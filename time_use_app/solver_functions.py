"""
solver_functions.py — pure functions for the 4-good Kenya time-use model.

Layered structure (top-down):

  numeric helpers           clamp, safe_pow, bisect_root
  CES / PIGL building blocks
  per-state household evaluation residuals
  per-state damped fixed-point solvers
  participation logit fixed point (2 x 2 over P_m, P_f)
  per-county wrapper        (4 states + participation -> E[V])
  spatial layer             (47 counties, market clearing, amenities)
  migration                 logit reallocation (population-weighted variant)
  counterfactual driver     re-solves everything under perturbed inputs

No I/O, no globals, no module-level mutable state.  All functions take
all dependencies as arguments and either return values or mutate the
Household / County dataclass passed in.
"""
from __future__ import annotations

import math
from dataclasses import replace as _dc_replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from classes import Household, ModelParams, County


# =========================================================================== #
# Numeric helpers                                                             #
# =========================================================================== #

EPS = 1e-12


def clamp(x: float, lo: float = EPS, hi: float = 1e300) -> float:
    """Floor x at lo and ceiling at hi.  Used everywhere to avoid log(0)."""
    return float(min(max(x, lo), hi))


def safe_pow(x: float, p: float) -> float:
    """Power with positive-base guard."""
    return float(clamp(x) ** p)


def bisect_root(f, lo: float, hi: float,
                max_iter: int = 200, tol: float = 1e-12) -> float:
    """Robust bisection.  Requires f(lo) * f(hi) < 0."""
    flo, fhi = f(lo), f(hi)
    if not (math.isfinite(flo) and math.isfinite(fhi)):
        raise ValueError("bisect_root: non-finite endpoint")
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0:
        raise ValueError("bisect_root: root not bracketed")
    for _ in range(max_iter):
        mid  = 0.5 * (lo + hi)
        fmid = f(mid)
        if not math.isfinite(fmid):
            mid  = 0.5 * (mid + lo)
            fmid = f(mid)
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if flo * fmid < 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


# =========================================================================== #
# CES / PIGL building blocks                                                  #
# =========================================================================== #

def ces_unit_cost(p_home: float, p_market: float,
                  omega: float, eta: float) -> float:
    """
    Composite price (model eq. 4):
        P_i = [ omega^eta · (P_iH)^(1-eta) + (1-omega)^eta · (p_i)^(1-eta) ]^(1/(1-eta))

    Cobb-Douglas limit at eta -> 1.
    """
    p_home   = clamp(p_home)
    p_market = clamp(p_market)
    omega    = clamp(omega, EPS, 1.0 - EPS)
    if abs(eta - 1.0) < 1e-10:
        return float((p_home ** omega) * (p_market ** (1.0 - omega)))
    inside = (omega ** eta) * safe_pow(p_home,   1.0 - eta) \
           + ((1.0 - omega) ** eta) * safe_pow(p_market, 1.0 - eta)
    return float(safe_pow(inside, 1.0 / (1.0 - eta)))


def ces_labour_aggregate(LM:  float, Lxf: float, Lc: float, Ld: float,
                         D_M: float, D_xf: float, D_c: float, D_d: float,
                         rho: float) -> float:
    """
    Per-gender labour aggregate over four activities (model eq. 7,
    extended to include food preparation):

        L^g = [ sum_j  D^g_j · (L^j_g)^((rho-1)/rho) ]^(rho/(rho-1))

    Inactive activities (L^j_g = 0) contribute zero in the (rho-1)/rho
    exponent because rho < 0; numerically they're floored at EPS.
    """
    r     = (rho - 1.0) / rho
    inner = (safe_pow(D_M  * LM,  r)
           + safe_pow(D_xf * Lxf, r)
           + safe_pow(D_c  * Lc,  r)
           + safe_pow(D_d  * Ld,  r))
    return float(safe_pow(inner, rho / (rho - 1.0)))


def pigl_B(P_xf: float, P_c: float, P_d: float,
           beta_xf: float, beta_c: float, beta_d: float) -> float:
    """
    PIGL price aggregator (model eq. 17), 4 goods, P_xn = 1 numeraire:
        log B(p) = beta_xf log P_xf + beta_xn log 1 + beta_c log P_c
                                                       + beta_d log P_d
    """
    return float(safe_pow(P_xf, beta_xf)
               * safe_pow(P_c,  beta_c)
               * safe_pow(P_d,  beta_d))


def pigl_lambda(E: float, B: float, eps: float) -> float:
    """Marginal utility of expenditure (model eq. 6): lambda = (1/B)(E/B)^(eps-1)."""
    return float((1.0 / clamp(B)) * safe_pow(clamp(E) / clamp(B), eps - 1.0))


def pigl_shares(E: float, B: float, eps: float,
                betas:  Tuple[float, float, float, float],
                kappas: Tuple[float, float, float, float]
                ) -> Tuple[float, float, float, float]:
    """
    PIGL shares (model eq. 16) for (xf, xn, c, d):
        theta_i = beta_i + kappa_i · (E/B)^(-eps)

    Returns the four shares without clamping.  Negative shares signal a
    parameter inconsistency upstream and are caught by the caller.
    """
    real = clamp(E) / clamp(B)
    adj  = safe_pow(real, -eps)
    return (float(betas[0] + kappas[0] * adj),
            float(betas[1] + kappas[1] * adj),
            float(betas[2] + kappas[2] * adj),
            float(betas[3] + kappas[3] * adj))


# =========================================================================== #
# Shadow prices and composite quantities                                      #
# =========================================================================== #

def _shadow_price_home(L_total: float, L_i_g: float, D_i_g: float,
                       A_i: float, lam: float, L: float,
                       phi: float, rho: float) -> float:
    """
    Shadow price of home good i, computed from any partner g's FOC
    (model eq. 13):

        P_iH = L^(1/phi + 1/rho) · D_i^g · (L_i^g)^(-1/rho) / (lambda · A_i)

    At the optimum both partners give the same value.  We always use the
    man's (g = m) here since he is the numeraire (D_M_m = D_xf_m = D_c_m = 1)
    and is active in three of the four participation states.
    """
    Lt = safe_pow(clamp(L), 1.0 / phi + 1.0 / rho)
    return float(Lt * safe_pow(clamp(D_i_g * L_i_g), -1.0 / rho)
                 / (clamp(A_i) * clamp(lam)))


# =========================================================================== #
# Per-state household evaluation                                              #
# =========================================================================== #
#
# All four states share the same back-end:  given current labour quantities,
# compute aggregates -> shadow prices -> composite prices -> lambda -> PIGL
# demands -> input splits, then return the residual vector that the
# fixed-point update should drive to zero.
#
# The differences across states:
#   (1,1)  both market-works active           -> 8 residuals
#   (1,0)  only man works (LM_f = 0)          -> 7 residuals (drop LM_f FOC)
#   (0,1)  only woman works (LM_m = 0)        -> 7 residuals (drop LM_m FOC)
#   (0,0)  neither works                      -> 6 residuals (drop both)
#
# In states with one or both market-works inactive, the corresponding
# wage income is zero so the budget collapses to E = a (when both off)
# or E = y_g · L^M_g + a (when one is on).
# =========================================================================== #


def _aggregate_and_prices(hh: Household, mp: ModelParams) -> None:
    """
    Compute L_m, L_f, L, then solve for lambda by fixed-point against
    the budget E and the home shadow prices.  Mutates hh in place.

    Steps (model section 5, generalised to 4 goods):
       1. L^g via the CES aggregator over 4 activities
       2. P_iH = (L^(1/phi+1/rho) D_i^m (L_i^m)^(-1/rho)) / (A_i lambda)
       3. P_i  = ces_unit_cost(P_iH, p_i, omega_i, eta_i)
       4. B(p) = prod P_i^beta_i
       5. lambda = (1/B)(E/B)^(eps-1)   given E from the budget
    """
    s_m, s_f = hh.state

    # Budget: market labours are pinned to zero in inactive states, so
    # this works uniformly.  hh.LM_m / hh.LM_f are set by the caller.
    hh.E = float(hh.y_m * hh.LM_m + hh.y_f * hh.LM_f + hh.a)

    # Per-gender CES aggregate (4 activities)
    hh.L_m = ces_labour_aggregate(
        hh.LM_m, hh.Lxf_m, hh.Lc_m, hh.Ld_m,
        mp.D_M_m, mp.D_xf_m, mp.D_c_m, mp.D_d_m, mp.rho)
    hh.L_f = ces_labour_aggregate(
        hh.LM_f, hh.Lxf_f, hh.Lc_f, hh.Ld_f,
        mp.D_M_f, mp.D_xf_f, mp.D_c_f, mp.D_d_f, mp.rho)
    hh.L = float(hh.L_m + hh.L_f)

    # We use the man's (g = m) shadow price.  In state (0, 0) the male
    # market labour is zero but his three home activities are active, so
    # the male side still works.  In all four states at least one of the
    # men's home activities is active.
    def _prices(lam: float) -> Tuple[float, float, float, float, float, float, float]:
        PxfH = _shadow_price_home(hh.L, hh.Lxf_m, mp.D_xf_m, hh.A_xf, lam,
                                  hh.L, mp.phi, mp.rho)
        PcH  = _shadow_price_home(hh.L, hh.Lc_m,  mp.D_c_m,  hh.A_c,  lam,
                                  hh.L, mp.phi, mp.rho)
        PdH  = _shadow_price_home(hh.L, hh.Ld_m,  mp.D_d_m,  hh.A_d,  lam,
                                  hh.L, mp.phi, mp.rho)
        Pxf  = ces_unit_cost(PxfH, hh.p_xf, mp.omega_xf, mp.eta_xf)
        Pc   = ces_unit_cost(PcH,  hh.pc,   mp.omega_c,  mp.eta_c)
        Pd   = ces_unit_cost(PdH,  hh.pd,   mp.omega_d,  mp.eta_d)
        B    = pigl_B(Pxf, Pc, Pd, mp.beta_xf, mp.beta_c, mp.beta_d)
        return PxfH, PcH, PdH, Pxf, Pc, Pd, B

    # Solve  lambda = (1/B)(E/B)^(eps-1)  by bracketed bisection
    def _residual(lam: float) -> float:
        _, _, _, _, _, _, B = _prices(lam)
        return lam - pigl_lambda(hh.E, B, mp.eps_engel)

    lo, hi = 1e-12, 1e6
    flo, fhi = _residual(lo), _residual(hi)
    it = 0
    while flo * fhi > 0 and it < 60:
        hi *= 10.0
        fhi = _residual(hi)
        it += 1
    if flo * fhi > 0:
        # Last-resort: extreme parameters; fall back to lambda = pigl_lambda
        # at B from a coarse guess.  Convergence will fail upstream and
        # the solver will mark the state as not converged.
        hh.lam = float(pigl_lambda(hh.E, 1.0, mp.eps_engel))
    else:
        hh.lam = float(bisect_root(_residual, lo, hi))

    hh.PxfH, hh.PcH, hh.PdH, hh.Pxf, hh.Pc, hh.Pd, hh.B = _prices(hh.lam)


def _pigl_demands_and_splits(hh: Household, mp: ModelParams) -> None:
    """
    PIGL share -> composite quantities -> CES home/market split for each
    home-producible good.  Mutates hh in place.
    """
    th = pigl_shares(hh.E, hh.B, mp.eps_engel,
                     (mp.beta_xf, mp.beta_xn, mp.beta_c, mp.beta_d),
                     (mp.kappa_xf, mp.kappa_xn, mp.kappa_c, mp.kappa_d))
    hh.th_xf, hh.th_xn, hh.th_c, hh.th_d = th

    hh.xn  = float(hh.th_xn * hh.E)
    hh.Sxf = float(hh.th_xf * hh.E / clamp(hh.Pxf))
    hh.Sc  = float(hh.th_c  * hh.E / clamp(hh.Pc))
    hh.Sd  = float(hh.th_d  * hh.E / clamp(hh.Pd))

    # CES quantity splits (model eq. 15)
    hh.SxfM = float((1.0 - mp.omega_xf) ** mp.eta_xf
                    * safe_pow(hh.Pxf / clamp(hh.p_xf), mp.eta_xf) * hh.Sxf)
    hh.ScM  = float((1.0 - mp.omega_c)  ** mp.eta_c
                    * safe_pow(hh.Pc  / clamp(hh.pc),   mp.eta_c)  * hh.Sc)
    hh.SdM  = float((1.0 - mp.omega_d)  ** mp.eta_d
                    * safe_pow(hh.Pd  / clamp(hh.pd),   mp.eta_d)  * hh.Sd)
    hh.SxfH = float((mp.omega_xf) ** mp.eta_xf
                    * safe_pow(hh.Pxf / clamp(hh.PxfH), mp.eta_xf) * hh.Sxf)
    hh.ScH  = float((mp.omega_c)  ** mp.eta_c
                    * safe_pow(hh.Pc  / clamp(hh.PcH),  mp.eta_c)  * hh.Sc)
    hh.SdH  = float((mp.omega_d)  ** mp.eta_d
                    * safe_pow(hh.Pd  / clamp(hh.PdH),  mp.eta_d)  * hh.Sd)


def evaluate(hh: Household, mp: ModelParams) -> None:
    """Full single-pass evaluation; mutates hh.  Used by all four state solvers."""
    _aggregate_and_prices(hh, mp)
    _pigl_demands_and_splits(hh, mp)


# --------------------------------------------------------------------------- #
# Fixed-point update map                                                      #
# --------------------------------------------------------------------------- #
#
# Same structure as the existing 3-good code, extended with food preparation:
#
#   Active L^M_g:  (lambda y_g) = L_term · D^M_g · (L^M_g)^(-1/rho)
#                  =>  L^M_g = (L_term / (lambda y_g))^rho / D^M_g
#
#   Each home activity i in {xf, c, d}:
#     L_i_total = S^i_H / A_i
#     L^i_m / L^i_f = (D^i_m / D^i_f)^rho
#     =>  L^i_m = L_i_total · ratio_i / (1 + ratio_i)
#         L^i_f = L_i_total            / (1 + ratio_i)
# --------------------------------------------------------------------------- #


def _fp_update(hh: Household, mp: ModelParams) -> Dict[str, float]:
    """
    Compute the fixed-point target values for the eight labour quantities.
    Inactive market-works (per hh.state) are returned as zero.
    """
    s_m, s_f = hh.state
    L_term = safe_pow(clamp(hh.L), 1.0 / mp.phi + 1.0 / mp.rho)

    # Market work
    if s_m:
        LM_m_new = safe_pow(L_term / clamp(hh.lam * hh.y_m), mp.rho) / mp.D_M_m
    else:
        LM_m_new = 0.0
    if s_f:
        LM_f_new = safe_pow(L_term / clamp(hh.lam * hh.y_f), mp.rho) / mp.D_M_f
    else:
        LM_f_new = 0.0

    # Home activities: total from CES split, gender split from intra-pair FOC
    Lxf_total = clamp(hh.SxfH) / clamp(hh.A_xf)
    Lc_total  = clamp(hh.ScH)  / clamp(hh.A_c)
    Ld_total  = clamp(hh.SdH)  / clamp(hh.A_d)

    r_xf = safe_pow(clamp(mp.D_xf_m / mp.D_xf_f), mp.rho)
    r_c  = safe_pow(clamp(mp.D_c_m  / mp.D_c_f),  mp.rho)
    r_d  = safe_pow(clamp(mp.D_d_m  / mp.D_d_f),  mp.rho)

    return dict(
        LM_m  = LM_m_new,
        LM_f  = LM_f_new,
        Lxf_m = Lxf_total * r_xf / (1.0 + r_xf),
        Lxf_f = Lxf_total          / (1.0 + r_xf),
        Lc_m  = Lc_total  * r_c  / (1.0 + r_c),
        Lc_f  = Lc_total           / (1.0 + r_c),
        Ld_m  = Ld_total  * r_d  / (1.0 + r_d),
        Ld_f  = Ld_total           / (1.0 + r_d),
    )


# =========================================================================== #
# Per-state damped fixed-point solver                                         #
# =========================================================================== #

# Initial guesses (1000-KSh / hour world, weekly hour-shares).  Calibrated to
# the existing 3-good solver's converged values, with food prep added at
# men's/women's TUS averages converted to hour-share-of-week proportions.
_DEFAULT_L0 = dict(
    LM_m  = 0.30, LM_f  = 0.10,
    Lxf_m = 0.05, Lxf_f = 0.15,
    Lc_m  = 0.04, Lc_f  = 0.10,
    Ld_m  = 0.05, Ld_f  = 0.10,
)


def _apply_state_constraints(L: Dict[str, float], state: Tuple[int, int]
                              ) -> Dict[str, float]:
    """Pin LM_m / LM_f to zero per the participation state."""
    s_m, s_f = state
    out = dict(L)
    if not s_m:
        out["LM_m"] = 0.0
    if not s_f:
        out["LM_f"] = 0.0
    return out


def solve_state(hh: Household, mp: ModelParams,
                L0: Optional[Dict[str, float]] = None,
                max_iter: int = 5000, tol: float = 1e-9,
                damping: float = 0.2) -> Tuple[bool, int, float]:
    """
    Damped fixed-point solver in the 6-to-8 dim labour space (depending on
    hh.state).  Updates hh in place; returns (converged, n_iter, res_norm).

    Adaptive damping: shrink by 0.5 if the residual norm grew, grow by
    1.05 (capped at 0.9) if it shrank.  Same scheme as the existing code.
    """
    L = _apply_state_constraints(L0 or _DEFAULT_L0, hh.state)
    omega = damping
    prev_norm = math.inf
    best_norm = math.inf
    best_L = dict(L)

    for k in range(max_iter):
        # Set hh labour from L
        hh.LM_m, hh.LM_f = L["LM_m"], L["LM_f"]
        hh.Lxf_m, hh.Lxf_f = L["Lxf_m"], L["Lxf_f"]
        hh.Lc_m, hh.Lc_f   = L["Lc_m"],  L["Lc_f"]
        hh.Ld_m, hh.Ld_f   = L["Ld_m"],  L["Ld_f"]

        # Evaluate prices and demands at this L
        try:
            evaluate(hh, mp)
        except Exception:
            return False, k, math.inf

        # Build target L from the fixed-point update, applying state mask
        L_new = _apply_state_constraints(_fp_update(hh, mp), hh.state)

        # Damped update
        L_next = {k_: (1.0 - omega) * L[k_] + omega * L_new[k_] for k_ in L}
        L_next = _apply_state_constraints(L_next, hh.state)

        # Floor strictly positive on active dimensions
        for k_ in L_next:
            s_m, s_f = hh.state
            if k_ == "LM_m" and not s_m:
                continue
            if k_ == "LM_f" and not s_f:
                continue
            L_next[k_] = max(L_next[k_], 1e-14)

        step  = sum((L_next[k_] - L[k_]) ** 2 for k_ in L) ** 0.5

        # Residual: difference between current L and the un-damped target
        res_vec = [L_new[k_] - L[k_] for k_ in L]
        norm = float(np.linalg.norm(res_vec))

        if norm < best_norm:
            best_norm = norm
            best_L = dict(L)

        if norm < tol and step < tol:
            # converged: keep the latest L
            L = L_next
            hh.LM_m, hh.LM_f = L["LM_m"], L["LM_f"]
            hh.Lxf_m, hh.Lxf_f = L["Lxf_m"], L["Lxf_f"]
            hh.Lc_m, hh.Lc_f   = L["Lc_m"],  L["Lc_f"]
            hh.Ld_m, hh.Ld_f   = L["Ld_m"],  L["Ld_f"]
            evaluate(hh, mp)
            _set_value(hh, mp)
            return True, k, norm

        # Adaptive damping
        if norm > prev_norm * 1.01:
            omega = max(1e-4, omega * 0.5)
        else:
            omega = min(0.9, omega * 1.05)
        prev_norm = norm
        L = L_next

    # max-iter hit: fall back to best-seen
    L = best_L
    hh.LM_m, hh.LM_f = L["LM_m"], L["LM_f"]
    hh.Lxf_m, hh.Lxf_f = L["Lxf_m"], L["Lxf_f"]
    hh.Lc_m, hh.Lc_f   = L["Lc_m"],  L["Lc_f"]
    hh.Ld_m, hh.Ld_f   = L["Ld_m"],  L["Ld_f"]
    try:
        evaluate(hh, mp)
        _set_value(hh, mp)
    except Exception:
        pass
    return False, max_iter, best_norm


def _set_value(hh: Household, mp: ModelParams) -> None:
    """V at this single state (model eq. 23):
           V = (E/B)^eps / eps  -  (L_m^(1+1/phi) + L_f^(1+1/phi)) / (1 + 1/phi)
    """
    eps = mp.eps_engel
    phi = mp.phi
    if not (math.isfinite(hh.E) and math.isfinite(hh.B) and
            math.isfinite(hh.L_m) and math.isfinite(hh.L_f)):
        hh.V_state = float("nan")
        return
    util = safe_pow(clamp(hh.E) / clamp(hh.B), eps) / eps
    disu = (safe_pow(clamp(hh.L_m), 1.0 + 1.0 / phi)
          + safe_pow(clamp(hh.L_f), 1.0 + 1.0 / phi)) / (1.0 + 1.0 / phi)
    hh.V_state = float(util - disu)


# =========================================================================== #
# Per-county four-state participation solve                                   #
# =========================================================================== #

def _make_household(county: County, mp: ModelParams,
                    h_m: float, h_f: float, a: float,
                    state: Tuple[int, int]) -> Household:
    """Build a fresh Household at a (county, h, state) point."""
    return Household(
        params = mp,
        y_m  = float(county.w_ell * h_m),
        y_f  = float(mp.wage_gap * county.w_ell * h_f),
        p_xf = float(county.p_xf),
        pc   = float(county.pc),
        pd   = float(county.pd),
        a    = a,
        A_xf = float(county.A_xf_home),
        A_c  = float(county.A_c_home),
        A_d  = float(county.A_d_home),
        state = state,
    )


def _participation_logit(V11: float, V10: float, V01: float, V00: float,
                         mp: ModelParams,
                         max_iter: int = 200, tol: float = 1e-10
                         ) -> Tuple[float, float, float]:
    """
    2 x 2 fixed point in (P_m, P_f) per model eq. 24-25.  Returns
    (P_m, P_f, EV) where EV is the expected value (eq. 26).
    """
    # All four state values must be finite.  If any is NaN, fall back to
    # the interior value with both participating.
    if not all(math.isfinite(v) for v in (V11, V10, V01, V00)):
        return 1.0, 1.0, V11 if math.isfinite(V11) else float("nan")

    P_m, P_f = 0.5, 0.5
    for _ in range(max_iter):
        Vbar_m_1 = P_f * V11 + (1.0 - P_f) * V10
        Vbar_m_0 = P_f * V01 + (1.0 - P_f) * V00
        Vbar_f_1 = P_m * V11 + (1.0 - P_m) * V01
        Vbar_f_0 = P_m * V10 + (1.0 - P_m) * V00

        # P^g = 1 / (1 + exp((Vbar_g_0 - Vbar_g_1 + u_bar^g) / sigma_u^g))
        z_m = (Vbar_m_0 - Vbar_m_1 + mp.u_bar_m) / max(mp.sigma_u_m, 1e-9)
        z_f = (Vbar_f_0 - Vbar_f_1 + mp.u_bar_f) / max(mp.sigma_u_f, 1e-9)
        P_m_new = 1.0 / (1.0 + math.exp(min(50.0, max(-50.0, z_m))))
        P_f_new = 1.0 / (1.0 + math.exp(min(50.0, max(-50.0, z_f))))

        if abs(P_m_new - P_m) < tol and abs(P_f_new - P_f) < tol:
            P_m, P_f = P_m_new, P_f_new
            break
        P_m, P_f = P_m_new, P_f_new

    EV = (P_m * P_f * V11
        + P_m * (1.0 - P_f) * V10
        + (1.0 - P_m) * P_f * V01
        + (1.0 - P_m) * (1.0 - P_f) * V00)
    return float(P_m), float(P_f), float(EV)


def solve_county_household(county: County, mp: ModelParams,
                           h_m: float, h_f: float, a: float = 0.2,
                           max_iter: int = 5000, tol: float = 1e-9,
                           damping: float = 0.2
                           ) -> Dict[str, object]:
    """
    Solve all four participation states + the participation logit fixed point
    at one (county, h) point.

    Hot-starting: state (1,1) is solved first; the converged labour vector
    is then used as the starting point for the constrained states with the
    relevant component(s) zeroed.

    Returns a dict with all four Household objects + (P_m, P_f, EV).
    """
    out: Dict[str, object] = {}

    # ----- (1,1) interior -------------------------------------------------- #
    hh11 = _make_household(county, mp, h_m, h_f, a, (1, 1))
    conv11, _, _ = solve_state(hh11, mp, L0=None,
                               max_iter=max_iter, tol=tol, damping=damping)
    out["hh11"], out["conv11"] = hh11, conv11

    # Hot-start vector from (1,1) for the constrained states
    L_hot = dict(
        LM_m=hh11.LM_m, LM_f=hh11.LM_f,
        Lxf_m=hh11.Lxf_m, Lxf_f=hh11.Lxf_f,
        Lc_m=hh11.Lc_m, Lc_f=hh11.Lc_f,
        Ld_m=hh11.Ld_m, Ld_f=hh11.Ld_f,
    )

    # ----- (1, 0) only the man works -------------------------------------- #
    hh10 = _make_household(county, mp, h_m, h_f, a, (1, 0))
    L0_10 = dict(L_hot); L0_10["LM_f"] = 0.0
    conv10, _, _ = solve_state(hh10, mp, L0=L0_10,
                               max_iter=max_iter, tol=tol, damping=damping)
    out["hh10"], out["conv10"] = hh10, conv10

    # ----- (0, 1) only the woman works ------------------------------------ #
    hh01 = _make_household(county, mp, h_m, h_f, a, (0, 1))
    L0_01 = dict(L_hot); L0_01["LM_m"] = 0.0
    conv01, _, _ = solve_state(hh01, mp, L0=L0_01,
                               max_iter=max_iter, tol=tol, damping=damping)
    out["hh01"], out["conv01"] = hh01, conv01

    # ----- (0, 0) neither works ------------------------------------------- #
    hh00 = _make_household(county, mp, h_m, h_f, a, (0, 0))
    L0_00 = dict(L_hot); L0_00["LM_m"] = 0.0; L0_00["LM_f"] = 0.0
    conv00, _, _ = solve_state(hh00, mp, L0=L0_00,
                               max_iter=max_iter, tol=tol, damping=damping)
    out["hh00"], out["conv00"] = hh00, conv00

    # ----- Participation logit -------------------------------------------- #
    P_m, P_f, EV = _participation_logit(
        hh11.V_state, hh10.V_state, hh01.V_state, hh00.V_state, mp)
    out["P_m"], out["P_f"], out["EV"] = P_m, P_f, EV

    return out


# =========================================================================== #
# Spatial layer                                                               #
# =========================================================================== #

def solve_county_grid(county: County, mp: ModelParams,
                      h_grid: np.ndarray, a: float = 0.2
                      ) -> Dict[str, np.ndarray]:
    """
    Solve every (county, h) pair on h_grid.  Returns arrays of length
    len(h_grid) for each reportable quantity, plus scalar means used by
    the spatial layer.

    The same h is used for both partners (h_m = h_f = h on the grid),
    matching the existing project's convention.
    """
    Nh = len(h_grid)
    arrs: Dict[str, np.ndarray] = {
        k: np.full(Nh, np.nan) for k in (
            "LM_m", "LM_f", "Lxf_m", "Lxf_f", "Lc_m", "Lc_f", "Ld_m", "Ld_f",
            "E", "V11", "V10", "V01", "V00", "EV", "P_m", "P_f",
            "th_xf", "th_xn", "th_c", "th_d",
            "SxfH_share", "SxfM_share",
            "ScH_share", "ScM_share",
            "SdH_share", "SdM_share",
            "Pxf", "Pc", "Pd",
            "SxfM", "ScM", "SdM",   # market quantity demands (for clearing)
            "xn",                    # non-food demand (for clearing)
        )
    }
    arrs["converged"] = np.zeros(Nh, dtype=bool)

    for i, h in enumerate(h_grid):
        try:
            res = solve_county_household(county, mp, float(h), float(h), a)
        except Exception:
            continue
        hh11 = res["hh11"]
        if not res["conv11"]:
            continue
        # Per-h hours and prices come from the (1,1) interior solve;
        # this is what the existing 3-good app shows.
        arrs["LM_m"][i]  = hh11.LM_m;  arrs["LM_f"][i]  = hh11.LM_f
        arrs["Lxf_m"][i] = hh11.Lxf_m; arrs["Lxf_f"][i] = hh11.Lxf_f
        arrs["Lc_m"][i]  = hh11.Lc_m;  arrs["Lc_f"][i]  = hh11.Lc_f
        arrs["Ld_m"][i]  = hh11.Ld_m;  arrs["Ld_f"][i]  = hh11.Ld_f
        arrs["E"][i]     = hh11.E
        arrs["th_xf"][i] = hh11.th_xf; arrs["th_xn"][i] = hh11.th_xn
        arrs["th_c"][i]  = hh11.th_c;  arrs["th_d"][i]  = hh11.th_d
        arrs["Pxf"][i]   = hh11.Pxf;   arrs["Pc"][i]    = hh11.Pc
        arrs["Pd"][i]    = hh11.Pd
        arrs["SxfM"][i]  = hh11.SxfM
        arrs["ScM"][i]   = hh11.ScM
        arrs["SdM"][i]   = hh11.SdM
        arrs["xn"][i]    = hh11.xn

        # Expenditure shares: P_iH·S_iH / (P_i·S_i).  These sum to 1 by the
        # CES envelope.  Quantity ratios do NOT sum to 1.
        if hh11.Pxf > 0 and hh11.Sxf > 0:
            arrs["SxfH_share"][i] = (hh11.PxfH * hh11.SxfH) / (hh11.Pxf * hh11.Sxf)
            arrs["SxfM_share"][i] = (hh11.p_xf * hh11.SxfM) / (hh11.Pxf * hh11.Sxf)
        if hh11.Pc > 0 and hh11.Sc > 0:
            arrs["ScH_share"][i]  = (hh11.PcH * hh11.ScH) / (hh11.Pc * hh11.Sc)
            arrs["ScM_share"][i]  = (hh11.pc  * hh11.ScM) / (hh11.Pc * hh11.Sc)
        if hh11.Pd > 0 and hh11.Sd > 0:
            arrs["SdH_share"][i]  = (hh11.PdH * hh11.SdH) / (hh11.Pd * hh11.Sd)
            arrs["SdM_share"][i]  = (hh11.pd  * hh11.SdM) / (hh11.Pd * hh11.Sd)

        # Per-state values and participation
        arrs["V11"][i] = res["hh11"].V_state
        arrs["V10"][i] = res["hh10"].V_state
        arrs["V01"][i] = res["hh01"].V_state
        arrs["V00"][i] = res["hh00"].V_state
        arrs["P_m"][i] = res["P_m"]
        arrs["P_f"][i] = res["P_f"]
        arrs["EV"][i]  = res["EV"]
        arrs["converged"][i] = True

    return arrs


def solve_all_counties(counties: List[County], mp: ModelParams,
                       h_grid: np.ndarray, a: float = 0.2
                       ) -> Dict[int, Dict[str, np.ndarray]]:
    """Solve every county on h_grid; returns {county_id: arrays}."""
    return {c.county_id: solve_county_grid(c, mp, h_grid, a)
            for c in counties}


def compute_market_clearing(counties: List[County],
                            results: Dict[int, Dict[str, np.ndarray]]
                            ) -> None:
    """
    Compute county sectoral employment via free-entry pricing
    (eq. 28-30) and the household demands.  Mutates each county in
    place, setting LM_xf_total, LM_c_total, LM_d_total, LM_xn_total.

    Per model eq. 34:
        LM_i_l = N_l · S_iM(l) / AM_i_l    for i in {xf, c, d}
    Per model eq. 35:
        LM_xn_l = N_l · ( h_m · LM_m + wage_gap · h_f · LM_f )
                  - sum_{i!=xn} LM_i_l

    For the working-baseline 4-good model we use household-mean
    market demands and household-mean market hours from the grid;
    we treat h = mean(h_grid) as the household efficiency scale.
    """
    for c in counties:
        r = results.get(c.county_id)
        if r is None:
            continue
        S_xf_M = float(np.nanmean(r["SxfM"]))
        S_c_M  = float(np.nanmean(r["ScM"]))
        S_d_M  = float(np.nanmean(r["SdM"]))
        LM_m   = float(np.nanmean(r["LM_m"]))
        LM_f   = float(np.nanmean(r["LM_f"]))

        c.LM_xf_total = float(c.N * S_xf_M / clamp(c.AM_xf))
        c.LM_c_total  = float(c.N * S_c_M  / clamp(c.AM_c))
        c.LM_d_total  = float(c.N * S_d_M  / clamp(c.AM_d))

        # Total market labour supplied (efficiency units, h grid mean = 1)
        # In our calibration h_m and h_f are the efficiency scaler already
        # baked into y_m, y_f.  For the per-county clearing diagnostic we
        # use the grid-mean LM values directly: LM_xn = N (LM_m + wage_gap LM_f) - sum others.
        wg = 1.0  # wage_gap factor on female efficiency hours: handled via y_f already in solver
        # (wage_gap scaling on female hours appears in the wage, not the time)
        total_market = float(c.N * (LM_m + wg * LM_f))
        c.LM_xn_total = total_market - (c.LM_xf_total + c.LM_c_total + c.LM_d_total)


def calibrate_amenities(counties: List[County],
                        results: Dict[int, Dict[str, np.ndarray]]
                        ) -> float:
    """
    Set xi_l = Ubar - V_star_l with population-weighted Ubar (model eq. 37).
    Mutates counties in place.  Returns Ubar.
    """
    Vs = []
    Pms, Pfs = [], []
    for c in counties:
        r = results.get(c.county_id, {})
        EV = r.get("EV", np.array([np.nan]))
        Vs.append(float(np.nanmean(EV)))
        Pms.append(float(np.nanmean(r.get("P_m", np.array([np.nan])))))
        Pfs.append(float(np.nanmean(r.get("P_f", np.array([np.nan])))))
    Vs = np.array(Vs); Pms = np.array(Pms); Pfs = np.array(Pfs)
    Ns = np.array([c.N for c in counties], dtype=float)

    finite = np.isfinite(Vs) & np.isfinite(Ns) & (Ns > 0)
    if finite.sum() == 0:
        Ubar = float("nan")
    else:
        Ubar = float(np.sum(Ns[finite] * Vs[finite]) / np.sum(Ns[finite]))

    for c, V, Pm, Pf in zip(counties, Vs, Pms, Pfs):
        c.V_star = float(V)
        c.P_m    = float(Pm)
        c.P_f    = float(Pf)
        c.xi     = float(Ubar - V) if math.isfinite(V) and math.isfinite(Ubar) else float("nan")
    return Ubar


# =========================================================================== #
# Migration                                                                   #
# =========================================================================== #

def migration_update(N_baseline: np.ndarray,
                     V_star_new: np.ndarray,
                     xi: np.ndarray,
                     sigma_mig: float = 1.0) -> np.ndarray:
    """
    Population-weighted logit migration (model eq. 38-39):

        N'_l = N_total · ( N_l · exp((V*'_l + xi_l - Ubar') / sigma_mig) )
                       / sum_k N_k · exp((V*'_k + xi_k - Ubar') / sigma_mig)

    where Ubar' is the population-weighted (with baseline N) mean of V*' + xi,
    so that at baseline (V*' = V*) we have N' = N exactly.
    """
    finite = np.isfinite(V_star_new) & np.isfinite(xi) & np.isfinite(N_baseline)
    if finite.sum() == 0:
        return N_baseline.copy()
    N_total = float(np.sum(N_baseline[finite]))
    Ubar_new = float(np.sum(N_baseline[finite] * (V_star_new[finite] + xi[finite]))
                     / N_total)
    log_w = (V_star_new + xi - Ubar_new) / max(sigma_mig, 1e-9)
    log_w = np.where(finite, log_w, -np.inf)
    log_w = log_w - np.nanmax(log_w[finite])
    w = N_baseline * np.exp(log_w)
    w = np.where(np.isfinite(w), w, 0.0)
    Z = float(np.sum(w))
    if Z <= 0:
        return N_baseline.copy()
    return N_total * w / Z


# =========================================================================== #
# Counterfactual driver                                                       #
# =========================================================================== #

def run_counterfactual(counties_baseline: List[County],
                       counties_new: List[County],
                       mp_new: ModelParams,
                       h_grid: np.ndarray,
                       results_baseline: Dict[int, Dict[str, np.ndarray]],
                       a: float = 0.2,
                       with_migration: bool = True
                       ) -> Dict[str, object]:
    """
    Re-solve the full spatial equilibrium under perturbed inputs.  Holds
    amenities xi from the baseline.  Returns:

        results_new   {cid: arrays}
        V_star_new    np.ndarray
        N_new         np.ndarray  (= baseline if with_migration is False)
        Ubar_new      float
        delta_V       np.ndarray  (V_star_new - V_star_base)
        delta_N_pct   np.ndarray  (100 * (N_new - N_base) / N_base)
    """
    # Re-solve households at the perturbed parameters/counties
    results_new = solve_all_counties(counties_new, mp_new, h_grid, a)

    # Pull V_star (E[V]) and baseline xi
    cids = [c.county_id for c in counties_baseline]
    V_star_new = np.array([
        float(np.nanmean(results_new.get(cid, {}).get("EV", np.array([np.nan]))))
        for cid in cids])
    V_star_base = np.array([c.V_star for c in counties_baseline])
    xi  = np.array([c.xi for c in counties_baseline])
    N0  = np.array([c.N  for c in counties_baseline], dtype=float)

    # New equilibrium utility (population-weighted with baseline N)
    finite = np.isfinite(V_star_new) & np.isfinite(xi) & (N0 > 0)
    if finite.sum() == 0:
        Ubar_new = float("nan")
    else:
        Ubar_new = float(np.sum(N0[finite] * (V_star_new[finite] + xi[finite]))
                         / float(np.sum(N0[finite])))

    if with_migration:
        N_new = migration_update(N0, V_star_new, xi, mp_new.sigma_mig)
    else:
        N_new = N0.copy()

    delta_N_pct = np.where(N0 > 0, 100.0 * (N_new / N0 - 1.0), 0.0)

    return dict(
        results_new = results_new,
        V_star_new  = V_star_new,
        V_star_base = V_star_base,
        N_new       = N_new,
        Ubar_new    = Ubar_new,
        delta_V     = V_star_new - V_star_base,
        delta_N_pct = delta_N_pct,
        cids        = cids,
    )
