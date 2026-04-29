#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classes.py  –  model parameters and household object for the extended model.

Extensions vs. baseline:
  • Gender-differentiated disutility weights (D_M, D_c, D_d separately for m/f).
  • Two effective wages y_m, y_f (= w_ell * h_m / h_f).
  • Household labor aggregate  L = L^m + L^f  where each L^g is a gender-
    specific CES aggregate of (L_M^g, L_c^g, L_d^g).
  • Six labor choice variables instead of three.
  • The solver works in the 6-dimensional labor space
    (LM_m, LM_f, Lc_m, Lc_f, Ld_m, Ld_f).
  • Home production uses total hours: S_H^i = A_i (L_i^m + L_i^f).
  • lambda fixed-point and PIGL demands are unchanged in structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from functions import (
    clamp, safe_pow, EPS,
    ces_unit_cost, pigl_B, pigl_lambda, pigl_shares,
    ces_labor_aggregate, bisect_root,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelParams:
    """
    Global (structural) parameters of the extended model.

    Gender-specific disutility weights are stored as separate fields
    for men (suffix _m) and women (suffix _f).
    """
    # ── PIGL ──────────────────────────────────────────────────────────────────
    eps_engel: float
    beta_x:    float
    beta_c:    float
    beta_d:    float
    kappa_x:   float
    kappa_c:   float
    kappa_d:   float

    # ── Home-market CES ───────────────────────────────────────────────────────
    omega_c: float
    omega_d: float
    eta_c:   float
    eta_d:   float

    # ── Disutility weights – men ───────────────────────────────────────────────
    D_M_m: float
    D_c_m: float
    D_d_m: float

    # ── Disutility weights – women ─────────────────────────────────────────────
    D_M_f: float
    D_c_f: float
    D_d_f: float

    # ── Labor aggregate ────────────────────────────────────────────────────────
    rho: float    # < 0  →  complementarity across activities
    phi: float    # Frisch elasticity


# ═══════════════════════════════════════════════════════════════════════════════
# Household
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Household:
    """
    Household with one man and one woman.

    The solver iterates over the 6-vector
      (LM_m, LM_f, Lc_m, Lc_f, Ld_m, Ld_f)
    and calls evaluate_from_labor() at each step.

    Everything else (lambda, prices, demands, residuals) is computed
    internally and stored as attributes so the Bokeh app can read them.
    """
    params: ModelParams

    # ── Exogenous (county-level) ───────────────────────────────────────────────
    y_m: float      # effective wage of man   = w_ell * h_m
    y_f: float      # effective wage of woman = w_ell * h_f
    pc:  float      # market price of care services
    pd:  float      # market price of domestic services
    a:   float      # non-labor income

    # ── Home productivity (county-level, gender-neutral) ──────────────────────
    A_c: float = 1.0
    A_d: float = 1.0

    # ── Labor choices (filled by evaluate_from_labor) ─────────────────────────
    LM_m: float = np.nan
    LM_f: float = np.nan
    Lc_m: float = np.nan
    Lc_f: float = np.nan
    Ld_m: float = np.nan
    Ld_f: float = np.nan

    # ── Aggregate labor ───────────────────────────────────────────────────────
    L_m: float = np.nan    # gender-specific CES aggregate
    L_f: float = np.nan
    L:   float = np.nan    # household total  L = L_m + L_f

    # ── Prices and lambda ────────────────────────────────────────────────────
    E:    float = np.nan
    lam:  float = np.nan
    PcH:  float = np.nan
    PdH:  float = np.nan
    Pc:   float = np.nan
    Pd:   float = np.nan
    B:    float = np.nan

    # ── PIGL demands ─────────────────────────────────────────────────────────
    th_x: float = np.nan
    th_c: float = np.nan
    th_d: float = np.nan
    x:    float = np.nan
    Sc:   float = np.nan
    Sd:   float = np.nan

    # ── Input splits ─────────────────────────────────────────────────────────
    ScH: float = np.nan
    SdH: float = np.nan
    ScM: float = np.nan
    SdM: float = np.nan

    # ── Residuals (6-vector) ─────────────────────────────────────────────────
    residuals: np.ndarray = field(default_factory=lambda: np.full(6, np.nan))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 0 – set labor
    # ─────────────────────────────────────────────────────────────────────────

    def set_labor(self, LM_m: float, LM_f: float,
                  Lc_m: float, Lc_f: float,
                  Ld_m: float, Ld_f: float) -> None:
        (self.LM_m, self.LM_f,
         self.Lc_m, self.Lc_f,
         self.Ld_m, self.Ld_f) = (float(LM_m), float(LM_f),
                                   float(Lc_m), float(Lc_f),
                                   float(Ld_m), float(Ld_f))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 – total expenditure  E = y_m L_M^m + y_f L_M^f + a   (D1)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_E(self) -> None:
        self.E = float(self.y_m * self.LM_m + self.y_f * self.LM_f + self.a)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 – gender-specific and household labor aggregates   (D3 extended)
    #
    # L^g = CES(D_M^g L_M^g,  D_c^g L_c^g,  D_d^g L_d^g)
    # L   = L^m + L^f
    # ─────────────────────────────────────────────────────────────────────────

    def compute_L_aggregate(self) -> None:
        p = self.params
        self.L_m = ces_labor_aggregate(
            self.LM_m, self.Lc_m, self.Ld_m,
            p.D_M_m, p.D_c_m, p.D_d_m, p.rho,
        )
        self.L_f = ces_labor_aggregate(
            self.LM_f, self.Lc_f, self.Ld_f,
            p.D_M_f, p.D_c_f, p.D_d_f, p.rho,
        )
        self.L = float(self.L_m + self.L_f)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 – shadow price of home-produced service i for gender g   (D4)
    #
    # P_i^{H,g} = L^{1/φ+1/ρ} · D_i^g · (L_i^g)^{-1/ρ}  /  (A_i λ)
    #
    # At the optimum the two genders have equal shadow prices, so we can
    # use either.  We use the sum-of-hours form directly:
    # the household shadow price is computed via the aggregate first-order
    # condition (which does not depend on the gender split).
    # ─────────────────────────────────────────────────────────────────────────

    def _shadow_price_home(self, Li_g: float, Di_g: float,
                           Ai: float, lam: float) -> float:
        """
        Shadow price for gender g in service i.
        P_i^{H,g} = L^{1/φ+1/ρ} · D_i^g (L_i^g)^{-1/ρ} / (A_i λ)
        """
        p      = self.params
        L_term = safe_pow(clamp(self.L), 1.0 / p.phi + 1.0 / p.rho)
        zi     = clamp(Di_g * Li_g)
        return float(L_term * safe_pow(zi, -1.0 / p.rho) / (clamp(Ai) * clamp(lam)))

    def _prices_given_lambda(self, lam: float) -> tuple[float, float, float, float, float]:
        """
        Given λ, compute PcH, PdH, Pc, Pd, B.
        We use the *man's* shadow price; at the optimum both genders agree.
        """
        p    = self.params
        PcH  = self._shadow_price_home(self.Lc_m, p.D_c_m, self.A_c, lam)
        PdH  = self._shadow_price_home(self.Ld_m, p.D_d_m, self.A_d, lam)
        Pc   = ces_unit_cost(PcH, self.pc, p.omega_c, p.eta_c)
        Pd   = ces_unit_cost(PdH, self.pd, p.omega_d, p.eta_d)
        B    = pigl_B(Pc, Pd, p.beta_c, p.beta_d)
        return PcH, PdH, Pc, Pd, B

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 – solve for λ from fixed-point  λ = (1/B)(E/B)^{ε-1}   (C1)
    # ─────────────────────────────────────────────────────────────────────────

    def solve_lambda(self) -> None:
        self.compute_E()
        self.compute_L_aggregate()

        eps = self.params.eps_engel

        def f(lam: float) -> float:
            _, _, _, _, B = self._prices_given_lambda(lam)
            return lam - pigl_lambda(self.E, B, eps)

        lo, hi = 1e-12, 1e6
        flo, fhi = f(lo), f(hi)
        it = 0
        while flo * fhi > 0 and it < 60:
            hi  *= 10.0
            fhi  = f(hi)
            it  += 1
        if flo * fhi > 0:
            raise RuntimeError("Could not bracket λ root; check parameters/inputs.")

        self.lam = float(bisect_root(f, lo, hi))
        self.PcH, self.PdH, self.Pc, self.Pd, self.B = \
            self._prices_given_lambda(self.lam)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 – PIGL demands   (D7, P1-P3)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_pigl_demands(self) -> None:
        p = self.params
        self.th_x, self.th_c, self.th_d = pigl_shares(
            self.E, self.B, p.eps_engel,
            p.beta_x, p.beta_c, p.beta_d,
            p.kappa_x, p.kappa_c, p.kappa_d,
        )
        self.x  = float(self.th_x * self.E)
        self.Sc = float(self.th_c * self.E / clamp(self.Pc))
        self.Sd = float(self.th_d * self.E / clamp(self.Pd))

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6 – conditional input demands   (37, 38)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_input_splits(self) -> None:
        p = self.params
        self.ScM = float((1.0 - p.omega_c)
                         * safe_pow(self.Pc / clamp(self.pc), p.eta_c) * self.Sc)
        self.SdM = float((1.0 - p.omega_d)
                         * safe_pow(self.Pd / clamp(self.pd), p.eta_d) * self.Sd)
        self.ScH = float(p.omega_c
                         * safe_pow(self.Pc / clamp(self.PcH), p.eta_c) * self.Sc)
        self.SdH = float(p.omega_d
                         * safe_pow(self.Pd / clamp(self.PdH), p.eta_d) * self.Sd)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7 – 6-equation residual system
    #
    # (1) Market work FOC – man
    #     λ y_m = L^{1/φ+1/ρ} D_M^m (L_M^m)^{-1/ρ}
    #
    # (2) Market work FOC – woman
    #     λ y_f = L^{1/φ+1/ρ} D_M^f (L_M^f)^{-1/ρ}
    #
    # (3) Home care identity – man (from conditional demand + home prod.)
    #     A_c L_c^m = ScH · [D_c^m (L_c^m)^{-1/ρ}]
    #                      / [D_c^m (L_c^m)^{-1/ρ} + D_c^f (L_c^f)^{-1/ρ}]
    #     i.e. man's share of ScH proportional to his marginal disutility weight
    #
    # (4) Home care identity – woman (symmetric)
    #
    # (5) Home domestic identity – man
    #
    # (6) Home domestic identity – woman
    #
    # Equations (3)-(6) combine:
    #   S_iH = A_i (L_i^m + L_i^f)                    (home production)
    #   S_iH^g / S_iH = D_i^g (L_i^g)^{-1/ρ}           (intrahousehold split L2/L3)
    #                   / Σ_{g'} D_i^{g'} (L_i^{g'})^{-1/ρ}
    #
    # The fixed-point update naturally uses:
    #   Total:  L_i^m + L_i^f  = ScH / A_i
    #   Split:  L_i^m / L_i^f  = (D_i^m / D_i^f)^ρ    (from L2/L3 rearranged)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_labor_residuals(self) -> None:
        p   = self.params
        rho = p.rho
        phi = p.phi

        L_term = safe_pow(clamp(self.L), 1.0 / phi + 1.0 / rho)

        # FOC market work (man, woman)
        eq1 = (self.lam * self.y_m
               - L_term * safe_pow(clamp(p.D_M_m * self.LM_m), -1.0 / rho))
        eq2 = (self.lam * self.y_f
               - L_term * safe_pow(clamp(p.D_M_f * self.LM_f), -1.0 / rho))

        # Home input identities: total hours consistent with ScH / A_i
        total_Lc = self.ScH / clamp(self.A_c)
        total_Ld = self.SdH / clamp(self.A_d)

        # Gender split from L2 / L3:  L_i^m / L_i^f = (D_i^m / D_i^f)^ρ
        # => L_i^m = total * ratio / (1 + ratio),  L_i^f = total / (1 + ratio)
        ratio_c = safe_pow(clamp(p.D_c_m / p.D_c_f), rho)
        ratio_d = safe_pow(clamp(p.D_d_m / p.D_d_f), rho)

        Lc_m_star = total_Lc * ratio_c / (1.0 + ratio_c)
        Lc_f_star = total_Lc / (1.0 + ratio_c)
        Ld_m_star = total_Ld * ratio_d / (1.0 + ratio_d)
        Ld_f_star = total_Ld / (1.0 + ratio_d)

        eq3 = self.Lc_m - Lc_m_star
        eq4 = self.Lc_f - Lc_f_star
        eq5 = self.Ld_m - Ld_m_star
        eq6 = self.Ld_f - Ld_f_star

        self.residuals = np.array([eq1, eq2, eq3, eq4, eq5, eq6], dtype=float)

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_from_labor(self,
                            LM_m: float, LM_f: float,
                            Lc_m: float, Lc_f: float,
                            Ld_m: float, Ld_f: float,
                            ) -> np.ndarray:
        """
        One call to evaluate everything and return the 6-residual vector.
        Also stores all intermediate objects as attributes.
        """
        self.set_labor(LM_m, LM_f, Lc_m, Lc_f, Ld_m, Ld_f)
        self.solve_lambda()
        self.compute_pigl_demands()
        self.compute_input_splits()
        self.compute_labor_residuals()
        return self.residuals

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience aggregates for reporting / plotting
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def LM(self) -> float:
        """Total market hours (man + woman)."""
        return float(self.LM_m + self.LM_f)

    @property
    def Lc(self) -> float:
        """Total home care hours."""
        return float(self.Lc_m + self.Lc_f)

    @property
    def Ld(self) -> float:
        """Total home domestic hours."""
        return float(self.Ld_m + self.Ld_f)
