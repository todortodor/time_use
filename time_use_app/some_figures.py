# make_figures.py
"""
Generate the model output graphs shown in the attached spec:

1) Hours worked vs human capital h (participants only): L_M, L_c, L_d
2) Labor force participation probability P(h) vs h
3) Expenditure shares vs total expenditure E: theta_x, theta_c, theta_d
4) Home vs market service shares vs h: S_iH/S_i and S_iM/S_i for i in {c,d}
5) Value functions Vm(h) and Vn(h) vs h (participant vs nonparticipant)
6) Optional policy comparative static: ΔP(h) when pc is reduced by 20%

Assumptions (because your current core code is static, and participation/value functions
aren’t fully specified there):
- Indirect utility from PIGL implied by lambda:
    dV/dE = lambda = 1/B * (E/B)^{eps-1}  =>  V_goods = (E/B)^eps / eps
- Disutility of time aggregate uses:
    V_time = L^{1 + 1/phi} / (1 + 1/phi)
- Total value V = V_goods - V_time
- Participation: compare "participant" (solve full labor system) vs "nonparticipant" (LM=0)
- Participation probability uses a logit smoothing:
    P(h) = 1 / (1 + exp(-(Vm - Vn)/sigma_part))
  (sigma_part is a plotting/behavioral smoothing parameter; set below)

This file expects the following modules in the same folder:
- classes.py  (ModelParams, Household)
- solver.py   (SolverState, solve_model)  [fixed-point solver version]
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from classes import ModelParams, Household
from solver import SolverState, solve_model


# -----------------------------
# User-configurable settings
# -----------------------------

@dataclass
class PlotConfig:
    # Human capital grid
    h_min: float = 0.2
    h_max: float = 5.0
    n_h: int = 60
    x_axis: str = "log"  # "log" or "level"

    # Wage mapping y(h) = w0 * h^w_elast
    w0: float = 20.0
    w_elast: float = 1.0

    # Exogenous prices + non-labor income (baseline)
    pc: float = 80.0
    pd: float = 40.0
    a: float = 200.0

    # Home productivity mapping (optional)
    # A_i(h) = A_i0 * h^A_elast_i
    A_c0: float = 1.0
    A_d0: float = 1.0
    A_c_elast: float = 0.0
    A_d_elast: float = 0.0

    # Participation probability smoothing (logit scale)
    sigma_part: float = 5.0

    # Solver controls
    fp_tol: float = 1e-10
    fp_max_iter: int = 20000
    fp_damping: float = 0.2

    # Nonparticipant inner loop (LM fixed to 0)
    np_max_iter: int = 20000
    np_tol: float = 1e-10
    np_damping: float = 0.2

    # Output
    out_dir: str = "figures"
    save_png: bool = True
    show_plots: bool = True


# -----------------------------
# Utility and participation
# -----------------------------

def value_from_household(hh: Household) -> float:
    """
    Compute Vm or Vn from the evaluated household.

    V_goods = (E/B)^eps / eps
    V_time  = L^{1 + 1/phi} / (1 + 1/phi)
    V = V_goods - V_time
    """
    eps = float(hh.params.eps_engel)
    phi = float(hh.params.phi)

    E = float(hh.E)
    B = float(hh.B)
    L = float(hh.L)

    V_goods = (E / B) ** eps / eps
    V_time = L ** (1.0 + 1.0 / phi) / (1.0 + 1.0 / phi)
    return float(V_goods - V_time)


def participation_prob(Vm: float, Vn: float, sigma: float) -> float:
    """Logit smoothing for participation probability."""
    z = (Vm - Vn) / max(1e-12, sigma)
    # stable logistic
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


# -----------------------------
# Nonparticipant solver (LM=0)
# -----------------------------

def solve_nonparticipant(hh: Household, cfg: PlotConfig, Lc0: float = 0.1, Ld0: float = 0.1) -> Tuple[Household, bool]:
    """
    Solve "nonparticipant" allocation with LM fixed to 0.
    We iterate only on (Lc, Ld) using the same internal lambda fixed point etc.

    Fixed-point map (LM=0):
      Lc <- ScH / A_c
      Ld <- SdH / A_d
    with damping.
    """
    LM_fixed = 1e-16  # keep strictly positive to avoid any power issues
    Lc = max(1e-14, float(Lc0))
    Ld = max(1e-14, float(Ld0))

    omega = float(cfg.np_damping)
    prev_step = np.inf

    for _ in range(cfg.np_max_iter):
        # Evaluate household at (LM_fixed, Lc, Ld)
        _ = hh.evaluate_from_labor(LM_fixed, Lc, Ld)

        # Update map using home-input identities
        Lc_new = float(hh.ScH) / float(hh.A_c)
        Ld_new = float(hh.SdH) / float(hh.A_d)

        Lc_new = max(1e-14, Lc_new)
        Ld_new = max(1e-14, Ld_new)

        Lc_next = (1.0 - omega) * Lc + omega * Lc_new
        Ld_next = (1.0 - omega) * Ld + omega * Ld_new

        step = float(np.linalg.norm([Lc_next - Lc, Ld_next - Ld]))

        if step < cfg.np_tol and prev_step < cfg.np_tol:
            # One more evaluation to store consistent objects
            _ = hh.evaluate_from_labor(LM_fixed, Lc_next, Ld_next)
            return hh, True

        prev_step = step
        Lc, Ld = Lc_next, Ld_next

    # Final eval at last guess
    _ = hh.evaluate_from_labor(LM_fixed, Lc, Ld)
    return hh, False


# -----------------------------
# Model parameter baseline
# -----------------------------

def baseline_model_params() -> ModelParams:
    """
    Baseline params (hard-coded from the earlier calibration you referenced).
    If your pdf baseline differs, update these numbers to match.
    """
    return ModelParams(
        eps_engel=0.30,
        beta_x=0.50,
        beta_c=0.25,
        beta_d=0.25,
        kappa_x=0.15,
        kappa_c=-0.05,
        kappa_d=-0.10,
        omega_c=0.75,
        omega_d=0.70,
        eta_c=2.5,
        eta_d=3.0,
        D_M=1.0,
        D_c=1.2,
        D_d=0.9,
        rho=-0.50,
        phi=0.50,
    )


# -----------------------------
# Main figure generation
# -----------------------------

def run_all_figures(cfg: PlotConfig) -> Dict[str, np.ndarray]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    mp = baseline_model_params()

    # Grid of human capital
    h_grid = np.linspace(cfg.h_min, cfg.h_max, cfg.n_h)
    x = np.log(h_grid) if cfg.x_axis == "log" else h_grid

    # Storage
    LM = np.full(cfg.n_h, np.nan)
    Lc = np.full(cfg.n_h, np.nan)
    Ld = np.full(cfg.n_h, np.nan)

    P = np.full(cfg.n_h, np.nan)
    Vm = np.full(cfg.n_h, np.nan)
    Vn = np.full(cfg.n_h, np.nan)

    E = np.full(cfg.n_h, np.nan)
    thx = np.full(cfg.n_h, np.nan)
    thc = np.full(cfg.n_h, np.nan)
    thd = np.full(cfg.n_h, np.nan)

    ScH_share = np.full(cfg.n_h, np.nan)
    ScM_share = np.full(cfg.n_h, np.nan)
    SdH_share = np.full(cfg.n_h, np.nan)
    SdM_share = np.full(cfg.n_h, np.nan)

    # Solver state (participant)
    state = SolverState(
        verbose=False,
        max_iter=cfg.fp_max_iter,
        tol=cfg.fp_tol,
        damping=cfg.fp_damping,
        adapt_damping=True,
    )

    # Warm start guesses for participant and nonparticipant
    L_guess = (0.2, 0.1, 0.1)
    Lc_guess_np, Ld_guess_np = 0.1, 0.1

    for i, h in enumerate(h_grid):
        y = cfg.w0 * (h ** cfg.w_elast)
        A_c = cfg.A_c0 * (h ** cfg.A_c_elast)
        A_d = cfg.A_d0 * (h ** cfg.A_d_elast)

        # Participant household
        hh = Household(
            params=mp,
            y=float(y),
            pc=float(cfg.pc),
            pd=float(cfg.pd),
            a=float(cfg.a),
            A_c=float(A_c),
            A_d=float(A_d),
        )

        hh, state = solve_model(mp, hh, state, L0=L_guess)

        # Store participant solution (even if not converged, store best point)
        LM[i], Lc[i], Ld[i] = state.LM, state.Lc, state.Ld
        L_guess = (max(1e-14, LM[i]), max(1e-14, Lc[i]), max(1e-14, Ld[i]))

        Vm[i] = value_from_household(hh)

        # Nonparticipant value: LM fixed 0
        hh_np = Household(
            params=mp,
            y=float(y),
            pc=float(cfg.pc),
            pd=float(cfg.pd),
            a=float(cfg.a),
            A_c=float(A_c),
            A_d=float(A_d),
        )
        hh_np, ok_np = solve_nonparticipant(hh_np, cfg, Lc0=Lc_guess_np, Ld0=Ld_guess_np)
        Vn[i] = value_from_household(hh_np)

        # update nonparticipant warm start
        Lc_guess_np, Ld_guess_np = float(hh_np.Lc), float(hh_np.Ld)

        # Participation probability (smoothed)
        P[i] = participation_prob(Vm[i], Vn[i], cfg.sigma_part)

        # Expenditure and shares (participant)
        E[i] = float(hh.E)
        thx[i], thc[i], thd[i] = float(hh.th_x), float(hh.th_c), float(hh.th_d)

        # Home vs market shares (participant)
        # Shares are well-defined only if totals > 0
        Sc = float(hh.Sc)
        Sd = float(hh.Sd)
        ScH_share[i] = float(hh.ScH) / Sc if Sc > 0 else np.nan
        ScM_share[i] = float(hh.ScM) / Sc if Sc > 0 else np.nan
        SdH_share[i] = float(hh.SdH) / Sd if Sd > 0 else np.nan
        SdM_share[i] = float(hh.SdM) / Sd if Sd > 0 else np.nan

    # -----------------------------
    # 1) Hours worked vs h (participants)
    # -----------------------------
    plt.figure()
    plt.plot(x, LM, label=r"$L^M$")
    plt.plot(x, Lc, label=r"$L^c$")
    plt.plot(x, Ld, label=r"$L^d$")
    plt.xlabel("log(h)" if cfg.x_axis == "log" else "h")
    plt.ylabel("Hours / time allocation")
    plt.title("Hours in each activity vs human capital (participants)")
    plt.legend()
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "hours_vs_h.png"), dpi=200, bbox_inches="tight")

    # -----------------------------
    # 2) Participation probability P(h)
    # -----------------------------
    plt.figure()
    plt.plot(x, P)
    plt.xlabel("log(h)" if cfg.x_axis == "log" else "h")
    plt.ylabel("Participation probability")
    plt.title("Labor force participation probability vs human capital")
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "participation_vs_h.png"), dpi=200, bbox_inches="tight")

    # -----------------------------
    # 3) Expenditure shares vs E
    # -----------------------------
    # Sort by E for a clean x-axis
    order = np.argsort(E)
    plt.figure()
    plt.plot(E[order], thx[order], label=r"$\vartheta_x$")
    plt.plot(E[order], thc[order], label=r"$\vartheta_c$")
    plt.plot(E[order], thd[order], label=r"$\vartheta_d$")
    plt.xlabel("Total expenditure E")
    plt.ylabel("Budget share")
    plt.title("Expenditure shares vs total expenditure (PIGL non-homotheticity)")
    plt.legend()
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "shares_vs_E.png"), dpi=200, bbox_inches="tight")

    # -----------------------------
    # 4) Home vs market service shares vs h
    # -----------------------------
    plt.figure()
    plt.plot(x, ScH_share, label=r"$S_{cH}/S_c$")
    plt.plot(x, ScM_share, label=r"$S_{cM}/S_c$")
    plt.xlabel("log(h)" if cfg.x_axis == "log" else "h")
    plt.ylabel("Share")
    plt.title("Care services: home vs market share")
    plt.legend()
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "care_home_market_shares_vs_h.png"), dpi=200, bbox_inches="tight")

    plt.figure()
    plt.plot(x, SdH_share, label=r"$S_{dH}/S_d$")
    plt.plot(x, SdM_share, label=r"$S_{dM}/S_d$")
    plt.xlabel("log(h)" if cfg.x_axis == "log" else "h")
    plt.ylabel("Share")
    plt.title("Domestic services: home vs market share")
    plt.legend()
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "domestic_home_market_shares_vs_h.png"), dpi=200, bbox_inches="tight")

    # -----------------------------
    # 5) Value functions Vm(h), Vn(h)
    # -----------------------------
    plt.figure()
    plt.plot(x, Vm, label=r"$V_m(h)$")
    plt.plot(x, Vn, label=r"$V_n(h)$")
    plt.xlabel("log(h)" if cfg.x_axis == "log" else "h")
    plt.ylabel("Value")
    plt.title("Value functions vs human capital")
    plt.legend()
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "value_functions_vs_h.png"), dpi=200, bbox_inches="tight")

    # -----------------------------
    # 6) Optional policy: reduce pc by 20% and plot ΔP(h)
    # -----------------------------
    P_policy = np.full(cfg.n_h, np.nan)
    pc_new = 0.8 * cfg.pc

    # Reuse warm starts
    L_guess = (0.2, 0.1, 0.1)
    Lc_guess_np, Ld_guess_np = 0.1, 0.1

    for i, h in enumerate(h_grid):
        y = cfg.w0 * (h ** cfg.w_elast)
        A_c = cfg.A_c0 * (h ** cfg.A_c_elast)
        A_d = cfg.A_d0 * (h ** cfg.A_d_elast)

        # Participant under policy
        hh = Household(
            params=mp,
            y=float(y),
            pc=float(pc_new),
            pd=float(cfg.pd),
            a=float(cfg.a),
            A_c=float(A_c),
            A_d=float(A_d),
        )
        hh, state = solve_model(mp, hh, state, L0=L_guess)
        L_guess = (max(1e-14, state.LM), max(1e-14, state.Lc), max(1e-14, state.Ld))
        Vm_pol = value_from_household(hh)

        # Nonparticipant under policy
        hh_np = Household(
            params=mp,
            y=float(y),
            pc=float(pc_new),
            pd=float(cfg.pd),
            a=float(cfg.a),
            A_c=float(A_c),
            A_d=float(A_d),
        )
        hh_np, _ = solve_nonparticipant(hh_np, cfg, Lc0=Lc_guess_np, Ld0=Ld_guess_np)
        Vn_pol = value_from_household(hh_np)
        Lc_guess_np, Ld_guess_np = float(hh_np.Lc), float(hh_np.Ld)

        P_policy[i] = participation_prob(Vm_pol, Vn_pol, cfg.sigma_part)

    dP = P_policy - P

    plt.figure()
    plt.plot(x, dP)
    plt.xlabel("log(h)" if cfg.x_axis == "log" else "h")
    plt.ylabel(r"$\Delta P(h)$")
    plt.title("Change in participation when care price is reduced by 20%")
    if cfg.save_png:
        plt.savefig(os.path.join(cfg.out_dir, "delta_participation_policy_pc_minus20.png"), dpi=200, bbox_inches="tight")

    if cfg.show_plots:
        plt.show()
    else:
        plt.close("all")

    return {
        "h": h_grid,
        "x": x,
        "LM": LM,
        "Lc": Lc,
        "Ld": Ld,
        "P": P,
        "Vm": Vm,
        "Vn": Vn,
        "E": E,
        "thx": thx,
        "thc": thc,
        "thd": thd,
        "ScH_share": ScH_share,
        "ScM_share": ScM_share,
        "SdH_share": SdH_share,
        "SdM_share": SdM_share,
        "P_policy": P_policy,
    }


if __name__ == "__main__":
    cfg = PlotConfig(
        # You can flip between log/level of h here
        x_axis="log",
        # If you want a wider wage range:
        # w0=15.0, w_elast=1.2,
        out_dir="figures",
        save_png=True,
        show_plots=True,
    )
    run_all_figures(cfg)
