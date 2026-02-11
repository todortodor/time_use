# one_solve.py
from __future__ import annotations

import numpy as np

from classes import ModelParams, Household
from solver import SolverState, solve_model


def run_one_solve() -> tuple[Household, SolverState]:
    """
    Solve the model once using the baseline parameter values in the PDF.

    Baseline calibration (Section 5.1–5.3) and illustrative exogenous values (Section 5.4):
      ε=0.30
      βx=0.50, βc=0.25, βd=0.25
      κx=0.15, κc=-0.05, κd=-0.10
      ωc=0.75, ωd=0.70
      ηc=2.5, ηd=3.0
      Ac=1.0, Ad=1.0
      DM=1.0, Dc=1.2, Dd=0.9
      φ=0.50, ρ=-0.50
      y=50, pc=80, pd=40, a=200
    """
    # ---- Model parameters (true/global) ----
    mp = ModelParams(
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

    # ---- Household instance (exogenous vars + productivities) ----
    hh = Household(
        params=mp,
        y=50.0,
        pc=80.0,
        pd=40.0,
        a=200.0,
        A_c=1.0,
        A_d=1.0,
    )

    # ---- Solver configuration ----
    state = SolverState(
        verbose=True,
        max_iter=20000,
        tol=1e-10,
        damping=0.20,          # initial relaxation weight
        adapt_damping=True,    # shrink/grow damping automatically
    )

    # Initial guess in labor space (LM, Lc, Ld) — must be strictly positive
    L0 = (0.20, 0.10, 0.10)

    # Solve (fixed-point solver over labor only)
    hh, state = solve_model(mp, hh, state, L0=L0)

    return hh, state


if __name__ == "__main__":
    hh, state = run_one_solve()

    print()
    state.print_solver_report()
    state.plot_convergence()

    print("\n=== Key household objects at solution ===")
    print(f"LM, Lc, Ld      : {hh.LM:.6g}, {hh.Lc:.6g}, {hh.Ld:.6g}")
    print(f"E              : {hh.E:.6g}")
    print(f"lambda         : {hh.lam:.6g}")
    print(f"PcH, PdH       : {hh.PcH:.6g}, {hh.PdH:.6g}")
    print(f"Pc, Pd         : {hh.Pc:.6g}, {hh.Pd:.6g}")
    print(f"B(p)           : {hh.B:.6g}")
    print(f"theta_x,c,d    : {hh.th_x:.6g}, {hh.th_c:.6g}, {hh.th_d:.6g}")
    print(f"x, Sc, Sd      : {hh.x:.6g}, {hh.Sc:.6g}, {hh.Sd:.6g}")
    print(f"ScH, ScM       : {hh.ScH:.6g}, {hh.ScM:.6g}")
    print(f"SdH, SdM       : {hh.SdH:.6g}, {hh.SdM:.6g}")

    # Optional plots
    # state.plot_convergence(logy=True)
