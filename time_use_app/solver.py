#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 15:20:43 2026

@author: slepot
"""


# solver.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np


@dataclass
class SolverState:
    """
    Stores everything related to numeric solving:
    - convergence status
    - iteration history
    - final solution
    - diagnostics + plotting/report helpers
    """
    method: str = "fixed_point"
    max_iter: int = 5000
    tol: float = 1e-10

    # fixed-point controls
    damping: float = 0.2          # relaxation weight ω in (0,1]
    min_damping: float = 1e-4
    adapt_damping: bool = True    # shrink ω if things get worse
    shrink: float = 0.5
    grow: float = 1.05
    grow_cap: float = 0.9

    # stopping rule uses both:
    tol_residual: Optional[float] = None   # if None, uses tol
    tol_step: Optional[float] = None       # if None, uses tol

    verbose: bool = True

    # Results
    converged: bool = False
    status: str = "not_started"
    n_iter: int = 0

    # final values (labor space)
    LM: float = np.nan
    Lc: float = np.nan
    Ld: float = np.nan
    res: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    res_norm: float = np.nan
    step_norm: float = np.nan

    # History
    iters: List[int] = field(default_factory=list)
    u_hist: List[np.ndarray] = field(default_factory=list)      # log labor
    L_hist: List[np.ndarray] = field(default_factory=list)      # (LM,Lc,Ld)
    res_hist: List[np.ndarray] = field(default_factory=list)    # residual vectors
    norm_hist: List[float] = field(default_factory=list)        # ||res||
    step_hist: List[float] = field(default_factory=list)        # ||ΔL||

    # If something goes wrong
    last_exception: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.converged = False
        self.status = "not_started"
        self.n_iter = 0
        self.LM = self.Lc = self.Ld = np.nan
        self.res = np.full(3, np.nan)
        self.res_norm = np.nan
        self.step_norm = np.nan

        self.iters.clear()
        self.u_hist.clear()
        self.L_hist.clear()
        self.res_hist.clear()
        self.norm_hist.clear()
        self.step_hist.clear()

        self.last_exception = None
        self.extra.clear()

    def record(self, k: int, L: np.ndarray, res: np.ndarray, step_norm: float) -> None:
        self.iters.append(int(k))
        self.L_hist.append(L.copy())
        self.u_hist.append(np.log(L).copy())
        self.res_hist.append(res.copy())
        self.norm_hist.append(float(np.linalg.norm(res)))
        self.step_hist.append(float(step_norm))

    def print_solver_report(self) -> None:
        tol_r = self.tol_residual if self.tol_residual is not None else self.tol
        tol_s = self.tol_step if self.tol_step is not None else self.tol

        lines = []
        lines.append("=== Solver report ===")
        lines.append(f"status        : {self.status}")
        lines.append(f"converged     : {self.converged}")
        lines.append(f"method        : {self.method}")
        lines.append(f"iterations    : {self.n_iter}")
        lines.append(f"tol_residual  : {tol_r:g}")
        lines.append(f"tol_step      : {tol_s:g}")
        lines.append(f"damping       : {self.damping:g}")
        lines.append(f"res_norm      : {self.res_norm:.6e}")
        lines.append(f"step_norm     : {self.step_norm:.6e}")
        lines.append(f"LM,Lc,Ld      : {self.LM:.6g}, {self.Lc:.6g}, {self.Ld:.6g}")
        if self.last_exception is not None:
            lines.append(f"last_error    : {self.last_exception}")
        print("\n".join(lines))

    def plot_convergence(self, logy: bool = True) -> None:
        if len(self.norm_hist) == 0:
            print("No history to plot.")
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available in this environment.")
            return

        x = np.array(self.iters, dtype=int)
        y1 = np.array(self.norm_hist, dtype=float)
        y2 = np.array(self.step_hist, dtype=float)

        plt.figure()
        plt.plot(x, y1, label="||res||")
        plt.plot(x, y2, label="||ΔL||")
        if logy:
            plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Norm")
        plt.title("Fixed-point convergence")
        plt.legend()
        plt.show()


# -------------------------
# Helpers
# -------------------------

def _ensure_positive(L: np.ndarray, floor: float = 1e-14) -> np.ndarray:
    L2 = np.asarray(L, dtype=float).copy()
    L2[~np.isfinite(L2)] = np.nan
    if np.any(np.isnan(L2)):
        raise ValueError("Non-finite labor guess encountered.")
    return np.maximum(L2, floor)


# -------------------------
# Public API
# -------------------------

def solve_model(model_params,
                household,
                state: SolverState,
                L0: Optional[Tuple[float, float, float]] = None):
    """
    Fixed-point solver over labor allocations (LM,Lc,Ld) only.

    Requires:
      household.evaluate_from_labor(LM,Lc,Ld) -> residuals (len 3)
    and after that call, household must have attributes:
      household.lam, household.y, household.L,
      household.params.D_M, household.params.rho, household.params.phi,
      household.ScH, household.SdH, household.A_c, household.A_d

    Returns (household, state) with household evaluated at final L.
    """
    state.reset()
    state.method = "fixed_point"
    state.status = "running"

    tol_r = state.tol_residual if state.tol_residual is not None else state.tol
    tol_s = state.tol_step if state.tol_step is not None else state.tol

    # initial guess
    if L0 is None:
        L = np.array([0.2, 0.1, 0.1], dtype=float)
    else:
        L = np.array(L0, dtype=float)

    L = _ensure_positive(L)

    # track best seen (optional but helpful)
    best = {"norm": np.inf, "L": L.copy()}

    prev_norm = np.inf
    omega = float(state.damping)

    for k in range(state.max_iter):
        try:
            res = np.asarray(household.evaluate_from_labor(float(L[0]), float(L[1]), float(L[2])), dtype=float)
        except Exception as e:
            state.last_exception = repr(e)
            state.status = "failed_eval"
            state.n_iter = k
            break

        norm = float(np.linalg.norm(res))
        if norm < best["norm"]:
            best["norm"] = norm
            best["L"] = L.copy()

        # Build fixed-point update g(L) from rearranged equilibrium conditions
        try:
            rho = float(household.params.rho)
            phi = float(household.params.phi)
            D_M = float(household.params.D_M)

            lam = float(household.lam)
            y = float(household.y)

            # L_term = L^{1/φ + 1/ρ}
            Lagg = float(household.L)
            L_term = Lagg ** (1.0 / phi + 1.0 / rho)

            # From λ y = L_term * (D_M L_M)^(-1/ρ)
            # => D_M L_M = (L_term / (λ y))^{ρ}
            LM_new = (L_term / (lam * y)) ** rho / D_M

            # From A_i L_i = S_iH
            Lc_new = float(household.ScH) / float(household.A_c)
            Ld_new = float(household.SdH) / float(household.A_d)

            L_new = np.array([LM_new, Lc_new, Ld_new], dtype=float)
            L_new = _ensure_positive(L_new)

        except Exception as e:
            state.last_exception = repr(e)
            state.status = "failed_update_map"
            state.n_iter = k
            break

        # Damped update
        L_next = (1.0 - omega) * L + omega * L_new
        L_next = _ensure_positive(L_next)

        step_norm = float(np.linalg.norm(L_next - L))
        state.record(k, L, res, step_norm)

        if state.verbose:
            print(f"[{k:04d}] ||res||={norm:.3e}  ||ΔL||={step_norm:.3e}  ω={omega:.3g}  L={L}")

        # Convergence check
        if (norm < tol_r) and (step_norm < tol_s):
            state.converged = True
            state.status = "converged"
            state.n_iter = k
            L = L_next
            break

        # Optional adaptive damping: if residual norm worsens, shrink ω; else maybe grow
        if state.adapt_damping:
            if np.isfinite(prev_norm):
                if norm > prev_norm * 1.01:   # got worse
                    omega = max(state.min_damping, omega * state.shrink)
                else:
                    omega = min(state.grow_cap, omega * state.grow)

        prev_norm = norm
        L = L_next

    # If not converged, fall back to best-seen point for final evaluation
    if not state.converged:
        state.status = state.status if state.status != "running" else "max_iter"
        state.n_iter = state.n_iter if state.n_iter != 0 else state.max_iter
        L = best["L"]

    # Final evaluation at chosen L so household attributes are consistent
    try:
        res_final = np.asarray(household.evaluate_from_labor(float(L[0]), float(L[1]), float(L[2])), dtype=float)
        state.res = res_final
        state.res_norm = float(np.linalg.norm(res_final))
        state.LM, state.Lc, state.Ld = float(L[0]), float(L[1]), float(L[2])
        # step_norm: last recorded if exists
        state.step_norm = float(state.step_hist[-1]) if len(state.step_hist) else np.nan

        if (not state.converged) and (state.res_norm < tol_r):
            # residual criterion met at best point even if step criterion wasn’t
            state.status = "residual_small_best_point"

    except Exception as e:
        state.last_exception = repr(e)
        state.status = "failed_final_eval"
        state.converged = False

    return household, state
