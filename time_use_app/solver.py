#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver.py  –  fixed-point solver over the 6-dimensional labor space
              (LM_m, LM_f, Lc_m, Lc_f, Ld_m, Ld_f).

The solver structure mirrors the baseline exactly:
  • SolverState records history and diagnostics.
  • solve_model() runs the damped fixed-point iteration.

Fixed-point update map  g : R^6 → R^6
──────────────────────────────────────
From the equilibrium conditions:

  (1) λ y_m = L_term · D_M^m (L_M^m)^{-1/ρ}
      → L_M^m_new = (L_term / (λ y_m))^ρ / D_M^m

  (2) λ y_f = L_term · D_M^f (L_M^f)^{-1/ρ}
      → L_M^f_new = (L_term / (λ y_f))^ρ / D_M^f

  (3)-(6) Home identities + gender split (L2/L3):
      L_c_total = ScH / A_c
      L_c^m / L_c^f = (D_c^m / D_c^f)^ρ
      → L_c^m_new = L_c_total · ratio_c / (1 + ratio_c)
      → L_c^f_new = L_c_total / (1 + ratio_c)
      (analogously for domestic)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# State object
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverState:
    """
    Stores convergence status, iteration history, and final solution.
    The solution lives in the 6-dimensional labor space.
    """
    method:   str   = "fixed_point"
    max_iter: int   = 5000
    tol:      float = 1e-10

    # fixed-point controls
    damping:      float = 0.2
    min_damping:  float = 1e-4
    adapt_damping: bool = True
    shrink:       float = 0.5
    grow:         float = 1.05
    grow_cap:     float = 0.9

    tol_residual: Optional[float] = None
    tol_step:     Optional[float] = None

    verbose: bool = True

    # Results
    converged: bool  = False
    status:    str   = "not_started"
    n_iter:    int   = 0

    # Final labor values
    LM_m: float = np.nan
    LM_f: float = np.nan
    Lc_m: float = np.nan
    Lc_f: float = np.nan
    Ld_m: float = np.nan
    Ld_f: float = np.nan

    res:       np.ndarray = field(default_factory=lambda: np.full(6, np.nan))
    res_norm:  float      = np.nan
    step_norm: float      = np.nan

    # History
    iters:     List[int]         = field(default_factory=list)
    L_hist:    List[np.ndarray]  = field(default_factory=list)
    res_hist:  List[np.ndarray]  = field(default_factory=list)
    norm_hist: List[float]       = field(default_factory=list)
    step_hist: List[float]       = field(default_factory=list)

    last_exception: Optional[str]    = None
    extra:          Dict[str, Any]   = field(default_factory=dict)

    def reset(self) -> None:
        self.converged = False
        self.status    = "not_started"
        self.n_iter    = 0
        self.LM_m = self.LM_f = self.Lc_m = self.Lc_f = self.Ld_m = self.Ld_f = np.nan
        self.res       = np.full(6, np.nan)
        self.res_norm  = np.nan
        self.step_norm = np.nan
        self.iters.clear()
        self.L_hist.clear()
        self.res_hist.clear()
        self.norm_hist.clear()
        self.step_hist.clear()
        self.last_exception = None
        self.extra.clear()

    def record(self, k: int, L: np.ndarray, res: np.ndarray, step_norm: float) -> None:
        self.iters.append(int(k))
        self.L_hist.append(L.copy())
        self.res_hist.append(res.copy())
        self.norm_hist.append(float(np.linalg.norm(res)))
        self.step_hist.append(float(step_norm))

    def print_solver_report(self) -> None:
        tol_r = self.tol_residual if self.tol_residual is not None else self.tol
        tol_s = self.tol_step    if self.tol_step    is not None else self.tol
        lines = [
            "=== Solver report ===",
            f"status        : {self.status}",
            f"converged     : {self.converged}",
            f"method        : {self.method}",
            f"iterations    : {self.n_iter}",
            f"tol_residual  : {tol_r:g}",
            f"tol_step      : {tol_s:g}",
            f"damping       : {self.damping:g}",
            f"res_norm      : {self.res_norm:.6e}",
            f"step_norm     : {self.step_norm:.6e}",
            f"LM_m,LM_f     : {self.LM_m:.6g}, {self.LM_f:.6g}",
            f"Lc_m,Lc_f     : {self.Lc_m:.6g}, {self.Lc_f:.6g}",
            f"Ld_m,Ld_f     : {self.Ld_m:.6g}, {self.Ld_f:.6g}",
        ]
        if self.last_exception is not None:
            lines.append(f"last_error    : {self.last_exception}")
        print("\n".join(lines))

    def plot_convergence(self, logy: bool = True) -> None:
        if not self.norm_hist:
            print("No history to plot.")
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available.")
            return
        x  = np.array(self.iters, dtype=int)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_positive(L: np.ndarray, floor: float = 1e-14) -> np.ndarray:
    L2 = np.asarray(L, dtype=float).copy()
    L2[~np.isfinite(L2)] = np.nan
    if np.any(np.isnan(L2)):
        raise ValueError("Non-finite labor guess encountered.")
    return np.maximum(L2, floor)


def _fp_update(hh, rho: float, phi: float) -> np.ndarray:
    """
    Build the fixed-point update vector g(L) ∈ R^6.

    Equations:
      LM_g_new = (L_term / (λ y_g))^ρ / D_M^g          for g ∈ {m, f}
      L_i_total = S_iH / A_i
      L_i^m / L_i^f = (D_i^m / D_i^f)^ρ
    """
    from functions import safe_pow, clamp

    lam    = float(hh.lam)
    L_agg  = float(hh.L)
    L_term = L_agg ** (1.0 / phi + 1.0 / rho)

    p = hh.params

    # market work
    LM_m_new = safe_pow(L_term / (lam * clamp(hh.y_m)), rho) / p.D_M_m
    LM_f_new = safe_pow(L_term / (lam * clamp(hh.y_f)), rho) / p.D_M_f

    # home activities
    Lc_total = float(hh.ScH) / clamp(hh.A_c)
    Ld_total = float(hh.SdH) / clamp(hh.A_d)

    ratio_c  = safe_pow(clamp(p.D_c_m / p.D_c_f), rho)
    ratio_d  = safe_pow(clamp(p.D_d_m / p.D_d_f), rho)

    Lc_m_new = Lc_total * ratio_c / (1.0 + ratio_c)
    Lc_f_new = Lc_total           / (1.0 + ratio_c)
    Ld_m_new = Ld_total * ratio_d / (1.0 + ratio_d)
    Ld_f_new = Ld_total           / (1.0 + ratio_d)

    return np.array([LM_m_new, LM_f_new,
                     Lc_m_new, Lc_f_new,
                     Ld_m_new, Ld_f_new], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def solve_model(model_params,
                household,
                state: SolverState,
                L0: Optional[Tuple[float, ...]] = None):
    """
    Damped fixed-point solver over the 6-vector
      L = (LM_m, LM_f, Lc_m, Lc_f, Ld_m, Ld_f).

    Requires household.evaluate_from_labor(*L) → residuals (len 6).
    Returns (household, state) with household evaluated at final L.
    """
    state.reset()
    state.method = "fixed_point"
    state.status = "running"

    tol_r = state.tol_residual if state.tol_residual is not None else state.tol
    tol_s = state.tol_step    if state.tol_step    is not None else state.tol

    # initial guess: [LM_m, LM_f, Lc_m, Lc_f, Ld_m, Ld_f]
    if L0 is None:
        L = np.array([0.15, 0.10, 0.08, 0.10, 0.05, 0.08], dtype=float)
    else:
        L = np.array(L0, dtype=float)

    L = _ensure_positive(L)

    best      = {"norm": np.inf, "L": L.copy()}
    prev_norm = np.inf
    omega     = float(state.damping)

    rho = float(model_params.rho)
    phi = float(model_params.phi)

    for k in range(state.max_iter):
        try:
            res = np.asarray(
                household.evaluate_from_labor(
                    float(L[0]), float(L[1]),
                    float(L[2]), float(L[3]),
                    float(L[4]), float(L[5]),
                ), dtype=float)
        except Exception as e:
            state.last_exception = repr(e)
            state.status = "failed_eval"
            state.n_iter = k
            break

        norm = float(np.linalg.norm(res))
        if norm < best["norm"]:
            best["norm"] = norm
            best["L"]    = L.copy()

        try:
            L_new = _fp_update(household, rho, phi)
            L_new = _ensure_positive(L_new)
        except Exception as e:
            state.last_exception = repr(e)
            state.status = "failed_update_map"
            state.n_iter = k
            break

        L_next     = (1.0 - omega) * L + omega * L_new
        L_next     = _ensure_positive(L_next)
        step_norm  = float(np.linalg.norm(L_next - L))

        state.record(k, L, res, step_norm)

        if state.verbose:
            print(f"[{k:04d}] ||res||={norm:.3e}  ||ΔL||={step_norm:.3e}"
                  f"  ω={omega:.3g}"
                  f"  LM=({L[0]:.4g},{L[1]:.4g})"
                  f"  Lc=({L[2]:.4g},{L[3]:.4g})"
                  f"  Ld=({L[4]:.4g},{L[5]:.4g})")

        if (norm < tol_r) and (step_norm < tol_s):
            state.converged = True
            state.status    = "converged"
            state.n_iter    = k
            L = L_next
            break

        if state.adapt_damping and np.isfinite(prev_norm):
            if norm > prev_norm * 1.01:
                omega = max(state.min_damping, omega * state.shrink)
            else:
                omega = min(state.grow_cap,    omega * state.grow)

        prev_norm = norm
        L         = L_next

    # fallback to best-seen
    if not state.converged:
        state.status = state.status if state.status != "running" else "max_iter"
        state.n_iter = state.n_iter if state.n_iter  != 0        else state.max_iter
        L = best["L"]

    # final evaluation
    try:
        res_final = np.asarray(
            household.evaluate_from_labor(
                float(L[0]), float(L[1]),
                float(L[2]), float(L[3]),
                float(L[4]), float(L[5]),
            ), dtype=float)
        state.res       = res_final
        state.res_norm  = float(np.linalg.norm(res_final))
        state.LM_m, state.LM_f = float(L[0]), float(L[1])
        state.Lc_m, state.Lc_f = float(L[2]), float(L[3])
        state.Ld_m, state.Ld_f = float(L[4]), float(L[5])
        state.step_norm = float(state.step_hist[-1]) if state.step_hist else np.nan

        if (not state.converged) and (state.res_norm < tol_r):
            state.status = "residual_small_best_point"

    except Exception as e:
        state.last_exception = repr(e)
        state.status         = "failed_final_eval"
        state.converged      = False

    return household, state
