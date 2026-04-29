#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial.py  –  spatial equilibrium layer.

Given a list of County objects and ModelParams, this module:

  1. solve_counties()     – solves the household problem in every county,
                            records V*_ℓ = V(p_ℓ,E*_ℓ) − D(L*_ℓ).

  2. calibrate_amenities() – sets ξ_ℓ = Ū − V*_ℓ for each county, where
                             Ū is the population-weighted mean of V*_ℓ
                             (normalization: Σ N_ℓ ξ_ℓ = 0).

  3. check_no_arbitrage()  – verifies V*_ℓ + ξ_ℓ = Ū  ∀ ℓ.

  4. counterfactual()      – given perturbed counties (e.g. a care subsidy),
                             holds ξ_ℓ fixed and solves for new V'*_ℓ and
                             a new equilibrium utility level Ū'.
                             Optionally re-equilibrates population N'_ℓ so
                             that every household achieves Ū'.

Household solve
───────────────
For each county ℓ and each point on the h-grid, we create a Household with
  y_m = w_ℓ · h_m,  y_f = wage_gap · w_ℓ · h_f,
  pc = county.pc,   pd = county.pd,
  A_c = county.A_c_home, A_d = county.A_d_home
and run solve_model().

The county-level V*_ℓ is the value of a *representative* household at the
median h on the grid (or the average, configurable via agg="median"/"mean").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

from classes    import ModelParams, Household
from solver     import SolverState, solve_model
from county     import County
from functions  import clamp


# ═══════════════════════════════════════════════════════════════════════════════
# Value function helper (mirrors main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _value(hh: Household) -> float:
    """V = (E/B)^ε/ε − L^{1+1/φ}/(1+1/φ)"""
    eps = float(hh.params.eps_engel)
    phi = float(hh.params.phi)
    return float((hh.E / hh.B) ** eps / eps
                 - hh.L ** (1.0 + 1.0 / phi) / (1.0 + 1.0 / phi))


# ═══════════════════════════════════════════════════════════════════════════════
# Per-county household solve result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CountyResult:
    """
    Stores the household solution for one (county, h-grid) combination.
    """
    county_name: str
    h_grid:  np.ndarray

    # per-h outcomes
    LM_m:  np.ndarray = field(default_factory=lambda: np.array([]))
    LM_f:  np.ndarray = field(default_factory=lambda: np.array([]))
    Lc_m:  np.ndarray = field(default_factory=lambda: np.array([]))
    Lc_f:  np.ndarray = field(default_factory=lambda: np.array([]))
    Ld_m:  np.ndarray = field(default_factory=lambda: np.array([]))
    Ld_f:  np.ndarray = field(default_factory=lambda: np.array([]))
    E:     np.ndarray = field(default_factory=lambda: np.array([]))
    V:     np.ndarray = field(default_factory=lambda: np.array([]))
    converged: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # representative value (mean or median over h-grid)
    V_rep: float = np.nan


# ═══════════════════════════════════════════════════════════════════════════════
# Core routines
# ═══════════════════════════════════════════════════════════════════════════════

def solve_counties(
        counties:        List[County],
        mp:              ModelParams,
        h_grid:          np.ndarray,
        wage_gap:        float = 1.0,
        agg:             str   = "mean",
        solver_kw:       Optional[Dict] = None,
        verbose:         bool  = False,
        plot_convergence: bool = True,
) -> List[CountyResult]:
    """
    Solve the household problem for every county on h_grid.

    Parameters
    ----------
    counties  : list of County objects
    mp        : ModelParams (shared across all counties)
    h_grid    : 1-D array of human capital levels
    wage_gap  : y_f = wage_gap * w_ell * h_f
    agg       : how to aggregate V over h_grid for V_rep
    solver_kw : optional overrides for SolverState fields
                (e.g. dict(max_iter=5000, tol=1e-9))
    verbose   : print per-county summary

    Returns
    -------
    List of CountyResult, one per county (same order as input).
    """
    kw = {"max_iter": 20000, "tol": 1e-10, "damping": 0.2,
          "adapt_damping": True, "verbose": False}
    if solver_kw:
        kw.update(solver_kw)

    N_h     = len(h_grid)
    results      = []
    _county_figs = []   # accumulate per-county figures for final PDF save

    for county in counties:
        # Use county-specific ModelParams if attached (county.mp), else shared mp.
        mp_eff = getattr(county, 'mp', None) or mp

        LM_m = np.full(N_h, np.nan); LM_f = np.full(N_h, np.nan)
        Lc_m = np.full(N_h, np.nan); Lc_f = np.full(N_h, np.nan)
        Ld_m = np.full(N_h, np.nan); Ld_f = np.full(N_h, np.nan)
        E_arr = np.full(N_h, np.nan)
        V_arr = np.full(N_h, np.nan)
        conv  = np.zeros(N_h, dtype=bool)

        state   = SolverState(**kw)
        L_guess = (0.15, 0.10, 0.05, 0.10, 0.04, 0.08)

        for i, h in enumerate(h_grid):
            y_m = county.y_m(h)
            y_f = county.y_f(h, wage_gap=wage_gap)

            hh = Household(
                params = mp_eff,
                y_m    = y_m,
                y_f    = y_f,
                pc     = county.pc,
                pd     = county.pd,
                # a: non-labour income in 1000-KSh units.
                # Set to 0.2 (200 KSh/month) to prevent PIGL shares from going
                # negative at low wage levels. Not calibrated from data —
                # the KCHS transfers/remittances module should be used here.
                a      = 0.2,
                A_c    = county.A_c_home,
                A_d    = county.A_d_home,
            )
            try:
                hh, state = solve_model(mp_eff, hh, state, L0=L_guess)
                LM_m[i] = state.LM_m; LM_f[i] = state.LM_f
                Lc_m[i] = state.Lc_m; Lc_f[i] = state.Lc_f
                Ld_m[i] = state.Ld_m; Ld_f[i] = state.Ld_f
                E_arr[i] = float(hh.E)
                V_arr[i] = _value(hh)
                conv[i]  = state.converged
                L_guess  = tuple(max(1e-14, v) for v in
                                 [state.LM_m, state.LM_f,
                                  state.Lc_m, state.Lc_f,
                                  state.Ld_m, state.Ld_f])
            except Exception:
                pass   # leave as nan

        # representative value
        finite = V_arr[np.isfinite(V_arr)]
        if len(finite) == 0:
            V_rep = np.nan
        elif agg == "median":
            V_rep = float(np.median(finite))
        else:
            V_rep = float(np.mean(finite))

        res = CountyResult(
            county_name = county.name,
            h_grid      = h_grid.copy(),
            LM_m=LM_m, LM_f=LM_f,
            Lc_m=Lc_m, Lc_f=Lc_f,
            Ld_m=Ld_m, Ld_f=Ld_f,
            E=E_arr, V=V_arr,
            converged=conv,
            V_rep=V_rep,
        )
        results.append(res)
        county.V_star = V_rep

        if verbose:
            n_conv = int(np.sum(conv))
            print(f"  {county.name}: V_rep={V_rep:.4g}  "
                  f"converged {n_conv}/{N_h}")

        if plot_convergence:
            try:
                import matplotlib.pyplot as plt

                # ── county label ──────────────────────────────────────────
                KENYA_COUNTIES = {
                    1:"Mombasa",2:"Kwale",3:"Kilifi",4:"Tana River",5:"Lamu",
                    6:"Taita Taveta",7:"Garissa",8:"Wajir",9:"Mandera",
                    10:"Marsabit",11:"Isiolo",12:"Meru",13:"Tharaka-Nithi",
                    14:"Embu",15:"Kitui",16:"Machakos",17:"Makueni",
                    18:"Nyandarua",19:"Nyeri",20:"Kirinyaga",21:"Murang'a",
                    22:"Kiambu",23:"Turkana",24:"West Pokot",25:"Samburu",
                    26:"Trans Nzoia",27:"Uasin Gishu",28:"Elgeyo-Marakwet",
                    29:"Nandi",30:"Baringo",31:"Laikipia",32:"Nakuru",
                    33:"Narok",34:"Kajiado",35:"Kericho",36:"Bomet",
                    37:"Kakamega",38:"Vihiga",39:"Bungoma",40:"Busia",
                    41:"Siaya",42:"Kisumu",43:"Homa Bay",44:"Migori",
                    45:"Kisii",46:"Nyamira",47:"Nairobi",
                }
                try:
                    code = int(county.name.split('_')[1])
                except Exception:
                    code = -1
                cname = KENYA_COUNTIES.get(code, county.name)

                # ── compute county-level diagnostics ──────────────────────
                w_ksh   = county.w_ell * 1000
                pc_ksh  = county.pc    * 1000
                pd_ksh  = county.pd    * 1000
                Ac      = county.A_c_home
                Ad      = county.A_d_home

                LM_m_arr = res.LM_m[np.isfinite(res.LM_m)]
                LM_f_arr = res.LM_f[np.isfinite(res.LM_f)]
                Lc_m_arr = res.Lc_m[np.isfinite(res.Lc_m)]
                Lc_f_arr = res.Lc_f[np.isfinite(res.Lc_f)]
                Ld_m_arr = res.Ld_m[np.isfinite(res.Ld_m)]
                Ld_f_arr = res.Ld_f[np.isfinite(res.Ld_f)]
                E_arr    = res.E[np.isfinite(res.E)]

                LM_m_mean = float(np.mean(LM_m_arr)) if len(LM_m_arr) else np.nan
                LM_f_mean = float(np.mean(LM_f_arr)) if len(LM_f_arr) else np.nan
                Lc_m_mean = float(np.mean(Lc_m_arr)) if len(Lc_m_arr) else np.nan
                Lc_f_mean = float(np.mean(Lc_f_arr)) if len(Lc_f_arr) else np.nan
                Ld_m_mean = float(np.mean(Ld_m_arr)) if len(Ld_m_arr) else np.nan
                Ld_f_mean = float(np.mean(Ld_f_arr)) if len(Ld_f_arr) else np.nan
                E_mean    = float(np.mean(E_arr))     if len(E_arr)    else np.nan

                n_conv    = int(np.sum(conv))
                conv_rate = n_conv / N_h

                # ── reference values (national approximations) ─────────────
                W_NAT  = 92.0;  PC_NAT = 68.7;  PD_NAT = 34.9
                LMM_NAT= 0.52;  LMF_NAT= 0.40
                LDM_NAT= 0.15;  LDF_NAT= 0.28
                LCM_NAT= 0.10;  LCF_NAT= 0.13
                AC_NAT = 1.0;   AD_NAT = 1.0

                def _rel(val, ref, pct=True):
                    if np.isnan(val): return "n/a"
                    diff = (val - ref) / ref * 100
                    sign = "+" if diff >= 0 else ""
                    return f"{sign}{diff:.0f}%"

                # ── build diagnostic text ──────────────────────────────────
                lines = []

                # wages and prices
                w_tag  = "high-wage" if w_ksh  > W_NAT  * 1.2 else \
                         "low-wage"  if w_ksh  < W_NAT  * 0.8 else "average-wage"
                pc_tag = "expensive care"   if pc_ksh > PC_NAT * 1.2 else \
                         "cheap care"       if pc_ksh < PC_NAT * 0.8 else "average care prices"
                pd_tag = "expensive domestic" if pd_ksh > PD_NAT * 1.2 else \
                         "cheap domestic"     if pd_ksh < PD_NAT * 0.8 else "average domestic prices"
                lines.append(f"• Economy: {w_tag} ({w_ksh:.0f} vs {W_NAT:.0f} KSh/hr national), "
                              f"{pc_tag} ({pc_ksh:.0f} KSh/hr), {pd_tag} ({pd_ksh:.0f} KSh/hr).")

                # home productivity
                if Ac > AC_NAT * 1.2:
                    lines.append(f"• High home care productivity (A_c={Ac:.2f}, {_rel(Ac,AC_NAT)} above national): "
                                  f"households substitute home care for market care.")
                elif Ac < AC_NAT * 0.8:
                    lines.append(f"• Low home care productivity (A_c={Ac:.2f}, {_rel(Ac,AC_NAT)} below national): "
                                  f"home care is less effective; market care more attractive.")
                if Ad > AD_NAT * 1.2:
                    lines.append(f"• High home domestic productivity (A_d={Ad:.2f}).")
                elif Ad < AD_NAT * 0.8:
                    lines.append(f"• Low home domestic productivity (A_d={Ad:.2f}).")

                # market work gender gap
                if not np.isnan(LM_m_mean) and not np.isnan(LM_f_mean):
                    gap_M = LM_m_mean - LM_f_mean
                    gap_tag = "large" if gap_M > 0.15 else "small" if gap_M < 0.05 else "moderate"
                    lines.append(f"• Market work: men {LM_m_mean:.2f}h vs women {LM_f_mean:.2f}h/wk "
                                  f"({gap_tag} gender gap of {gap_M:+.2f}h).")

                # domestic gender gap
                if not np.isnan(Ld_m_mean) and not np.isnan(Ld_f_mean):
                    gap_d = Ld_f_mean - Ld_m_mean
                    lines.append(f"• Domestic work: women do {gap_d:.2f}h/wk more than men "
                                  f"({Ld_f_mean:.2f} vs {Ld_m_mean:.2f}).")

                # care work
                if not np.isnan(Lc_m_mean) and not np.isnan(Lc_f_mean):
                    gap_c = Lc_f_mean - Lc_m_mean
                    lines.append(f"• Care work: women do {gap_c:+.2f}h/wk relative to men "
                                  f"({Lc_f_mean:.2f} vs {Lc_m_mean:.2f}).")

                # expenditure
                if not np.isnan(E_mean):
                    lines.append(f"• Mean household expenditure: {E_mean*1000:.0f} KSh/month.")

                # convergence
                if conv_rate == 1.0:
                    lines.append(f"• Solver: fully converged on all {N_h} h-grid points.")
                elif conv_rate >= 0.5:
                    lines.append(f"• Solver: partial convergence ({n_conv}/{N_h} points). "
                                  f"Results approximate.")
                else:
                    lines.append(f"• Solver: poor convergence ({n_conv}/{N_h} points). "
                                  f"Interpret with caution — likely numerical instability at these parameters.")

                # V* and xi
                if np.isfinite(V_rep) and np.isfinite(county.xi):
                    xi_tag = "above-average amenity" if county.xi > 0.05 else \
                             "below-average amenity" if county.xi < -0.05 else "average amenity"
                    lines.append(f"• Equilibrium: V*={V_rep:.3f}, ξ={county.xi:.3f} ({xi_tag}).")

                diag_text = "\n".join(lines)

                # ── hours profile across h-grid ────────────────────────────
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                fig.suptitle(f"{cname}  |  wage={w_ksh:.0f} KSh/hr  |  "
                             f"V*={V_rep:.3f}  |  ξ={county.xi:.3f}",
                             fontsize=12, fontweight='bold')

                h_ticks = [f"h={h:.1f}" for h in h_grid]
                COLORS = {'LM_m':'#1f77b4','LM_f':'#aec7e8',
                          'Lc_m':'#ff7f0e','Lc_f':'#ffbb78',
                          'Ld_m':'#2ca02c','Ld_f':'#98df8a'}

                # Panel 1: market hours
                axes[0].plot(h_grid, res.LM_m, 'o-', color=COLORS['LM_m'], label='Man')
                axes[0].plot(h_grid, res.LM_f, 's--',color=COLORS['LM_f'], label='Woman')
                axes[0].set_title("Market hours $L^M$")
                axes[0].set_xlabel("Human capital h")
                axes[0].set_ylabel("Hours / week")
                axes[0].legend(); axes[0].grid(alpha=0.3)

                # Panel 2: domestic hours
                axes[1].plot(h_grid, res.Ld_m, 'o-', color=COLORS['Ld_m'], label='Man')
                axes[1].plot(h_grid, res.Ld_f, 's--',color=COLORS['Ld_f'], label='Woman')
                axes[1].set_title("Domestic hours $L^d$")
                axes[1].set_xlabel("Human capital h")
                axes[1].legend(); axes[1].grid(alpha=0.3)

                # Panel 3: care hours
                axes[2].plot(h_grid, res.Lc_m, 'o-', color=COLORS['Lc_m'], label='Man')
                axes[2].plot(h_grid, res.Lc_f, 's--',color=COLORS['Lc_f'], label='Woman')
                axes[2].set_title("Care hours $L^c$")
                axes[2].set_xlabel("Human capital h")
                axes[2].legend(); axes[2].grid(alpha=0.3)

                # Diagnostic text below plots
                fig.text(0.01, -0.02, diag_text, fontsize=8,
                         verticalalignment='top', family='monospace',
                         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

                fig.tight_layout(rect=[0, 0.0, 1, 1])
                plt.show()

                # store figure for final save
                _county_figs.append((cname, fig))

            except Exception as e:
                print(f"  (Per-county plot skipped: {e})")

    # ── final save: all county plots in one PDF ───────────────────────────
    if plot_convergence and _county_figs:
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages("county_convergence.pdf") as pdf:
                for cname, fig in _county_figs:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            print(f"  Saved all {len(_county_figs)} county plots → county_convergence.pdf")
        except Exception as e:
            print(f"  (PDF save skipped: {e})")

    return results


def calibrate_amenities(
        counties: List[County],
        results:  List[CountyResult],
) -> float:
    """
    Back out amenities from the spatial equilibrium condition:
        ξ_ℓ = Ū − V*_ℓ
    with population-weighted normalization:
        Ū = Σ_ℓ N_ℓ V*_ℓ / Σ_ℓ N_ℓ   (so Σ_ℓ N_ℓ ξ_ℓ = 0)

    Sets county.xi on each County in-place.
    Returns Ū.
    """
    V_stars = np.array([r.V_rep for r in results])
    Ns      = np.array([c.N    for c in counties])

    # population-weighted mean
    total_N = float(np.sum(Ns))
    U_bar   = float(np.sum(Ns * V_stars) / clamp(total_N))

    for county, V_star in zip(counties, V_stars):
        county.xi      = float(U_bar - V_star)
        county.V_star  = float(V_star)

    return U_bar


def check_no_arbitrage(
        counties: List[County],
        U_bar:    float,
        tol:      float = 1e-8,
) -> bool:
    """
    Verify V*_ℓ + ξ_ℓ = Ū  for all counties.
    Returns True if all pass.
    """
    ok = True
    for county in counties:
        residual = abs(county.V_star + county.xi - U_bar)
        if residual > tol:
            print(f"  WARNING: {county.name}  |V* + ξ − Ū| = {residual:.2e}")
            ok = False
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Counterfactual
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CounterfactualResult:
    """Outcome of a counterfactual experiment."""
    U_bar_new:   float
    county_names: List[str]
    V_star_new:  np.ndarray    # new V*_ℓ for each county
    xi:          np.ndarray    # unchanged amenities
    delta_V:     np.ndarray    # V*_ℓ_new − V*_ℓ_baseline
    # if reequilibrate=True, also store new populations
    N_new:       Optional[np.ndarray] = None
    delta_N:     Optional[np.ndarray] = None


def counterfactual(
        counties_baseline: List[County],
        counties_new:      List[County],
        mp:                ModelParams,
        h_grid:            np.ndarray,
        wage_gap:          float = 1.0,
        reequilibrate:     bool  = False,
        solver_kw:         Optional[Dict] = None,
        verbose:           bool  = False,
        plot_convergence:  bool  = False,
) -> CounterfactualResult:
    """
    Run a counterfactual experiment.

    Parameters
    ----------
    counties_baseline : calibrated baseline counties (with xi set)
    counties_new      : perturbed counties (same list structure, new prices/wages)
                        xi is NOT used from these; we hold xi from baseline.
    mp                : ModelParams (unchanged)
    h_grid            : human capital grid
    wage_gap          : wage gap (unchanged)
    reequilibrate     : if True, compute new populations N'_ℓ consistent with
                        V*'_ℓ + ξ_ℓ = Ū'  (assuming a simple logit migration)
    solver_kw         : solver overrides

    Returns
    -------
    CounterfactualResult
    """
    # copy amenities from baseline onto new counties (structural parameter)
    for c_base, c_new in zip(counties_baseline, counties_new):
        c_new.xi = c_base.xi

    if verbose:
        print("Counterfactual: solving households in perturbed counties...")

    results_new = solve_counties(
        counties_new, mp, h_grid,
        wage_gap=wage_gap, solver_kw=solver_kw, verbose=verbose,
        plot_convergence=plot_convergence,
    )

    V_star_new   = np.array([r.V_rep for r in results_new])
    V_star_base  = np.array([c.V_star for c in counties_baseline])
    xi_arr       = np.array([c.xi    for c in counties_baseline])
    Ns           = np.array([c.N     for c in counties_baseline])

    # new equilibrium utility: population-weighted average of V*' + ξ
    # (in a free-migration equilibrium U'_bar = Σ N'_ℓ (V*'_ℓ + ξ_ℓ) / Σ N'_ℓ
    #  but since population is fixed here, we use the baseline weights)
    U_bar_new = float(np.sum(Ns * (V_star_new + xi_arr)) / clamp(float(np.sum(Ns))))

    N_new = delta_N = None
    if reequilibrate:
        # Logit migration: N'_ℓ ∝ N_ℓ · exp((V*'_ℓ + ξ_ℓ − U'_bar) / sigma_mig)
        # sigma_mig = 1.0 is a placeholder — it is not calibrated from data.
        # It governs the strength of the migration response: lower values
        # mean households are more sensitive to welfare differences.
        # Should be estimated from observed migration flows (e.g. KCHS panel).
        sigma_mig = 1.0
        log_weights = (V_star_new + xi_arr - U_bar_new) / sigma_mig
        log_weights -= log_weights.max()
        weights  = np.exp(log_weights)
        N_bar    = float(np.sum(Ns))
        N_new    = N_bar * weights / float(np.sum(weights))
        delta_N  = N_new - Ns

    return CounterfactualResult(
        U_bar_new    = U_bar_new,
        county_names = [c.name for c in counties_baseline],
        V_star_new   = V_star_new,
        xi           = xi_arr,
        delta_V      = V_star_new - V_star_base,
        N_new        = N_new,
        delta_N      = delta_N,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: build a synthetic county panel
# ═══════════════════════════════════════════════════════════════════════════════

def make_synthetic_counties(
        n_counties: int = 10,
        w_min: float = 10.0,
        w_max: float = 40.0,
        A_c_range: Tuple[float, float] = (0.8, 1.5),
        A_d_range: Tuple[float, float] = (0.8, 1.5),
        N_total: float = 1000.0,
        seed: int = 42,
) -> List[County]:
    """
    Generate a synthetic panel of counties for testing/demonstration.

    Wages vary linearly from w_min to w_max.
    TFPs are drawn from a log-uniform distribution.
    Populations are drawn from a Dirichlet distribution.
    """
    rng = np.random.default_rng(seed)

    wages     = np.linspace(w_min, w_max, n_counties)
    A_c_arr   = np.exp(rng.uniform(np.log(A_c_range[0]),
                                    np.log(A_c_range[1]), n_counties))
    A_d_arr   = np.exp(rng.uniform(np.log(A_d_range[0]),
                                    np.log(A_d_range[1]), n_counties))
    pop_shares = rng.dirichlet(np.ones(n_counties))
    Ns         = N_total * pop_shares

    counties = []
    for k in range(n_counties):
        c = County(
            name      = f"County_{k+1:02d}",
            w_ell     = float(wages[k]),
            A_x       = 1.0,
            A_c       = float(A_c_arr[k]),
            A_d       = float(A_d_arr[k]),
            A_c_home  = float(A_c_arr[k]),   # same as market TFP in baseline
            A_d_home  = float(A_d_arr[k]),
            N         = float(Ns[k]),
        )
        counties.append(c)

    return counties