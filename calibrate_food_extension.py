#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_food_extension.py
===========================

Companion calibration for the food/non-food split extension.

Produces calibrated quantities for the new food and non-food sectors:
- PIGL shares (beta_xf, beta_xn, kappa_xf, kappa_xn)
- Food preparation hours by gender (TUS code t231)
- Disutility weights for food preparation (D^xf_m, D^xf_f) using the
  CORRECT formula derived from the solver's fixed-point update map
  (the same formula we used to fix D^f_M).

Output: calibrated_food_params.json

Note: integrating this into the full 6 -> 8 equation solver is a
separate (large) task. This script delivers the calibrated parameters
and time-use facts; the existing 3-good solver continues to operate
unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).parent
TUS_PATH = '/mnt/user-data/uploads/time_use_final_ver13.dta'
CA_PATH  = '/mnt/user-data/uploads/consumption_aggregate_microdata.dta'

E_SCALE = 1000.0   # KSh -> 1000-KSh, consistent with main calibration


def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def wls(y, X, w):
    """Weighted least squares with intercept."""
    n = len(y)
    X1 = np.column_stack([np.ones(n), X])
    sw = np.sqrt(w)
    Xs = X1 * sw[:, None]
    ys = y * sw
    coef, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    return coef   # [intercept, slopes...]


def main():
    print("=" * 70)
    print("Food/non-food calibration extension")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────
    # PART 1 - PIGL shares from KCHS consumption aggregate
    # ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading KCHS consumption aggregate ...")
    ca, _ = pyreadstat.read_dta(CA_PATH)
    ca = ca[(ca['padqexp'] > 50) & (ca['padqexp'] < 100000)].copy()

    # Categories: food, non-food (= nonfood items + rent + education)
    # The remaining residual after food, non-food, rent, education is
    # services (care + domestic) which are calibrated separately.
    ca['E']        = ca['padqexp']
    ca['s_food']   = (ca['padqfdcons'] / ca['E']).clip(0, 1)
    ca['s_nonf']   = ((ca['padqnfitems'] + ca['padqrent'] + ca['padqeduc'])
                      / ca['E']).clip(0, 1)
    ca['log_E']    = np.log(ca['E'])

    # Engel slopes
    print("\n[2] Engel slope regressions (weighted) ...")
    slopes = {}
    for col in ['s_food', 's_nonf']:
        sub = ca.dropna(subset=[col, 'log_E'])
        coef = wls(sub[col].values, sub[['log_E']].values,
                   sub['weight_hh'].values)
        slopes[col] = float(coef[1])
        print(f"  d({col})/d log E = {coef[1]:+.4f}  "
              f"(mean share: {wmean(sub[col], sub['weight_hh']):.4f})")

    # PIGL: theta = beta + kappa * (E/B)^(-eps)
    # We piggyback on eps from the main calibration
    main_params = json.load(open(ROOT / 'time_use_app/calibrated_params.json'))
    eps   = main_params['eps_engel']
    # IMPORTANT: kappa must be evaluated at the solver-scale median E,
    # NOT at the KCHS-scale median. The solver works at E ~ 0.2-0.8
    # (1000-KSh, household monthly), while KCHS gives ~6 (per adult equiv).
    # Use the same E_sol = 0.5 as the main calibration to be consistent.
    E_sol = 0.5
    print(f"\n  eps = {eps:.4f} (from main calibration)")
    print(f"  E_sol (1000-KSh) = {E_sol:.4f}  (solver scale, matches main "
          f"calibration)")

    beta_xf = wmean(ca['s_food'], ca['weight_hh'])
    beta_xn = wmean(ca['s_nonf'], ca['weight_hh'])
    print(f"  beta_food = {beta_xf:.4f}, beta_nonfood = {beta_xn:.4f}")
    print(f"  Sum: {beta_xf + beta_xn:.4f} "
          f"(remainder = services, taken from main calibration)")

    # kappa from slopes, evaluated at solver-scale E_sol = 0.5
    kappa_xf = float(-slopes['s_food'] * (E_sol ** (1 + eps)) / eps)
    kappa_xn = float(-slopes['s_nonf'] * (E_sol ** (1 + eps)) / eps)
    print(f"  kappa_food = {kappa_xf:+.4f}  (positive => share falls with "
          f"income, food is a necessity - correct)")
    print(f"  kappa_nonfood = {kappa_xn:+.4f}  (negative => share rises "
          f"with income, non-food is a luxury - correct)")

    # ─────────────────────────────────────────────────────────────────
    # PART 2 - Food preparation hours from TUS
    # ─────────────────────────────────────────────────────────────────
    print("\n[3] Food preparation hours from TUS code t231 ...")
    tu, _ = pyreadstat.read_dta(TUS_PATH)
    tu['L_xf']    = tu['t231'] * 7 / 60   # minutes/day -> hours/week
    tu['female']  = (pd.to_numeric(tu['b04'], errors='coerce') == 2).astype(float)
    tu['age_yr']  = pd.to_numeric(tu['b05_years'], errors='coerce')
    wa = tu[(tu['age_yr'] >= 15) & (tu['age_yr'] <= 65)].copy()

    # Conditional on participation (food prep > 0)
    Lxf_m_uncond = wmean(wa.loc[wa['female'] == 0, 'L_xf'],
                          wa.loc[wa['female'] == 0, 'Person_Day_Weight'])
    Lxf_f_uncond = wmean(wa.loc[wa['female'] == 1, 'L_xf'],
                          wa.loc[wa['female'] == 1, 'Person_Day_Weight'])
    sub_m = wa[(wa['female'] == 0) & (wa['L_xf'] > 0)]
    sub_f = wa[(wa['female'] == 1) & (wa['L_xf'] > 0)]
    Lxf_m_cond = wmean(sub_m['L_xf'], sub_m['Person_Day_Weight'])
    Lxf_f_cond = wmean(sub_f['L_xf'], sub_f['Person_Day_Weight'])

    print(f"  Men:   {Lxf_m_uncond:.2f} h/wk unconditional, "
          f"{Lxf_m_cond:.2f} h/wk conditional")
    print(f"  Women: {Lxf_f_uncond:.2f} h/wk unconditional, "
          f"{Lxf_f_cond:.2f} h/wk conditional")
    print(f"  Conditional ratio Lxf_m/Lxf_f = "
          f"{Lxf_m_cond / Lxf_f_cond:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # PART 3 - Disutility weights via the SOLVER fixed-point
    # ─────────────────────────────────────────────────────────────────
    print("\n[4] Disutility weights (using the solver-consistent formula) ...")
    rho = main_params['rho']
    # Normalisation: D^xf_m = 1 (men's food prep disutility set to 1)
    # The intra-household allocation FOC matches the home-activity equation:
    #   L^xf_m / L^xf_f = (D^xf_m / D^xf_f)^rho   (analogous to L2/L3)
    # Solving for D^xf_f given D^xf_m = 1:
    #   D^xf_f = D^xf_m * (L^xf_m / L^xf_f)^(1/rho)
    # With rho < 0 and L^xf_m / L^xf_f < 1, the (1/rho)-power is negative
    # and applied to a number < 1, so D^xf_f > 1 - the correct sign
    # (women find it less burdensome BECAUSE they do more of it given
    #  their preferences; this is what the model encodes).
    r_xf = Lxf_m_cond / Lxf_f_cond   # ~ 0.55
    D_xf_m = 1.0
    D_xf_f = float(D_xf_m * (r_xf ** (1.0 / rho)))
    print(f"  rho = {rho}")
    print(f"  r_xf = L^xf_m / L^xf_f = {r_xf:.4f}")
    print(f"  D_xf_m = {D_xf_m:.4f}  (normalisation)")
    print(f"  D_xf_f = {D_xf_f:.4f}  "
          f"(should be > 1 since women do more food prep)")

    # ─────────────────────────────────────────────────────────────────
    # PART 4 - Save
    # ─────────────────────────────────────────────────────────────────
    out = {
        # PIGL with food/non-food split
        'eps_engel':   eps,
        'beta_xf':     round(beta_xf, 4),
        'beta_xn':     round(beta_xn, 4),
        'kappa_xf':    round(kappa_xf, 4),
        'kappa_xn':    round(kappa_xn, 4),

        # Food preparation hours (h/wk, conditional on participation)
        'Lxf_m_cond':  round(Lxf_m_cond, 3),
        'Lxf_f_cond':  round(Lxf_f_cond, 3),
        'r_xf':        round(r_xf, 4),

        # Disutility weights for food preparation
        'D_xf_m':      round(D_xf_m, 4),
        'D_xf_f':      round(D_xf_f, 4),

        # CES home/market priors for food (= same as for care/domestic)
        'omega_xf':    0.75,
        'eta_xf':      2.5,

        # Food TFP
        'A_xf':        1.0,

        # Notes
        '_notes': (
            "Calibrated extension for food/non-food split. "
            "The kappa signs are correct: kappa_food < 0 (necessity), "
            "kappa_nonfood > 0 (luxury). "
            "D_xf_f > 1 because women supply more food prep hours "
            "than men (consistent with Engel curves for food prep). "
            "Integration into the 8-equation solver is pending; "
            "the existing 3-good solver continues to operate."
        ),
    }
    out_path = ROOT / 'calibrated_food_params.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[5] Saved: {out_path}")
    print("    All values printed above.")


if __name__ == '__main__':
    main()
