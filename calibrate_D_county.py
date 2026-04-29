#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_D_county.py
=====================

Calibrate county-specific disutility weights D^g_j for j in {M, c, d}
and g in {m, f}, using the SOLVER-CONSISTENT fixed-point formulas
(NOT the FOC-rearranged formula that originally gave D^f_M < 1).

Identification (one formula per weight):
  Normalisations:
    D^m_M(l) = 1 for all l                 (men's market is the numeraire)

  From observed conditional hours ratios in the TUS:
    D^f_M(l) = r_M(l) / wage_gap^rho        (solver fixed-point match)
    D^m_d(l) = (L^c_m(l) / L^d_m(l))^(1/rho)    (intra-male FOC c vs d)
    D^m_c(l) = 1                            (normalisation)

  Female home weights, intra-household match:
    D^f_c(l) = D^m_c(l) * (L^c_m(l) / L^c_f(l))^(1/rho)
    D^f_d(l) = D^m_d(l) * (L^d_m(l) / L^d_f(l))^(1/rho)

Sample sizes are insufficient for some counties on some activities,
so we apply a Bayesian shrinkage rule:

  r_county_shrunk = (n*r_county + n0*r_national) / (n + n0)

with n = effective per-cell sample size and n0 = 20 (prior weight).

Output: county_D_weights.csv
        Columns: county, D_M_m, D_M_f, D_c_m, D_c_f, D_d_m, D_d_f, n_M, n_c, n_d
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).parent
TUA  = ROOT / 'time_use_app'
TUS_PATH = '/mnt/user-data/uploads/time_use_final_ver13.dta'

PRIOR_N = 20.0   # shrinkage prior weight (effective "national" obs count)


def wmean(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def main():
    print("=" * 70)
    print("County-specific D-weight calibration")
    print("=" * 70)

    # ── Load main calibration to get rho and wage_gap ────────────────
    main = json.load(open(TUA / 'calibrated_params.json'))
    rho       = main['rho']
    wage_gap  = main['wage_gap']
    print(f"\nrho = {rho},  wage_gap = {wage_gap:.4f}")

    # ── Load TUS ─────────────────────────────────────────────────────
    print("\nLoading TUS ...")
    tu, _ = pyreadstat.read_dta(TUS_PATH)
    tu['L_M']    = tu['t11'] * 7 / 60
    tu['L_c']    = tu['t14'] * 7 / 60
    tu['L_d']    = tu['t13'] * 7 / 60
    tu['female'] = (pd.to_numeric(tu['b04'], errors='coerce') == 2).astype(float)
    tu['age_yr'] = pd.to_numeric(tu['b05_years'], errors='coerce')
    tu['county'] = pd.to_numeric(tu['county'], errors='coerce').astype('Int64')
    ms = pd.to_numeric(tu['b07'], errors='coerce')
    tu['married'] = ms.isin([1, 2, 3]).astype(float)   # all union types
    tu['w']       = pd.to_numeric(tu['Person_Day_Weight'], errors='coerce')

    wa = tu[(tu['age_yr'] >= 15) & (tu['age_yr'] <= 65) &
            (tu['married'] == 1)].dropna(subset=['county']).copy()
    wa['county'] = wa['county'].astype(int)
    print(f"  Married working-age: {len(wa)} individuals")
    print(f"  County coverage: {wa['county'].nunique()} counties")

    # ── National-level conditional means (for shrinkage prior) ───────
    print("\nNational conditional means (h/wk):")
    nat = {}
    for sex_lbl, sex_val in [('m', 0), ('f', 1)]:
        for act, col in [('M', 'L_M'), ('c', 'L_c'), ('d', 'L_d')]:
            sub = wa[(wa['female'] == sex_val) & (wa[col] > 0)]
            v = wmean(sub[col], sub['w']) if len(sub) > 0 else np.nan
            nat[f'{col}_{sex_lbl}'] = v
            print(f"  {act}_{sex_lbl}: {v:.2f}h  (n={len(sub)})")

    r_M_nat = nat['L_M_m'] / nat['L_M_f']
    r_c_nat = nat['L_c_m'] / nat['L_c_f']    # for L^c_m / L^c_f intra-household
    r_d_nat = nat['L_d_m'] / nat['L_d_f']
    rcd_m_nat = nat['L_c_m'] / nat['L_d_m']  # intra-male c/d
    print(f"\nNational ratios: r_M = {r_M_nat:.4f}, "
          f"r_c (m/f) = {r_c_nat:.4f}, r_d (m/f) = {r_d_nat:.4f}")
    print(f"Intra-male c/d ratio: {rcd_m_nat:.4f}")

    # ── County-level conditional means and counts ────────────────────
    rows = []
    for ctyc in sorted(wa['county'].unique()):
        sub = wa[wa['county'] == ctyc]
        rec = {'county': int(ctyc)}
        for sex_lbl, sex_val in [('m', 0), ('f', 1)]:
            for act, col in [('M', 'L_M'), ('c', 'L_c'), ('d', 'L_d')]:
                ssub = sub[(sub['female'] == sex_val) & (sub[col] > 0)]
                rec[f'{col}_{sex_lbl}_mean'] = (
                    wmean(ssub[col], ssub['w']) if len(ssub) > 0 else np.nan)
                rec[f'{col}_{sex_lbl}_n'] = int(len(ssub))
        rows.append(rec)
    cty = pd.DataFrame(rows)

    # ── Shrinkage estimator on RATIOS ────────────────────────────────
    # ratio_shrunk = (n_eff * ratio_raw + N_PRIOR * ratio_national) / (n_eff + N_PRIOR)
    def shrink_ratio(r_raw, n_eff, r_nat, n0=PRIOR_N):
        if not np.isfinite(r_raw):
            return r_nat
        return float((n_eff * r_raw + n0 * r_nat) / (n_eff + n0))

    out = {'county': cty['county'].astype(int).tolist()}
    out['D_M_m'] = []   # all 1
    out['D_c_m'] = []   # all 1
    out['D_M_f'] = []
    out['D_d_m'] = []
    out['D_c_f'] = []
    out['D_d_f'] = []
    out['n_M']   = []
    out['n_c']   = []
    out['n_d']   = []

    for _, row in cty.iterrows():
        # Effective sample sizes (geometric mean of m and f counts)
        n_M = float(np.sqrt(row['L_M_m_n'] * row['L_M_f_n']))
        n_c = float(np.sqrt(row['L_c_m_n'] * row['L_c_f_n']))
        n_d = float(np.sqrt(row['L_d_m_n'] * row['L_d_f_n']))

        # Raw ratios
        r_M_raw = (row['L_M_m_mean'] / row['L_M_f_mean']
                   if row['L_M_f_mean'] > 0 else np.nan)
        r_c_raw = (row['L_c_m_mean'] / row['L_c_f_mean']
                   if row['L_c_f_mean'] > 0 else np.nan)
        r_d_raw = (row['L_d_m_mean'] / row['L_d_f_mean']
                   if row['L_d_f_mean'] > 0 else np.nan)
        # Intra-male c/d ratio
        rcd_raw = (row['L_c_m_mean'] / row['L_d_m_mean']
                   if row['L_d_m_mean'] > 0 else np.nan)
        n_cd_m = float(np.sqrt(row['L_c_m_n'] * row['L_d_m_n']))

        # Apply shrinkage
        r_M  = shrink_ratio(r_M_raw,  n_M,    r_M_nat)
        r_c  = shrink_ratio(r_c_raw,  n_c,    r_c_nat)
        r_d  = shrink_ratio(r_d_raw,  n_d,    r_d_nat)
        rcd  = shrink_ratio(rcd_raw,  n_cd_m, rcd_m_nat)

        # ── D weights (solver fixed-point formulas) ──
        # The solver implements (see classes.py compute_labor_residuals):
        #   Market:        L^M_m / L^M_f = wage_gap^rho * D^M_f / D^M_m
        #   Intra-male:    L^c_m / L^d_m = (D^d_m / D^c_m)^rho     (analogous derivation)
        #   Home c split:  L^c_m / L^c_f = (D^c_m / D^c_f)^rho
        #   Home d split:  L^d_m / L^d_f = (D^d_m / D^d_f)^rho
        #
        # Solving each for the unknown D, with the appropriate normalisation:
        D_M_m = 1.0
        D_c_m = 1.0
        # D^M_f from market FOC ratio
        D_M_f = float(r_M / (wage_gap ** rho))
        # D^d_m from intra-male c/d FOC: rcd = L^c_m/L^d_m = (D^d_m/D^c_m)^rho
        # => D^d_m = D^c_m * rcd^(1/rho)
        D_d_m = float(D_c_m * (rcd ** (1.0 / rho)))
        # D^c_f from intra-household c-split: r_c = L^c_m/L^c_f = (D^c_m/D^c_f)^rho
        # => D^c_f = D^c_m * r_c^(-1/rho)
        D_c_f = float(D_c_m * (r_c ** (-1.0 / rho)))
        # D^d_f from intra-household d-split: r_d = L^d_m/L^d_f = (D^d_m/D^d_f)^rho
        # => D^d_f = D^d_m * r_d^(-1/rho)
        D_d_f = float(D_d_m * (r_d ** (-1.0 / rho)))

        out['D_M_m'].append(round(D_M_m, 4))
        out['D_c_m'].append(round(D_c_m, 4))
        out['D_d_m'].append(round(D_d_m, 4))
        out['D_M_f'].append(round(D_M_f, 4))
        out['D_c_f'].append(round(D_c_f, 4))
        out['D_d_f'].append(round(D_d_f, 4))
        out['n_M'].append(int(n_M))
        out['n_c'].append(int(n_c))
        out['n_d'].append(int(n_d))

    df = pd.DataFrame(out)

    # ── Save ─────────────────────────────────────────────────────────
    out_path = TUA / 'county_D_weights.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\nDistribution of county D weights:")
    for col in ['D_M_f', 'D_d_m', 'D_c_f', 'D_d_f']:
        print(f"  {col}: mean={df[col].mean():.3f}  "
              f"range=[{df[col].min():.3f}, {df[col].max():.3f}]")

    # Sample of counties
    print("\nSample (every 5th county):")
    print(df[['county', 'D_M_f', 'D_d_m', 'D_c_f', 'D_d_f',
              'n_M', 'n_c', 'n_d']].iloc[::5].to_string(index=False))


if __name__ == '__main__':
    main()
