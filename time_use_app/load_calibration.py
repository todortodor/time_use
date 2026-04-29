#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_calibration.py  –  Load calibrated parameters into model objects.

Reads:
  calibrated_params.json     – scalar ModelParams
  county_fundamentals.csv    – county-level w_ell, pc, pd, A_c_proxy, A_d_proxy, N_tus

Returns:
  mp        : ModelParams  (all scalar structural parameters)
  counties  : List[County] (one per county, with county-specific wages/prices/TFP)
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from classes import ModelParams
from county  import County


def load(
    params_path:  str = 'calibrated_params.json',
    county_path:  str = 'county_fundamentals.csv',
    D_county_path: str | None = None,    # if provided, county-specific D weights
    a:            float = 200.0,   # non-labor income (KSh/month, 1000-KSh units → 0.2)
    E_scale:      float = 1000.0,  # expenditure unit (KSh → 1000-KSh)
) -> tuple[ModelParams, list[County]]:
    """
    Load calibrated model parameters and county fundamentals.

    Parameters
    ----------
    params_path   : path to calibrated_params.json
    county_path   : path to county_fundamentals.csv
    D_county_path : optional path to county_D_weights.csv. If given,
                    each county gets its own ModelParams with the
                    county-specific D weights (other parameters shared).
                    The returned `mp` then contains the NATIONAL means
                    (for reporting), and each County has a `.mp` attribute
                    with its own ModelParams.
    a             : non-labor income in KSh/month (will be divided by E_scale)
    E_scale       : expenditure normalisation factor (default 1000 → 1000-KSh units)

    Returns
    -------
    mp       : ModelParams (scalar, national-mean D weights)
    counties : list of County objects, each with .mp = county-specific
               ModelParams attached (or `None` if D_county_path is None
               — in which case all counties share `mp`).
    """
    from dataclasses import replace as dc_replace

    # ── Load scalar parameters ────────────────────────────────────────────
    with open(params_path) as f:
        p = json.load(f)

    mp = ModelParams(
        eps_engel = p['eps_engel'],
        beta_x    = p['beta_x'],
        beta_c    = p['beta_c'],
        beta_d    = p['beta_d'],
        kappa_x   = p['kappa_x'],
        kappa_c   = p['kappa_c'],
        kappa_d   = p['kappa_d'],
        omega_c   = p['omega_c'],
        omega_d   = p['omega_d'],
        eta_c     = p['eta_c'],
        eta_d     = p['eta_d'],
        D_M_m     = p['D_M_m'],
        D_c_m     = p['D_c_m'],
        D_d_m     = p['D_d_m'],
        D_M_f     = p['D_M_f'],
        D_c_f     = p['D_c_f'],
        D_d_f     = p['D_d_f'],
        rho       = p['rho'],
        phi       = p['phi'],
    )

    # ── Optionally load per-county D weights ──────────────────────────────
    D_county_df = None
    if D_county_path is not None:
        D_county_df = pd.read_csv(D_county_path).set_index('county')

    # ── Load county fundamentals ──────────────────────────────────────────
    cty = pd.read_csv(county_path)
    wage_gap = float(p['wage_gap'])

    counties = []
    for _, row in cty.iterrows():
        w       = float(row['w_ell'])        / E_scale
        pc_raw  = row.get('pc', None)
        if pc_raw is None or (isinstance(pc_raw, float) and np.isnan(pc_raw)):
            pc_raw = row.get('w_care', np.nan)
        if isinstance(pc_raw, float) and np.isnan(pc_raw):
            pc_raw = 68.7
        pc      = float(pc_raw) / E_scale

        pd_raw  = row.get('pd', None)
        if pd_raw is None or (isinstance(pd_raw, float) and np.isnan(pd_raw)):
            pd_raw = row.get('w_domestic', np.nan)
        if isinstance(pd_raw, float) and np.isnan(pd_raw):
            pd_raw = 34.9
        pd_val  = float(pd_raw) / E_scale
        Ac = float(row.get('A_c_proxy', 1.0))
        Ad = float(row.get('A_d_proxy', 1.0))
        N  = float(row.get('N_tus', 1.0))
        cname = str(int(row['county'])) if 'county' in row.index else str(_)
        cnum  = int(row['county']) if 'county' in row.index else int(_)

        c = County(
            name       = f"county_{cname}",
            w_ell      = w,
            A_x        = 1.0,
            A_c        = 1.0,
            A_d        = 1.0,
            A_c_home   = Ac,
            A_d_home   = Ad,
            N          = N,
            derive_prices = False,
            _pc        = pc,
            _pd        = pd_val,
        )
        c.wage_gap = wage_gap

        # Attach county-specific ModelParams if D_county_df provided
        if D_county_df is not None and cnum in D_county_df.index:
            d = D_county_df.loc[cnum]
            c.mp = dc_replace(
                mp,
                D_M_m = float(d['D_M_m']),
                D_c_m = float(d['D_c_m']),
                D_d_m = float(d['D_d_m']),
                D_M_f = float(d['D_M_f']),
                D_c_f = float(d['D_c_f']),
                D_d_f = float(d['D_d_f']),
            )
        else:
            c.mp = None     # signals to use the shared `mp`

        counties.append(c)

    return mp, counties


def print_summary(mp: ModelParams, counties: list[County]) -> None:
    print("=== ModelParams ===")
    for field in mp.__dataclass_fields__:
        print(f"  {field:<15} = {getattr(mp, field)}")
    print(f"\n=== Counties ({len(counties)}) ===")
    wages = [c.w_ell for c in counties]
    pcs   = [c.pc    for c in counties]
    pds   = [c.pd    for c in counties]
    print(f"  w_ell : mean={np.mean(wages):.4f}  min={np.min(wages):.4f}  max={np.max(wages):.4f}  (1000-KSh/hr)")
    print(f"  pc    : mean={np.mean(pcs):.4f}  min={np.min(pcs):.4f}  max={np.max(pcs):.4f}")
    print(f"  pd    : mean={np.mean(pds):.4f}  min={np.min(pds):.4f}  max={np.max(pds):.4f}")
    print(f"  A_c_home range: [{min(c.A_c_home for c in counties):.3f}, {max(c.A_c_home for c in counties):.3f}]")
    print(f"  A_d_home range: [{min(c.A_d_home for c in counties):.3f}, {max(c.A_d_home for c in counties):.3f}]")


if __name__ == '__main__':
    mp, counties = load()
    print_summary(mp, counties)
