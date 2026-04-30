"""
main.py — Bokeh app for the Kenya 4-good time-use spatial equilibrium model.

Two side-by-side scenario columns (A and B) for comparison.  Each column has:

  • A clickable Kenya map (centroid circles) at the top for county selection.
  • Global parameters section (PIGL 4 goods, CES, rho, phi, wage_gap, p_xf,
    sigma_u, u_bar, sigma_mig).
  • County-specific parameters section (auto-loaded for the selected county):
    w_ell, p_xf, pc, pd, AM_xn, AM_xf, AM_c, AM_d, N, plus 5 free D weights.
    Constants D_M_m = D_xf_m = D_c_m = 1 are hidden.
  • Solve / Reset / Solve-all buttons.
  • Counterfactual panel with three buttons (wage_gap=1, p^c×0.7, p^d×0.7).
  • Tabs:
      - Per-h plots     (selected county): hours (8 lines), gaps (4),
                        PIGL shares (4 goods), home/market shares (6),
                        participation rates P_m / P_f
      - Spatial summary (requires "Solve all 47"): V* vs wage scatter,
                        ξ ranking bar, gender-ratio scatter, sectoral shares
      - Counterfactuals (requires "Solve all 47"): summary table +
                        12 choropleth maps (3 scenarios × 4 indicators)

Run with:
    bokeh serve --show /home/claude/project/main.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import (
    Button, ColumnDataSource, Div, HoverTool, NumericInput,
    Select, Tabs, TabPanel, DataTable, TableColumn, NumberFormatter,
    LinearColorMapper, ColorBar, BasicTicker,
)
from bokeh.plotting import figure
from bokeh.palettes import RdBu11

# Local imports
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from classes import ModelParams, County
from solver_functions import (
    solve_county_household, solve_all_counties, calibrate_amenities,
    compute_market_clearing, run_counterfactual,
)


# =========================================================================== #
# Paths and constants                                                         #
# =========================================================================== #

PARAMS_PATH = HERE / "calibrated_params.json"
COUNTY_PATH = HERE / "county_data.csv"
GEOMETRY_PATH = HERE / "kenya_counties.json"

E_SCALE = 1000.0          # KSh -> 1000-KSh
H_GRID  = np.array([0.5, 1.0, 2.0, 3.0])

# Plot dims
H_FIG = 230
W_FIG = 380


# =========================================================================== #
# County geometry                                                             #
# =========================================================================== #
# kenya_counties.json (built from GADM 4.1 level-1, simplified to 0.01°)
# stores per-county lists of polygon patches (one or more, for islands etc.).
# Bokeh's `patches` glyph wants one row per patch; we keep the county_id as
# a column on every row so a single tap-select still gives us the county.

def _load_county_geometry():
    if not GEOMETRY_PATH.exists():
        return None
    with GEOMETRY_PATH.open() as f:
        raw = json.load(f)
    # Build flat per-patch lists keyed back to county_id.
    patches_xs   = []
    patches_ys   = []
    patches_id   = []
    patches_name = []
    for cid_str in sorted(raw.keys(), key=int):
        cid = int(cid_str)
        entry = raw[cid_str]
        for patch in entry["patches"]:
            patches_xs.append(patch["xs"])
            patches_ys.append(patch["ys"])
            patches_id.append(cid)
            patches_name.append(entry["name"])
    return dict(xs=patches_xs, ys=patches_ys,
                id=patches_id, name=patches_name)


COUNTY_GEOMETRY = _load_county_geometry()


# =========================================================================== #
# Data loading                                                                #
# =========================================================================== #

def _load_baseline():
    with PARAMS_PATH.open() as f:
        params = json.load(f)
    cdf = pd.read_csv(COUNTY_PATH)
    cdf["county_id"] = cdf["county_id"].astype(int)

    # Build the baseline dictionary (used for reset)
    counties: Dict[int, dict] = {}
    for _, r in cdf.iterrows():
        cid = int(r["county_id"])
        counties[cid] = {
            "name": r["name"], "lat": float(r["lat"]), "lon": float(r["lon"]),
            "w_ell": float(r["w_ell"]),
            "p_xf":  float(r["p_xf"]),
            "pc":    float(r["pc"]),
            "pd":    float(r["pd"]),
            "AM_xn": float(r["AM_xn"]),
            "AM_xf": float(r["AM_xf"]),
            "AM_c":  float(r["AM_c"]),
            "AM_d":  float(r["AM_d"]),
            "A_xf_home": float(r["A_xf_home"]),
            "A_c_home":  float(r["A_c_home"]),
            "A_d_home":  float(r["A_d_home"]),
            "N":     float(r["N"]),
            "D_M_m": float(r["D_M_m"]), "D_xf_m": float(r["D_xf_m"]),
            "D_c_m": float(r["D_c_m"]), "D_d_m":  float(r["D_d_m"]),
            "D_M_f": float(r["D_M_f"]), "D_xf_f": float(r["D_xf_f"]),
            "D_c_f": float(r["D_c_f"]), "D_d_f":  float(r["D_d_f"]),
            # Calibrated quantities (optional: blank -> NaN)
            "V_star": float(r["V_star"]) if r["V_star"] != "" else float("nan"),
            "P_m":    float(r["P_m"])    if r["P_m"]    != "" else float("nan"),
            "P_f":    float(r["P_f"])    if r["P_f"]    != "" else float("nan"),
            "xi":     float(r["xi"])     if r["xi"]     != "" else float("nan"),
        }
    return params, counties


BASELINE_PARAMS, BASELINE_COUNTIES = _load_baseline()
COUNTY_NAMES = {cid: BASELINE_COUNTIES[cid]["name"]
                for cid in BASELINE_COUNTIES}


# =========================================================================== #
# UI field definitions                                                        #
# =========================================================================== #

GLOBAL_FIELDS = [
    ("eps_engel", "ε (Engel)"),
    ("rho",       "ρ (CES disutility)"),
    ("phi",       "φ (Frisch elast.)"),
    ("wage_gap",  "wage_gap"),
    ("p_xf",      "p_xf (national)"),
    ("beta_xf",   "β_xf"),  ("beta_xn",  "β_xn"),
    ("beta_c",    "β_c"),   ("beta_d",   "β_d"),
    ("kappa_xf",  "κ_xf"),  ("kappa_xn", "κ_xn"),
    ("kappa_c",   "κ_c"),   ("kappa_d",  "κ_d"),
    ("omega_xf",  "ω_xf"),  ("omega_c",  "ω_c"),  ("omega_d",  "ω_d"),
    ("eta_xf",    "η_xf"),  ("eta_c",    "η_c"),  ("eta_d",    "η_d"),
    ("sigma_u_m", "σ_u^m"), ("sigma_u_f","σ_u^f"),
    ("u_bar_m",   "ū^m"),   ("u_bar_f",  "ū^f"),
    ("sigma_mig", "σ_mig"),
]

# Constant D weights (D_M_m = D_xf_m = D_c_m = 1) are hidden in the UI.
# D_d_m varies by county (intra-male c-vs-d ratio), so it shows.
COUNTY_FIELDS = [
    ("w_ell",  "wage (1000-KSh/hr)"),
    ("p_xf",   "p_xf (county; usually = global)"),
    ("pc",     "p_c (1000-KSh/hr)"),
    ("pd",     "p_d (1000-KSh/hr)"),
    ("AM_xn",  "A^M,xn (= w)"),
    ("AM_xf",  "A^M,xf (= w/p_xf)"),
    ("AM_c",   "A^M,c (= w/p_c)"),
    ("AM_d",   "A^M,d (= w/p_d)"),
    ("N",      "N (population)"),
    ("D_M_f",  "D_M^f"),
    ("D_xf_f", "D_xf^f"),
    ("D_c_f",  "D_c^f"),
    ("D_d_m",  "D_d^m"),
    ("D_d_f",  "D_d^f"),
]


def _build_mp(global_p: dict) -> ModelParams:
    """Build a ModelParams from the (possibly user-edited) global_p dict.

    The 8 D weights in ModelParams are placeholder national defaults; the
    actual per-county D weights live in the County dict and override these
    for each county solve.
    """
    return ModelParams(
        eps_engel=float(global_p["eps_engel"]),
        beta_xf=float(global_p["beta_xf"]),
        beta_xn=float(global_p["beta_xn"]),
        beta_c=float(global_p["beta_c"]),
        beta_d=float(global_p["beta_d"]),
        kappa_xf=float(global_p["kappa_xf"]),
        kappa_xn=float(global_p["kappa_xn"]),
        kappa_c=float(global_p["kappa_c"]),
        kappa_d=float(global_p["kappa_d"]),
        omega_xf=float(global_p["omega_xf"]),
        omega_c=float(global_p["omega_c"]),
        omega_d=float(global_p["omega_d"]),
        eta_xf=float(global_p["eta_xf"]),
        eta_c=float(global_p["eta_c"]),
        eta_d=float(global_p["eta_d"]),
        D_M_m=1.0, D_xf_m=1.0, D_c_m=1.0,  # placeholder (county overrides)
        D_d_m=float(BASELINE_PARAMS["D_d_m"]),
        D_M_f=float(BASELINE_PARAMS["D_M_f"]),
        D_xf_f=float(BASELINE_PARAMS["D_xf_f"]),
        D_c_f=float(BASELINE_PARAMS["D_c_f"]),
        D_d_f=float(BASELINE_PARAMS["D_d_f"]),
        rho=float(global_p["rho"]),
        phi=float(global_p["phi"]),
        sigma_u_m=float(global_p["sigma_u_m"]),
        sigma_u_f=float(global_p["sigma_u_f"]),
        u_bar_m=float(global_p["u_bar_m"]),
        u_bar_f=float(global_p["u_bar_f"]),
        wage_gap=float(global_p["wage_gap"]),
        p_xf=float(global_p["p_xf"]),
        sigma_mig=float(global_p["sigma_mig"]),
    )


def _build_county(cp: dict, cid: int, mp: ModelParams) -> County:
    """Build a County from the (possibly user-edited) per-county dict.

    The county's D weights override the placeholder ones in mp via the
    Household solver code path, which uses mp.D_*_g.  Therefore for per-
    county solves we construct a *county-specific* mp with the county's D
    weights inserted.  This mirrors the existing 3-good app pattern.
    """
    return County(
        name=cp["name"], county_id=cid,
        lat=float(cp["lat"]), lon=float(cp["lon"]),
        w_ell=float(cp["w_ell"]),
        p_xf=float(cp["p_xf"]),
        pc=float(cp["pc"]),  pd=float(cp["pd"]),
        AM_xn=float(cp["AM_xn"]), AM_xf=float(cp["AM_xf"]),
        AM_c=float(cp["AM_c"]),   AM_d=float(cp["AM_d"]),
        A_xf_home=float(cp["A_xf_home"]),
        A_c_home=float(cp["A_c_home"]),
        A_d_home=float(cp["A_d_home"]),
        N=float(cp["N"]),
        D_M_m=float(cp["D_M_m"]),   D_xf_m=float(cp["D_xf_m"]),
        D_c_m=float(cp["D_c_m"]),   D_d_m=float(cp["D_d_m"]),
        D_M_f=float(cp["D_M_f"]),   D_xf_f=float(cp["D_xf_f"]),
        D_c_f=float(cp["D_c_f"]),   D_d_f=float(cp["D_d_f"]),
    )


def _county_specific_mp(global_p: dict, cp: dict) -> ModelParams:
    """ModelParams with the county's own D weights baked in."""
    base = _build_mp(global_p)
    return ModelParams(
        eps_engel=base.eps_engel,
        beta_xf=base.beta_xf, beta_xn=base.beta_xn,
        beta_c=base.beta_c,   beta_d=base.beta_d,
        kappa_xf=base.kappa_xf, kappa_xn=base.kappa_xn,
        kappa_c=base.kappa_c,   kappa_d=base.kappa_d,
        omega_xf=base.omega_xf, omega_c=base.omega_c, omega_d=base.omega_d,
        eta_xf=base.eta_xf,     eta_c=base.eta_c,     eta_d=base.eta_d,
        D_M_m=float(cp["D_M_m"]),
        D_xf_m=float(cp["D_xf_m"]),
        D_c_m=float(cp["D_c_m"]),
        D_d_m=float(cp["D_d_m"]),
        D_M_f=float(cp["D_M_f"]),
        D_xf_f=float(cp["D_xf_f"]),
        D_c_f=float(cp["D_c_f"]),
        D_d_f=float(cp["D_d_f"]),
        rho=base.rho, phi=base.phi,
        sigma_u_m=base.sigma_u_m, sigma_u_f=base.sigma_u_f,
        u_bar_m=base.u_bar_m,     u_bar_f=base.u_bar_f,
        wage_gap=base.wage_gap, p_xf=base.p_xf,
        sigma_mig=base.sigma_mig,
    )


# =========================================================================== #
# Solve wrappers (return arrays sized to H_GRID)                              #
# =========================================================================== #

def solve_one_county(global_p: dict, cp: dict) -> dict:
    """Solve a single county over H_GRID at the (possibly edited) parameters.

    Returns a dict of arrays length len(H_GRID) for plotting, plus
    convergence diagnostics.
    """
    mp = _county_specific_mp(global_p, cp)
    cid = int(cp.get("county_id", 0))
    cty = _build_county(cp, cid, mp)

    Nh = len(H_GRID)
    arrs = {k: np.full(Nh, np.nan) for k in (
        "LM_m","LM_f","Lxf_m","Lxf_f","Lc_m","Lc_f","Ld_m","Ld_f",
        "E","V11","V10","V01","V00","EV","P_m","P_f",
        "th_xf","th_xn","th_c","th_d",
        "SxfH_share","SxfM_share","ScH_share","ScM_share",
        "SdH_share","SdM_share",
        "Pxf","Pc","Pd",
        "SxfM","ScM","SdM","xn",
    )}
    arrs["converged"] = np.zeros(Nh, dtype=bool)

    for i, h in enumerate(H_GRID):
        try:
            res = solve_county_household(cty, mp, float(h), float(h), a=0.2)
        except Exception:
            continue
        if not res["conv11"]:
            continue
        hh = res["hh11"]
        arrs["LM_m"][i]=hh.LM_m; arrs["LM_f"][i]=hh.LM_f
        arrs["Lxf_m"][i]=hh.Lxf_m; arrs["Lxf_f"][i]=hh.Lxf_f
        arrs["Lc_m"][i]=hh.Lc_m; arrs["Lc_f"][i]=hh.Lc_f
        arrs["Ld_m"][i]=hh.Ld_m; arrs["Ld_f"][i]=hh.Ld_f
        arrs["E"][i]=hh.E
        arrs["th_xf"][i]=hh.th_xf; arrs["th_xn"][i]=hh.th_xn
        arrs["th_c"][i]=hh.th_c;   arrs["th_d"][i]=hh.th_d
        arrs["Pxf"][i]=hh.Pxf; arrs["Pc"][i]=hh.Pc; arrs["Pd"][i]=hh.Pd
        arrs["SxfM"][i]=hh.SxfM; arrs["ScM"][i]=hh.ScM
        arrs["SdM"][i]=hh.SdM; arrs["xn"][i]=hh.xn
        # Expenditure shares (sum to 1 by CES envelope)
        if hh.Pxf > 0 and hh.Sxf > 0:
            arrs["SxfH_share"][i] = (hh.PxfH * hh.SxfH) / (hh.Pxf * hh.Sxf)
            arrs["SxfM_share"][i] = (hh.p_xf * hh.SxfM) / (hh.Pxf * hh.Sxf)
        if hh.Pc > 0 and hh.Sc > 0:
            arrs["ScH_share"][i] = (hh.PcH * hh.ScH) / (hh.Pc * hh.Sc)
            arrs["ScM_share"][i] = (hh.pc  * hh.ScM) / (hh.Pc * hh.Sc)
        if hh.Pd > 0 and hh.Sd > 0:
            arrs["SdH_share"][i] = (hh.PdH * hh.SdH) / (hh.Pd * hh.Sd)
            arrs["SdM_share"][i] = (hh.pd  * hh.SdM) / (hh.Pd * hh.Sd)
        arrs["V11"][i]=res["hh11"].V_state; arrs["V10"][i]=res["hh10"].V_state
        arrs["V01"][i]=res["hh01"].V_state; arrs["V00"][i]=res["hh00"].V_state
        arrs["P_m"][i]=res["P_m"]; arrs["P_f"][i]=res["P_f"]; arrs["EV"][i]=res["EV"]
        arrs["converged"][i] = True

    arrs["n_conv"]   = int(arrs["converged"].sum())
    arrs["wage_KSh"] = float(cp["w_ell"] * E_SCALE)
    arrs["N"]        = float(cp["N"])
    return arrs


def solve_all(global_p: dict, counties_p: Dict[int, dict]
              ) -> Dict[int, dict]:
    """Solve every county; returns {cid: arrays}."""
    return {cid: solve_one_county(global_p, counties_p[cid])
            for cid in sorted(counties_p)}


def summarise_spatial(spatial: Dict[int, dict],
                      counties_p: Dict[int, dict]) -> dict:
    """National summary metrics from a fully-solved spatial result."""
    ids = sorted(spatial.keys())
    Vstar = np.array([float(np.nanmean(spatial[c]["EV"])) for c in ids])
    GDP   = np.array([float(np.nanmean(spatial[c]["E"]))
                      * float(spatial[c]["N"]) for c in ids])
    Pf    = np.array([float(np.nanmean(spatial[c]["P_f"])) for c in ids])
    Pm    = np.array([float(np.nanmean(spatial[c]["P_m"])) for c in ids])
    LMm   = np.array([float(np.nanmean(spatial[c]["LM_m"])) for c in ids])
    LMf   = np.array([float(np.nanmean(spatial[c]["LM_f"])) for c in ids])
    ratio = LMf / np.maximum(LMm, 1e-9)
    wages = np.array([spatial[c]["wage_KSh"] for c in ids])
    Ns    = np.array([float(counties_p[c]["N"]) for c in ids])

    # Sectoral employment from compute_market_clearing-style accounting
    # We do the same accounting inline using the grid means, mirroring
    # solver_functions.compute_market_clearing.
    LM_xf = np.array([float(np.nanmean(spatial[c]["SxfM"]))
                      * float(counties_p[c]["N"])
                      / max(float(counties_p[c]["AM_xf"]), 1e-9)
                      for c in ids])
    LM_c  = np.array([float(np.nanmean(spatial[c]["ScM"]))
                      * float(counties_p[c]["N"])
                      / max(float(counties_p[c]["AM_c"]), 1e-9)
                      for c in ids])
    LM_d  = np.array([float(np.nanmean(spatial[c]["SdM"]))
                      * float(counties_p[c]["N"])
                      / max(float(counties_p[c]["AM_d"]), 1e-9)
                      for c in ids])
    LM_total_supplied = np.array([float(np.nanmean(spatial[c]["LM_m"])
                                          + np.nanmean(spatial[c]["LM_f"]))
                                  * float(counties_p[c]["N"])
                                  for c in ids])
    LM_xn = LM_total_supplied - (LM_xf + LM_c + LM_d)

    # Population-weighted Ubar; xi_l = Ubar - Vstar_l
    finite = np.isfinite(Vstar) & (Ns > 0)
    Ubar = (float(np.average(Vstar[finite], weights=Ns[finite]))
            if finite.sum() > 0 else float("nan"))
    xi = Ubar - Vstar

    return dict(counties=ids, Vstar=Vstar, GDP=GDP,
                Pm=Pm, Pf=Pf, LM_m=LMm, LM_f=LMf, ratio=ratio,
                wages=wages, xi=xi, Ubar=Ubar,
                LM_xf=LM_xf, LM_c=LM_c, LM_d=LM_d, LM_xn=LM_xn,
                N=Ns)


# =========================================================================== #
# UI factory                                                                  #
# =========================================================================== #

def _empty_per_h() -> dict:
    return {k: [float("nan")] * len(H_GRID) for k in (
        "LM_m","LM_f","Lxf_m","Lxf_f","Lc_m","Lc_f","Ld_m","Ld_f",
        "th_xf","th_xn","th_c","th_d",
        "SxfH","SxfM","ScH","ScM","SdH","SdM",
        "P_m","P_f",
        "gap_M","gap_xf","gap_c","gap_d")} | {"h": list(H_GRID)}


def _empty_spatial() -> dict:
    ids = sorted(BASELINE_COUNTIES.keys())
    return dict(
        county=ids,
        name=[BASELINE_COUNTIES[c]["name"] for c in ids],
        Vstar=[float("nan")] * len(ids),
        xi   =[float("nan")] * len(ids),
        wage =[float("nan")] * len(ids),
        Pm   =[float("nan")] * len(ids),
        Pf   =[float("nan")] * len(ids),
        ratio=[float("nan")] * len(ids),
        LM_m =[float("nan")] * len(ids),
        LM_f =[float("nan")] * len(ids),
        GDP  =[float("nan")] * len(ids),
        N    =[float("nan")] * len(ids),
    )


def _empty_map_data() -> dict:
    """Initial column data for the map source.

    If COUNTY_GEOMETRY is loaded, the source is patches-shaped: one row per
    polygon patch, with `xs`, `ys`, `id`, `name`, `value`.  If the geometry
    file is missing, fall back to centroids (one row per county).
    """
    if COUNTY_GEOMETRY is not None:
        n = len(COUNTY_GEOMETRY["id"])
        return dict(
            xs   = list(COUNTY_GEOMETRY["xs"]),
            ys   = list(COUNTY_GEOMETRY["ys"]),
            id   = list(COUNTY_GEOMETRY["id"]),
            name = list(COUNTY_GEOMETRY["name"]),
            value= [float("nan")] * n,
        )
    # Centroid fallback (matches the pre-geometry behaviour)
    ids = sorted(BASELINE_COUNTIES.keys())
    return dict(
        id=ids,
        name=[BASELINE_COUNTIES[c]["name"] for c in ids],
        lat=[BASELINE_COUNTIES[c]["lat"] for c in ids],
        lon=[BASELINE_COUNTIES[c]["lon"] for c in ids],
        value=[float(BASELINE_COUNTIES[c]["w_ell"]) * E_SCALE for c in ids],
    )


# =========================================================================== #
# ScenarioUI — one of two side-by-side scenarios                              #
# =========================================================================== #

class ScenarioUI:
    """One scenario column. Two of these go side by side."""

    def __init__(self, label: str, default_county: int = 47):
        self.label = label
        self.selected_county = default_county

        # State (mutable; user edits flow into these)
        self.global_p   = dict(BASELINE_PARAMS)
        self.counties_p = {cid: dict(BASELINE_COUNTIES[cid])
                           for cid in BASELINE_COUNTIES}
        for cid in self.counties_p:
            self.counties_p[cid]["county_id"] = cid

        # Cached results
        self.spatial: Optional[Dict[int, dict]] = None
        self.spatial_summary: Optional[dict] = None
        self.cf_results: Dict[str, Dict[int, dict]] = {}
        self.cf_summaries: Dict[str, dict] = {}

        # Status div
        self.status = Div(
            text=f"<i>{label}: ready.</i>",
            width=W_FIG * 2 + 30, height=30,
            styles={"font-family":"sans-serif","font-size":"11pt"})
        self.county_label = Div(
            text=f"<b>Selected county:</b> "
                 f"{COUNTY_NAMES[default_county]} (#{default_county})",
            width=W_FIG * 2 + 30,
            styles={"font-family":"sans-serif","font-size":"11pt"})

        # Data sources
        self.src_hours    = ColumnDataSource(_empty_per_h())
        self.src_gap      = ColumnDataSource(_empty_per_h())
        self.src_shares   = ColumnDataSource(_empty_per_h())
        self.src_homemkt  = ColumnDataSource(_empty_per_h())
        self.src_part     = ColumnDataSource(_empty_per_h())
        self.src_spatial  = ColumnDataSource(_empty_spatial())
        self.src_xi       = ColumnDataSource(dict(rank=[], name=[],
                                                   xi=[], color=[]))
        self.src_emp      = ColumnDataSource(dict(name=[], LM_xf=[],
                                                   LM_xn=[], LM_c=[], LM_d=[]))
        self.map_source   = ColumnDataSource(_empty_map_data())
        self.cf_table_src = ColumnDataSource(dict(
            scenario=[], dGDP=[], dPm=[], dPf=[], dratio=[], dN=[]))

        # CF map sources: (scenario × indicator)
        self.cf_maps = {(sc, ind): ColumnDataSource(_empty_map_data())
                        for sc in ("wage", "pc", "pd")
                        for ind in ("dGDP", "dPf", "dratio", "dN")}

        # Build widgets
        self._build_widgets()
        self._build_plots()
        self._build_layout()
        self._wire_callbacks()

        # Initial solve to populate the per-h tab
        self._solve_selected_and_update()

    # ------------------------------------------------------------------ #
    # Widgets                                                            #
    # ------------------------------------------------------------------ #

    def _build_widgets(self):
        # Buttons
        self.solve_btn     = Button(label="Solve selected county",
                                     button_type="primary", width=200)
        self.reset_btn     = Button(label="Reset to baseline", width=140)
        self.solve_all_btn = Button(label="Solve all 47 counties",
                                     button_type="success", width=200)
        self.cf_btn_wage = Button(label="CF: wage_gap = 1",
                                   button_type="warning", width=180)
        self.cf_btn_pc   = Button(label="CF: p_c × 0.7",
                                   button_type="warning", width=180)
        self.cf_btn_pd   = Button(label="CF: p_d × 0.7",
                                   button_type="warning", width=180)
        self.map_metric  = Select(title="Map metric:",
                                   value="wage",
                                   options=["wage", "Pf (baseline)",
                                            "ΔGDP (last CF)"],
                                   width=200)
        # Global parameter inputs
        self.global_inputs = {}
        for k, lab in GLOBAL_FIELDS:
            self.global_inputs[k] = NumericInput(
                title=lab, value=float(self.global_p[k]),
                mode="float", width=130)
        # County parameter inputs
        self.county_inputs = {}
        for k, lab in COUNTY_FIELDS:
            self.county_inputs[k] = NumericInput(
                title=lab,
                value=float(self.counties_p[self.selected_county][k]),
                mode="float", width=130)

    # ------------------------------------------------------------------ #
    # Plots                                                              #
    # ------------------------------------------------------------------ #

    def _build_plots(self):
        # ----- Per-h tab plots ---------------------------------------- #
        # 1. Hours by gender × activity (8 lines)
        p_hours = figure(height=H_FIG, width=W_FIG,
                          title="Hours per week by gender × activity")
        for col_, color, label in [
            ("LM_m",  "#1f77b4", "L^M man"),
            ("LM_f",  "#1f77b4", "L^M woman"),
            ("Lxf_m", "#ff7f0e", "L^xf man"),
            ("Lxf_f", "#ff7f0e", "L^xf woman"),
            ("Lc_m",  "#d62728", "L^c man"),
            ("Lc_f",  "#d62728", "L^c woman"),
            ("Ld_m",  "#2ca02c", "L^d man"),
            ("Ld_f",  "#2ca02c", "L^d woman"),
        ]:
            dash = "dashed" if "_f" in col_ else "solid"
            p_hours.line("h", col_, source=self.src_hours,
                         color=color, line_dash=dash, line_width=2,
                         legend_label=label)
        p_hours.xaxis.axis_label = "h"
        p_hours.yaxis.axis_label = "hours / week"
        p_hours.legend.location = "top_left"
        p_hours.legend.label_text_font_size = "7pt"

        # 2. Gender gaps (woman - man) for 4 activities
        p_gap = figure(height=H_FIG, width=W_FIG,
                        title="Gender gap (woman − man), 4 activities")
        for col_, color, label in [
            ("gap_M",  "#1f77b4", "ΔL^M"),
            ("gap_xf", "#ff7f0e", "ΔL^xf"),
            ("gap_c",  "#d62728", "ΔL^c"),
            ("gap_d",  "#2ca02c", "ΔL^d"),
        ]:
            p_gap.line("h", col_, source=self.src_gap,
                       color=color, line_width=2, legend_label=label)
        p_gap.xaxis.axis_label = "h"
        p_gap.yaxis.axis_label = "Δ hours / week"
        p_gap.legend.location = "center_right"
        p_gap.legend.label_text_font_size = "7pt"

        # 3. PIGL expenditure shares (4 goods)
        p_sh = figure(height=H_FIG, width=W_FIG,
                       title="PIGL expenditure shares (4 goods)")
        for col_, color, label in [
            ("th_xf", "#ff7f0e", "θ_xf (food)"),
            ("th_xn", "#9467bd", "θ_xn (non-food)"),
            ("th_c",  "#d62728", "θ_c (care)"),
            ("th_d",  "#2ca02c", "θ_d (domestic)"),
        ]:
            p_sh.line("h", col_, source=self.src_shares,
                      color=color, line_width=2, legend_label=label)
        p_sh.xaxis.axis_label = "h"
        p_sh.yaxis.axis_label = "share"
        p_sh.legend.location = "center_right"
        p_sh.legend.label_text_font_size = "7pt"

        # 4. Home/market expenditure shares (6 lines)
        p_hm = figure(height=H_FIG, width=W_FIG,
                       title="Home vs market expenditure shares")
        for col_, color, dash, label in [
            ("SxfH", "#ff7f0e", "solid",  "Food home"),
            ("SxfM", "#ff7f0e", "dashed", "Food market"),
            ("ScH",  "#d62728", "solid",  "Care home"),
            ("ScM",  "#d62728", "dashed", "Care market"),
            ("SdH",  "#2ca02c", "solid",  "Domestic home"),
            ("SdM",  "#2ca02c", "dashed", "Domestic market"),
        ]:
            p_hm.line("h", col_, source=self.src_homemkt,
                      color=color, line_dash=dash, line_width=2,
                      legend_label=label)
        p_hm.xaxis.axis_label = "h"
        p_hm.yaxis.axis_label = "share of total"
        p_hm.legend.location = "center_right"
        p_hm.legend.label_text_font_size = "7pt"

        # 5. Participation rates
        p_part = figure(height=H_FIG, width=W_FIG,
                         title="Participation rates P_m, P_f")
        p_part.line("h", "P_m", source=self.src_part,
                    color="#1f77b4", line_width=2, legend_label="P_m")
        p_part.line("h", "P_f", source=self.src_part,
                    color="#d62728", line_width=2, legend_label="P_f")
        p_part.xaxis.axis_label = "h"
        p_part.yaxis.axis_label = "participation"
        p_part.legend.location = "center_right"
        p_part.legend.label_text_font_size = "8pt"

        self.p_hours, self.p_gap, self.p_sh, self.p_hm, self.p_part = (
            p_hours, p_gap, p_sh, p_hm, p_part)

        # ----- Spatial summary plots ---------------------------------- #
        p_vs = figure(height=H_FIG, width=W_FIG, title="V* vs wage")
        p_vs.scatter("wage", "Vstar", source=self.src_spatial,
                     size=8, fill_color="#1f77b4",
                     line_color="black", line_width=0.5)
        p_vs.xaxis.axis_label = "wage (KSh/hr)"
        p_vs.yaxis.axis_label = "V*"
        p_vs.add_tools(HoverTool(tooltips=[("County","@name"),
                                            ("Wage","@wage{0}"),
                                            ("V*","@Vstar{0.000}")]))

        p_xi = figure(height=H_FIG, width=W_FIG,
                       title="Amenity ξ ranking (highest → lowest)")
        p_xi.vbar(x="rank", top="xi", width=0.85, source=self.src_xi,
                  fill_color="color", line_color="black", line_width=0.3)
        p_xi.xaxis.axis_label = "county rank"
        p_xi.yaxis.axis_label = "ξ"
        p_xi.add_tools(HoverTool(tooltips=[("County","@name"),
                                            ("ξ","@xi{0.000}")]))

        p_ratio = figure(height=H_FIG, width=W_FIG,
                          title="Female/male market hours ratio")
        p_ratio.scatter("wage", "ratio", source=self.src_spatial, size=8,
                        fill_color="#d62728", line_color="black",
                        line_width=0.5)
        p_ratio.xaxis.axis_label = "wage (KSh/hr)"
        p_ratio.yaxis.axis_label = "L^M_f / L^M_m"
        p_ratio.add_tools(HoverTool(tooltips=[("County","@name"),
                                               ("Wage","@wage{0}"),
                                               ("ratio","@ratio{0.000}")]))

        p_emp = figure(height=H_FIG, width=W_FIG,
                        title="Sectoral market employment by county",
                        x_range=[BASELINE_COUNTIES[c]["name"]
                                  for c in sorted(BASELINE_COUNTIES)])
        p_emp.vbar_stack(["LM_xf","LM_xn","LM_c","LM_d"],
                         x="name", source=self.src_emp,
                         color=["#ff7f0e","#9467bd","#d62728","#2ca02c"],
                         legend_label=["food","non-food","care","domestic"],
                         width=0.8)
        p_emp.xaxis.major_label_orientation = math.pi / 2.5
        p_emp.xaxis.major_label_text_font_size = "6pt"
        p_emp.yaxis.axis_label = "employment (efficiency hours)"
        p_emp.legend.location = "top_left"
        p_emp.legend.label_text_font_size = "7pt"

        self.p_vs, self.p_xi, self.p_ratio, self.p_emp = (
            p_vs, p_xi, p_ratio, p_emp)

        # ----- Map ---------------------------------------------------- #
        self.map_color_mapper = LinearColorMapper(palette=RdBu11,
                                                    low=0, high=1)
        p_map = figure(height=300, width=W_FIG,
                        title=f"{self.label}: click a county",
                        match_aspect=True, tools="tap,reset,pan,wheel_zoom")
        if COUNTY_GEOMETRY is not None:
            p_map.patches(xs="xs", ys="ys", source=self.map_source,
                          fill_color={"field":"value",
                                       "transform":self.map_color_mapper},
                          line_color="#444", line_width=0.5,
                          # Make selected/non-selected appearance distinct
                          selection_line_color="black",
                          selection_line_width=2.0,
                          nonselection_fill_alpha=1.0,
                          nonselection_line_alpha=1.0)
        else:
            p_map.scatter("lon", "lat", source=self.map_source, size=14,
                          fill_color={"field":"value",
                                       "transform":self.map_color_mapper},
                          line_color="black", line_width=0.5)
        p_map.add_tools(HoverTool(tooltips=[("County","@name"),
                                              ("Value","@value{0.000}")]))
        cb = ColorBar(color_mapper=self.map_color_mapper,
                       ticker=BasicTicker(), location=(0, 0),
                       label_standoff=4, height=200, width=12)
        p_map.add_layout(cb, "right")
        p_map.xaxis.axis_label = "lon"
        p_map.yaxis.axis_label = "lat"
        self.p_map = p_map

        # ----- Counterfactual table ----------------------------------- #
        self.cf_table = DataTable(
            source=self.cf_table_src, width=W_FIG * 2, height=140,
            columns=[
                TableColumn(field="scenario", title="Scenario"),
                TableColumn(field="dGDP", title="ΔGDP %",
                             formatter=NumberFormatter(format="0.00")),
                TableColumn(field="dPm",  title="ΔP_m (pp)",
                             formatter=NumberFormatter(format="0.00")),
                TableColumn(field="dPf",  title="ΔP_f (pp)",
                             formatter=NumberFormatter(format="0.00")),
                TableColumn(field="dratio", title="Δ ratio",
                             formatter=NumberFormatter(format="0.000")),
                TableColumn(field="dN",   title="ΔN %",
                             formatter=NumberFormatter(format="0.00")),
            ])

        # ----- CF choropleth grid (3 × 4) ----------------------------- #
        # Per-indicator value format for the tooltip
        _cf_fmt = {
            "dGDP":   "{+0.00}%",     # percentage change
            "dPf":    "{+0.00} pp",   # percentage points
            "dratio": "{+0.000}",     # change in L^M_f / L^M_m
            "dN":     "{+0.00}%",     # percentage change in population
        }
        self.cf_map_figs = {}
        for sc, sc_title in [("wage","wage_gap=1"),
                              ("pc","p_c × 0.7"),
                              ("pd","p_d × 0.7")]:
            for ind, ind_title in [("dGDP","ΔGDP%"),
                                    ("dPf","ΔP_f"),
                                    ("dratio","Δratio"),
                                    ("dN","ΔN%")]:
                src = self.cf_maps[(sc, ind)]
                cm  = LinearColorMapper(palette=RdBu11, low=-1, high=1)
                # Slightly wider (235 vs 200) to make room for the colorbar
                # without squashing the map.
                f = figure(height=180, width=235,
                            title=f"{sc_title}: {ind_title}",
                            match_aspect=True,
                            tools="pan,wheel_zoom,zoom_in,zoom_out,reset",
                            toolbar_location="right")
                f.toolbar.logo = None
                f.toolbar.autohide = False
                if COUNTY_GEOMETRY is not None:
                    f.patches(xs="xs", ys="ys", source=src,
                              fill_color={"field":"value",
                                           "transform":cm},
                              line_color="#444", line_width=0.3)
                else:
                    f.scatter("lon", "lat", source=src, size=10,
                              fill_color={"field":"value",
                                           "transform":cm},
                              line_color="black", line_width=0.3)
                f.add_tools(HoverTool(tooltips=[
                    ("County", "@name"),
                    (ind_title, f"@value{_cf_fmt[ind]}"),
                ]))
                # Colorbar on the right.  Compact: 8 px wide, no title
                # (the figure title already names the indicator).
                cf_cb = ColorBar(color_mapper=cm,
                                 ticker=BasicTicker(desired_num_ticks=4),
                                 location=(0, 0), label_standoff=4,
                                 height=130, width=8,
                                 major_label_text_font_size="7pt")
                f.add_layout(cf_cb, "right")
                f.xaxis.visible = False
                f.yaxis.visible = False
                self.cf_map_figs[(sc, ind, "fig")] = f
                self.cf_map_figs[(sc, ind, "cmap")] = cm

    # ------------------------------------------------------------------ #
    # Layout                                                             #
    # ------------------------------------------------------------------ #

    def _build_layout(self):
        # Tab 1: per-h
        tab1 = TabPanel(child=column(
            row(self.p_hours, self.p_gap),
            row(self.p_sh,    self.p_hm),
            row(self.p_part),
        ), title="Per-h plots (selected county)")

        # Tab 2: spatial summary
        tab2 = TabPanel(child=column(
            row(self.solve_all_btn),
            row(self.p_vs, self.p_ratio),
            row(self.p_xi, self.p_emp),
        ), title="Spatial summary")

        # Tab 3: counterfactuals
        cf_grid_rows = []
        for sc in ("wage", "pc", "pd"):
            row_figs = [self.cf_map_figs[(sc, ind, "fig")]
                        for ind in ("dGDP", "dPf", "dratio", "dN")]
            cf_grid_rows.append(row(*row_figs))
        tab3 = TabPanel(child=column(
            row(self.cf_btn_wage, self.cf_btn_pc, self.cf_btn_pd),
            self.cf_table,
            *cf_grid_rows,
        ), title="Counterfactuals")

        self.tabs = Tabs(tabs=[tab1, tab2, tab3])

        # Parameter blocks (two-per-row), each with an explicit width=380 so
        # the controls column has a definite size and never collapses.
        def two_per_row(inputs_dict, fields):
            rows = []
            it = iter(fields)
            for k1, _ in it:
                try:
                    k2, _ = next(it)
                    rows.append(row(inputs_dict[k1], inputs_dict[k2]))
                except StopIteration:
                    rows.append(row(inputs_dict[k1]))
            return column(*rows, width=380)

        global_block = column(
            Div(text="<details open><summary><b>Global parameters</b> "
                     "(click to collapse)</summary>", width=380),
            two_per_row(self.global_inputs, GLOBAL_FIELDS),
            Div(text="</details>", width=380),
            width=380,
        )
        county_block = column(
            self.county_label,
            Div(text="<details open><summary><b>County-specific parameters"
                     "</b></summary>", width=380),
            two_per_row(self.county_inputs, COUNTY_FIELDS),
            Div(text="</details>", width=380),
            width=380,
        )

        # Controls column: header, map metric, map, buttons, status,
        # then the parameter blocks.  Explicit width=420 (380 plus padding).
        controls = column(
            Div(text=f"<h2 style='margin:0'>{self.label}</h2>", width=400),
            self.map_metric,
            self.p_map,
            row(self.solve_btn, self.reset_btn),
            self.status,
            global_block,
            county_block,
            width=420,
        )

        # Top-level layout: side-by-side row of controls and tabs (matches
        # the layout shape of the existing 3-good app).
        self.layout = row(controls, self.tabs)

    # ------------------------------------------------------------------ #
    # Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def _wire_callbacks(self):
        def on_tap(*_):
            sel = self.map_source.selected.indices
            if not sel:
                return
            idx  = sel[0]
            cnum = self.map_source.data["id"][idx]
            self._select_county(int(cnum))
        self.map_source.selected.on_change(
            "indices", lambda attr, old, new: on_tap())

        self.solve_btn.on_click(self._solve_selected_and_update)
        self.reset_btn.on_click(self._reset)
        self.solve_all_btn.on_click(self._solve_all_and_update)
        self.cf_btn_wage.on_click(lambda: self._run_cf("wage"))
        self.cf_btn_pc.on_click(lambda: self._run_cf("pc"))
        self.cf_btn_pd.on_click(lambda: self._run_cf("pd"))
        self.map_metric.on_change("value",
                                   lambda attr, old, new: self._refresh_map())

    def _select_county(self, cnum: int):
        # Save the current county_inputs before switching
        self._sync_inputs_to_county(self.selected_county)
        self.selected_county = cnum
        cp = self.counties_p[cnum]
        for k, _ in COUNTY_FIELDS:
            self.county_inputs[k].value = float(cp[k])
        self.county_label.text = (
            f"<b>Selected county:</b> {COUNTY_NAMES[cnum]} (#{cnum})")
        self._solve_selected_and_update()

    def _sync_inputs_to_state(self):
        for k, _ in GLOBAL_FIELDS:
            v = self.global_inputs[k].value
            if v is not None:
                self.global_p[k] = float(v)
        self._sync_inputs_to_county(self.selected_county)

    def _sync_inputs_to_county(self, cnum: int):
        for k, _ in COUNTY_FIELDS:
            v = self.county_inputs[k].value
            if v is not None:
                self.counties_p[cnum][k] = float(v)

    def _solve_selected_and_update(self):
        self._sync_inputs_to_state()
        self.status.text = (f"<i>{self.label}: solving "
                             f"{COUNTY_NAMES[self.selected_county]} ...</i>")
        try:
            r = solve_one_county(self.global_p,
                                  self.counties_p[self.selected_county])
            # Hours
            self.src_hours.data = dict(
                h=list(H_GRID),
                LM_m=list(r["LM_m"]), LM_f=list(r["LM_f"]),
                Lxf_m=list(r["Lxf_m"]), Lxf_f=list(r["Lxf_f"]),
                Lc_m=list(r["Lc_m"]), Lc_f=list(r["Lc_f"]),
                Ld_m=list(r["Ld_m"]), Ld_f=list(r["Ld_f"]))
            # Gaps (woman − man)
            self.src_gap.data = dict(
                h=list(H_GRID),
                gap_M =list(r["LM_f"]  - r["LM_m"]),
                gap_xf=list(r["Lxf_f"] - r["Lxf_m"]),
                gap_c =list(r["Lc_f"]  - r["Lc_m"]),
                gap_d =list(r["Ld_f"]  - r["Ld_m"]))
            # Shares
            self.src_shares.data = dict(
                h=list(H_GRID),
                th_xf=list(r["th_xf"]), th_xn=list(r["th_xn"]),
                th_c =list(r["th_c"]),  th_d =list(r["th_d"]))
            # Home/market
            self.src_homemkt.data = dict(
                h=list(H_GRID),
                SxfH=list(r["SxfH_share"]), SxfM=list(r["SxfM_share"]),
                ScH =list(r["ScH_share"]),  ScM =list(r["ScM_share"]),
                SdH =list(r["SdH_share"]),  SdM =list(r["SdM_share"]))
            # Participation
            self.src_part.data = dict(
                h=list(H_GRID),
                P_m=list(r["P_m"]), P_f=list(r["P_f"]))

            self.status.text = (
                f"<b>{self.label}: {COUNTY_NAMES[self.selected_county]}</b> "
                f"solved · convergence {r['n_conv']}/4 · "
                f"L^M ratio (m/f) = "
                f"{np.nanmean(r['LM_m'])/max(np.nanmean(r['LM_f']),1e-9):.2f}")
        except Exception as e:
            self.status.text = (f"<b style='color:#b00'>"
                                 f"{self.label}: error</b>: {e!r}")

    def _solve_all_and_update(self):
        self._sync_inputs_to_state()
        self.status.text = (f"<i>{self.label}: solving all 47 counties "
                             f"(this can take 20-40s) ...</i>")
        try:
            self.spatial = solve_all(self.global_p, self.counties_p)
            self.spatial_summary = summarise_spatial(
                self.spatial, self.counties_p)
            s = self.spatial_summary
            ids = s["counties"]

            self.src_spatial.data = dict(
                county=ids,
                name=[COUNTY_NAMES[c] for c in ids],
                Vstar=list(s["Vstar"]), xi=list(s["xi"]),
                wage =list(s["wages"]),
                Pm   =list(s["Pm"]),    Pf =list(s["Pf"]),
                ratio=list(s["ratio"]),
                LM_m =list(s["LM_m"]),  LM_f=list(s["LM_f"]),
                GDP  =list(s["GDP"]),   N=list(s["N"]),
            )

            order = np.argsort(s["xi"])[::-1]
            cols = ["#1a9850" if x > 0 else "#d73027"
                    for x in s["xi"][order]]
            self.src_xi.data = dict(
                rank=list(range(len(order))),
                name=[COUNTY_NAMES[ids[i]] for i in order],
                xi=list(s["xi"][order]),
                color=cols,
            )
            # Sectoral employment, by name (order of vbar_stack x_range)
            name_order = [COUNTY_NAMES[c] for c in sorted(BASELINE_COUNTIES)]
            self.src_emp.data = dict(
                name=name_order,
                LM_xf=list(s["LM_xf"]),
                LM_xn=list(s["LM_xn"]),
                LM_c =list(s["LM_c"]),
                LM_d =list(s["LM_d"]),
            )
            self._refresh_map()
            self.status.text = (
                f"<b>{self.label}: 47 counties solved</b> · "
                f"Ū = {s['Ubar']:.3f} · "
                f"V* range [{np.nanmin(s['Vstar']):.3f}, "
                f"{np.nanmax(s['Vstar']):.3f}]")
        except Exception as e:
            self.status.text = (f"<b style='color:#b00'>"
                                 f"{self.label}: solve-all error</b>: {e!r}")

    def _reset(self):
        self.global_p   = dict(BASELINE_PARAMS)
        self.counties_p = {cid: dict(BASELINE_COUNTIES[cid])
                           for cid in BASELINE_COUNTIES}
        for cid in self.counties_p:
            self.counties_p[cid]["county_id"] = cid
        for k, _ in GLOBAL_FIELDS:
            self.global_inputs[k].value = float(self.global_p[k])
        cp = self.counties_p[self.selected_county]
        for k, _ in COUNTY_FIELDS:
            self.county_inputs[k].value = float(cp[k])
        self.spatial = None
        self.spatial_summary = None
        self.cf_results = {}
        self.cf_summaries = {}
        self.cf_table_src.data = dict(
            scenario=[], dGDP=[], dPm=[], dPf=[], dratio=[], dN=[])
        for src in self.cf_maps.values():
            src.data = _empty_map_data()
        self._refresh_map()
        self._solve_selected_and_update()
        self.status.text = f"<i>{self.label}: reset to baseline.</i>"

    def _refresh_map(self):
        ids = sorted(BASELINE_COUNTIES.keys())
        metric = self.map_metric.value
        if metric == "wage":
            vals = {c: float(self.counties_p[c]["w_ell"]) * E_SCALE
                    for c in ids}
            finite = list(vals.values())
            lo, hi = min(finite), max(finite)
        elif metric == "Pf (baseline)" and self.spatial_summary is not None:
            s = self.spatial_summary
            vals = {c: float(v) for c, v in zip(s["counties"], s["Pf"])}
            finite = [v for v in vals.values() if np.isfinite(v)]
            lo, hi = (min(finite), max(finite)) if finite else (0, 1)
        elif metric == "ΔGDP (last CF)" and self.cf_summaries:
            last = list(self.cf_summaries.keys())[-1]
            base_gdp = self.spatial_summary["GDP"]
            new_gdp  = self.cf_summaries[last]["GDP"]
            with np.errstate(divide="ignore", invalid="ignore"):
                pct = 100.0 * (new_gdp / base_gdp - 1.0)
            vals = {c: float(v)
                    for c, v in zip(self.spatial_summary["counties"], pct)}
            finite = [v for v in vals.values() if np.isfinite(v)]
            if finite:
                m = max(abs(min(finite)), abs(max(finite))) or 1.0
                lo, hi = -m, m
            else:
                lo, hi = -1, 1
        else:
            vals = {c: float(self.counties_p[c]["w_ell"]) * E_SCALE
                    for c in ids}
            finite = list(vals.values())
            lo, hi = min(finite), max(finite)

        if COUNTY_GEOMETRY is not None:
            # Patches-shaped: one row per polygon patch.  Look up the
            # value for each patch's county_id.
            patch_ids = COUNTY_GEOMETRY["id"]
            self.map_source.data = dict(
                xs   = list(COUNTY_GEOMETRY["xs"]),
                ys   = list(COUNTY_GEOMETRY["ys"]),
                id   = list(patch_ids),
                name = list(COUNTY_GEOMETRY["name"]),
                value= [vals.get(cid, float("nan")) for cid in patch_ids],
            )
        else:
            # Centroid fallback
            self.map_source.data = dict(
                id=ids,
                name=[COUNTY_NAMES[c] for c in ids],
                lat =[BASELINE_COUNTIES[c]["lat"] for c in ids],
                lon =[BASELINE_COUNTIES[c]["lon"] for c in ids],
                value=[vals.get(c, float("nan")) for c in ids],
            )
        self.map_color_mapper.low  = lo
        self.map_color_mapper.high = hi

    # ------------------------------------------------------------------ #
    # Counterfactuals                                                    #
    # ------------------------------------------------------------------ #

    def _run_cf(self, kind: str):
        if self.spatial is None:
            self.status.text = (
                f"<b style='color:#b00'>{self.label}: run "
                f"\"Solve all 47\" first.</b>")
            return
        self._sync_inputs_to_state()
        cf_global = dict(self.global_p)
        cf_counties = {c: dict(p) for c, p in self.counties_p.items()}
        if kind == "wage":
            cf_global["wage_gap"] = 1.0
            scen_label = "wage_gap = 1"
        elif kind == "pc":
            for c in cf_counties:
                cf_counties[c]["pc"] *= 0.7
                cf_counties[c]["AM_c"] = (cf_counties[c]["w_ell"]
                                            / max(cf_counties[c]["pc"], 1e-9))
            scen_label = "p_c × 0.7"
        elif kind == "pd":
            for c in cf_counties:
                cf_counties[c]["pd"] *= 0.7
                cf_counties[c]["AM_d"] = (cf_counties[c]["w_ell"]
                                            / max(cf_counties[c]["pd"], 1e-9))
            scen_label = "p_d × 0.7"
        else:
            return

        self.status.text = (f"<i>{self.label}: running CF "
                             f"\"{scen_label}\" ...</i>")
        try:
            spatial_cf = solve_all(cf_global, cf_counties)
            summary_cf = summarise_spatial(spatial_cf, cf_counties)

            # Migration: population reallocates by logit on V* + xi
            # (population-weighted variant; see solver_functions.migration_update).
            # We use baseline xi (not re-calibrated) so the counterfactual
            # picks up only the structural change, not the amenity reset.
            from solver_functions import migration_update
            base = self.spatial_summary
            ids  = base["counties"]
            N0   = np.array([float(self.counties_p[c]["N"]) for c in ids])
            xi   = base["xi"]
            sigma_mig = float(cf_global.get("sigma_mig", 1.0))
            N_new = migration_update(N0, summary_cf["Vstar"], xi, sigma_mig)
            # Inject the migration-updated populations into the summary
            summary_cf = dict(summary_cf)
            summary_cf["N"]   = N_new
            summary_cf["GDP"] = (summary_cf["GDP"] / np.maximum(N0, 1e-9)) * N_new

            self.cf_results[kind]   = spatial_cf
            self.cf_summaries[kind] = summary_cf
            self._update_cf_outputs()
            self._refresh_map()
            self.status.text = (f"<b>{self.label}: CF \"{scen_label}\" "
                                 f"done.</b>")
        except Exception as e:
            self.status.text = (f"<b style='color:#b00'>"
                                 f"{self.label}: CF error</b>: {e!r}")

    def _update_cf_outputs(self):
        base = self.spatial_summary
        if base is None:
            return
        ids = base["counties"]
        rows = []
        for kind, sc_label in [("wage","wage_gap=1"),
                                ("pc","p_c×0.7"),
                                ("pd","p_d×0.7")]:
            if kind not in self.cf_summaries:
                continue
            cf = self.cf_summaries[kind]
            # National (population-weighted) deltas
            Ns = base["N"]
            with np.errstate(invalid="ignore", divide="ignore"):
                dGDP_pct = 100.0 * (cf["GDP"] / base["GDP"] - 1.0)
                dN_pct   = 100.0 * (cf["N"]   / base["N"]   - 1.0)
            dPm = (cf["Pm"]   - base["Pm"])    * 100.0   # percentage points
            dPf = (cf["Pf"]   - base["Pf"])    * 100.0
            dratio = cf["ratio"] - base["ratio"]
            wgt = Ns / max(Ns.sum(), 1e-9)
            rows.append(dict(
                scenario=sc_label,
                dGDP=float(np.nansum(dGDP_pct * wgt)),
                dPm =float(np.nansum(dPm    * wgt)),
                dPf =float(np.nansum(dPf    * wgt)),
                dratio=float(np.nansum(dratio * wgt)),
                dN  =float(np.nansum(dN_pct * wgt)),
            ))
            # Update the per-county choropleth maps for this scenario
            for ind, vec in [("dGDP", dGDP_pct),
                              ("dPf",  dPf),
                              ("dratio", dratio),
                              ("dN",   dN_pct)]:
                src = self.cf_maps[(kind, ind)]
                # Build a county_id -> value lookup
                vmap = {c: float(v) for c, v in zip(ids, vec)}
                if COUNTY_GEOMETRY is not None:
                    patch_ids = COUNTY_GEOMETRY["id"]
                    src.data = dict(
                        xs   = list(COUNTY_GEOMETRY["xs"]),
                        ys   = list(COUNTY_GEOMETRY["ys"]),
                        id   = list(patch_ids),
                        name = list(COUNTY_GEOMETRY["name"]),
                        value= [vmap.get(cid, float("nan"))
                                 for cid in patch_ids],
                    )
                else:
                    src.data = dict(
                        id=ids,
                        name=[COUNTY_NAMES[c] for c in ids],
                        lat=[BASELINE_COUNTIES[c]["lat"] for c in ids],
                        lon=[BASELINE_COUNTIES[c]["lon"] for c in ids],
                        value=list(vec),
                    )
                cm = self.cf_map_figs[(kind, ind, "cmap")]
                finite = vec[np.isfinite(vec)]
                if finite.size:
                    m = max(abs(np.nanmin(finite)),
                             abs(np.nanmax(finite))) or 1.0
                    cm.low, cm.high = -m, m

        if rows:
            self.cf_table_src.data = dict(
                scenario=[r["scenario"] for r in rows],
                dGDP=[r["dGDP"] for r in rows],
                dPm =[r["dPm"]  for r in rows],
                dPf =[r["dPf"]  for r in rows],
                dratio=[r["dratio"] for r in rows],
                dN  =[r["dN"]   for r in rows],
            )


# =========================================================================== #
# Document                                                                    #
# =========================================================================== #

ui_A = ScenarioUI("Scenario A")
ui_B = ScenarioUI("Scenario B")

page = row(
    ui_A.layout,
    Spacer(width=24),
    ui_B.layout,
    sizing_mode="stretch_width",
)

curdoc().add_root(page)
curdoc().title = "Kenya Time-Use Model"
