#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — Bokeh app for the Kenya Time-Use spatial equilibrium model.

Two side-by-side scenario columns (A and B) for comparison. Each column has:

  • A clickable Kenya map (centroid circles) at the top for county selection.
  • Global parameters section (PIGL, CES, rho, phi, wage_gap, sigma_u, u_bar).
  • County-specific parameters section (auto-loaded for the selected county):
    w_ell, pc, pd, A_c, A_d, N, and the six D weights.
  • Solve / Reset buttons.
  • Counterfactual panel with three buttons (wage_gap=1, p^c×0.7, p^d×0.7).
  • Tabs:
      - Per-h plots (for the selected county): hours, gaps, shares, home/market
      - Spatial: V*, xi, gender gap by county (requires "Solve all 47")
      - Counterfactuals: summary table + nine choropleth maps
                         (requires "Solve all 47" first)

Run with:
    bokeh serve --show main.py

Notes:
  - The food/non-food extension is documented in the model PDF and calibrated
    in calibrated_food_params.json, but the underlying solver does not yet
    integrate it. The food parameters appear in the app as read-only display
    fields with a note. Activation pending the 8-equation solver rewrite.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row, Spacer
from bokeh.models import (
    Button, ColumnDataSource, Div, HoverTool, Legend, NumericInput,
    Select, Tabs, TabPanel, DataTable, TableColumn, NumberFormatter,
    LinearColorMapper, ColorBar, BasicTicker,
)
from bokeh.plotting import figure
from bokeh.palettes import RdBu11, PiYG11, PuOr11
from bokeh.events import Tap

# Local imports — model code in the same folder
from classes import ModelParams, Household
from solver import SolverState, solve_model

# ═══════════════════════════════════════════════════════════════════════
# Paths and constants
# ═══════════════════════════════════════════════════════════════════════

HERE = Path(__file__).parent
ROOT = HERE.parent

PARAMS_PATH      = HERE / "calibrated_params.json"
FUND_PATH        = HERE / "county_fundamentals.csv"
DCNTY_PATH       = HERE / "county_D_weights.csv"
FOOD_PATH        = ROOT / "calibrated_food_params.json"
COUNTIES_GEO     = ROOT / "kenya_counties.json"

E_SCALE = 1000.0   # KSh -> 1000-KSh, consistent with the rest of the project

# County names (1..47) and centroids — stored in kenya_counties.json
def _load_county_centroids():
    with open(COUNTIES_GEO) as f:
        d = json.load(f)
    # JSON: {"1": ["Mombasa", lat, lon], ...}
    centroids = {}
    for k, v in d.items():
        centroids[int(k)] = {"name": v[0], "lat": v[1], "lon": v[2]}
    return centroids


COUNTY = _load_county_centroids()
N_COUNTIES = len(COUNTY)


# ═══════════════════════════════════════════════════════════════════════
# Calibration loader
# ═══════════════════════════════════════════════════════════════════════

def load_baseline() -> dict:
    """
    Load the full baseline calibration into a single nested dict:
      {
        'global':  {eps_engel, beta_x, beta_c, beta_d, kappa_*, omega_*,
                    eta_*, rho, phi, wage_gap,
                    sigma_u_m, sigma_u_f, u_bar_m, u_bar_f},
        'food':    {beta_xf, beta_xn, kappa_xf, kappa_xn, D_xf_m, D_xf_f,
                    omega_xf, eta_xf, A_xf, ...},
        'counties': {1: {w_ell, pc, pd, A_c, A_d, N, D_M_m, D_c_m, D_d_m,
                         D_M_f, D_c_f, D_d_f}, ..., 47: {...}}
      }
    """
    import pandas as pd

    with open(PARAMS_PATH) as f:
        p = json.load(f)
    glob = {
        "eps_engel": p["eps_engel"],
        "beta_x":    p["beta_x"], "beta_c": p["beta_c"], "beta_d": p["beta_d"],
        "kappa_x":   p["kappa_x"],"kappa_c":p["kappa_c"],"kappa_d":p["kappa_d"],
        "omega_c":   p["omega_c"],"omega_d":p["omega_d"],
        "eta_c":     p["eta_c"],  "eta_d":  p["eta_d"],
        "rho":       p["rho"],    "phi":    p["phi"],
        "wage_gap":  p["wage_gap"],
        "sigma_u_m": p.get("sigma_u_m", 3.144),
        "sigma_u_f": p.get("sigma_u_f", 1.894),
        "u_bar_m":   p.get("u_bar_m",   1.933),
        "u_bar_f":   p.get("u_bar_f",   3.762),
    }

    food = {}
    if FOOD_PATH.exists():
        with open(FOOD_PATH) as f:
            food = json.load(f)

    fund = pd.read_csv(FUND_PATH)
    Dcty = pd.read_csv(DCNTY_PATH).set_index("county") if DCNTY_PATH.exists() else None

    counties: Dict[int, Dict[str, float]] = {}
    for _, row in fund.iterrows():
        cnum = int(row["county"])
        # wages and prices in KSh -> 1000-KSh units
        w_ell = float(row["w_ell"]) / E_SCALE
        pc_raw = row.get("pc")
        if pc_raw is None or (isinstance(pc_raw, float) and np.isnan(pc_raw)):
            pc_raw = row.get("w_care", np.nan)
        if isinstance(pc_raw, float) and np.isnan(pc_raw):
            pc_raw = 68.7
        pc = float(pc_raw) / E_SCALE
        pd_raw = row.get("pd")
        if pd_raw is None or (isinstance(pd_raw, float) and np.isnan(pd_raw)):
            pd_raw = row.get("w_domestic", np.nan)
        if isinstance(pd_raw, float) and np.isnan(pd_raw):
            pd_raw = 34.9
        pd_v = float(pd_raw) / E_SCALE

        d = {
            "w_ell": w_ell, "pc": pc, "pd": pd_v,
            "A_c":   float(row.get("A_c_proxy", 1.0)),
            "A_d":   float(row.get("A_d_proxy", 1.0)),
            "N":     float(row.get("N_tus", 1.0)),
        }
        if Dcty is not None and cnum in Dcty.index:
            r = Dcty.loc[cnum]
            d.update({
                "D_M_m": float(r["D_M_m"]), "D_c_m": float(r["D_c_m"]),
                "D_d_m": float(r["D_d_m"]),
                "D_M_f": float(r["D_M_f"]), "D_c_f": float(r["D_c_f"]),
                "D_d_f": float(r["D_d_f"]),
            })
        else:
            d.update({"D_M_m": p["D_M_m"], "D_c_m": p["D_c_m"],
                      "D_d_m": p["D_d_m"], "D_M_f": p["D_M_f"],
                      "D_c_f": p["D_c_f"], "D_d_f": p["D_d_f"]})
        counties[cnum] = d

    return {"global": glob, "food": food, "counties": counties}


BASELINE = load_baseline()


# ═══════════════════════════════════════════════════════════════════════
# Solver wrappers
# ═══════════════════════════════════════════════════════════════════════

def build_mp(global_params: dict, county_params: dict) -> ModelParams:
    """Construct a ModelParams from current global + county-specific values."""
    return ModelParams(
        eps_engel=global_params["eps_engel"],
        beta_x=global_params["beta_x"], beta_c=global_params["beta_c"],
        beta_d=global_params["beta_d"],
        kappa_x=global_params["kappa_x"], kappa_c=global_params["kappa_c"],
        kappa_d=global_params["kappa_d"],
        omega_c=global_params["omega_c"], omega_d=global_params["omega_d"],
        eta_c=global_params["eta_c"],     eta_d=global_params["eta_d"],
        D_M_m=county_params["D_M_m"], D_c_m=county_params["D_c_m"],
        D_d_m=county_params["D_d_m"],
        D_M_f=county_params["D_M_f"], D_c_f=county_params["D_c_f"],
        D_d_f=county_params["D_d_f"],
        rho=global_params["rho"], phi=global_params["phi"],
    )


H_GRID = np.array([0.5, 1.0, 2.0, 3.0])


def solve_county(global_params: dict, county_params: dict,
                 a: float = 0.2) -> dict:
    """Solve one county over the h-grid. Returns dict of arrays."""
    mp = build_mp(global_params, county_params)
    wage_gap = float(global_params["wage_gap"])
    Nh = len(H_GRID)

    LM_m = np.full(Nh, np.nan); LM_f = np.full(Nh, np.nan)
    Lc_m = np.full(Nh, np.nan); Lc_f = np.full(Nh, np.nan)
    Ld_m = np.full(Nh, np.nan); Ld_f = np.full(Nh, np.nan)
    E_arr = np.full(Nh, np.nan); V_arr = np.full(Nh, np.nan)
    th_x = np.full(Nh, np.nan); th_c = np.full(Nh, np.nan); th_d = np.full(Nh, np.nan)
    ScH_share = np.full(Nh, np.nan); ScM_share = np.full(Nh, np.nan)
    SdH_share = np.full(Nh, np.nan); SdM_share = np.full(Nh, np.nan)
    conv = np.zeros(Nh, dtype=bool)

    state = SolverState(verbose=False, max_iter=20000, tol=1e-8,
                        damping=0.15, adapt_damping=True)
    L_guess = (0.15, 0.10, 0.05, 0.10, 0.04, 0.08)

    for i, h in enumerate(H_GRID):
        y_m = float(county_params["w_ell"] * h)
        y_f = float(wage_gap * county_params["w_ell"] * h)
        hh = Household(
            params=mp, y_m=y_m, y_f=y_f,
            pc=float(county_params["pc"]), pd=float(county_params["pd"]),
            a=a, A_c=float(county_params["A_c"]), A_d=float(county_params["A_d"]),
        )
        try:
            hh, state = solve_model(mp, hh, state, L0=L_guess)
            LM_m[i] = state.LM_m; LM_f[i] = state.LM_f
            Lc_m[i] = state.Lc_m; Lc_f[i] = state.Lc_f
            Ld_m[i] = state.Ld_m; Ld_f[i] = state.Ld_f
            E_arr[i] = float(hh.E)
            th_x[i] = float(hh.th_x); th_c[i] = float(hh.th_c); th_d[i] = float(hh.th_d)
            Sc_tot = float(hh.Sc); Sd_tot = float(hh.Sd)
            ScH_share[i] = hh.ScH / Sc_tot if Sc_tot > 0 else np.nan
            ScM_share[i] = hh.ScM / Sc_tot if Sc_tot > 0 else np.nan
            SdH_share[i] = hh.SdH / Sd_tot if Sd_tot > 0 else np.nan
            SdM_share[i] = hh.SdM / Sd_tot if Sd_tot > 0 else np.nan
            # Value (per Block E)
            V_arr[i] = (hh.E / hh.B) ** mp.eps_engel / mp.eps_engel - \
                       hh.L ** (1.0 + 1.0 / mp.phi) / (1.0 + 1.0 / mp.phi)
            conv[i] = state.converged
            L_guess = tuple(max(1e-14, v) for v in
                            [state.LM_m, state.LM_f, state.Lc_m,
                             state.Lc_f, state.Ld_m, state.Ld_f])
        except Exception:
            pass

    # Female participation from logit (Block E)
    sigma_u = global_params["sigma_u_f"]; u_bar = global_params["u_bar_f"]
    V_P = global_params["phi"] * county_params["w_ell"] * wage_gap * H_GRID
    Pf = 1.0 / (1.0 + np.exp((u_bar - V_P) / max(sigma_u, 1e-9)))

    return {
        "h": H_GRID, "LM_m": LM_m, "LM_f": LM_f,
        "Lc_m": Lc_m, "Lc_f": Lc_f, "Ld_m": Ld_m, "Ld_f": Ld_f,
        "E": E_arr, "V": V_arr, "Pf": Pf,
        "th_x": th_x, "th_c": th_c, "th_d": th_d,
        "ScH_share": ScH_share, "ScM_share": ScM_share,
        "SdH_share": SdH_share, "SdM_share": SdM_share,
        "converged": conv, "n_conv": int(conv.sum()),
        "N_county": float(county_params["N"]),
        "wage": float(county_params["w_ell"] * E_SCALE),  # back to KSh/hr
    }


def solve_all_counties(global_params: dict,
                       counties_params: Dict[int, dict]) -> Dict[int, dict]:
    """Solve every county. Returns {county_id: result_dict}."""
    out = {}
    for cnum, cp in counties_params.items():
        out[cnum] = solve_county(global_params, cp)
    return out


def summarise_spatial(results: Dict[int, dict]) -> dict:
    """National summary metrics."""
    Vstar = np.array([np.nanmean(results[c]["V"]) for c in sorted(results)])
    GDP   = np.array([np.nanmean(results[c]["E"]) * results[c]["N_county"]
                      for c in sorted(results)])
    Pf    = np.array([np.nanmean(results[c]["Pf"]) for c in sorted(results)])
    LMm   = np.array([np.nanmean(results[c]["LM_m"]) for c in sorted(results)])
    LMf   = np.array([np.nanmean(results[c]["LM_f"]) for c in sorted(results)])
    ratio = LMf / np.maximum(LMm, 1e-9)
    wages = np.array([results[c]["wage"] for c in sorted(results)])
    # Calibrate amenities so weighted mean V* + xi = 0
    weights = np.array([results[c]["N_county"] for c in sorted(results)])
    Ubar = float(np.average(Vstar, weights=weights))
    xi = Ubar - Vstar
    return {"counties": sorted(results.keys()),
            "Vstar": Vstar, "GDP": GDP, "Pf": Pf,
            "LM_m": LMm, "LM_f": LMf, "ratio": ratio,
            "wages": wages, "xi": xi, "Ubar": Ubar}


# ═══════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════

GLOBAL_FIELDS = [
    ("eps_engel", "ε (Engel)"), ("rho", "ρ (CES disutility)"),
    ("phi", "φ (Frisch elast.)"), ("wage_gap", "wage_gap"),
    ("beta_x", "β_x"), ("beta_c", "β_c"), ("beta_d", "β_d"),
    ("kappa_x", "κ_x"), ("kappa_c", "κ_c"), ("kappa_d", "κ_d"),
    ("omega_c", "ω_c"), ("omega_d", "ω_d"),
    ("eta_c", "η_c"), ("eta_d", "η_d"),
    ("sigma_u_m", "σ_u^m"), ("sigma_u_f", "σ_u^f"),
    ("u_bar_m", "ū^m"),     ("u_bar_f", "ū^f"),
]

COUNTY_FIELDS = [
    ("w_ell", "wage (1000-KSh/hr)"),
    ("pc", "p_c (1000-KSh/hr)"), ("pd", "p_d (1000-KSh/hr)"),
    ("A_c", "A_c (home TFP)"),  ("A_d", "A_d (home TFP)"),
    ("N", "N (population)"),
    ("D_M_m", "D_M^m"), ("D_M_f", "D_M^f"),
    ("D_c_m", "D_c^m"), ("D_c_f", "D_c^f"),
    ("D_d_m", "D_d^m"), ("D_d_f", "D_d^f"),
]


def make_county_map(map_source: ColumnDataSource,
                    cmap_source: ColumnDataSource,
                    title: str = "Click a county") -> figure:
    """Centroid-circle Kenya map. Tap selects a county."""
    p = figure(
        title=title, width=380, height=420,
        x_range=(33.2, 42.6), y_range=(-5.2, 5.2),
        tools="tap,hover,reset",
        active_tap="auto",
        toolbar_location="right",
    )
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.background_fill_color = "#f0f4f8"
    # Faint outline of Kenya
    p.line([33.7, 42.2, 42.2, 33.7, 33.7],
           [-4.8, -4.8, 4.5, 4.5, -4.8],
           color="#888", line_dash="dotted", line_width=1)

    cmap = LinearColorMapper(palette=RdBu11, low=-1, high=1)
    r = p.scatter(
        x="lon", y="lat", source=map_source, size=14,
        fill_color={"field": "value", "transform": cmap},
        line_color="black", line_width=0.5,
    )
    cb = ColorBar(color_mapper=cmap, location=(0, 0), width=10,
                  ticker=BasicTicker(desired_num_ticks=5))
    p.add_layout(cb, "right")

    hover = p.select_one(HoverTool)
    hover.tooltips = [("County", "@name"), ("Value", "@value{0.000}")]

    # cmap_source carries (low, high) so callbacks can update the colour scale
    return p


def participation_rate(LM_f: np.ndarray, h_grid: np.ndarray,
                       w_ell: float, wage_gap: float, sigma_u: float,
                       u_bar: float, phi: float) -> float:
    V_P = phi * w_ell * wage_gap * np.asarray(h_grid)
    P = 1.0 / (1.0 + np.exp((u_bar - V_P) / max(sigma_u, 1e-9)))
    return float(np.mean(P))


# ═══════════════════════════════════════════════════════════════════════
# ScenarioUI — one column (A or B)
# ═══════════════════════════════════════════════════════════════════════

class ScenarioUI:
    def __init__(self, label: str):
        self.label = label
        self.selected_county = 47  # Nairobi by default
        # Working copies of parameters (user can edit)
        self.global_p = dict(BASELINE["global"])
        self.counties_p = {k: dict(v) for k, v in BASELINE["counties"].items()}
        # Cached spatial results for the Spatial / Counterfactual tabs
        self.spatial_results: Optional[Dict[int, dict]] = None
        self.cf_results: Optional[Dict[str, Dict[int, dict]]] = None

        # ── Build widgets ──
        self.global_inputs: Dict[str, NumericInput] = {}
        for k, lab in GLOBAL_FIELDS:
            self.global_inputs[k] = NumericInput(
                title=lab, value=float(self.global_p[k]), mode="float", width=130)

        self.county_inputs: Dict[str, NumericInput] = {}
        cp = self.counties_p[self.selected_county]
        for k, lab in COUNTY_FIELDS:
            self.county_inputs[k] = NumericInput(
                title=lab, value=float(cp[k]), mode="float", width=130)

        self.county_label = Div(
            text=f"<b>Selected county:</b> {COUNTY[self.selected_county]['name']} "
                 f"(#{self.selected_county})", width=380)

        self.solve_btn = Button(label="Solve selected county",
                                button_type="primary", width=200)
        self.reset_btn = Button(label="Reset to baseline",
                                button_type="default", width=180)

        self.solve_all_btn = Button(label="Solve all 47 counties",
                                    button_type="success", width=200)
        self.cf_btn_wage = Button(label="CF: wage_gap = 1",
                                  button_type="warning", width=180)
        self.cf_btn_pc = Button(label="CF: p_c × 0.7",
                                button_type="warning", width=180)
        self.cf_btn_pd = Button(label="CF: p_d × 0.7",
                                button_type="warning", width=180)

        self.status = Div(text="<i>Ready.</i>", width=380)

        # Map data
        self.map_source = ColumnDataSource(self._initial_map_data())
        self.cmap_holder = ColumnDataSource(data=dict(low=[-1], high=[1]))
        self.map_metric = Select(
            title="Colour map by:",
            value="wage",
            options=["wage", "Pf (baseline)", "ΔGDP (last CF)"],
            width=200,
        )
        self.map = make_county_map(
            self.map_source, self.cmap_holder,
            title=f"{label} — click a county to select")

        # Per-h plot data sources
        self.src_hours = ColumnDataSource(
            dict(h=[], LM_m=[], LM_f=[], Lc_m=[], Lc_f=[], Ld_m=[], Ld_f=[]))
        self.src_gap = ColumnDataSource(
            dict(h=[], gap_M=[], gap_c=[], gap_d=[]))
        self.src_shares = ColumnDataSource(
            dict(h=[], th_x=[], th_c=[], th_d=[]))
        self.src_homemkt = ColumnDataSource(
            dict(h=[], ScH=[], ScM=[], SdH=[], SdM=[]))

        # Spatial summary plot data
        self.src_spatial = ColumnDataSource(
            dict(county=[], name=[], Vstar=[], xi=[], wage=[],
                 Pf=[], ratio=[], LM_m=[], LM_f=[], GDP=[]))

        # Counterfactual results
        self.cf_table_source = ColumnDataSource(
            dict(scenario=[], dGDP=[], dPf=[], dratio=[]))
        self.cf_map_sources = {
            (cf, ind): ColumnDataSource(self._initial_map_data())
            for cf in ("wage", "pc", "pd")
            for ind in ("dGDP", "dPf", "dratio")
        }

        # Build plots and tabs
        self.plots = self._make_plots()
        self.tabs = self._make_tabs()

        # Wire callbacks
        self._wire_callbacks()

        # Initial solve to populate the per-h tab
        self._solve_selected_and_update()

    # ── Map data ──
    def _initial_map_data(self) -> dict:
        ids = sorted(COUNTY.keys())
        return dict(
            id=ids,
            name=[COUNTY[c]["name"] for c in ids],
            lat=[COUNTY[c]["lat"] for c in ids],
            lon=[COUNTY[c]["lon"] for c in ids],
            value=[float(self.counties_p[c]["w_ell"] * E_SCALE) for c in ids],
        )

    def _refresh_map(self) -> None:
        ids = sorted(COUNTY.keys())
        metric = self.map_metric.value
        if metric == "wage":
            vals = [float(self.counties_p[c]["w_ell"] * E_SCALE) for c in ids]
            cmap = "RdBu11"; lo = min(vals); hi = max(vals)
        elif metric == "Pf (baseline)" and self.spatial_results is not None:
            s = summarise_spatial(self.spatial_results)
            vals = list(s["Pf"])
            lo = min(vals); hi = max(vals)
        elif metric == "ΔGDP (last CF)" and self.cf_results is not None:
            # Use the most recently run CF
            last_cf = list(self.cf_results.keys())[-1] if self.cf_results else None
            if last_cf is None:
                vals = [0.0] * len(ids); lo = -1; hi = 1
            else:
                base = self.spatial_results
                cf = self.cf_results[last_cf]
                vals = []
                for c in ids:
                    bE = np.nanmean(base[c]["E"]) * base[c]["N_county"]
                    cE = np.nanmean(cf[c]["E"]) * cf[c]["N_county"]
                    vals.append(100.0 * (cE / bE - 1.0) if bE > 0 else 0.0)
                m = max(abs(min(vals)), abs(max(vals))) or 1.0
                lo, hi = -m, m
        else:
            vals = [float(self.counties_p[c]["w_ell"] * E_SCALE) for c in ids]
            lo, hi = min(vals), max(vals)

        self.map_source.data = dict(
            id=ids,
            name=[COUNTY[c]["name"] for c in ids],
            lat=[COUNTY[c]["lat"] for c in ids],
            lon=[COUNTY[c]["lon"] for c in ids],
            value=vals,
        )
        # Update color mapper range
        for r in self.map.renderers:
            try:
                r.glyph.fill_color["transform"].low = lo
                r.glyph.fill_color["transform"].high = hi
            except Exception:
                pass

    # ── Plots ──
    def _make_plots(self) -> Dict[str, figure]:
        H, W = 240, 380

        p_h = figure(height=H, width=W, title="Hours by gender × activity")
        p_h.line("h", "LM_m", source=self.src_hours, color="#1f77b4",
                 line_width=2, legend_label="L^M man")
        p_h.line("h", "LM_f", source=self.src_hours, color="#1f77b4",
                 line_width=2, line_dash="dashed", legend_label="L^M woman")
        p_h.line("h", "Lc_m", source=self.src_hours, color="#d62728",
                 line_width=2, legend_label="L^c man")
        p_h.line("h", "Lc_f", source=self.src_hours, color="#d62728",
                 line_width=2, line_dash="dashed", legend_label="L^c woman")
        p_h.line("h", "Ld_m", source=self.src_hours, color="#2ca02c",
                 line_width=2, legend_label="L^d man")
        p_h.line("h", "Ld_f", source=self.src_hours, color="#2ca02c",
                 line_width=2, line_dash="dashed", legend_label="L^d woman")
        p_h.xaxis.axis_label = "h"; p_h.yaxis.axis_label = "h/wk"
        p_h.legend.location = "top_left"; p_h.legend.label_text_font_size = "8pt"

        p_g = figure(height=H, width=W, title="Gender gaps (woman − man)")
        p_g.line("h", "gap_M", source=self.src_gap, color="#1f77b4",
                 line_width=2, legend_label="ΔL^M")
        p_g.line("h", "gap_c", source=self.src_gap, color="#d62728",
                 line_width=2, legend_label="ΔL^c")
        p_g.line("h", "gap_d", source=self.src_gap, color="#2ca02c",
                 line_width=2, legend_label="ΔL^d")
        p_g.xaxis.axis_label = "h"; p_g.yaxis.axis_label = "Δh/wk"
        p_g.legend.location = "center_right"; p_g.legend.label_text_font_size = "8pt"

        p_s = figure(height=H, width=W, title="PIGL expenditure shares")
        p_s.line("h", "th_x", source=self.src_shares, color="#9467bd",
                 line_width=2, legend_label="θ_x")
        p_s.line("h", "th_c", source=self.src_shares, color="#d62728",
                 line_width=2, legend_label="θ_c")
        p_s.line("h", "th_d", source=self.src_shares, color="#2ca02c",
                 line_width=2, legend_label="θ_d")
        p_s.xaxis.axis_label = "h"; p_s.yaxis.axis_label = "share"
        p_s.legend.location = "center_right"; p_s.legend.label_text_font_size = "8pt"

        p_hm = figure(height=H, width=W, title="Home vs market service shares")
        p_hm.line("h", "ScH", source=self.src_homemkt, color="#d62728",
                  line_width=2, legend_label="Care home")
        p_hm.line("h", "ScM", source=self.src_homemkt, color="#d62728",
                  line_width=2, line_dash="dashed", legend_label="Care market")
        p_hm.line("h", "SdH", source=self.src_homemkt, color="#2ca02c",
                  line_width=2, legend_label="Domestic home")
        p_hm.line("h", "SdM", source=self.src_homemkt, color="#2ca02c",
                  line_width=2, line_dash="dashed", legend_label="Domestic market")
        p_hm.xaxis.axis_label = "h"; p_hm.yaxis.axis_label = "share of total"
        p_hm.legend.location = "center_right"; p_hm.legend.label_text_font_size = "8pt"

        # Spatial summary plots
        p_vs = figure(height=H, width=W, title="V* vs wage")
        p_vs.scatter("wage", "Vstar", source=self.src_spatial, size=8,
                     fill_color="#1f77b4", line_color="black", line_width=0.5)
        p_vs.xaxis.axis_label = "wage (KSh/hr)"; p_vs.yaxis.axis_label = "V*"
        h2 = HoverTool(renderers=[p_vs.renderers[0]],
                       tooltips=[("County", "@name"), ("Wage", "@wage{0}"),
                                 ("V*", "@Vstar{0.000}")])
        p_vs.add_tools(h2)

        p_xi = figure(height=H, width=W, title="Amenity ξ (sorted)")
        # We render a vbar after sorting in callback
        self.src_spatial_sorted = ColumnDataSource(
            dict(rank=[], name=[], xi=[], color=[]))
        p_xi.vbar(x="rank", top="xi", width=0.8, source=self.src_spatial_sorted,
                  fill_color="color", line_color="black", line_width=0.3)
        p_xi.xaxis.axis_label = "county rank (highest → lowest)"
        p_xi.yaxis.axis_label = "ξ"
        h3 = HoverTool(renderers=[p_xi.renderers[0]],
                       tooltips=[("County", "@name"), ("ξ", "@xi{0.000}")])
        p_xi.add_tools(h3)

        p_gap_county = figure(height=H, width=W,
                              title="Gender market-hours ratio L^M_f / L^M_m")
        p_gap_county.scatter("wage", "ratio", source=self.src_spatial, size=8,
                             fill_color="#d62728", line_color="black",
                             line_width=0.5)
        p_gap_county.xaxis.axis_label = "wage (KSh/hr)"
        p_gap_county.yaxis.axis_label = "L^M_f / L^M_m"
        h4 = HoverTool(renderers=[p_gap_county.renderers[0]],
                       tooltips=[("County", "@name"),
                                 ("Ratio", "@ratio{0.000}")])
        p_gap_county.add_tools(h4)

        return {
            "hours": p_h, "gap": p_g, "shares": p_s, "homemkt": p_hm,
            "Vstar": p_vs, "xi": p_xi, "gap_county": p_gap_county,
        }

    def _make_tabs(self) -> Tabs:
        per_h = column(
            row(self.plots["hours"], self.plots["gap"]),
            row(self.plots["shares"], self.plots["homemkt"]),
        )
        spatial = column(
            self.solve_all_btn,
            row(self.plots["Vstar"], self.plots["xi"]),
            row(self.plots["gap_county"]),
        )
        # Counterfactual tab — table + nine maps (3 maps × 3 scenarios)
        cf_table = DataTable(
            source=self.cf_table_source,
            columns=[
                TableColumn(field="scenario", title="Scenario"),
                TableColumn(field="dGDP", title="ΔGDP (%)",
                            formatter=NumberFormatter(format="+0.00")),
                TableColumn(field="dPf", title="ΔP_f (pp)",
                            formatter=NumberFormatter(format="+0.000")),
                TableColumn(field="dratio", title="Δratio (pp)",
                            formatter=NumberFormatter(format="+0.000")),
            ],
            width=720, height=140, index_position=None,
        )
        cf_buttons = row(self.cf_btn_wage, self.cf_btn_pc, self.cf_btn_pd)
        cf_maps = self._make_cf_maps_grid()
        cf_panel = column(
            Div(text="<b>Counterfactual results.</b> Run any of the three "
                     "scenarios. The table shows national means; the maps show "
                     "per-county changes. Requires the spatial baseline to be "
                     "solved first (Spatial tab).",
                width=720),
            cf_buttons, cf_table, cf_maps,
        )

        return Tabs(tabs=[
            TabPanel(child=per_h, title="Per-h plots (selected county)"),
            TabPanel(child=spatial, title="Spatial summary"),
            TabPanel(child=cf_panel, title="Counterfactuals"),
        ])

    def _make_cf_maps_grid(self):
        scenarios = [("wage", "Wage gap = 1"),
                     ("pc", "p_c × 0.7"),
                     ("pd", "p_d × 0.7")]
        indicators = [("dGDP", "ΔGDP (%)", "RdBu11"),
                      ("dPf", "ΔP_f (pp)", "PiYG11"),
                      ("dratio", "Δratio (pp)", "PuOr11")]
        rows_ = []
        for scen_key, scen_lab in scenarios:
            map_row = []
            for ind_key, ind_lab, palette in indicators:
                src = self.cf_map_sources[(scen_key, ind_key)]
                p = figure(width=240, height=240,
                           title=f"{scen_lab} — {ind_lab}",
                           x_range=(33.2, 42.6), y_range=(-5.2, 5.2),
                           tools="hover", toolbar_location=None)
                p.xaxis.visible = False; p.yaxis.visible = False
                p.xgrid.visible = False; p.ygrid.visible = False
                p.background_fill_color = "#f0f4f8"
                p.line([33.7, 42.2, 42.2, 33.7, 33.7],
                       [-4.8, -4.8, 4.5, 4.5, -4.8],
                       color="#888", line_dash="dotted", line_width=1)
                pal = {"RdBu11": RdBu11, "PiYG11": PiYG11,
                       "PuOr11": PuOr11}[palette]
                cmap = LinearColorMapper(palette=pal, low=-1, high=1)
                p.scatter(x="lon", y="lat", source=src, size=8,
                          fill_color={"field": "value", "transform": cmap},
                          line_color="black", line_width=0.4)
                hover = p.select_one(HoverTool)
                hover.tooltips = [("County", "@name"),
                                  ("Value", "@value{+0.00}")]
                map_row.append(p)
            rows_.append(row(*map_row))
        return column(*rows_)

    # ── Callbacks ──
    def _wire_callbacks(self) -> None:
        # County selection via tap on the map
        def on_tap(event):
            if not self.map_source.selected.indices:
                return
            idx = self.map_source.selected.indices[0]
            cnum = self.map_source.data["id"][idx]
            self._select_county(int(cnum))

        self.map_source.selected.on_change("indices", lambda attr, old, new:
                                            on_tap(None))

        self.solve_btn.on_click(self._solve_selected_and_update)
        self.reset_btn.on_click(self._reset)
        self.solve_all_btn.on_click(self._solve_all_and_update)
        self.cf_btn_wage.on_click(lambda: self._run_cf("wage"))
        self.cf_btn_pc.on_click(lambda: self._run_cf("pc"))
        self.cf_btn_pd.on_click(lambda: self._run_cf("pd"))
        self.map_metric.on_change("value", lambda attr, old, new:
                                   self._refresh_map())

    def _select_county(self, cnum: int) -> None:
        # Save the current county_inputs back before switching
        self._sync_inputs_to_county(self.selected_county)
        self.selected_county = cnum
        cp = self.counties_p[cnum]
        for k, _ in COUNTY_FIELDS:
            self.county_inputs[k].value = float(cp[k])
        self.county_label.text = (
            f"<b>Selected county:</b> {COUNTY[cnum]['name']} (#{cnum})")
        self._solve_selected_and_update()

    def _sync_inputs_to_state(self) -> None:
        """Read all input widgets back into self.global_p and self.counties_p."""
        for k, _ in GLOBAL_FIELDS:
            v = self.global_inputs[k].value
            if v is not None:
                self.global_p[k] = float(v)
        self._sync_inputs_to_county(self.selected_county)

    def _sync_inputs_to_county(self, cnum: int) -> None:
        for k, _ in COUNTY_FIELDS:
            v = self.county_inputs[k].value
            if v is not None:
                self.counties_p[cnum][k] = float(v)

    def _solve_selected_and_update(self) -> None:
        self._sync_inputs_to_state()
        self.status.text = "<i>Solving selected county…</i>"
        try:
            r = solve_county(self.global_p,
                             self.counties_p[self.selected_county])
            self.src_hours.data = dict(
                h=H_GRID, LM_m=r["LM_m"], LM_f=r["LM_f"],
                Lc_m=r["Lc_m"], Lc_f=r["Lc_f"],
                Ld_m=r["Ld_m"], Ld_f=r["Ld_f"])
            self.src_gap.data = dict(
                h=H_GRID,
                gap_M=r["LM_f"] - r["LM_m"],
                gap_c=r["Lc_f"] - r["Lc_m"],
                gap_d=r["Ld_f"] - r["Ld_m"])
            self.src_shares.data = dict(
                h=H_GRID, th_x=r["th_x"], th_c=r["th_c"], th_d=r["th_d"])
            self.src_homemkt.data = dict(
                h=H_GRID,
                ScH=r["ScH_share"], ScM=r["ScM_share"],
                SdH=r["SdH_share"], SdM=r["SdM_share"])
            self.status.text = (
                f"<b>{COUNTY[self.selected_county]['name']}</b> solved · "
                f"convergence {r['n_conv']}/4 · "
                f"L^M ratio (m/f) = {np.nanmean(r['LM_m'])/max(np.nanmean(r['LM_f']),1e-9):.2f}")
        except Exception as e:
            self.status.text = f"<b style='color:#b00'>Error:</b> {e!r}"

    def _reset(self) -> None:
        self.global_p = dict(BASELINE["global"])
        self.counties_p = {k: dict(v) for k, v in BASELINE["counties"].items()}
        for k, _ in GLOBAL_FIELDS:
            self.global_inputs[k].value = float(self.global_p[k])
        cp = self.counties_p[self.selected_county]
        for k, _ in COUNTY_FIELDS:
            self.county_inputs[k].value = float(cp[k])
        self.spatial_results = None
        self.cf_results = None
        self._refresh_map()
        self._solve_selected_and_update()
        self.status.text = "<i>Reset to baseline.</i>"

    def _solve_all_and_update(self) -> None:
        self._sync_inputs_to_state()
        self.status.text = "<i>Solving all 47 counties…</i>"
        try:
            self.spatial_results = solve_all_counties(self.global_p,
                                                      self.counties_p)
            s = summarise_spatial(self.spatial_results)
            ids = s["counties"]
            self.src_spatial.data = dict(
                county=ids,
                name=[COUNTY[c]["name"] for c in ids],
                Vstar=list(s["Vstar"]), xi=list(s["xi"]),
                wage=list(s["wages"]), Pf=list(s["Pf"]),
                ratio=list(s["ratio"]), LM_m=list(s["LM_m"]),
                LM_f=list(s["LM_f"]), GDP=list(s["GDP"]),
            )
            order = np.argsort(s["xi"])[::-1]
            cols = ["#1a9850" if x > 0 else "#d73027" for x in s["xi"][order]]
            self.src_spatial_sorted.data = dict(
                rank=list(range(len(order))),
                name=[COUNTY[ids[i]]["name"] for i in order],
                xi=list(s["xi"][order]),
                color=cols,
            )
            self._refresh_map()
            self.status.text = (
                f"All 47 counties solved. Ū = {s['Ubar']:.3f}, "
                f"V* range [{s['Vstar'].min():.3f}, {s['Vstar'].max():.3f}]")
        except Exception as e:
            self.status.text = f"<b style='color:#b00'>Error:</b> {e!r}"

    def _run_cf(self, kind: str) -> None:
        if self.spatial_results is None:
            self.status.text = ("<b style='color:#b00'>Run \"Solve all 47\" "
                                "first (Spatial tab).</b>")
            return
        self._sync_inputs_to_state()
        # Build counterfactual params
        cf_global = dict(self.global_p)
        cf_counties = {k: dict(v) for k, v in self.counties_p.items()}
        if kind == "wage":
            cf_global["wage_gap"] = 1.0
            scen_label = "Wage gap = 1"
        elif kind == "pc":
            for c in cf_counties:
                cf_counties[c]["pc"] *= 0.7
            scen_label = "p_c × 0.7"
        elif kind == "pd":
            for c in cf_counties:
                cf_counties[c]["pd"] *= 0.7
            scen_label = "p_d × 0.7"
        else:
            return

        self.status.text = f"<i>Running counterfactual: {scen_label}…</i>"
        try:
            cf_res = solve_all_counties(cf_global, cf_counties)
            if self.cf_results is None:
                self.cf_results = {}
            self.cf_results[kind] = cf_res
            # Compute deltas and update table + maps
            self._update_cf_outputs()
            self.status.text = f"<b>Counterfactual {scen_label} done.</b>"
        except Exception as e:
            self.status.text = f"<b style='color:#b00'>CF error:</b> {e!r}"

    def _update_cf_outputs(self) -> None:
        base = self.spatial_results
        ids = sorted(COUNTY.keys())
        rows_table = []
        scen_labels = {"wage": "Wage gap = 1",
                       "pc": "p_c × 0.7", "pd": "p_d × 0.7"}
        for kind, lab in scen_labels.items():
            if kind not in self.cf_results:
                continue
            cf = self.cf_results[kind]
            # Per-county deltas
            dGDP = []; dPf = []; dratio = []
            for c in ids:
                bE = np.nanmean(base[c]["E"]) * base[c]["N_county"]
                cE = np.nanmean(cf[c]["E"]) * cf[c]["N_county"]
                dGDP.append(100.0 * (cE / bE - 1.0) if bE > 0 else 0.0)
                bPf = float(np.nanmean(base[c]["Pf"]))
                cPf = float(np.nanmean(cf[c]["Pf"]))
                dPf.append(100.0 * (cPf - bPf))
                bLm = np.nanmean(base[c]["LM_m"]); bLf = np.nanmean(base[c]["LM_f"])
                cLm = np.nanmean(cf[c]["LM_m"]);   cLf = np.nanmean(cf[c]["LM_f"])
                br = bLf / max(bLm, 1e-9); cr = cLf / max(cLm, 1e-9)
                dratio.append(100.0 * (cr - br))
            # Update map sources
            for ind_key, vals in [("dGDP", dGDP), ("dPf", dPf),
                                  ("dratio", dratio)]:
                src = self.cf_map_sources[(kind, ind_key)]
                src.data = dict(
                    id=ids,
                    name=[COUNTY[c]["name"] for c in ids],
                    lat=[COUNTY[c]["lat"] for c in ids],
                    lon=[COUNTY[c]["lon"] for c in ids],
                    value=vals,
                )
                # Update color scale to symmetric around 0
                m = max(abs(min(vals)), abs(max(vals))) or 1.0
                # Find the color mapper for this map (first scatter renderer)
            rows_table.append(dict(
                scenario=lab,
                dGDP=float(np.mean(dGDP)),
                dPf=float(np.mean(dPf)),
                dratio=float(np.mean(dratio)),
            ))
        # Update table source
        if rows_table:
            self.cf_table_source.data = dict(
                scenario=[r["scenario"] for r in rows_table],
                dGDP=[r["dGDP"] for r in rows_table],
                dPf=[r["dPf"] for r in rows_table],
                dratio=[r["dratio"] for r in rows_table],
            )

    # ── Top-level layout ──
    def root_layout(self):
        # Two-column controls: globals on the left of the column,
        # county-specific on the right
        global_grid = column(
            Div(text="<details open><summary><b>Global parameters</b> "
                     "(click to collapse)</summary>", width=380),
            *[row(*[self.global_inputs[k] for k, _ in GLOBAL_FIELDS[i:i+2]])
              for i in range(0, len(GLOBAL_FIELDS), 2)],
            Div(text="</details>", width=380),
            width=380,
        )
        county_grid = column(
            self.county_label,
            Div(text="<details open><summary><b>County-specific parameters</b>"
                     "</summary>", width=380),
            *[row(*[self.county_inputs[k] for k, _ in COUNTY_FIELDS[i:i+2]])
              for i in range(0, len(COUNTY_FIELDS), 2)],
            Div(text="</details>", width=380),
            width=380,
        )
        controls = column(
            Div(text=f"<h2 style='margin:0'>{self.label}</h2>", width=400),
            self.map_metric, self.map,
            row(self.solve_btn, self.reset_btn),
            self.status,
            global_grid, county_grid,
            width=420,
        )
        return row(controls, self.tabs)


# ═══════════════════════════════════════════════════════════════════════
# Build the page
# ═══════════════════════════════════════════════════════════════════════

scenA = ScenarioUI("Scenario A")
scenB = ScenarioUI("Scenario B")

page = row(
    scenA.root_layout(),
    Spacer(width=24),
    scenB.root_layout(),
    sizing_mode="stretch_width",
)

curdoc().add_root(page)
curdoc().title = "Kenya Time-Use Model — Spatial Bokeh App"
