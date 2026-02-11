#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 17:14:31 2026

@author: slepot
"""

# main.py
"""
Bokeh server app with two side-by-side scenarios for comparing model outcomes.

Each column contains:
- Input boxes for all parameters (defaults set to baseline values from your PDF)
- "Reset all" button (restore defaults)
- "Solve" button (run solver over h_grid and update all plots)

Run with:
    bokeh serve --show main.py

Assumptions:
- You have these files in the same folder (or in PYTHONPATH):
    classes.py  (ModelParams, Household with evaluate_from_labor)
    solver.py   (SolverState, solve_model)  [fixed-point solver version]

Notes:
- Participation/value functions are implemented as in make_figures.py:
    V = (E/B)^eps / eps  -  L^{1+1/phi}/(1+1/phi)
  Participation probability is a smoothed logit:
    P(h)=1/(1+exp(-(Vm-Vn)/sigma_part))
"""

# from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource, Div, Button, TextInput, Select,
    Tabs, TabPanel, Spacer, Legend
)

from bokeh.plotting import figure

from classes import ModelParams, Household
from solver import SolverState, solve_model


# ----------------------------
# Defaults (baseline from PDF)
# ----------------------------

DEFAULTS: Dict[str, object] = {
    # human capital grid
    "h_min": 0.2,
    "h_max": 5.0,
    "n_h": 10,
    "x_axis": "log",  # "log" or "level"

    # wage mapping y(h) = w0 * h^w_elast
    "w0": 20.0,
    "w_elast": 1.0,

    # exogenous prices, non-labor income
    "pc": 80.0,
    "pd": 40.0,
    "a": 200.0,

    # home productivity mapping A_i(h)=A_i0*h^A_elast
    "A_c0": 1.0,
    "A_d0": 1.0,
    "A_c_elast": 0.0,
    "A_d_elast": 0.0,

    # participation smoothing
    "sigma_part": 5.0,

    # solver controls (outer fixed point)
    "fp_tol": 1e-10,
    "fp_max_iter": 20000,
    "fp_damping": 0.2,

    # nonparticipant inner fixed point (Lc,Ld with LM fixed)
    "np_tol": 1e-10,
    "np_max_iter": 20000,
    "np_damping": 0.2,

    # ModelParams (baseline)
    "eps_engel": 0.30,
    "beta_x": 0.50,
    "beta_c": 0.25,
    "beta_d": 0.25,
    "kappa_x": 0.15,
    "kappa_c": -0.05,
    "kappa_d": -0.10,
    "omega_c": 0.75,
    "omega_d": 0.70,
    "eta_c": 2.5,
    "eta_d": 3.0,
    "D_M": 1.0,
    "D_c": 1.2,
    "D_d": 0.9,
    "rho": -0.50,
    "phi": 0.50,
}


# ----------------------------
# Parsing helpers
# ----------------------------

def _parse_float(s: str, default: float) -> float:
    try:
        x = float(s)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _parse_int(s: str, default: int) -> int:
    try:
        x = int(float(s))
        return int(x)
    except Exception:
        return int(default)


def _safe_logistic(z: float) -> float:
    # stable logistic
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ----------------------------
# Value + participation
# ----------------------------

def value_from_household(hh: Household) -> float:
    eps = float(hh.params.eps_engel)
    phi = float(hh.params.phi)
    E = float(hh.E)
    B = float(hh.B)
    L = float(hh.L)

    V_goods = (E / B) ** eps / eps
    V_time = L ** (1.0 + 1.0 / phi) / (1.0 + 1.0 / phi)
    return float(V_goods - V_time)


def participation_prob(Vm: float, Vn: float, sigma_part: float) -> float:
    z = (Vm - Vn) / max(1e-12, float(sigma_part))
    return float(_safe_logistic(z))


# ----------------------------
# Nonparticipant solve (LM fixed ~ 0)
# ----------------------------

def solve_nonparticipant(hh: Household, np_tol: float, np_max_iter: int, np_damping: float,
                         Lc0: float = 0.1, Ld0: float = 0.1) -> Tuple[Household, bool]:
    LM_fixed = 1e-16  # keep strictly positive
    Lc = max(1e-14, float(Lc0))
    Ld = max(1e-14, float(Ld0))
    omega = float(np_damping)

    prev_step = np.inf

    for _ in range(int(np_max_iter)):
        _ = hh.evaluate_from_labor(LM_fixed, Lc, Ld)

        Lc_new = float(hh.ScH) / float(hh.A_c)
        Ld_new = float(hh.SdH) / float(hh.A_d)

        Lc_new = max(1e-14, Lc_new)
        Ld_new = max(1e-14, Ld_new)

        Lc_next = (1.0 - omega) * Lc + omega * Lc_new
        Ld_next = (1.0 - omega) * Ld + omega * Ld_new

        step = float(np.linalg.norm([Lc_next - Lc, Ld_next - Ld]))
        if step < np_tol and prev_step < np_tol:
            _ = hh.evaluate_from_labor(LM_fixed, Lc_next, Ld_next)
            return hh, True

        prev_step = step
        Lc, Ld = Lc_next, Ld_next

    _ = hh.evaluate_from_labor(LM_fixed, Lc, Ld)
    return hh, False


# ----------------------------
# Simulation over h_grid
# ----------------------------

@dataclass
class SimConfig:
    h_min: float
    h_max: float
    n_h: int
    x_axis: str
    w0: float
    w_elast: float
    pc: float
    pd: float
    a: float
    A_c0: float
    A_d0: float
    A_c_elast: float
    A_d_elast: float
    sigma_part: float
    fp_tol: float
    fp_max_iter: int
    fp_damping: float
    np_tol: float
    np_max_iter: int
    np_damping: float


def run_simulation(mp: ModelParams, cfg: SimConfig) -> Dict[str, np.ndarray]:
    h_grid = np.linspace(cfg.h_min, cfg.h_max, int(cfg.n_h))
    x = np.log(h_grid) if cfg.x_axis == "log" else h_grid

    # arrays
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

    # warm starts
    L_guess = (0.2, 0.1, 0.1)
    Lc_guess_np, Ld_guess_np = 0.1, 0.1

    state = SolverState(
        verbose=False,
        max_iter=int(cfg.fp_max_iter),
        tol=float(cfg.fp_tol),
        damping=float(cfg.fp_damping),
        adapt_damping=True,
    )

    for i, h in enumerate(h_grid):
        y = float(cfg.w0 * (h ** cfg.w_elast))
        A_c = float(cfg.A_c0 * (h ** cfg.A_c_elast))
        A_d = float(cfg.A_d0 * (h ** cfg.A_d_elast))

        # participant
        hh = Household(
            params=mp,
            y=y,
            pc=float(cfg.pc),
            pd=float(cfg.pd),
            a=float(cfg.a),
            A_c=A_c,
            A_d=A_d,
        )
        hh, state = solve_model(mp, hh, state, L0=L_guess)
        LM[i], Lc[i], Ld[i] = state.LM, state.Lc, state.Ld
        L_guess = (max(1e-14, LM[i]), max(1e-14, Lc[i]), max(1e-14, Ld[i]))

        Vm[i] = value_from_household(hh)

        # nonparticipant
        hh_np = Household(
            params=mp,
            y=y,
            pc=float(cfg.pc),
            pd=float(cfg.pd),
            a=float(cfg.a),
            A_c=A_c,
            A_d=A_d,
        )
        hh_np, _ = solve_nonparticipant(
            hh_np,
            np_tol=float(cfg.np_tol),
            np_max_iter=int(cfg.np_max_iter),
            np_damping=float(cfg.np_damping),
            Lc0=Lc_guess_np,
            Ld0=Ld_guess_np,
        )
        Vn[i] = value_from_household(hh_np)
        Lc_guess_np, Ld_guess_np = float(hh_np.Lc), float(hh_np.Ld)

        P[i] = participation_prob(Vm[i], Vn[i], float(cfg.sigma_part))

        E[i] = float(hh.E)
        thx[i], thc[i], thd[i] = float(hh.th_x), float(hh.th_c), float(hh.th_d)

        Sc_tot = float(hh.Sc)
        Sd_tot = float(hh.Sd)
        ScH_share[i] = float(hh.ScH) / Sc_tot if Sc_tot > 0 else np.nan
        ScM_share[i] = float(hh.ScM) / Sc_tot if Sc_tot > 0 else np.nan
        SdH_share[i] = float(hh.SdH) / Sd_tot if Sd_tot > 0 else np.nan
        SdM_share[i] = float(hh.SdM) / Sd_tot if Sd_tot > 0 else np.nan

    # policy: pc reduced by 20%
    pc_policy = 0.8 * float(cfg.pc)
    P_policy = np.full(cfg.n_h, np.nan)

    # reset warm starts
    L_guess = (0.2, 0.1, 0.1)
    Lc_guess_np, Ld_guess_np = 0.1, 0.1

    for i, h in enumerate(h_grid):
        y = float(cfg.w0 * (h ** cfg.w_elast))
        A_c = float(cfg.A_c0 * (h ** cfg.A_c_elast))
        A_d = float(cfg.A_d0 * (h ** cfg.A_d_elast))

        hh = Household(params=mp, y=y, pc=pc_policy, pd=float(cfg.pd), a=float(cfg.a), A_c=A_c, A_d=A_d)
        hh, state = solve_model(mp, hh, state, L0=L_guess)
        L_guess = (max(1e-14, state.LM), max(1e-14, state.Lc), max(1e-14, state.Ld))
        Vm_pol = value_from_household(hh)

        hh_np = Household(params=mp, y=y, pc=pc_policy, pd=float(cfg.pd), a=float(cfg.a), A_c=A_c, A_d=A_d)
        hh_np, _ = solve_nonparticipant(
            hh_np,
            np_tol=float(cfg.np_tol),
            np_max_iter=int(cfg.np_max_iter),
            np_damping=float(cfg.np_damping),
            Lc0=Lc_guess_np,
            Ld0=Ld_guess_np,
        )
        Vn_pol = value_from_household(hh_np)
        Lc_guess_np, Ld_guess_np = float(hh_np.Lc), float(hh_np.Ld)

        P_policy[i] = participation_prob(Vm_pol, Vn_pol, float(cfg.sigma_part))

    dP = P_policy - P

    # sort order for shares vs E
    order = np.argsort(E)

    return {
        "h": h_grid,
        "x": x,
        "LM": LM, "Lc": Lc, "Ld": Ld,
        "P": P,
        "Vm": Vm, "Vn": Vn,
        "E": E, "order_E": order,
        "thx": thx, "thc": thc, "thd": thd,
        "ScH_share": ScH_share, "ScM_share": ScM_share,
        "SdH_share": SdH_share, "SdM_share": SdM_share,
        "dP": dP,
    }


# ----------------------------
# Bokeh UI: one scenario column
# ----------------------------

class ScenarioUI:
    def __init__(self, title: str, defaults: Dict[str, object]):
        self.title = title
        self.defaults = defaults

        # --- widgets ---
        self.widgets: Dict[str, object] = {}

        # selectors
        self.widgets["x_axis"] = Select(
            title="x-axis for h",
            value=str(defaults["x_axis"]),
            options=["log", "level"],
            width=220,
        )

        # numeric inputs: TextInput for robustness across Bokeh versions
        def add_float(name: str, label: str, width: int = 220):
            w = TextInput(title=label, value=str(self.defaults[name]), width=width)
            self.widgets[name] = w

        def add_int(name: str, label: str, width: int = 220):
            w = TextInput(title=label, value=str(self.defaults[name]), width=width)
            self.widgets[name] = w

        # grid + mappings
        add_float("h_min", "h_min")
        add_float("h_max", "h_max")
        add_int("n_h", "n_h")
        add_float("w0", "w0 (wage scale)")
        add_float("w_elast", "w_elast (wage elasticity - irrelevant)")

        # exogenous
        add_float("pc", "pc (care price)")
        add_float("pd", "pd (domestic price)")
        add_float("a", "a (non-labor income)")

        # home productivity
        add_float("A_c0", "A_c0")
        add_float("A_d0", "A_d0")
        add_float("A_c_elast", "A_c_elast - irrelevant")
        add_float("A_d_elast", "A_d_elast - irrelevant")

        # participation smoothing
        add_float("sigma_part", "sigma_part (logit scale) - prob irrelevant")

        # solver controls
        add_float("fp_tol", "fp_tol")
        add_int("fp_max_iter", "fp_max_iter")
        add_float("fp_damping", "fp_damping")
        add_float("np_tol", "np_tol")
        add_int("np_max_iter", "np_max_iter")
        add_float("np_damping", "np_damping")

        # model params
        add_float("eps_engel", "eps_engel")
        add_float("beta_x", "beta_x")
        add_float("beta_c", "beta_c")
        add_float("beta_d", "beta_d")
        add_float("kappa_x", "kappa_x")
        add_float("kappa_c", "kappa_c")
        add_float("kappa_d", "kappa_d")
        add_float("omega_c", "omega_c")
        add_float("omega_d", "omega_d")
        add_float("eta_c", "eta_c")
        add_float("eta_d", "eta_d")
        add_float("D_M", "D_M")
        add_float("D_c", "D_c")
        add_float("D_d", "D_d")
        add_float("rho", "rho")
        add_float("phi", "phi")

        self.reset_btn = Button(label="Reset all", button_type="warning", width=110)
        self.solve_btn = Button(label="Solve", button_type="primary", width=110)
        self.status_div = Div(text="", width=440)

        self.reset_btn.on_click(self.reset_all)
        self.solve_btn.on_click(self.solve_and_update)

        # --- plots ---
        self.sources = self._make_sources()
        self.plots = self._make_plots()

        # initial solve so plots aren’t empty
        self.solve_and_update()

        # --- layout ---
        controls = self._make_controls_layout()
        plot_tabs = self._make_plot_tabs()

        self.root = column(
            Div(text=f"<h2>{self.title}</h2>", width=460),
            controls,
            Spacer(height=10),
            plot_tabs,
            sizing_mode="stretch_width",
        )

    def _make_sources(self) -> Dict[str, ColumnDataSource]:
        # initialize empty sources with required columns
        return {
            "hours": ColumnDataSource(data=dict(x=[], LM=[], Lc=[], Ld=[])),
            "participation": ColumnDataSource(data=dict(x=[], P=[])),
            "shares": ColumnDataSource(data=dict(E=[], thx=[], thc=[], thd=[])),
            "care_share": ColumnDataSource(data=dict(x=[], home=[], market=[])),
            "dom_share": ColumnDataSource(data=dict(x=[], home=[], market=[])),
            "values": ColumnDataSource(data=dict(x=[], Vm=[], Vn=[])),
            "policy": ColumnDataSource(data=dict(x=[], dP=[])),
        }

    def _make_plots(self) -> Dict[str, object]:
        p_hours = figure(height=240, width=460, title="Hours vs h (participants)")
        r1 = p_hours.line("x", "LM", source=self.sources["hours"], color="#1f77b4")
        r2 = p_hours.line("x", "Lc", source=self.sources["hours"], color="#ff7f0e")
        r3 = p_hours.line("x", "Ld", source=self.sources["hours"], color="#2ca02c")
        
        legend = Legend(items=[
            ("L^M", [r1]),
            ("L^c", [r2]),
            ("L^d", [r3]),
        ])
        p_hours.add_layout(legend, "right")
        p_hours.xaxis.axis_label = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        p_hours.yaxis.axis_label = "Hours / time"

        p_part = figure(height=240, width=460, title="Participation probability P(h)")
        p_part.line("x", "P", source=self.sources["participation"])
        p_part.xaxis.axis_label = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        p_part.yaxis.axis_label = "P(h)"

        p_shares = figure(height=240, width=460, title="Budget shares vs total expenditure E")
        r1 = p_shares.line("E", "thx", source=self.sources["shares"], color="#1f77b4")
        r2 = p_shares.line("E", "thc", source=self.sources["shares"], color="#ff7f0e")
        r3 = p_shares.line("E", "thd", source=self.sources["shares"], color="#2ca02c")
        
        legend = Legend(items=[
            ("theta_x", [r1]),
            ("theta_c", [r2]),
            ("theta_d", [r3]),
        ])
        p_shares.add_layout(legend, "right")
        p_shares.xaxis.axis_label = "E"
        p_shares.yaxis.axis_label = "share"

        p_care = figure(height=240, width=460, title="Care: home vs market shares")
        r1 = p_care.line("x", "home", source=self.sources["care_share"], color="#1f77b4")
        r2 = p_care.line("x", "market", source=self.sources["care_share"], color="#d62728")
        
        legend = Legend(items=[
            ("ScH/Sc", [r1]),
            ("ScM/Sc", [r2]),
        ])
        p_care.add_layout(legend, "right")
        p_care.xaxis.axis_label = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        p_care.yaxis.axis_label = "share"

        p_dom = figure(height=240, width=460, title="Domestic: home vs market shares")
        r1 = p_dom.line("x", "home", source=self.sources["dom_share"], color="#1f77b4")
        r2 = p_dom.line("x", "market", source=self.sources["dom_share"], color="#d62728")
        
        legend = Legend(items=[
            ("SdH/Sd", [r1]),
            ("SdM/Sd", [r2]),
        ])
        p_dom.add_layout(legend, "right")
        p_dom.xaxis.axis_label = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        p_dom.yaxis.axis_label = "share"

        p_vals = figure(height=240, width=460, title="Value functions")
        r1 = p_vals.line("x", "Vm", source=self.sources["values"], color="#1f77b4")
        r2 = p_vals.line("x", "Vn", source=self.sources["values"], color="#ff7f0e")
        
        legend = Legend(items=[
            ("Vm(h)", [r1]),
            ("Vn(h)", [r2]),
        ])
        p_vals.add_layout(legend, "right")
        p_vals.xaxis.axis_label = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        p_vals.yaxis.axis_label = "value"

        p_pol = figure(height=240, width=460, title="Policy: ΔP(h) when pc reduced by 20%")
        p_pol.line("x", "dP", source=self.sources["policy"])
        p_pol.xaxis.axis_label = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        p_pol.yaxis.axis_label = "ΔP(h)"

        return {
            "hours": p_hours,
            "participation": p_part,
            "shares": p_shares,
            "care": p_care,
            "domestic": p_dom,
            "values": p_vals,
            "policy": p_pol,
        }

    def _make_plot_tabs(self) -> Tabs:
        tabs = [
            TabPanel(child=column(self.plots["hours"], self.plots["participation"]), title="Work & Participation"),
            TabPanel(child=column(self.plots["shares"]), title="Shares"),
            TabPanel(child=column(self.plots["care"], self.plots["domestic"]), title="Home vs Market"),
            TabPanel(child=column(self.plots["values"]), title="Values"),
            TabPanel(child=column(self.plots["policy"]), title="Policy"),
        ]
        return Tabs(tabs=tabs)

    def _make_controls_layout(self):
        # group widgets for readability
        def section(title: str):
            return Div(text=f"<b>{title}</b>", width=440)

        w = self.widgets

        grid_box = column(
            section("Grid"),
            row(w["h_min"], w["h_max"]),
            row(w["n_h"], w["x_axis"]),
        )

        exog_box = column(
            section("Exogenous"),
            row(w["w0"]),# w["w_elast"]),
            row(w["pc"], w["pd"]),
            row(w["a"]),
        )

        home_box = column(
            section("Home productivity"),
            row(w["A_c0"], w["A_d0"]),
            # row(w["A_c_elast"], w["A_d_elast"]),
        )

        num_box = column(
            section("Solver & participation"),
            row(w["sigma_part"]),
            row(w["fp_tol"], w["fp_max_iter"]),
            row(w["fp_damping"]),
            row(w["np_tol"], w["np_max_iter"]),
            row(w["np_damping"]),
        )

        model_box = column(
            section("Model parameters"),
            row(w["eps_engel"], w["rho"]),
            row(w["phi"], w["D_M"]),
            row(w["D_c"], w["D_d"]),
            row(w["beta_x"], w["beta_c"]),
            row(w["beta_d"], w["kappa_x"]),
            row(w["kappa_c"], w["kappa_d"]),
            row(w["omega_c"], w["omega_d"]),
            row(w["eta_c"], w["eta_d"]),
        )

        buttons = row(self.reset_btn, Spacer(width=10), self.solve_btn)

        return column(
            grid_box,
            exog_box,
            home_box,
            num_box,
            model_box,
            buttons,
            self.status_div,
        )

    def reset_all(self) -> None:
        for k, default in self.defaults.items():
            if k in self.widgets:
                if isinstance(self.widgets[k], Select):
                    self.widgets[k].value = str(default)
                else:
                    self.widgets[k].value = str(default)
        self.status_div.text = "<i>Reset to defaults.</i>"

        # update axis labels
        xlab = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        for key in ["hours", "participation", "care", "domestic", "values", "policy"]:
            self.plots[key].xaxis.axis_label = xlab

    def _read_config(self) -> Tuple[ModelParams, SimConfig]:
        # read model params
        mp = ModelParams(
            eps_engel=_parse_float(self.widgets["eps_engel"].value, float(self.defaults["eps_engel"])),
            beta_x=_parse_float(self.widgets["beta_x"].value, float(self.defaults["beta_x"])),
            beta_c=_parse_float(self.widgets["beta_c"].value, float(self.defaults["beta_c"])),
            beta_d=_parse_float(self.widgets["beta_d"].value, float(self.defaults["beta_d"])),
            kappa_x=_parse_float(self.widgets["kappa_x"].value, float(self.defaults["kappa_x"])),
            kappa_c=_parse_float(self.widgets["kappa_c"].value, float(self.defaults["kappa_c"])),
            kappa_d=_parse_float(self.widgets["kappa_d"].value, float(self.defaults["kappa_d"])),
            omega_c=_parse_float(self.widgets["omega_c"].value, float(self.defaults["omega_c"])),
            omega_d=_parse_float(self.widgets["omega_d"].value, float(self.defaults["omega_d"])),
            eta_c=_parse_float(self.widgets["eta_c"].value, float(self.defaults["eta_c"])),
            eta_d=_parse_float(self.widgets["eta_d"].value, float(self.defaults["eta_d"])),
            D_M=_parse_float(self.widgets["D_M"].value, float(self.defaults["D_M"])),
            D_c=_parse_float(self.widgets["D_c"].value, float(self.defaults["D_c"])),
            D_d=_parse_float(self.widgets["D_d"].value, float(self.defaults["D_d"])),
            rho=_parse_float(self.widgets["rho"].value, float(self.defaults["rho"])),
            phi=_parse_float(self.widgets["phi"].value, float(self.defaults["phi"])),
        )

        cfg = SimConfig(
            h_min=_parse_float(self.widgets["h_min"].value, float(self.defaults["h_min"])),
            h_max=_parse_float(self.widgets["h_max"].value, float(self.defaults["h_max"])),
            n_h=_parse_int(self.widgets["n_h"].value, int(self.defaults["n_h"])),
            x_axis=str(self.widgets["x_axis"].value),
            w0=_parse_float(self.widgets["w0"].value, float(self.defaults["w0"])),
            w_elast=_parse_float(self.widgets["w_elast"].value, float(self.defaults["w_elast"])),
            pc=_parse_float(self.widgets["pc"].value, float(self.defaults["pc"])),
            pd=_parse_float(self.widgets["pd"].value, float(self.defaults["pd"])),
            a=_parse_float(self.widgets["a"].value, float(self.defaults["a"])),
            A_c0=_parse_float(self.widgets["A_c0"].value, float(self.defaults["A_c0"])),
            A_d0=_parse_float(self.widgets["A_d0"].value, float(self.defaults["A_d0"])),
            A_c_elast=_parse_float(self.widgets["A_c_elast"].value, float(self.defaults["A_c_elast"])),
            A_d_elast=_parse_float(self.widgets["A_d_elast"].value, float(self.defaults["A_d_elast"])),
            sigma_part=_parse_float(self.widgets["sigma_part"].value, float(self.defaults["sigma_part"])),
            fp_tol=_parse_float(self.widgets["fp_tol"].value, float(self.defaults["fp_tol"])),
            fp_max_iter=_parse_int(self.widgets["fp_max_iter"].value, int(self.defaults["fp_max_iter"])),
            fp_damping=_parse_float(self.widgets["fp_damping"].value, float(self.defaults["fp_damping"])),
            np_tol=_parse_float(self.widgets["np_tol"].value, float(self.defaults["np_tol"])),
            np_max_iter=_parse_int(self.widgets["np_max_iter"].value, int(self.defaults["np_max_iter"])),
            np_damping=_parse_float(self.widgets["np_damping"].value, float(self.defaults["np_damping"])),
        )
        return mp, cfg

    def solve_and_update(self) -> None:
        self.solve_btn.disabled = True
        self.status_div.text = "<i>Solving…</i>"

        # update axis labels
        xlab = "log(h)" if self.widgets["x_axis"].value == "log" else "h"
        for key in ["hours", "participation", "care", "domestic", "values", "policy"]:
            self.plots[key].xaxis.axis_label = xlab

        try:
            mp, cfg = self._read_config()
            out = run_simulation(mp, cfg)

            # update sources
            self.sources["hours"].data = dict(x=out["x"], LM=out["LM"], Lc=out["Lc"], Ld=out["Ld"])
            self.sources["participation"].data = dict(x=out["x"], P=out["P"])

            order = out["order_E"]
            self.sources["shares"].data = dict(
                E=out["E"][order],
                thx=out["thx"][order],
                thc=out["thc"][order],
                thd=out["thd"][order],
            )

            self.sources["care_share"].data = dict(x=out["x"], home=out["ScH_share"], market=out["ScM_share"])
            self.sources["dom_share"].data = dict(x=out["x"], home=out["SdH_share"], market=out["SdM_share"])
            self.sources["values"].data = dict(x=out["x"], Vm=out["Vm"], Vn=out["Vn"])
            self.sources["policy"].data = dict(x=out["x"], dP=out["dP"])

            self.status_div.text = "<b>Done.</b>"

        except Exception as e:
            self.status_div.text = f"<b style='color:#b00'>Error:</b> {e!r}"

        finally:
            self.solve_btn.disabled = False


# ----------------------------
# Build page with two columns
# ----------------------------

left = ScenarioUI("Scenario A", DEFAULTS)
right = ScenarioUI("Scenario B", DEFAULTS)

page = row(
    left.root,
    Spacer(width=20),
    right.root,
    sizing_mode="stretch_width",
)

curdoc().add_root(page)
curdoc().title = "Household Time Use Model: Two-Scenario Comparison"
