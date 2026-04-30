"""
classes.py — dataclasses for the 4-good Kenya time-use model.

Three classes:

  ModelParams   global structural parameters (frozen)
  Household     state container for one household at one (county, h) point
  County        county-level fundamentals (mutable; the spatial layer fills xi
                and V_star after the baseline solve)

Goods
  xf  food          (home-producible, market substitute)
  xn  non-food      (market only, numeraire: P_xn = 1)
  c   care services (home-producible, market substitute)
  d   domestic      (home-producible, market substitute)

Activities
  M   market work
  xf  home food preparation
  c   home care
  d   home domestic
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# ModelParams                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelParams:
    """Global, structural parameters shared across counties."""

    # PIGL preferences (4 goods)
    eps_engel: float
    beta_xf:   float
    beta_xn:   float
    beta_c:    float
    beta_d:    float
    kappa_xf:  float
    kappa_xn:  float
    kappa_c:   float
    kappa_d:   float

    # Home/market CES (3 home-producible goods)
    omega_xf: float
    omega_c:  float
    omega_d:  float
    eta_xf:   float
    eta_c:    float
    eta_d:    float

    # Disutility weights (8 = 4 activities × 2 genders).
    # Men's market and home-care are normalised to 1 by convention
    # (D_M_m = D_xf_m = D_c_m = 1).
    D_M_m:  float
    D_xf_m: float
    D_c_m:  float
    D_d_m:  float
    D_M_f:  float
    D_xf_f: float
    D_c_f:  float
    D_d_f:  float

    # Labour aggregator
    rho: float   # < 0  =>  complementarity across activities
    phi: float   # Frisch elasticity

    # Participation logit
    sigma_u_m: float
    sigma_u_f: float
    u_bar_m:   float
    u_bar_f:   float

    # National
    wage_gap: float   # y_f / y_m at h_m = h_f, level (≈ 0.83)
    p_xf:     float   # uniform national food price (1000-KSh/unit)
    sigma_mig: float = 1.0   # migration scale, hardcoded literature value


# --------------------------------------------------------------------------- #
# Household                                                                   #
# --------------------------------------------------------------------------- #

@dataclass
class Household:
    """
    One household at one (county, h) point.  The solver writes back into this
    object so the Bokeh app can read every intermediate quantity.

    `state = (s_m, s_f)` selects which of the four participation states is
    being solved.  Inactive market-work components are pinned to zero by the
    solver wrappers; this dataclass carries them all.
    """

    # ---- inputs ----------------------------------------------------------- #
    params: ModelParams
    y_m:  float    # effective male wage   = w_l · h_m
    y_f:  float    # effective female wage = wage_gap · w_l · h_f
    p_xf: float    # price of food (national, but stored locally for the solver)
    pc:   float
    pd:   float
    a:    float    # non-labour income (1000-KSh/period)

    # Home productivities (county-level; literature priors leave them at 1)
    A_xf: float = 1.0
    A_c:  float = 1.0
    A_d:  float = 1.0

    # Participation state being solved
    state: Tuple[int, int] = (1, 1)

    # ---- labour (filled by solver) --------------------------------------- #
    LM_m:  float = np.nan
    LM_f:  float = np.nan
    Lxf_m: float = np.nan
    Lxf_f: float = np.nan
    Lc_m:  float = np.nan
    Lc_f:  float = np.nan
    Ld_m:  float = np.nan
    Ld_f:  float = np.nan

    # Aggregates
    L_m: float = np.nan
    L_f: float = np.nan
    L:   float = np.nan

    # ---- prices, lambda, demands ----------------------------------------- #
    E:    float = np.nan
    lam:  float = np.nan
    PxfH: float = np.nan
    PcH:  float = np.nan
    PdH:  float = np.nan
    Pxf:  float = np.nan
    Pc:   float = np.nan
    Pd:   float = np.nan
    B:    float = np.nan

    th_xf: float = np.nan
    th_xn: float = np.nan
    th_c:  float = np.nan
    th_d:  float = np.nan

    xn:  float = np.nan
    Sxf: float = np.nan
    Sc:  float = np.nan
    Sd:  float = np.nan

    SxfH: float = np.nan
    ScH:  float = np.nan
    SdH:  float = np.nan
    SxfM: float = np.nan
    ScM:  float = np.nan
    SdM:  float = np.nan

    # Value at this single state (eq. 23)
    V_state: float = np.nan


# --------------------------------------------------------------------------- #
# County                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class County:
    """
    All county-l fundamentals needed for the spatial equilibrium.

    Productivities follow free-entry pricing (eq. 28-30):
        AM_xn = w_l                   (numeraire)
        AM_xf = w_l / p_xf            (uniform national p_xf)
        AM_c  = w_l / pc_l
        AM_d  = w_l / pd_l

    These are stored explicitly so the UI can show / edit them.
    """

    name:      str
    county_id: int
    lat:       float = np.nan
    lon:       float = np.nan

    # Wage and prices (all in 1000-KSh units)
    w_ell: float = 0.05
    p_xf:  float = 0.05    # uniform national, copied from ModelParams
    pc:    float = 0.05
    pd:    float = 0.05

    # Market-sector productivities (free-entry identities)
    AM_xn: float = 0.05
    AM_xf: float = 1.0
    AM_c:  float = 1.0
    AM_d:  float = 1.0

    # Home productivities (= 1 in baseline; literature)
    A_xf_home: float = 1.0
    A_c_home:  float = 1.0
    A_d_home:  float = 1.0

    # Population (TUS person-day weights summed; not census)
    N: float = 1.0

    # County-specific D weights (men's market, food-prep, care normalised to 1)
    D_M_m:  float = 1.0
    D_xf_m: float = 1.0
    D_c_m:  float = 1.0
    D_d_m:  float = 1.0
    D_M_f:  float = 1.0
    D_xf_f: float = 1.0
    D_c_f:  float = 1.0
    D_d_f:  float = 1.0

    # Calibrated equilibrium objects (filled by spatial solve)
    V_star: float = np.nan   # E[V] from four-state participation
    P_m:    float = np.nan
    P_f:    float = np.nan
    xi:     float = np.nan   # amenity, calibrated as Ubar - V_star

    # Market-clearing diagnostics (filled by spatial layer)
    LM_xf_total: float = np.nan
    LM_c_total:  float = np.nan
    LM_d_total:  float = np.nan
    LM_xn_total: float = np.nan
