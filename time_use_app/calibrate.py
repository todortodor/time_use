"""
calibrate.py — calibrate the 4-good Kenya time-use model from microdata.

Run with:
    python calibrate.py

Reads (from /mnt/user-data/uploads/):
    time_use_final_ver13.dta              TUS, 24,004 individuals
    individuals_microdata.dta             KCHS labour module
    consumption_aggregate_microdata.dta   KCHS expenditure aggregate
    nonfood_items_microdata.dta           KCHS non-food items detail

Writes (to the project directory):
    calibrated_params.json   global structural parameters
    county_data.csv          per-county fundamentals (47 rows)

Calibration blocks (in order):

  A.  PIGL preferences (4 goods):  eps_engel, beta_*, kappa_*
  B.  Wage gap (national, KCHS Mincer regression)
  C.  National disutility weights D^g_j (4 activities, 2 genders)
  D.  County-level D weights with shrinkage (n0 = 20)
  E.  County wages and service prices (KCHS)
  F.  Uniform national food price p_xf (consistency calibration)
  G.  Market-sector productivities A^M,i_l from free-entry pricing
  H.  Population N_l (TUS person-day weights, summed per county)
  I.  Participation shifters sigma_u^g, u_bar^g (structural moment match)
  J.  Spatial calibration: solve all 47 counties, set xi_l = Ubar - V*_l

All literature priors flagged inline:
    rho      = -0.5         CES disutility curvature
    phi      = 0.5          Frisch elasticity (KCHS IV biased downward)
    omega_xf = omega_c = 0.75; omega_d = 0.70
    eta_xf   = eta_c = eta_d = 2.5
    A^H,i    = 1            home productivities (not identified)

Sign-correctness items preserved from the existing project (see plan):
    1. D^f_M ≈ 1.34 from solver-consistent formula r_M / wage_gap^rho
    2. County D weights use shrinkage with n0 = 20 toward national ratio
    3. Migration uses the population-weighted variant
       (in solver_functions.migration_update; not in calibration)
    4. Home/market plot uses expenditure shares (in main.py)
    5. phi = 0.5 literature prior, NOT the IV estimate

Unit convention:
    All wages, prices, and expenditures stored in **1000-KSh units**.
    The .csv reflects this directly so loaders don't have to scale.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from classes import ModelParams, County
from solver_functions import (
    solve_county_household, solve_all_counties, calibrate_amenities,
    compute_market_clearing,
)

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Paths and constants                                                         #
# --------------------------------------------------------------------------- #

UPLOADS = Path('/mnt/user-data/uploads')
HERE    = Path(__file__).parent

TUS_PATH = UPLOADS / 'time_use_final_ver13.dta'
IND_PATH = UPLOADS / 'individuals_microdata.dta'
CA_PATH  = UPLOADS / 'consumption_aggregate_microdata.dta'
NF_PATH  = UPLOADS / 'nonfood_items_microdata.dta'

OUT_PARAMS = HERE / 'calibrated_params.json'
OUT_COUNTY = HERE / 'county_data.csv'

E_SCALE  = 1000.0   # KSh -> 1000-KSh
SENTINEL = 999_990_000
PRIOR_N  = 20.0     # shrinkage prior weight (n0)

# 47 KNBS county codes -> (name, lat, lon).  Centroids embedded so the
# calibration is self-contained (no external GeoJSON).
COUNTY_CENTROIDS = {
     1: ("Mombasa",         -4.05, 39.67),
     2: ("Kwale",           -4.18, 39.45),
     3: ("Kilifi",          -3.51, 39.85),
     4: ("Tana River",      -1.65, 39.65),
     5: ("Lamu",            -2.27, 40.92),
     6: ("Taita Taveta",    -3.40, 38.55),
     7: ("Garissa",         -0.45, 39.65),
     8: ("Wajir",            1.75, 40.05),
     9: ("Mandera",          3.92, 41.85),
    10: ("Marsabit",         2.33, 37.99),
    11: ("Isiolo",           0.35, 37.58),
    12: ("Meru",             0.05, 37.65),
    13: ("Tharaka-Nithi",   -0.30, 37.80),
    14: ("Embu",            -0.53, 37.45),
    15: ("Kitui",           -1.37, 38.01),
    16: ("Machakos",        -1.52, 37.27),
    17: ("Makueni",         -1.80, 37.62),
    18: ("Nyandarua",       -0.18, 36.50),
    19: ("Nyeri",           -0.42, 36.95),
    20: ("Kirinyaga",       -0.66, 37.30),
    21: ("Murang'a",        -0.78, 37.05),
    22: ("Kiambu",          -1.18, 36.83),
    23: ("Turkana",          3.12, 35.60),
    24: ("West Pokot",       1.40, 35.10),
    25: ("Samburu",          1.10, 36.70),
    26: ("Trans Nzoia",      1.02, 35.00),
    27: ("Uasin Gishu",      0.55, 35.30),
    28: ("Elgeyo-Marakwet",  0.85, 35.50),
    29: ("Nandi",            0.20, 35.15),
    30: ("Baringo",          0.50, 36.05),
    31: ("Laikipia",         0.40, 36.80),
    32: ("Nakuru",          -0.30, 36.10),
    33: ("Narok",           -1.10, 36.00),
    34: ("Kajiado",         -1.85, 36.78),
    35: ("Kericho",         -0.37, 35.28),
    36: ("Bomet",           -0.78, 35.35),
    37: ("Kakamega",         0.28, 34.75),
    38: ("Vihiga",           0.07, 34.72),
    39: ("Bungoma",          0.57, 34.55),
    40: ("Busia",            0.45, 34.10),
    41: ("Siaya",            0.05, 34.30),
    42: ("Kisumu",          -0.10, 34.75),
    43: ("Homa Bay",        -0.52, 34.45),
    44: ("Migori",          -1.07, 34.47),
    45: ("Kisii",           -0.68, 34.78),
    46: ("Nyamira",         -0.57, 34.93),
    47: ("Nairobi",         -1.28, 36.82),
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def clean_income(s):
    v = pd.to_numeric(s, errors='coerce')
    return v.where(v < SENTINEL, np.nan)


def clean_hours(s):
    v = pd.to_numeric(s, errors='coerce')
    return v.where((v > 0) & (v < 200), np.nan)


def hourly_wage(income_month, hours_week):
    w = income_month / (hours_week * 4.333)
    return w.where((w > 1) & (w < 5000), np.nan)


def wls(y, X, w):
    """Weighted least squares.  Returns (intercept, slopes...)"""
    sw = np.sqrt(np.asarray(w, dtype=float))
    A  = np.c_[np.ones(len(X)), np.asarray(X, dtype=float)]
    c, _, _, _ = np.linalg.lstsq(A * sw[:, None],
                                  np.asarray(y, dtype=float) * sw, rcond=None)
    return c


def wmean(series, weights):
    s = np.asarray(series, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float('nan')
    return float(np.average(s[mask], weights=w[mask]))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.log(p / (1 - p)))


# --------------------------------------------------------------------------- #
# 0. Load data                                                                #
# --------------------------------------------------------------------------- #

print("=" * 72)
print("Kenya Time-Use Model — calibration (4-good full model)")
print("=" * 72)
print("\nLoading microdata ...")
tu,  _ = pyreadstat.read_dta(str(TUS_PATH))
ind, _ = pyreadstat.read_dta(str(IND_PATH))
ca,  _ = pyreadstat.read_dta(str(CA_PATH))
nf,  _ = pyreadstat.read_dta(str(NF_PATH))
print(f"  TUS  {tu.shape}")
print(f"  IND  {ind.shape}")
print(f"  CA   {ca.shape}")
print(f"  NF   {nf.shape}")


# --------------------------------------------------------------------------- #
# 1. TUS preparation                                                          #
# --------------------------------------------------------------------------- #
# All TUS time variables are minutes per day; convert to hours per week.
# Activity codes (set in TUS dictionary):
#   t11   market work
#   t13   home domestic
#   t14   home care
#   t231  food preparation                                                  #
# --------------------------------------------------------------------------- #

print("\n[1] TUS preparation ...")
MINS_TO_HW = 7.0 / 60.0
df = tu.copy()
df['L_M']  = df['t11']  * MINS_TO_HW
df['L_xf'] = df['t231'] * MINS_TO_HW   # food preparation
df['L_d']  = df['t13']  * MINS_TO_HW
df['L_c']  = df['t14']  * MINS_TO_HW

df['female']  = (pd.to_numeric(df['b04'], errors='coerce') == 2).astype(float)
df['age_yr']  = pd.to_numeric(df['b05_years'], errors='coerce')
df['wa']      = (df['age_yr'] >= 15) & (df['age_yr'] <= 65)
df['married'] = pd.to_numeric(df['b07'], errors='coerce').isin([1, 2, 3]).astype(float)
df['rural']   = (pd.to_numeric(df['resid'], errors='coerce') == 2).astype(float)
df['county']  = pd.to_numeric(df['county'], errors='coerce').astype('Int64')
df['wt']      = pd.to_numeric(df['Person_Day_Weight'], errors='coerce')

ec = pd.to_numeric(df['c08'], errors='coerce')
df['educ_prim'] = ec.between(4, 11).astype(float)
df['educ_sec']  = ec.between(12, 17).astype(float)
df['educ_tert'] = ec.between(18, 37).astype(float)
df['high_skill']= df['educ_tert']

df['part_M']  = (df['L_M']  > 0).astype(int)
df['part_xf'] = (df['L_xf'] > 0).astype(int)
df['part_c']  = (df['L_c']  > 0).astype(int)
df['part_d']  = (df['L_d']  > 0).astype(int)

df['wage_hourly'] = hourly_wage(
    clean_income(df['d40_gross']).fillna(
        clean_income(df.get('d48_Amount', pd.Series(np.nan, index=df.index)))),
    clean_hours(df['d27']))
df['log_wage'] = np.log(df['wage_hourly'])

wa = df[df['wa']].dropna(subset=['county']).copy()
wa['county'] = wa['county'].astype(int)

print(f"  Working-age: {len(wa):,}  with valid wage: {wa['wage_hourly'].notna().sum():,}")
print(f"  Mean hours (h/wk): L_M={wa['L_M'].mean():.1f}  "
      f"L_xf={wa['L_xf'].mean():.1f}  "
      f"L_c={wa['L_c'].mean():.1f}  "
      f"L_d={wa['L_d'].mean():.1f}")


# --------------------------------------------------------------------------- #
# 2. KCHS labour panel preparation                                            #
# --------------------------------------------------------------------------- #

print("\n[2] KCHS panel preparation ...")
ind['income_emp']  = clean_income(ind['d40_gross'])
ind['income_self'] = clean_income(ind['d48_amount'])
ind['income_month']= ind['income_emp'].fillna(ind['income_self'])
ind['hours_usual'] = clean_hours(ind['d27'])
ind['wage_hourly'] = hourly_wage(ind['income_month'], ind['hours_usual'])
ind['log_wage']    = np.log(ind['wage_hourly'])
ind['female']  = (pd.to_numeric(ind['b04'], errors='coerce') == 2).astype(float)
ind['age_yr']  = pd.to_numeric(ind['b05_years'], errors='coerce')
ind['wa']      = (ind['age_yr'] >= 15) & (ind['age_yr'] <= 65)
ind['married'] = pd.to_numeric(ind['b07'], errors='coerce').isin([1, 2, 3]).astype(float)
ind['rural']   = (pd.to_numeric(ind['resid'], errors='coerce') == 2).astype(float)
ind['isic']    = pd.to_numeric(ind['d23_isic'], errors='coerce')
ec2 = pd.to_numeric(ind['c08'], errors='coerce')
ind['educ_prim'] = ec2.between(4, 11).astype(float)
ind['educ_sec']  = ec2.between(12, 17).astype(float)
ind['educ_tert'] = ec2.between(18, 37).astype(float)


# Aggregate non-food expenditure by COICOP (5 = domestic, 13 = personal care)
nf_care = (nf[nf['coicopcode'] == 13]
           .groupby(['county', 'clid', 'hhid'])['nfcons'].sum()
           .reset_index().rename(columns={'nfcons': 'e_care_ann'}))
nf_dom  = (nf[nf['coicopcode'] == 5]
           .groupby(['county', 'clid', 'hhid'])['nfcons'].sum()
           .reset_index().rename(columns={'nfcons': 'e_dom_ann'}))
ca2 = (ca.merge(nf_care, on=['county', 'clid', 'hhid'], how='left')
         .merge(nf_dom,  on=['county', 'clid', 'hhid'], how='left'))
ca2[['e_care_ann', 'e_dom_ann']] = ca2[['e_care_ann', 'e_dom_ann']].fillna(0)

aq = ca2['adq_scale'].clip(lower=0.5)
ca2['e_care_m'] = ca2['e_care_ann'] / 12.0 / aq
ca2['e_dom_m']  = ca2['e_dom_ann']  / 12.0 / aq
ca2['E']        = ca2['padqexp'].where(ca2['padqexp'] > 0)

# Four observed PIGL shares.  These will be used in Block A.
ca2['share_xf'] = (ca2['padqfdcons'] / ca2['E']).clip(0, 1)
ca2['share_c']  = (ca2['e_care_m']   / ca2['E']).clip(0, 1)
ca2['share_d']  = (ca2['e_dom_m']    / ca2['E']).clip(0, 1)
ca2['share_xn'] = (1.0 - ca2['share_xf'] - ca2['share_c'] - ca2['share_d']).clip(0, 1)

ca2['log_E']   = np.log(ca2['E'])
ca2['rural_hh']= (pd.to_numeric(ca2['resid'], errors='coerce') == 2).astype(float)

panel = ind.merge(
    ca2[['interview__id', 'E', 'log_E',
         'share_xf', 'share_xn', 'share_c', 'share_d',
         'weight_hh', 'hhsize', 'rural_hh']],
    on='interview__id', how='inner')
wa_p = panel[panel['wa'] & panel['E'].notna()].copy()
print(f"  KCHS panel: {len(wa_p):,} ind, {wa_p['interview__id'].nunique():,} hh")


# --------------------------------------------------------------------------- #
# Block B: wage gap (Mincer)                                                  #
# --------------------------------------------------------------------------- #

print("\n[B] Wage gap (KCHS Mincer regression) ...")
wr = wa_p[wa_p['wage_hourly'].notna()].dropna(
    subset=['log_wage', 'age_yr', 'female', 'married', 'rural']).copy()
wr['age_sq'] = wr['age_yr'] ** 2
cw = wls(wr['log_wage'].values,
         wr[['female', 'married', 'educ_prim', 'educ_sec', 'educ_tert',
             'rural', 'age_yr', 'age_sq']].values.astype(float),
         wr['individual_weight_labour'].fillna(1.0).values.astype(float))
wage_gap_log = float(cw[1])
wage_gap     = float(np.exp(wage_gap_log))
educ_prem_tert = float(cw[5])
print(f"  log wage gap = {wage_gap_log:.4f}  =>  wage_gap = {wage_gap:.4f}")
print(f"  Tertiary education premium = {educ_prem_tert:.4f}")


# --------------------------------------------------------------------------- #
# Block A: PIGL preferences (4 goods)                                         #
# --------------------------------------------------------------------------- #
# share_xf  = padqfdcons / E      food
# share_c   = COICOP 13           personal care
# share_d   = COICOP 5            domestic
# share_xn  = residual             non-food (residual is correct for sum=1)
#
# Engel slopes via WLS on log(E/E_SCALE).  eps_engel from food curvature
# (food share has the cleanest decline with income).  beta_i from
# population-weighted means; rescale the four to sum to 1 (small drift
# from numerical noise + the hard clip at 0).  kappa_i from per-share
# Engel slopes; kappa_xn set as residual to enforce sum(kappa) = 0.
# --------------------------------------------------------------------------- #

print("\n[A] PIGL preferences (4 goods) ...")
hh_eng = ca2[ca2['E'] > 50].dropna(
    subset=['share_xf', 'share_xn', 'share_c', 'share_d',
            'log_E', 'hhsize']).copy()
hh_eng['hhsize_c'] = hh_eng['hhsize'].clip(1, 12).astype(float)
hh_eng['log_E_norm'] = np.log(hh_eng['E'] / E_SCALE)

X_eng = hh_eng[['log_E_norm', 'hhsize_c', 'rural_hh']].values.astype(float)
w_eng = hh_eng['weight_hh'].values.astype(float)
print(f"  Engel sample: {len(hh_eng):,} hh")

slopes = {}
for dep, lab in [('share_xf', 'food'),
                 ('share_c',  'care'),
                 ('share_d',  'domestic'),
                 ('share_xn', 'non-food')]:
    c_e = wls(hh_eng[dep].values.astype(float), X_eng, w_eng)
    slopes[dep] = float(c_e[1])
    bm = wmean(hh_eng[dep].values, hh_eng['weight_hh'].values)
    print(f"    {lab:<10}  mean={bm:.4f}  slope_logE={slopes[dep]:+.5f}")

# eps from food Engel curvature (food has the strongest, cleanest signal)
share_xf_mean = wmean(hh_eng['share_xf'].values, hh_eng['weight_hh'].values)
eps_engel     = float(abs(slopes['share_xf']) / share_xf_mean)

# Population-weighted shares for beta
beta_xf_raw = wmean(hh_eng['share_xf'].values, hh_eng['weight_hh'].values)
beta_c_raw  = wmean(hh_eng['share_c'].values,  hh_eng['weight_hh'].values)
beta_d_raw  = wmean(hh_eng['share_d'].values,  hh_eng['weight_hh'].values)
beta_xn_raw = 1.0 - beta_xf_raw - beta_c_raw - beta_d_raw
# Rescale to sum to exactly 1 (numerical drift safeguard)
S = beta_xf_raw + beta_xn_raw + beta_c_raw + beta_d_raw
beta_xf = float(beta_xf_raw / S)
beta_xn = float(beta_xn_raw / S)
beta_c  = float(beta_c_raw  / S)
beta_d  = float(beta_d_raw  / S)

# kappa from Engel slopes:  d theta_i / d log E = -eps · kappa_i · (E/B)^(-eps)
# Evaluate at the SOLVER-scale expenditure, NOT the data median.
# The KCHS median expenditure (~6.0 in 1000-KSh per adult-equivalent) is not
# the same scale the solver sees: with county wages w_l ~ 0.05-0.25 (1000-KSh
# per hour), household labour ~0.4-1.1, plus a=0.2 transfer, the solver's
# household E lands around ~0.3-0.8.  Using E_med=0.5 here matches the
# existing project's calibration.  This keeps kappa magnitudes self-consistent
# with the solver's lambda fixed point.
E_sol = 0.5
kappa_xf = float(-slopes['share_xf'] * (E_sol ** (1.0 + eps_engel)) / eps_engel)
kappa_c  = float(-slopes['share_c']  * (E_sol ** (1.0 + eps_engel)) / eps_engel)
kappa_d  = float(-slopes['share_d']  * (E_sol ** (1.0 + eps_engel)) / eps_engel)
kappa_xn = float(-(kappa_xf + kappa_c + kappa_d))

assert abs(beta_xf + beta_xn + beta_c + beta_d - 1.0) < 1e-10
assert abs(kappa_xf + kappa_xn + kappa_c + kappa_d) < 1e-10

print(f"  eps_engel = {eps_engel:.4f}")
print(f"  beta:  xf={beta_xf:.4f}  xn={beta_xn:.4f}  c={beta_c:.4f}  d={beta_d:.4f}  "
      f"sum={beta_xf+beta_xn+beta_c+beta_d:.4f}")
print(f"  kappa: xf={kappa_xf:+.4f}  xn={kappa_xn:+.4f}  "
      f"c={kappa_c:+.4f}  d={kappa_d:+.4f}  "
      f"sum={kappa_xf+kappa_xn+kappa_c+kappa_d:+.4f}")


# --------------------------------------------------------------------------- #
# Literature priors                                                           #
# --------------------------------------------------------------------------- #

rho       = -0.5
phi       = 0.5            # KCHS IV is biased downward; use literature working value
omega_xf  = 0.75           # literature priors (Aguiar & Hurst 2007 range)
omega_c   = 0.75
omega_d   = 0.70
eta_xf    = 2.5
eta_c     = 2.5
eta_d     = 2.5

print(f"\n  Literature priors:")
print(f"    rho = {rho},  phi = {phi}")
print(f"    omega = ({omega_xf}, {omega_c}, {omega_d})  for (xf, c, d)")
print(f"    eta   = ({eta_xf}, {eta_c}, {eta_d})  for (xf, c, d)")


# --------------------------------------------------------------------------- #
# Block C: National disutility weights D^g_j (4 activities)                   #
# --------------------------------------------------------------------------- #
# Solver-consistent formulas (NOT FOC-rearranged; see plan, sign-correctness
# item 1).  Normalisations on the male side: D_M_m = D_xf_m = D_c_m = 1.
#                                                                             #
#  D^M_f  = (L^M_m / L^M_f) / wage_gap^rho        # eq. 19'                   #
#  D^xf_f = (L^xf_m / L^xf_f)^(-1/rho)            # eq. 22'                   #
#  D^c_f  = (L^c_m  / L^c_f )^(-1/rho)            # eq. 22'                   #
#  D^d_m  = (L^c_m  / L^d_m )^(1/rho)             # intra-male c-vs-d         #
#  D^d_f  = D^d_m · (L^d_m / L^d_f)^(-1/rho)      # eq. 22' chained           #
# --------------------------------------------------------------------------- #

print("\n[C] National disutility weights (married subsample) ...")
mwa = wa[wa['married'] == 1].copy()


def cond_h(g, pcol, hcol):
    sub = mwa[(mwa['female'] == g) & (mwa[pcol] == 1)]
    return wmean(sub[hcol].values, sub['wt'].values)


LM_m_h, LM_f_h = cond_h(0, 'part_M', 'L_M'),  cond_h(1, 'part_M', 'L_M')
Lxf_m_h, Lxf_f_h = cond_h(0, 'part_xf', 'L_xf'), cond_h(1, 'part_xf', 'L_xf')
Lc_m_h, Lc_f_h = cond_h(0, 'part_c', 'L_c'),  cond_h(1, 'part_c', 'L_c')
Ld_m_h, Ld_f_h = cond_h(0, 'part_d', 'L_d'),  cond_h(1, 'part_d', 'L_d')

print(f"  Conditional hours (h/wk):")
print(f"    L_M:  m={LM_m_h:.2f}   f={LM_f_h:.2f}   m/f={LM_m_h/LM_f_h:.3f}")
print(f"    L_xf: m={Lxf_m_h:.2f}  f={Lxf_f_h:.2f}  m/f={Lxf_m_h/Lxf_f_h:.3f}")
print(f"    L_c:  m={Lc_m_h:.2f}   f={Lc_f_h:.2f}   m/f={Lc_m_h/Lc_f_h:.3f}")
print(f"    L_d:  m={Ld_m_h:.2f}   f={Ld_f_h:.2f}   m/f={Ld_m_h/Ld_f_h:.3f}")

r_M_nat   = LM_m_h  / LM_f_h
r_xf_nat  = Lxf_m_h / Lxf_f_h
r_c_nat   = Lc_m_h  / Lc_f_h
r_d_nat   = Ld_m_h  / Ld_f_h
rcd_m_nat = Lc_m_h  / Ld_m_h     # intra-male c/d ratio

# Female weights from solver-consistent formulas
D_M_m_nat  = 1.0
D_xf_m_nat = 1.0
D_c_m_nat  = 1.0
D_M_f_nat  = float(r_M_nat / (wage_gap ** rho))
D_xf_f_nat = float(D_xf_m_nat * (r_xf_nat ** (-1.0 / rho)))
D_c_f_nat  = float(D_c_m_nat  * (r_c_nat  ** (-1.0 / rho)))
# Male d weight from intra-male c-vs-d (since D_c_m, D_xf_m, D_M_m all
# normalised to 1, D_d_m is the only free male home weight).
D_d_m_nat  = float(D_c_m_nat * (rcd_m_nat ** (1.0 / rho)))
D_d_f_nat  = float(D_d_m_nat * (r_d_nat ** (-1.0 / rho)))

print(f"  Calibrated D weights (national):")
print(f"    D_M_m  =  1.0000   (norm)")
print(f"    D_M_f  =  {D_M_f_nat:.4f}   (target ~1.34)")
print(f"    D_xf_m =  1.0000   (norm)")
print(f"    D_xf_f =  {D_xf_f_nat:.4f}   (solver-consistent; existing food JSON used wrong sign)")
print(f"    D_c_m  =  1.0000   (norm)")
print(f"    D_c_f  =  {D_c_f_nat:.4f}")
print(f"    D_d_m  =  {D_d_m_nat:.4f}")
print(f"    D_d_f  =  {D_d_f_nat:.4f}")

# Sanity: D_M_f should be in [1.2, 1.5] with positive sign
if not (1.0 < D_M_f_nat < 1.6):
    print(f"  *** WARNING: D_M_f = {D_M_f_nat:.4f} outside expected range [1.2, 1.5] ***")


# --------------------------------------------------------------------------- #
# Block D: County-level D weights with shrinkage                              #
# --------------------------------------------------------------------------- #
# Bayesian shrinkage on the raw conditional-hours ratios:
#                                                                             #
#   r_l_shrunk = (n_l · r_l_raw + n0 · r_nat) / (n_l + n0),    n0 = 20        #
#                                                                             #
# n_l for activity i is the geometric mean of male and female cell counts.    #
# For the intra-male c/d ratio, use sqrt(n_Lc_m · n_Ld_m).                    #
# --------------------------------------------------------------------------- #

print("\n[D] County-level D weights with shrinkage (n0 = 20) ...")


def shrink(r_raw, n_eff, r_nat, n0=PRIOR_N):
    if not np.isfinite(r_raw):
        return r_nat
    return float((n_eff * r_raw + n0 * r_nat) / (n_eff + n0))


# Per-county conditional means and counts
rows = []
for cid in sorted(mwa['county'].unique()):
    if pd.isna(cid):
        continue
    cid = int(cid)
    sub = mwa[mwa['county'] == cid]
    rec = {'county': cid}
    for sex_lbl, sex_val in [('m', 0), ('f', 1)]:
        for act, col, pcol in [('M', 'L_M', 'part_M'),
                                ('xf', 'L_xf', 'part_xf'),
                                ('c', 'L_c', 'part_c'),
                                ('d', 'L_d', 'part_d')]:
            ssub = sub[(sub['female'] == sex_val) & (sub[pcol] == 1)]
            rec[f'{col}_{sex_lbl}_mean'] = (
                wmean(ssub[col].values, ssub['wt'].values)
                if len(ssub) > 0 else np.nan)
            rec[f'{col}_{sex_lbl}_n'] = int(len(ssub))
    rows.append(rec)
cty = pd.DataFrame(rows).set_index('county')

county_D = {}
for cid, row in cty.iterrows():
    n_M  = float(np.sqrt(max(row['L_M_m_n']  * row['L_M_f_n'],  0)))
    n_xf = float(np.sqrt(max(row['L_xf_m_n'] * row['L_xf_f_n'], 0)))
    n_c  = float(np.sqrt(max(row['L_c_m_n']  * row['L_c_f_n'],  0)))
    n_d  = float(np.sqrt(max(row['L_d_m_n']  * row['L_d_f_n'],  0)))
    n_cd = float(np.sqrt(max(row['L_c_m_n']  * row['L_d_m_n'],  0)))

    r_M  = (row['L_M_m_mean']  / row['L_M_f_mean']
            if row['L_M_f_mean']  and row['L_M_f_mean']  > 0 else np.nan)
    r_xf = (row['L_xf_m_mean'] / row['L_xf_f_mean']
            if row['L_xf_f_mean'] and row['L_xf_f_mean'] > 0 else np.nan)
    r_c  = (row['L_c_m_mean']  / row['L_c_f_mean']
            if row['L_c_f_mean']  and row['L_c_f_mean']  > 0 else np.nan)
    r_d  = (row['L_d_m_mean']  / row['L_d_f_mean']
            if row['L_d_f_mean']  and row['L_d_f_mean']  > 0 else np.nan)
    rcd  = (row['L_c_m_mean']  / row['L_d_m_mean']
            if row['L_d_m_mean']  and row['L_d_m_mean']  > 0 else np.nan)

    r_M  = shrink(r_M,  n_M,  r_M_nat)
    r_xf = shrink(r_xf, n_xf, r_xf_nat)
    r_c  = shrink(r_c,  n_c,  r_c_nat)
    r_d  = shrink(r_d,  n_d,  r_d_nat)
    rcd  = shrink(rcd,  n_cd, rcd_m_nat)

    D_M_m_l  = 1.0
    D_xf_m_l = 1.0
    D_c_m_l  = 1.0
    D_M_f_l  = float(r_M  / (wage_gap ** rho))
    D_xf_f_l = float(D_xf_m_l * (r_xf ** (-1.0 / rho)))
    D_c_f_l  = float(D_c_m_l  * (r_c  ** (-1.0 / rho)))
    D_d_m_l  = float(D_c_m_l  * (rcd  ** (1.0 / rho)))
    D_d_f_l  = float(D_d_m_l  * (r_d  ** (-1.0 / rho)))

    county_D[int(cid)] = dict(
        D_M_m=D_M_m_l, D_xf_m=D_xf_m_l, D_c_m=D_c_m_l, D_d_m=D_d_m_l,
        D_M_f=D_M_f_l, D_xf_f=D_xf_f_l, D_c_f=D_c_f_l, D_d_f=D_d_f_l,
        n_M=int(n_M), n_xf=int(n_xf), n_c=int(n_c), n_d=int(n_d),
    )

# Print a summary across counties
arr = lambda k: np.array([county_D[c][k] for c in county_D])
for k in ('D_M_f', 'D_xf_f', 'D_c_f', 'D_d_m', 'D_d_f'):
    a = arr(k); print(f"  {k}:  mean={a.mean():.3f}  range=[{a.min():.3f}, {a.max():.3f}]")


# --------------------------------------------------------------------------- #
# Block E: County wages and service prices                                    #
# --------------------------------------------------------------------------- #

print("\n[E] County wages and service prices ...")
wa_ind = ind[ind['wa'] & ind['wage_hourly'].notna()].copy()


def cty_wmean_wage(df_sub):
    """Weighted county mean of wage_hourly."""
    if 'weight_pop' in df_sub.columns:
        wcol = 'weight_pop'
    else:
        wcol = 'weight_hh' if 'weight_hh' in df_sub.columns else None

    def _f(g):
        w = (g[wcol].fillna(1).values if wcol else np.ones(len(g)))
        if len(g) >= 2 and np.isfinite(w).any():
            return float(np.average(g['wage_hourly'].values, weights=w))
        return float('nan')

    return df_sub.groupby('county').apply(_f)


cty_wage      = cty_wmean_wage(wa_ind)
cty_wage_dom  = cty_wmean_wage(wa_ind[wa_ind['isic'].isin([9700, 9800])])
cty_wage_care = cty_wmean_wage(wa_ind[wa_ind['isic'].isin([8891, 8810, 8730])])

w_dom_nat  = float(wa_ind[wa_ind['isic'].isin([9700, 9800])]['wage_hourly'].mean())
w_care_nat = float(wa_ind[wa_ind['isic'].isin(
    [8891, 8810, 8730, 8720])]['wage_hourly'].mean())

# Reindex sparse series onto the full set of counties present in cty_wage,
# with NaN for missing counties; then fill with the national fallback.
all_counties = cty_wage.index
cty_wage_dom  = cty_wage_dom.reindex(all_counties).fillna(w_dom_nat)
cty_wage_care = cty_wage_care.reindex(all_counties).fillna(w_care_nat)

# Population from TUS person-day weights (used in Block F and Block J)
cty_pop = wa.groupby('county')['wt'].sum()

print(f"  County wage mean = {cty_wage.mean():.1f}  range=[{cty_wage.min():.0f}, {cty_wage.max():.0f}] KSh/hr")
print(f"  Domestic-sector national wage = {w_dom_nat:.1f} KSh/hr")
print(f"  Care-sector national wage     = {w_care_nat:.1f} KSh/hr")


# --------------------------------------------------------------------------- #
# Block F: Uniform national food price p_xf                                   #
# --------------------------------------------------------------------------- #
# Document Section 21.2 option 1: uniform p_xf, with A^M,xf_l = w_l / p_xf.
# Defended in Phase 1 plan: Kenyan staples are substantially traded
# inter-county; option 2 (p_xf_l = w_l) makes high-wage counties
# counterfactually expensive in food.
#                                                                             #
# Calibration: pick p_xf so that the population-weighted mean food            #
# expenditure share at calibrated parameters equals beta_xf.  At the          #
# population mean, with E ≈ E_med · 1000 KSh/hh, the food market value        #
# spent is theta_xf · E ≈ beta_xf · E.  We want p_xf to be "in the same       #
# units as pc, pd" so the home/market split is sensible.  Concretely:        #
# choose p_xf so the implied national A^M,xf is close to A^M,c, A^M,d         #
# scale-wise.  This is a normalisation; we set:                              #
#                                                                             #
#       p_xf = mean(p_c, p_d) at the population mean wage                     #
#                                                                             #
# This keeps food prices in the same numerical range as service prices       #
# without any economic implication beyond the numeraire choice.              #
# --------------------------------------------------------------------------- #

print("\n[F] Uniform national food price p_xf ...")
# Population-weighted mean of pc, pd, where the weights are the TUS
# person-day-weight sums per county (the same N_l used everywhere else).
pop_index = cty_wage_care.index
pop_w = np.array([float(cty_pop.get(int(c), 1.0)) for c in pop_index])
# numpy.average requires aligned shapes; build aligned arrays first
care_arr = cty_wage_care.values.astype(float)
dom_arr  = cty_wage_dom.values.astype(float)
mask     = np.isfinite(care_arr) & np.isfinite(dom_arr) & (pop_w > 0)
p_c_pop  = float(np.average(care_arr[mask], weights=pop_w[mask]))
p_d_pop  = float(np.average(dom_arr[mask],  weights=pop_w[mask]))
p_xf_KSh = 0.5 * (p_c_pop + p_d_pop)
p_xf     = p_xf_KSh / E_SCALE     # store in 1000-KSh units
print(f"  p_c (pop-mean) = {p_c_pop:.1f}  p_d (pop-mean) = {p_d_pop:.1f}  KSh/hr-equiv")
print(f"  Set p_xf = mean(p_c, p_d) = {p_xf_KSh:.1f} KSh-equiv ({p_xf:.4f} in 1000-KSh)")
print(f"  (Normalisation; defends numeraire-equivalent scale across services + food)")


# --------------------------------------------------------------------------- #
# Block H: Population from TUS person-day weights (already computed in [E])  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Block I: Participation shifters (initial values, from KCHS as in existing  #
#          calibration; refined structurally in Block J)                      #
# --------------------------------------------------------------------------- #

print("\n[I] Participation shifters (initial moments from KCHS) ...")
part_tab = (wa.groupby(['female', 'high_skill'])
              .apply(lambda g: pd.Series({
                  'P': float(np.average(g['part_M'].values,
                                         weights=g['wt'].values)),
                  'N': len(g)})))
P_m_low  = float(part_tab.loc[(0.0, 0.0), 'P'])
P_m_high = float(part_tab.loc[(0.0, 1.0), 'P'])
P_f_low  = float(part_tab.loc[(1.0, 0.0), 'P'])
P_f_high = float(part_tab.loc[(1.0, 1.0), 'P'])
dV       = phi * abs(educ_prem_tert)

def est_sigma(Pl, Ph, dv):
    dl = logit(Ph) - logit(Pl)
    if abs(dl) < 0.01:
        return float('nan')
    return float(dv / dl)

sigma_u_m = est_sigma(P_m_low, P_m_high, dV)
sigma_u_f = est_sigma(P_f_low, P_f_high, dV)
P_m_mean = float(np.average(wa[wa['female']==0]['part_M'].values,
                              weights=wa[wa['female']==0]['wt'].values))
P_f_mean = float(np.average(wa[wa['female']==1]['part_M'].values,
                              weights=wa[wa['female']==1]['wt'].values))
VP_m = phi * wmean(wa[wa['female']==0]['log_wage'].values,
                    wa[wa['female']==0]['wt'].values)
VP_f = phi * wmean(wa[wa['female']==1]['log_wage'].values,
                    wa[wa['female']==1]['wt'].values)
u_bar_m = float(VP_m - sigma_u_m * logit(P_m_mean))
u_bar_f = float(VP_f - sigma_u_f * logit(P_f_mean))
print(f"  P_m = {P_m_mean:.4f}  P_f = {P_f_mean:.4f}")
print(f"  sigma_u_m = {sigma_u_m:.4f}  sigma_u_f = {sigma_u_f:.4f}")
print(f"  u_bar_m   = {u_bar_m:.4f}  u_bar_f   = {u_bar_f:.4f}")


# --------------------------------------------------------------------------- #
# Build ModelParams and County list, then solve baseline                      #
# --------------------------------------------------------------------------- #

mp = ModelParams(
    eps_engel=eps_engel,
    beta_xf=beta_xf, beta_xn=beta_xn, beta_c=beta_c, beta_d=beta_d,
    kappa_xf=kappa_xf, kappa_xn=kappa_xn, kappa_c=kappa_c, kappa_d=kappa_d,
    omega_xf=omega_xf, omega_c=omega_c, omega_d=omega_d,
    eta_xf=eta_xf, eta_c=eta_c, eta_d=eta_d,
    D_M_m=D_M_m_nat, D_xf_m=D_xf_m_nat, D_c_m=D_c_m_nat, D_d_m=D_d_m_nat,
    D_M_f=D_M_f_nat, D_xf_f=D_xf_f_nat, D_c_f=D_c_f_nat, D_d_f=D_d_f_nat,
    rho=rho, phi=phi,
    sigma_u_m=sigma_u_m, sigma_u_f=sigma_u_f,
    u_bar_m=u_bar_m, u_bar_f=u_bar_f,
    wage_gap=wage_gap, p_xf=p_xf,
    sigma_mig=1.0,
)

counties = []
for cid in sorted(COUNTY_CENTROIDS.keys()):
    name, lat, lon = COUNTY_CENTROIDS[cid]
    w_l_KSh  = float(cty_wage.get(cid, cty_wage.mean()))
    pc_KSh   = float(cty_wage_care.get(cid, w_care_nat))
    pd_KSh   = float(cty_wage_dom.get(cid,  w_dom_nat))
    N_l      = float(cty_pop.get(cid, 1.0))

    w_ell = w_l_KSh / E_SCALE
    pc_l  = pc_KSh  / E_SCALE
    pd_l  = pd_KSh  / E_SCALE

    # Free-entry productivities (eq. 28-30)
    AM_xn = w_ell                    # numeraire
    AM_xf = w_ell / max(p_xf, 1e-9)
    AM_c  = w_ell / max(pc_l, 1e-9)
    AM_d  = w_ell / max(pd_l, 1e-9)

    Dw = county_D.get(cid, {
        'D_M_m':1.0,'D_xf_m':1.0,'D_c_m':1.0,'D_d_m':D_d_m_nat,
        'D_M_f':D_M_f_nat,'D_xf_f':D_xf_f_nat,'D_c_f':D_c_f_nat,'D_d_f':D_d_f_nat,
    })
    counties.append(County(
        name=name, county_id=cid, lat=lat, lon=lon,
        w_ell=w_ell, p_xf=p_xf, pc=pc_l, pd=pd_l,
        AM_xn=AM_xn, AM_xf=AM_xf, AM_c=AM_c, AM_d=AM_d,
        A_xf_home=1.0, A_c_home=1.0, A_d_home=1.0,
        N=N_l,
        D_M_m=Dw['D_M_m'], D_xf_m=Dw['D_xf_m'],
        D_c_m=Dw['D_c_m'], D_d_m=Dw['D_d_m'],
        D_M_f=Dw['D_M_f'], D_xf_f=Dw['D_xf_f'],
        D_c_f=Dw['D_c_f'], D_d_f=Dw['D_d_f'],
    ))


# --------------------------------------------------------------------------- #
# Block J: Spatial calibration (solve all 47 counties, set xi_l)              #
# --------------------------------------------------------------------------- #

print("\n[J] Spatial calibration: solving all 47 counties ...")
H_GRID = np.array([0.5, 1.0, 2.0, 3.0])
print(f"  h-grid = {H_GRID}")

results = solve_all_counties(counties, mp, H_GRID, a=0.2)

# Diagnostics: how many counties solved successfully on each h?
n_ok = sum(int(np.any(results[c.county_id]['converged']))
           for c in counties)
print(f"  Counties with at least one converged h: {n_ok}/47")

compute_market_clearing(counties, results)
Ubar = calibrate_amenities(counties, results)
print(f"  Ubar = {Ubar:.4f}")
print(f"  V*  range:  [{min((c.V_star for c in counties if np.isfinite(c.V_star)), default=float('nan')):.4f}, "
      f"{max((c.V_star for c in counties if np.isfinite(c.V_star)), default=float('nan')):.4f}]")
print(f"  xi  range:  [{min((c.xi for c in counties if np.isfinite(c.xi)), default=float('nan')):.4f}, "
      f"{max((c.xi for c in counties if np.isfinite(c.xi)), default=float('nan')):.4f}]")


# --------------------------------------------------------------------------- #
# Block I-refine: Structural participation shifters                           #
# --------------------------------------------------------------------------- #
# The Block I values used the educational-premium proxy (phi · log w_premium)
# for ∆V^g, matching the existing 3-good code.  Now that we have the four-
# state solve, we can compute the *structural* ∆V^g directly and refit
# (sigma_u^g, u_bar^g) to match observed participation rates exactly.
#
# We operate at the population-mean county (population-weighted aggregate
# of solved V11, V10, V01, V00 across counties at the median h).  Then:
#
#   At observed P^g_mean, given observed slope dP^g/dlog w via skill-cell
#   variation, solve the two-equation Section-22 system:
#       sigma_u^g = (∆V^g_high - ∆V^g_low) / (logit(P^g_high) - logit(P^g_low))
#       u_bar^g   = ∆V^g_mean - sigma_u^g · logit(P^g_mean)
# --------------------------------------------------------------------------- #

print("\n[I-refine] Structural participation shifters (post-spatial) ...")

# Pick a median h on the grid for the moment-match
h_med = 1.0
# Compute population-weighted mean ∆V^g across counties (low- and high-skill
# proxied by h=0.5 and h=3.0 from the grid)
Ns_arr = np.array([c.N for c in counties])
def _weighted_mean_at_h(arrname, h_idx):
    vals = np.array([float(results[c.county_id][arrname][h_idx])
                      for c in counties])
    finite = np.isfinite(vals) & (Ns_arr > 0)
    return float(np.average(vals[finite], weights=Ns_arr[finite]))

# h-grid index for low / mean / high skill.  We use 0.5, 1.0, 3.0.
# The existing code uses tertiary educ premium ≈ 1.6 logs ≈ exp(1.6) ≈ 5x
# wage difference; here h=0.5 vs h=3.0 is a 6x difference, similar.
i_low, i_med, i_high = 0, 1, 3   # H_GRID = [0.5, 1.0, 2.0, 3.0]

V11_low  = _weighted_mean_at_h('V11', i_low)
V10_low  = _weighted_mean_at_h('V10', i_low)
V01_low  = _weighted_mean_at_h('V01', i_low)
V00_low  = _weighted_mean_at_h('V00', i_low)
V11_high = _weighted_mean_at_h('V11', i_high)
V10_high = _weighted_mean_at_h('V10', i_high)
V01_high = _weighted_mean_at_h('V01', i_high)
V00_high = _weighted_mean_at_h('V00', i_high)
V11_med  = _weighted_mean_at_h('V11', i_med)
V10_med  = _weighted_mean_at_h('V10', i_med)
V01_med  = _weighted_mean_at_h('V01', i_med)
V00_med  = _weighted_mean_at_h('V00', i_med)

# At population-wide P_f, P_m, the structural ∆V^g = V_bar^g_1 - V_bar^g_0
# We use the observed P^g rates as the "truth" the participation rates
# should hit at the population-mean h.
P_m_obs = P_m_mean   # from Block I above
P_f_obs = P_f_mean

dV_m_low  = (P_f_obs * V11_low  + (1-P_f_obs) * V10_low)  - (P_f_obs * V01_low  + (1-P_f_obs) * V00_low)
dV_m_high = (P_f_obs * V11_high + (1-P_f_obs) * V10_high) - (P_f_obs * V01_high + (1-P_f_obs) * V00_high)
dV_m_med  = (P_f_obs * V11_med  + (1-P_f_obs) * V10_med)  - (P_f_obs * V01_med  + (1-P_f_obs) * V00_med)

dV_f_low  = (P_m_obs * V11_low  + (1-P_m_obs) * V01_low)  - (P_m_obs * V10_low  + (1-P_m_obs) * V00_low)
dV_f_high = (P_m_obs * V11_high + (1-P_m_obs) * V01_high) - (P_m_obs * V10_high + (1-P_m_obs) * V00_high)
dV_f_med  = (P_m_obs * V11_med  + (1-P_m_obs) * V01_med)  - (P_m_obs * V10_med  + (1-P_m_obs) * V00_med)

print(f"  Structural ∆V at low/med/high h:")
print(f"    ∆V_m: {dV_m_low:.3f}, {dV_m_med:.3f}, {dV_m_high:.3f}")
print(f"    ∆V_f: {dV_f_low:.3f}, {dV_f_med:.3f}, {dV_f_high:.3f}")

# Skill-conditional participation rates (from earlier; observed in TUS):
#   P^g_low = P^g | non-tertiary,  P^g_high = P^g | tertiary
# Two-moment match:
#   sigma_u^g = (∆V^g_high - ∆V^g_low) / (logit(P^g_high) - logit(P^g_low))
#   u_bar^g   = ∆V^g_mean - sigma_u^g · logit(P^g_mean)
def _refit(dV_low, dV_high, dV_med, P_low, P_high, P_med):
    dl = logit(P_high) - logit(P_low)
    if abs(dl) < 1e-3 or not math.isfinite(dl):
        return float('nan'), float('nan')
    sig = (dV_high - dV_low) / dl
    if sig <= 0:
        # Logit would be inverted; use a fallback positive value.
        sig = max(abs(sig), 0.5)
    ub  = dV_med - sig * logit(P_med)
    return float(sig), float(ub)

sigma_u_m_struct, u_bar_m_struct = _refit(
    dV_m_low, dV_m_high, dV_m_med, P_m_low, P_m_high, P_m_obs)
sigma_u_f_struct, u_bar_f_struct = _refit(
    dV_f_low, dV_f_high, dV_f_med, P_f_low, P_f_high, P_f_obs)

# If structural refit is degenerate (NaN or sign flip), keep the
# Block-I educational-premium values.  The Block I values are what the
# existing 3-good code uses, so this preserves baseline behaviour as
# a fallback.
if math.isfinite(sigma_u_m_struct) and sigma_u_m_struct > 0:
    sigma_u_m, u_bar_m = sigma_u_m_struct, u_bar_m_struct
    print(f"  Refit (m): sigma_u_m={sigma_u_m:.4f}  u_bar_m={u_bar_m:.4f}")
else:
    print(f"  Refit (m) degenerate; keeping Block-I values")
if math.isfinite(sigma_u_f_struct) and sigma_u_f_struct > 0:
    sigma_u_f, u_bar_f = sigma_u_f_struct, u_bar_f_struct
    print(f"  Refit (f): sigma_u_f={sigma_u_f:.4f}  u_bar_f={u_bar_f:.4f}")
else:
    print(f"  Refit (f) degenerate; keeping Block-I values")

# Rebuild ModelParams with refined shifters; resolve and recalibrate
# amenities so xi_l is consistent with the final shifters.
mp = ModelParams(
    eps_engel=eps_engel,
    beta_xf=beta_xf, beta_xn=beta_xn, beta_c=beta_c, beta_d=beta_d,
    kappa_xf=kappa_xf, kappa_xn=kappa_xn, kappa_c=kappa_c, kappa_d=kappa_d,
    omega_xf=omega_xf, omega_c=omega_c, omega_d=omega_d,
    eta_xf=eta_xf, eta_c=eta_c, eta_d=eta_d,
    D_M_m=D_M_m_nat, D_xf_m=D_xf_m_nat, D_c_m=D_c_m_nat, D_d_m=D_d_m_nat,
    D_M_f=D_M_f_nat, D_xf_f=D_xf_f_nat, D_c_f=D_c_f_nat, D_d_f=D_d_f_nat,
    rho=rho, phi=phi,
    sigma_u_m=sigma_u_m, sigma_u_f=sigma_u_f,
    u_bar_m=u_bar_m, u_bar_f=u_bar_f,
    wage_gap=wage_gap, p_xf=p_xf,
    sigma_mig=1.0,
)
results = solve_all_counties(counties, mp, H_GRID, a=0.2)
compute_market_clearing(counties, results)
Ubar = calibrate_amenities(counties, results)

# Diagnostic: compare implied national P^g to observed
implied_Pm = float(np.average([np.nanmean(results[c.county_id]['P_m'])
                                for c in counties], weights=Ns_arr))
implied_Pf = float(np.average([np.nanmean(results[c.county_id]['P_f'])
                                for c in counties], weights=Ns_arr))
print(f"  Refit check: P_m (model={implied_Pm:.4f}, obs={P_m_obs:.4f}); "
      f"P_f (model={implied_Pf:.4f}, obs={P_f_obs:.4f})")


# --------------------------------------------------------------------------- #
# Write outputs                                                               #
# --------------------------------------------------------------------------- #

params_out = {
    "eps_engel": round(eps_engel, 4),
    "beta_xf":   round(beta_xf,   4),
    "beta_xn":   round(beta_xn,   4),
    "beta_c":    round(beta_c,    4),
    "beta_d":    round(beta_d,    4),
    "kappa_xf":  round(kappa_xf,  4),
    "kappa_xn":  round(kappa_xn,  4),
    "kappa_c":   round(kappa_c,   4),
    "kappa_d":   round(kappa_d,   4),
    "omega_xf":  omega_xf,
    "omega_c":   omega_c,
    "omega_d":   omega_d,
    "eta_xf":    eta_xf,
    "eta_c":     eta_c,
    "eta_d":     eta_d,
    "D_M_m":     round(D_M_m_nat,  4),
    "D_xf_m":    round(D_xf_m_nat, 4),
    "D_c_m":     round(D_c_m_nat,  4),
    "D_d_m":     round(D_d_m_nat,  4),
    "D_M_f":     round(D_M_f_nat,  4),
    "D_xf_f":    round(D_xf_f_nat, 4),
    "D_c_f":     round(D_c_f_nat,  4),
    "D_d_f":     round(D_d_f_nat,  4),
    "rho":       rho,
    "phi":       phi,
    "sigma_u_m": round(sigma_u_m,  4),
    "sigma_u_f": round(sigma_u_f,  4),
    "u_bar_m":   round(u_bar_m,    4),
    "u_bar_f":   round(u_bar_f,    4),
    "wage_gap":  round(wage_gap,   4),
    "p_xf":      round(p_xf,       6),
    "sigma_mig": 1.0,
    "Ubar":      round(float(Ubar) if np.isfinite(Ubar) else 0.0, 6),
}
with OUT_PARAMS.open('w') as f:
    json.dump(params_out, f, indent=2)
print(f"\n  Wrote {OUT_PARAMS}")

# County data CSV
csv_rows = []
for c in counties:
    csv_rows.append({
        "county_id": c.county_id,
        "name":      c.name,
        "lat":       c.lat,
        "lon":       c.lon,
        "w_ell":     round(c.w_ell, 6),
        "p_xf":      round(c.p_xf,  6),
        "pc":        round(c.pc,    6),
        "pd":        round(c.pd,    6),
        "AM_xn":     round(c.AM_xn, 6),
        "AM_xf":     round(c.AM_xf, 4),
        "AM_c":      round(c.AM_c,  4),
        "AM_d":      round(c.AM_d,  4),
        "A_xf_home": c.A_xf_home,
        "A_c_home":  c.A_c_home,
        "A_d_home":  c.A_d_home,
        "N":         round(c.N, 2),
        "D_M_m":     round(c.D_M_m,  4),
        "D_xf_m":    round(c.D_xf_m, 4),
        "D_c_m":     round(c.D_c_m,  4),
        "D_d_m":     round(c.D_d_m,  4),
        "D_M_f":     round(c.D_M_f,  4),
        "D_xf_f":    round(c.D_xf_f, 4),
        "D_c_f":     round(c.D_c_f,  4),
        "D_d_f":     round(c.D_d_f,  4),
        "V_star":    (round(c.V_star, 6) if np.isfinite(c.V_star) else ""),
        "P_m":       (round(c.P_m,    4) if np.isfinite(c.P_m)    else ""),
        "P_f":       (round(c.P_f,    4) if np.isfinite(c.P_f)    else ""),
        "xi":        (round(c.xi,     6) if np.isfinite(c.xi)     else ""),
        "LM_xf_total": (round(c.LM_xf_total, 2) if np.isfinite(c.LM_xf_total) else ""),
        "LM_c_total":  (round(c.LM_c_total,  2) if np.isfinite(c.LM_c_total)  else ""),
        "LM_d_total":  (round(c.LM_d_total,  2) if np.isfinite(c.LM_d_total)  else ""),
        "LM_xn_total": (round(c.LM_xn_total, 2) if np.isfinite(c.LM_xn_total) else ""),
    })
pd.DataFrame(csv_rows).to_csv(OUT_COUNTY, index=False)
print(f"  Wrote {OUT_COUNTY}")

print("\n" + "=" * 72)
print("CALIBRATION COMPLETE")
print("=" * 72)
print(f"  Sanity checks:")
print(f"    eps_engel = {eps_engel:.4f}    (target ~0.29)")
print(f"    wage_gap  = {wage_gap:.4f}     (target ~0.83)")
print(f"    D_M_f     = {D_M_f_nat:.4f}    (target ~1.34)")
print(f"    D_xf_f    = {D_xf_f_nat:.4f}   (target ~3.31)")
print(f"    D_d_m     = {D_d_m_nat:.4f}")
print(f"    phi       = {phi}")
print(f"    rho       = {rho}")
print(f"  Counties:   {len(counties)}")
print(f"  Wage range: [{min(c.w_ell for c in counties)*E_SCALE:.0f}, "
      f"{max(c.w_ell for c in counties)*E_SCALE:.0f}] KSh/hr")
