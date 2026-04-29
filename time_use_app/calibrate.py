#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate.py  –  Calibrate the extended time-use model from microdata.

Datasets
────────
TUS  : time_use_final_ver13.dta          24,004 individuals, 1 diary day each
IND  : individuals_microdata.dta         71,239 individuals, KCHS labour module
CA   : consumption_aggregate_microdata.dta  17,894 households, KCHS expenditure
NF   : nonfood_items_microdata.dta       279,356 item×household rows, KCHS

Survey structure
────────────────
TUS and KCHS (IND + CA + NF) are INDEPENDENT samples from the same population.
They cannot be merged at the household level.  They share the same county codes
(1-47) and can be combined at that level.
IND links to CA on interview__id (70,194 matched rows -> 16,849 households).

Data facts established by exploration
──────────────────────────────────────
- TUS time vars (t11, t13, t14): MINUTES PER DAY. t2100 ~= 1440 confirms.
  Convert: hours/week = minutes/day * 7 / 60.
- Model mapping: L_M=t11, L_d=t13, L_c=t14
- Wages: d40_gross=monthly employee KSh, d48_amount=monthly self-emp KSh
  sentinel=999999999; hourly = monthly / (d27_usual_hours * 4.333)
- COICOP: 5=Furnishings/maintenance (domestic proxy), 13=Personal care (care proxy)
- ISIC: 9700=domestic household employers, 8891/8810=care services
- County codes 1-47 are CONSISTENT across TUS and KCHS.
"""
from __future__ import annotations
import json, warnings
import numpy as np
import pandas as pd
import pyreadstat
from numpy.linalg import lstsq

warnings.filterwarnings('ignore')

TUS_PATH = './time_use_final_ver13.dta'
IND_PATH = './individuals_microdata.dta'
CA_PATH  = './consumption_aggregate_microdata.dta'
NF_PATH  = './nonfood_items_microdata.dta'

SENTINEL = 999_990_000

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
    sw = np.sqrt(np.asarray(w, dtype=float))
    A  = np.c_[np.ones(len(X)), np.asarray(X, dtype=float)]
    c, _, _, _ = lstsq(A * sw[:, None], np.asarray(y, dtype=float) * sw, rcond=None)
    return c

def wmean(series, weights):
    mask = pd.Series(series).notna() & np.isfinite(pd.Series(series))
    if mask.sum() == 0:
        return np.nan
    return float(np.average(np.asarray(series)[mask], weights=np.asarray(weights)[mask]))

def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.log(p / (1 - p)))

# ═══════════════════════════════════════════════════════════════════════════
# 0. Load
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data ...")
tu, _ = pyreadstat.read_dta(TUS_PATH)
ind,_ = pyreadstat.read_dta(IND_PATH)
ca, _ = pyreadstat.read_dta(CA_PATH)
nf, _ = pyreadstat.read_dta(NF_PATH)
print(f"  TUS {tu.shape}  IND {ind.shape}  CA {ca.shape}  NF {nf.shape}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. TUS preparation
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Step 1: TUS --")
MINS_TO_HW = 7.0 / 60.0
df = tu.copy()
df['L_M']  = df['t11'] * MINS_TO_HW
df['L_d']  = df['t13'] * MINS_TO_HW
df['L_c']  = df['t14'] * MINS_TO_HW
df['female']  = (pd.to_numeric(df['b04'], errors='coerce') == 2).astype(float)
df['age_yr']  = pd.to_numeric(df['b05_years'], errors='coerce')
df['wa']      = (df['age_yr'] >= 15) & (df['age_yr'] <= 65)
df['married'] = pd.to_numeric(df['b07'], errors='coerce').isin([1,2,3]).astype(float)
ec = pd.to_numeric(df['c08'], errors='coerce')
df['educ_prim'] = ec.between(4,11).astype(float)
df['educ_sec']  = ec.between(12,17).astype(float)
df['educ_tert'] = ec.between(18,37).astype(float)
df['high_skill']= df['educ_tert']
df['rural']     = (pd.to_numeric(df['resid'], errors='coerce') == 2).astype(float)
df['part_M'] = (df['L_M'] > 0).astype(int)
df['part_d'] = (df['L_d'] > 0).astype(int)
df['part_c'] = (df['L_c'] > 0).astype(int)
df['wage_hourly'] = hourly_wage(
    clean_income(df['d40_gross']).fillna(
        clean_income(df.get('d48_Amount', pd.Series(np.nan, index=df.index)))),
    clean_hours(df['d27']))
df['log_wage'] = np.log(df['wage_hourly'])
wa = df[df['wa']].copy()
print(f"  Working-age: {len(wa):,}  with wage: {wa['wage_hourly'].notna().sum():,}")
print(f"  Mean L_M={wa['L_M'].mean():.1f}h/wk  L_d={wa['L_d'].mean():.1f}  L_c={wa['L_c'].mean():.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. IND + CA preparation
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Step 2: KCHS panel --")
ind['income_emp']  = clean_income(ind['d40_gross'])
ind['income_self'] = clean_income(ind['d48_amount'])
ind['income_month']= ind['income_emp'].fillna(ind['income_self'])
ind['hours_usual'] = clean_hours(ind['d27'])
ind['wage_hourly'] = hourly_wage(ind['income_month'], ind['hours_usual'])
ind['log_wage']    = np.log(ind['wage_hourly'])
ind['female']   = (pd.to_numeric(ind['b04'], errors='coerce') == 2).astype(float)
ind['age_yr']   = pd.to_numeric(ind['b05_years'], errors='coerce')
ind['wa']       = (ind['age_yr'] >= 15) & (ind['age_yr'] <= 65)
ind['married']  = pd.to_numeric(ind['b07'], errors='coerce').isin([1,2,3]).astype(float)
ec2             = pd.to_numeric(ind['c08'], errors='coerce')
ind['educ_prim']= ec2.between(4,11).astype(float)
ind['educ_sec'] = ec2.between(12,17).astype(float)
ind['educ_tert']= ec2.between(18,37).astype(float)
ind['high_skill']= ind['educ_tert']
ind['rural']    = (pd.to_numeric(ind['resid'], errors='coerce') == 2).astype(float)
ind['isic']     = pd.to_numeric(ind['d23_isic'], errors='coerce')

nf_care = (nf[nf['coicopcode']==13].groupby(['county','clid','hhid'])['nfcons']
           .sum().reset_index().rename(columns={'nfcons':'e_care_ann'}))
nf_dom  = (nf[nf['coicopcode']==5].groupby(['county','clid','hhid'])['nfcons']
           .sum().reset_index().rename(columns={'nfcons':'e_dom_ann'}))
ca2 = (ca.merge(nf_care,on=['county','clid','hhid'],how='left')
         .merge(nf_dom, on=['county','clid','hhid'],how='left'))
ca2[['e_care_ann','e_dom_ann']] = ca2[['e_care_ann','e_dom_ann']].fillna(0)
aq = ca2['adq_scale'].clip(lower=0.5)
ca2['e_care_m'] = ca2['e_care_ann'] / 12.0 / aq
ca2['e_dom_m']  = ca2['e_dom_ann']  / 12.0 / aq
ca2['E']        = ca2['padqexp'].where(ca2['padqexp'] > 0)
ca2['share_c']  = (ca2['e_care_m'] / ca2['E']).clip(0, 1)
ca2['share_d']  = (ca2['e_dom_m']  / ca2['E']).clip(0, 1)
ca2['share_x']  = (ca2['padqfdcons'] / ca2['E']).clip(0, 1)
ca2['log_E']    = np.log(ca2['E'])
ca2['rural_hh'] = (pd.to_numeric(ca2['resid'], errors='coerce') == 2).astype(float)

panel = ind.merge(ca2[['interview__id','E','log_E','share_c','share_d','share_x',
                         'weight_hh','hhsize','rural_hh']],
                  on='interview__id', how='inner')
wa_p = panel[panel['wa'] & panel['E'].notna()].copy()
print(f"  KCHS panel: {len(wa_p):,} individuals, {wa_p['interview__id'].nunique():,} households")
print(f"  With valid wage: {wa_p['wage_hourly'].notna().sum():,}")

# ═══════════════════════════════════════════════════════════════════════════
# Block F early: County wages + service prices
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Block F (early): county wages --")
wa_ind = ind[ind['wa'] & ind['wage_hourly'].notna()].copy()

def cty_wmean(df_sub, wcol='weight_pop'):
    return (df_sub.groupby('county')
            .apply(lambda g: np.average(g['wage_hourly'],
                                         weights=g[wcol].fillna(1))
                   if len(g) >= 2 else np.nan))

cty_wage     = cty_wmean(wa_ind).rename('w_ell')
cty_wage_dom = cty_wmean(wa_ind[wa_ind['isic'].isin([9700,9800])]).rename('w_dom')
cty_wage_care= cty_wmean(wa_ind[wa_ind['isic'].isin([8891,8810,8730])]).rename('w_care')

w_dom_nat  = float(wa_ind[wa_ind['isic'].isin([9700,9800])]['wage_hourly'].mean())
w_care_nat = float(wa_ind[wa_ind['isic'].isin([8891,8810,8730,8720])]['wage_hourly'].mean())
cty_wage_dom  = cty_wage_dom.fillna(w_dom_nat)
cty_wage_care = cty_wage_care.fillna(w_care_nat)

cty_pop   = wa.groupby('county')['Person_Day_Weight'].sum().rename('N_tus')
cty_rural = (wa.groupby('county')
              .apply(lambda g: np.average(g['rural'], weights=g['Person_Day_Weight']))
              .rename('share_rural'))

print(f"  County wages: mean={cty_wage.mean():.1f}  range=[{cty_wage.min():.0f},{cty_wage.max():.0f}]")
print(f"  Domestic service national wage: {w_dom_nat:.1f} KSh/hr")
print(f"  Care service national wage:     {w_care_nat:.1f} KSh/hr")

# Gender wage gap from KCHS (larger, more reliable sample)
wr = wa_p[wa_p['wage_hourly'].notna()].dropna(
    subset=['log_wage','age_yr','female','married','rural']).copy()
wr['age_sq'] = wr['age_yr'] ** 2
cw = wls(wr['log_wage'].values,
         wr[['female','married','educ_prim','educ_sec','educ_tert',
              'rural','age_yr','age_sq']].values.astype(float),
         wr['individual_weight_labour'].fillna(1.0).values.astype(float))
wage_gap_log   = float(cw[1])
wage_gap       = float(np.exp(wage_gap_log))
educ_prem_tert = float(cw[5])
print(f"\n  Gender wage gap: {wage_gap_log:.4f} log pts => y_f/y_m = {wage_gap:.4f}")
print(f"  Tertiary educ premium: {educ_prem_tert:.4f} log pts")

# ═══════════════════════════════════════════════════════════════════════════
# Block B: phi, rho
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Block B: phi, rho --")

# First stage: predict log wage from education + age (instrument excludes gender/marital)
has_wage = wa[wa['wage_hourly'].notna()].dropna(
    subset=['log_wage','age_yr','educ_prim','educ_sec','educ_tert','rural']).copy()
has_wage['age_sq'] = has_wage['age_yr'] ** 2
c_fs = wls(has_wage['log_wage'].values,
           has_wage[['educ_prim','educ_sec','educ_tert','rural','age_yr','age_sq']].values.astype(float),
           has_wage['Person_Day_Weight'].values.astype(float))
# Apply first stage to full sample
wa2 = wa.copy()
wa2['age_sq'] = wa2['age_yr'] ** 2
wa2['log_wage_hat'] = (
    np.c_[np.ones(len(wa2)),
          wa2[['educ_prim','educ_sec','educ_tert','rural','age_yr','age_sq']].values.astype(float)]
    @ c_fs)

res_B = {}
for act, hcol, pcol in [('M','L_M','part_M'),('d','L_d','part_d'),('c','L_c','part_c')]:
    for g, gn in [(0,'men'),(1,'women')]:
        sub = wa2[(wa2['female']==g)&(wa2[pcol]==1)].dropna(
            subset=[hcol,'age_yr','log_wage_hat']).copy()
        sub['age_sq'] = sub['age_yr']**2
        if len(sub) < 30:
            continue
        c2 = wls(sub[hcol].values.astype(float),
                 sub[['log_wage_hat','married','educ_sec','educ_tert',
                       'rural','age_yr','age_sq']].values.astype(float),
                 sub['Person_Day_Weight'].values.astype(float))
        beta = float(c2[1])
        mh   = wmean(sub[hcol], sub['Person_Day_Weight'])
        se   = beta/mh if mh > 0 else np.nan
        res_B[(act,gn)] = {'beta':beta,'mean_h':mh,'semi_elas':se,'n':len(sub)}
        print(f"  L_{act} {gn}: beta={beta:.3f}  mean={mh:.1f}h  se={se:.4f}  n={len(sub)}")

phi_raw = abs(res_B.get(('M','men'),{}).get('semi_elas', np.nan))
# phi_raw is the uncompensated Frisch elasticity from IV.
# No correction applied — the income-effect adjustment requires
# estimating the income elasticity of hours, which we have not done.
# NOTE: this gives phi ~ 0.03, which is low but honest.
# If you prefer a literature prior (0.3-0.5), set it explicitly here.
phi_est = float(phi_raw) if np.isfinite(phi_raw) else 0.3

# rho: not identified from this data. Use literature prior directly.
# The heuristic formula gave -0.10 which caused solver divergence.
# Set to -0.5 (standard macro calibration) and document explicitly.
rho_est = -0.5
print(f"\n  phi={phi_est:.4f} (raw IV, no correction)")
print(f"  rho={rho_est:.4f} (literature prior; not identified from data)")

# ═══════════════════════════════════════════════════════════════════════════
# Block C: Gender disutility weights
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Block C: gender disutility --")
mwa = wa[wa['married']==1].copy()

def cond_h(g, pcol, hcol):
    sub = mwa[(mwa['female']==g)&(mwa[pcol]==1)]
    return wmean(sub[hcol], sub['Person_Day_Weight'])

Lc_m = cond_h(0,'part_c','L_c'); Lc_f = cond_h(1,'part_c','L_c')
Ld_m = cond_h(0,'part_d','L_d'); Ld_f = cond_h(1,'part_d','L_d')
LM_m = cond_h(0,'part_M','L_M'); LM_f = cond_h(1,'part_M','L_M')

r_c = Lc_m/Lc_f; r_d = Ld_m/Ld_f; r_M = LM_m/LM_f
print(f"  Hours ratios (men/women, married): r_M={r_M:.4f}  r_d={r_d:.4f}  r_c={r_c:.4f}")

rho = rho_est
D_c_ratio = float(r_c**(1.0/rho))
D_d_ratio = float(r_d**(1.0/rho))

# D_M_f: derived from the solver's fixed-point equation, NOT from the FOC.
# At the fixed point the solver satisfies:
#   LM_m / LM_f = wage_gap^rho * D_M_f   (with D_M_m = 1)
# So to match the observed hours ratio r_M = LM_m/LM_f:
#   D_M_f = r_M / wage_gap^rho
# This is the ONLY formula consistent with the solver update map.
# The previous formula (wage_gap * r_M^(1/rho)) was derived from a
# different rearrangement of the FOC and gives the wrong sign.
D_f_M = float(r_M / (wage_gap ** rho))
D_m_c     = 1.0
D_f_c     = float(D_m_c / D_c_ratio)
# D_m_d from within-man care/domestic ratio (L5 condition).
# No clip applied — if the value is extreme it reflects a data problem,
# not a reason to override.
D_m_d     = float((Lc_m/Ld_m)**(1.0/rho)) if Ld_m > 0 else 1.0
D_f_d     = float(D_m_d / D_d_ratio)

print(f"  D^m_M=1.000  D^f_M={D_f_M:.4f}")
print(f"  D^m_c={D_m_c:.4f}  D^f_c={D_f_c:.4f}")
print(f"  D^m_d={D_m_d:.4f}  D^f_d={D_f_d:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Block A: PIGL
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Block A: PIGL --")
hh_eng = ca2[ca2['E']>50].dropna(subset=['share_c','share_d','share_x','log_E','hhsize']).copy()
hh_eng['hhsize_c'] = hh_eng['hhsize'].clip(1,12).astype(float)
X_eng = hh_eng[['log_E','hhsize_c','rural_hh']].values.astype(float)
w_eng = hh_eng['weight_hh'].values.astype(float)
print(f"  Sample: {len(hh_eng):,} households")

slopes = {}
for dep, lab in [('share_c','care'),('share_d','domestic'),('share_x','food')]:
    c_e = wls(hh_eng[dep].values.astype(float), X_eng, w_eng)
    slopes[dep] = float(c_e[1])
    bm = wmean(hh_eng[dep], hh_eng['weight_hh'])
    print(f"  {lab}: mean={bm:.4f}  slope_logE={slopes[dep]:.5f}")

# Normalise E to units of 1000 KSh so that (E/B)^eps is order-1 at the median.
# The PIGL model is scale-free in (E/B), so re-running with E_norm = E/1000
# gives the same model fit with tractable kappa magnitudes.
E_SCALE = 1000.0
hh_eng['log_E_norm'] = np.log(hh_eng['E'] / E_SCALE)
X_eng_norm = hh_eng[['log_E_norm','hhsize_c','rural_hh']].values.astype(float)
slopes_norm = {}
for dep, lab in [('share_c','care'),('share_d','domestic'),('share_x','food')]:
    c_e2 = wls(hh_eng[dep].values.astype(float), X_eng_norm, w_eng)
    slopes_norm[dep] = float(c_e2[1])

E_med        = float(hh_eng['E'].median() / E_SCALE)   # in 1000-KSh units
share_x_mean = wmean(hh_eng['share_x'], hh_eng['weight_hh'])
# eps from food Engel slope: no clip, report the raw estimate.
eps_est      = float(abs(slopes_norm['share_x']) / share_x_mean)
# beta_c and beta_d from data; beta_x is the residual so betas sum to 1.
# The model has three goods: x (everything non-service), c (care), d (domestic).
# We use observed care and domestic shares directly, and set beta_x = 1 - beta_c - beta_d.
beta_c = wmean(hh_eng['share_c'], hh_eng['weight_hh'])
beta_d = wmean(hh_eng['share_d'], hh_eng['weight_hh'])
beta_x = float(1.0 - beta_c - beta_d)   # residual: enforces sum = 1

# kappa from Engel slopes on care and domestic (the two identified service shares).
# kappa_x is set as the residual so that sum(kappa) = 0.
kappa_c = float(-slopes_norm['share_c']*(E_med**(1+eps_est))/eps_est)
kappa_d = float(-slopes_norm['share_d']*(E_med**(1+eps_est))/eps_est)
kappa_x = float(-(kappa_c + kappa_d))   # enforce sum = 0

slopes = {k: slopes_norm[k] for k in slopes_norm}
assert abs(beta_x + beta_c + beta_d - 1.0) < 1e-10, "betas must sum to 1"
assert abs(kappa_x + kappa_c + kappa_d) < 1e-10,    "kappas must sum to 0"
print(f"  eps={eps_est:.4f}  beta_x={beta_x:.4f}  beta_c={beta_c:.4f}  beta_d={beta_d:.4f}  sum={beta_x+beta_c+beta_d:.4f}")
print(f"  kappa_x={kappa_x:.4f}  kappa_c={kappa_c:.4f}  kappa_d={kappa_d:.4f}  sum={kappa_x+kappa_c+kappa_d:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Block D: omega, eta, A_home
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Block D: home production --")
zero_c = float((hh_eng['share_c']==0).mean())
zero_d = float((hh_eng['share_d']==0).mean())
# omega: not identified from this data without matching TUS hours to KCHS
# expenditure at household level (impossible — independent surveys).
# Set to literature priors and document explicitly.
# High zero-market shares confirm that omega > 0.5 is plausible,
# but the exact value is not pinned down.
omega_c_est = 0.75   # literature prior (Aguiar & Hurst 2007 range)
omega_d_est = 0.70   # literature prior
print(f"  Zero market care: {zero_c:.1%}  omega_c={omega_c_est:.3f} (literature prior)")
print(f"  Zero market dom:  {zero_d:.1%}  omega_d={omega_d_est:.3f} (literature prior)")

cty_tus = (wa.groupby('county').apply(lambda g: pd.Series({
    'Lc_mean': wmean(g['L_c'], g['Person_Day_Weight']),
    'Ld_mean': wmean(g['L_d'], g['Person_Day_Weight']),
})))
cty_joint = cty_tus.join(pd.DataFrame({'log_w': np.log(cty_wage)}), how='inner').dropna()
# eta from cross-county regression — report raw, no clip.
# Data gives ~1.0; literature gives 2.5. Both reported.
eta_c_est = eta_d_est = 2.5   # literature prior as baseline
if len(cty_joint) >= 20:
    for lab, lcol in [('care','Lc_mean'),('domestic','Ld_mean')]:
        yc = np.log(cty_joint[lcol].clip(lower=0.01).values.astype(float))
        Xc = cty_joint['log_w'].values.astype(float).reshape(-1,1)
        cc = wls(yc, Xc, np.ones(len(yc)))
        eta_data = float(-cc[1])
        print(f"  {lab}: d ln(L^i)/d ln(w)={cc[1]:.3f} => eta_data={eta_data:.3f} "
              f"(using literature prior {2.5})")

# A_home: not identified — set to 1 everywhere.
# Using county mean hours / national mean has no structural content.
cty_A_c = pd.Series(1.0, index=cty_tus.index).rename('A_c_proxy')
cty_A_d = pd.Series(1.0, index=cty_tus.index).rename('A_d_proxy')
print("  A_c_home = A_d_home = 1.0 everywhere (not identified from data)")

# ═══════════════════════════════════════════════════════════════════════════
# Block E: Participation
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- Block E: participation --")
part_tab = (wa.groupby(['female','high_skill'])
              .apply(lambda g: pd.Series({
                  'P': float(np.average(g['part_M'],
                                         weights=g['Person_Day_Weight'])),
                  'N': len(g)})))
print(part_tab.to_string())

P_m_low  = float(part_tab.loc[(0.0,0.0),'P'])
P_m_high = float(part_tab.loc[(0.0,1.0),'P'])
P_f_low  = float(part_tab.loc[(1.0,0.0),'P'])
P_f_high = float(part_tab.loc[(1.0,1.0),'P'])
dV       = phi_est * abs(educ_prem_tert)

def est_sigma(Pl, Ph, dv):
    dl = logit(Ph) - logit(Pl)
    # No clip — if sigma is extreme it means participation barely
    # varies with skill, which is a genuine data feature.
    if abs(dl) < 0.01:
        return np.nan   # not identified: no variation in participation by skill
    return float(dv / dl)

sigma_u_m = est_sigma(P_m_low, P_m_high, dV)
sigma_u_f = est_sigma(P_f_low, P_f_high, dV)
P_m_mean = float(np.average(wa[wa['female']==0]['part_M'],
                              weights=wa[wa['female']==0]['Person_Day_Weight']))
P_f_mean = float(np.average(wa[wa['female']==1]['part_M'],
                              weights=wa[wa['female']==1]['Person_Day_Weight']))
VP_m = phi_est * wmean(wa[wa['female']==0]['log_wage'],
                        wa[wa['female']==0]['Person_Day_Weight'])
VP_f = phi_est * wmean(wa[wa['female']==1]['log_wage'],
                        wa[wa['female']==1]['Person_Day_Weight'])
u_bar_m = float(VP_m - sigma_u_m * logit(P_m_mean))
u_bar_f = float(VP_f - sigma_u_f * logit(P_f_mean))
print(f"  sigma_u_m={sigma_u_m:.4f}  u_bar_m={u_bar_m:.4f}")
print(f"  sigma_u_f={sigma_u_f:.4f}  u_bar_f={u_bar_f:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# Build county fundamentals and export
# ═══════════════════════════════════════════════════════════════════════════
cty_df = pd.DataFrame({
    'w_ell':       cty_wage,
    'w_domestic':  cty_wage_dom,
    'w_care':      cty_wage_care,
    'share_rural': cty_rural,
    'N_tus':       cty_pop,
    'A_c_proxy':   cty_A_c,
    'A_d_proxy':   cty_A_d,
}).dropna(subset=['w_ell'])
cty_df['pc'] = cty_df['w_care']
cty_df['pd'] = cty_df['w_domestic']

params = {
    "eps_engel": round(eps_est,4),
    "beta_x":    round(float(beta_x),4),
    "beta_c":    round(float(beta_c),4),
    "beta_d":    round(float(beta_d),4),
    "kappa_x":   round(float(kappa_x),4),
    "kappa_c":   round(float(kappa_c),4),
    "kappa_d":   round(float(kappa_d),4),
    "phi":       round(float(phi_est),4),
    "rho":       round(float(rho_est),4),
    "D_M_m":     1.0,
    "D_c_m":     round(float(D_m_c),4),
    "D_d_m":     round(float(D_m_d),4),
    "D_M_f":     round(float(D_f_M),4),
    "D_c_f":     round(float(D_f_c),4),
    "D_d_f":     round(float(D_f_d),4),
    "omega_c":   round(float(omega_c_est),4),
    "omega_d":   round(float(omega_d_est),4),
    "eta_c":     round(float(eta_c_est),4),
    "eta_d":     round(float(eta_d_est),4),
    "sigma_u_m": round(float(sigma_u_m),4),
    "sigma_u_f": round(float(sigma_u_f),4),
    "u_bar_m":   round(float(u_bar_m),4),
    "u_bar_f":   round(float(u_bar_f),4),
    "wage_gap":  round(float(wage_gap),4),
}

print("\n" + "="*55)
print("CALIBRATED PARAMETERS")
print("="*55)
for k,v in params.items():
    print(f"  {k:<15} = {v}")

cty_df.reset_index().to_csv('output/county_fundamentals.csv', index=False)
with open('output/calibrated_params.json','w') as f:
    json.dump(params, f, indent=2)

# Auxiliary data for the calibration PDF
aux = {
    "n_tus": int(len(wa)),
    "n_kchs_hh": int(wa_p['interview__id'].nunique()),
    "n_wage_obs": int(wa_p['wage_hourly'].notna().sum()),
    "n_dom_workers": int(wa_ind[wa_ind['isic'].isin([9700,9800])].shape[0]),
    "E_median_ksh": round(float(E_med),1),
    "w_mean_ksh":   round(float(cty_wage.mean()),1),
    "w_dom_nat":    round(float(w_dom_nat),1),
    "w_care_nat":   round(float(w_care_nat),1),
    "wage_coef_female":    round(float(cw[1]),4),
    "wage_coef_secondary": round(float(cw[4]),4),
    "wage_coef_tertiary":  round(float(cw[5]),4),
    "wage_coef_rural":     round(float(cw[6]),4),
    "hours_LM_m": round(float(LM_m),2), "hours_LM_f": round(float(LM_f),2),
    "hours_Ld_m": round(float(Ld_m),2), "hours_Ld_f": round(float(Ld_f),2),
    "hours_Lc_m": round(float(Lc_m),2), "hours_Lc_f": round(float(Lc_f),2),
    "P_m_low":  round(P_m_low,4),  "P_m_high": round(P_m_high,4),
    "P_f_low":  round(P_f_low,4),  "P_f_high": round(P_f_high,4),
    "engel_slope_care":     round(float(slopes['share_c']),6),
    "engel_slope_domestic": round(float(slopes['share_d']),6),
    "engel_slope_food":     round(float(slopes['share_x']),6),
    "zero_care_pct":  round(float(zero_c),4),
    "zero_dom_pct":   round(float(zero_d),4),
}
with open('output/calibration_aux.json','w') as f:
    json.dump(aux, f, indent=2)

print("\n  Saved: calibrated_params.json  county_fundamentals.csv  calibration_aux.json")
