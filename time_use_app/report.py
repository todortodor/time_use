#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py  –  Run full simulation and write all output to a single PDF.

Sections:
  1. Model parameters
  2. County economic fundamentals (Table 1)
  3. Gender division of labour (Table 2)
  4. Amenity ranking (Table 3)
  5. Cross-county correlations
  6. Counterfactual: 20% care subsidy in 5 poorest counties (Table 4)
  7. Per-county pages: hours profiles (men vs women) + diagnostic text

Run with:
    python report.py
Output: simulation_report.pdf
"""
from __future__ import annotations

import copy, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table

from load_calibration import load
from spatial import solve_counties, calibrate_amenities, counterfactual

# ── Kenya county names ────────────────────────────────────────────────────
COUNTY_NAMES = {
    1:"Mombasa",2:"Kwale",3:"Kilifi",4:"Tana River",5:"Lamu",
    6:"Taita Taveta",7:"Garissa",8:"Wajir",9:"Mandera",10:"Marsabit",
    11:"Isiolo",12:"Meru",13:"Tharaka-Nithi",14:"Embu",15:"Kitui",
    16:"Machakos",17:"Makueni",18:"Nyandarua",19:"Nyeri",20:"Kirinyaga",
    21:"Murang'a",22:"Kiambu",23:"Turkana",24:"West Pokot",25:"Samburu",
    26:"Trans Nzoia",27:"Uasin Gishu",28:"Elgeyo-Marakwet",29:"Nandi",
    30:"Baringo",31:"Laikipia",32:"Nakuru",33:"Narok",34:"Kajiado",
    35:"Kericho",36:"Bomet",37:"Kakamega",38:"Vihiga",39:"Bungoma",
    40:"Busia",41:"Siaya",42:"Kisumu",43:"Homa Bay",44:"Migori",
    45:"Kisii",46:"Nyamira",47:"Nairobi",
}

# ── Helpers ───────────────────────────────────────────────────────────────

def add_table(ax, col_labels, rows, col_widths=None, title=None,
              header_color='#2c5f8a', row_colors=('#f0f4f8','#ffffff')):
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6, loc='left')
    n_rows = len(rows)
    n_cols = len(col_labels)
    all_rows = [col_labels] + rows
    t = ax.table(cellText=all_rows, loc='center', cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(7.5)
    t.scale(1, 1.4)
    for j in range(n_cols):
        t[0, j].set_facecolor(header_color)
        t[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, n_rows + 1):
        bg = row_colors[(i - 1) % 2]
        for j in range(n_cols):
            t[i, j].set_facecolor(bg)
    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(n_rows + 1):
                t[i, j].set_width(w)

def diag_text(c, r, N_h, n_conv):
    """Build diagnostic bullet string for one county."""
    W_NAT=92.; PC_NAT=68.7; PD_NAT=34.9; AC_NAT=1.0; AD_NAT=1.0
    w_ksh  = c.w_ell*1000; pc_ksh=c.pc*1000; pd_ksh=c.pd*1000
    Ac=c.A_c_home; Ad=c.A_d_home
    LM_m=float(np.nanmean(r.LM_m)); LM_f=float(np.nanmean(r.LM_f))
    Lc_m=float(np.nanmean(r.Lc_m)); Lc_f=float(np.nanmean(r.Lc_f))
    Ld_m=float(np.nanmean(r.Ld_m)); Ld_f=float(np.nanmean(r.Ld_f))
    E_m =float(np.nanmean(r.E))*1000

    w_tag  = "high-wage"  if w_ksh>W_NAT*1.2  else "low-wage"  if w_ksh<W_NAT*0.8  else "average-wage"
    pc_tag = "expensive care"    if pc_ksh>PC_NAT*1.2 else "cheap care"    if pc_ksh<PC_NAT*0.8 else "average care price"
    pd_tag = "expensive domestic" if pd_ksh>PD_NAT*1.2 else "cheap domestic" if pd_ksh<PD_NAT*0.8 else "average domestic price"
    xi_tag = "above-average amenity" if c.xi>0.05 else "below-average amenity" if c.xi<-0.05 else "average amenity"
    conv_tag = f"fully converged ({n_conv}/{N_h})" if n_conv==N_h else \
               f"partial convergence ({n_conv}/{N_h}) — interpret with caution" if n_conv>0 else \
               f"did not converge ({n_conv}/{N_h}) — results unreliable"

    lines = [
        f"Economy: {w_tag} ({w_ksh:.0f} vs {W_NAT:.0f} KSh/hr national).",
        f"Prices: {pc_tag} ({pc_ksh:.0f} KSh/hr), {pd_tag} ({pd_ksh:.0f} KSh/hr).",
        f"Home TFP: A_c={Ac:.2f} ({'above' if Ac>1.1 else 'below' if Ac<0.9 else 'at'} national), "
        f"A_d={Ad:.2f}.",
        f"Market work: men {LM_m:.2f}h vs women {LM_f:.2f}h/wk "
        f"(gap {LM_m-LM_f:+.2f}h).",
        f"Domestic: women {Ld_f:.2f}h vs men {Ld_m:.2f}h/wk "
        f"(gap {Ld_f-Ld_m:+.2f}h).",
        f"Care: women {Lc_f:.2f}h vs men {Lc_m:.2f}h/wk.",
        f"Mean HH expenditure: {E_m:.0f} KSh/month.",
        f"Spatial: V*={r.V_rep:.3f}, xi={c.xi:.3f} ({xi_tag}).",
        f"Solver: {conv_tag}.",
    ]
    return "\n".join(f"• {l}" for l in lines)


# ═══════════════════════════════════════════════════════════════════════════
# Run simulation
# ═══════════════════════════════════════════════════════════════════════════
print("Loading calibration ...")
mp, counties = load(params_path='calibrated_params.json',
                    county_path='county_fundamentals.csv')
for c in counties:
    code = int(c.name.split('_')[1])
    c.label = COUNTY_NAMES.get(code, c.name)
    c.code  = code

h_grid   = np.array([0.5, 1.0, 2.0, 3.0])
wage_gap = counties[0].wage_gap

print("Solving households across 47 counties ...")
results = solve_counties(
    counties, mp, h_grid, wage_gap=wage_gap, agg="mean",
    plot_convergence=False,
    solver_kw=dict(max_iter=20000, tol=1e-8, damping=0.15,
                   adapt_damping=True, verbose=False),
    verbose=True,
)
U_bar = calibrate_amenities(counties, results)
print(f"U_bar = {U_bar:.4f}")

# ── Build summary dataframe ───────────────────────────────────────────────
rows_data = []
for c, r in zip(counties, results):
    n_conv = int(np.sum(r.converged))
    rows_data.append({
        'code':c.code,'county':c.label,
        'w_ksh':round(c.w_ell*1000,1), 'pc_ksh':round(c.pc*1000,1),
        'pd_ksh':round(c.pd*1000,1),
        'A_c':round(c.A_c_home,3),'A_d':round(c.A_d_home,3),
        'V_star':round(r.V_rep,4),'xi':round(c.xi,4),
        'E_mean':round(float(np.nanmean(r.E))*1000,0),
        'LM_m':round(float(np.nanmean(r.LM_m)),3),
        'LM_f':round(float(np.nanmean(r.LM_f)),3),
        'Lc_m':round(float(np.nanmean(r.Lc_m)),3),
        'Lc_f':round(float(np.nanmean(r.Lc_f)),3),
        'Ld_m':round(float(np.nanmean(r.Ld_m)),3),
        'Ld_f':round(float(np.nanmean(r.Ld_f)),3),
        'gap_M':round(float(np.nanmean(r.LM_m))-float(np.nanmean(r.LM_f)),3),
        'gap_d':round(float(np.nanmean(r.Ld_f))-float(np.nanmean(r.Ld_m)),3),
        'gap_c':round(float(np.nanmean(r.Lc_f))-float(np.nanmean(r.Lc_m)),3),
        'n_conv':n_conv,
    })
df = pd.DataFrame(rows_data).sort_values('w_ksh')

# ── Counterfactual ────────────────────────────────────────────────────────
print("Running counterfactual ...")
sorted_idx  = np.argsort([c.w_ell for c in counties])
subsidy_idx = set(sorted_idx[:5].tolist())
counties_new = []
for k, c in enumerate(counties):
    cn = copy.deepcopy(c); cn._pc = c.pc*(0.8 if k in subsidy_idx else 1.0)
    counties_new.append(cn)
cf = counterfactual(
    counties, counties_new, mp, h_grid, wage_gap=wage_gap,
    reequilibrate=True, plot_convergence=False,
    solver_kw=dict(max_iter=20000, tol=1e-8, damping=0.15,
                   adapt_damping=True, verbose=False),
    verbose=False,
)
df_cf = pd.DataFrame({
    'county': [COUNTY_NAMES.get(counties[k].code, cf.county_names[k])
               for k in range(len(cf.county_names))],
    'subsidy':['YES' if k in subsidy_idx else '' for k in range(len(cf.county_names))],
    'w_ksh':  [round(counties[k].w_ell*1000,1) for k in range(len(cf.county_names))],
    'dV':     np.round(cf.delta_V,4),
    'N_new':  np.round(cf.N_new,1)  if cf.N_new   is not None else [np.nan]*len(counties),
    'dN':     np.round(cf.delta_N,1) if cf.delta_N is not None else [np.nan]*len(counties),
}).sort_values('dV', ascending=False)

# ═══════════════════════════════════════════════════════════════════════════
# Write PDF
# ═══════════════════════════════════════════════════════════════════════════
OUT = 'simulation_report.pdf'
print(f"Writing {OUT} ...")

with PdfPages(OUT) as pdf:

    # ── Cover page ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('#1a3a5c')
    ax = fig.add_axes([0,0,1,1]); ax.axis('off')
    ax.text(0.5, 0.65, 'Kenya Time-Use Model',
            ha='center', va='center', fontsize=28, fontweight='bold',
            color='white', transform=ax.transAxes)
    ax.text(0.5, 0.52, 'Spatial Equilibrium Simulation Report',
            ha='center', va='center', fontsize=16, color='#aaccee',
            transform=ax.transAxes)
    ax.text(0.5, 0.42, f'47 counties  |  h-grid: {list(h_grid)}  |  U_bar = {U_bar:.4f}',
            ha='center', va='center', fontsize=11, color='#88aacc',
            transform=ax.transAxes)
    with open('calibrated_params.json') as f:
        params = json.load(f)
    param_str = "  |  ".join(f"{k}={v}" for k,v in params.items()
                              if k not in ['sigma_u_m','sigma_u_f','u_bar_m','u_bar_f'])
    for i, chunk in enumerate(textwrap.wrap(param_str, 90)):
        ax.text(0.5, 0.30 - i*0.04, chunk, ha='center', va='center',
                fontsize=7, color='#aabbcc', transform=ax.transAxes,
                family='monospace')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Page: Model parameters ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    p_rows = [[k, str(v)] for k, v in params.items()]
    add_table(ax, ['Parameter','Value'], p_rows,
              title='Model Parameters (calibrated_params.json)')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Page: Table 1 – economic fundamentals ─────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    t1_rows = [[row['county'], f"{row['w_ksh']:.1f}", f"{row['pc_ksh']:.1f}",
                f"{row['pd_ksh']:.1f}", f"{row['A_c']:.3f}", f"{row['A_d']:.3f}",
                f"{row['V_star']:.4f}", f"{row['xi']:+.4f}", f"{row['E_mean']:.0f}"]
               for _, row in df.iterrows()]
    add_table(ax,
              ['County','Wage\n(KSh/hr)','p_care\n(KSh/hr)','p_dom\n(KSh/hr)',
               'A_c_home','A_d_home','V*','ξ','E_mean\n(KSh/mo)'],
              t1_rows,
              title='Table 1: County Economic Fundamentals (sorted by wage)')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Page: Table 2 – gender division of labour ─────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    t2_rows = [[row['county'],
                f"{row['LM_m']:.3f}", f"{row['LM_f']:.3f}",
                f"{row['Ld_m']:.3f}", f"{row['Ld_f']:.3f}",
                f"{row['Lc_m']:.3f}", f"{row['Lc_f']:.3f}",
                f"{row['gap_M']:+.3f}", f"{row['gap_d']:+.3f}", f"{row['gap_c']:+.3f}"]
               for _, row in df.iterrows()]
    add_table(ax,
              ['County','LM_m','LM_f','Ld_m','Ld_f','Lc_m','Lc_f',
               'Gap_M\nm−f','Gap_d\nf−m','Gap_c\nf−m'],
              t2_rows,
              title='Table 2: Gender Division of Labour (mean h/wk across h-grid)')
    fig.text(0.02, 0.02,
             'Gap_M = LM_m − LM_f   |   Gap_d = Ld_f − Ld_m   |   Gap_c = Lc_f − Lc_m',
             fontsize=8, color='grey')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Page: Table 3 – amenity ranking ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    df_xi = df.sort_values('xi', ascending=False).reset_index(drop=True)
    t3_rows = [[str(i+1), row['county'], f"{row['xi']:+.4f}",
                f"{row['V_star']:.4f}", f"{row['w_ksh']:.1f}",
                '▲' if row['xi']>0 else '▼']
               for i, (_, row) in enumerate(df_xi.iterrows())]
    add_table(ax, ['Rank','County','ξ','V*','Wage\n(KSh/hr)',''],
              t3_rows,
              title='Table 3: Amenity Ranking  (ξ = Ū − V*;  ▲ above avg, ▼ below avg)')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Page: cross-county correlations ───────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('Cross-County Correlations', fontsize=13, fontweight='bold')

    pairs = [
        ('w_ksh','V_star','Wage vs V*'),
        ('w_ksh','xi',    'Wage vs Amenity ξ'),
        ('w_ksh','gap_M', 'Wage vs Market-hours gap'),
        ('w_ksh','gap_d', 'Wage vs Domestic-hours gap'),
        ('A_c',  'Lc_f',  'A_c_home vs Women care hours'),
        ('pc_ksh','gap_M','Care price vs Market-hours gap'),
    ]
    for idx, (xc, yc, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        x = df[xc]; y = df[yc]
        ax.scatter(x, y, s=18, alpha=0.7, color='#2c5f8a')
        m, b = np.polyfit(x, y, 1)
        xl = np.linspace(x.min(), x.max(), 50)
        ax.plot(xl, m*xl+b, 'r-', lw=1)
        r = x.corr(y)
        ax.set_title(f'{title}\nr={r:+.3f}', fontsize=8)
        ax.set_xlabel(xc, fontsize=7); ax.set_ylabel(yc, fontsize=7)
        ax.tick_params(labelsize=6)
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Page: Table 4 – counterfactual ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    t4_rows = [[row['county'], row['subsidy'], f"{row['w_ksh']:.1f}",
                f"{row['dV']:+.4f}", f"{row['N_new']:.1f}", f"{row['dN']:+.1f}"]
               for _, row in df_cf.iterrows()]
    subsidy_names = ', '.join(counties[k].label for k in subsidy_idx)
    add_table(ax,
              ['County','Subsidy','Wage\n(KSh/hr)','ΔV*','N_new','ΔN'],
              t4_rows,
              title=f'Table 4: Counterfactual — 20% care price subsidy in 5 poorest counties\n'
                    f'Subsidised: {subsidy_names}\n'
                    f'Ū = {U_bar:.4f}  →  Ū\' = {cf.U_bar_new:.4f}  '
                    f'(Δ = {cf.U_bar_new-U_bar:+.4f})')
    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    # ── Per-county pages ──────────────────────────────────────────────────
    COLORS = {'LM_m':'#1f77b4','LM_f':'#aec7e8',
              'Lc_m':'#ff7f0e','Lc_f':'#ffbb78',
              'Ld_m':'#2ca02c','Ld_f':'#98df8a'}

    for c, r in zip(counties, results):
        n_conv = int(np.sum(r.converged))
        conv_rate = n_conv / len(h_grid)
        col_title = '#2c5f8a' if conv_rate==1 else '#cc7700' if conv_rate>0 else '#aa0000'

        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(
            f"{c.label}  |  Wage: {c.w_ell*1000:.0f} KSh/hr  |  "
            f"V*: {r.V_rep:.3f}  |  ξ: {c.xi:.3f}  |  "
            f"Converged: {n_conv}/{len(h_grid)}",
            fontsize=11, fontweight='bold', color=col_title)

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                               top=0.88, bottom=0.32)

        # Panel 1: market hours
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(h_grid, r.LM_m, 'o-', color=COLORS['LM_m'], lw=2, label='Man')
        ax1.plot(h_grid, r.LM_f, 's--',color=COLORS['LM_f'], lw=2, label='Woman')
        ax1.set_title('Market hours $L^M$', fontsize=9)
        ax1.set_xlabel('h', fontsize=8); ax1.set_ylabel('h/wk', fontsize=8)
        ax1.legend(fontsize=7); ax1.grid(alpha=0.3); ax1.tick_params(labelsize=7)

        # Panel 2: domestic hours
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(h_grid, r.Ld_m, 'o-', color=COLORS['Ld_m'], lw=2, label='Man')
        ax2.plot(h_grid, r.Ld_f, 's--',color=COLORS['Ld_f'], lw=2, label='Woman')
        ax2.set_title('Domestic hours $L^d$', fontsize=9)
        ax2.set_xlabel('h', fontsize=8); ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3); ax2.tick_params(labelsize=7)

        # Panel 3: care hours
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(h_grid, r.Lc_m, 'o-', color=COLORS['Lc_m'], lw=2, label='Man')
        ax3.plot(h_grid, r.Lc_f, 's--',color=COLORS['Lc_f'], lw=2, label='Woman')
        ax3.set_title('Care hours $L^c$', fontsize=9)
        ax3.set_xlabel('h', fontsize=8); ax3.legend(fontsize=7)
        ax3.grid(alpha=0.3); ax3.tick_params(labelsize=7)

        # Panel 4: gender gaps across h
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(h_grid, r.LM_m - r.LM_f, 'o-', color='#1f77b4', label='ΔL^M')
        ax4.plot(h_grid, r.Ld_f - r.Ld_m, 's--',color='#2ca02c', label='ΔL^d')
        ax4.plot(h_grid, r.Lc_f - r.Lc_m, '^:', color='#ff7f0e', label='ΔL^c')
        ax4.axhline(0, color='k', lw=0.7, ls=':')
        ax4.set_title('Gender gaps vs h', fontsize=9)
        ax4.set_xlabel('h', fontsize=8); ax4.set_ylabel('Δ h/wk', fontsize=8)
        ax4.legend(fontsize=7); ax4.grid(alpha=0.3); ax4.tick_params(labelsize=7)

        # Panel 5: total hours by gender
        ax5 = fig.add_subplot(gs[1, 1])
        total_m = r.LM_m + r.Lc_m + r.Ld_m
        total_f = r.LM_f + r.Lc_f + r.Ld_f
        ax5.plot(h_grid, total_m, 'o-', color='#1f77b4', lw=2, label='Man')
        ax5.plot(h_grid, total_f, 's--',color='#d62728', lw=2, label='Woman')
        ax5.set_title('Total work hours', fontsize=9)
        ax5.set_xlabel('h', fontsize=8); ax5.set_ylabel('h/wk', fontsize=8)
        ax5.legend(fontsize=7); ax5.grid(alpha=0.3); ax5.tick_params(labelsize=7)

        # Panel 6: expenditure
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(h_grid, r.E*1000, 'o-', color='#9467bd', lw=2)
        ax6.set_title('Household expenditure', fontsize=9)
        ax6.set_xlabel('h', fontsize=8); ax6.set_ylabel('KSh/month', fontsize=8)
        ax6.grid(alpha=0.3); ax6.tick_params(labelsize=7)

        # Diagnostic text box
        txt = diag_text(c, r, len(h_grid), n_conv)
        fig.text(0.02, 0.28, txt, fontsize=7.5, va='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8',
                           edgecolor='#2c5f8a', alpha=0.9),
                 wrap=True)

        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

print(f"Done. Saved: {OUT}")
