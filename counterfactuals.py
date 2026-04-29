#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
counterfactuals.py
==================

Runs three counterfactual policy experiments on the Kenya time-use spatial
equilibrium model and produces a non-specialist PDF report with maps.

Experiments
-----------
1. Closing the gender wage gap   (wage_gap = 1.00 in all counties).
2. Care price reduction          (p^c <- 0.7 * p^c in all counties).
3. Domestic price reduction      (p^d <- 0.7 * p^d in all counties).

Outputs (per experiment)
------------------------
For each county:
- Delta GDP            (proxied by N * E)
- Delta female participation rate
- Delta ratio  L^M_f / L^M_m

Output file: counterfactuals.pdf
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Local imports
sys.path.insert(0, str(Path(__file__).parent / 'time_use_app'))
from load_calibration import load
from spatial import solve_counties, calibrate_amenities

ROOT = Path(__file__).parent
TUA  = ROOT / 'time_use_app'

# ── Kenya county metadata (centroids and names) ───────────────────────────
with open(ROOT / 'kenya_counties.json') as f:
    KENYA = {int(k): v for k, v in json.load(f).items()}

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def participation_rate(LM_f_grid: np.ndarray, h_grid: np.ndarray,
                       w_ell: float, wage_gap: float,
                       sigma_u: float, u_bar: float, phi: float) -> float:
    """
    Female participation rate from the Block E logit equation:
      P^g(h) = 1 / (1 + exp((u_bar - V_P) / sigma_u))
    where V_P(h) = phi * w_g * h is the value of working at human capital h.
    Average across the h-grid.
    """
    w_f = wage_gap * w_ell
    V_P = phi * w_f * np.asarray(h_grid)
    # Logit
    P = 1.0 / (1.0 + np.exp((u_bar - V_P) / sigma_u))
    return float(np.mean(P))


def solve_one_scenario(counties, mp, h_grid, label: str, block_e: dict):
    """Solve the model for a given list of counties; return per-county metrics.
    block_e is a dict with keys sigma_u_f, u_bar_f (and male equivalents)."""
    print(f"  Solving scenario: {label}...")
    results = solve_counties(
        counties, mp, h_grid,
        wage_gap=counties[0].wage_gap,
        plot_convergence=False,
        solver_kw=dict(max_iter=20000, tol=1e-8, damping=0.15,
                       adapt_damping=True, verbose=False),
        verbose=False,
    )
    rows = []
    for c, r in zip(counties, results):
        E_mean = float(np.nanmean(r.E))
        gdp = float(c.N * E_mean)
        Pf = participation_rate(
            r.LM_f, h_grid,
            w_ell=c.w_ell, wage_gap=c.wage_gap,
            sigma_u=block_e['sigma_u_f'], u_bar=block_e['u_bar_f'], phi=mp.phi,
        )
        LM_m_mean = float(np.nanmean(r.LM_m))
        LM_f_mean = float(np.nanmean(r.LM_f))
        ratio_fm  = LM_f_mean / max(LM_m_mean, 1e-9)
        rows.append({
            'code': c.code,
            'county': c.label,
            'GDP': gdp,
            'Pf': Pf,
            'ratio_fm': ratio_fm,
            'LM_m': LM_m_mean,
            'LM_f': LM_f_mean,
            'E': E_mean,
            'V_star': r.V_rep,
            'converged': int(np.sum(r.converged)) == len(h_grid),
        })
    return pd.DataFrame(rows), results


# ──────────────────────────────────────────────────────────────────────────
# Map plotting
# ──────────────────────────────────────────────────────────────────────────

def plot_kenya_map(ax, df, value_col, title, fmt='{:+.1f}%',
                   cmap_name='RdBu_r', vcenter=0.0, units=''):
    """
    Stylised Kenya county map: each county shown as a circle at its centroid,
    coloured by `value_col`. Labels show value.
    """
    lats = np.array([KENYA[int(r.code)][1] for _, r in df.iterrows()])
    lons = np.array([KENYA[int(r.code)][2] for _, r in df.iterrows()])
    vals = df[value_col].values

    # Symmetric colour scale around vcenter
    vmax = max(abs(vals.max() - vcenter), abs(vals.min() - vcenter))
    if vmax == 0:
        vmax = 1e-6
    norm = TwoSlopeNorm(vmin=vcenter - vmax, vcenter=vcenter, vmax=vcenter + vmax)
    cmap = plt.get_cmap(cmap_name)

    # Draw rough Kenya outline (axis-aligned bounding box for context)
    ax.add_patch(mpatches.Rectangle(
        (33.7, -4.8), 8.5, 9.0, fill=False,
        edgecolor='lightgrey', linewidth=0.8, linestyle='--'))

    # Plot circles
    for lat, lon, v, code in zip(lats, lons, vals, df['code']):
        col = cmap(norm(v))
        ax.scatter(lon, lat, s=520, c=[col], edgecolors='black',
                   linewidths=0.6, zorder=3)
        ax.text(lon, lat, fmt.format(v), ha='center', va='center',
                fontsize=5.5, fontweight='bold', zorder=4,
                color='white' if abs(v - vcenter) > vmax * 0.4 else 'black')

    # Label only the largest counties to avoid clutter
    for _, row in df.iterrows():
        code = int(row.code)
        name, lat, lon = KENYA[code]
        # Hard-wired short labels for big/well-known counties only
        if name in ('Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret',
                    'Turkana', 'Mandera', 'Marsabit', 'Garissa', 'Tana River',
                    'Trans Nzoia', 'Kwale', 'Lamu', 'Kakamega'):
            ax.annotate(name, (lon, lat), xytext=(0, -14),
                        textcoords='offset points', fontsize=5,
                        ha='center', color='dimgrey', alpha=0.85)

    ax.set_xlim(33.5, 42.5)
    ax.set_ylim(-5.0, 4.5)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04,
                      pad=0.02, shrink=0.7)
    cb.ax.tick_params(labelsize=7)
    if units:
        cb.set_label(units, fontsize=7)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

COUNTY_NAMES = {k: v[0] for k, v in KENYA.items()}

def attach_metadata(counties):
    for c in counties:
        code = int(c.name.split('_')[1])
        c.code = code
        c.label = COUNTY_NAMES.get(code, c.name)


def main():
    print("=" * 70)
    print("Kenya Time-Use Counterfactuals")
    print("=" * 70)

    # ── Baseline ──
    print("\nLoading calibration ...")
    D_path = TUA / 'county_D_weights.csv'
    if D_path.exists():
        print(f"  Using county-specific D weights from {D_path.name}")
        mp_base, counties_base = load(
            params_path=str(TUA / 'calibrated_params.json'),
            county_path=str(TUA / 'county_fundamentals.csv'),
            D_county_path=str(D_path),
        )
    else:
        print(f"  Using shared (national) D weights")
        mp_base, counties_base = load(
            params_path=str(TUA / 'calibrated_params.json'),
            county_path=str(TUA / 'county_fundamentals.csv'),
        )
    # Block E parameters (not in ModelParams dataclass)
    with open(TUA / 'calibrated_params.json') as f:
        _full = json.load(f)
    BLOCK_E = {
        'sigma_u_m': _full['sigma_u_m'],
        'sigma_u_f': _full['sigma_u_f'],
        'u_bar_m':   _full['u_bar_m'],
        'u_bar_f':   _full['u_bar_f'],
    }
    attach_metadata(counties_base)
    h_grid = np.array([0.5, 1.0, 2.0, 3.0])

    print("\n[Baseline]")
    df_base, _ = solve_one_scenario(counties_base, mp_base, h_grid, 'baseline', BLOCK_E)
    print(f"  Mean GDP: {df_base['GDP'].mean():.2e}")
    print(f"  Mean female participation: {df_base['Pf'].mean():.3f}")
    print(f"  Mean LM_f/LM_m: {df_base['ratio_fm'].mean():.3f}")

    # ── Scenario 1: Wage gap closure ──
    print("\n[Scenario 1] Closing the gender wage gap (wage_gap = 1.00)")
    counties_wg = copy.deepcopy(counties_base)
    for c in counties_wg:
        c.wage_gap = 1.00
    attach_metadata(counties_wg)
    df_wg, _ = solve_one_scenario(counties_wg, mp_base, h_grid, 'wage_gap=1', BLOCK_E)

    # ── Scenario 2: Care price reduction ──
    print("\n[Scenario 2] Care price reduction (-30%)")
    counties_pc = copy.deepcopy(counties_base)
    for c in counties_pc:
        c._pc = c.pc * 0.7
    attach_metadata(counties_pc)
    df_pc, _ = solve_one_scenario(counties_pc, mp_base, h_grid, 'p_care -30%', BLOCK_E)

    # ── Scenario 3: Domestic price reduction ──
    print("\n[Scenario 3] Domestic price reduction (-30%)")
    counties_pd = copy.deepcopy(counties_base)
    for c in counties_pd:
        c._pd = c.pd * 0.7
    attach_metadata(counties_pd)
    df_pd, _ = solve_one_scenario(counties_pd, mp_base, h_grid, 'p_dom -30%', BLOCK_E)

    # ── Build delta dataframes ──
    def deltas(df_new, df_base):
        d = df_new.copy()
        d['dGDP_pct']   = 100.0 * (df_new['GDP']      / df_base['GDP']      - 1)
        d['dPf_pp']     = 100.0 * (df_new['Pf']       - df_base['Pf'])           # percentage points
        d['dratio_pp']  = 100.0 * (df_new['ratio_fm'] - df_base['ratio_fm'])     # percentage points
        return d

    d_wg = deltas(df_wg, df_base)
    d_pc = deltas(df_pc, df_base)
    d_pd = deltas(df_pd, df_base)

    # Save numerical results
    out_csv = ROOT / 'counterfactual_results.csv'
    out = pd.concat({
        'baseline': df_base.set_index('code'),
        'wage_gap_closed': d_wg.set_index('code')[['dGDP_pct','dPf_pp','dratio_pp']],
        'care_price_-30%': d_pc.set_index('code')[['dGDP_pct','dPf_pp','dratio_pp']],
        'dom_price_-30%':  d_pd.set_index('code')[['dGDP_pct','dPf_pp','dratio_pp']],
    }, axis=1)
    out.to_csv(out_csv)
    print(f"\nSaved CSV: {out_csv}")

    # ──────────────────────────────────────────────────────────────────
    # Build PDF report
    # ──────────────────────────────────────────────────────────────────
    pdf_path = ROOT / 'counterfactuals.pdf'
    print(f"Writing PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:

        # Cover page
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('#1a3a5c')
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
        ax.text(0.5, 0.66, 'Kenya Time-Use Model',
                ha='center', va='center', fontsize=28, fontweight='bold',
                color='white', transform=ax.transAxes)
        ax.text(0.5, 0.55, 'Three Counterfactual Policy Experiments',
                ha='center', va='center', fontsize=15, color='#aaccee',
                transform=ax.transAxes)
        ax.text(0.5, 0.45,
                '1. Closing the gender wage gap\n'
                '2. Reducing the price of care services by 30%\n'
                '3. Reducing the price of domestic services by 30%',
                ha='center', va='center', fontsize=11, color='white',
                transform=ax.transAxes, linespacing=1.6)
        ax.text(0.5, 0.18,
                'For each policy, we report county-level changes in:\n'
                '  • GDP (population × mean household expenditure)\n'
                '  • Female labour-force participation rate\n'
                '  • Ratio of female to male market hours',
                ha='center', va='center', fontsize=10, color='#cccccc',
                transform=ax.transAxes, linespacing=1.5)
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # ── Page 1: executive summary table ──
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Executive Summary — National Averages',
                     fontsize=14, fontweight='bold', pad=20, loc='left')

        summary = pd.DataFrame({
            'Scenario': [
                'Baseline',
                'Wage gap closed',
                'Care price -30%',
                'Domestic price -30%',
            ],
            'GDP (vs baseline)': [
                '—',
                f"{d_wg['dGDP_pct'].mean():+.2f}%",
                f"{d_pc['dGDP_pct'].mean():+.2f}%",
                f"{d_pd['dGDP_pct'].mean():+.2f}%",
            ],
            'Female participation (vs baseline)': [
                f"{df_base['Pf'].mean():.1%}",
                f"{d_wg['dPf_pp'].mean():+.2f} pp",
                f"{d_pc['dPf_pp'].mean():+.2f} pp",
                f"{d_pd['dPf_pp'].mean():+.2f} pp",
            ],
            'Female/male hours ratio (vs baseline)': [
                f"{df_base['ratio_fm'].mean():.3f}",
                f"{d_wg['dratio_pp'].mean():+.2f} pp",
                f"{d_pc['dratio_pp'].mean():+.2f} pp",
                f"{d_pd['dratio_pp'].mean():+.2f} pp",
            ],
        })
        tbl = ax.table(cellText=summary.values, colLabels=summary.columns,
                       loc='center', cellLoc='center', colLoc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
        for j in range(len(summary.columns)):
            tbl[0, j].set_facecolor('#2c5f8a')
            tbl[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(1, len(summary) + 1):
            bg = '#f0f4f8' if i % 2 == 0 else '#ffffff'
            for j in range(len(summary.columns)):
                tbl[i, j].set_facecolor(bg)

        # Plain-language footnote
        ax.text(0.05, 0.18,
                'How to read this:\n'
                '  • A positive % for GDP means the policy raises economic '
                'activity (households earn and spend more).\n'
                '  • "+1 pp" for participation means 1 additional percentage '
                'point of women in paid work.\n'
                '  • "+1 pp" for the hours ratio means women work 1% more '
                'paid hours relative to men.\n\n'
                'GDP is proxied as: county population × mean monthly household '
                'expenditure.\n'
                'Note: county populations come from the survey weights; they '
                'are not exact census numbers.',
                fontsize=9, va='top', transform=ax.transAxes,
                family='sans-serif', color='#333333',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffbe6',
                          edgecolor='#cca300'))
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # ── One page per scenario, three maps each ──
        scenarios = [
            ('Closing the gender wage gap',
             'In this scenario, women earn the same hourly wage as men.',
             d_wg),
            ('Reducing the price of care services by 30%',
             'In this scenario, market care services (childcare, eldercare) '
             'cost 30% less in every county.',
             d_pc),
            ('Reducing the price of domestic services by 30%',
             'In this scenario, market domestic services (cleaning, '
             'laundry, etc.) cost 30% less in every county.',
             d_pd),
        ]

        for title, desc, d in scenarios:
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
            fig.text(0.5, 0.93, desc, ha='center', fontsize=10,
                     color='#444444', style='italic')

            # Three maps in a row
            gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.18],
                                  left=0.04, right=0.96, top=0.88, bottom=0.04,
                                  wspace=0.12, hspace=0.0)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])

            plot_kenya_map(ax1, d, 'dGDP_pct',
                           'Change in county GDP', fmt='{:+.1f}',
                           cmap_name='RdBu_r', vcenter=0.0, units='%')
            plot_kenya_map(ax2, d, 'dPf_pp',
                           'Change in female participation', fmt='{:+.1f}',
                           cmap_name='PiYG', vcenter=0.0, units='percentage points')
            plot_kenya_map(ax3, d, 'dratio_pp',
                           'Change in F/M hours ratio', fmt='{:+.1f}',
                           cmap_name='PuOr_r', vcenter=0.0,
                           units='percentage points')

            # Bottom: short text summary
            ax_text = fig.add_subplot(gs[1, :])
            ax_text.axis('off')
            txt = (
                f"National averages:  "
                f"GDP {d['dGDP_pct'].mean():+.2f}%   |   "
                f"Female participation {d['dPf_pp'].mean():+.2f} pp   |   "
                f"F/M hours ratio {d['dratio_pp'].mean():+.2f} pp"
            )
            ax_text.text(0.5, 0.5, txt, ha='center', va='center', fontsize=11,
                         fontweight='bold', color='#1a3a5c',
                         transform=ax_text.transAxes,
                         bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='#e8f0f8', edgecolor='#2c5f8a'))

            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

        # ── Final page: methodology + caveats ──
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        ax.set_title('Methodology and caveats', fontsize=14,
                     fontweight='bold', pad=20, loc='left')
        body = (
            "How the model works:\n"
            "• Each Kenyan county is represented by a household with one "
            "man and one woman.\n"
            "• Household members allocate their time across paid market "
            "work, home care, home domestic work, and leisure.\n"
            "• Household members purchase goods, market care services, "
            "and market domestic services.\n"
            "• Each county faces an observed wage and observed prices "
            "for care and domestic services.\n"
            "• People are free to move across counties; in equilibrium, "
            "high-wage counties have lower amenity values and "
            "vice-versa.\n\n"
            "Counterfactual experiments:\n"
            "• A counterfactual changes one input (the wage gap, or a "
            "service price) and re-solves the entire spatial "
            "equilibrium.\n"
            "• We compare the new equilibrium to the baseline at the "
            "county level.\n\n"
            "Caveats to keep in mind:\n"
            "• GDP is proxied by population × mean expenditure. "
            "County populations come from survey weights, not the "
            "official census. Relative changes (the percentages) are "
            "more reliable than absolute levels.\n"
            "• Three counties have unreliable service prices due to "
            "data sparsity: Homa Bay (domestic), Kakamega (care), "
            "Tharaka-Nithi (domestic). Their results should be "
            "interpreted with caution.\n"
            "• The labour supply elasticity phi is set to 0.50, a "
            "literature prior. The data alone gives a smaller value "
            "(~0.03), which would shrink all counterfactual responses.\n"
            "• Migration in response to counterfactuals is included "
            "but uses a free parameter (sigma_mig = 1.0) that has not "
            "been calibrated to actual Kenyan migration data."
        )
        ax.text(0.04, 0.86, body, fontsize=10, va='top', linespacing=1.5,
                transform=ax.transAxes, family='sans-serif', color='#333333')
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    print(f"\n✓ Done. PDF: {pdf_path}")
    print(f"✓ CSV: {out_csv}")


if __name__ == '__main__':
    main()
