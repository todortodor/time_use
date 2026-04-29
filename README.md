# Kenya Time-Use Model

Spatial equilibrium model of household time allocation across 47 Kenyan counties.

## Folder structure

```
project/
├── README.md                       (this file)
├── model.tex                       Model specification document
├── model.pdf                       Compiled model document
├── counterfactuals.py              Counterfactual experiments script
├── counterfactuals.pdf             Counterfactual results (non-specialist report)
├── counterfactual_results.csv      Per-county numerical results
├── calibrate_food_extension.py     Food/non-food calibration
├── calibrated_food_params.json     Calibrated food extension parameters
├── kenya_counties.json             County centroids for the maps
└── time_use_app/                   Main model code
    ├── calibrated_params.json      Main calibration (phi=0.50, D_M_f=1.344)
    ├── county_fundamentals.csv     Per-county wages, prices, populations
    ├── calibrate.py                Main calibration script (from microdata)
    ├── classes.py                  ModelParams + Household dataclasses
    ├── functions.py                CES, PIGL, root-finding helpers
    ├── solver.py                   6-equation fixed-point solver
    ├── spatial.py                  47-county solver + counterfactual
    ├── load_calibration.py         Loader for params + county data
    ├── county.py                   County dataclass
    ├── report.py                   Per-county simulation report (PDF)
    └── simulation_report.pdf       (output of report.py)
```

## How to use

1. **Run counterfactuals**: `python counterfactuals.py`
   Produces `counterfactuals.pdf` with three policy experiments and
   nine choropleth maps (3 indicators x 3 scenarios).

2. **Inspect calibration**: open `model.pdf` for the full equation
   listing and parameter values.

3. **Re-run calibration from raw microdata** (requires the KCHS and
   TUS files):
   ```
   cd time_use_app
   python calibrate.py
   ```

4. **Generate per-county simulation report** (47 county pages with
   plots): `cd time_use_app && python report.py`. Slow (~10 min).

## Current calibration

| Parameter | Value | Source |
|-----------|-------|--------|
| eps_engel | 0.291 | KCHS Engel curve |
| beta_x    | 0.889 | KCHS share, residual goods |
| beta_c    | 0.055 | KCHS market care share |
| beta_d    | 0.056 | KCHS market domestic share |
| kappa_x   | +0.069 | KCHS Engel slope |
| kappa_c   | -0.034 | KCHS Engel slope |
| kappa_d   | -0.035 | KCHS Engel slope |
| phi       | 0.50  | Literature prior (was 0.032 from raw IV) |
| rho       | -0.50 | Literature prior |
| D_M_m     | 1.0   | Normalisation |
| D_M_f     | 1.344 | Solver fixed-point match (formerly wrong: 0.383) |
| D_c_m     | 1.0   | Normalisation |
| D_c_f     | 0.766 | TUS hours ratio |
| D_d_m     | 1.632 | TUS hours ratio |
| D_d_f     | 0.337 | TUS hours ratio (sign issue, see model.pdf) |
| omega_c   | 0.75  | Literature prior |
| omega_d   | 0.70  | Literature prior |
| eta_c     | 2.5   | Literature prior |
| eta_d     | 2.5   | Literature prior |
| wage_gap  | 0.831 | KCHS wage regression |

See `model.pdf` Section 4 for the calibration strategy and
`time_use_app/calibrated_params.json` for the full machine-readable
file.

## Food/non-food extension status

The food/non-food split is **calibrated** (see
`calibrated_food_params.json`) but not yet integrated into the solver.
The 8-equation extension to the solver is a follow-on task. Calibrated
quantities for the extension:

| Parameter   | Value  |
|-------------|--------|
| beta_xf     | 0.569  |
| beta_xn     | 0.432  |
| kappa_xf    | +0.283 |
| kappa_xn    | -0.283 |
| D_xf_m      | 1.0    |
| D_xf_f      | 3.314  |
| omega_xf    | 0.75   |
| eta_xf      | 2.5    |

Food prep hours (TUS code t231):
- Men:   11.3 h/wk conditional, 23.7% participate
- Women: 20.6 h/wk conditional, 90.7% participate

## Counterfactual results (national averages)

| Scenario              | GDP    | Female participation | F/M hours ratio |
|-----------------------|--------|----------------------|-----------------|
| Wage gap closed       | +3.83% | +0.07 pp             | +6.56 pp        |
| Care price -30%       | +0.01% |  0.00 pp             |  0.00 pp        |
| Domestic price -30%   | +0.03% |  0.00 pp             |  0.00 pp        |

Care and domestic price reductions give near-zero effects because the
expenditure shares of those services are tiny (~5% of budget) and
service demand is income-inelastic. The wage gap closure is the only
counterfactual with a clear macroeconomic effect.

## Known limitations

- County populations (`N_tus` in `county_fundamentals.csv`) are raw
  TUS person-day weight sums (~10^8), not census populations. This
  affects absolute GDP levels; relative changes are unaffected.
- Three counties have unreliable service prices: Homa Bay (p_d=3),
  Kakamega (p_c=351), Tharaka-Nithi (p_d=110).
- Home TFP A_c, A_d set to 1 everywhere (not identified).
- sigma_mig = 1.0 in the spatial migration equation is uncalibrated.
- The Block E parameters (sigma_u_g, u_bar_g) were estimated at the
  old phi=0.032; with the new phi=0.50 they would shift.

## Bokeh app

A two-panel side-by-side comparison app lives at
`time_use_app/main.py`.

### Run

```bash
cd <project_root>
bokeh serve time_use_app/ --show
```

The app opens at `http://localhost:5006/time_use_app`.

### Features

- **Two scenario columns (A / B)** with identical controls, allowing you
  to compare two parameter configurations side-by-side.
- **Kenya map at the top** of each column. Click a county to load its
  parameters into the editor below. Color-code by wage, baseline female
  participation, or last counterfactual ΔGDP.
- **Global parameters editor** (18 fields): PIGL, CES, ρ, φ, wage_gap,
  σ_u, ū. These are shared across all counties in that scenario.
- **County-specific parameters editor** (12 fields): w_ell, p_c, p_d,
  A_c, A_d, N, and the six D weights. Loaded from the calibrated CSVs
  on first display, editable thereafter.
- **Solve buttons**:
  - "Solve selected county" (~0.1 s) — re-solves only the displayed
    county and updates the per-h plots.
  - "Solve all 47 counties" (~3-5 s, on the Spatial tab) — re-solves
    every county; required before counterfactuals.
- **Counterfactual buttons** on the Counterfactuals tab: wage_gap = 1,
  p^c × 0.7, p^d × 0.7. Each updates both the summary table and the
  nine choropleth maps (3 indicators × 3 scenarios).
- **Tabs**:
  - Per-h plots (selected county): hours, gaps, shares, home/market.
  - Spatial: V* vs wage, ξ ranking, gender hours-ratio.
  - Counterfactuals: summary table + 9 maps.

### Caveats

- Page is wide (~2400 px). Use a monitor ≥ 1920 px or expect horizontal
  scrolling.
- The food/non-food extension is calibrated (see
  `calibrated_food_params.json`) but not integrated into the 6-equation
  solver yet. Activation pending the 8-equation solver rewrite. The
  food parameters can still be inspected by reading the JSON.
