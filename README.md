# Kenya Time-Use Model — 4-good full implementation

This is the full-document implementation of the Kenya time-use spatial
equilibrium model with four goods (food, non-food, care, domestic), four
participation states per household, market clearing, and population
re-allocation under counterfactuals.

The earlier project under `existing_project/` implemented the same model
with three goods (the food/non-food split was held in a side JSON but
never integrated into the solver). This is a clean rebuild of the full
specification, calibrated end-to-end from microdata.

## Files

```
project/
├── README.md                this file
├── classes.py               ModelParams, Household, County dataclasses
├── solver_functions.py      pure functions: CES, PIGL, household solver,
│                            participation logit, spatial loop, migration
├── calibrate.py             calibration script — reads .dta, writes outputs
├── main.py                  Bokeh app
├── calibrated_params.json   global structural parameters
└── county_data.csv          47 county fundamentals
```

## How to run

### Calibrate from microdata

```bash
cd project
python calibrate.py
```

Reads `time_use_final_ver13.dta`, `individuals_microdata.dta`,
`consumption_aggregate_microdata.dta`, and `nonfood_items_microdata.dta`
from `/mnt/user-data/uploads/`, and writes `calibrated_params.json` and
`county_data.csv`. Runs end-to-end in roughly 60 seconds: about 2 seconds
for the data manipulation blocks and the rest for two passes of solving
all 47 counties (initial calibration + structural refinement of
participation shifters).

### Launch the Bokeh app

```bash
bokeh serve --show project/main.py
```

Two side-by-side scenario columns. In each: a Kenya map, parameter
inputs, Solve / Reset / Solve-all buttons, three counterfactual buttons,
and three tabs (per-h plots, spatial summary, counterfactuals).

Tap a county on the map to load its parameters; edit any input and click
"Solve selected county" (200 ms) or "Solve all 47 counties" (~30–35 s)
to update the spatial summary and enable counterfactuals. Each
counterfactual takes another ~30–35 s.

## Sanity-check values

Headline calibrated quantities, with the targets I expected at calibration time:

| Parameter   | Value    | Target              | Source                                       |
| ----------- | -------- | ------------------- | -------------------------------------------- |
| `eps_engel` | 0.291    | ≈ 0.29              | KCHS food Engel curvature                    |
| `wage_gap`  | 0.831    | ≈ 0.83              | KCHS Mincer regression                       |
| `D_M_f`     | 1.343    | ≈ 1.34              | TUS market hours, married subsample          |
| `D_xf_f`    | 0.269    | (see correction below) | TUS food-prep hours                       |
| `D_c_f`     | 0.766    |                     | TUS home-care hours                          |
| `D_d_m`     | 1.630    |                     | Intra-male home-domestic vs. home-care ratio |
| `D_d_f`     | 0.336    |                     | TUS home-domestic hours                      |
| `phi`       | 0.5      | literature prior    | Frisch elasticity (KCHS IV biased downward)  |
| `rho`       | −0.5     | literature prior    | CES disutility curvature                     |
| `Ubar`      | 3.4368   |                     | Population-weighted V*                       |
| `V*` range  | [3.40, 3.56] |                 | After spatial solve                          |
| `ξ` range   | [−0.12, 0.04] |                | After amenity calibration                    |
| County wages | [30, 261] KSh/hr |             | KCHS labour module                           |

Participation rates after the structural refinement of `(σ_u, ū)`:
- Implied national `P_m = 0.58` against observed `P_m_obs = 0.55`
- Implied national `P_f = 0.39` against observed `P_f_obs = 0.34`

Both within roughly 5 percentage points of observed — close enough that
the counterfactuals reflect the data-anchored elasticity rather than
arbitrary calibration choices.

### `D_xf_f` — correction relative to the existing food JSON

The existing project's `calibrated_food_params.json` has `D_xf_f = 3.314`,
calibrated via `D_xf_f = D_xf_m × (L^xf_m / L^xf_f)^(1/ρ)`. This sign is
**wrong** for solver consistency: my solver's update map (matching
eq. 22 of the document) implies

```
L^j_m / L^j_f = (D^m_j / D^f_j)^ρ        ⇒        D^f_j = D^m_j × r_j^(−1/ρ)
```

where `r_j = L^j_m / L^j_f` is the observed gender ratio. With
`L^xf_m = 11.52`, `L^xf_f = 22.20`, and `ρ = −0.5`, the correct value is

```
D^f_xf = 1 × 0.519^(−1/−0.5) = 0.519² = 0.269
```

The same formula was used (correctly) in the existing project for
`D_c_f` and `D_d_f`. Only the food extension JSON had the inverted sign.
Because the food extension JSON was never integrated into the existing
solver, this never caused observable issues in the existing 3-good app
— but the new value (0.27) is the correct one and is what this solver
expects.

### `kappa` scale — `E_sol = 0.5`, not data median

`κ_i = −slope_i · E^(1+ε) / ε`, evaluated at the expenditure scale that
the **solver** sees, not the **data** median. KCHS gives a median per-
adult-equivalent annual expenditure of `~6.0` (in 1000-KSh units), but
when the household solver evaluates one of the calibrated counties at
the median wage, it sees `E ≈ 0.3–0.5` (1000-KSh per period). Using
`E_sol = 0.5` keeps the κ magnitudes self-consistent with the solver's
λ fixed point. This matches the choice made in the existing project's
food extension calibration.

### `p_xf` — uniform national, defended for peer review

The document offers two normalisations for the food sector
(Section 21.2): (1) a uniform national price `p_xf` with
`A^M,xf_l = w_l / p_xf`; or (2) a uniform national TFP `A^M,xf = 1`
with `p_xf_l = w_l`. We adopt option (1), with `p_xf` set to the
population-weighted mean of `(p_c, p_d)` so that food prices live on
the same numerical scale as service prices (`p_xf = 0.0561` in
1000-KSh units, ≈ 56 KSh/hr-equivalent).

The defence: option (2) makes high-wage counties counterfactually
expensive in food, which would mechanically depress their food
expenditure share relative to KCHS data — a check easy for a reviewer
to fail. Option (1) is closer to reality for Kenyan staples (maize,
sugar, cooking oil), which trade substantially across counties and have
prices that don't co-move 1-for-1 with wages. The remaining county
heterogeneity in food production sits in `A^M,xf_l = w_l / p_xf`,
defensible as "labour productivity in the food-producing sector tracks
the local wage", which is exactly what the free-entry pricing condition
implies.

## Sign-correctness items preserved from the existing project

1. **`D^f_M ≈ 1.34`** — solver-consistent formula `r_M / wage_gap^ρ` (calibrated to 1.343).
2. **County `D` weight shrinkage** — Bayesian prior with `n_0 = 20` toward the national ratio; `n_eff` is the geometric mean of male/female cell counts.
3. **Population-weighted migration** — `solver_functions.migration_update` uses the variant from eq. 38–39 with population-weighted `Ubar`, so that at the baseline equilibrium `N' = N` exactly.
4. **Home/market plot uses expenditure shares** — the home-vs-market plot in main.py shows `P_iH·S_iH / (P_i·S_i)` and `p_i·S_iM / (P_i·S_i)`, which sum to 1 by the CES envelope. Quantity ratios would not.
5. **`φ = 0.5` literature prior** — used directly, not the KCHS IV estimate (which is biased downward by classical measurement error in hours).

New for the 4-good model:

6. **`D^f_xf ≈ 0.27`** from `(L^xf_m / L^xf_f)^(−1/ρ)` — see correction note above.
7. **Participation shifters refit structurally** — using actual ΔV from the four-state solve at the population-mean county, rather than the educational-premium proxy used in the existing 3-good code.

## Known limitations

- **Population is TUS-weighted, not census.** `N_l` is the sum of TUS
  person-day weights per county. This is what the existing project does
  and what we agreed for this rebuild; for a paper-grade run, swap in
  KNBS census counts.
- **`σ_mig = 1.0` is uncalibrated.** No internal-migration-elasticity
  estimate exists in our data. The literature value is fine for
  comparative-statics counterfactuals (which is what the app does);
  not fine for absolute migration magnitudes.
- **Home productivities `A^H,i_l = 1`.** Not separately identified from
  TUS time-use alone (would require home-output prices we don't have).
- **`h`-grid is coarse: `[0.5, 1.0, 2.0, 3.0]`.** Per-h plots are
  jagged for that reason. Adding more grid points scales the per-county
  solve time linearly.
- **No endogenous `h_m, h_f`.** As locked in the Phase 1 plan.
- **No endogenous prices.** Productivities are primitives; prices follow
  via the free-entry identities, but inter-county trade in `xn` is
  reported as a residual diagnostic rather than enforced as a binding
  constraint.

## Performance notes

On the calibration container:

- **`python calibrate.py`** — about 60 seconds end-to-end (two passes of
  47-county solve plus the data manipulation blocks).
- **Single-county Solve** in the app — about 200 ms (essentially feels
  instant).
- **"Solve all 47 counties"** in the app — about 30–35 seconds.
- **One counterfactual** in the app — another 30–35 seconds.

The solver is intentionally not parallelised; the 4-state participation
and adaptive damping make per-county work fairly heterogeneous and the
overhead of multiprocessing across only 47 counties wouldn't help much.
If the 30s wait becomes painful, the cleanest optimisation is to cache
results per county and only re-solve those whose parameters changed
between successive Solve-all clicks.

## Calibration sources

- TUS activity codes: `t11` (market work), `t13` (home domestic),
  `t14` (home care), `t231` (food preparation). All in minutes per day,
  converted to hours per week with factor `7/60`.
- KCHS labour module: `d40_gross` (employee monthly income),
  `d48_amount` (self-employed monthly income), `d27` (usual weekly
  hours). Hourly wage = monthly income / (weekly hours × 4.333).
- KCHS expenditure aggregate: `padqfdcons` (food); COICOP 13 from the
  non-food items file (personal care); COICOP 5 (domestic services).
  Non-food residual = 1 − food − care − domestic.
- ISIC codes for service-sector wages: 9700/9800 for domestic,
  8891/8810/8730 for care.

All quantities stored in **1000-KSh units** in `calibrated_params.json`
and `county_data.csv` so loaders don't have to scale.
