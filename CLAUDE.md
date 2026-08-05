# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is NordPSA

Nordic power system model built on PyPSA. Combines LP dispatch optimization with capacity expansion (investment) for 6 aggregated zones: SE-N, SE-S, NO-N, NO-S, DK, FI. Covers 2023–2025 at hourly/3h resolution.

## Setup

```bash
conda env create -f environment.yml   # create environment
conda activate nordpsa-env
pip install -e .                       # install nordpsa package
```

## Workflow

Data must be fetched and built before solving. Full pipeline:

```bash
make fetch        # fetch eSett load + production data (scripts/fetch_esett.py)
make fetch-ec     # fetch Energy Charts VRE profiles + DE-LU day-ahead price
make build        # build processed parquet inputs (scripts/build_inputs.py)
make solve        # run model (scripts/run_model.py)
```

**The bare command runs the canonical expansion baseline** (see below) — no flags needed:
```bash
python scripts/run_model.py --output run01_description              # baseline: 2h, 2023-2025
python scripts/run_model.py --year 2024 --output run02_2024only     # baseline, single year
python scripts/run_model.py --resolution 3 --output run03_coarse    # coarser, everything else baseline
```

**Run discipline:** commit *after* a successful simulation, not before. This ensures only good runs are traced to code state. Always propose the commit message and wait for user approval before committing. Name the output directory in the commit message so results are traceable:
```bash
python scripts/run_model.py --output run02_fleet_factors
# verify results, then:
git commit -m "Change X → ran: run02_fleet_factors"
```
Results go to `results/<output>/` and are gitignored (large files). `network.nc` contains full PyPSA network including inflow timeseries — verify correct hydrology with `n.storage_units_t.inflow`.

Visualize results:
```bash
python scripts/plot_dispatch.py results/run01_spring_flood_cyclic/ --resample 7D
```

## Architecture

### Data flow

```
eSett API → data/raw/production_*.parquet   (actual load + generation by carrier)
            data/raw/consumption_*.parquet
EC API    → data/raw/vre_*.parquet          (VRE capacity factor profiles)
            data/raw/price_market_*.parquet (DE-LU day-ahead price)
            ↓
scripts/build_inputs.py
            ↓
data/processed/
  load.parquet            (6 zones × hourly MW)
  vre_profiles.parquet    (columns: {zone}_{carrier}, capacity factors 0-1)
  vre_pnom.yaml           (installed capacities per zone/carrier)
  nuclear_profile.parquet (availability factor per zone, 0-1)
  thermal_profile.parquet (must-run thermal MW per zone)
  hydro_params.yaml       (GENERATED — do not edit, auto-fitted from production data)
  market_price.parquet    (DE-LU hourly price)
            ↓
nordpsa/network.py → pypsa.Network
            ↓
scripts/run_model.py → results/
```

### nordpsa/ package

- `network.py` — builds PyPSA network from processed inputs. Key function: `build_network(cfg, snapshots, load, vre_profiles, vre_noms, nuclear_profile, thermal_profile, hydro_params, market_price)`
- `esett.py` — eSett open data client, aggregates MBAs into NordPSA zones
- `ec.py` — Energy Charts API client for VRE profiles and DE-LU price
- `hydro.py` — parametric inflow model (Gaussian spring flood + seasonal cosine); fits against actual data, generates `inflow_timeseries()` for PyPSA StorageUnits

### Network components

| Component | Carrier | Notes |
|-----------|---------|-------|
| Bus | AC | One per zone |
| Link | — | Bidirectional NTC (p_min_pu=-1) |
| StorageUnit | hydro | Reservoir with parametric inflow, cyclic SOC, no pumping |
| Generator | nuclear | Must-run: p_min_pu = p_max_pu (NUCLEAR_MIN_FRACTION = 1.0). Dispatch = availability profile × p_nom, no optimizer freedom |
| Generator | wind_onshore/offshore, solar | VRE with capacity factor profiles |
| Generator | thermal | Must-run: p_min_pu = p_max_pu = actual profile |
| Generator | gas | Dispatchable peaker, extendable |
| Generator | market | Import/export valve: p_min_pu=-1, marginal_cost=DE-LU price |
| Generator | slack | Load shedding (3000 EUR/MWh), only zones without market connection |

### Cost model

Capital cost = `overnight_eur_per_w × 1e6 × (CRF + fom_fraction) × n_years`
CRF = `r × (1+r)^L / ((1+r)^L − 1)` with r=0.06, fom_fraction=0.02.
Capital cost is charged on `p_nom_opt` (total installed capacity, not just increment). All extendable generators have `p_nom_min = existing_capacity`.

### Zones and market connections

SE-N and NO-N have no direct continental market connection — only slack generators.
SE-S, NO-S, DK, FI have `market` generators (p_nom from config, price = DE-LU day-ahead).

## Important design decisions

**Canonical expansion baseline (defaults since 2026-08-05):** `python scripts/run_model.py` with no flags *is* the baseline — the configuration of `run250_hydroops_2h`, adopted so that experiments differ from the baseline only by the flags they actually name. Every default has an off-switch:

| Setting | Default | Off-switch |
|---|---|---|
| resolution | 2h (`snapshots.resolution_hours` in `zones.yaml`) | `--resolution N` |
| `--spill-cost` | 50 EUR/MWh | `--spill-cost 0.1` |
| `--cost-scenario` | `svk_2040` | `--cost-scenario none` |
| `--demand-scenario` | `svk_2040_mm` | `--demand-scenario none` |
| `--add-heat` | ON | `--no-add-heat` |
| `--hydro-restrictions` | ON | `--no-hydro-restrictions` |
| `--onwind-capfac-increase` | 0.30 | `--onwind-capfac-increase 0` |
| `--offwind-capfac-increase` | 0.10 | `--offwind-capfac-increase 0` |
| `--nuclear-min-load` | 0.6 | `--nuclear-min-load 1.0` |
| `--add-nuclear` | `SE-S:10:201 SE-N:10:202 FI:10:203` | `--no-add-nuclear` |
| `--voll` | 3000 EUR/MWh in **all** zones | `--no-voll` |
| `--market-elasticity` | ON (predates this change) | `--no-market-elast` |

`--voll` is the one default that does *not* reproduce `run250_hydroops_2h` exactly. It does not change the shedding price — `MC_SLACK` is already 3000 — only *where* slack exists: without it only the zones lacking a market connection (SE-N, NO-N) have a backstop, with it all six do. In an expansion run this gives the optimizer a new option: shed load at 3000 EUR/MWh instead of building peak capacity. Break-even against gas (~180 kEUR/MW/yr annualized) is roughly 60 scarcity hours per year, so FI — whose price tail is ~29 scarcity h/yr — is where a difference is most likely to show up as slightly less gas capacity. Also note `--voll` is taken from the *new* command on a `--dispatch` replay (as is `--no-voll`), so redispatch now gets VOLL by default; this removes the old trap where the flag had to be repeated manually.

Two traps this created, both handled in `scripts/run_model.py`:

- **`--add-nuclear` uses `action="extend"`**, so a non-empty argparse default would make a user-supplied `--add-nuclear` *append to* the baseline list instead of replacing it. The default is therefore the sentinel `None`, resolved to `DEFAULT_ADD_NUCLEAR` after `parse_args`.
- **`--dispatch` replays a source run's argv**, which was written against whatever defaults existed then. Replaying a pre-change run unmodified would silently inject the new defaults (e.g. give run240 hydro restrictions it never had). `write_run_meta` therefore stamps a `defaults:` line (`BASELINE_DEFAULTS_TAG`); `apply_dispatch_replay` restores `PRE_BASELINE_DEFAULTS` only for source runs *lacking* that line. Runs made after the change keep today's defaults.

`--resolution` deliberately stays `default=None` in argparse: `apply_dispatch_replay` relies on `args.resolution or 1` to keep redispatch at 1h, so the 2h baseline lives in config instead.

**IPM with crossover:** Solver must use `run_crossover: "on"` for capacity expansion runs. Without crossover, p_nom_opt stays near p_nom_min even when investment is profitable (interior-point primal solution, not a vertex).

**Nuclear — existing fleet must-run, new build load-following:** `p_min = p_max × min_frac` in both the dispatch branch (actual `nuclear_profile`) and the synthetic-nuclear branch (`--add-nuclear`, `availability_timeseries`), via `NUCLEAR_MIN_FRACTION = 1.0` (`nordpsa/network.py`).

- **Existing fleet: `min_frac = 1.0`** → `p_min_pu = p_max_pu`, the optimizer cannot down-regulate it, and dispatch (`n.generators_t.p`) equals the availability profile × p_nom in every snapshot.
- **New nuclear (`--add-nuclear` / `--add-nuclear-fixed`): `min_frac = 0.6` by default** since the canonical baseline (`--nuclear-min-load`, sets `min_load_frac_exp`). `--nuclear-min-load 1.0` makes new nuclear pure must-run too.

Verify with `--dry-run`, which prints `must-run` per generator: in the baseline `SE-S nuclear` shows 0.837 (= its CF, pure must-run) while `SE-S nuclear exp` shows 0.517 (= 0.6 × CF).

**Thermal as must-run Generator:** `p_min_pu = p_max_pu = profile/p_nom`. Dispatch is fully determined by data; optimizer has no freedom. Thermal is NOT subtracted from load.

**Hydro inflow model:** Parameters are manually calibrated spring-flood profiles stored in `config/hydro_params.yaml` (NOT `data/processed/hydro_params.yaml` which is auto-generated and must never be used). SE-N: A=10000 MW spring flood, mu=day 135 (May 15), phi=183 (summer-high cosine). `build_inputs.py` does NOT regenerate these — they are a config artifact. Verify correct hydrology after each run: SE-N inflow should peak ~15000 MW in May, ~2600 MW in January; reservoir SOC should peak ~85% in July.

**Hydro SOC cycling:** `cyclic_state_of_charge=True` + `extra_functionality` callback pins SOC[t=0] = target from `hydro_soc_initial` in `zones.yaml`. This forces start = end = target (e.g. 70%) while the LP optimizes freely in between.

**Hydro operation restrictions (`--hydro-restrictions`, default ON since 2026-08-05, off via `--no-hydro-restrictions`):** Constraints on the *reservoir* StorageUnits (after any RoR split), configured under `hydro_operation` in `zones.yaml` and implemented as an `extra_functionality` callback (`hydro_operation_constraints` in `nordpsa/network.py`):

| Constraint | Form | Default |
|---|---|---|
| `min_hourly_frac` | `p_dispatch[t] ≥ f × p_nom` | 0.10 |
| `min_daily_frac` | `Σ_day p·w ≥ f × p_nom × H_day` | 0.20 |
| `max_weekly_frac` | `Σ_week p·w ≤ f × p_nom × H_week` | 0.77 |
| `bypass_spill` | `Σ_week spill·w ≥ κ × (Σ_week p·w − threshold)` | off |

Purpose: stop the LP from (a) shutting hydro off entirely through long low-price periods (small river reservoirs would overflow) and (b) running at full power week after week — a common ELLI-type artefact. Window sums use the actual `snapshot_weightings`, so they are correct at 1h/2h/3h and partial windows at the series edges are not over-tightened. Constraint names are prefixed `custom-`.

With cyclic SOC, annual production equals inflow, so the constraints are mutually consistent only if `min_daily_frac ≤ inflow/(p_nom×H) ≤ max_weekly_frac`. `hydro_operation_bounds()` reports that ratio per zone and `run_model.py` prints it plus warnings before solving. For 2024 the ratio is 0.46–0.55 in all zones, comfortably inside [0.20, 0.77].

Per-zone values come from Ek Fälth et al. (2025) supplementary material (`docs/hydro_restrictions.pdf`), read off the violin plots (±2–3 pp) for *Present regime*, 1 week: Fig A.2 (sustained capacity) SE1 0.77 / SE2 0.87 / SE3 0.77, aggregated capacity-weighted to SE-N 0.83 and SE-S 0.77; Fig A.5 (annual production loss 0.12–0.22 %) calibrates the `bypass_spill` κ to SE-N 0.34 / SE-S 0.59. ⚠️ The study covers **Sweden only** — NO-N/NO-S (0.85) and FI (0.80) are flagged assumptions argued from reservoir hours (NO ~3130 h vs SE-N 2723 h vs SE-S 1514 h). ⚠️ Fig A.4 shows strong seasonality (SE1 ~72 % in March, ~91 % in May–June) that a year-constant cap cannot capture; monthly caps are not implemented. ⚠️ `bypass_spill` is off by default: PyPSA bounds spill by the inflow in the same snapshot, so a high-production/low-inflow week can go infeasible; κ is not given in the source and must be calibrated. Verify with `python scripts/test_hydro_operation.py`.

**p_nom_max bounds:** All extendable generators have finite `p_nom_max_mw` in config (20k for nuclear/gas, 50k for VRE per zone). Without these, HiGHS sees ~3e10 column bounds and prints scaling warnings (harmless but ugly).

## Config

All parameters in `config/zones.yaml`:
- Zone definitions with hydro/nuclear existing capacity
- NTC links between zones
- Market connection capacities
- Technology costs (overnight, lifetime, VOM, extendable flag, p_nom_max)
- Solver settings (HiGHS IPM + crossover)
- Simulation period and resolution
