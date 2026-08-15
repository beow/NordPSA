# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is NordPSA

Nordic power system model built on PyPSA. Combines LP dispatch optimization with capacity expansion (investment) for 6 aggregated zones: SE-N, SE-S, NO-N, NO-S, DK, FI. Covers 2023–2025; the canonical baseline runs at 2h, and 1h/3h are available via `--resolution`.

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

**The bare `--dispatch` runs the canonical dispatch template** (the configuration of `run340_minh005_seg20_2h`) — capacities frozen from the source expansion run, rolling horizon 1+3 weeks, the calibrated terminal water-value curve, and no price proxy:

```bash
python scripts/run_model.py --output run01_expansion                      # kanonisk EXPANSION
python scripts/run_model.py --dispatch run01_expansion --output run02_disp # kanonisk DISPATCH
```

`--dispatch` implies five things that previously had to be spelled out on every command. Forgetting any of them made the run fall back *silently* to a cyclic LP with the price proxy — the opposite of what was intended, which is exactly the trap run319 fell into. Each has an off-switch:

| implied by `--dispatch` | off-switch |
|---|---|
| `--rolling-horizon`, 1 week/window + 3 weeks look-ahead | `--no-rolling-horizon` |
| `--terminal-curve` = `config/terminal_curve_2040_calibrated.yaml` | `--no-terminal-curve` |
| `--no-hydro-price-proxy` (the curve *is* the water value) | `--hydro-price-proxy` |
| resolution 2h (`DEFAULT_DISPATCH_RESOLUTION`, was 1h) | `--resolution N` |
| `--spill-cost 0.1` (`DEFAULT_SPILL_COST_DISPATCH`, was 50 — see below) | `--spill-cost N` |

`--expansion` exists as an explicit synonym for "no `--dispatch`"; it changes nothing but lets scripts say which mode they mean, and errors if combined with `--dispatch`.

⚠️ **Reproducing dispatch runs made before this change** (run316 and older) needs `--no-rolling-horizon --no-terminal-curve --hydro-price-proxy --resolution 2 --hydro-min-hourly 0.10` — verified to reproduce run316's flag string exactly. This follows the same precedent as `--vre-curtailment-cost`: new defaults apply to newly typed commands, and older runs are reproduced by naming the old values.

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
| Generator | nuclear | Existing fleet must-run: p_min_pu = p_max_pu, dispatch = availability × p_nom. New build (`--add-nuclear`) load-follows at 0.6 × p_max by default |
| Generator | wind_onshore/offshore, solar | VRE with capacity factor profiles |
| Generator | thermal | Must-run: p_min_pu = p_max_pu = actual profile |
| Generator | gas | Dispatchable peaker, extendable |
| Generator | market | Import/export valve: p_min_pu=-1, marginal_cost=DE-LU price |
| Generator | slack | Load shedding (3000 EUR/MWh). All six zones by default (`--voll`); `--no-voll` limits it to the zones without a market connection |

### Cost model

Capital cost = `overnight_eur_per_w × 1e6 × (CRF + fom_fraction) × n_years`
CRF = `r × (1+r)^L / ((1+r)^L − 1)` with r=0.06, fom_fraction=0.02.
Capital cost is charged on `p_nom_opt` (total installed capacity, not just increment). All extendable generators have `p_nom_min = existing_capacity`.

### Zones and market connections

SE-N and NO-N have no direct continental market connection — with `--no-voll` they have only slack generators.
SE-S, NO-S, DK, FI have `market` generators (p_nom from config, price = DE-LU day-ahead).

## Important design decisions

**Canonical expansion baseline (defaults since 2026-08-05):** `python scripts/run_model.py` with no flags *is* the baseline — the configuration of `run250_hydroops_2h`, adopted so that experiments differ from the baseline only by the flags they actually name. Every default has an off-switch:

| Setting | Default | Off-switch |
|---|---|---|
| resolution | 2h (`snapshots.resolution_hours` in `zones.yaml`) | `--resolution N` |
| `--spill-cost` | 50 EUR/MWh (expansion; 0.1 med frysta kapaciteter — se nedan) | `--spill-cost N` |
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

**Dispatch-only default: `--vre-curtailment-cost 5` (since 2026-08-07).** Applies *only* when capacities are frozen — `--dispatch` or `--no-expansion` — and is 0 otherwise. It sets `marginal_cost = vom − C` on all wind/solar, which is algebraically identical to charging C per MWh curtailed (`mc·p + C·(A−p) = C·A + (mc−C)·p`; at fixed capacity `C·A` is a constant). This is the **only** route to negative zone prices: without it curtailment is free, no bid is negative, and there is no unit commitment (`committable=0`, `start_up_cost=0`), so the price floor is the cheapest bid (VOM). Verified in `run258_ref_dispatch_3h`: min 0.10 and zero negative hours in all six zones.

- ⚠️ **Never in expansion.** With extendable VRE, `A ∝ p_nom_opt`, so `C·A` is not constant — the term becomes a production subsidy that grows with built capacity and the optimizer builds to `p_nom_max`. `apply_vre_curtailment_cost` raises `SystemExit` if any VRE generator is still extendable; that is why the default is a `None` sentinel resolved *after* `apply_dispatch_replay`, not an argparse default.
- The flag is taken from the *new* command on a `--dispatch` replay (like `--voll`), since the source's argv can never contain it.
- **Calibration:** C represents what is actually lost per MWh when curtailing in 2040 — guarantees of origin only (forecast 2–5 EUR/MWh). Elcertifikat is zero (closed to new plants after 2021, ends 2035, early closure under investigation) and CfDs pay nothing in negative hours (CEEAG suspends support at negative prices; German EEG §51 tightens to a 1-hour rule in 2027). **5 is the upper end of the GO range**; the central estimate is 2.
- ⚠️ Reproducing runs made before this change (run255–run258, and any older dispatch) requires `--vre-curtailment-cost 0` explicitly.

Two traps this created, both handled in `scripts/run_model.py`:

- **`--add-nuclear` uses `action="extend"`**, so a non-empty argparse default would make a user-supplied `--add-nuclear` *append to* the baseline list instead of replacing it. The default is therefore the sentinel `None`, resolved to `DEFAULT_ADD_NUCLEAR` after `parse_args`.
- **`--dispatch` replays a source run's argv**, which was written against whatever defaults existed then. Replaying a pre-change run unmodified would silently inject the new defaults (e.g. give run240 hydro restrictions it never had). `write_run_meta` therefore stamps a `defaults:` line (`BASELINE_DEFAULTS_TAG`); `apply_dispatch_replay` restores `PRE_BASELINE_DEFAULTS` only for source runs *lacking* that line. Runs made after the change keep today's defaults.

`--resolution` deliberately stays `default=None` in argparse: `apply_dispatch_replay` relies on `args.resolution or 1` to keep redispatch at 1h, so the 2h baseline lives in config instead.

**`--spill-cost` is mode-dependent: 50 in expansion, 0.1 with frozen capacities (since 2026-08-15).** It is the mirror image of `--vre-curtailment-cost` — a modelling guardrail in one mode and a physical cost in the other — and an explicit value wins in both.

- **Expansion 50 = guardrail, not a real cost.** With free spill the optimizer can build excess wind/solar and dump the displaced hydro almost for nothing, so the system's true ability to absorb VRE is hidden and it overinvests. Measured 2026-05-31: run43 (spill 1) gave 63 TWh phantom spill and NO-S wind 20.8 GW; run44 (spill 50) gave 0 spill and 13.1 GW — **−37 %**.
- **Dispatch 0.1, because 50 double-counts the value of water.** With frozen capacities there is no investment decision to distort, and the cost of spilling *is* the value of the water lost — already carried by the LP as the shadow price on SOC (λ ≈ 73–80). At the ceiling marginal water has zero storage value, so the choice is produce at price *p* or pay *c*; the model produces as long as `p > −c`. At c = 50 it runs hydro down to **−50 EUR/MWh** rather than spill, which no real operator does — they bypass.
- ⚠️ **Empirically inert, so the change is principled rather than numerical — measured, not assumed.** Spill is 0.0000 TWh in every run in the terminal-curve track (run316–run345), max SOC reaches 94 % (run340) / 99.9 % (run320 NO-S), and in *all* hours above 90 % fill the price sits at 52–74 EUR/MWh with **zero** hours below 5. The fingerprint of the double-count — nearly full reservoir *and* collapsed price *and* hydro running hard — appears nowhere.
- **A/B `run346_spillcost01_2h` vs `run344_default_dispatch_2h`** (identical except 50 → 0.1): `hydro_spill.csv` **bit-identical** (0 both ways), zone prices move ≤ 0.018 EUR/MWh in the mean, negative-hour counts identical to the hour, hydro 531.91 → 531.96 TWh, v/s SE 1.547 → 1.548 · NO 2.021 → 2.017 · FI 0.989 → 0.992, facit score 0.4185 → 0.4189, drift −1.25 → −1.30 TWh. The residual differences are **degenerate-LP tie-breaking, not behaviour**: since spill is 0 in both, the coefficient contributes nothing to either objective and only changes which optimal vertex the solver reports. Window objectives sum to −1.041535e12 vs −1.041483e12 (0.005 %); note the rolling horizon is a *sequence* of coupled LPs (SOC carries over), so the totals have no optimality relation to each other and a different tie-break in one window shifts the next window's starting point.
- ⛔ **Falsified by the same A/B:** the hypothesis that a high spill cost pushes production *earlier* (the known ~5-week phase error, v49 against facit's v0.6). The first harmonic of weekly production is **v49.2, amplitude 0.458, in both runs** — unmoved to the first decimal. The phase error is not spill-driven.
- ⚠️ Note the model **under-spills relative to reality**: Ek Fälth et al. put real annual production loss from bypass spill at 0.12–0.22 %, the model at 0.000 %.

Two traps this closed: `--spill-cost` was **not** copied from the new command in `apply_dispatch_replay`, so `--dispatch X --spill-cost 0.1` was *silently ignored* (the run319 class of trap); and the flag did not appear in the `flaggor:` line of `run_meta.txt`, so a run's spill cost could only be read from `console.log`. Both fixed. The `PRE_BASELINE_DEFAULTS` entry for `spill_cost` was removed as unreachable — the new dispatch default 0.1 is exactly what its restore path produced via `network.py`'s `.get(..., 0.1)` fallback.

**IPM with crossover:** Solver must use `run_crossover: "on"` for capacity expansion runs. Without crossover, p_nom_opt stays near p_nom_min even when investment is profitable (interior-point primal solution, not a vertex).

**Nuclear — existing fleet must-run, new build load-following:** `p_min = p_max × min_frac` in both the dispatch branch (actual `nuclear_profile`) and the synthetic-nuclear branch (`--add-nuclear`, `availability_timeseries`), via `NUCLEAR_MIN_FRACTION = 1.0` (`nordpsa/network.py`).

- **Existing fleet: `min_frac = 1.0`** → `p_min_pu = p_max_pu`, the optimizer cannot down-regulate it, and dispatch (`n.generators_t.p`) equals the availability profile × p_nom in every snapshot.
- **New nuclear (`--add-nuclear` / `--add-nuclear-fixed`): `min_frac = 0.6` by default** since the canonical baseline (`--nuclear-min-load`, sets `min_load_frac_exp`). `--nuclear-min-load 1.0` makes new nuclear pure must-run too.

Verify with `--dry-run`, which prints `must-run` per generator: in the baseline `SE-S nuclear` shows 0.837 (= its CF, pure must-run) while `SE-S nuclear exp` shows 0.517 (= 0.6 × CF).

**⚠️ Hydro bids at the HISTORICAL zone price (water-value proxy, `--no-hydro-price-proxy` to disable):** the reservoir StorageUnit's `marginal_cost` is set to that zone's *actual observed day-ahead price* (from `market_prices.parquet`, floored at hydro VOM 0.6) — verified identical to the 2h mean of the historical series in all 13152 snapshots of run260. Hydro's effective bid is therefore

```
historical price[t]  +  mu_energy_balance[t] / efficiency_dispatch
```

Introduced in `947ec81` (run23, 2026-05-18) explicitly as a *"water value proxy"*, replacing a flat VOM, at a time when the model did not yet extract a water value. Thirteen days later `c08906f` (run57) added `assign_all_duals=True` and the genuine endogenous water value — the SOC-balance dual — but the proxy was never removed. They have been stacked ever since; the line has not been touched since run23.

This matters because the endogenous water value is nearly constant (1–6 unique values per zone over three years; SE-S has exactly one), so **all** the time variation in hydro's bid comes from the 2023–25 price series, none from the model. In a 2040 expansion that is circular: hydro's dispatch, and hence the price shape the model produces, is anchored to the price shape it is meant to predict. Use `--no-hydro-price-proxy` (flat VOM) to isolate the effect; the default is unchanged, so existing runs stay reproducible.

**Thermal as must-run Generator:** `p_min_pu = p_max_pu = profile/p_nom`. Dispatch is fully determined by data; optimizer has no freedom. Thermal is NOT subtracted from load.

**Hydro inflow model:** Parameters are manually calibrated spring-flood profiles stored in `config/hydro_params.yaml` (NOT `data/processed/hydro_params.yaml` which is auto-generated and must never be used). SE-N: A=10000 MW spring flood, mu=day 135 (May 15), phi=183 (summer-high cosine). `build_inputs.py` does NOT regenerate these — they are a config artifact. Verify correct hydrology after each run: SE-N inflow should peak ~15000 MW in May, ~2600 MW in January; reservoir SOC should peak ~85% in July.

**Hydro SOC cycling:** `cyclic_state_of_charge=True` + `extra_functionality` callback pins SOC[t=0] = target from `hydro_soc_initial` in `zones.yaml`. This forces start = end = target (e.g. 70%) while the LP optimizes freely in between.

**Hydro operation restrictions (`--hydro-restrictions`, default ON since 2026-08-05, off via `--no-hydro-restrictions`):** Constraints on the *reservoir* StorageUnits (after any RoR split), configured under `hydro_operation` in `zones.yaml` and implemented as an `extra_functionality` callback (`hydro_operation_constraints` in `nordpsa/network.py`):

| Constraint | Form | Default |
|---|---|---|
| `min_hourly_frac` | `p_dispatch[t] ≥ f × p_nom` | 0.05 |
| `min_daily_frac` | `Σ_day p·w ≥ f × p_nom × H_day` | 0.20 |
| `max_weekly_frac` | `Σ_week p·w ≤ f × p_nom × H_week` | 0.77 |
| `bypass_spill` | `Σ_week spill·w ≥ κ × (Σ_week p·w − threshold)` | off |

Purpose: stop the LP from (a) shutting hydro off entirely through long low-price periods (small river reservoirs would overflow) and (b) running at full power week after week — a common ELLI-type artefact. Window sums use the actual `snapshot_weightings`, so they are correct at 1h/2h/3h and partial windows at the series edges are not over-tightened. Constraint names are prefixed `custom-`.

**⚠️ `min_hourly_frac` and `min_daily_frac` are LOCKED (2026-08-14) — do not change without a new measurement.** They were calibrated against hydro's *bid curve* in observed data (production vs price, adelsfors.se June 2026, SE1+SE2), which is a different observable from the seasonal ratio. `min_hourly_frac` 0.10 → **0.05**: at 0.10 the model ran 3.3 GW in SE-N at zero price against ~2.0 observed; at 0.05 it runs 2.78 and the mean error below 10 EUR/MWh falls 1.40 → 0.89 GW. The remainder is *run-of-river* (2.34 GW in June), which this parameter does not control. It costs 0.09 in facit score (0.338 → 0.425) — the hourly floor and the seasonal measure pull in opposite directions, and the observed behaviour won. `min_daily_frac` stays at **0.20**: run341 (0.10) and run342 (0.15) tested and **rejected** — the daily floor does not bind at low prices at all (production below 40 EUR/MWh is identical at 2.78 GW for 0.10/0.15/0.20), it binds in the 40–70 mid-range where reality produces *even more* than the model, so relaxing it worsened the fit (MAE over 20–60 EUR/MWh: 1.96 → 2.21 → 2.28) and drove reservoir drift positive (−1.0 → +2.6 → +3.7 TWh, i.e. hoarding). Neither value comes from Ek Fälth et al. — they are modelling guardrails; only `max_weekly_frac_by_zone` and `bypass_spill` have a source.

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
