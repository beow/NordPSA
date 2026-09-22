# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is NordPSA

Nordic power system model built on PyPSA. Combines LP dispatch optimization with capacity expansion (investment) for 6 aggregated zones: SE-N, SE-S, NO-N, NO-S, DK, FI. Production and consumption data are mostly from 2023–2025 and scaled in the capacity expansion step; the canonical baseline runs at 2h, and 1h/3h are available via `--resolution`.

## Setup

```bash
conda env create -f environment.yml   # create environment
conda activate nordpsa-env
pip install -e .                       # install nordpsa package
```

## Workflow

Data must be fetched and built before solving:

```bash
make data         # full pipeline: fetch (eSett, Energy Charts, NVE/ENTSO-E, ninja, Open-Meteo),
                  # synthetic SE run-of-river, then build data/processed/
make build        # only rebuild data/processed/ from data/raw/
make fetch-nve    # single steps: see Makefile (needs ENTSOE_API_TOKEN / NINJA_TOKEN)
```

### Running the model: three commands

```bash
nordpsa expand   --output run500_baseline                   # 2040 expansion, capacities optimized
nordpsa dispatch --from run500_baseline --output run501     # source's world + frozen fleet
nordpsa today    --output run502_today                      # today's system, dispatch
```

(`python -m nordpsa …` works too; the `nordpsa` command needs `pip install -e .`.)

| command | mode | world (default) | capacities |
|---|---|---|---|
| `expand` | expansion: one cyclic LP, hydro mc from the terminal curve | `2040_svk_mm` (`--world`) | optimized |
| `dispatch --from RUN` | dispatch: rolling horizon 1+3 weeks, terminal curve | inherited from RUN | frozen at RUN's `p_nom_opt` |
| `today` | dispatch | `today` | config values (today's fleet) |

**Settings are layered, later wins:** `config/defaults.yaml` ← `config/worlds/<world>.yaml` ← `--experiment FILE …` ← `--set KEY=VALUE …`. Every default lives in exactly one place (`defaults.yaml`); unknown keys are an error. The only named flags are `--output --desc --resolution --year --experiment --set --dry-run --print-config` (+ `--world`/`--from`). Everything else is a settings key:

```bash
nordpsa expand --experiment lowhydro06 --output run503_dry          # config/experiments/lowhydro06.yaml
nordpsa expand --set market.ntc_scale=0.5 --set hydro.bid_ladder=[5,34.6] --output run504
nordpsa expand --print-config                                       # show the resolved settings, run nothing
```

The fully resolved settings are saved as `results/<run>/run_config.yaml`. **`dispatch --from RUN` inherits RUN's whole world from that file** (scenarios, nuclear, VRE, NTC, dry year, heat, hydro) — nothing has to be repeated. Only the per-command sections (`period`, `solver`, `dispatch`, `expansion`) come from the current defaults + the world file, so a recalibrated terminal curve reaches a redispatch of an old expansion automatically.

- `modes:` in a world file restricts which commands may use it (`today` is dispatch-only).
- Mode-dependent values are sections, not sentinels: `expansion.spill_cost` 50 / `dispatch.spill_cost` 0.1, `dispatch.vre_curtailment_cost` 5 (never in expansion — see below), `expansion.hydro_mc_curve`, `dispatch.terminal_curve` / `terminal_anchor` / `rolling_weeks` / `lookahead_weeks`.

**Run discipline:** commit *after* a successful simulation, not before. This ensures only good runs are traced to code state. Always propose the commit message and wait for user approval before committing. Name the output directory in the commit message so results are traceable:
```bash
nordpsa expand --output run02_fleet_factors
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
nordpsa/network/ (build_network) → pypsa.Network
            ↓
nordpsa expand | dispatch | today → results/<run>/  (+ run_config.yaml)
```

### nordpsa/ package

- `cli.py` — the three commands; `settings.py` — layered settings, `--set`, validation, `run_config.yaml`
- `run.py` — one run from resolved settings to results (the pipeline, in order)
- `world.py` — applies the world to the config: cost/demand scenarios, NTC, market cables, dry year, potentials
- `modes.py` — what differs between modes: freezing from a source run, VRE curtailment cost, hydro mc curve
- `solve.py` — cyclic LP, rolling horizon with terminal curve, result extraction/saving
- `inputs.py` — config and processed inputs, snapshots, resampling, CF boost
- `network/` — builds the PyPSA network: `build.py` (`build_network`) calls one module per component type — `core` (buses, links, load, slack), `generation` (thermal, nuclear, VRE, gas), `hydropower`, `storage`, `market`, `hydrogen`, `heat`, `ev`, `dsr`; `costs` annualizes capital costs
- `constraints/` — the `extra_functionality` callbacks: `soc` (cyclic SOC anchor), `terminal_value`, `bid_ladder`, `hydro_ops` (operation restrictions)
- `profiles/` — time series the model is built on: `hydro_inflow` (NVE/ENTSO-E inflow + RoR, parametric spring-flood model, RoR high-frequency), `nuclear_availability` (synthetic stochastic availability), `heat_load` (When2Heat district-heating profiles)
- `data/` — clients for external sources, used only by `scripts/fetch_*.py` and `build_inputs.py`: `esett`, `ec` (Energy Charts), `entsoe` (+ Elexon), `ninja` (Renewables.ninja)
- `wv/` — terminal curve + scoring

### Network components

| Component | Carrier | Notes |
|-----------|---------|-------|
| Bus | AC | One per zone |
| Link | — | Bidirectional NTC (p_min_pu = −1) |
| StorageUnit | hydro | Reservoir: inflow from data, SOC anchored (see below), no pumping |
| Generator | hydro | Run-of-river, must-run (p_min_pu = p_max_pu = profile) |
| Generator | nuclear | Existing fleet must-run; new build (`nuclear.add`) load-follows down to `nuclear.new_min_load` |
| Generator | wind_onshore/offshore, solar | VRE with capacity-factor profiles |
| Generator | thermal | Must-run: p_min_pu = p_max_pu = actual profile |
| Generator | gas | Dispatchable peaker, extendable |
| Generator | market | Continental price valve, p_min_pu = −1, with a price staircase (`market.elasticity`) |
| Generator | slack | Load shedding at `voll` (3000 EUR/MWh) in all six zones; `voll: null` limits it to zones without a market connection |
| StorageUnit | battery | Fixed, free (sunk): today's fleet, or the demand scenario's |
| Buses/Links/Stores | heat, H2, EV, DSR | Sector coupling from the demand scenario (`network/heat.py`, `hydrogen.py`, `ev.py`, `dsr.py`) |

### Cost model

Capital cost = `overnight_eur_per_w × 1e6 × (CRF + fom_fraction) × n_years`
CRF = `r × (1+r)^L / ((1+r)^L − 1)` with r = 0.06, fom_fraction = 0.02 unless the technology sets its own.
Capital cost is charged on `p_nom_opt` (total installed capacity, not just the increment). All extendable generators have `p_nom_min` = existing capacity and a finite `p_nom_max_mw` (20 GW nuclear/gas, 50 GW VRE per zone; without it HiGHS sees ~3e10 column bounds). Cost scenarios add interest during construction: `overnight' = OC · (1 + build_years/2 · r)`.

⚠️ **The raw objective is not a system cost.** It contains the bid-ladder offsets and the VRE curtailment cost. Real cost = `objective + objective_constant − Σ_t Σ_k (mc_t + offset_k)·d_k·w_t`. ⚠️ Demand-scenario components (batteries, electrolysers, H2 stores) are built *free* (sunk), so system cost is understated by roughly 7 % in the 2040 world.

### Zones and market connections

SE-N and NO-N have no direct continental connection; SE-S, NO-S, DK and FI have `market` generators priced at the neighbouring zone's historical day-ahead price (rescaled to 2040 levels by the demand scenario). Cable capacities are set in `zones.yaml` and overridden, in this order, by the demand scenario's `market_ntc_overrides`, then `market.ntc_scale` (all ten cables), then `market.ntc_override` (named cables). With VOLL in all zones a cable cut cannot make the model infeasible, only expensive.

Internal NTCs are **flow-based equivalents** calibrated on a flow-based-free window (locked 2026-08-22); the 2040 world adds `links_expansion_overrides` (Aurora SE-N–FI, NO-N↔NO-S) and the demand scenario's `ntc_overrides` (snitt 2 = 9000 MW, derated from the TSO's 10 700).

## Important design decisions

### Two passes: expansion, then dispatch

Expansion is **one cyclic LP with perfect foresight** over 2023–2025 weather; capacities are optimized. Dispatch freezes those capacities and solves a **rolling horizon** (1-week windows + 3 weeks look-ahead that is solved and discarded), with a **terminal water-value curve** at each window end. The rolling horizon exists because perfect foresight makes the endogenous water value nearly constant (1–2 unique values per zone over three years), which erases the price formation the model is meant to study.

Consequences worth knowing when reading results:
- **Per-year prices from an expansion carry a foresight rent** the dispatch cannot reproduce: the cyclic LP knows 2024 is wet and drains hard in dry 2023. Compare three-year means across modes, not single years.
- **Reservoir drift:** cyclic SOC makes expansion stock-neutral by construction. A dispatch is judged against the *measured* stock change over the window (EC: +3.13 TWh 2023-01-02 → 2025-12-31), not against zero.

### Mode-dependent costs

| setting | expansion | dispatch | why |
|---|---|---|---|
| `spill_cost` | 50 | 0.1 | In expansion a guardrail: free spill let the optimizer overbuild VRE and dump displaced hydro (measured: −37 % NO-S wind at 50). In dispatch 50 double-counts water value, which the SOC dual already carries. Empirically inert in dispatch (spill 0 either way). |
| `vre_curtailment_cost` | 0 (forbidden) | 5 | VRE bids `vom − C`, identical to charging C per curtailed MWh when capacity is fixed — the **only** route to negative prices. With extendable VRE it becomes a production subsidy that grows with capacity, so `apply_vre_curtailment_cost` refuses to run if any VRE is extendable. C = 5 is the upper end of the guarantees-of-origin range (2–5); elcertifikat and CfDs pay nothing in negative hours. |
| hydro marginal cost | terminal curve along the normal path | VOM 0.6 + SOC dual | see next section |

### Hydro marginal cost in expansion

The reservoir's `marginal_cost` in expansion is the terminal curve evaluated along its reference path, `λ_bas · A(v)` (`expansion.hydro_mc_curve`, file `…_v12_exp73.yaml`). Hydro needs *some* time shape in its bid — a flat bid makes the LP degenerate — and this one comes from zone tables and measured reservoir data, not from prices.

- **λ_bas = 73 uniformly, locked.** Read directly off full-foresight reference runs: 73.02 and 72.87 EUR/MWh in all five zones and all hours, on two different grids (NO-N export capacity −39 % moved λ by 0.2 %). The dispatch file's per-zone anchors are a drift correction and must **not** be used in expansion.
- The level is nearly inert (the SOC dual absorbs it), but the curve is multiplicative, so the *shape* matters: a flat A(v) moved +5.5 GW solar and −1.0 GW offshore wind.
- `b_mean` is inert here: along the reference path the deviation term is 1 by construction.
- ⛔ **The historical-price proxy is removed.** Until 2026-08-20 the expansion bid was each zone's observed 2023–25 day-ahead price, which made the 2040 price shape circular (anchored to the prices it is meant to predict).

### Terminal water-value curve (dispatch)

`λ_k(v, z) = λ_bas[z] · A(v, z) · P_k(x − x_ref(v, z))` — v = week, x = fill, k = SOC segment (20 segments, concave). A(v) is the seasonal level, `x_ref(v)` the normal reservoir path, and B = `b_mean` sets how steeply the value responds to being off that path. Production file: `config/terminal_curves/terminal_curve_2040_gemini_v12.yaml`; every earlier version is in `archive/` with an index.

**Three global degrees of freedom, everything else is shape:**

| parameter | value | basis |
|---|---|---|
| `b_mean` | **4.0** uniform | First non-circular measurement of B: regression of log price on reservoir deviation with year fixed effects, EC reservoir data + ENTSO-E prices 2015–2025, gives B = 1.5–3.3; the model is flatter than reality even at 4.0 (B is sublinear in `b_mean`). Bid-curve MAE 2.94 → 2.03. |
| `a_scale` | **0.15** | Robustness: A(v) is the only seasonal signal outside `x_ref`. |
| `b_amp` | **0** | Measured near-inert in a 2×2; the measured seasonal amplitude is not distinguishable from zero. |
| `x_ref` | two harmonics, **per zone** | Least squares against EC median reservoir paths; two harmonics halve the error of one (FI R² 0.64 → 0.92). SE-S is rain-fed and needs its own path (v/s error −92 %). 52 free weekly values were tried and rejected. |
| anchors λ_bas | SE-N 66.0 · SE-S 64.5 · NO-N 70.2 · NO-S 69.6 · FI 72.5 | A drift correction for limited foresight, not a claim that water is worth different amounts per zone. `today` has its own anchors (27–58), set in `worlds/today.yaml`. |

**Calibration targets, in order:** the observed hydro *bid curve* (production vs price) and the measured price elasticity B; seasonal ratio (winter/summer, eSett, measured on *all* hydro including RoR, per zone) and EC reservoir band are validation. ⛔ Never calibrate the curve against the price distribution — that reintroduces the circularity the proxy had.

⚠️ **Open risks:**
- **`x_ref` sensitivity is the main risk at b = 4:** a 10 pp error in the reference path costs 49 % (22 % at b = 2), and `x_ref` is measured on 2015–2025, not 2040.
- **The northern price tail overshoots** (h > 100: SE-N ~3100 vs 828 observed, NO-N ~2800 vs 320) — the known cost of b = 4.
- **FI is too sensitive** (model B 3.1–3.6 already at `b_mean` 0.8–2.0, vs measured 1.7; a zone-specific FI value is the open question), and its seasonal ratio stays short of eSett — four levers tried, none moved it.
- B is regime dependent (2015–20 ≈ 1.9, 2021–25 ≈ 4.3); choosing 4 implicitly says 2040 resembles the scarcity regime.

### Hydro bid ladder (both modes)

A single `marginal_cost` per reservoir makes the whole 52 GW fleet bid at one price, so the zone price sits *on* hydro's bid in 58–79 % of hours and hydro runs bang-bang (at its cap or floor 17–52 % of the time). `hydro.bid_ladder: [K, WIDTH]` splits dispatch into K tiers of `p_nom/K` with offsets `WIDTH·((k+½)/K − ½)`, added only to the objective, so the ladder centres on whatever the bid is in each mode.

- **Motivation is the aggregation error** — hundreds of reservoirs lumped into one per zone — mirroring the two-layer Nordic practice (seasonal water-value table → short-term bid curve). The model only has layer one.
- **Default `[3, 36]`** (offsets −12/0/+12): in a clean expansion A/B it improved all three validation observables at once (bid-curve MAE 3.67 → 3.53, seasonal-ratio error 0.83 → 0.49) for +0.41 % real cost; investment effect is a pure substitution (solar −2.7 GW, offshore wind +0.3 GW).
- ⚠️ WIDTH is not the bid span (span = WIDTH·(K−1)/K), and matching the span does not match strength: the perturbation is `SD = WIDTH·√((1 − 1/K²)/12)`. A fair K comparison holds SD fixed (`[5, 34.6]` vs `[3, 36]`); that run has not been made.
- ⛔ Calibrate WIDTH against the observed bid curve, never the price distribution.

### Hydro operation restrictions (both modes)

`hydro_operation` in `zones.yaml`, on the reservoir after the RoR split (`constraints/hydro_ops.py`); on by default (`hydro.restrictions`). They stop the LP from shutting hydro off for weeks or running at full power week after week.

| constraint | form | value |
|---|---|---|
| `min_hourly_frac` | `p[t] ≥ f · p_nom` | **0.05** |
| `min_daily_frac` | `Σ_day p·w ≥ f · p_nom · H_day` | **0.20** |
| `max_weekly_frac_by_zone` | `Σ_week p·w ≤ f · p_nom · H_week` | SE-N 0.83 · SE-S 0.77 · NO 0.85 · FI 0.80 |
| `bypass_spill` | weekly spill ≥ κ·(production − threshold) | off |

- ⚠️ **`min_hourly_frac` and `min_daily_frac` are locked against the observed bid curve** (production vs price, SE1+SE2): 0.10 → 0.05 cut the error below 10 EUR/MWh from 1.40 to 0.89 GW; relaxing the daily floor to 0.10/0.15 worsened the 20–60 EUR/MWh fit and caused hoarding. Neither comes from the literature — they are guardrails. Do not change without a new measurement.
- The weekly caps and the bypass κ come from Ek Fälth et al. (2025), Sweden only; NO and FI are assumptions argued from reservoir hours. A year-constant cap cannot capture the strong seasonality in the source.
- With cyclic SOC the constraints are consistent only if `min_daily ≤ inflow/(p_nom·H) ≤ max_weekly`; `run.py` prints that ratio per zone before solving (0.46–0.55 for 2024).
- ⚠️ Known leak: rolling windows run Sunday–Saturday while the weekly cap groups ISO weeks, so realized weeks can reach 0.848 against a 0.83 cap (1–4 weeks of 156 per zone). Expansion is exact.
- `bypass_spill` is off because PyPSA bounds spill by the same-snapshot inflow, which can make a high-production/low-inflow week infeasible. Verify with `python scripts/test_hydro_operation.py`.

### Hydro SOC anchor

Both modes start from the same stock: the **measured EC level on 2023-01-02** — `hydro_soc_initial` SE-N/SE-S 0.567 · NO-N/NO-S 0.630 · FI 0.575, 76.6 of 125.4 TWh. Expansion pins SOC[0] = anchor with cyclic SOC (start = end); the rolling horizon uses it as its initial condition.

The rolling horizon has an **attractor around 77 TWh**: runs end there regardless of start, so drift is the distance to the attractor. Anchoring both modes on the measured level (which lies on it) makes them comparable; moving only dispatch to the old 3-year mean gave −7.3 TWh drift and was rejected.

### Inflow, run-of-river and other data-driven profiles

- **Inflow:** NVE/ENTSO-E measured series for SE and NO (`profiles/hydro_inflow.py`); FI uses the parametric spring-flood model, with parameters that are **hand-calibrated** in `config/hydro_params.yaml` (not `data/processed/hydro_params.yaml`, which is auto-generated and must never be used). Verify hydrology after a run with `n.storage_units_t.inflow`: SE-N should peak ~15 GW in May and ~2.6 GW in January.
- **Run-of-river** is a must-run generator split from the reservoir, which keeps the storage volume `p_nom·max_hours`. SE's reported RoR has only ~52 distinct values per year (weekly steps; SvK reports no B11), so SE-N/SE-S/FI get synthetic high-frequency structure (`hydro.ror_hifreq`, σ 0.22), with p_nom locked and weekly energy preserved. The reservoir/RoR split barely affects the aggregate seasonal ratio: the reservoir compensates about two thirds of any change.
- **Nuclear:** existing fleet must-run on its actual availability profile; with `nuclear.add` the existing fleet switches to a synthetic stochastic profile and new reactors are extendable, load-following down to `new_min_load` (0.6 in the 2040 world). `--dry-run` prints must-run fractions per generator.
- **Thermal** is must-run on its actual profile and is not subtracted from load.

### Solver

HiGHS IPM **with crossover** (`run_crossover: "on"`) is required for expansion: without it `p_nom_opt` stays near `p_nom_min` even where investment pays (interior primal, not a vertex).

## Config

- `config/zones.yaml` — the model's DATA: zones with hydro/nuclear capacity, NTC links (+ `links_expansion_overrides`), market connections, technology costs, `cost_scenarios`/`demand_scenarios` definitions, `hydro_operation` parameters, solver settings, simulation period.
- `config/defaults.yaml` — run SETTINGS and their defaults (which parts of the data are used, and how).
- `config/worlds/*.yaml` — `2040_svk_mm` (default for `expand`), `today` (default for `today`, dispatch only).
- `config/experiments/*.yaml` — named deviations, e.g. the seven standard scenarios run by `scripts/run_batch.py`.
- `config/terminal_curves/` — the two active terminal curves (v12 for dispatch, v12_exp73 for expansion); `archive/` holds every earlier version and sweep variant, indexed in `archive/README.md`.
