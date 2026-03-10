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

Common solve variants:
```bash
python scripts/run_model.py --resolution 3       # 3h timesteps, 2023-2025
python scripts/run_model.py --resolution 3 --year 2024   # single year
```

Visualize results:
```bash
python scripts/plot_dispatch.py results/res3h_2023_2025/ --resample 7D
python scripts/plot_dispatch.py results/res3h_2023_2025/ --zone SE-S
```

Results are saved to `results/<label>/` with dispatch CSVs, flows, prices, hydro SoC, and `network.nc`.

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
  hydro_params.yaml       (fitted inflow model parameters)
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
| Generator | nuclear | Load-following: p_min_pu = 0.6 × p_max_pu |
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

**IPM with crossover:** Solver must use `run_crossover: "on"` for capacity expansion runs. Without crossover, p_nom_opt stays near p_nom_min even when investment is profitable (interior-point primal solution, not a vertex).

**Thermal as must-run Generator:** `p_min_pu = p_max_pu = profile/p_nom`. Dispatch is fully determined by data; optimizer has no freedom. Thermal is NOT subtracted from load.

**Hydro inflow model:** Fitted once per zone (mean over 2023-2025). Same seasonal curve repeated each year — no inter-year reservoir carryover. Cyclic SOC means reservoir refills annually.

**p_nom_max bounds:** All extendable generators have finite `p_nom_max_mw` in config (20k for nuclear/gas, 50k for VRE per zone). Without these, HiGHS sees ~3e10 column bounds and prints scaling warnings (harmless but ugly).

## Config

All parameters in `config/zones.yaml`:
- Zone definitions with hydro/nuclear existing capacity
- NTC links between zones
- Market connection capacities
- Technology costs (overnight, lifetime, VOM, extendable flag, p_nom_max)
- Solver settings (HiGHS IPM + crossover)
- Simulation period and resolution
