---
name: project-nordpsa-overview
description: NordPSA — Nordic power system model on PyPSA; architecture, zones, key design decisions
metadata:
  type: project
---

Nordic power system capacity expansion + dispatch model built on PyPSA. 6 aggregated zones: SE-N, SE-S, NO-N, NO-S, DK, FI. Covers 2023–2025 at 1h or 3h resolution.

**Why:** Research/analysis tool for studying Nordic electricity system investments and dispatch under continental price signals.

**How to apply:** Understand that all parameters live in `config/zones.yaml` (zones, NTCs, costs) and `config/hydro_params.yaml` (manually calibrated — never overwrite from auto-fit). The `data/processed/hydro_params.yaml` is auto-generated and must never be used as model input.

## Key architecture

- `nordpsa/network.py` — builds PyPSA network (`build_network()` + `hydro_soc_initial_constraint()`)
- `nordpsa/hydro.py` — parametric inflow model: Gaussian spring flood + cosine seasonal + base; fits vs eSett data
- `nordpsa/entsoe.py` — ENTSO-E price/flow fetcher + Elexon BMRS for GB prices (post-Brexit)
- `nordpsa/esett.py` — eSett open data client for Nordic production/load
- `nordpsa/ec.py` — Energy Charts VRE profiles + DE-LU price
- `scripts/run_model.py` — main solve script (args: --resolution, --year, --output, --no-extra-load, --no-expansion)
- `scripts/build_inputs.py` — builds processed parquet from raw data
- `scripts/plot_dispatch.py` — visualization

## Important design decisions

- Solver: HiGHS IPM **with crossover** (required for correct capacity expansion vertex solutions)
- Hydro: `cyclic_state_of_charge=True` + SOC[t=0]=target constraint via `extra_functionality`
- Thermal: must-run Generator (p_min_pu = p_max_pu = profile), NOT subtracted from load
- Market connections: p_min_pu=-1 Generator per cable (import/export valve)
- SE-N and NO-N: no continental market → slack generators for LP feasibility
- additional_load_mw: 1000 MW added per zone (datacenter emulation) — can be disabled with --no-extra-load

## Hydro inflow params (config/hydro_params.yaml)
SE-N: A=10000, mu=135, sigma=22, B=2500, phi=183, C=5037
SE-S: A=500, mu=120, sigma=25, B=700, phi=183, C=1379
NO-N: A=6000, mu=130, sigma=20, B=1800, phi=183, C=4104
NO-S: A=9000, mu=115, sigma=28, B=3500, phi=183, C=10167
FI: A=1800, mu=120, sigma=22, B=500, phi=183, C=1389
