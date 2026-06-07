"""
Bygger värmelastprofiler per NordPSA-zon (When2Heat + Open-Meteo) →
data/processed/heat_load.parquet  (total termisk MWh/h, 6 zoner, 2023-2025).

Kräver att scripts/fetch_openmeteo.py körts (cachade ort-temperaturer i data/raw/).
Metodik: se nordpsa/heat.py. Kalibreras mot config/heat_annual_twh.yaml.

Sparar även komponenter (space/water) i heat_components.parquet för analys.

Användning:
    python scripts/build_heat.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa import heat

ROOT      = Path(__file__).resolve().parents[1]
PROC_DIR  = ROOT / "data" / "processed"
ANNUAL    = ROOT / "config" / "heat_annual_twh.yaml"


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    annual = yaml.safe_load(open(ANNUAL))["heat_annual_twh"]

    print("Bygger zon-temperatur (befolkningsvägd) ...")
    temp = heat.zone_temperature()
    print("Bygger värmeprofiler (When2Heat-sigmoid + DHW + timfaktorer) ...")
    raw  = heat.heat_profiles_raw(temp)
    hl   = heat.build_heat_load(annual)

    hl.to_parquet(PROC_DIR / "heat_load.parquet")
    print(f"  → heat_load.parquet  ({len(hl)} rader, {len(hl.columns)} zoner)")

    # komponenter (space/water, normaliserade) — för analys
    comp = pd.concat({z: raw[z][["space", "water"]] for z in raw}, axis=1)
    comp.to_parquet(PROC_DIR / "heat_components.parquet")
    print(f"  → heat_components.parquet  (space/water normaliserade)")

    print("\nÅrsvolym per zon (TWh, kalibreringskoll):")
    for z in hl.columns:
        for yr, grp in hl[z].groupby(hl[z].index.year):
            pass
        twh = {yr: g.sum() / 1e6 for yr, g in hl[z].groupby(hl[z].index.year)}
        print(f"  {z:<6} " + "  ".join(f"{y}:{v:.1f}" for y, v in twh.items()))


if __name__ == "__main__":
    main()
