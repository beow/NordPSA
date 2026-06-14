"""
Hämtar syntetiska offshore-vindprofiler per zon från Renewables.ninja
och sparar som Parquet (timvis kapacitetsfaktor, UTC).

Profilerna ger EXPANDERBAR havsbaserad vind i alla zoner — även de utan
offshore idag. build_inputs.py kalibrerar dem mot DK:s faktiska produktion
(DK behåller sin egen faktiska profil; ninja-DK används bara som
kalibreringsankare).

En representativ koordinat per zon, vald utifrån faktiska/planerade
havsområden. NO-N är flytande havsvind (Norska havet) — tidigt skede och
mest spekulativt.

Utdata:
  data/raw/offshore_ninja_{ZONE}_{year}.parquet  — kolumn 'cf' (0–1)

Kräver token:  export NINJA_TOKEN=...   (https://www.renewables.ninja/)

Användning:
    python scripts/fetch_ninja.py
    python scripts/fetch_ninja.py --force
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.ninja import RenewablesNinjaClient

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS   = [2023, 2024, 2025]
# Ninja: 50 req/timme, burst 6. 6 zoner × 3 år = 18 req → snäll paus.
SLEEP_S = 15.0

# Representativ offshore-koordinat per zon (lat, lon, beskrivning).
SITES = {
    "DK":   (56.0,  7.5,  "Nordsjön (Horns Rev-omr.) — kalibreringsankare"),
    "SE-S": (55.0, 14.5,  "Östersjön (Kriegers Flak / S. Midsjöbanken)"),
    "SE-N": (65.0, 22.5,  "Bottenviken (utpekade områden Norrbotten)"),
    "NO-S": (56.8,  5.0,  "Nordsjön (Sørlige Nordsjø II)"),
    "NO-N": (67.5, 11.5,  "Norska havet — flytande (NVE-områden, tidigt skede)"),
    "FI":   (62.9, 21.0,  "Bottenhavet (Korsnäs m.fl.)"),
}


def raw_path(zone: str, year: int) -> Path:
    return RAW_DIR / f"offshore_ninja_{zone}_{year}.parquet"


def fetch_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cli = RenewablesNinjaClient()  # kastar tydligt fel om NINJA_TOKEN saknas

    total = len(SITES) * len(YEARS)
    done  = 0
    for zone, (lat, lon, desc) in SITES.items():
        print(f"\n=== {zone}: {desc}  ({lat}, {lon}) ===")
        for year in YEARS:
            done += 1
            out = raw_path(zone, year)
            if out.exists() and not force:
                print(f"[{done}/{total}] hoppar över {out.name} (finns redan)")
                continue

            print(f"[{done}/{total}] hämtar {zone} {year} ...", end=" ", flush=True)
            try:
                cf = cli.fetch_wind(lat, lon, year)
            except Exception as e:
                print(f"FEL: {e}")
                continue

            if cf is None or cf.empty:
                print("tom respons — hoppar över")
                continue

            cf.to_frame().to_parquet(out)
            print(f"OK ({len(cf)} timmar, medel-CF={cf.mean():.3f})")
            time.sleep(SLEEP_S)

    print("\nKlart!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="skriv över befintliga filer")
    args = parser.parse_args()
    fetch_all(force=args.force)
