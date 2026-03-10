"""
Laddar ner produktion och konsumtion från eSett för alla NordPSA-zoner,
år 2023-2025, och sparar som Parquet i data/raw/.

Användning:
    python scripts/fetch_esett.py           # hämtar allt som saknas
    python scripts/fetch_esett.py --force   # skriv över befintliga filer
"""
import argparse
import sys
import time
from pathlib import Path

# Tillåt import av nordpsa-paketet oavsett från vilken katalog scriptet körs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.esett import eSettClient, NORDPSA_ZONES

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS   = [2023, 2024, 2025]
SERIES  = ["production", "consumption"]
SLEEP_S = 0.5   # paus mellan anrop för att inte hammra API:et


def raw_path(series: str, zone: str, year: int) -> Path:
    return RAW_DIR / f"{series}_{zone}_{year}.parquet"


def fetch_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cli = eSettClient()

    total = len(NORDPSA_ZONES) * len(YEARS) * len(SERIES)
    done  = 0

    for zone in NORDPSA_ZONES:
        for year in YEARS:
            for series in SERIES:
                done += 1
                out = raw_path(series, zone, year)

                if out.exists() and not force:
                    print(f"[{done}/{total}] hoppar över {out.name} (finns redan)")
                    continue

                print(f"[{done}/{total}] hämtar {series} {zone} {year} ...", end=" ", flush=True)
                try:
                    df = cli.fetch_year(series, zone, year, resolution="hour")
                except Exception as e:
                    print(f"FEL: {e}")
                    continue

                if df.empty:
                    print("tom respons — hoppar över")
                    continue

                df.to_parquet(out, index=False)
                print(f"OK ({len(df)} rader → {out.name})")
                time.sleep(SLEEP_S)

    print("\nKlart!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="skriv över befintliga filer")
    args = parser.parse_args()
    fetch_all(force=args.force)
