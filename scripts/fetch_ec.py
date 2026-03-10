"""
Hämtar dansk produktion (vind + sol) och kontinentala day-ahead-priser
från Energy Charts och sparar som Parquet.

Utdata:
  data/raw/production_DK_ec_{year}.parquet  — dansk produktion
  data/raw/price_market_{year}.parquet      — DE-LU day-ahead-priser

Användning:
    python scripts/fetch_ec.py           # hämtar allt som saknas
    python scripts/fetch_ec.py --force   # skriv över befintliga filer
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.ec import EnergyChartsClient, EC_COUNTRY_DK

RAW_DIR    = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS      = [2023, 2024, 2025]
SLEEP_S    = 1.0    # Energy Charts är ett gratisAPI — var snäll mot servern
PRICE_BZN  = "DE-LU"   # Tyska day-ahead-priser som proxy för kontinentalt marknadspris


def raw_path_dk(year: int) -> Path:
    return RAW_DIR / f"production_DK_ec_{year}.parquet"


def raw_path_price(year: int) -> Path:
    return RAW_DIR / f"price_market_{year}.parquet"


def fetch_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cli = EnergyChartsClient()

    # --- Dansk produktion ---
    print("=== Dansk produktion ===")
    for i, year in enumerate(YEARS, 1):
        out = raw_path_dk(year)
        if out.exists() and not force:
            print(f"[{i}/{len(YEARS)}] hoppar över {out.name} (finns redan)")
            continue

        print(f"[{i}/{len(YEARS)}] hämtar DK {year} ...", end=" ", flush=True)
        try:
            df = cli.fetch_year(EC_COUNTRY_DK, year)
        except Exception as e:
            print(f"FEL: {e}")
            continue

        if df is None or df.empty:
            print("tom respons — hoppar över")
            continue

        df.to_parquet(out)
        print(f"OK ({len(df)} rader)")
        time.sleep(SLEEP_S)

    # --- Kontinentala marknadspriser (DE-LU day-ahead) ---
    print("\n=== Kontinentala day-ahead-priser (DE-LU) ===")
    for i, year in enumerate(YEARS, 1):
        out = raw_path_price(year)
        if out.exists() and not force:
            print(f"[{i}/{len(YEARS)}] hoppar över {out.name} (finns redan)")
            continue

        print(f"[{i}/{len(YEARS)}] hämtar {PRICE_BZN} priser {year} ...", end=" ", flush=True)
        try:
            s = cli.fetch_price_year(PRICE_BZN, year)
        except Exception as e:
            print(f"FEL: {e}")
            continue

        s.to_frame().to_parquet(out)
        print(f"OK ({len(s)} timmar, medel={s.mean():.1f} EUR/MWh)")
        if i < len(YEARS):
            time.sleep(SLEEP_S)

    print("\nKlart!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="skriv över befintliga filer")
    args = parser.parse_args()
    fetch_all(force=args.force)
