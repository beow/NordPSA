"""
Hämtar timvis 2m-temperatur per befolkningsort från Open-Meteo Historical Weather
API (ERA5-baserad reanalys, ~11 km, gratis, ingen nyckel) för 2023-2025.

Orterna definieras i config/heat_cities.yaml (per NordPSA-zon). Temperaturen används
sedan av nordpsa/heat.py för att bygga befolkningsvägda värmelastprofiler per zon
(When2Heat-metodik).

Utdata: data/raw/openmeteo_t2m_{ort}.parquet  (kolumn 'temp', UTC-timindex)

Användning:
    python scripts/fetch_openmeteo.py                 # 2023-2025, cachar per ort
    python scripts/fetch_openmeteo.py --force         # ignorera cache
    python scripts/fetch_openmeteo.py --start 2024-01-01 --end 2024-12-31
"""
import argparse
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw"
CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "heat_cities.yaml"
API_URL  = "https://archive-api.open-meteo.com/v1/archive"
SLEEP_S  = 0.5   # snäll mot API:t


def slug(name: str) -> str:
    """Ortnamn → filsäker ascii-slug (Umeå → umea, Mo i Rana → mo_i_rana)."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return s.lower().replace(" ", "_")


def raw_path(name: str) -> Path:
    return RAW_DIR / f"openmeteo_t2m_{slug(name)}.parquet"


def fetch_city(name: str, lat: float, lon: float, start: str, end: str,
               force: bool) -> pd.Series:
    """Hämtar (och cachar) timtemperatur för en ort över [start, end]."""
    path = raw_path(name)
    if path.exists() and not force:
        return pd.read_parquet(path)["temp"]

    print(f"    {name} ({lat:.2f},{lon:.2f}): hämtar {start}..{end} ...", end=" ", flush=True)
    resp = requests.get(API_URL, params={
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "hourly":    "temperature_2m",
        "timezone":  "UTC",
    }, timeout=120)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    s = pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), name="temp")
    s = s.astype(float)
    s.to_frame().to_parquet(path)
    print(f"OK ({len(s)} h, medel {s.mean():.1f}°C)")
    time.sleep(SLEEP_S)
    return s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end",   default="2025-12-31")
    p.add_argument("--force", action="store_true", help="Ignorera cache")
    args = p.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cities = yaml.safe_load(open(CFG_PATH))["heat_cities"]

    n_total = sum(len(v) for v in cities.values())
    print(f"Hämtar 2m-temperatur för {n_total} orter ({args.start}..{args.end}) från Open-Meteo:")
    done = 0
    for zone, orter in cities.items():
        print(f"  {zone}:")
        for c in orter:
            fetch_city(c["namn"], c["lat"], c["lon"], args.start, args.end, args.force)
            done += 1
    print(f"Klart — {done} orter cachade i {RAW_DIR}/")


if __name__ == "__main__":
    main()
