"""
Bygger alla förberedda indata för NordPSA från rådata i data/raw/.

Utdata (data/processed/):
  load.parquet             — lastprofil per zon (MW, timvis)
  vre_profiles.parquet     — p_max_pu för vind och sol per zon
  vre_pnom.yaml            — estimerade installerade effekter för vind/sol
  nuclear_profile.parquet  — kärnkraftsprofil p_max_pu per zon
  thermal_profile.parquet  — must-run termisk (MW) per zon
  hydro_params.yaml        — fittade inflödesparametrar

Användning:
    python scripts/build_inputs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.esett import NORDPSA_ZONES
from nordpsa import hydro as hydro_mod

RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CONFIG_PATH   = Path(__file__).resolve().parents[1] / "config" / "zones.yaml"
YEARS         = [2023, 2024, 2025]

# Standardperiod: hela 2023-2025 i UTC
PERIOD_START = pd.Timestamp("2023-01-01 00:00", tz="UTC")
PERIOD_END   = pd.Timestamp("2025-12-31 23:00", tz="UTC")


def _trim(df: pd.DataFrame) -> pd.DataFrame:
    """Klipper DataFrame till standardperioden."""
    return df.loc[PERIOD_START:PERIOD_END]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_raw(series: str, zone: str) -> pd.DataFrame:
    """Laddar och sammanfogar rådata för alla år för en zon, index = UTC."""
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"{series}_{zone}_{year}.parquet"
        df = pd.read_parquet(path)
        df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values("timestampUTC")
    return df.set_index("timestampUTC")


# ---------------------------------------------------------------------------
# Last
# ---------------------------------------------------------------------------

def build_load() -> pd.DataFrame:
    """
    Lastprofil per zon (MW).
    DK: hämtas från Energy Charts 'load'-kolumn (eSett saknar zonuppdelad last).
    Övriga: eSett consumption.total (abs-värde).
    """
    print("Bygger lastprofil ...")
    dk_ec  = _load_dk_ec()
    result = {}
    for zone in NORDPSA_ZONES:
        if zone == "DK":
            result[zone] = dk_ec["load"].abs()
        else:
            df = load_raw("consumption", zone)
            result[zone] = df["total"].abs()

    out = _trim(pd.DataFrame(result))
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "load.parquet")
    print(f"  → load.parquet  ({len(out)} rader, {len(out.columns)} zoner)")
    return out


# ---------------------------------------------------------------------------
# VRE-profiler (vind + sol)
# ---------------------------------------------------------------------------

def _load_dk_ec() -> pd.DataFrame:
    """Laddar och sammanfogar Energy Charts-data för DK (alla år)."""
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"production_DK_ec_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"EC-data saknas: {path}\n"
                "Kör 'python scripts/fetch_ec.py' först."
            )
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames).sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_vre_profiles() -> pd.DataFrame:
    """
    Normaliserade kapacitetsfaktorer (p_max_pu) för vind (on/offshore) och sol.
    DK: hämtas från Energy Charts. Övriga: eSett.
    p_nom estimeras som 99:e percentilen av faktisk produktion.
    Kolumnnamn: {zon}_wind_onshore, {zon}_wind_offshore, {zon}_solar
    """
    print("Bygger VRE-profiler ...")
    result   = {}
    vre_noms = {}

    dk_ec = _load_dk_ec()

    for zone in NORDPSA_ZONES:
        if zone == "DK":
            onshore  = dk_ec["wind"].fillna(0)
            offshore = dk_ec.get("windOffshore", pd.Series(0.0, index=dk_ec.index)).fillna(0)
            solar    = dk_ec["solar"].fillna(0)
            source   = "EC"
        else:
            df       = load_raw("production", zone)
            onshore  = df["wind"].fillna(0)
            offshore = df["windOffshore"].fillna(0)
            solar    = df["solar"].fillna(0)
            source   = "eSett"

        def _normalise(series: pd.Series, fleet_factor: float = 1.0) -> tuple[pd.Series, float]:
            """p_nom = p99 / fleet_factor; p_max_pu = series / p_nom.

            fleet_factor < 1 ger p_nom > p99, vilket speglar att hela flottan
            sällan körs på max samtidigt.
            wind_onshore_fleet_factor=0.7: p99 motsvarar 70% av installerad kapacitet.
            solar_fleet_factor=0.85: sol har smalare fördelning, högre samordning.
            """
            p_nom = float(np.percentile(series, 99)) / fleet_factor
            if p_nom > 1:
                return (series / p_nom).clip(0, 1), p_nom
            return series * 0.0, 0.0

        onshore_pu,  on_nom  = _normalise(onshore,  fleet_factor=0.70)  # wind_onshore_fleet_factor
        offshore_pu, off_nom = _normalise(offshore, fleet_factor=0.80)  # wind_offshore_fleet_factor
        solar_pu,    sol_nom = _normalise(solar,     fleet_factor=0.85)  # solar_fleet_factor

        result[f"{zone}_wind_onshore"]  = onshore_pu
        result[f"{zone}_wind_offshore"] = offshore_pu
        result[f"{zone}_solar"]         = solar_pu

        vre_noms[zone] = {
            "wind_onshore_p_nom_mw":  round(on_nom),
            "wind_offshore_p_nom_mw": round(off_nom),
            "solar_p_nom_mw":         round(sol_nom),
        }
        print(
            f"  {zone} ({source}): "
            f"onshore={on_nom:.0f} MW  offshore={off_nom:.0f} MW  sol={sol_nom:.0f} MW"
        )

    out = _trim(pd.DataFrame(result))
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "vre_profiles.parquet")
    print(f"  → vre_profiles.parquet")

    with open(PROCESSED_DIR / "vre_pnom.yaml", "w") as f:
        yaml.dump(vre_noms, f, default_flow_style=False)
    print(f"  → vre_pnom.yaml")

    return out


# ---------------------------------------------------------------------------
# Kärnkraft
# ---------------------------------------------------------------------------

def build_nuclear_profile(cfg: dict) -> pd.DataFrame:
    """
    p_max_pu = faktisk_produktion / nuclear_p_nom per zon.
    Zoner utan kärnkraft (p_nom = 0) får p_max_pu = 0.
    """
    print("Bygger kärnkraftsprofil ...")
    result = {}

    for zone, zcfg in cfg["zones"].items():
        p_nom = zcfg.get("nuclear_p_nom_mw", 0)
        df    = load_raw("production", zone)

        if p_nom == 0:
            result[zone] = pd.Series(0.0, index=df.index, dtype=float)
            continue

        p_max_pu = (df["nuclear"] / p_nom).clip(0, 1.05)
        result[zone] = p_max_pu
        print(
            f"  {zone}: p_nom={p_nom} MW, "
            f"medel={p_max_pu.mean():.2f}, "
            f"min={p_max_pu.min():.2f}"
        )

    out = pd.DataFrame(result)
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "nuclear_profile.parquet")
    print(f"  → nuclear_profile.parquet")
    return out


# ---------------------------------------------------------------------------
# Termisk (must-run)
# ---------------------------------------------------------------------------

def build_thermal_profile() -> pd.DataFrame:
    """
    Must-run termisk produktion (MW) = thermal + other per zon.
    Sparas som absoluta MW; network.py beräknar p_nom = max per zon.
    """
    print("Bygger termisk profil ...")
    result = {}

    for zone in NORDPSA_ZONES:
        df      = load_raw("production", zone)
        thermal = df["thermal"].fillna(0) + df["other"].fillna(0)
        result[zone] = thermal.clip(lower=0)

    out = pd.DataFrame(result)
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "thermal_profile.parquet")
    print(f"  → thermal_profile.parquet  ({len(out)} rader)")
    return out


# ---------------------------------------------------------------------------
# Hydro
# ---------------------------------------------------------------------------
# Hydro-parametrar (vårflodsprofil) är manuellt kalibrerade och lagras i
# config/hydro_params.yaml — de skrivs INTE över av build_inputs.py.
# Se nordpsa/hydro.py för modellbeskrivning.


# ---------------------------------------------------------------------------
# Marknadspriser
# ---------------------------------------------------------------------------

def build_market_price() -> pd.Series:
    """
    Sammanfogar rådata för kontinentala day-ahead-priser (DE-LU) 2023-2025.
    Klipps till standardperioden och sparas som market_price.parquet.
    """
    print("Bygger marknadsprisserie ...")
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"price_market_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Prisdata saknas: {path}\n"
                "Kör 'python scripts/fetch_ec.py' först."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        frames.append(df)

    s = pd.concat(frames).sort_index()["price_eur_mwh"]
    s = s.loc[PERIOD_START:PERIOD_END]
    s.index.name = "time"

    # Fyll ev. glapp med forward-fill (max 2h) och sedan medelvärde
    s = s.resample("h").mean().ffill(limit=2).fillna(s.mean())

    s.to_frame().to_parquet(PROCESSED_DIR / "market_price.parquet")
    print(f"  → market_price.parquet  ({len(s)} timmar, medel={s.mean():.1f} EUR/MWh)")
    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    build_load()
    build_vre_profiles()
    build_nuclear_profile(cfg)
    build_thermal_profile()
    build_market_price()

    print("\nKlart! Alla indata sparade i data/processed/")
