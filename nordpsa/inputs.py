"""Indata: config, förberedda tidsserier, snapshots och resampling."""
from __future__ import annotations

import pandas as pd
import yaml

from nordpsa.settings import CONFIG_DIR, ROOT

PROC_DIR    = ROOT / "data" / "processed"
CONFIG_PATH = CONFIG_DIR / "zones.yaml"


def load_config() -> dict:
    """Modellens data: zoner, NTC, kostnader, scenariodefinitioner, lösare."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_inputs() -> dict:
    """Laddar alla förberedda indata från data/processed/."""
    load_df   = pd.read_parquet(PROC_DIR / "load.parquet")
    vre       = pd.read_parquet(PROC_DIR / "vre_profiles.parquet")
    nuclear   = pd.read_parquet(PROC_DIR / "nuclear_profile.parquet")
    thermal   = pd.read_parquet(PROC_DIR / "thermal_profile.parquet")
    prices_df = pd.read_parquet(PROC_DIR / "market_prices.parquet")

    with open(PROC_DIR / "vre_pnom.yaml") as f:
        vre_noms = yaml.safe_load(f)
    with open(CONFIG_DIR / "hydro_params.yaml") as f:
        hydro_params = yaml.safe_load(f)

    # Fjärrvärme-värmebehov (valfritt; gitignorerad, byggs av scripts/build_heat.py)
    heat_path = PROC_DIR / "heat_load.parquet"
    heat_load = pd.read_parquet(heat_path) if heat_path.exists() else None

    # EV-laddningsprofiler (valfritt; byggs av build_ev_profiles i build_inputs.py)
    ev_path     = PROC_DIR / "ev_profiles.parquet"
    ev_profiles = pd.read_parquet(ev_path) if ev_path.exists() else None

    # Sätt UTC-index och ta bort timezone (PyPSA kräver tz-naivt)
    dfs = [load_df, vre, nuclear, thermal, prices_df]
    if heat_load is not None:
        dfs.append(heat_load)
    if ev_profiles is not None:
        dfs.append(ev_profiles)
    for df in dfs:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    market_prices = {col: prices_df[col] for col in prices_df.columns}

    return dict(
        load=load_df, vre_profiles=vre, vre_noms=vre_noms,
        nuclear_profile=nuclear, thermal_profile=thermal,
        hydro_params=hydro_params, market_prices=market_prices,
        heat_load=heat_load, ev_profiles=ev_profiles,
    )


def make_snapshots(cfg: dict, resolution: int, year: int | None) -> pd.DatetimeIndex:
    """Snapshot-index för hela config-perioden eller ett enskilt år."""
    start = pd.Timestamp(cfg["snapshots"]["start"], tz="UTC")
    end   = pd.Timestamp(cfg["snapshots"]["end"],   tz="UTC") - pd.Timedelta(hours=1)

    if year is not None:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end   = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")

    idx = pd.date_range(start, end, freq=f"{resolution}h")
    return idx.tz_localize(None)  # PyPSA kräver timezone-naiva snapshots


def resample_inputs(inputs: dict, snapshots: pd.DatetimeIndex, resolution: int) -> dict:
    """Resamplar alla tidsserier till snapshot-frekvensen (medelvärde)."""
    freq = f"{resolution}h"
    out  = {}
    for key in ("load", "vre_profiles", "nuclear_profile", "thermal_profile"):
        out[key] = inputs[key].resample(freq).mean().reindex(snapshots).ffill()
    out["market_prices"] = {
        bzn: s.resample(freq).mean().reindex(snapshots).ffill()
        for bzn, s in inputs["market_prices"].items()
    }
    out["vre_noms"]     = inputs["vre_noms"]
    out["hydro_params"] = inputs["hydro_params"]
    hl = inputs.get("heat_load")
    out["heat_load"] = (hl.resample(freq).mean().reindex(snapshots).ffill()
                        if hl is not None else None)
    ev = inputs.get("ev_profiles")
    out["ev_profiles"] = (ev.resample(freq).mean().reindex(snapshots).ffill()
                          if ev is not None else None)
    return out


def boost_capfac(profiles: pd.DataFrame, increase: float, carrier: str) -> pd.DataFrame:
    """Höjer `carrier`-kolumnernas (wind_onshore/wind_offshore) kapacitetsfaktor med
    relativ andel `increase` (0.1 = +10 %) via en olinjär potens-transform per zon:

        x   = cf / cf_max            (cf_max = profilens max per zon, ~0.8 pga
                                      geografisk spridning — INTE installerad effekt)
        cf' = cf_max · x^γ,  γ < 1

    Konkav (γ<1) → lyfter låga/mellan effektnivåer mest (relativt; "mer effektiv
    vid lätta vindar"), fixerar bägge ändar: cf=0→0 (vindstilla) och cf=cf_max→cf_max
    (märkeffekt oförändrad). cf' överskrider aldrig cf_max (geografisk envelopp).

    γ löses per zon med bisektion (mean är monotont avtagande i γ) så att
    mean(cf') = (1+increase)·mean(cf). Speglar 2040:s nybyggnadsflotta.
    """
    if increase <= 0:
        return profiles
    out = profiles.copy()
    print(f"Höjer {carrier}-CF med {increase*100:.0f}% (olinjär potens-transform):")
    suffix = f"_{carrier}"
    for col in [c for c in profiles.columns if c.endswith(suffix)]:
        cf     = profiles[col].astype(float)
        cf_max = float(cf.max())
        if cf_max <= 0:
            continue
        x      = (cf / cf_max).clip(0.0, 1.0)
        mean0  = float(cf.mean())
        target = mean0 * (1.0 + increase)
        # Tak: γ→0 ger cf'→cf_max för alla cf>0
        max_mean = cf_max * float((cf > 0).mean())
        zone = col.replace(suffix, "")
        if target > max_mean:
            print(f"  Varning: {zone} — mål {target:.3f} > tak {max_mean:.3f}; klampar till taket")
            target = max_mean
        # Bisektion i γ ∈ (eps, 1]; mean(γ) avtagande → m>target ⇒ höj γ
        lo, hi, g = 1e-4, 1.0, 1.0
        for _ in range(60):
            g = 0.5 * (lo + hi)
            m = float((cf_max * x.pow(g)).mean())
            if m > target:
                lo = g
            else:
                hi = g
        cf1 = cf_max * x.pow(g)
        out[col] = cf1
        print(f"  {zone:6s} CF {mean0:.3f}→{float(cf1.mean()):.3f} "
              f"(+{100*(cf1.mean()/mean0-1):.1f}%), γ={g:.3f}, cf_max={cf_max:.3f} (oförändrad)")
    return out
