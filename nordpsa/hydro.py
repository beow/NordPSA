"""
Parametrisk inflödesmodell för vattenkraft per zon.

Modell (daglig upplösning):
    inflow(doy) = A * exp(-((doy - mu) / sigma)^2)   # vårflod (Gausskurva)
                + B * cos(2π * (doy - phi) / 365)     # säsongskomponent
                + C                                    # basflöde

Enheter: MW (medeleffekt per timme)
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
YEARS         = [2023, 2024, 2025]


def _model(doy: np.ndarray, A: float, mu: float, sigma: float,
           B: float, phi: float, C: float) -> np.ndarray:
    spring   = A * np.exp(-((doy - mu) / sigma) ** 2)
    seasonal = B * np.cos(2 * np.pi * (doy - phi) / 365)
    return spring + seasonal + C


def load_actual_hydro(zone: str) -> pd.Series:
    """
    Laddar faktisk vattenkraftproduktion (MW, timvis) för alla år.
    Returnerar dagliga medelvärden indexerade på dag-på-året (1-365),
    medlade över alla tillgängliga år.
    """
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"production_{zone}_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
        frames.append(df.set_index("timestampUTC")["hydro"])

    if not frames:
        raise FileNotFoundError(f"Ingen produktionsdata för {zone}")

    ts = pd.concat(frames).sort_index()
    daily = ts.resample("D").mean()
    daily.index = daily.index.dayofyear
    return daily.groupby(daily.index).mean()


def fit_zone(zone: str, actual: pd.Series) -> Dict[str, float]:
    """Fittar modellparametrar mot faktisk daglig produktion för en zon."""
    doy = actual.index.values.astype(float)
    y   = actual.values

    C0     = float(np.percentile(y, 10))
    A0     = float(np.percentile(y, 90) - C0)
    # Startgissning: vårflod topp kring dag 120 (maj), säsongsmin kring februari (phi≈65)
    p0     = [A0, 120.0, 30.0, A0 * 0.3, 65.0, C0]
    bounds = (
        [0,   60,  10,  0,    0,   0      ],
        [np.inf, 180, 90, np.inf, 180, np.inf],
    )

    try:
        popt, _ = curve_fit(_model, doy, y, p0=p0, bounds=bounds, maxfev=10_000)
    except RuntimeError:
        popt = p0  # fallback till startgissning

    A, mu, sigma, B, phi, C = popt
    return {
        "A": round(float(A), 2),
        "mu": round(float(mu), 2),
        "sigma": round(float(sigma), 2),
        "B": round(float(B), 2),
        "phi": round(float(phi), 2),
        "C": round(float(C), 2),
    }


def fit_and_save_all(zones: list[str]) -> Dict[str, Dict[str, float]]:
    """Fittar parametrar för alla zoner med hydro och sparar till hydro_params.yaml."""
    all_params: Dict[str, Dict[str, float]] = {}

    for zone in zones:
        path = RAW_DIR / f"production_{zone}_2023.parquet"
        if not path.exists():
            continue
        # Kolla om zonen har hydro
        df_check = pd.read_parquet(path)
        if df_check["hydro"].fillna(0).sum() == 0:
            print(f"  {zone}: ingen vattenkraft — hoppar över")
            continue

        print(f"  Fittar inflödesmodell för {zone} ...", end=" ", flush=True)
        try:
            actual = load_actual_hydro(zone)
            params = fit_zone(zone, actual)
            all_params[zone] = params
            print(
                f"OK  C={params['C']:.0f} MW  "
                f"A={params['A']:.0f} MW  "
                f"mu=dag {params['mu']:.0f}"
            )
        except Exception as e:
            print(f"FEL: {e}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "hydro_params.yaml"
    with open(out, "w") as f:
        yaml.dump(all_params, f, default_flow_style=False, sort_keys=True)
    print(f"  → {out.name}")
    return all_params


def load_nve_inflow(zone: str, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Laddar NVE faktisk veckovist inflöde (MW) och expanderar till snapshot-index
    via forward-fill (konstantvärde inom varje vecka, energikonservativt).

    Kräver data/raw/inflow_nve_{zone}_{year}.parquet för varje år i snapshots.
    Finns bara för NO-N och NO-S.
    """
    years = sorted(set(snapshots.year))
    frames = []
    for year in years:
        path = RAW_DIR / f"inflow_nve_{zone}_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Saknar NVE-inflödesdata: {path.name} "
                f"— kör 'python scripts/fetch_nve.py' först."
            )
        df = pd.read_parquet(path)
        df["week_start"] = pd.to_datetime(df["week_start"], utc=True).dt.tz_localize(None)
        frames.append(df.set_index("week_start")["inflow_mw"])

    weekly = pd.concat(frames).sort_index()

    # Expandera till snapshot-frekvens: lägg snapshots i det veckoglesade indexet
    # och ffill så att varje snapshot får veckans MW-värde
    combined_idx = weekly.index.union(snapshots)
    result = weekly.reindex(combined_idx).ffill().reindex(snapshots).fillna(0.0)
    result.name = "inflow_mw"
    return result


def load_nve_ror(zone: str, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Laddar run-of-river (B11) timprofil (MW) och reindexerar till snapshot-index.

    Returnerar must-run-profilen för strömkraft. Saknas filen returneras nollserie
    (zonen har ingen separat RoR-data). Resamplas till snapshot-frekvens via medel.
    """
    years = sorted(set(snapshots.year))
    frames = []
    for year in years:
        path = RAW_DIR / f"ror_nve_{zone}_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True).dt.tz_localize(None)
        frames.append(df.set_index("timestampUTC")["ror_mw"])

    if not frames:
        return pd.Series(0.0, index=snapshots, name="ror_mw")

    hourly = pd.concat(frames).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    # Resampla till snapshot-frekvens (medel), reindexera, fyll luckor
    dt_h = (snapshots[1] - snapshots[0]).total_seconds() / 3600
    resampled = hourly.resample(f"{int(dt_h)}h").mean()
    result = resampled.reindex(snapshots).ffill().bfill().fillna(0.0)
    result.name = "ror_mw"
    return result


_MODEL_KEYS = ("A", "mu", "sigma", "B", "phi", "C")


def inflow_timeseries(params: Dict[str, float],
                      timestamps: pd.DatetimeIndex,
                      annual_scales: Dict[int, float] | None = None,
                      peak_gamma: float | None = None,
                      target_annual_twh: Dict[int, float] | None = None) -> pd.Series:
    """
    Genererar inflödestidsserie (MW) för ett godtyckligt tidsstämpelindex.
    Om annual_scales är givet skalas inflödet per kalenderår mot faktisk produktion.

    peak_gamma:        potens-transform av den parametriska formen (>1 spetsar
                       vårfloden; FI-kalibrering mot ENTSO-E+eSett-härledd
                       tillrinning gav γ≈1.5).
    target_annual_twh: dict {år: TWh} → varje kalenderår normaliseras (efter
                       ev. peakning) så att årssumman exakt matchar faktisk
                       tillrinning. Ersätter annual_scales när angiven.

    Används av network.py för att sätta inflow_t på StorageUnits.
    """
    mparams = {k: params[k] for k in _MODEL_KEYS if k in params}
    doy     = timestamps.dayofyear.values.astype(float)
    values  = _model(doy, **mparams)
    values  = np.maximum(values, 0.0)
    if peak_gamma is not None and peak_gamma != 1.0:
        values = values ** peak_gamma
    result = pd.Series(values, index=timestamps, name="inflow_mw")

    if target_annual_twh:
        # Steg-timmar ur tidsstämpelsteget (uniformt rutnät, t.ex. 3h-upplösning).
        step_h = ((timestamps[1] - timestamps[0]).total_seconds() / 3600.0
                  if len(timestamps) > 1 else 1.0)
        for year, twh in target_annual_twh.items():
            mask = timestamps.year == year
            cur_mwh = result.loc[mask].sum() * step_h
            if cur_mwh > 0:
                result.loc[mask] *= twh * 1e6 / cur_mwh
    elif annual_scales:
        for year, scale in annual_scales.items():
            result.loc[timestamps.year == year] *= scale
    return result
