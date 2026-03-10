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


def inflow_timeseries(params: Dict[str, float],
                      timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Genererar inflödestidsserie (MW) för ett godtyckligt tidsstämpelindex.
    Används av network.py för att sätta inflow_t på StorageUnits.
    """
    doy    = timestamps.dayofyear.values.astype(float)
    values = _model(doy, **params)
    values = np.maximum(values, 0.0)
    return pd.Series(values, index=timestamps, name="inflow_mw")
