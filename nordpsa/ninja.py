"""
Renewables.ninja API-klient — syntetiska offshore-vindprofiler.

Används för havsbaserad vind i zoner som saknar (tillräcklig) faktisk
produktion, så att kapacitetsexpansion kan bygga offshore där det inte
finns idag. Ger timvis kapacitetsfaktor (0–1) för en given koordinat.

Profilerna kalibreras mot DK:s faktiska produktion i build_inputs.py
(reanalys-CF från ninja är brutto och vindbias-känslig — kalibreringen mot
den enda zon med verklig offshore-produktion förankrar nivån).

Kräver API-token (NINJA_TOKEN i miljön):
  1. Skapa konto på https://www.renewables.ninja/
  2. Kopiera token från profilsidan
  3. export NINJA_TOKEN=...
"""
import os
from typing import Optional

import pandas as pd
import requests

# Greenfield offshore-turbin (hög effekt, modern klass). Nivån absorberas
# ändå av DK-kalibreringen — formen/säsongsvariationen är det viktiga.
DEFAULT_TURBINE = "Vestas V164 9500"
DEFAULT_HEIGHT  = 140     # navhöjd, m
# MERRA-2 uppdateras mer löpande än ninjas ERA5 → bättre täckning 2024–2025.
DEFAULT_DATASET = "merra2"


class RenewablesNinjaClient:
    BASE_URL = "https://www.renewables.ninja/api"

    def __init__(self, token: Optional[str] = None, timeout: int = 60):
        self.token = token or os.environ.get("NINJA_TOKEN")
        if not self.token:
            raise ValueError(
                "Renewables.ninja-token saknas. Sätt NINJA_TOKEN i miljön "
                "(skapa konto på https://www.renewables.ninja/ och kopiera "
                "token från profilsidan)."
            )
        self.timeout = timeout
        self.sess    = requests.Session()
        self.sess.headers["Authorization"] = f"Token {self.token}"

    def fetch_wind(self,
                   lat:     float,
                   lon:     float,
                   year:    int,
                   *,
                   turbine: str   = DEFAULT_TURBINE,
                   height:  float = DEFAULT_HEIGHT,
                   dataset: str   = DEFAULT_DATASET) -> pd.Series:
        """
        Hämtar timvis kapacitetsfaktor för vind vid en koordinat under ett år.

        capacity=1 ⇒ 'electricity' returneras per installerad enhet, dvs CF.
        Returnerar pd.Series (UTC-index, namn 'cf', 0–1).
        """
        url    = f"{self.BASE_URL}/data/wind"
        params = {
            "lat":       lat,
            "lon":       lon,
            "date_from": f"{year}-01-01",
            "date_to":   f"{year}-12-31",
            "dataset":   dataset,
            "capacity":  1.0,        # CF = electricity när capacity = 1
            "height":    height,
            "turbine":   turbine,
            "format":    "json",
            "raw":       "false",
        }

        r = self.sess.get(url, params=params, timeout=self.timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text[:500]}",
                response=r,
            ) from None

        payload = r.json()
        if "data" not in payload:
            raise ValueError(f"Oväntat ninja-svar: {list(payload.keys())}")

        # data = {epoch_ms: {"electricity": x, ...}} — ninja nycklar = UTC epoch-ms.
        df  = pd.DataFrame(payload["data"]).T
        idx = pd.to_datetime(pd.to_numeric(df.index), unit="ms", utc=True).as_unit("ns")
        cf  = pd.Series(df["electricity"].astype(float).values, index=idx,
                        name="cf").clip(0, 1)
        cf.index.name = "timestampUTC"

        # Resampla till timvis om finare upplösning skulle returneras.
        if len(cf) > 1:
            dt = (cf.index[1] - cf.index[0]).total_seconds()
            if dt < 3600:
                cf = cf.resample("h").mean()

        return cf
