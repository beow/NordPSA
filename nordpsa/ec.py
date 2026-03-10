"""
Energy Charts API-klient — används enbart för Danmark (DK).

eSett returnerar all dansk produktion i kolumnen 'other' utan uppdelning,
så vi hämtar vind och sol från Energy Charts public_power-endpoint istället.

Baserad på ECapi.py av Bengt Söderström.
Ändringar:
  - Inget beroende av 'helpers'-modulen; kolumnnamn mappas lokalt
  - Tidsstämplar hanteras som UTC (ingen +1h-offset) för konsistens med eSett
  - Skräddarsydd för NordPSA: returnerar samma kolumnnamn som eSett-datan
"""
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Mappning från Energy Charts kolumnnamn till NordPSA/eSett-kolumnnamn
_EC_COL_MAP = {
    "Wind onshore":                 "wind",
    "Wind offshore":                "windOffshore",
    "Photovoltaics":                "solar",
    "Run-of-river":                 "hydro",
    "Biomass":                      "biomass",
    "Nuclear":                      "nuclear",
    "Fossil hard coal":             "coal",
    "Fossil brown coal / Lignite":  "lignite",
    "Fossil gas":                   "gas",
    "Hydro pumped storage":         "hydro_phs",
    "Battery storage":              "energyStorage",
    "Others":                       "other",
    "Geothermal":                   "geothermal",
    "Waste":                        "waste",
    "Load":                         "load",
    "Residual load":                "residual_load",
    "Renewable share of generation": "renewable_share",
    "Renewable share of load":      "renewable_share_load",
}

EC_COUNTRY_DK = "dk"


class EnergyChartsClient:
    def __init__(self,
                 base_url: str = "https://api.energy-charts.info",
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.sess     = requests.Session()

    def fetch_public_power(self,
                           country: str,
                           start_date: str,
                           end_date: str) -> Optional[pd.DataFrame]:
        """
        Hämtar offentlig kraftproduktion per produktionstyp.

        Parametrar
        ----------
        country    : Energy Charts landskod, t.ex. 'dk'
        start_date : 'YYYY-MM-DD'
        end_date   : 'YYYY-MM-DD'  (inklusiv)

        Returnerar DataFrame med UTC-index och kolumner mappade till NordPSA-namn.
        """
        url    = f"{self.base_url}/public_power"
        params = {
            "country": country,
            "start":   start_date,
            "end":     end_date,
            "format":  "json",
        }

        r = self.sess.get(url, params=params, timeout=self.timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text}",
                response=r,
            ) from None

        data = r.json()

        if "unix_seconds" not in data or "production_types" not in data:
            raise ValueError(f"Oväntat API-svar: {list(data.keys())}")

        # Tidsstämplar — unix_seconds är UTC epoch
        timestamps = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)

        cols = {}
        for pt in data["production_types"]:
            raw_name = pt["name"]
            col_name = _EC_COL_MAP.get(raw_name, raw_name.lower().replace(" ", "_"))
            cols[col_name] = pt["data"]

        df = pd.DataFrame(cols, index=timestamps)
        df.index.name = "timestampUTC"

        # Resampla till timvis om kortare intervall (t.ex. 15 min)
        if len(df) > 1:
            dt = (df.index[1] - df.index[0]).total_seconds()
            if dt < 3600:
                df = df.resample("h").mean()

        return df

    def fetch_year(self, country: str, year: int) -> Optional[pd.DataFrame]:
        """Hämtar ett helt kalenderår."""
        return self.fetch_public_power(
            country,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
        )

    def fetch_price(self, bzn: str, start_date: str, end_date: str) -> pd.Series:
        """
        Hämtar day-ahead-spotpris för ett budzoneomr.

        Parametrar
        ----------
        bzn        : budzondkod, t.ex. 'DE-LU' (Tyskland) eller 'FR' (Frankrike)
        start_date : 'YYYY-MM-DD'
        end_date   : 'YYYY-MM-DD'  (inklusiv)

        Returnerar pd.Series med UTC-index och pris i EUR/MWh, timvis.
        """
        url    = f"{self.base_url}/price"
        params = {"bzn": bzn, "start": start_date, "end": end_date}

        r = self.sess.get(url, params=params, timeout=self.timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text}",
                response=r,
            ) from None

        data = r.json()
        timestamps = pd.to_datetime(data["unix_seconds"], unit="s", utc=True)
        prices = pd.Series(data["price"], index=timestamps, name="price_eur_mwh",
                           dtype=float)

        # Resampla till timvis om kortare intervall
        if len(prices) > 1:
            dt = (prices.index[1] - prices.index[0]).total_seconds()
            if dt < 3600:
                prices = prices.resample("h").mean()

        return prices

    def fetch_price_year(self, bzn: str, year: int) -> pd.Series:
        """Hämtar day-ahead-pris för ett helt kalenderår."""
        return self.fetch_price(bzn, f"{year}-01-01", f"{year}-12-31")
