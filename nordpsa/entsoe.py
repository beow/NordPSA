"""
ENTSO-E Transparency Platform klient — day-ahead priser + cross-border flöden.
Elexon BMRS klient — GB day-ahead priser (Market Index Data).

GB saknas på ENTSO-E post-Brexit; används ElexonClient för GB-priser istället.

Användning:
    from nordpsa.entsoe import ENTSOEClient, ElexonClient
    cli = ENTSOEClient()   # läser ENTSOE_token eller ENTSOE_API_TOKEN från env
    s = cli.fetch_price_year('GB', 2024)
    flows = cli.fetch_cross_border_flows('SE2', 'SE3',
                pd.Timestamp('2023-01-01', tz='UTC'),
                pd.Timestamp('2023-12-31 23:00', tz='UTC'))
"""
import os
import time
import xml.etree.ElementTree as ET
from typing import Optional

import pandas as pd
import requests

ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"

# EIC-koder för relevanta budzoner
# Nordiska MBA:er finns i nordpsa/esett.py (MBA_TO_EIC)
EIC_CODES: dict[str, str] = {
    # Svenska budzoner
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    # Norska budzoner
    "NO1": "10YNO-1--------2",
    "NO2": "10YNO-2--------T",
    "NO3": "10YNO-3--------J",
    "NO4": "10YNO-4--------9",
    "NO5": "10Y1001A1001A48H",
    # Övriga nordiska
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "FI":  "10YFI-1--------U",
    # Kontinentala grannar
    "DE-LU": "10Y1001A1001A83F",
    "NL":    "10YNL----------L",
    "GB":    "10YGB----------A",
    "LT":    "10YLT-1001A0008Q",
    "PL":    "10YPL-AREA-----S",
    "EE":    "10Y1001A1001A39I",
}

# Valutaomräkning till EUR (årsgenomsnitt 2023-2025)
# GBP/EUR: ca 1.17 (1 GBP ≈ 1.17 EUR)
CURRENCY_TO_EUR: dict[str, float] = {
    "EUR": 1.0,
    "GBP": 1.17,
}


class ENTSOEClient:
    def __init__(self, api_token: Optional[str] = None, base_url: str = ENTSOE_BASE_URL):
        # Stöder både ENTSOE_token (befintlig) och ENTSOE_API_TOKEN
        self.api_token = (
            api_token
            or os.environ.get("ENTSOE_API_TOKEN", "")
            or os.environ.get("ENTSOE_token", "")
        )
        if not self.api_token:
            raise ValueError(
                "ENTSO-E API-token saknas. Sätt ENTSOE_token eller ENTSOE_API_TOKEN "
                "i miljön, eller skicka api_token= till konstruktorn."
            )
        self.base_url = base_url
        self.sess = requests.Session()

    def fetch_day_ahead_price(
        self,
        bzn: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series:
        """
        Hämtar day-ahead priser för en budzon.

        Parametrar
        ----------
        bzn   : budzonskod, t.ex. 'GB', eller direkt EIC-kod
        start : startdatum (UTC)
        end   : slutdatum (UTC, inklusivt)

        Returnerar pd.Series med timvis index (UTC) och EUR/MWh som värden.
        Max 1 år per anrop (ENTSO-E-begränsning).
        """
        eic = EIC_CODES.get(bzn, bzn)
        params = {
            "securityToken": self.api_token,
            "documentType":  "A44",
            "in_Domain":     eic,
            "out_Domain":    eic,
            "periodStart":   start.strftime("%Y%m%d%H%M"),
            "periodEnd":     (end + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M"),
        }
        r = self.sess.get(self.base_url, params=params, timeout=60)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text[:500]}",
                response=r,
            ) from None

        return self._parse_xml(r.text)

    def fetch_price_year(self, bzn: str, year: int) -> pd.Series:
        """Hämtar day-ahead priser för ett helt kalenderår."""
        start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
        end   = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")
        return self.fetch_day_ahead_price(bzn, start, end)

    def fetch_cross_border_flows(
        self,
        out_domain: str,
        in_domain: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        sleep_s: float = 0.5,
    ) -> pd.Series:
        """
        Hämtar timvisa fysiska cross-border flöden (A11) från out_domain → in_domain.
        Positiva värden = flöde i riktningen out → in.

        Delar automatiskt upp i årsblock om perioden är längre än 1 år.
        """
        out_eic = EIC_CODES.get(out_domain, out_domain)
        in_eic  = EIC_CODES.get(in_domain,  in_domain)

        # Dela upp i årsblock (ENTSO-E tillåter max 1 år per anrop)
        chunks = []
        t = start
        while t <= end:
            t_end = min(pd.Timestamp(f"{t.year}-12-31 23:00", tz="UTC"), end)
            params = {
                "securityToken": self.api_token,
                "documentType":  "A11",
                "in_Domain":     in_eic,
                "out_Domain":    out_eic,
                "periodStart":   t.strftime("%Y%m%d%H%M"),
                "periodEnd":     (t_end + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M"),
            }
            r = self.sess.get(self.base_url, params=params, timeout=60)
            try:
                r.raise_for_status()
            except requests.HTTPError:
                raise requests.HTTPError(
                    f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text[:500]}",
                    response=r,
                ) from None
            chunk = self._parse_xml_quantity(r.text)
            if not chunk.empty:
                chunks.append(chunk)
            t = pd.Timestamp(f"{t.year + 1}-01-01 00:00", tz="UTC")
            if t <= end and sleep_s > 0:
                time.sleep(sleep_s)

        if not chunks:
            return pd.Series(dtype=float, name=f"{out_domain}→{in_domain}")
        s = pd.concat(chunks).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        s.name = f"{out_domain}→{in_domain}"
        return s

    def fetch_reservoir_year(self, bzn: str, year: int) -> pd.Series:
        """
        Hämtar veckovisa reservoarnivåer (MWh) för en budzon, A72/A16.
        Returnerar pd.Series med ISO-veckostart (måndag UTC) som index.

        ENTSO-E A72 använder P7D-upplösning med start söndag 23:00 UTC
        (= CET måndag 00:00). Tidsstämplarna normaliseras till måndag 00:00 UTC.
        """
        eic = EIC_CODES.get(bzn, bzn)
        params = {
            "securityToken": self.api_token,
            "documentType":  "A72",
            "processType":   "A16",
            "in_Domain":     eic,
            "periodStart":   f"{year}01010000",
            "periodEnd":     f"{year}12312300",
        }
        r = self.sess.get(self.base_url, params=params, timeout=60)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason}\nURL: {r.url}\nBody: {r.text[:300]}",
                response=r,
            ) from None
        return self._parse_xml_reservoir(r.text, year)

    def _parse_xml_reservoir(self, xml_text: str, year: int) -> pd.Series:
        """Parsar ENTSO-E A72 (P7D) och returnerar veckovis MWh med måndag UTC-index."""
        root = ET.fromstring(xml_text)
        ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        def t(name: str) -> str:
            return "{%s}%s" % (ns, name) if ns else name

        records: dict[pd.Timestamp, float] = {}
        for ts in root.findall(".//" + t("TimeSeries")):
            for period in ts.findall(t("Period")):
                interval  = period.find(t("timeInterval"))
                start_str = interval.find(t("start")).text          # type: ignore[union-attr]
                start_dt  = pd.Timestamp(start_str)

                for point in period.findall(t("Point")):
                    pos    = int(point.find(t("position")).text)     # type: ignore[union-attr]
                    qty_el = point.find(t("quantity"))
                    if qty_el is None:
                        continue
                    # P7D: position n → start + (n-1) × 7 dagar
                    ts_pt = start_dt + pd.Timedelta(days=7 * (pos - 1))
                    # Normalisera söndag 23:00 UTC → måndag 00:00 UTC
                    ts_pt = (ts_pt + pd.Timedelta(hours=1)).normalize()
                    if ts_pt.tz is None:
                        ts_pt = ts_pt.tz_localize("UTC")
                    records[ts_pt] = float(qty_el.text)

        s = pd.Series(records).sort_index()
        # Filtrera till begärt år (± 1 vecka)
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end   = pd.Timestamp(f"{year}-12-31 23:59", tz="UTC")
        return s.loc[start:end]

    def fetch_hydro_generation(
        self,
        bzn: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        sleep_s: float = 0.5,
    ) -> pd.Series:
        """
        Hämtar faktisk vattenkraftgenerering (A75/A16) för en budzon, MW timvis.

        Summerar B11 (run-of-river), B12 (reservoar) och B10 (pumped storage,
        nettogenerering). Delar upp i årsblock vid behov.
        """
        PSR_TYPES = ["B11", "B12", "B10"]
        eic = EIC_CODES.get(bzn, bzn)
        chunks_all: list[pd.Series] = []

        t = start
        while t <= end:
            t_end = min(pd.Timestamp(f"{t.year}-12-31 23:00", tz="UTC"), end)
            for psr in PSR_TYPES:
                params = {
                    "securityToken": self.api_token,
                    "documentType":  "A75",
                    "processType":   "A16",
                    "in_Domain":     eic,
                    "psrType":       psr,
                    "periodStart":   t.strftime("%Y%m%d%H%M"),
                    "periodEnd":     (t_end + pd.Timedelta(hours=1)).strftime("%Y%m%d%H%M"),
                }
                r = self.sess.get(self.base_url, params=params, timeout=60)
                try:
                    r.raise_for_status()
                except requests.HTTPError:
                    continue   # psrType kanske inte finns för denna zon
                chunk = self._parse_xml_quantity(r.text)
                if not chunk.empty:
                    chunks_all.append(chunk)
                time.sleep(sleep_s)
            t = pd.Timestamp(f"{t.year + 1}-01-01 00:00", tz="UTC")

        if not chunks_all:
            return pd.Series(dtype=float, name=f"hydro_{bzn}")

        # Summera alla psrType per tidsstämpel; klämma till begärd period
        combined = pd.concat(chunks_all)
        s = combined.groupby(combined.index).sum()
        s = s.sort_index().loc[start:end]
        s.name = f"hydro_{bzn}"
        return s

    def fetch_hydro_year(self, bzn: str, year: int) -> pd.Series:
        """Hämtar faktisk vattenkraftgenerering för ett helt kalenderår."""
        start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
        end   = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")
        return self.fetch_hydro_generation(bzn, start, end)

    def _parse_xml_quantity(self, xml_text: str) -> pd.Series:
        """Parsar ENTSO-E XML och returnerar timvis MW-flöde (quantity).

        Hanterar blandade upplösningar (PT15M, PT30M, PT60M) inom ett svar
        genom att behandla varje Period separat och resampla till timvis.
        """
        root = ET.fromstring(xml_text)
        ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        def t(name: str) -> str:
            return "{%s}%s" % (ns, name) if ns else name

        period_series: list[pd.Series] = []
        for ts in root.findall(".//" + t("TimeSeries")):
            for period in ts.findall(t("Period")):
                interval   = period.find(t("timeInterval"))
                start_str  = interval.find(t("start")).text       # type: ignore[union-attr]
                resolution = period.find(t("resolution")).text     # type: ignore[union-attr]

                if "PT15M" in resolution:
                    freq_min = 15
                elif "PT30M" in resolution:
                    freq_min = 30
                else:
                    freq_min = 60

                start_dt = pd.Timestamp(start_str)
                records: dict[pd.Timestamp, float] = {}
                for point in period.findall(t("Point")):
                    pos    = int(point.find(t("position")).text)   # type: ignore[union-attr]
                    qty_el = point.find(t("quantity"))
                    if qty_el is None:
                        continue
                    ts_pt  = start_dt + pd.Timedelta(minutes=freq_min * (pos - 1))
                    records[ts_pt] = float(qty_el.text)

                if not records:
                    continue
                s = pd.Series(records).sort_index()
                s.index = pd.to_datetime(s.index, utc=True)
                if freq_min < 60:
                    s = s.resample("h").mean()
                period_series.append(s)

        if not period_series:
            return pd.Series(dtype=float)

        combined = pd.concat(period_series)
        # Deduplicera per tidsstämpel — ta medel vid överlapp (MW-data)
        return combined.groupby(combined.index).mean().sort_index()

    def _parse_xml(self, xml_text: str) -> pd.Series:
        root = ET.fromstring(xml_text)

        # Extrahera namespace-prefix från rot-taggen
        ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
        def t(name: str) -> str:
            return "{%s}%s" % (ns, name) if ns else name

        # Detektera valuta
        currency = "EUR"
        cur_el = root.find(".//" + t("currency_Unit.name"))
        if cur_el is not None:
            currency = cur_el.text.strip()
        multiplier = CURRENCY_TO_EUR.get(currency, 1.0)

        records: dict[pd.Timestamp, float] = {}
        for ts in root.findall(".//" + t("TimeSeries")):
            for period in ts.findall(t("Period")):
                interval = period.find(t("timeInterval"))
                start_str = interval.find(t("start")).text       # type: ignore[union-attr]
                resolution = period.find(t("resolution")).text   # type: ignore[union-attr]
                freq_min = 30 if "PT30M" in resolution else 60

                start_dt = pd.Timestamp(start_str)
                for point in period.findall(t("Point")):
                    pos   = int(point.find(t("position")).text)   # type: ignore[union-attr]
                    price = float(point.find(t("price.amount")).text)  # type: ignore[union-attr]
                    ts_pt = start_dt + pd.Timedelta(minutes=freq_min * (pos - 1))
                    records[ts_pt] = price * multiplier

        if not records:
            raise ValueError("Inga prisdata hittades i ENTSO-E-svaret.")

        s = pd.Series(records).sort_index()
        s.index = pd.to_datetime(s.index, utc=True)
        s.name = "price_eur_mwh"

        # Resampla till timvis om 30-minutersdata
        if len(s) > 1:
            dt_min = (s.index[1] - s.index[0]).total_seconds() / 60
            if dt_min < 60:
                s = s.resample("h").mean()

        return s


# ---------------------------------------------------------------------------
# Elexon BMRS — GB day-ahead priser (Market Index Data)
# ---------------------------------------------------------------------------

ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
GBP_TO_EUR = 1.17   # årsgenomsnitt 2023-2025


class ElexonClient:
    """
    Hämtar GB day-ahead priser från Elexon BMRS (Market Index Data).
    Ingen API-token krävs.

    Priser returneras i EUR/MWh (konverterat från GBP via GBP_TO_EUR).
    Källdata är halvtimmes MID-priser; resamplas till timvis.
    """

    def __init__(self, base_url: str = ELEXON_BASE_URL, gbp_to_eur: float = GBP_TO_EUR):
        self.base_url   = base_url
        self.gbp_to_eur = gbp_to_eur
        self.sess       = requests.Session()

    def fetch_price(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """
        Hämtar MID-priser (GBP/MWh) och returnerar timvis EUR/MWh.
        Hämtar i 30-dagarsbitar för att hålla sig inom API-gränser.
        """
        chunks = []
        t = start.normalize()  # börja vid midnatt
        while t <= end:
            t_end = min(t + pd.Timedelta(days=7), end)
            chunk = self._fetch_chunk(t, t_end)
            if not chunk.empty:
                chunks.append(chunk)
            t = t_end + pd.Timedelta(hours=1)

        if not chunks:
            raise ValueError("Inga MID-prisdata från Elexon för angiven period.")

        s = pd.concat(chunks).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        return s

    def fetch_price_year(self, year: int) -> pd.Series:
        """Hämtar MID-priser för ett helt kalenderår."""
        start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
        end   = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")
        return self.fetch_price(start, end)

    def _fetch_chunk(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        params = {
            "from": start.strftime("%Y-%m-%d"),
            "to":   end.strftime("%Y-%m-%d"),
        }
        r = self.sess.get(f"{self.base_url}/datasets/MID", params=params, timeout=60)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise requests.HTTPError(
                f"{r.status_code} {r.reason}\nURL: {r.url}", response=r
            ) from None

        data = r.json().get("data", [])
        if not data:
            return pd.Series(dtype=float, name="price_eur_mwh")

        df = pd.DataFrame(data)
        df = df[df["price"] > 0]   # filtrera bort noll-poster (saknade data)
        df["startTime"] = pd.to_datetime(df["startTime"], utc=True)

        # Volymvägt medelpris per tidssteg (medlar APXMIDP + N2EXMIDP)
        grouped = (
            df.groupby("startTime")
            .apply(lambda g: (g["price"] * g["volume"]).sum() / g["volume"].sum()
                   if g["volume"].sum() > 0 else g["price"].mean())
            .rename("price_gbp_mwh")
        )

        # Konvertera GBP → EUR och resampla till timvis
        s = (grouped * self.gbp_to_eur).resample("h").mean()
        s.name = "price_eur_mwh"
        return s.loc[start:end]
