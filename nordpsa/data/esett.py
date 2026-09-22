"""
eSett open data client — utökad för NordPSA-zoner.

Baserad på eSettAPI.py av Bengt Söderström.
Ändringar:
  - NORDPSA_ZONES tillagda i COUNTRY_TO_MBAS
  - _fmt_utc fixad: lägger till millisekunder som eSett kräver
  - fetch_year(): bekvämlighetsmetod för att hämta ett helt år
"""
import datetime as _dt
from typing import Any, Dict, List, Optional

import pandas as pd
import pytz
import requests

# --- Mappningar ---

MBA_TO_EIC: Dict[str, str] = {
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "NO1": "10YNO_1________2",
    "NO2": "10YNO_2________T",
    "NO3": "10YNO_3________J",
    "NO4": "10YNO_4________9",
    "NO5": "10Y1001A1001A48H",
    "DK1": "10YDK-1--------W",
    "DK2": "10YDK-2--------M",
    "FI":  "10YFI_1________U",
}

COUNTRY_TO_MBAS: Dict[str, List[str]] = {
    # Länder
    "SE": ["SE1", "SE2", "SE3", "SE4"],
    "NO": ["NO1", "NO2", "NO3", "NO4", "NO5"],
    "DK": ["DK1", "DK2"],
    "FI": ["FI"],
    # NordPSA-zoner
    "SE-N": ["SE1", "SE2"],
    "SE-S": ["SE3", "SE4"],
    "NO-N": ["NO3", "NO4"],
    "NO-S": ["NO1", "NO2", "NO5"],
    # DK och FI matchar redan länder ovan
}

NORDPSA_ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]

CET_TZ = pytz.timezone("Europe/Stockholm")


def _parse_iso8601_z(s: str) -> _dt.datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return _dt.datetime.fromisoformat(s)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_local_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    ts_utc = pd.to_datetime(df["timestampUTC"], utc=True)
    local = ts_utc.dt.tz_convert(CET_TZ)
    df = df.copy()
    df["timestamp"] = local.dt.tz_localize(None)
    return df


def _maybe_local_index(df: pd.DataFrame,
                       local_index: bool,
                       keep_time_columns: bool,
                       index_tz: Optional[str]) -> pd.DataFrame:
    if not local_index or df.empty:
        return df
    tz = pytz.timezone(index_tz) if index_tz else CET_TZ
    ts_local = pd.to_datetime(df["timestampUTC"], utc=True).dt.tz_convert(tz)
    out = df.copy()
    if not keep_time_columns:
        out = out.drop(columns=[c for c in ["timestamp", "timestampUTC"] if c in out.columns])
    out = out.set_index(ts_local)
    out.index.name = "time"
    return out


class eSettClient:
    def __init__(self, base_url: str = "https://api.opendata.esett.com", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sess = requests.Session()

    # --- Publikt API ---

    def fetch(self,
              series_type: str,
              area_code_or_eic: str,
              start_utc: str | _dt.datetime,
              end_utc: str | _dt.datetime,
              resolution: Optional[str] = None,
              extra_params: Optional[Dict[str, Any]] = None,
              local_index: bool = False,
              index_tz: Optional[str] = None,
              keep_time_columns: bool = False) -> pd.DataFrame:
        st = series_type.lower()
        if st not in ("consumption", "production"):
            raise ValueError("series_type must be 'Consumption' or 'Production'")
        exp = "EXP15" if st == "consumption" else "EXP16"
        df = self._fetch_aggregate_with_country_support(
            exp, series_type.capitalize(), area_code_or_eic,
            start_utc, end_utc, resolution, extra_params
        )
        return _maybe_local_index(df, local_index, keep_time_columns, index_tz)

    def fetch_year(self,
                   series_type: str,
                   zone: str,
                   year: int,
                   resolution: str = "hour") -> pd.DataFrame:
        """Hämtar ett helt kalenderår (UTC) för en NordPSA-zon."""
        start = f"{year}-01-01T00:00:00.000Z"
        end   = f"{year + 1}-01-01T00:00:00.000Z"
        return self.fetch(series_type, zone, start, end, resolution=resolution)

    # --- Inre helpers ---

    def _fetch_aggregate_with_country_support(self,
                                              exp: str,
                                              series: str,
                                              area_code_or_eic: str,
                                              start_utc: str | _dt.datetime,
                                              end_utc: str | _dt.datetime,
                                              resolution: Optional[str],
                                              extra_params: Optional[Dict[str, Any]]) -> pd.DataFrame:
        token = area_code_or_eic.strip().upper()
        if token in COUNTRY_TO_MBAS:
            frames: List[pd.DataFrame] = []
            for mba in COUNTRY_TO_MBAS[token]:
                df_zone = self._fetch_aggregate(
                    exp=exp, series=series, mba=MBA_TO_EIC.get(mba, mba),
                    start_utc=start_utc, end_utc=end_utc,
                    resolution=resolution, extra_params=extra_params
                )
                if not df_zone.empty:
                    df_zone = df_zone.copy()
                    df_zone["mba"] = mba
                    frames.append(df_zone)
            return self._aggregate_country_frames(token, frames)

        mba_param = MBA_TO_EIC.get(token, token)
        return self._fetch_aggregate(exp, series, mba_param, start_utc, end_utc, resolution, extra_params)

    def _aggregate_country_frames(self, zone_code: str, frames: List[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=["timestamp", "timestampUTC", "mba"])
        df_all = pd.concat(frames, ignore_index=True)
        value_cols = [c for c in df_all.columns if c not in ("timestamp", "timestampUTC", "mba")]
        df_all = _coerce_numeric(df_all, value_cols)
        if not value_cols:
            return pd.DataFrame(columns=["timestamp", "timestampUTC", "mba"])
        grouped = (
            df_all
            .groupby(["timestampUTC"], dropna=False)[value_cols]
            .sum(min_count=1)
            .reset_index()
        )
        grouped = _add_local_time_cols(grouped)
        grouped["mba"] = zone_code
        ordered = ["timestamp", "timestampUTC", "mba"] + value_cols
        return grouped[ordered]

    def _fetch_aggregate(self,
                         exp: str,
                         series: str,
                         mba: str,
                         start_utc: str | _dt.datetime,
                         end_utc: str | _dt.datetime,
                         resolution: Optional[str],
                         extra_params: Optional[Dict[str, Any]]) -> pd.DataFrame:
        res = (resolution or "").strip().lower()
        aggregate_resolutions = {"year", "month", "week", "day", "hour"}

        if res in aggregate_resolutions:
            url = f"{self.base_url}/{exp}/Aggregate"
            params: Dict[str, Any] = {
                "series": series,
                "mba": mba,
                "start": self._fmt_utc(start_utc),
                "end": self._fmt_utc(end_utc),
                "resolution": res,
            }
        else:
            url = f"{self.base_url}/{exp}/Consumption" if exp == "EXP15" else f"{self.base_url}/{exp}/Volumes"
            params = {
                "mba": [mba] if isinstance(mba, str) else mba,
                "start": self._fmt_utc(start_utc),
                "end": self._fmt_utc(end_utc),
            }

        if extra_params:
            params.update(extra_params)

        r = self.sess.get(url, params=params, timeout=self.timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError:
            msg = f"{r.status_code} {r.reason} for URL: {r.url}\nBody: {r.text}"
            raise requests.HTTPError(msg, response=r) from None

        text = (r.text or "").strip()
        if not text:
            return pd.DataFrame(columns=["timestamp", "timestampUTC", "mba"])

        payload = r.json()
        rows = self._rows_from_payload(payload)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "timestampUTC", "mba"])

        df = pd.DataFrame(rows)
        if "timestampUTC" in df.columns:
            df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
            df = _add_local_time_cols(df)

        if "mba" not in df.columns or df["mba"].isna().all():
            short = next((k for k, v in MBA_TO_EIC.items() if v == mba), None)
            df["mba"] = short or mba

        return df

    def _rows_from_payload(self, payload: Any) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("items", []) or payload.get("Items", [])
        else:
            return []

        rows: List[Dict[str, Any]] = []
        for it in items:
            ts  = it.get("timestampUTC") or it.get("TimestampUTC")
            mba = it.get("mba") or it.get("MBA") or it.get("biddingZone") or it.get("BiddingZone")
            if not ts:
                continue
            row: Dict[str, Any] = {"timestampUTC": ts, "mba": mba}
            for k, v in it.items():
                kl = str(k).lower()
                if kl in ("timestamp", "timestamputc", "mba", "biddingzone", "area", "series"):
                    continue
                if isinstance(v, (int, float)) or (
                    isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit()
                ):
                    try:
                        row[k] = float(v)
                    except Exception:
                        row[k] = pd.to_numeric(v, errors="coerce")
                else:
                    row[k] = v
            rows.append(row)
        return rows

    def _fmt_utc(self, t: str | _dt.datetime) -> str:
        if isinstance(t, str):
            # Lägg till millisekunder om de saknas (eSett kräver .SSSZ-format)
            if t.endswith("Z") and "." not in t:
                t = t[:-1] + ".000Z"
            return t
        if t.tzinfo is None:
            t = pytz.utc.localize(t)
        return t.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
