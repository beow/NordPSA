"""
Värmelastprofiler per NordPSA-zon (When2Heat-metodik, Ruhnau & Muessel 2019).

Rekonstruktion av When2Heat (github.com/oruhnau/when2heat) anpassad för NordPSA:
istället för ett ERA5-grid viktat med befolkning använder vi befolkningsvägd
timtemperatur per zon (från utvalda orter, se config/heat_cities.yaml,
hämtade via scripts/fetch_openmeteo.py).

Algoritm (per zon):
  1. Befolkningsvägd timtemperatur (orter viktade med befolkning).
  2. Referenstemperatur = 4-dygns viktat medel (0.5^i), fångar termisk tröghet.
  3. Landjustering: T_ref − (tröskel_land − tröskel_DE)  (heating_thresholds.csv).
  4. Daglig värme (BDEW-sigmoid) per byggnadstyp SFH/MFH/COM:
       sigmoid = A/(1+(B/(T−40))^C) + D ;  linjär = max(m_s·T+b_s, m_w·T+b_w)
       daily_heat = sigmoid + linjär ;  daily_water = m_w·clip(T,15)+b_w + D
  5. Timfördelning via BGW-timfaktorer (temperaturklass; för COM även veckodag).
       hourly_heat = total (space+water) ;  hourly_water = DHW (klass 30, flack)
       hourly_space = (heat − water).clip(0)
  6. Normalisera + skala per zon/år mot årsvolym (config/heat_annual_twh.yaml).

Förenklingar v1: 'normal' windiness (ingen vinddata); fast byggnadsmix
(SFH 0.49 / MFH 0.21 / COM 0.30); befolkningsvägd temp (ej heat) per zon.

Parameterfiler (pinnad kopia, BGW/BDEW via When2Heat, CC-BY): data/when2heat/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT       = Path(__file__).resolve().parents[1]
W2H_DIR    = ROOT / "data" / "when2heat"
RAW_DIR    = ROOT / "data" / "raw"
CITIES_CFG = ROOT / "config" / "heat_cities.yaml"

ZONE_COUNTRY = {"SE-N": "SE", "SE-S": "SE", "NO-N": "NO", "NO-S": "NO",
                "DK": "DK", "FI": "FI"}
BUILDINGS    = ["SFH", "MFH", "COM"]
BUILDING_W   = {"SFH": 0.49, "MFH": 0.21, "COM": 0.30}   # 0.7 bostad (70/30) + 0.3 tertiär


# --------------------------------------------------------------------------- I/O
def _slug(name: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return s.lower().replace(" ", "_")


def zone_temperature(cities_cfg_path: Path = CITIES_CFG, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Befolkningsvägd timtemperatur (°C) per zon, från cachade ort-parquets."""
    cities = yaml.safe_load(open(cities_cfg_path))["heat_cities"]
    out = {}
    for zone, orter in cities.items():
        num = den = None
        for c in orter:
            s = pd.read_parquet(raw_dir / f"openmeteo_t2m_{_slug(c['namn'])}.parquet")["temp"]
            w = float(c["pop"])
            num = s * w if num is None else num + s * w
            den = w if den is None else den + w
        out[zone] = num / den
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_daily_params() -> dict:
    """daily_demand.csv → {(building, windiness): {A,B,C,D,m_s,b_s,m_w,b_w}}."""
    raw = pd.read_csv(W2H_DIR / "daily_demand.csv", sep=";", decimal=",", header=None)
    bts  = raw.iloc[0, 1:].tolist()
    winds = raw.iloc[1, 1:].tolist()
    names = raw.iloc[2:, 0].tolist()
    params = {}
    for j, (bt, wnd) in enumerate(zip(bts, winds), start=1):
        col = raw.iloc[2:, j].astype(str).str.replace(",", ".", regex=False)
        vals = pd.to_numeric(col, errors="coerce").tolist()
        params[(bt, wnd)] = dict(zip(names, vals))
    return params


def _load_hourly_factors() -> dict:
    """hourly_factors_*.csv → {building: DataFrame}. SFH/MFH index=time;
    COM index=(weekday, time). Kolumner = temperaturklasser (str)."""
    hf = {}
    for b in ["SFH", "MFH"]:
        d = pd.read_csv(W2H_DIR / f"hourly_factors_{b}.csv", sep=";", decimal=",", index_col=0)
        d.columns = [str(int(c)) for c in d.columns]
        hf[b] = d
    com = pd.read_csv(W2H_DIR / "hourly_factors_COM.csv", sep=";", decimal=",")
    com = com.set_index(["weekday", "time"])
    com.columns = [str(int(c)) for c in com.columns]
    hf["COM"] = com
    return hf


def _load_thresholds() -> pd.Series:
    d = pd.read_csv(W2H_DIR / "heating_thresholds.csv", sep=";", decimal=",", index_col=0)
    return d.iloc[:, 0]


# ------------------------------------------------------------------ When2Heat
def reference_temperature(temp: pd.DataFrame) -> pd.DataFrame:
    """4-dygns viktat medel av dygnsmedeltemperaturen (termisk tröghet)."""
    daily = temp.resample("D").mean()
    num = sum(0.5 ** i * daily.shift(i).bfill() for i in range(4))
    return num / sum(0.5 ** i for i in range(4))


def _daily_components(adj_ref: pd.Series, params: dict) -> tuple[dict, dict]:
    """Daglig heat (total) och water per byggnadstyp för en zon (adj_ref = °C, daglig)."""
    heat_b, water_b = {}, {}
    for b in BUILDINGS:
        p = params[(b, "normal")]
        sig = p["A"] / (1 + (p["B"] / (adj_ref - 40)) ** p["C"]) + p["D"]
        lin = np.maximum(p["m_s"] * adj_ref + p["b_s"], p["m_w"] * adj_ref + p["b_w"])
        heat_b[b]  = sig + lin
        water_b[b] = p["m_w"] * adj_ref.clip(lower=15) + p["b_w"] + p["D"]
    return heat_b, water_b


def _hourly_factor_series(hf_b: pd.DataFrame, building: str,
                          index: pd.DatetimeIndex, klass: pd.Series) -> np.ndarray:
    """Slå upp timfaktor per tidssteg från BGW-tabell (klass = temperaturklass-str/timme)."""
    times = index.strftime("%H:%M")
    if building == "COM":
        wdays = index.strftime("%w").astype(int)
        stacked = hf_b.stack()                      # (weekday, time, class) → faktor
        keys = list(zip(wdays, times, klass.values))
    else:
        stacked = hf_b.stack()                      # (time, class) → faktor
        keys = list(zip(times, klass.values))
    return stacked.reindex(keys).values


def heat_profiles_raw(temp: pd.DataFrame) -> dict:
    """Returnerar {zon: DataFrame[total, space, water]} (normaliserade MW/godtycklig skala)."""
    params  = _load_daily_params()
    hf      = _load_hourly_factors()
    thr     = _load_thresholds()
    ref_d   = reference_temperature(temp)           # daglig per zon

    out = {}
    for zone in temp.columns:
        country = ZONE_COUNTRY[zone]
        shift   = thr[country] - thr["DE"]
        adj_d   = ref_d[zone] - shift               # landjusterad referens (daglig)
        heat_b, water_b = _daily_components(adj_d, params)

        # temperaturklass per dygn (av OJUSTERAD referens), upsamplad till timme
        klass_d = (np.ceil(ref_d[zone] / 5) * 5).clip(-15, 30).astype(int).astype(str)
        hidx    = temp.index
        klass_h = klass_d.reindex(hidx, method="ffill")
        # daglig → timme (ffill inom dygn)
        def up(s):
            return s.reindex(hidx, method="ffill")

        total_h = pd.Series(0.0, index=hidx)
        water_h = pd.Series(0.0, index=hidx)
        for b in BUILDINGS:
            w = BUILDING_W[b]
            f_heat  = _hourly_factor_series(hf[b], b, hidx, klass_h)
            f_water = _hourly_factor_series(hf[b], b, hidx, pd.Series("30", index=hidx))
            total_h += w * up(heat_b[b]).values  * f_heat
            water_h += w * up(water_b[b]).values * f_water
        space_h = (total_h - water_h).clip(lower=0)
        out[zone] = pd.DataFrame({"total": total_h, "space": space_h, "water": water_h})
    return out


def build_heat_load(annual_twh: dict, cities_cfg_path: Path = CITIES_CFG) -> pd.DataFrame:
    """Kalibrerad värmelast (MWh/h) per zon. annual_twh = {land: TWh/år (total värme)}.
    Landsvolym splittas till zoner via befolkningsandel; skalas per zon och år."""
    temp = zone_temperature(cities_cfg_path)
    raw  = heat_profiles_raw(temp)

    # befolkningsandel per zon inom sitt land
    cities = yaml.safe_load(open(cities_cfg_path))["heat_cities"]
    zone_pop = {z: sum(c["pop"] for c in orter) for z, orter in cities.items()}
    country_pop = {}
    for z, p in zone_pop.items():
        country_pop[ZONE_COUNTRY[z]] = country_pop.get(ZONE_COUNTRY[z], 0) + p

    out = {}
    for zone, df in raw.items():
        country = ZONE_COUNTRY[zone]
        target_zone_twh = annual_twh[country] * zone_pop[zone] / country_pop[country]
        prof = df["total"].copy()
        # skala per kalenderår så årssumman (MWh) = mål-TWh × 1e6
        scaled = prof.copy()
        for yr, grp in prof.groupby(prof.index.year):
            target_mwh = target_zone_twh * 1e6
            scaled.loc[grp.index] = grp * (target_mwh / grp.sum())
        out[zone] = scaled
    res = pd.DataFrame(out)
    res.index.name = "time"
    return res
