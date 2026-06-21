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
    df = df.set_index("timestampUTC")
    df = df[~df.index.duplicated(keep="first")]
    return df


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
    df = df[~df.index.duplicated(keep="first")]
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

        def _normalise_solar_by_year(series: pd.Series, fleet_factor: float = 1.0) -> tuple[pd.Series, float]:
            """Skalningsmetod för sol: normalisera varje år till 2025 års p99-nivå.

            Solflottan växer år för år. Rå produktion 2023 är lägre än 2025 inte för
            att solen skiner mindre, utan för att färre paneler fanns installerade.
            Lösning: beräkna p99 per kalenderår, skala upp 2023/2024 med faktorn
            p99_2025 / p99_år. På så vis antar modellen att 2025 års flotta gäller
            under hela perioden — konsistent med ett kapacitetsexpansionsperspektiv.
            p_nom sätts till 2025 års p99 / fleet_factor.
            """
            p99_per_year = {}
            for yr in YEARS:
                yr_data = series[series.index.year == yr]
                p99_per_year[yr] = float(np.percentile(yr_data, 99)) if len(yr_data) > 0 else 0.0

            ref_p99 = p99_per_year.get(2025, 0.0)
            if ref_p99 <= 1:
                return series * 0.0, 0.0

            scaled = series.copy()
            for yr, p99_yr in p99_per_year.items():
                if p99_yr > 1:
                    mask = series.index.year == yr
                    scaled.loc[mask] = series.loc[mask] * (ref_p99 / p99_yr)

            p_nom = ref_p99 / fleet_factor
            return (scaled / p_nom).clip(0, 1), p_nom

        onshore_pu,  on_nom  = _normalise(onshore,  fleet_factor=0.70)  # wind_onshore_fleet_factor
        # Offshore: fleet_factor=1.0 → p_nom = p99. Till havs blåser det så jämnt
        # att p99 i princip ligger PÅ nameplate (DK: p99=2613 ≈ verkligt 2650),
        # så ingen nedtryckning behövs (till skillnad från onshore). Ger sann
        # flott-CF som profilmedel (DK 0.41).
        offshore_pu, off_nom = _normalise(offshore, fleet_factor=1.0)   # wind_offshore_fleet_factor
        solar_pu,    sol_nom = _normalise_solar_by_year(solar, fleet_factor=0.85)

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

    _apply_ninja_offshore(result)

    out = _trim(pd.DataFrame(result))
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "vre_profiles.parquet")
    print(f"  → vre_profiles.parquet")

    with open(PROCESSED_DIR / "vre_pnom.yaml", "w") as f:
        yaml.dump(vre_noms, f, default_flow_style=False)
    print(f"  → vre_pnom.yaml")

    return out


# Zoner som får syntetiska (ninja) offshore-profiler. DK behåller sin
# faktiska profil och fungerar som empiriskt kalibreringsankare.
OFFSHORE_NINJA_ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "FI"]

# Faktor som omvandlar ninjas enskild-plats-CF (modern turbin) till realiserad
# flott-CF för en framtida offshore-flotta. DK ger empiriskt realiserad/ninja
# ≈ 0.737 (blandade årgångar); avrundat UPP till 0.80 som modern-uplift (~8%),
# eftersom nybyggen får moderna 15 MW-turbiner som slår DK:s gamla flotta.
OFFSHORE_NINJA_NET_FACTOR = 0.80


def _load_ninja_offshore(zone: str) -> pd.Series | None:
    """Laddar ninjas offshore-CF för alla år, UTC-index. None om filer saknas."""
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"offshore_ninja_{zone}_{year}.parquet"
        if not path.exists():
            return None
        frames.append(pd.read_parquet(path))
    s = pd.concat(frames)
    s.index = pd.to_datetime(s.index, utc=True)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s["cf"]


def _apply_ninja_offshore(result: dict) -> None:
    """Ersätter offshore-profilen i zoner utan faktisk produktion med
    Renewables.ninja-profiler, kalibrerade mot DK:s faktiska CF.

    Ger expanderbar havsbaserad vind i alla zoner. DK behåller sin faktiska
    profil (sann flott-CF ~0.41 med fleet_factor=1.0). p_nom påverkas INTE —
    ninja ger bara formen (p_max_pu); installerad effekt = befintligt
    (p_nom_min, ofta 0) och utbyggnad styrs av p_nom_max i config/zones.yaml.

    Nivå: ninjas enskild-plats-CF skalas med OFFSHORE_NINJA_NET_FACTOR (0.80)
    till realiserad flott-CF. DK fungerar som diagnostiskt ankare (empiriskt
    realiserad/ninja ≈ 0.737; vi kör 0.80 som modern-uplift).
    """
    corr = OFFSHORE_NINJA_NET_FACTOR

    # Diagnostik: jämför vald faktor mot DK:s empiriska realiserad/ninja-ankare.
    dk_ninja = _load_ninja_offshore("DK")
    if dk_ninja is not None:
        dk_actual = result["DK_wind_offshore"]
        idx = dk_actual.index.intersection(dk_ninja.index)
        dk_emp = float(dk_actual.loc[idx].mean() / dk_ninja.loc[idx].mean())
        print(f"  ninja offshore-faktor={corr:.2f} (modern-uplift) — "
              f"DK empiriskt {dk_emp:.3f} "
              f"(flott-CF {dk_actual.loc[idx].mean():.3f} / "
              f"ninja {dk_ninja.loc[idx].mean():.3f})")
    else:
        print(f"  ninja offshore-faktor={corr:.2f} (modern-uplift)")

    applied = 0
    for zone in OFFSHORE_NINJA_ZONES:
        cf = _load_ninja_offshore(zone)
        if cf is None:
            print(f"  OBS: ninja-offshore saknas för {zone} — hoppar över")
            continue
        col = f"{zone}_wind_offshore"
        cal = (cf * corr).clip(0, 1).reindex(result[col].index).fillna(0.0)
        result[col] = cal
        applied += 1
        print(f"  {zone}: offshore-profil ← ninja (medel-CF {cal.mean():.3f})")
    if applied == 0:
        print("  OBS: inga ninja-offshore-filer (kör 'python scripts/fetch_ninja.py')")


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
    Must-run termisk produktion (MW) per zon.

    SE/NO/FI (eSett): thermal + other — eSett har ingen finare uppdelning.
    DK (Energy Charts): coal + gas + biomass + waste + fossil_oil.
      eSett DK samlar ALL produktion i 'other' (ingen teknikuppdelning),
      vilket skulle dubbelräkna vind/sol som redan modelleras separat.
    Sparas som absoluta MW; network.py beräknar p_nom = max per zon.
    """
    print("Bygger termisk profil ...")
    result = {}

    for zone in NORDPSA_ZONES:
        if zone == "DK":
            # Must-run för DK: 100% waste + 50% biomass.
            # Kol, gas och olja exkluderas — fossilt fasas ut och gas
            # modelleras separat som dispatchbar peaklastresurs.
            dk_ec = _load_dk_ec()
            thermal = (
                dk_ec.get("waste",   pd.Series(0.0, index=dk_ec.index)).fillna(0) * 1.0
              + dk_ec.get("biomass", pd.Series(0.0, index=dk_ec.index)).fillna(0) * 0.5
            )
        else:
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

# Budzoner att läsa in — måste matcha PRICE_BZNS_EC + PRICE_BZNS_ENTSOE i fetch_ec.py
PRICE_BZNS = ["DE-LU", "EE", "LT", "PL", "NL", "GB"]

# NordPSA-zon → ingående MBAs
ZONE_MBAS = {
    "SE-N": ["SE1", "SE2"],
    "SE-S": ["SE3", "SE4"],
    "NO-N": ["NO3", "NO4"],
    "NO-S": ["NO1", "NO2", "NO5"],
    "DK":   ["DK1", "DK2"],
    "FI":   ["FI"],
}

# Statiska DK-vikter (eSett saknar MBA-uppdelad DK-last)
DK_LOAD_WEIGHTS = {"DK1": 0.60, "DK2": 0.40}


def _load_price_bzn(bzn: str, fallback_bzn: str | None = None) -> pd.Series:
    """Laddar och sammanfogar råprisdata för en budzon (alla år).
    Om filen saknas och fallback_bzn är satt används fallback-zonen istället.
    """
    files_missing = any(
        not (RAW_DIR / f"price_{bzn}_{year}.parquet").exists()
        for year in YEARS
    )
    if files_missing:
        if fallback_bzn is not None:
            print(f"  OBS: price_{bzn}_*.parquet saknas — använder {fallback_bzn} som proxy")
            return _load_price_bzn(fallback_bzn)
        raise FileNotFoundError(
            f"Prisdata saknas för {bzn}.\n"
            "Kör 'python scripts/fetch_ec.py' först."
        )

    frames = []
    for year in YEARS:
        path = RAW_DIR / f"price_{bzn}_{year}.parquet"
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        frames.append(df)

    s = pd.concat(frames).sort_index()["price_eur_mwh"]
    s = s[~s.index.duplicated(keep="first")]
    s = s.loc[PERIOD_START:PERIOD_END]
    # Fyll ev. glapp med forward-fill (max 2h) och sedan medelvärde
    s = s.resample("h").mean().ffill(limit=2).fillna(s.mean())
    return s


def _load_mba_price(mba: str) -> pd.Series:
    """Laddar och sammanfogar timsprisdata för en nordisk MBA."""
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"price_mba_{mba}_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"MBA-prisdata saknas: {path}\n"
                "Kör 'python scripts/fetch_ec.py' först."
            )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        frames.append(df)
    s = pd.concat(frames).sort_index()["price_eur_mwh"]
    s = s[~s.index.duplicated(keep="first")]
    s = s.loc[PERIOD_START:PERIOD_END]
    return s.resample("h").mean().ffill(limit=2).fillna(s.mean())


def _load_mba_consumption(mba: str) -> pd.Series:
    """Laddar eSett per-MBA konsumtionsdata (lastvolym för viktning)."""
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"consumption_mba_{mba}_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"MBA-konsumtionsdata saknas: {path}\n"
                "Kör 'python scripts/fetch_esett.py' först."
            )
        df = pd.read_parquet(path)
        df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values("timestampUTC")
    df = df.set_index("timestampUTC")
    df = df[~df.index.duplicated(keep="first")]
    s = df["total"].abs()
    s = s.loc[PERIOD_START:PERIOD_END]
    return s.resample("h").mean().ffill(limit=2).fillna(s.mean())


def build_zone_prices() -> pd.DataFrame:
    """
    Timsvisa zonepriser för alla 6 NordPSA-zoner som lastvolymvägda medelvärden
    av MBA-priserna inom varje zon.
    DK: statiska vikter (DK1=60%, DK2=40%) — eSett saknar MBA-uppdelad DK-last.
    """
    print("Bygger zonpriser (lastvolymvägda MBA-medelvärden) ...")
    result = {}
    for zone, mbas in ZONE_MBAS.items():
        prices, loads = {}, {}
        for mba in mbas:
            try:
                prices[mba] = _load_mba_price(mba)
            except FileNotFoundError as e:
                print(f"  OBS: {e}")
                continue
            if zone == "DK":
                loads[mba] = DK_LOAD_WEIGHTS.get(mba, 1.0 / len(mbas))
            else:
                try:
                    loads[mba] = _load_mba_consumption(mba)
                except FileNotFoundError as e:
                    print(f"  OBS: {e} — faller tillbaka på enkelt medelvärde")

        if not prices:
            print(f"  {zone}: inga prisdata — hoppar över")
            continue

        valid = list(prices.keys())
        price_df = pd.DataFrame({mba: prices[mba] for mba in valid})

        if zone == "DK":
            w = [DK_LOAD_WEIGHTS.get(mba, 1.0 / len(valid)) for mba in valid]
            w_sum = sum(w)
            zone_price = sum(price_df[mba] * (wi / w_sum) for mba, wi in zip(valid, w))
        elif all(isinstance(loads.get(mba), pd.Series) for mba in valid):
            w_df = pd.DataFrame({mba: loads[mba] for mba in valid})
            w_df = w_df.reindex(price_df.index).ffill().fillna(0).clip(lower=0)
            total = w_df.sum(axis=1).replace(0, np.nan)
            zone_price = (price_df * w_df).sum(axis=1) / total
            zone_price = zone_price.fillna(price_df.mean(axis=1))
        else:
            zone_price = price_df.mean(axis=1)

        result[zone] = zone_price
        print(f"  {zone} ({'+'.join(valid)}): medel={zone_price.mean():.1f} EUR/MWh")

    return pd.DataFrame(result)


def build_market_prices() -> pd.DataFrame:
    """
    Sammanfogar råprisdata för alla budzoner 2023-2025.
    Returnerar och sparar en DataFrame med en kolumn per budzon.
    Utdatafil: market_prices.parquet
    """
    # Fallback-zoner om rådata saknas (t.ex. GB innan ENTSO-E-token aktiverats)
    FALLBACKS = {"GB": "NL"}

    print("Bygger marknadspriser ...")
    result = {}
    for bzn in PRICE_BZNS:
        s = _load_price_bzn(bzn, fallback_bzn=FALLBACKS.get(bzn))
        result[bzn] = s
        print(f"  {bzn:<8} {len(s)} timmar  medel={s.mean():.1f} EUR/MWh")

    zone_df = build_zone_prices()
    for zone in zone_df.columns:
        result[zone] = zone_df[zone]

    out = pd.DataFrame(result)
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "market_prices.parquet")
    print(f"  → market_prices.parquet  ({len(out)} rader, {len(out.columns)} budzoner)")
    return out


# ---------------------------------------------------------------------------
# EV-laddningsprofiler (syntetiska vecka×timme, per fordonsklass)
# ---------------------------------------------------------------------------

# 24-timmars formmallar (timme 0–23, UTC). PLATSHÅLLARE = nordiska medel, TRIMMA.
# drive = relativ körenergi (skalas till årsmål i network); avail = andel inkopplad
# (0–1, → laddar-Link p_max_pu); minsoc = avgångs-/körbarhets-golv (0–1, → Store e_min_pu).
_EV_SHAPES = {
    "car": {
        "drive_wd": [0.2,0.1,0.1,0.1,0.2,0.5,1.5,3.0,2.5,1.2,0.8,0.8,0.9,0.8,0.9,1.5,3.0,3.2,2.0,1.2,0.9,0.7,0.5,0.3],
        "drive_we": [0.3,0.2,0.1,0.1,0.1,0.2,0.4,0.7,1.2,1.8,2.2,2.4,2.3,2.2,2.1,2.0,1.8,1.6,1.4,1.1,0.9,0.7,0.5,0.4],
        "avail_wd": [0.95,0.95,0.95,0.95,0.95,0.90,0.70,0.50,0.55,0.60,0.60,0.60,0.60,0.60,0.60,0.55,0.50,0.55,0.70,0.85,0.90,0.92,0.95,0.95],
        "avail_we": [0.95,0.95,0.95,0.95,0.95,0.95,0.92,0.90,0.85,0.80,0.78,0.78,0.78,0.78,0.80,0.82,0.85,0.88,0.90,0.92,0.93,0.95,0.95,0.95],
        "minsoc":   [0.45,0.50,0.55,0.62,0.68,0.70,0.68,0.55,0.45,0.40,0.35,0.35,0.35,0.35,0.35,0.38,0.42,0.45,0.45,0.45,0.45,0.45,0.45,0.45],
    },
    "heavy": {
        "drive_wd": [0.1,0.1,0.1,0.1,0.2,0.5,1.5,2.5,3.0,3.0,3.0,2.8,2.5,2.8,3.0,3.0,2.8,2.0,1.2,0.6,0.3,0.2,0.1,0.1],
        "drive_we": [0.1,0.1,0.1,0.1,0.1,0.2,0.4,0.6,0.8,0.9,1.0,1.0,1.0,1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0.1,0.1],
        "avail_wd": [0.90,0.90,0.90,0.90,0.85,0.70,0.40,0.20,0.15,0.15,0.15,0.20,0.25,0.20,0.15,0.15,0.20,0.40,0.70,0.85,0.90,0.90,0.90,0.90],
        "avail_we": [0.90,0.90,0.90,0.90,0.90,0.85,0.70,0.60,0.50,0.45,0.45,0.45,0.45,0.45,0.50,0.60,0.70,0.80,0.85,0.90,0.90,0.90,0.90,0.90],
        "minsoc":   [0.60,0.65,0.70,0.75,0.80,0.80,0.70,0.50,0.35,0.30,0.30,0.30,0.30,0.30,0.30,0.30,0.35,0.40,0.45,0.50,0.50,0.55,0.55,0.60],
    },
}


def _bridge_inflex(drive, thr: float = 0.4):
    """Oflexibel "ladda-direkt"-form ur körprofilen (SvK snabb-/kortparkeringsladdning).

    Behåller morgon- och kvällspucklarna men ersätter middagsdippen MELLAN dem med en
    konstant linje (snabb/jobb-laddning gör dagprofilen jämnare än den momentana körningen),
    och nollar natten (ingen oflexibel laddning utanför aktiv dag). Linjenivå = lägre puckeln
    → den högre puckeln sticker upp över plateauen. v1-platshållare, trimma."""
    a   = np.asarray(drive, dtype=float)
    hrs = np.arange(len(a))
    hm  = int(np.argmax(a[:12]))            # morgonpuckel (FM)
    he  = 12 + int(np.argmax(a[12:]))       # kvällspuckel (EM)
    lvl = min(a[hm], a[he])                 # konstant brygg-nivå
    out = a.copy()
    mid = (hrs > hm) & (hrs < he)           # middagsdippen mellan pucklarna
    out[mid] = np.maximum(out[mid], lvl)
    night = (a < thr) & ((hrs < hm) | (hrs > he))   # natt före/efter aktiv dag
    out[night] = 0.0
    return out


def build_ev_profiles() -> pd.DataFrame:
    """Syntetiska EV-profiler per fordonsklass (zon-oberoende vecka×timme-mall).

    Kolumner per klass: {c}_drive (relativ körenergi), {c}_avail (inkopplings-
    tillgänglighet 0–1), {c}_minsoc (körbarhets-golv 0–1), {c}_inflex (oflexibel
    ladda-direkt-form, _bridge_inflex). Skalas/appliceras per zon i network._add_ev.
    v1: ingen säsongsvariation (vinterräckvidd) — TODO."""
    print("Bygger EV-profiler (syntetiska vecka×timme) ...")
    idx  = pd.date_range(PERIOD_START, PERIOD_END, freq="h")
    hour = idx.hour.to_numpy()
    wknd = idx.dayofweek.to_numpy() >= 5           # lör/sön

    def _expand(wd, we):
        return np.where(wknd, np.array(we)[hour], np.array(wd)[hour])

    result = {}
    for c, s in _EV_SHAPES.items():
        result[f"{c}_drive"]  = _expand(s["drive_wd"], s["drive_we"])
        result[f"{c}_avail"]  = _expand(s["avail_wd"], s["avail_we"])
        result[f"{c}_minsoc"] = np.array(s["minsoc"])[hour]   # samma alla dagar (v1)
        result[f"{c}_inflex"] = _expand(_bridge_inflex(s["drive_wd"]),
                                        _bridge_inflex(s["drive_we"]))

    out = pd.DataFrame(result, index=idx)
    out.index.name = "time"
    out.to_parquet(PROCESSED_DIR / "ev_profiles.parquet")
    print(f"  → ev_profiles.parquet  ({len(out)} rader, {len(out.columns)} kolumner)")
    return out


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
    build_market_prices()
    build_ev_profiles()

    print("\nKlart! Alla indata sparade i data/processed/")
