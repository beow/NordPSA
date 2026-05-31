"""
Beräknar faktiska vattenkraftstillflöden för norska och svenska NordPSA-zoner
från reservoardata och ENTSO-E faktisk generering.

Metodik
-------
NVE:s eget sätt att beräkna "nyttbart tilsig" (sedan 2015):

    inflow_week ≈ hydro_production_week + Δreservoir_week

Δreservoir = (fylling_TWh[t] - fylling_TWh[t-1]) per vecka, summerat per zon.
Produktion: ENTSO-E A75/A16 faktisk generering (B11 + B12 + B10) per budzon.

Notat: spill inkluderas inte — resultatet är en underskattning av det naturliga
inflödet vid perioder med spillning.

Datakällor per land
-------------------
  Norge  – reservoar: NVE Magasinstatistikk API (veckovis, per priszon)
  Sverige – reservoar: ENTSO-E A72/A16 (veckovis, per budzon)
  Båda   – produktion: ENTSO-E A75/A16 (timvis, per budzon)

Zonmappning
-----------
  NO-N  ←  NVE omrnr 3 (NO3) + 4 (NO4)        ENTSO-E prod: NO3, NO4
  NO-S  ←  NVE omrnr 1 (NO1) + 2 (NO2) + 5   ENTSO-E prod: NO1, NO2, NO5
  SE-N  ←  ENTSO-E reservoar + prod: SE1, SE2
  SE-S  ←  ENTSO-E reservoar + prod: SE3, SE4

Run-of-river separeras
----------------------
ENTSO-E B11 (run-of-river) modelleras som separat must-run-generator i network.py,
inte som lagringsbart reservoarinflöde. Reservoarinflödet (inflow_nve) använder därför
endast reservoarproduktion (B12+B10). RoR-timprofilen sparas separat.

Utdata
------
  data/raw/inflow_nve_{zone}_{year}.parquet  reservoarinflöde (vecka), exkl. RoR
  data/raw/ror_nve_{zone}_{year}.parquet     run-of-river must-run-profil (timme)

Kolumner (inflow_nve):
  week_start   UTC-tidsstämpel för veckostart (måndag 00:00)
  inflow_gwh   GWh reservoarinflöde under veckan
  inflow_mw    Medeleffekt (MW) = inflow_gwh * 1000 / 168

Användning:
    python scripts/fetch_nve.py                  # beräkna allt som saknas
    python scripts/fetch_nve.py --force          # skriv över befintliga filer
    python scripts/fetch_nve.py --fetch-entsoe   # hämta ENTSO-E-data först
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.entsoe import ENTSOEClient

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS   = [2023, 2024, 2025]

# Smoothing-fönster (veckor) för inflödesprofilen. 1 = ingen smoothing (skarp vårflod, default).
# Reservoaren (spill_cost) hanterar topparna; skarp vårflod är mer realistisk.
SMOOTH_WEEKS = 1

NVE_API_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

# NordPSA-zon → NVE omrnr och ENTSO-E budzoner
NO_ZONE_MAP = {
    "NO-N": {"omrnr": [3, 4],    "bzns": ["NO3", "NO4"]},
    "NO-S": {"omrnr": [1, 2, 5], "bzns": ["NO1", "NO2", "NO5"]},
}

# NordPSA-zon → ENTSO-E budzoner (reservoar + produktion)
SE_ZONE_MAP = {
    "SE-N": {"bzns": ["SE1", "SE2"]},
    "SE-S": {"bzns": ["SE3", "SE4"]},
}

HOURS_PER_WEEK = 168.0


# ---------------------------------------------------------------------------
# Sökvägar
# ---------------------------------------------------------------------------

def raw_path(zone: str, year: int) -> Path:
    return RAW_DIR / f"inflow_nve_{zone}_{year}.parquet"


def entsoe_hydro_path(bzn: str, year: int) -> Path:
    return RAW_DIR / f"hydro_entsoe_{bzn}_{year}.parquet"


def entsoe_reservoir_path(bzn: str, year: int) -> Path:
    return RAW_DIR / f"reservoir_entsoe_{bzn}_{year}.parquet"


def entsoe_ror_path(bzn: str, year: int) -> Path:
    return RAW_DIR / f"ror_entsoe_{bzn}_{year}.parquet"


def ror_profile_path(zone: str, year: int) -> Path:
    return RAW_DIR / f"ror_nve_{zone}_{year}.parquet"


# ---------------------------------------------------------------------------
# ENTSO-E: hämta hydrogenerering (A75) och reservoar (A72)
# ---------------------------------------------------------------------------

def fetch_entsoe_hydro(force: bool = False) -> None:
    """Hämtar och cachar ENTSO-E hydrogenerering för NO- och SE-budzoner."""
    cli = ENTSOEClient()
    all_bzns = sorted(
        {b for v in NO_ZONE_MAP.values() for b in v["bzns"]}
        | {b for v in SE_ZONE_MAP.values() for b in v["bzns"]}
    )
    total = len(all_bzns) * len(YEARS)
    done = 0
    for bzn in all_bzns:
        for year in YEARS:
            done += 1
            out = entsoe_hydro_path(bzn, year)
            if out.exists() and not force:
                print(f"  [{done}/{total}] hoppar {out.name}")
                continue
            print(f"  [{done}/{total}] A75 {bzn} {year} ...", end=" ", flush=True)
            try:
                s = cli.fetch_hydro_year(bzn, year)
            except Exception as e:
                print(f"FEL: {e}")
                continue
            s.rename("hydro_mw").reset_index().rename(
                columns={"index": "timestampUTC"}
            ).to_parquet(out, index=False)
            print(f"OK  {len(s)} h  {round(s.sum()/1e6, 2)} TWh")
            time.sleep(1.0)


def fetch_entsoe_reservoirs(force: bool = False) -> None:
    """Hämtar och cachar ENTSO-E reservoarnivåer (A72) för SE-budzoner."""
    cli = ENTSOEClient()
    all_bzns = sorted({b for v in SE_ZONE_MAP.values() for b in v["bzns"]})
    total = len(all_bzns) * len(YEARS)
    done = 0
    for bzn in all_bzns:
        for year in YEARS:
            done += 1
            out = entsoe_reservoir_path(bzn, year)
            if out.exists() and not force:
                print(f"  [{done}/{total}] hoppar {out.name}")
                continue
            print(f"  [{done}/{total}] A72 {bzn} {year} ...", end=" ", flush=True)
            try:
                s = cli.fetch_reservoir_year(bzn, year)
            except Exception as e:
                print(f"FEL: {e}")
                continue
            s.rename("reservoir_mwh").reset_index().rename(
                columns={"index": "timestampUTC"}
            ).to_parquet(out, index=False)
            print(f"OK  {len(s)} punkter  max={round(s.max()/1e6, 1)} TWh")
            time.sleep(1.0)


def fetch_entsoe_ror(force: bool = False) -> None:
    """Hämtar och cachar ENTSO-E run-of-river (B11) per timme för NO- och SE-budzoner."""
    cli = ENTSOEClient()
    all_bzns = sorted(
        {b for v in NO_ZONE_MAP.values() for b in v["bzns"]}
        | {b for v in SE_ZONE_MAP.values() for b in v["bzns"]}
    )
    total = len(all_bzns) * len(YEARS)
    done = 0
    for bzn in all_bzns:
        for year in YEARS:
            done += 1
            out = entsoe_ror_path(bzn, year)
            if out.exists() and not force:
                print(f"  [{done}/{total}] hoppar {out.name}")
                continue
            print(f"  [{done}/{total}] B11 {bzn} {year} ...", end=" ", flush=True)
            try:
                s = cli.fetch_hydro_year(bzn, year, psr_types=["B11"])
            except Exception as e:
                print(f"FEL: {e}")
                continue
            s.rename("ror_mw").reset_index().rename(
                columns={"index": "timestampUTC"}
            ).to_parquet(out, index=False)
            print(f"OK  {len(s)} h  {round(s.sum()/1e6, 2)} TWh")
            time.sleep(1.0)


def load_entsoe_hydro(zone: str, kind: str = "total") -> pd.Series:
    """
    Laddar ENTSO-E hydrogenerering (MW, timvis) summerad för en NordPSA-zon.

    kind="total"     → B11+B12+B10 (cachad hydro_entsoe-fil)
    kind="ror"       → B11 (run-of-river, cachad ror_entsoe-fil)
    kind="reservoir" → total − ror (reservoarkraft, B12+B10)
    """
    cfg = NO_ZONE_MAP.get(zone) or SE_ZONE_MAP.get(zone)
    assert cfg, f"Okänd zon: {zone}"

    def _load(path_fn, col):
        frames = []
        for bzn in cfg["bzns"]:
            for year in YEARS:
                p = path_fn(bzn, year)
                if not p.exists():
                    raise FileNotFoundError(f"Saknar fil: {p.name} — kör med --fetch-entsoe.")
                df = pd.read_parquet(p)
                df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
                frames.append(df.set_index("timestampUTC")[col].fillna(0.0))
        combined = pd.concat(frames).sort_index()
        return combined.groupby(combined.index).sum()

    if kind == "total":
        return _load(entsoe_hydro_path, "hydro_mw")
    if kind == "ror":
        return _load(entsoe_ror_path, "ror_mw")
    if kind == "reservoir":
        total = _load(entsoe_hydro_path, "hydro_mw")
        ror   = _load(entsoe_ror_path, "ror_mw").reindex(total.index).fillna(0.0)
        return (total - ror).clip(lower=0.0)
    raise ValueError(f"Okänd kind: {kind}")


# ---------------------------------------------------------------------------
# NVE: reservoarnivåer (Norge)
# ---------------------------------------------------------------------------

def fetch_nve_reservoir() -> pd.DataFrame:
    """Hämtar all magasinstatistik från NVE i ett anrop."""
    print("Hämtar NVE Magasinstatistikk ...", end=" ", flush=True)
    resp = requests.get(NVE_API_URL, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    print(f"OK ({len(df)} rader, {df['dato_Id'].min()[:4]}–{df['dato_Id'].max()[:4]})")
    return df


def build_reservoir_weekly_nve(nve_df: pd.DataFrame, omrnr_list: list[int]) -> pd.DataFrame:
    """
    Aggregerar NVE-reservoarnivåer (TWh) per vecka för en mängd omrnr.

    NVE publicerar två parallella rapporteringssystem per zon (alternerar veckovis).
    Filtrerar till fullständig serie (max kapacitet), interpolerar luckorna.
    """
    nve_df = nve_df.copy()
    nve_df["dato_Id"] = pd.to_datetime(nve_df["dato_Id"])

    per_omrnr = []
    for omrnr in omrnr_list:
        sub = nve_df[nve_df["omrnr"] == omrnr].copy()
        max_kap = sub["kapasitet_TWh"].max()
        full = sub[sub["kapasitet_TWh"] >= max_kap * 0.95].copy()
        full = full.drop_duplicates(subset=["iso_aar", "iso_uke"], keep="last")
        full["week_start"] = pd.to_datetime(
            full["iso_aar"].astype(str)
            + "-W" + full["iso_uke"].astype(str).str.zfill(2)
            + "-1",
            format="%G-W%V-%u",
            utc=True,
        )
        ts = full.set_index("week_start")["fylling_TWh"].sort_index()
        per_omrnr.append(ts.rename(f"omrnr_{omrnr}"))

    combined = pd.concat(per_omrnr, axis=1)
    full_idx = pd.date_range(
        combined.index.min(), combined.index.max(), freq="W-MON", tz="UTC"
    )
    combined = combined.reindex(full_idx).interpolate(method="linear")
    total = combined.sum(axis=1).rename("fylling_TWh").reset_index()
    total.columns = ["week_start", "fylling_TWh"]
    total["delta_twh"] = total["fylling_TWh"].diff().fillna(0.0)
    return total


# ---------------------------------------------------------------------------
# ENTSO-E: reservoarnivåer (Sverige)
# ---------------------------------------------------------------------------

def build_reservoir_weekly_entsoe(bzn_list: list[str]) -> pd.DataFrame:
    """
    Aggregerar ENTSO-E-reservoarnivåer (TWh) per vecka för en mängd SE-budzoner.
    Läser cachade A72-filer. Returnerar DataFrame med week_start, fylling_TWh, delta_twh.
    """
    per_bzn = []
    for bzn in bzn_list:
        frames = []
        for year in YEARS:
            p = entsoe_reservoir_path(bzn, year)
            if not p.exists():
                raise FileNotFoundError(
                    f"Saknar ENTSO-E reservoar: {p.name} — kör med --fetch-entsoe."
                )
            df = pd.read_parquet(p)
            df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
            frames.append(df.set_index("timestampUTC")["reservoir_mwh"].ffill())
        ts = pd.concat(frames).sort_index()
        ts = ts[~ts.index.duplicated(keep="last")]
        # Konvertera MWh → TWh och resampla till veckostart (måndag)
        ts_twh = (ts / 1e6).resample("W-MON", label="left", closed="left").last()
        per_bzn.append(ts_twh.rename(bzn))

    combined = pd.concat(per_bzn, axis=1)
    full_idx = pd.date_range(
        combined.index.min(), combined.index.max(), freq="W-MON", tz="UTC"
    )
    combined = combined.reindex(full_idx).interpolate(method="linear")
    total = combined.sum(axis=1).rename("fylling_TWh").reset_index()
    total.columns = ["week_start", "fylling_TWh"]
    total["delta_twh"] = total["fylling_TWh"].diff().fillna(0.0)
    return total


# ---------------------------------------------------------------------------
# Inflödesberäkning (gemensam)
# ---------------------------------------------------------------------------

def build_inflow(zone: str, year: int, reservoir_wkly: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar veckovist RESERVOAR-inflöde för en zon och ett år.
    Run-of-river (B11) exkluderas — den modelleras som separat must-run-generator.
    Returnerar DataFrame med: week_start, inflow_gwh, inflow_mw.
    """
    hydro_h = load_entsoe_hydro(zone, kind="reservoir")

    hydro_wk = (
        (hydro_h.resample("W-MON", label="left", closed="left").sum() / 1000.0)
        .rename("hydro_gwh")
        .reset_index()
        .rename(columns={"timestampUTC": "week_start"})
    )

    res_year = reservoir_wkly[reservoir_wkly["week_start"].dt.year == year].copy()
    merged = pd.merge(res_year, hydro_wk, on="week_start", how="left")
    merged["hydro_gwh"] = merged["hydro_gwh"].fillna(0.0)

    inflow_raw = merged["hydro_gwh"] + merged["delta_twh"] * 1000.0

    # Smoothing: SMOOTH_WEEKS-veckors centrerat rullande medel (default 1 = ingen).
    # annual_target beräknas på osmoothat för korrekt energibalans.
    annual_target = float(inflow_raw.sum())
    inflow_smooth = inflow_raw.rolling(window=SMOOTH_WEEKS, center=True, min_periods=1).mean()

    # Clip till ≥0 och rescala till korrekt årstotal
    inflow_clipped = inflow_smooth.clip(lower=0.0)
    clipped_sum = float(inflow_clipped.sum())
    if clipped_sum > 0 and 0 < annual_target < clipped_sum:
        inflow_clipped = inflow_clipped * (annual_target / clipped_sum)

    merged["inflow_gwh"] = inflow_clipped
    merged["inflow_mw"]  = merged["inflow_gwh"] * 1000.0 / HOURS_PER_WEEK
    return merged[["week_start", "inflow_gwh", "inflow_mw"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Huvudflöde
# ---------------------------------------------------------------------------

def fetch_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Norge: NVE-reservoar ---
    nve_df = fetch_nve_reservoir()
    for zone, cfg in NO_ZONE_MAP.items():
        print(f"\n=== {zone} (NVE omrnr {cfg['omrnr']}, ENTSO-E prod {cfg['bzns']}) ===")
        reservoir_wkly = build_reservoir_weekly_nve(nve_df, cfg["omrnr"])
        _compute_and_save(zone, reservoir_wkly, force)

    # --- Sverige: ENTSO-E-reservoar ---
    for zone, cfg in SE_ZONE_MAP.items():
        print(f"\n=== {zone} (ENTSO-E reservoar + prod {cfg['bzns']}) ===")
        try:
            reservoir_wkly = build_reservoir_weekly_entsoe(cfg["bzns"])
        except FileNotFoundError as e:
            print(f"  FEL: {e}")
            continue
        _compute_and_save(zone, reservoir_wkly, force)

    print("\nKlar.")


def _compute_and_save(zone: str, reservoir_wkly: pd.DataFrame, force: bool) -> None:
    for year in YEARS:
        out = raw_path(zone, year)
        if out.exists() and not force:
            print(f"  {year}: hoppar över {out.name} (finns redan)")
            continue
        print(f"  {year}: beräknar ...", end=" ", flush=True)
        try:
            df = build_inflow(zone, year, reservoir_wkly)
        except FileNotFoundError as e:
            print(f"FEL: {e}")
            continue
        total_twh = df["inflow_gwh"].sum() / 1000
        df.to_parquet(out, index=False)
        print(f"OK  {len(df)} veckor  {total_twh:.1f} TWh → {out.name}")
    _save_ror_profile(zone, force)


def _save_ror_profile(zone: str, force: bool) -> None:
    """Sparar run-of-river (B11) timprofil per NordPSA-zon, ett år per fil."""
    try:
        ror = load_entsoe_hydro(zone, kind="ror")
    except FileNotFoundError as e:
        print(f"  RoR: {e}")
        return
    for year in YEARS:
        out = ror_profile_path(zone, year)
        if out.exists() and not force:
            continue
        s = ror[ror.index.year == year]
        if s.empty:
            continue
        df = s.rename("ror_mw").reset_index().rename(columns={"index": "timestampUTC"})
        df.to_parquet(out, index=False)
        print(f"  RoR {year}: {s.sum()/1e6:.1f} TWh  max={s.max():.0f} MW → {out.name}")


def print_summary() -> None:
    print(f"\n{'Zon':<8} {'År':<6} {'TWh':>7}  {'Medel MW':>9}")
    print("-" * 34)
    for zone in list(NO_ZONE_MAP) + list(SE_ZONE_MAP):
        for year in YEARS:
            out = raw_path(zone, year)
            if not out.exists():
                continue
            df = pd.read_parquet(out)
            print(f"{zone:<8} {year:<6} {df['inflow_gwh'].sum()/1000:>7.1f}  {df['inflow_mw'].mean():>9.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force",        action="store_true", help="Skriv över befintliga filer")
    parser.add_argument("--fetch-entsoe", action="store_true", help="Hämta ENTSO-E-data innan beräkning")
    parser.add_argument("--summary",      action="store_true", help="Visa sammanfattning utan att hämta")
    parser.add_argument("--smooth-weeks", type=int, default=SMOOTH_WEEKS,
                        help="Smoothing-fönster i veckor (1 = ingen smoothing, skarp vårflod). Standard: 4")
    args = parser.parse_args()
    SMOOTH_WEEKS = args.smooth_weeks

    if args.summary:
        print_summary()
    else:
        if args.fetch_entsoe:
            print("=== Hämtar ENTSO-E hydrogenerering ===")
            fetch_entsoe_hydro(force=args.force)
            print("\n=== Hämtar ENTSO-E run-of-river (B11) ===")
            fetch_entsoe_ror(force=args.force)
            print("\n=== Hämtar ENTSO-E reservoarnivåer (SE) ===")
            fetch_entsoe_reservoirs(force=args.force)
        fetch_all(force=args.force)
        print_summary()
