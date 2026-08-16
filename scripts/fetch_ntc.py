"""Hämtar PUBLICERAD överföringskapacitet och fysiska flöden från ENTSO-E.

Kompletterar `calibrate_ntc.py`, som hämtar scheduled commercial exchanges — alltså
HANDELSVOLYMER. De är uppåt begränsade av kapaciteten men lika med den bara när gränsen
är trängselutsatt, så q95 av dem underskattar systematiskt taket på gränser som sällan
binder (mätt 2026-08-16: bara NO-S–DK och SE-S–FI binder >15 % av timmarna).

Tre serier hämtas, alla per riktning och kalenderår, cachade som parquet:

  cap_da    query_offered_capacity(contract_marketagreement_type="A01") — den PUBLICERADE
            day-ahead-kapaciteten per riktning och timme. Det här är "installerad NTC" i
            den mening frågan brukar avse, och den VARIERAR med avbrott och driftläge
            (SE1→FI 1230-1560 MW under en enda vecka).
            ⚠️ query_net_transfer_capacity_dayahead ger NoMatchingDataError för samtliga
            nordiska gränser — den vägen finns inte, testat 2026-08-16.
            ⛔ Publiceringen UPPHÖR vid flow-based go-live 2024-10-30: mätt
            SE2→SE3 median 5211 och NO2→DK1 1680 i september 2024, NoMatchingDataError i
            november 2024 och hela 2025. Under FB finns ingen kapacitet per gräns att
            publicera. Att serien tar slut är alltså ett RESULTAT, inte ett hämtningsfel.
            Validerat att den ÄR ett tak: schemat ligger under den i 90-100 % av timmarna
            (de få överskridandena är ≤200 MW, sannolikt intradag ovanpå day-ahead).
  phys      query_crossborder_flows — fysiskt flöde. Skiljer sig från det kommersiella
            schemat på maskat AC-nät (loop- och transitflöden följer Kirchhoff, inte
            kontrakt). NordPSA:s länkar är kommersiella zonöverföringar, så phys är till
            för jämförelse, inte för kalibrering.
  netpos    query_net_position — zonens nettoposition, den storhet flow-based faktiskt
            begränsar. Per zon, inte per gräns.

⛔ FB-domänen (CNEC/PTDF/RAM) finns INTE i entsoe-py 0.8.0 och inte på ENTSO-E:s
transparensplattform i hämtbar form. Den publiceras av JAO (Nordic publication tool) med
eget API — ett separat jobb.

Kräver ENTSOE_token eller ENTSOE_API_TOKEN i miljön.

    python scripts/fetch_ntc.py                      # NTC + fysiska flöden, 2023-2025
    python scripts/fetch_ntc.py --what cap_da        # bara publicerad kapacitet
    python scripts/fetch_ntc.py --years 2024 2025 --force
    python scripts/fetch_ntc.py --report             # bara tabellen, hämtar inget nytt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.calibrate_ntc import INTERNAL_BORDERS, MARKET_BORDERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "capacity"
FLOWS = ROOT / "data" / "raw" / "flows"
CFG = ROOT / "config" / "zones.yaml"
SLEEP_S = 0.4
FB_GOLIVE = pd.Timestamp("2024-10-30")

WHAT = {
    "cap_da": ("query_offered_capacity", {"contract_marketagreement_type": "A01"}),
    "phys": ("query_crossborder_flows", {}),
}


def path_for(what: str, a: str, b: str, year: int) -> Path:
    return RAW / f"{what}_{a}_{b}_{year}.parquet"


def fetch_direction(client, what: str, a: str, b: str, year: int, force: bool) -> pd.Series:
    p = path_for(what, a, b, year)
    if p.exists() and not force:
        return pd.read_parquet(p).iloc[:, 0]
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    print(f"    {what} {a}→{b} {year}: ", end="", flush=True)
    try:
        meth, kw = WHAT[what]
        s = getattr(client, meth)(a, b, start=start, end=end, **kw)
    except Exception as e:  # tomt svar är vanligt och förväntat efter FB
        print(f"– ({type(e).__name__}: {str(e)[:60]})")
        return pd.Series(dtype=float)
    if s is None or len(s) == 0:
        print("tomt")
        return pd.Series(dtype=float)
    s = s.resample("h").mean()
    s.name = f"{a}→{b} {year}"
    p.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(p)
    last = s.dropna().index.max()
    print(f"OK ({len(s.dropna())} h, max {s.max():.0f} MW, sista {last:%Y-%m-%d})")
    time.sleep(SLEEP_S)
    return s


def read_cached(what: str, a: str, b: str) -> pd.Series | None:
    out = []
    for p in sorted(RAW.glob(f"{what}_{a}_{b}_*.parquet")):
        s = pd.read_parquet(p).iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
        out.append(s)
    return pd.concat(out).sort_index() if out else None


def read_sched(a: str, b: str) -> pd.Series | None:
    out = []
    for p in sorted(FLOWS.glob(f"sched_{a}_{b}_*.parquet")):
        s = pd.read_parquet(p).iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True).tz_localize(None)
        out.append(s)
    return pd.concat(out).sort_index() if out else None


def config_ntc() -> dict[str, float]:
    cfg = yaml.safe_load(open(CFG))
    return {f"{z1} ↔ {z2}": cap for z1, z2, cap in cfg.get("links", [])}


def report(borders: dict) -> None:
    """En rad per gräns: config mot publicerad NTC mot handelsvolymens q95."""
    cfg = config_ntc()
    print(f"\n{'gräns':14s} {'config':>7s} | {'NTC pub':>8s} {'NTC min':>8s} "
          f"{'sista NTC':>10s} | {'sched q95':>9s} {'q95/NTC':>8s}")
    print("-" * 74)
    for name, pairs in borders.items():
        ntc_vals, last_dates = [], []
        sched = None
        for a, b in pairs:
            for x, y, sign in ((a, b, +1), (b, a, -1)):
                s = read_cached("cap_da", x, y)
                if s is not None and s.notna().any():
                    ntc_vals.append(s.dropna())
                    last_dates.append(s.dropna().index.max())
                f = read_sched(x, y)
                if f is not None:
                    f = sign * f
                    sched = f if sched is None else sched.add(f, fill_value=0)
        key = name.replace(" ↔ ", " ↔ ")
        c = cfg.get(key, float("nan"))
        if ntc_vals:
            allv = pd.concat(ntc_vals)
            mode = allv.round(-1).mode()
            ntc = float(mode.iloc[0]) if len(mode) else float(allv.max())
            lo = float(allv.quantile(0.05))
            last = max(last_dates)
            q95 = float(sched.abs().quantile(0.95)) if sched is not None else float("nan")
            ratio = f"{q95 / ntc:.0%}" if ntc else "–"
            print(f"{name:14s} {c:7.0f} | {ntc:8.0f} {lo:8.0f} {last:%Y-%m-%d} | "
                  f"{q95:9.0f} {ratio:>8s}")
        else:
            q95 = float(sched.abs().quantile(0.95)) if sched is not None else float("nan")
            print(f"{name:14s} {c:7.0f} | {'–':>8s} {'–':>8s} {'–':>10s} | {q95:9.0f} {'–':>8s}")
    print("\nNTC pub = vanligaste publicerade day-ahead-värdet (avrundat till 10 MW).")
    print("NTC min = 5:e percentilen — kapaciteten varierar med avbrott och driftläge.")
    print(f"⚠️ Slutar serien nära {FB_GOLIVE:%Y-%m-%d} är det flow-based, inte ett fel:")
    print("   under FB publiceras ingen kapacitet per gräns för nordiska interna snitt.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--what", nargs="+", default=["cap_da", "phys"], choices=list(WHAT))
    ap.add_argument("--borders", default="internal", choices=["internal", "market", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true", help="hämta inget, skriv bara tabellen")
    args = ap.parse_args()

    borders = {"internal": INTERNAL_BORDERS, "market": MARKET_BORDERS,
               "all": {**INTERNAL_BORDERS, **MARKET_BORDERS}}[args.borders]

    if not args.report:
        token = os.environ.get("ENTSOE_API_TOKEN") or os.environ.get("ENTSOE_token", "")
        if not token:
            raise SystemExit("Sätt ENTSOE_token eller ENTSOE_API_TOKEN i miljön.")
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=token)
        for what in args.what:
            print(f"\n=== {what} ({WHAT[what][0]}) ===")
            for name, pairs in borders.items():
                print(f"  {name}")
                for a, b in pairs:
                    for year in args.years:
                        fetch_direction(client, what, a, b, year, args.force)
                        fetch_direction(client, what, b, a, year, args.force)

    report(borders)


if __name__ == "__main__":
    main()
