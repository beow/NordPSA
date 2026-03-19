"""
Kalibrerar NTC-kapaciteter via 95:e percentilen av timvisa scheduled commercial
exchanges (entsoe-py query_scheduled_exchanges), i linje med Ruhnau et al. (2024).

För varje NordPSA-gräns hämtas flöden i BÅDA riktningarna per ENTSO-E BZN-par,
netto beräknas (forward – reverse), summeras över par och q95(|netto|) returneras.

Utdata: jämförelsetabell  border | pairs | current_mw | q95_mw | ratio

Användning:
    python scripts/calibrate_ntc.py              # år 2025
    python scripts/calibrate_ntc.py --years 2024 2025
    python scripts/calibrate_ntc.py --force      # ignorera cache
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RAW_DIR  = Path(__file__).resolve().parents[1] / "data" / "raw" / "flows"
CFG_PATH = Path(__file__).resolve().parents[1] / "config" / "zones.yaml"
SLEEP_S  = 0.4   # vara snäll mot ENTSO-E API

# ---------------------------------------------------------------------------
# Gränsdefinitioner: NordPSA-länk → lista av (bzn_a, bzn_b) entsoe-py Area-koder
# Netto = flow(a→b) – flow(b→a); vi tar abs() och q95.
# ---------------------------------------------------------------------------
INTERNAL_BORDERS: dict[str, list[tuple[str, str]]] = {
    "NO-N ↔ SE-N": [("NO_4", "SE_2"), ("NO_3", "SE_2")],
    "SE-N ↔ SE-S": [("SE_2", "SE_3")],
    "SE-N ↔ FI":   [("SE_1", "FI")],
    "NO-N ↔ NO-S": [("NO_3", "NO_1"), ("NO_3", "NO_5")],
    "NO-S ↔ SE-S": [("NO_1", "SE_3")],
    "NO-S ↔ DK":   [("NO_2", "DK_1")],
    "SE-S ↔ DK":   [("SE_4", "DK_2")],
    "SE-S ↔ FI":   [("SE_3", "FI")],
}

MARKET_BORDERS: dict[str, list[tuple[str, str]]] = {
    "FI ↔ EE":   [("FI",   "EE")],
    "SE-S ↔ LT": [("SE_4", "LT")],
    "SE-S ↔ PL": [("SE_4", "PL")],
    "SE-S ↔ DE": [("SE_4", "DE_LU")],
    "DK ↔ DE":   [("DK_1", "DE_LU"), ("DK_2", "DE_LU")],
    "DK ↔ NL":   [("DK_1", "NL")],
    "DK ↔ GB":   [("DK_1", "GB")],
    "NO-S ↔ DE": [("NO_2", "DE_LU")],
    "NO-S ↔ NL": [("NO_2", "NL")],
    "NO-S ↔ GB": [("NO_2", "GB")],
}

ALL_BORDERS = {**INTERNAL_BORDERS, **MARKET_BORDERS}


def raw_path(bzn_a: str, bzn_b: str, year: int) -> Path:
    return RAW_DIR / f"sched_{bzn_a}_{bzn_b}_{year}.parquet"


def fetch_one_direction(client, bzn_a: str, bzn_b: str, year: int, force: bool) -> pd.Series:
    """Hämtar scheduled exchanges a→b för ett kalenderår; cachar resultatet."""
    path = raw_path(bzn_a, bzn_b, year)
    if path.exists() and not force:
        return pd.read_parquet(path).iloc[:, 0]

    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end   = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    label = f"{bzn_a}→{bzn_b} {year}"
    print(f"    {label}: hämtar ...", end=" ", flush=True)
    try:
        s = client.query_scheduled_exchanges(bzn_a, bzn_b, start=start, end=end)
    except Exception as e:
        print(f"FEL ({e})")
        return pd.Series(dtype=float)

    if s is None or (hasattr(s, "empty") and s.empty):
        print("tomt")
        return pd.Series(dtype=float)

    s = s.resample("h").mean()
    s.name = label
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    print(f"OK ({len(s)} h, max={s.abs().max():.0f} MW)")
    time.sleep(SLEEP_S)
    return s


def net_flow_for_pair(client, bzn_a: str, bzn_b: str, year: int, force: bool) -> pd.Series:
    """Netto-flöde a→b (positiv = export från a till b)."""
    fwd = fetch_one_direction(client, bzn_a, bzn_b, year, force)
    rev = fetch_one_direction(client, bzn_b, bzn_a, year, force)
    if fwd.empty and rev.empty:
        return pd.Series(dtype=float)
    if fwd.empty:
        return -rev
    if rev.empty:
        return fwd
    return fwd.sub(rev, fill_value=0.0)


def border_net_flow(client, pairs: list[tuple[str, str]], year: int, force: bool) -> pd.Series:
    """Summerar netto-flöden för alla par som utgör en NordPSA-gräns."""
    total: pd.Series | None = None
    for bzn_a, bzn_b in pairs:
        net = net_flow_for_pair(client, bzn_a, bzn_b, year, force)
        if net.empty:
            continue
        total = net if total is None else total.add(net, fill_value=0.0)
    return total if total is not None else pd.Series(dtype=float)


def load_current_ntc(cfg_path: Path) -> dict[str, float]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ntc: dict[str, float] = {}
    for z1, z2, cap in cfg.get("links", []):
        ntc[f"{z1} ↔ {z2}"] = cap
        ntc[f"{z2} ↔ {z1}"] = cap
    for entry in cfg.get("market_connections", []):
        name, zone, cap, _ = entry
        parts = name.split()
        if len(parts) == 2:
            ntc[f"{parts[0]} ↔ {parts[1]}"] = cap
            ntc[f"{parts[1]} ↔ {parts[0]}"] = cap
    return ntc


def run(years: list[int], force: bool) -> None:
    import os
    token = os.environ.get("ENTSOE_API_TOKEN") or os.environ.get("ENTSOE_token", "")
    if not token:
        raise ValueError("Sätt ENTSOE_token eller ENTSOE_API_TOKEN i miljön.")

    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    current_ntc = load_current_ntc(CFG_PATH)

    rows = []
    for border, pairs in ALL_BORDERS.items():
        print(f"\n{border}  ({', '.join(f'{a}↔{b}' for a,b in pairs)})")
        yearly_q95 = []
        for year in years:
            net = border_net_flow(client, pairs, year, force)
            if net.empty:
                print(f"  {year}: inga data")
                continue
            q95 = net.abs().quantile(0.95)
            print(f"  {year}: q95 = {q95:.0f} MW  (n={len(net)})")
            yearly_q95.append(q95)

        if not yearly_q95:
            continue
        q95_mean = sum(yearly_q95) / len(yearly_q95)

        # Slå upp nuvarande config
        current = current_ntc.get(border)
        if current is None:
            parts = border.split(" ↔ ")
            current = current_ntc.get(f"{parts[1]} ↔ {parts[0]}")

        rows.append({
            "border":     border,
            "pairs":      " + ".join(f"{a}↔{b}" for a, b in pairs),
            "current_mw": current,
            "q95_mw":     round(q95_mean),
            "ratio":      round(q95_mean / current, 2) if current else None,
        })

    print("\n" + "=" * 80)
    print(f"{'Border':<22} {'Pairs':<28} {'Current':>8} {'q95':>8} {'Ratio':>6}  {'':>3}")
    print("-" * 80)
    for r in rows:
        cur_s   = f"{r['current_mw']:.0f}" if r["current_mw"] else "   —"
        rat_s   = f"{r['ratio']:.2f}"       if r["ratio"]     else "  —"
        flag    = " <-- HÖJNING" if r["ratio"] and r["ratio"] > 1.15 else (
                  " (sänkning?)" if r["ratio"] and r["ratio"] < 0.80 else "")
        print(f"{r['border']:<22} {r['pairs']:<28} {cur_s:>8} {r['q95_mw']:>8} {rat_s:>6}{flag}")
    print("=" * 80)
    print("Ratio > 1.15: q95 > 15% över config  |  ratio < 0.80: config > 25% över q95")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=[2025])
    parser.add_argument("--force", action="store_true", help="Ignorera cache")
    args = parser.parse_args()
    run(args.years, args.force)
