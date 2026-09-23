"""Rotationsenergi och kortslutningseffekt i en löst körning (ingen LP, fungerar på gamla körningar).

Skriver stability_capacity.csv, stability_timeseries.csv och stability_summary.csv i
körningens katalog och skriver ut kapacitets- och sammanfattningstabellen.
Metoden (lo/hi/max) beskrivs i nordpsa/analysis/stability.py.

Användning:
    python scripts/stability_report.py results/run460_baseline_2h
    python scripts/stability_report.py results/run460_baseline_dispatch_1h --sync-weight DK=0.35
    python scripts/stability_report.py results/run460_baseline_2h --thresholds 100 120 145 --no-write
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.analysis.stability import stability_report, write_stability_reports  # noqa: E402


def _weights(items: list[str]) -> dict:
    out = {}
    for item in items:
        zone, _, w = item.partition("=")
        out[zone] = float(w)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--sync-weight", nargs="*", default=["DK=0.35"],
                    help="zonvikt i det nordiska systemvärdet, ZON=VIKT (default DK=0.35, ⚠️ okalibrerat)")
    ap.add_argument("--thresholds", nargs="*", type=float, default=[100, 120, 145],
                    help="E_k-trösklar för systemet [GWs] att räkna timmar under")
    ap.add_argument("--no-write", action="store_true", help="skriv inga CSV:er")
    args = ap.parse_args()

    logging.getLogger("pypsa").setLevel(logging.WARNING)
    n = pypsa.Network(args.run_dir / "network.nc")
    sw = _weights(args.sync_weight)

    rep = stability_report(n, sw, tuple(args.thresholds))
    cap, summ = rep["capacity"], rep["summary"]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    print(f"\n{args.run_dir.name}: {len(n.snapshots)} snapshots, sync_weight {sw}")
    print("\nKapacitet per zon och teknik (allt inkopplat, utan p_max_pu):")
    tot = cap.groupby("tech").sum()
    tot.index = pd.MultiIndex.from_product([["TOTAL"], tot.index])
    print(pd.concat([cap, tot]).round(1).to_string())
    print("\nE_k [GWs], S_k [GVA], SCR per zon (lo = u=p, hi = u=p/m_min, max = allt tillgängligt; SCR mot inmatad IBR):")
    print(summ.round(1).to_string())

    if not args.no_write:
        write_stability_reports(rep, args.run_dir)
        print(f"\nSkrev stability_*.csv till {args.run_dir}")


if __name__ == "__main__":
    main()
