#!/usr/bin/env python
"""
Plotta EV-laddning uppdelad på fast (oflexibel) och flexibel del för en
vald vecka/period ur en löst NordPSA-körning.

Datakällor (SvK-formuleringen, se nordpsa/network.py:_add_ev):
  - Fast/oflexibel: Load '{zon} EV {klass} inflex'  (fast AC-last, körprofilform)
  - Flexibel:       Link '{zon} EV {klass} charger' (laddflöde p0, klippt ≥0)
  klass ∈ {car, heavy}. Systemtotal = summa över alla zoner (eller --zone).

Exempel:
  python scripts/plot_ev_charging.py --input run147_chpfixed5_3h --start 2024-06-03
  python scripts/plot_ev_charging.py results/run147_chpfixed5_3h --start 2024-06-03
  python scripts/plot_ev_charging.py -i run155_nuc3_off3_onshore80_3h \
      --start 2024-01-15 --days 7 --output temp/run155_ev_jan.png
  python scripts/plot_ev_charging.py -i run147_chpfixed5_3h --start 2024-06-03 --zone SE-S
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pypsa


CLASSES = ("car", "heavy")


def ev_series(n: pypsa.Network, zones: list[str]):
    """Returnerar (fast_MW, flexibel_MW) som pd.Series över alla snapshots."""
    idx = n.snapshots
    fixed = pd.Series(0.0, index=idx)
    flex = pd.Series(0.0, index=idx)
    for z in zones:
        for c in CLASSES:
            inflex = f"{z} EV {c} inflex"
            if inflex in n.loads.index:
                # Fast last: använd faktiskt levererad (loads_t.p) annars p_set
                if inflex in n.loads_t.p.columns:
                    fixed = fixed.add(n.loads_t.p[inflex], fill_value=0.0)
                elif inflex in n.loads_t.p_set.columns:
                    fixed = fixed.add(n.loads_t.p_set[inflex], fill_value=0.0)
            charger = f"{z} EV {c} charger"
            if charger in n.links.index and charger in n.links_t.p0.columns:
                flex = flex.add(n.links_t.p0[charger].clip(lower=0), fill_value=0.0)
    return fixed, flex


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", default=None,
                    help="Sökväg till results/<run> (innehåller network.nc)")
    ap.add_argument("--input", "-i", dest="input", default=None,
                    help="Run att plotta: runnamn (slås upp under results/) eller full sökväg. "
                         "Alternativ till det positionella results_dir.")
    ap.add_argument("--start", required=True, help="Periodens startdatum, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=7, help="Antal dagar (default 7)")
    ap.add_argument("--end", default=None, help="Slutdatum YYYY-MM-DD (override på --days)")
    ap.add_argument("--zone", default=None,
                    help="Begränsa till en zon (default: systemtotal över alla EV-zoner)")
    ap.add_argument("--output", default=None,
                    help="Utdata-PNG (default temp/<run>_ev_charging_<start>.png)")
    args = ap.parse_args()

    target = args.input or args.results_dir
    if not target:
        raise SystemExit("Ange körningen via --input <run> eller som positionellt argument.")
    results_root = Path(__file__).resolve().parents[1] / "results"
    rd = Path(target)
    if not (rd / "network.nc").exists():
        # 1) exakt runnamn under results/
        if (results_root / target / "network.nc").exists():
            rd = results_root / target
        else:
            # 2) prefix-matchning: mappar under results/ som börjar med target
            matches = sorted(d for d in results_root.glob(f"{target}*")
                             if (d / "network.nc").exists())
            if len(matches) == 1:
                rd = matches[0]
            elif len(matches) > 1:
                names = ", ".join(m.name for m in matches)
                raise SystemExit(f"Flera körningar matchar '{target}*': {names}. "
                                 f"Ange ett mer specifikt namn.")
    ncfile = rd / "network.nc"
    if not ncfile.exists():
        raise SystemExit(f"Hittar ingen network.nc för '{target}' "
                         f"(provade sökväg, results/{target}/ och results/{target}*/)")
    label = rd.name
    n = pypsa.Network(str(ncfile))

    # EV-zoner i nätet
    all_zones = sorted({lab.split(" EV ")[0] for lab in n.loads.index if " EV " in lab}
                       | {lab.split(" EV ")[0] for lab in n.links.index if " EV " in lab})
    zones = [args.zone] if args.zone else all_zones
    if not zones:
        raise SystemExit("Inga EV-komponenter i nätet — kördes körningen utan EV?")

    fixed, flex = ev_series(n, zones)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else start + pd.Timedelta(days=args.days)
    mask = (fixed.index >= start) & (fixed.index < end)
    if not mask.any():
        raise SystemExit(f"Inga snapshots i [{start}, {end}) — kontrollera --start/perioden.")
    t = fixed.index[mask]
    f_gw = fixed[mask] / 1e3
    x_gw = flex[mask] / 1e3
    tot_gw = f_gw + x_gw

    # Stapelbredd = snapshot-längd i dagar (för datetime-x)
    dt_h = (t[1] - t[0]).total_seconds() / 3600 if len(t) > 1 else 3.0
    width = dt_h / 24.0

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(t, f_gw, width=width, align="edge", color="#6f8fb0",
           label="Fast (oflexibel)")
    ax.bar(t, x_gw, width=width, align="edge", bottom=f_gw, color="#e0a85a",
           label="Flexibel")
    # Total-linje centrerad i intervallet
    ax.plot(t + pd.Timedelta(hours=dt_h / 2), tot_gw, color="black", lw=1.3,
            label="Total EV-laddning")

    scope = args.zone if args.zone else "systemtotal"
    ax.set_ylabel("EV-laddning från elbussen [GW]")
    ax.set_title(f"EV-laddning per typ — {start.date()}–{(end - pd.Timedelta(days=1)).date()} "
                 f"({label}, {scope}, {int(dt_h)}h)")
    # Legendordning: Fast, Flexibel, Total (matchar bar-staplingen)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: ("Total" in labels[i], "Flexibel" in labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(args.output) if args.output else \
        results_root.parent / "temp" / f"{label}_ev_charging_{start.date()}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Sparad: {out}")
    print(f"  zoner: {', '.join(zones)}")
    print(f"  period: {start.date()} → {(end - pd.Timedelta(days=1)).date()}  ({mask.sum()} snapshots)")
    print(f"  EV-energi i perioden: fast {f_gw.sum()*dt_h:.0f} GWh, "
          f"flexibel {x_gw.sum()*dt_h:.0f} GWh, total {tot_gw.sum()*dt_h:.0f} GWh")


if __name__ == "__main__":
    main()
