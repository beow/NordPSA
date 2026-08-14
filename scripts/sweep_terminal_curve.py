#!/usr/bin/env python
"""Svep EN kurvparameter och tabellera v/s, poäng och drift.

    python scripts/sweep_terminal_curve.py --field a_peak --values 36 40 44 48
    python scripts/sweep_terminal_curve.py --field a_amp  --values 0.15 0.55

Parametern sätts på ALLA zoner samtidigt; syftet är att hitta rätt storleksordning
och riktning, inte att finkalibrera per zon.

## Varför screening vid fast nivå är försvarbart

k-svepet (calibrate_lambda_drift) mätte att v/s knappt rör sig med nivån: över ett
driftspann på 29 TWh flyttade sig SE:s v/s bara 0,89 → 1,01. Formens effekt på
säsongen går alltså att ranka utan att driften nollställs varje varv. Driften
rapporteras ändå, eftersom den behövs för slutkalibreringen.

⚠️ MEN nivå och form är INTE separerbara i modellsvaret, bara i kurvans algebra:
b_amp 0,6 → 0 slog drift −0,8 → −11,0 utan att röra λ_bas. Vinnaren måste därför
alltid nivåkalibreras om innan den bedöms slutgiltigt.

⚠️ FASFÖRSKJUTNING: terminalvärdet sätts i den LÖSTA horisontens slut, alltså
`--rolling-lookahead-weeks` veckor efter den vecka som behålls. Med look-ahead 3
verkar A(w) tre veckor "för tidigt" — en önskad effektiv topp vecka W kräver
a_peak ≈ W + 3.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nordpsa.wv import targets                                    # noqa: E402
from nordpsa.wv.terminal_curve import load_params, save_params    # noqa: E402


def assert_no_run_in_flight() -> None:
    out = subprocess.run(["pgrep", "-af", "[r]un_model.py"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        sys.exit("✖ En modellkörning pågår redan.\n  " + out.splitlines()[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field", required=True,
                    choices=["a_amp", "a_peak", "b_mean", "b_amp", "b_peak"])
    ap.add_argument("--values", nargs="+", type=float, required=True)
    ap.add_argument("--params", default="config/terminal_curve_2040.yaml")
    ap.add_argument("--dispatch", default="run260_baseline_2h")
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--rolling-weeks", type=int, default=1)
    ap.add_argument("--rolling-lookahead-weeks", type=int, default=3)
    ap.add_argument("--tag", default=None, help="Prefix för körningsnamnen")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    params, anchor = load_params(args.params)
    tag = args.tag or args.field
    print(f"Svep {args.field} = {args.values} (alla zoner), bas {args.params}")
    print("λ_bas: " + ", ".join(f"{z} {v:.1f}" for z, v in sorted(anchor.items())))
    print(f"{len(args.values)} körningar à ~11 min = ~{len(args.values)*11} min\n")
    if args.dry_run:
        return
    assert_no_run_in_flight()

    log = ROOT / "temp" / f"sweep_{tag}.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([args.field, "poang", "drift_twh", "SE_vs", "NO_vs", "FI_vs", "label"])
        print(f"{args.field:>8s}{'poäng':>9s}{'drift':>8s}{'SE v/s':>8s}{'NO v/s':>8s}"
              f"{'FI v/s':>8s}   (mål 1,38 / 1,30 / 1,13)")
        for v in args.values:
            trial = {z: replace(p, **{args.field: v}) for z, p in params.items()}
            f = ROOT / "temp" / f"curve_{tag}_{v:g}.yaml".replace(".", "p", 1)
            save_params(trial, anchor, str(f), note=f"svep {args.field}={v:g}")
            label = f"tmp_sweep_{tag}_{v:g}".replace(".", "p")
            (ROOT / "results" / label).mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "scripts/run_model.py",
                   "--dispatch", args.dispatch, "--resolution", str(args.resolution),
                   "--rolling-horizon", "--rolling-weeks", str(args.rolling_weeks),
                   "--rolling-lookahead-weeks", str(args.rolling_lookahead_weeks),
                   "--terminal-curve", str(f), "--no-hydro-price-proxy",
                   "--output", label, "--desc", f"svep {args.field}={v:g}"]
            t0 = time.time()
            with open(ROOT / "results" / label / "console.log", "w") as out:
                r = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"{v:8g}   ✖ föll (exit {r.returncode})")
                continue
            s = targets.score(label)
            d = targets.reservoir_drift(label)["SUMMA"]
            pl = s["per_land"]
            print(f"{v:8g}{s['poang']:9.4f}{d:+8.1f}"
                  + "".join(f"{pl[l]['vs']:8.2f}" for l in ("SE", "NO", "FI"))
                  + f"   ({(time.time()-t0)/60:.0f} min)")
            wr.writerow([f"{v:g}", f"{s['poang']:.4f}", f"{d:.2f}"]
                        + [f"{pl[l]['vs']:.3f}" for l in ("SE", "NO", "FI")] + [label])
            fh.flush()
    print(f"\nlogg: {log}")


if __name__ == "__main__":
    main()
