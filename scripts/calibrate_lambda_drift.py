#!/usr/bin/env python
"""Sök λ_bas med MAGASINDRIFTEN som mätare.

En rullande dispatch har inget cykliskt villkor, så lagret får driva fritt över
perioden. Driften är därför en direkt avläsning av om vattnet är rätt prissatt:

    drift > 0   för DYRT   → hamstring
    drift < 0   för BILLIGT → lagret säljs ut
    drift ≈ 0   nivån passar världen

Sökningen skalar HELA ankaret med en skalär k (λ_bas[z] × k) och bisekterar på
totaldriften. En skalär räcker som första ansats eftersom driften är monotont
växande i k — mätt över tre körningar: run318 (dagens nivå, dagens värld) −0,3 ·
run317 (2040-nivå, dagens värld, ~3× för högt) +20,0.

    python scripts/calibrate_lambda_drift.py --dispatch run260_baseline_2h --dry-run
    python scripts/calibrate_lambda_drift.py --dispatch run260_baseline_2h

⚠️ Noll drift är ett NÖDVÄNDIGT villkor, inte ett tillräckligt. Formen kan vara fel
även när nivån är rätt — run319 dumpade vatten i maj OCH tömde lagret. Skriptet
rapporterar därför v/s och poäng vid varje steg så att formfelet syns medan
nivån söks.

⚠️ ÅTERKOPPLING: höjt λ får hydron att hålla igen, vilket HÖJER priserna, vilket
minskar effekten av höjningen. Bisektionen hanterar det (den kräver bara monotoni,
inte linjäritet) men konvergensen kan bli långsammare än en ren halvering antyder.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nordpsa.wv import targets                                    # noqa: E402
from nordpsa.wv.terminal_curve import load_params, save_params    # noqa: E402

TRIAL_YAML = ROOT / "temp" / "terminal_curve_lambda_trial.yaml"


def assert_no_run_in_flight() -> None:
    out = subprocess.run(["pgrep", "-af", "[r]un_model.py"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        sys.exit("✖ En modellkörning pågår redan — två samtidiga tar ~9 GB var.\n  "
                 + "\n  ".join(out.splitlines()[:2]))


def evaluate(k: float, params, anchor0, args) -> tuple[float, dict]:
    """Kör dispatchen med λ_bas × k och returnera (totaldrift TWh, mätvärden)."""
    anchor = {z: v * k for z, v in anchor0.items()}
    TRIAL_YAML.parent.mkdir(parents=True, exist_ok=True)
    save_params(params, anchor, str(TRIAL_YAML),
                note=f"λ-sökning: ankaret × {k:.4f}")
    label = f"tmp_lamsearch_k{k:.3f}".replace(".", "p")
    cmd = [sys.executable, "scripts/run_model.py",
           "--dispatch", args.dispatch, "--resolution", str(args.resolution),
           "--rolling-horizon", "--rolling-weeks", str(args.rolling_weeks),
           "--rolling-lookahead-weeks", str(args.rolling_lookahead_weeks),
           "--terminal-curve", str(TRIAL_YAML), "--no-hydro-price-proxy",
           "--output", label, "--desc", f"lambda-sokning, ankaret x {k:.4f}"]
    (ROOT / "results" / label).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(ROOT / "results" / label / "console.log", "w") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    ✖ k={k:.3f}: körningen föll (exit {r.returncode})")
        return float("nan"), {}
    drift = targets.reservoir_drift(label)
    sc = targets.score(label)
    dt = (time.time() - t0) / 60
    print(f"    k={k:.3f}  λ_bas {min(anchor.values()):.0f}-{max(anchor.values()):.0f}"
          f"   DRIFT {drift['SUMMA']:+6.1f} TWh   poäng {sc['poang']:.4f}   "
          + " ".join(f"{lz} v/s {m['vs']:.2f}" for lz, m in sc["per_land"].items())
          + f"   ({dt:.0f} min)")
    return drift["SUMMA"], {"drift": drift, "score": sc, "label": label}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dispatch", default="run260_baseline_2h")
    ap.add_argument("--resolution", type=int, default=2)
    ap.add_argument("--rolling-weeks", type=int, default=1)
    ap.add_argument("--rolling-lookahead-weeks", type=int, default=3)
    ap.add_argument("--params", default=None, help="Kurvparametrar (default config/terminal_curve.yaml)")
    ap.add_argument("--k-lo", type=float, default=1.0, help="Startgräns med NEGATIV drift")
    ap.add_argument("--k-hi", type=float, default=None, help="Startgräns med POSITIV drift (söks annars)")
    ap.add_argument("--tol", type=float, default=1.0, metavar="TWH",
                    help="Acceptabel |drift| över hela perioden (default 1.0 TWh)")
    ap.add_argument("--max-evals", type=int, default=7)
    ap.add_argument("--out", default="config/terminal_curve_2040.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    params, anchor0 = load_params(args.params)
    print(f"Källa (frysta kapaciteter): {args.dispatch}, {args.resolution}h, "
          f"fönster {args.rolling_weeks}v +{args.rolling_lookahead_weeks}v look-ahead")
    print("λ_bas (k=1): " + ", ".join(f"{z} {v:.1f}" for z, v in sorted(anchor0.items())))
    print(f"Mål: |totaldrift| < {args.tol} TWh. Högst {args.max_evals} körningar "
          f"à ~11 min = ~{args.max_evals*11} min.")
    if args.dry_run:
        print("\n--dry-run: inget kört.")
        return

    assert_no_run_in_flight()
    log = ROOT / "temp" / "lambda_drift_search.csv"
    log.parent.mkdir(parents=True, exist_ok=True)
    hist = []

    with open(log, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["k", "drift_twh", "poang", "SE_vs", "NO_vs", "FI_vs", "label"])

        def rec(k, d, meta):
            hist.append((k, d))
            s = meta.get("score", {}).get("per_land", {})
            wr.writerow([f"{k:.4f}", f"{d:.2f}",
                         f"{meta.get('score', {}).get('poang', float('nan')):.4f}"]
                        + [f"{s.get(lz, {}).get('vs', float('nan')):.3f}" for lz in ("SE", "NO", "FI")]
                        + [meta.get("label", "")])
            fh.flush()

        # 1. Nedre gräns
        print(f"\n[1] nedre gräns k={args.k_lo}")
        lo_d, meta = evaluate(args.k_lo, params, anchor0, args); rec(args.k_lo, lo_d, meta)
        if lo_d != lo_d:
            sys.exit("✖ Nedre gränsen föll.")
        if lo_d > 0:
            sys.exit(f"✖ k={args.k_lo} ger redan POSITIV drift ({lo_d:+.1f}). "
                     "Sänk --k-lo; sökningen förutsätter negativ drift nedtill.")
        lo, hi = args.k_lo, args.k_hi

        # 2. Expandera uppåt tills driften vänder
        if hi is None:
            hi = args.k_lo
            while len(hist) < args.max_evals:
                hi = hi * 1.5
                print(f"\n[{len(hist)+1}] söker övre gräns k={hi:.3f}")
                d, meta = evaluate(hi, params, anchor0, args); rec(hi, d, meta)
                if d != d:
                    sys.exit("✖ Körning föll under gränssökningen.")
                if d > 0:
                    break
                lo = hi
            else:
                sys.exit(f"✖ Hittade ingen övre gräns inom {args.max_evals} körningar.")

        # 3. Bisektion
        while len(hist) < args.max_evals:
            mid = 0.5 * (lo + hi)
            print(f"\n[{len(hist)+1}] bisektion k={mid:.3f}  (intervall {lo:.3f}–{hi:.3f})")
            d, meta = evaluate(mid, params, anchor0, args); rec(mid, d, meta)
            if d != d:
                sys.exit("✖ Körning föll under bisektionen.")
            if abs(d) < args.tol:
                print(f"\n⭐ KLAR: k={mid:.4f} ger drift {d:+.1f} TWh (< {args.tol})")
                out = save_params(params, {z: v * mid for z, v in anchor0.items()}, args.out,
                                  note=f"2040: λ_bas kalibrerat på magasindrift, k={mid:.4f}, "
                                       f"drift {d:+.1f} TWh över perioden. ⚠️ Formen är EJ kalibrerad.")
                print(f"   → {out}")
                break
            lo, hi = (mid, hi) if d < 0 else (lo, mid)
        else:
            print(f"\n⚠️ Avbröt efter {args.max_evals} körningar utan att nå toleransen. "
                  f"Bästa intervall: {lo:.3f}–{hi:.3f}")

    print(f"\nlogg: {log}")
    print(f"{'k':>8s}{'drift':>9s}")
    for k, d in hist:
        print(f"{k:8.3f}{d:+9.1f}")


if __name__ == "__main__":
    main()
