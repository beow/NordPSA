#!/usr/bin/env python
"""Kalibrera terminalvärdeskurvan mot de två icke-cirkulära observablerna.

Yttre loop runt den rullande dispatchen: sätt kurvparametrar → kör → poängsätt mot
eSetts fysiska hydrosäsong och EC:s magasinband → behåll det bästa.

    python scripts/calibrate_terminal_curve.py --dry-run          # visa planen + kostnad
    python scripts/calibrate_terminal_curve.py --source run260_baseline_2h

## Sökstrategin, och varför den är så snål

Full koordinatsökning över fem parametrar × fem zoner vore 75 körningar per varv.
I stället två steg:

  STEG 1  GLOBALT: alla zoner delar (a_amp, b_mean). Grovt rutnät, 3×3 = 9 körningar.
          Fångar nivån på säsong och lutning innan zonerna får skilja sig åt.
  STEG 2  PER ZON: bara b_mean, en zon i taget, 3 kandidater = 15 körningar.
          b_mean är den parameter run268 visade är ZONBEROENDE — brant profil gav
          de inlåsta zonerna bias −2,2/−2,9 medan de kontinentkopplade underskjöt
          15-22. Det är den enda dimensionen där zonskillnaden är MÄTT.

`a_peak` och `b_peak` hålls fasta på de hydrologiska priorerna (v5 respektive v22)
och kalibreras inte i grundplanen. ⚠️ ANTAGANDE: att faserna är rätt. De kan
frisläppas med --phases, till priset av 30 körningar till per varv.

## Räcken

- Vägrar starta om en annan run_model.py redan kör. Två samtidiga körningar tar
  ~9 GB var och har OOM-dödat WSL en gång (run256).
- Körningarna heter `tmp_curvecal_*` och förbrukar därför inga runXXX-nummer.
- Varje utvärdering loggas till kalibreringens CSV, så en avbruten loop kan läsas av.

⚠️ KRÄVER inkopplingen `--terminal-curve` i den rullande lösaren (grenen
`rolling-horizon-watervalue`). Utan den ignoreras kurvan tyst och alla varv får
samma poäng — loopen upptäcker det och avbryter.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nordpsa.wv import targets                       # noqa: E402
from nordpsa.wv.terminal_curve import (              # noqa: E402
    DEFAULTS, CurveParams, anchor_from_run, load_params, save_params)

WORKDIR = ROOT / "results"
PARAM_TMP = ROOT / "temp" / "terminal_curve_trial.yaml"


# ── Räcke: aldrig två körningar samtidigt ───────────────────────────────────────

def assert_no_run_in_flight() -> None:
    out = subprocess.run(["pgrep", "-af", "[r]un_model.py"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        sys.exit("✖ En modellkörning pågår redan:\n  "
                 + "\n  ".join(out.splitlines()[:3])
                 + "\n\nTvå samtidiga körningar tar ~9 GB var och har OOM-dödat WSL. "
                   "Vänta tills den är klar.")


# ── En utvärdering ──────────────────────────────────────────────────────────────

def evaluate(params, anchor, tag: str, args) -> tuple[float, dict]:
    """Kör dispatchen med de här parametrarna och returnera (poäng, mätvärden)."""
    save_params(params, anchor, str(PARAM_TMP), note=f"kalibreringsvarv {tag}")
    label = f"tmp_curvecal_{tag}"
    cmd = [
        sys.executable, "scripts/run_model.py",
        "--dispatch", args.source,
        "--resolution", str(args.resolution),
        "--rolling-horizon",
        "--rolling-weeks", str(args.rolling_weeks),
        "--no-hydro-price-proxy",          # λ måste ligga på samma skala som marginalen
        "--terminal-curve", str(PARAM_TMP),
        "--output", label,
        "--desc", f"kalibrering av terminalkurvan, varv {tag}",
    ]
    if args.rolling_lookahead_weeks:
        cmd += ["--rolling-lookahead-weeks", str(args.rolling_lookahead_weeks)]
    if args.year:
        cmd += ["--year", str(args.year)]

    t0 = time.time()
    log = WORKDIR / label / "console.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"    ✖ {tag}: körningen föll (exit {r.returncode}) — se {log}")
        return float("inf"), {}
    s = targets.score(label)
    print(f"    {tag:24s} poäng {s['poang']:.4f}   ({dt/60:.1f} min)  "
          + "  ".join(f"{k} v/s {v['vs']:.2f}" for k, v in s["per_land"].items()))
    return s["poang"], s


# ── Sökningen ───────────────────────────────────────────────────────────────────

def plan(args) -> list:
    """Returnerar sökplanen som en lista av (beskrivning, kandidatgenerator)."""
    steps = [("globalt (a_amp, b_mean)", "global")]
    steps += [(f"per zon b_mean: {z}", ("zone", z)) for z in sorted(DEFAULTS)]
    if args.phases:
        steps += [(f"per zon a_peak: {z}", ("phase_a", z)) for z in sorted(DEFAULTS)]
        steps += [(f"per zon b_peak: {z}", ("phase_b", z)) for z in sorted(DEFAULTS)]
    return steps


def n_evals(args) -> int:
    n = len(args.a_amp) * len(args.b_mean)
    n += len(DEFAULTS) * len(args.b_mean)
    if args.phases:
        n += len(DEFAULTS) * (len(args.a_peak) + len(args.b_peak))
    return n * args.rounds


def search(params, anchor, args, writer) -> tuple[dict, float]:
    best, best_score = dict(params), float("inf")
    k = 0
    for rnd in range(1, args.rounds + 1):
        for desc, step in plan(args):
            print(f"\n  [varv {rnd}] {desc}")
            cands = []
            if step == "global":
                for a, b in itertools.product(args.a_amp, args.b_mean):
                    cands.append(({z: replace(best.get(z, CurveParams()), a_amp=a, b_mean=b)
                                   for z in best}, f"a{a:g}_b{b:g}"))
            else:
                kind, zone = step
                field = {"zone": "b_mean", "phase_a": "a_peak", "phase_b": "b_peak"}[kind]
                grid = {"zone": args.b_mean, "phase_a": args.a_peak,
                        "phase_b": args.b_peak}[kind]
                for v in grid:
                    trial = dict(best)
                    trial[zone] = replace(best.get(zone, CurveParams()), **{field: v})
                    cands.append((trial, f"{zone}_{field}{v:g}"))

            for trial, tag in cands:
                k += 1
                sc, metrics = evaluate(trial, anchor, f"r{rnd}_{tag}", args)
                writer.writerow({"varv": rnd, "steg": desc, "tag": tag, "poang": sc,
                                 **{f"{lz}_vs": m["vs"] for lz, m in
                                    (metrics.get("per_land") or {}).items()}})
                if sc < best_score - 1e-9:
                    best, best_score = trial, sc
                    print(f"      ⭐ nytt bästa: {best_score:.4f}")
            if k >= 2 and best_score == float("inf"):
                sys.exit("✖ Alla körningar föll — avbryter.")
    return best, best_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="run260_baseline_2h",
                    help="Expansionskörning vars kapaciteter fryses (--dispatch)")
    ap.add_argument("--resolution", type=int, default=3)
    ap.add_argument("--year", type=int, default=None,
                    help="Enskilt år. Utelämna för alla tre — säkrare, ett enskilt år "
                         "kan vara vått (2024 +12 %%) och snedvrida kalibreringen.")
    ap.add_argument("--rolling-weeks", type=int, default=4)
    ap.add_argument("--rolling-lookahead-weeks", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--phases", action="store_true",
                    help="Släpp även a_peak/b_peak fria (dyrt)")
    ap.add_argument("--a-amp", type=float, nargs="+", default=[0.15, 0.35, 0.55])
    ap.add_argument("--b-mean", type=float, nargs="+", default=[1.0, 2.5, 4.0])
    ap.add_argument("--a-peak", type=float, nargs="+", default=[1.0, 5.0, 9.0])
    ap.add_argument("--b-peak", type=float, nargs="+", default=[18.0, 22.0, 26.0])
    ap.add_argument("--anchor-from", default=None, metavar="RUN",
                    help="Mät λ_bas ur den här körningen (default: DEFAULT_ANCHOR)")
    ap.add_argument("--out", default=None, help="Var de bästa parametrarna skrivs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    params, anchor = load_params()
    if args.anchor_from:
        anchor = anchor_from_run(args.anchor_from)

    total = n_evals(args)
    print(f"Plan: {total} körningar ({args.rounds} varv), "
          f"{args.resolution}h, {'år ' + str(args.year) if args.year else 'alla år'}, "
          f"fönster {args.rolling_weeks} v"
          + (f" +{args.rolling_lookahead_weeks} v look-ahead"
             if args.rolling_lookahead_weeks else ""))
    print(f"Källa (frysta kapaciteter): {args.source}")
    print("λ_bas: " + ", ".join(f"{z} {v:.1f}" for z, v in sorted(anchor.items())))
    print("Steg: " + " → ".join(d for d, _ in plan(args)))
    if args.dry_run:
        print("\n--dry-run: inget kört. Mät en enskild körnings tid först och "
              "multiplicera med antalet ovan innan du startar.")
        return

    assert_no_run_in_flight()

    logp = ROOT / "temp" / "terminal_curve_calibration.csv"
    logp.parent.mkdir(parents=True, exist_ok=True)   # temp/ finns inte i en färsk worktree
    fields = ["varv", "steg", "tag", "poang"] + [f"{lz}_vs" for lz in targets.LAND]
    with open(logp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        best, best_score = search(params, anchor, args, writer)

    out = save_params(best, anchor, args.out,
                      note=f"kalibrerad mot eSett v/s + EC-band, poäng {best_score:.4f}, "
                           f"källa {args.source}")
    print(f"\n⭐ Bästa poäng {best_score:.4f} → {out}")
    print(f"   utvärderingslogg: {logp}")
    for z, c in sorted(best.items()):
        print(f"   {z:6s} a_amp={c.a_amp:.2f} a_peak=v{c.a_peak:.0f} "
              f"b_mean={c.b_mean:.2f} b_amp={c.b_amp:.2f} b_peak=v{c.b_peak:.0f}")


if __name__ == "__main__":
    main()
