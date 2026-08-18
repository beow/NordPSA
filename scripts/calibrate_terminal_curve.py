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
        # ⛔ `--no-hydro-price-proxy` STOD HÄR till 2026-08-17 och gjorde varje körning
        # omöjlig: flaggan togs bort 2026-08-15 (c3dd52e) när proxyn blev LÄGESBUNDEN,
        # så argparse svarade `unrecognized arguments` på allt. Behovet är oförändrat —
        # λ måste ligga på samma skala som marginalen — men det uppfylls nu av sig själv,
        # eftersom `--dispatch` fryser kapaciteterna och proxyn därmed alltid är AV.
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
    # ⚠️ EN OMKÖRNING vid fel. HiGHS/linopy-lagret kraschar sporadiskt med
    # `free(): invalid next size (fast)` EFTER att LP:t lösts optimalt — 5 av 37 i svep 1
    # och 2 av 49 i svep 2. Kraschar är inte bara bortfall: de biasar koordinatsökningen,
    # eftersom en gren kan väljas bort av slumpen i stället för av data.
    for forsok in (1, 2):
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            break
        print(f"    ↻ {tag}: föll (exit {r.returncode})"
              + (", kör om en gång" if forsok == 1 else " även vid omkörning"))
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
    # a_peak GLOBALT, inte per zon (2026-08-17). Motivet är kalibratorns eget: run268
    # mätte att `b_mean` är zonberoende — "det är den ENDA dimensionen där zonskillnaden
    # är MÄTT". Nivåns säsongsfas är hydrologi och delas därför av zonerna. Kostar 4
    # körningar i stället för 20, och den per-zon-varianten finns kvar bakom --phases.
    steps += [("globalt a_peak", "peak_a")]
    # VÅRFLODSSTEGET, nytt 2026-08-17. b_amp söktes inte alls tidigare, och eftersom
    # `load_params()` ger b_amp = 0 var `--phases`-sökningen över b_peak VERKNINGSLÖS —
    # den varierade en fas på en amplitud som var noll. (b_amp, b_peak) söks globalt:
    # spillrisk vid flodtoppen är hydrologi, inte zonens trängselläge.
    steps += [("globalt vårflod (b_amp, b_peak)", "flood")]
    # LÅGA HALVAN, nytt 2026-08-18. Utan den binder EN lutning ihop kurvans båda ändar och
    # eSetts krav på brant tilt betalas med orimliga λ vid tomt magasin (388 EUR/MWh vid
    # 10 % fyllnad). Driftintervallet är 7-83 %, så den låga änden BESÖKS varje senvinter.
    steps += [("globalt b_low_frac", "low")]
    # ASYMMETRISTEGET. Andra harmoniskan söks globalt av samma skäl som vårflodssteget:
    # skevheten är hydrologi. Hålls bakom en flagga så att den symmetriska formen kan
    # svepas för sig och de två formvarianterna jämföras rent.
    if args.asym:
        steps += [("globalt asymmetri (a_amp2, a_peak2)", "asym")]
    steps += [(f"per zon b_mean: {z}", ("zone", z)) for z in sorted(DEFAULTS)]
    if args.phases:
        steps += [(f"per zon a_peak: {z}", ("phase_a", z)) for z in sorted(DEFAULTS)]
        steps += [(f"per zon b_peak: {z}", ("phase_b", z)) for z in sorted(DEFAULTS)]
    return steps


def n_evals(args) -> int:
    n = len(args.a_amp) * len(args.b_mean)
    n += len(args.a_peak)                            # globalt a_peak
    n += len(args.b_amp) * len(args.b_peak)          # vårflodssteget
    n += len(args.b_low_frac)                        # låga halvan
    if args.asym:
        n += len(args.a_amp2) * len(args.a_peak2)    # asymmetristeget
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
            elif step == "peak_a":
                for ap in args.a_peak:
                    cands.append(({z: replace(best.get(z, CurveParams()), a_peak=ap)
                                   for z in best}, f"apeak{ap:g}"))
            elif step == "low":
                for lf in args.b_low_frac:
                    cands.append(({z: replace(best.get(z, CurveParams()), b_low_frac=lf)
                                   for z in best}, f"blow{lf:g}"))
            elif step == "asym":
                for a2, p2 in itertools.product(args.a_amp2, args.a_peak2):
                    cands.append(({z: replace(best.get(z, CurveParams()),
                                              a_amp2=a2, a_peak2=p2)
                                   for z in best}, f"aamp2{a2:g}_apeak2{p2:g}"))
            elif step == "flood":
                for ba, bp in itertools.product(args.b_amp, args.b_peak):
                    cands.append(({z: replace(best.get(z, CurveParams()),
                                              b_amp=ba, b_peak=bp)
                                   for z in best}, f"bamp{ba:g}_bpeak{bp:g}"))
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
    # 27 = den facit-kalibrerade kurvans värde. Med i rutnätet så att DAGENS lösning
    # måste försvara sig mot de hydrologiska priorerna 1/5/9 i stället för att uteslutas.
    ap.add_argument("--a-peak", type=float, nargs="+", default=[1.0, 5.0, 9.0, 27.0])
    ap.add_argument("--b-peak", type=float, nargs="+", default=[18.0, 22.0, 26.0])
    # 0,0 = dagens värde, alltså ingen säsong i brantheten — samma princip som a_peak 27.
    ap.add_argument("--b-amp", type=float, nargs="+", default=[0.0, 0.4, 0.8])
    # 1,0 = av, dvs en enda lutning — dagens form försvarar sig, samma princip som
    # a_peak 27 och b_amp 0.
    ap.add_argument("--b-low-frac", type=float, nargs="+", default=[1.0, 0.6, 0.4, 0.25])
    ap.add_argument("--asym", action="store_true",
                    help="Sök även andra harmoniskan i A(v) (a_amp2, a_peak2), dvs en "
                         "SKEV nivåkurva: snabb kollaps vid vårfloden, långsam "
                         "återuppbyggnad. Kräver |a_amp| + |a_amp2| < 1.")
    # 0,0 = av, alltså ren cosinus — den symmetriska formen försvarar sig.
    ap.add_argument("--a-amp2", type=float, nargs="+", default=[0.0, 0.15, 0.30])
    # Andra harmoniskan har PERIOD 26 v, så fyra faser täcker hela dess period.
    ap.add_argument("--a-peak2", type=float, nargs="+", default=[2.0, 8.0, 14.0, 20.0])
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
    # buffering=1 = radbuffrat. Utan det är filen BLOCKbuffrad och står tom till
    # processen avslutas, så räcket "en avbruten loop kan läsas av" höll inte i praktiken.
    with open(logp, "w", newline="", buffering=1) as fh:
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
