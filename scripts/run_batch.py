#!/usr/bin/env python
"""Batchkörning av 2040-expansionsscenarier med fast samtidighet.

Kör en uppsättning varianter av den kanoniska expansionen (`nordpsa expand`, världen
2040_svk_mm). Varje scenario är en experimentfil i config/experiments/ som bara bär
sin egen avvikelse; scenario 0 (baseline) har ingen.

Scenarionamnen är KONSTANTA mellan batchar — bara run-numrets prefix ändras, så
samma scenario har samma sista siffra i varje batch:

  run{PREFIX}0_baseline_{R}h     Baseline 2040-expansion
  run{PREFIX}1_senuc15exo_{R}h   + 1,5 GW exogen fast kärnkraft SE-S
  run{PREFIX}2_se_disc3_{R}h     + 3% diskontering SE kärnkraft + havsvind
  run{PREFIX}3_onshore80_{R}h    + onshore-tak −20%
  run{PREFIX}4_lowhydro06_{R}h   + torrt 2024 (hydro ×0,6)
  run{PREFIX}5_batt25_4h_{R}h    + 25 GW 4h batterier (ist. för 12 GW 2h)
  run{PREFIX}6_notax_{R}h        + ingen elskatt på värme-el (VP+el-panna)
  run{PREFIX}7_market50_{R}h     + ALLA kontinentkablar halverade (13 820 → 6 910 MW)

⚠️ Plats 7 BYTTE INNEHÅLL i batch 42. Till och med batch 36 var den `hansa`
(Hansa Power Bridge, SE-S↔DE 615→1315 MW, kvar i run198/run367). Konventionen att
scenarionamnen är konstanta mellan batchar gäller alltså inte över den gränsen: ett
`runXX7_hansa_*` är ≤ batch 36, ett `runXX7_market50_*` är ≥ batch 42.

Scenarier som redan har ett FÄRDIGT resultat (results/<namn>/network.nc) hoppas över;
--force kör om dem. Det skyddar bl.a. run260_baseline_2h, som redan ÄR batch 26:s
scenario 0 — `--prefix 26` kör alltså run261..run267 och återanvänder run260 som
batchens baseline. En mapp utan network.nc är en kraschad körning och körs om, så
en avbruten batch kan återupptas med samma kommando.

Med --dispatch-resolution H följs varje expansion av `nordpsa dispatch --from` i samma
arbetare: run{PREFIX}X_<namn>_dispatch_{H}h. Dispatchen ärver expansionens värld,
inklusive experimentet (t.ex. torråret), så inga flaggor behöver upprepas.

Samtidighet räknas i PIPELINES (expansion + ev. dispatch). Default 2: en 2h-expansion
≈ 4–5h / ~5–6 GB RAM, så 3 parallellt spräcker 15 GB. En 3h-expansion ≈ 1–1,5h /
~3,5 GB (då är 3 OK). Varje körnings stdout hamnar i results/<output>/run.log;
modellens egna filer (network.nc, highs.log, run_meta.txt) i samma mapp.

Användning:
  python scripts/run_batch.py --prefix 27                       # kanonisk baseline @2h
  python scripts/run_batch.py --prefix 27 --resolution 3 --concurrency 3 --label "..."
  python scripts/run_batch.py --prefix 27 --dispatch-resolution 1
  python scripts/run_batch.py --prefix 27 --only 0,5            # delmängd
  python scripts/run_batch.py --prefix 27 --dry-run             # visa kommandon, kör inget
  python scripts/run_batch.py --prefix 27 --set hydro.restrictions=false   # i alla körningar
"""
import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (idx, namn, beskrivning, experimentfil i config/experiments/ eller None)
# ⚠️ Plats 7 är `market50` sedan batch 42 (var `hansa` t.o.m. batch 36).
SCENARIOS = [
    (0, "baseline",   "Baseline 2040-expansion (SvK MM, DK-DE 4000)", None),
    (1, "senuc15exo", "+ 1,5 GW exogen fast karnkraft SE-S", "senuc15exo"),
    (2, "se_disc3",   "+ 3% diskontering SE karnkraft + havsvind", "se_disc3"),
    (3, "onshore80",  "+ onshore-tak -20%", "onshore80"),
    (4, "lowhydro06", "+ torrt 2024 (hydro x0.6)", "lowhydro06"),
    (5, "batt25_4h",  "+ 25 GW 4h batterier (ist for 12 GW 2h)", "batt25_4h"),
    (6, "notax",      "+ ingen elskatt pa varme-el (VP+el-panna)", "notax"),
    (7, "market50",   "+ ALLA kontinentkablar halverade (13 820 -> 6 910 MW)", "market50"),
]


def build_cmd(prefix, idx, name, desc, experiment, label, res, common):
    out = f"run{prefix}{idx}_{name}_{res}h"
    full_desc = f"[batch {prefix}] {desc}" + (f"; {label}" if label else "")
    cmd = [sys.executable, "-m", "nordpsa", "expand", "--resolution", str(res), *common]
    if experiment:
        cmd += ["--experiment", experiment]
    cmd += ["--output", out, "--desc", full_desc]
    return out, cmd


def build_dispatch_cmd(exp_out, prefix, idx, name, res_disp, label, common):
    """Kanonisk dispatch av expansionen: källans värld (inkl. experimentet) och frysta
    kapaciteter, rullande horisont 1+3 veckor, terminalkurvan."""
    out = f"run{prefix}{idx}_{name}_dispatch_{res_disp}h"
    desc = (f"[batch {prefix}] {res_disp}h dispatch av {exp_out} (frysta p_nom_opt)"
            + (f"; {label}" if label else ""))
    cmd = [sys.executable, "-m", "nordpsa", "dispatch", "--from", exp_out,
           "--resolution", str(res_disp), *common, "--output", out, "--desc", desc]
    return out, cmd


def _run(out, cmd):
    rdir = ROOT / "results" / out
    rdir.mkdir(parents=True, exist_ok=True)
    with open(rdir / "run.log", "w") as log:
        return subprocess.run(cmd, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT).returncode


def run_one(out, cmd, follow=None):
    """Kör expansionen; vid rc=0 körs ev. follow (label, cmd) i SAMMA arbetare, så att
    samtidigheten räknas i pipelines och inte i enskilda processer."""
    t0 = time.time()
    rc = _run(out, cmd)
    fout, frc = None, None
    if rc == 0 and follow is not None:
        fout, fcmd = follow
        frc = _run(fout, fcmd)
    return out, rc, time.time() - t0, fout, frc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True,
                    help="Run-nummerprefix, t.ex. 17 → run170..run175")
    ap.add_argument("--label", default="",
                    help="Batchetikett (vad som ändrats) → in i varje körnings --desc")
    ap.add_argument("--only", default="",
                    help="Kör bara dessa scenario-index, t.ex. '0,2,5' (default: alla)")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Antal samtidiga pipelines (default 2; 2h-expansion ≈ 5-6 GB styck)")
    ap.add_argument("--resolution", type=int, default=2,
                    help="Upplösning för expansionskörningarna (default 2)")
    ap.add_argument("--set", action="append", default=[], metavar="NYCKEL=VÄRDE",
                    help="nordpsa --set till ALLA körningar i batchen; kan upprepas")
    ap.add_argument("--dispatch-resolution", type=int, default=None, metavar="H",
                    help="Kör en dispatch av varje expansion på H h upplösning "
                         "(t.ex. 1). Utelämnad = ingen dispatch.")
    ap.add_argument("--force", action="store_true",
                    help="Kör om scenarier som redan har ett FÄRDIGT resultat (network.nc) "
                         "och skriv över det. Utan flaggan hoppas de över.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Visa kommandona och kör inget")
    args = ap.parse_args()

    sel = {int(x) for x in args.only.split(",") if x.strip() != ""}
    scen = [s for s in SCENARIOS if not sel or s[0] in sel]

    common = [tok for spec in args.set for tok in ("--set", spec)]

    jobs, skipped = [], []
    for idx, name, desc, extra in scen:
        out, cmd = build_cmd(args.prefix, idx, name, desc, extra,
                             args.label, args.resolution, common)
        # Ett FÄRDIGT resultat (network.nc finns) skrivs inte över av misstag. Skyddar
        # bl.a. run260_baseline_2h, som redan ÄR batch 26:s scenario 0. En mapp utan
        # network.nc är en kraschad/avbruten körning och körs om.
        if not args.force and (ROOT / "results" / out / "network.nc").exists():
            skipped.append(out)
            continue
        follow = None
        if args.dispatch_resolution:
            follow = build_dispatch_cmd(out, args.prefix, idx, name,
                                        args.dispatch_resolution, args.label, common)
        jobs.append((out, cmd, follow))

    if skipped:
        print(f"Hoppar över {len(skipped)} scenario(er) med färdigt resultat "
              f"(--force kör om dem):")
        for s in skipped:
            print(f"  • {s}")
        print()
    if not jobs:
        print("Inget att köra.")
        return

    print(f"Batch {args.prefix}: {len(jobs)} pipelines @ {args.resolution}h, "
          f"samtidighet {args.concurrency}"
          + (f" + {args.dispatch_resolution}h rullande dispatch (kanonisk mall)"
             if args.dispatch_resolution else "")
          + (f", etikett: {args.label}" if args.label else ""))
    for out, cmd, follow in jobs:
        print(f"\n  {out}")
        print("    " + " ".join(shlex.quote(c) for c in cmd))
        if follow:
            print(f"  → {follow[0]}")
            print("    " + " ".join(shlex.quote(c) for c in follow[1]))

    if args.dry_run:
        print("\n[dry-run] kör inget.")
        return

    print(f"\nStartar (loggar → results/<output>/run.log) ...\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, out, cmd, follow): out for out, cmd, follow in jobs}
        for fut in as_completed(futs):
            out, rc, dt, fout, frc = fut.result()
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            extra_txt = ""
            if fout is not None:
                extra_txt = f"  + dispatch {fout}: {'OK' if frc == 0 else f'FAIL (rc={frc})'}"
            elif rc == 0 and args.dispatch_resolution:
                extra_txt = "  + dispatch HOPPAD (expansionen misslyckades)"
            print(f"  [{status}] {out}  ({dt/60:.0f} min){extra_txt}")
            results.append((out, rc, dt, fout, frc))

    print("\n=== Batch klar ===")
    for out, rc, dt, fout, frc in sorted(results):
        line = f"  {'OK ' if rc == 0 else 'FAIL'}  {out}  ({dt/60:.0f} min)"
        if fout is not None:
            line += f"  | {'OK ' if frc == 0 else 'FAIL'} {fout}"
        print(line)
    nfail = sum(1 for _, rc, _, _, frc in results if rc != 0 or (frc not in (None, 0)))
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
