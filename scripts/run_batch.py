#!/usr/bin/env python
"""Batchkörning av 2040-expansionsscenarier (3h) med fast samtidighet.

Kör en uppsättning varianter av SvK 2040 MM-expansionsbaselinen. Scenarionamnen är
KONSTANTA mellan batchar — bara run-numrets prefix ändras, så samma scenario har
samma sista siffra i varje batch:

  run{PREFIX}0_baseline_{R}h     Baseline 2040-expansion
  run{PREFIX}1_senuc15exo_{R}h   + 1,5 GW exogen fast kärnkraft SE-S
  run{PREFIX}2_se_disc3_{R}h     + 3% diskontering SE kärnkraft + havsvind
  run{PREFIX}3_onshore80_{R}h    + onshore-tak −20%
  run{PREFIX}4_lowhydro06_{R}h   + torrt 2024 (hydro ×0,6)
  run{PREFIX}5_batt25_4h_{R}h    + 25 GW 4h batterier (ist. för 12 GW 2h)
  run{PREFIX}6_notax_{R}h        + ingen elskatt på värme-el (VP+el-panna)
  run{PREFIX}7_hansa_{R}h        + Hansa Power Bridge: SE-S↔DE 615→1315 MW

Med --dispatch-resolution H följs varje expansion av en pinnad omdispatch i samma
arbetare: run{PREFIX}X_<namn>_dispatch_{H}h (kapaciteter frysta till expansionens
p_nom_opt, hydro-SOC pinnad mot dess lagerbana per --dispatch-pin-freq).

Samtidighet räknas i PIPELINES (expansion + ev. dispatch). Default 2: en 2h-expansion
≈ 4–5h / ~5–6 GB RAM, så 3 parallellt spräcker 15 GB. En 3h-expansion ≈ 1–1,5h /
~3,5 GB (då är 3 OK). Varje körnings stdout hamnar i results/<output>/run.log;
modellens egna filer (network.nc, highs.log, run_meta.txt) i samma mapp.

Användning:
  python scripts/run_batch.py --prefix 17 --resolution 3 --concurrency 3 --label "..."
  python scripts/run_batch.py --prefix 24 --nuclear-min-load 0.6 --dispatch-resolution 1
  python scripts/run_batch.py --prefix 24 --only 0,5            # delmängd
  python scripts/run_batch.py --prefix 24 --dry-run             # visa kommandon, kör inget
"""
import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Gemensam baseline (run164-receptet) — delas av alla scenarier. Upplösningen sätts
# per batch via --resolution (default 2 fr.o.m. batch 24).
BASE = [
    "--spill-cost", "50", "--market-elasticity",
    "--cost-scenario", "svk_2040", "--demand-scenario", "svk_2040_mm", "--add-heat",
    "--onwind-capfac-increase", "0.30", "--offwind-capfac-increase", "0.10",
]
# Endogen ny kärnkraft (baseline + de flesta varianter).
NUC_BASE = ["--add-nuclear", "SE-S:10:201", "SE-N:10:202", "FI:10:203"]

# (idx, namn, beskrivning, scenario-specifika args inkl. kärnkraftsblock)
SCENARIOS = [
    (0, "baseline", "Baseline 2040-expansion (SvK MM, DK-DE 4000)",
        NUC_BASE),
    (1, "senuc15exo", "+ 1,5 GW exogen fast karnkraft SE-S",
        ["--add-nuclear", "SE-N:10:202", "FI:10:203", "--add-nuclear-fixed", "SE-S:3:500"]),
    (2, "se_disc3", "+ 3%% diskontering SE karnkraft + havsvind",
        NUC_BASE + ["--nuclear-discount-rate", "SE-N:0.03", "SE-S:0.03",
                    "--offwind-discount-rate", "SE-N:0.03", "SE-S:0.03"]),
    (3, "onshore80", "+ onshore-tak -20%%",
        NUC_BASE + ["--onshore-lower", "0.8"]),
    (4, "lowhydro06", "+ torrt 2024 (hydro x0.6)",
        NUC_BASE + ["--low-hydro", "0.6"]),
    (5, "batt25_4h", "+ 25 GW 4h batterier (ist for 12 GW 2h)",
        NUC_BASE + ["--scenario-battery", "25:4"]),
    (6, "notax", "+ ingen elskatt pa varme-el (VP+el-panna)",
        NUC_BASE + ["--no-tax-heatpower"]),
    (7, "hansa", "+ Hansa Power Bridge: SE-S<->DE 615->1315 MW",
        NUC_BASE + ["--market-ntc-override", "SE-S DE:1315"]),
]


def build_cmd(prefix, idx, name, desc, extra, label, res, common):
    out = f"run{prefix}{idx}_{name}_{res}h"
    full_desc = f"[batch {prefix}] {desc}" + (f"; {label}" if label else "")
    cmd = [sys.executable, "scripts/run_model.py", "--resolution", str(res),
           *BASE, *common, *extra, "--output", out, "--desc", full_desc]
    return out, cmd


def build_dispatch_cmd(exp_out, prefix, idx, name, res_disp, pin_freq, label):
    """1h (res_disp) pinnad omdispatch av expansionskörningen: kapaciteter frysta till
    dess p_nom_opt och hydro-SOC pinnad mot dess lagerbana i fönster om pin_freq."""
    out = f"run{prefix}{idx}_{name}_dispatch_{res_disp}h"
    desc = (f"[batch {prefix}] {res_disp}h omdispatch av {exp_out} (frysta p_nom_opt) "
            f"med hydro-SOC pinnad mot dess lagerbana, fonster {pin_freq}"
            + (f"; {label}" if label else ""))
    cmd = [sys.executable, "scripts/run_model.py",
           "--dispatch", exp_out, "--resolution", str(res_disp),
           "--soc-pin-from", exp_out, "--soc-pin-freq", pin_freq,
           "--output", out, "--desc", desc]
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
    ap.add_argument("--nuclear-min-load", type=float, default=None, metavar="FRAC",
                    help="Lastföljande NY kärnkraft, p_min_pu = FRAC × p_max_pu (t.ex. 0.6)")
    ap.add_argument("--dispatch-resolution", type=int, default=None, metavar="H",
                    help="Kör en pinnad omdispatch av varje expansion på H h upplösning "
                         "(t.ex. 1). Utelämnad = ingen dispatch.")
    ap.add_argument("--dispatch-pin-freq", default="MS", metavar="FREQ",
                    help="Fönsterlängd för dispatchens SOC-pin (default MS = månad)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Visa kommandona och kör inget")
    args = ap.parse_args()

    sel = {int(x) for x in args.only.split(",") if x.strip() != ""}
    scen = [s for s in SCENARIOS if not sel or s[0] in sel]

    common = []
    if args.nuclear_min_load is not None:
        common += ["--nuclear-min-load", str(args.nuclear_min_load)]

    jobs = []
    for idx, name, desc, extra in scen:
        out, cmd = build_cmd(args.prefix, idx, name, desc.replace("%%", "%"), extra,
                             args.label, args.resolution, common)
        follow = None
        if args.dispatch_resolution:
            follow = build_dispatch_cmd(out, args.prefix, idx, name,
                                        args.dispatch_resolution, args.dispatch_pin_freq,
                                        args.label)
        jobs.append((out, cmd, follow))

    print(f"Batch {args.prefix}: {len(jobs)} pipelines @ {args.resolution}h, "
          f"samtidighet {args.concurrency}"
          + (f" + {args.dispatch_resolution}h pinnad dispatch ({args.dispatch_pin_freq})"
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
