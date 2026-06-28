#!/usr/bin/env python
"""Batchkörning av 2040-expansionsscenarier (3h) med fast samtidighet.

Kör en uppsättning varianter av SvK 2040 MM-expansionsbaselinen. Scenarionamnen är
KONSTANTA mellan batchar — bara run-numrets prefix ändras, så samma scenario har
samma sista siffra i varje batch:

  run{PREFIX}0_baseline_3h     Baseline 2040-expansion
  run{PREFIX}1_senuc15exo_3h   + 1,5 GW exogen fast kärnkraft SE-S
  run{PREFIX}2_se_disc3_3h     + 3% diskontering SE kärnkraft + havsvind
  run{PREFIX}3_onshore80_3h    + onshore-tak −20%
  run{PREFIX}4_lowhydro06_3h   + torrt 2024 (hydro ×0,6)
  run{PREFIX}5_batt25_4h_3h    + 25 GW 4h batterier (ist. för 12 GW 2h)
  run{PREFIX}6_notax_3h        + ingen elskatt på värme-el (VP+el-panna)

Kör 3 samtidigt (default). En 3h-expansion ≈ 1–1,5h / ~3,5 GB RAM → en full batch
om 6 körningar ≈ 2 vågor ≈ 3–3,5h. Varje körnings stdout hamnar i
results/<output>/run.log; modellens egna filer (network.nc, highs.log, run_meta.txt)
i samma mapp.

Användning:
  python scripts/run_batch.py --prefix 17 --label "EV morgongolv 0.7 + 8.48 GWh/TWh"
  python scripts/run_batch.py --prefix 17 --only 0,5            # delmängd
  python scripts/run_batch.py --prefix 17 --dry-run             # visa kommandon, kör inget
"""
import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Gemensam baseline (run164-receptet) — delas av alla scenarier.
BASE = [
    "--resolution", "3", "--spill-cost", "50", "--market-elasticity",
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
]


def build_cmd(prefix, idx, name, desc, extra, label):
    out = f"run{prefix}{idx}_{name}_3h"
    full_desc = f"[batch {prefix}] {desc}" + (f"; {label}" if label else "")
    cmd = [sys.executable, "scripts/run_model.py", *BASE, *extra,
           "--output", out, "--desc", full_desc]
    return out, cmd


def run_one(out, cmd):
    rdir = ROOT / "results" / out
    rdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(rdir / "run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    return out, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True,
                    help="Run-nummerprefix, t.ex. 17 → run170..run175")
    ap.add_argument("--label", default="",
                    help="Batchetikett (vad som ändrats) → in i varje körnings --desc")
    ap.add_argument("--only", default="",
                    help="Kör bara dessa scenario-index, t.ex. '0,2,5' (default: alla)")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Antal samtidiga körningar (default 3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Visa kommandona och kör inget")
    args = ap.parse_args()

    sel = {int(x) for x in args.only.split(",") if x.strip() != ""}
    scen = [s for s in SCENARIOS if not sel or s[0] in sel]

    jobs = []
    for idx, name, desc, extra in scen:
        out, cmd = build_cmd(args.prefix, idx, name, desc.replace("%%", "%"), extra, args.label)
        jobs.append((out, cmd))

    print(f"Batch {args.prefix}: {len(jobs)} körningar, samtidighet {args.concurrency}"
          + (f", etikett: {args.label}" if args.label else ""))
    for out, cmd in jobs:
        print(f"\n  {out}")
        print("    " + " ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        print("\n[dry-run] kör inget.")
        return

    print(f"\nStartar (loggar → results/<output>/run.log) ...\n")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, out, cmd): out for out, cmd in jobs}
        for fut in as_completed(futs):
            out, rc, dt = fut.result()
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"  [{status}] {out}  ({dt/60:.0f} min)")
            results.append((out, rc, dt))

    print("\n=== Batch klar ===")
    for out, rc, dt in sorted(results):
        print(f"  {'OK ' if rc == 0 else 'FAIL'}  {out}  ({dt/60:.0f} min)")
    nfail = sum(1 for _, rc, _ in results if rc != 0)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
