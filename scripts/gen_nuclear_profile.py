#!/usr/bin/env python
"""Fristående syntetisk kärnkrafts-tillgänglighetsgenerator (ingen solve).

Egen implementation — rör INTE nordpsa/nuclear_availability.py. Skillnad mot
modellens generator: forcerade avbrott dras ur en SKIFTAD exponentialfördelning
med ett varaktighetsGOLV (min_outage_hours), vilket tar bort sub-dygns-spikarna
("taggigheten") utan att ändra CF — medelvaraktigheten är fortfarande mttr_hours,
så lambda-kalibreringen är identisk med modellens.

    varaktighet = min_outage + Exp(medel = mttr - min_outage)
      → min = min_outage,  medel = mttr  (oförändrad → CF hålls)

Exempel:
  python scripts/gen_nuclear_profile.py --preset SE-S --seed 111 --mttr 250 --min-outage 48 --plot
  python scripts/gen_nuclear_profile.py --n-reactors 5 --target-cf 0.86 --seed 113 \
         --mttr 300 --min-outage 72 --plot --output temp/fi_profile.parquet
"""
import argparse
import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760.0

# Presetflottor (reaktor-MW-listor speglar run191)
PRESETS = {
    "SE-S": {"reactor_mw": [6900 / 6] * 6 + [500.0] * 3, "target_cf": 0.85},
    "FI":   {"reactor_mw": [4400 / 5] * 5,                "target_cf": 0.85},
}


def calibrate_lambda(target_cf, d_mean_days, mttr_hours, biennial):
    """Forcerad felrat (händelser/reaktor/år) så årsmedel-CF ≈ target_cf.
    Identisk formel som modellen (alternerande förnyelseprocess)."""
    U      = 1.0 - target_cf
    p_plan = d_mean_days / 365.25 * (0.5 if biennial else 1.0)
    U_f    = float(np.clip((U - p_plan) / (1.0 - p_plan), 0.0, 0.95))
    if U_f <= 0.0:
        return 0.0
    return U_f / (1.0 - U_f) * HOURS_PER_YEAR / mttr_hours


def _apply(up, start_h, dur_h, dt_h):
    T = len(up)
    i0 = max(int(np.floor(start_h / dt_h)), 0)
    i1 = min(int(np.ceil((start_h + dur_h) / dt_h)), T)
    if i1 > i0:
        up[i0:i1] = 0.0


def _schedule_maintenance(up, rng, reactor_idx, n_reactors, ts, dt_h,
                          window_doy, days_range, biennial):
    """Staggrad sommarrevision/reaktor/år (vartannat om biennial)."""
    doy_lo, doy_hi = float(window_doy[0]), float(window_doy[1])
    d_lo, d_hi     = float(days_range[0]), float(days_range[1])
    t0    = ts[0]
    years = sorted({t.year for t in ts})
    for k, year in enumerate(years):
        if biennial and ((reactor_idx + k) % 2 == 1):
            continue
        d_days    = rng.uniform(d_lo, d_hi)
        slot      = (doy_hi - doy_lo) / n_reactors
        slot_lo   = doy_lo + reactor_idx * slot
        latest    = max(slot_lo, slot_lo + slot - d_days)
        start_doy = rng.uniform(slot_lo, latest) if latest > slot_lo else slot_lo
        year_start = pd.Timestamp(year=year, month=1, day=1, tz=t0.tz)
        start_h = (year_start - t0).total_seconds() / 3600.0 + (start_doy - 1.0) * 24.0
        _apply(up, start_h, d_days * 24.0, dt_h)


def _forced_outages(up, rng, lam, mttr_hours, min_outage_hours, total_hours, dt_h):
    """Poisson-process; SKIFTAD exp-varaktighet med golv min_outage_hours.
    Medelvaraktighet = mttr_hours (golv + (mttr-golv)) → CF oförändrad."""
    if lam <= 0.0:
        return
    scale = max(mttr_hours - min_outage_hours, 1e-6)   # exp-medel efter skift
    n_events = rng.poisson(lam * total_hours / HOURS_PER_YEAR)
    for _ in range(int(n_events)):
        start_h = rng.uniform(0.0, total_hours)
        dur_h   = min_outage_hours + rng.exponential(scale)
        _apply(up, start_h, dur_h, dt_h)


def generate(reactor_mw, snapshots, seed, target_cf, mttr_hours,
             min_outage_hours, biennial=False,
             maintenance_window_doy=(105, 273), maintenance_days=(25, 45)):
    mw = np.asarray(reactor_mw, dtype=float)
    n  = len(mw)
    rng = np.random.default_rng(int(seed))
    dt_h        = (snapshots[1] - snapshots[0]).total_seconds() / 3600.0
    T           = len(snapshots)
    total_hours = T * dt_h
    d_mean = 0.5 * (maintenance_days[0] + maintenance_days[1])
    lam    = calibrate_lambda(target_cf, d_mean, mttr_hours, biennial)
    avail_mw = np.zeros(T)
    for r in range(n):
        up = np.ones(T)
        _schedule_maintenance(up, rng, r, n, snapshots, dt_h,
                              maintenance_window_doy, maintenance_days, biennial)
        _forced_outages(up, rng, lam, mttr_hours, min_outage_hours, total_hours, dt_h)
        avail_mw += mw[r] * up
    return pd.Series(avail_mw / mw.sum(), index=snapshots, name="nuclear_avail"), lam


def _count_dips(daily, thresh=0.02):
    """Antal lokala dippar i dagsserien (grovt taggighetsmått)."""
    v = daily.values
    return int(np.sum((v[1:-1] < v[:-2] - thresh) & (v[1:-1] < v[2:] - thresh)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", choices=sorted(PRESETS), help="SE-S eller FI (run191-flottor)")
    g.add_argument("--n-reactors", type=int, help="N lika stora reaktorer (generisk flotta)")
    ap.add_argument("--reactor-mw", type=float, nargs="+",
                    help="explicit per-reaktor-MW-lista (överstyr --n-reactors/--preset-storlekar)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-cf", type=float, default=None, help="default: preset eller 0.85")
    ap.add_argument("--mttr", dest="mttr_hours", type=float, default=250.0,
                    help="medel forcerad varaktighet (h); högre = färre/längre = mindre taggigt")
    ap.add_argument("--min-outage", dest="min_outage_hours", type=float, default=48.0,
                    help="varaktighetsgolv (h); kapar sub-dygns-spikar")
    ap.add_argument("--biennial", action="store_true", help="revision vartannat år")
    ap.add_argument("--maint-days", type=float, nargs=2, default=(25, 45),
                    help="revisionslängd min max (dagar)")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end",   default="2025-12-31 23:00")
    ap.add_argument("--resolution", type=int, default=3, help="tidssteg i timmar")
    ap.add_argument("--tz", default="UTC")
    ap.add_argument("--output", default=None, help="spara profil som parquet")
    ap.add_argument("--plot", nargs="?", const="temp/nuclear_profile_gen.png", default=None,
                    help="spara PNG (default temp/nuclear_profile_gen.png)")
    args = ap.parse_args()

    if args.reactor_mw:
        reactor_mw = args.reactor_mw
        target_cf  = args.target_cf if args.target_cf is not None else 0.85
        name = f"{len(reactor_mw)}react"
    elif args.preset:
        reactor_mw = PRESETS[args.preset]["reactor_mw"]
        target_cf  = args.target_cf if args.target_cf is not None else PRESETS[args.preset]["target_cf"]
        name = args.preset
    else:
        reactor_mw = [1000.0] * args.n_reactors
        target_cf  = args.target_cf if args.target_cf is not None else 0.85
        name = f"{args.n_reactors}react"

    snaps = pd.date_range(args.start, args.end, freq=f"{args.resolution}h", tz=args.tz)
    prof, lam = generate(reactor_mw, snaps, args.seed, target_cf,
                         args.mttr_hours, args.min_outage_hours,
                         biennial=args.biennial, maintenance_days=tuple(args.maint_days))

    daily = prof.resample("1D").mean()
    scale = max(args.mttr_hours - args.min_outage_hours, 1e-6)
    p_short_orig = 1 - np.exp(-24.0 / args.mttr_hours)   # ren exp, för jämförelse
    print(f"flotta {name}: {len(reactor_mw)} reaktorer, target_cf {target_cf:.3f}")
    print(f"  mttr {args.mttr_hours:.0f} h · golv {args.min_outage_hours:.0f} h · "
          f"lambda {lam:.2f} avbrott/reaktor/år")
    print(f"  realiserad CF {prof.mean():.3f} · min(dag) {daily.min():.3f} · "
          f"dippar i dagsserie {_count_dips(daily)}")
    print(f"  (utan golv skulle {p_short_orig*100:.0f}% av forcerade avbrott vara <24 h; "
          f"nu är ALLA ≥ {args.min_outage_hours:.0f} h)")

    if args.output:
        prof.to_frame(name).to_parquet(args.output)
        print(f"  parquet: {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        fig, ax = plt.subplots(figsize=(13, 3.6))
        idx = pd.to_datetime(prof.index)
        ax.plot(idx, prof.values * 100, color="#9ec5e8", lw=0.4, alpha=0.7, label="3h")
        ax.plot(daily.index, daily.values * 100, color="#2c6fbb", lw=1.0, label="dagsmedel")
        ax.axhline(prof.mean() * 100, color="#2c6fbb", ls="--", lw=1.0, alpha=0.8)
        ax.set_ylim(30, 102); ax.set_ylabel("Tillgänglighet (%)")
        ax.grid(axis="y", color="#ddd", lw=0.6)
        ax.set_title(f"Kärnkraftsprofil {name} · seed {args.seed} · mttr {args.mttr_hours:.0f}h · "
                     f"golv {args.min_outage_hours:.0f}h · CF {prof.mean():.3f}",
                     loc="left", fontweight="bold", fontsize=11)
        ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.legend(loc="lower left", fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"  plot: {args.plot}")


if __name__ == "__main__":
    main()
