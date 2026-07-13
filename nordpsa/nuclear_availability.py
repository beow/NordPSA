"""
Syntetisk tillgänglighetsprofil för kärnkraft per zon (stokastisk Monte Carlo).

Modell (per reaktor, timvis): tillgänglighet a(t) ∈ {0,1} byggs av två lager
ovanpå full tillgänglighet:

  1. Revision (planerat): en sommarrevision/reaktor/år (eller vartannat år),
     staggrad över april–sep så reaktorerna inte ligger nere samtidigt.
     Längd ~ Uniform(min_days, max_days).
  2. Forcerade avbrott (stokastiskt): Poisson-process med rat lambda_fail
     (händelser/reaktor/år); varje avbrott varar ~ min_outage_hours +
     Exp(mean = mttr_hours - min_outage_hours). Golvet (min_outage_hours) kapar
     sub-dygns-avbrott som annars gör profilen "taggig"; det SKIFTADE upplägget
     bevarar medelvaraktigheten = mttr_hours så CF-kalibreringen är oförändrad.

Zonprofilen p_max_pu(t) = andel tillgängliga reaktorer (lika reaktorstorlek).

Kalibrering: lambda_fail härleds analytiskt så att årlig medeltillgänglighet
träffar target_cf givet revisionsschemat och vald mttr_hours (alternerande
förnyelseprocess). Realiserad CF rapporteras efter generering.

Enheter: p_max_pu (0–1). Returnerar pd.Series alignad mot snapshots.
"""
from typing import Dict

import numpy as np
import pandas as pd

# Defaultparametrar (kan överstyras via params-dict, jfr nuclear_synth i zones.yaml)
DEFAULTS = {
    "target_cf":              0.85,
    "mttr_hours":             300.0,        # medel reparationstid forcerat avbrott (~12.5 dygn)
    "min_outage_hours":       72.0,         # varaktighetsGOLV: kapar sub-3dygns-avbrott (mot taggighet)
    "maintenance_window_doy": [105, 273],   # ~15 apr – 30 sep
    "maintenance_days":       [25, 45],     # uniform min/max revisionslängd
    "revision_frequency":     "annual",     # annual | biennial
    "seed":                   42,
}

HOURS_PER_YEAR = 8760.0


def _calibrate_lambda(target_cf: float, d_mean_days: float,
                      mttr_hours: float, biennial: bool) -> float:
    """
    Härleder forcerad felrat lambda_fail (händelser/reaktor/år) så att den
    årliga medeltillgängligheten träffar target_cf.

        U   = 1 - target_cf                      (total target-otillgänglighet)
        p_p = D_mean/365.25 [· 0.5 vid biennial]  (planerad andel, revision)
        U_f = (U - p_p)/(1 - p_p)                 (marginell forcerad target)
        lam = U_f/(1 - U_f) · 8760/mttr           (alternerande förnyelseprocess)
    """
    U      = 1.0 - target_cf
    p_plan = d_mean_days / 365.25 * (0.5 if biennial else 1.0)
    U_f    = (U - p_plan) / (1.0 - p_plan)
    U_f    = float(np.clip(U_f, 0.0, 0.95))
    if U_f <= 0.0:
        return 0.0
    return U_f / (1.0 - U_f) * HOURS_PER_YEAR / mttr_hours


def _apply_outage(up: np.ndarray, start_h: float, dur_h: float, dt_h: float) -> None:
    """Sätter tillgängligheten till 0 över intervallet [start_h, start_h+dur_h)."""
    T = len(up)
    i0 = int(np.floor(start_h / dt_h))
    i1 = int(np.ceil((start_h + dur_h) / dt_h))
    i0 = max(i0, 0)
    i1 = min(i1, T)
    if i1 > i0:
        up[i0:i1] = 0.0


def _schedule_maintenance(up: np.ndarray, rng: np.random.Generator,
                          reactor_idx: int, n_reactors: int,
                          timestamps: pd.DatetimeIndex, dt_h: float,
                          window_doy, days_range, biennial: bool) -> None:
    """En staggrad sommarrevision/reaktor/år (vartannat år om biennial)."""
    doy_lo, doy_hi = float(window_doy[0]), float(window_doy[1])
    d_lo, d_hi     = float(days_range[0]), float(days_range[1])
    t0    = timestamps[0]
    years = sorted({ts.year for ts in timestamps})

    for k, year in enumerate(years):
        # Vartannat år: halva reaktorerna jämna år, halva udda år.
        if biennial and ((reactor_idx + k) % 2 == 1):
            continue

        d_days = rng.uniform(d_lo, d_hi)
        # Stagger: dela revisionsfönstret i n_reactors slottar, en per reaktor,
        # med liten jitter inom slotten.
        slot      = (doy_hi - doy_lo) / n_reactors
        slot_lo   = doy_lo + reactor_idx * slot
        latest    = max(slot_lo, slot_lo + slot - d_days)
        start_doy = rng.uniform(slot_lo, latest) if latest > slot_lo else slot_lo

        year_start = pd.Timestamp(year=year, month=1, day=1, tz=t0.tz)
        start_h = (year_start - t0).total_seconds() / 3600.0 + (start_doy - 1.0) * 24.0
        _apply_outage(up, start_h, d_days * 24.0, dt_h)


def _sample_forced_outages(up: np.ndarray, rng: np.random.Generator,
                           lambda_fail: float, mttr_hours: float,
                           total_hours: float, dt_h: float,
                           min_outage_hours: float = 0.0) -> None:
    """Poisson-process av forcerade avbrott med SKIFTAD exp-varaktighet.

    varaktighet = min_outage_hours + Exp(medel = mttr_hours - min_outage_hours)
      → minsta varaktighet = min_outage_hours, MEDEL = mttr_hours (oförändrad).
    Eftersom medlet är oförändrat är lambda-kalibreringen (och därmed CF) intakt;
    golvet tar bort de korta avbrotten som annars ger taggiga sub-dygns-dippar.
    """
    if lambda_fail <= 0.0:
        return
    scale = max(mttr_hours - min_outage_hours, 1e-6)   # exp-medel efter skiftet
    n_events = rng.poisson(lambda_fail * total_hours / HOURS_PER_YEAR)
    for _ in range(int(n_events)):
        start_h = rng.uniform(0.0, total_hours)
        dur_h   = min_outage_hours + rng.exponential(scale)
        _apply_outage(up, start_h, dur_h, dt_h)


def availability_timeseries(params: Dict, timestamps: pd.DatetimeIndex,
                            reactor_mw, seed: int | None = None) -> pd.Series:
    """
    Genererar syntetisk kärnkrafts-tillgänglighet (p_max_pu) alignad mot snapshots.

    Args:
        params:     dict med nyckparna i DEFAULTS (saknade fylls med default).
        timestamps: snapshot-index (jämn tidssteg, t.ex. 1h eller 3h).
        reactor_mw: per-reaktor-effekt (lista/array, godtyckliga storlekar) ELLER
                    ett heltal N → N lika stora reaktorer. Profilen viktas efter
                    effekt: p_max_pu(t) = Σ_r mw_r·up_r(t) / Σ_r mw_r.
        seed:       RNG-seed (överstyrs params['seed'] om angivet).

    Returns:
        pd.Series (p_max_pu ∈ [0,1]) indexerad på timestamps.
    """
    p = {**DEFAULTS, **(params or {})}
    if np.isscalar(reactor_mw):
        reactor_mw = [1.0] * int(reactor_mw)
    mw = np.asarray(reactor_mw, dtype=float)
    n_reactors = len(mw)
    if n_reactors < 1:
        raise ValueError(f"Minst en reaktor krävs, fick {n_reactors}")

    if seed is None:
        seed = int(p["seed"])
    rng = np.random.default_rng(seed)

    dt_h        = (timestamps[1] - timestamps[0]).total_seconds() / 3600.0
    T           = len(timestamps)
    total_hours = T * dt_h
    biennial    = str(p["revision_frequency"]).lower() == "biennial"

    d_mean      = 0.5 * (p["maintenance_days"][0] + p["maintenance_days"][1])
    lambda_fail = _calibrate_lambda(p["target_cf"], d_mean, p["mttr_hours"], biennial)

    avail_mw = np.zeros(T, dtype=float)
    for r in range(n_reactors):
        up = np.ones(T, dtype=float)
        _schedule_maintenance(up, rng, r, n_reactors, timestamps, dt_h,
                              p["maintenance_window_doy"], p["maintenance_days"], biennial)
        _sample_forced_outages(up, rng, lambda_fail, p["mttr_hours"], total_hours, dt_h,
                               min_outage_hours=p["min_outage_hours"])
        avail_mw += mw[r] * up

    p_max_pu = avail_mw / mw.sum()
    return pd.Series(p_max_pu, index=timestamps, name="nuclear_avail")


if __name__ == "__main__":
    # Fristående sanity-check (ingen solve): generera och rapportera för en zon.
    import sys

    n_reactors = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    seed       = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    ts = pd.date_range("2023-01-01", "2025-12-31 23:00", freq="h", tz="UTC")

    s = availability_timeseries({}, ts, n_reactors, seed=seed)
    levels = np.unique(np.round(s.values, 6))
    print(f"Reaktorer:        {n_reactors}  (seed={seed})")
    print(f"Timmar:           {len(s)}")
    print(f"Realiserad CF:    {s.mean():.4f}  (target {DEFAULTS['target_cf']})")
    print(f"Min/max p_max_pu: {s.min():.3f} / {s.max():.3f}")
    print(f"Distinkta nivåer: {len(levels)}  (väntat ≤ {n_reactors + 1})  {levels}")

    # Reproducerbarhet
    s2 = availability_timeseries({}, ts, n_reactors, seed=seed)
    print(f"Reproducerbar:    {bool((s.values == s2.values).all())}")

    # CF per år
    for yr, g in s.groupby(s.index.year):
        print(f"  {yr}: CF={g.mean():.4f}")
