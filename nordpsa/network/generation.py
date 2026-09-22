"""Produktion: must-run-termik, kärnkraft (befintlig, ny, exogen), VRE, gas."""
import zlib

import numpy as np
import pandas as pd
import pypsa

from nordpsa.profiles.nuclear_availability import availability_timeseries
from nordpsa.network.costs import annualized_cost, crf, scenario_overnight_mw


# Nuclear: 1.0 = must-run låst till faktisk produktion (p_min_pu = p_max_pu).
# Sätt <1.0 för att tillåta load-following ned till den andelen av p_max_pu(t).
NUCLEAR_MIN_FRACTION = 1.0


def add_thermal(n: pypsa.Network, thermal_profile: pd.DataFrame, cfg: dict | None = None) -> None:
    """Termisk must-run som fast Generator (p_min_pu = p_max_pu = profil).

    Dispatch är helt given av data — optimeraren har inget val.
    Zoner utan termisk produktion (max = 0) hoppas över.

    Zoner med KVV-config (heat.zones[z].chp) får sin termisk-el reducerad med
    share_of_thermal → den delen produceras endogent av bakpress-KVV (se add_chp),
    så att KVV-elen inte dubbelräknas. ENDAST när värmesektorn är aktiv (heat.enabled);
    annars behålls full must-run-termik (add_chp lägger ju ej tillbaka KVV utan heat).
    """
    heat_on = bool(((cfg or {}).get("heat") or {}).get("enabled"))
    hz = ((cfg or {}).get("heat") or {}).get("zones") or {}
    for zone in thermal_profile.columns:
        profile = thermal_profile[zone].clip(lower=0)
        chp = (hz.get(zone, {}) or {}).get("chp")
        if chp and heat_on:
            profile = profile * (1.0 - float(chp.get("share_of_thermal", 1.0)))
        p_nom = float(profile.max())
        if p_nom == 0:
            continue
        pu = (profile / p_nom).clip(0, 1)
        n.add(
            "Generator", f"{zone} thermal",
            bus=zone,
            carrier="thermal",
            p_nom=p_nom,
            p_nom_extendable=False,
            p_min_pu=pu,
            p_max_pu=pu,
            marginal_cost=0.0,
        )


def add_nuclear(
    n:                 pypsa.Network,
    cfg:               dict,
    nuclear_profile:   pd.DataFrame,
    ccfg:              dict,
    r:                 float,
    fom_fraction:      float,
    n_years:           float,
    snapshots:         pd.DatetimeIndex | None = None,
    synthetic_nuclear: dict | None = None,
) -> None:
    tcfg     = ccfg["nuclear"]
    mc       = tcfg["vom_eur_per_mwh"]
    cap_cost = annualized_cost(
        tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r, tcfg.get("fom_fraction", fom_fraction)
    ) * n_years

    # Expansionsläge (--add-nuclear angivet → synthetic_nuclear['active']): befintlig
    # flotta blir FAST (ej extendable) + SYNTETISK profil; ny kärnkraft expanderas
    # separat i add_extra_nuclear. Dispatchläge: faktisk profil, extendable per config.
    # OBS: exogen FAST kärnkraft (--add-nuclear-fixed) byggs numera i en EGEN generator
    # (add_fixed_nuclear) och rör INTE denna befintliga-flotta-funktion.
    syn          = synthetic_nuclear or {}
    syn_active   = bool(syn.get("active"))
    syn_existing = syn.get("existing", {}) or {}
    syn_params   = syn.get("params", {}) or {}
    min_frac     = float(syn_params.get("min_load_frac", NUCLEAR_MIN_FRACTION))
    extendable   = tcfg["extendable"] and not syn_active

    for zone, zcfg in cfg["zones"].items():
        p_nom_existing = zcfg.get("nuclear_p_nom_mw", 0)
        if p_nom_existing == 0 and not extendable:
            continue

        if syn_active and zone in syn_existing and p_nom_existing > 0:
            # Befintlig flotta som syntetisk blandflotta (expansionsläge, --add-nuclear).
            sc   = syn_existing[zone]
            n_ex = int(sc["n_reactors"]); seed = int(sc["seed"])
            reactor_mw = [p_nom_existing / n_ex] * n_ex
            p_max = availability_timeseries(syn_params, snapshots, reactor_mw, seed=seed)
            p_min = (p_max * min_frac).clip(lower=0)
            print(f"  → kärnkraft {zone}: {p_nom_existing:.0f} MW "
                  f"({len(reactor_mw)} reaktorer synth, seed {seed}), "
                  f"realiserad CF={p_max.mean():.3f}")
        else:
            p_max = nuclear_profile[zone]
            p_min = (p_max * NUCLEAR_MIN_FRACTION).clip(lower=0)

        p_nom_total = p_nom_existing
        p_nom_max   = max(tcfg.get("p_nom_max_mw", np.inf), p_nom_total)
        n.add(
            "Generator", f"{zone} nuclear",
            bus=zone,
            carrier="nuclear",
            p_nom=p_nom_total,
            p_nom_min=p_nom_total,
            p_nom_max=p_nom_max,
            p_nom_extendable=extendable,
            p_max_pu=p_max,
            p_min_pu=p_min,
            marginal_cost=mc,
            capital_cost=cap_cost if extendable else 0.0,
        )


def add_extra_nuclear(n: pypsa.Network, extra_nuclear: list | None, ccfg: dict,
                       r: float, n_years: float, snapshots=None,
                       synth_params: dict | None = None, fom_fraction: float = 0.02) -> None:
    """Ny kärnkraft via --add-nuclear ZON:N:SEED (Generator '{zon} nuclear exp').

    extra_nuclear: lista av (zon, n_reactors, seed). N nya reaktorer med SYNTETISK
    stokastisk tillgänglighet (seed per zon → dekorrelerade avbrott), EXTENDABLE —
    kapaciteten optimeras (implicit reaktorstorlek ≈ p_nom_opt/N), tak
    p_nom_max = N × mw_per_reactor (synth_params['mw_per_reactor'], default 1500 MW).
    Kapital laddas på p_nom_opt. Dispatch must-run (p_min = min_load_frac × p_max).
    """
    if not extra_nuclear:
        return
    tcfg     = ccfg["nuclear"]
    mc       = tcfg["vom_eur_per_mwh"]
    # Per-zon diskontoränta (--nuclear-discount-rate ZON:RATE) → annualiserad kapital-
    # kostnad räknas om per zon. Default = global r. Påverkar bara EXTENDABLE expansion.
    disc_by_zone = tcfg.get("discount_rate_by_zone") or {}
    params   = synth_params or {}
    # min_load_frac_exp (--nuclear-min-load) gäller BARA NY kärnkraft; befintliga flottan
    # styrs av min_load_frac och förblir ren must-run. <1.0 = lastföljande ny kärnkraft.
    min_frac = float(params.get("min_load_frac_exp", params.get("min_load_frac", 1.0)))
    mw_each  = float(params.get("mw_per_reactor", 1500.0))
    for zone, n_react, seed in extra_nuclear:
        if zone not in n.buses.index:
            print(f"  Varning: kärnkrafts-zon {zone} saknas — hoppar över")
            continue
        r_zone   = float(disc_by_zone.get(zone, r))
        cap_cost = annualized_cost(
            tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r_zone,
            tcfg.get("fom_fraction", fom_fraction)
        ) * n_years
        p_max     = availability_timeseries(params, snapshots, int(n_react), seed=int(seed))
        p_min     = (p_max * min_frac).clip(lower=0)
        p_nom_max = n_react * mw_each
        n.add(
            "Generator", f"{zone} nuclear exp",
            bus=zone,
            carrier="nuclear",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_max=p_nom_max,
            p_nom_extendable=True,
            p_max_pu=p_max,
            p_min_pu=p_min,
            marginal_cost=mc,
            capital_cost=cap_cost,
        )
        print(f"  → ny kärnkraft {zone}: {int(n_react)} reaktorer synth (seed {seed}), "
              f"tak {p_nom_max:.0f} MW, realiserad CF={p_max.mean():.3f}, "
              f"r={r_zone:.0%}{' (override)' if zone in disc_by_zone else ''}, "
              f"kapital {cap_cost/n_years/1e3:.0f} €/kW/år→p_nom_opt")


def add_fixed_nuclear(n: pypsa.Network, fixed_nuclear: dict | None, cfg: dict,
                       r: float, n_years: float, snapshots=None,
                       synth_params: dict | None = None) -> None:
    """Exogen FAST kärnkraft via --add-nuclear-fixed ZON:N:MW[:SEED] som EGEN generator
    '{zon} nuclear fixed' (separat från befintliga flottan). Must-run (p_min=p_max) om
    inte min_load_frac_exp/--nuclear-min-load sänker golvet (lastföljande ny kärnkraft),
    SYNTETISK stokastisk tillgänglighet (seed → dekorrelerade avbrott), FAST p_nom.
    Bär verklig annualiserad SvK-2040-kapex (inkl. IDC) + VOM — laddas på p_nom ÄVEN i
    dispatch (konstant i objektivet → synliggör kostnaden). Befintliga flottan lämnas
    orörd (faktisk profil, capital 0)."""
    if not fixed_nuclear:
        return
    params = synth_params or {}
    oc_mw, life, fom_n, vom = scenario_overnight_mw(cfg, "nuclear", "svk_2040")
    cap_cost = oc_mw * (crf(life, r) + fom_n) * n_years
    for zone, adds in fixed_nuclear.items():
        if zone not in n.buses.index:
            print(f"  Varning: kärnkrafts-zon {zone} saknas — hoppar över")
            continue
        # adds: lista av (n_react, mw_each[, seed]); bygg heterogen reaktorlista + seed
        reactor_mw, seed = [], None
        for a in adds:
            n_r, mw = int(a[0]), float(a[1])
            reactor_mw += [mw] * n_r
            if len(a) > 2 and a[2] is not None:
                seed = int(a[2])
        if seed is None:
            # Deterministisk härledd seed (hash() randomiseras per process → ej reproducerbart)
            seed = int(params.get("seed", 0)) + (zlib.crc32(zone.encode()) % 1000)
        p_nom = float(sum(reactor_mw))
        p_max = availability_timeseries(params, snapshots, reactor_mw, seed=seed)
        # min_load_frac_exp (--nuclear-min-load) gäller även denna exogena NYA kärnkraft;
        # utan flaggan = 1.0 = must-run (oförändrat beteende).
        min_frac = float(params.get("min_load_frac_exp", params.get("min_load_frac", 1.0)))
        p_min = (p_max * min_frac).clip(lower=0)
        # Extendable-pinnat (p_nom_min=p_nom_max=p_nom) → FAST kapacitet men kapexen
        # hamnar i objective_constant (PyPSA släpper annars fasta kapitalkostnader).
        # Must-run bevaras via p_min_pu=p_max_pu på den pinnade p_nom_opt.
        n.add(
            "Generator", f"{zone} nuclear fixed",
            bus=zone,
            carrier="nuclear",
            p_nom=p_nom,
            p_nom_min=p_nom,
            p_nom_max=p_nom,
            p_nom_extendable=True,
            p_max_pu=p_max,
            p_min_pu=p_min,
            marginal_cost=vom,
            capital_cost=cap_cost,
        )
        print(f"  → fast kärnkraft {zone}: {len(reactor_mw)} reaktorer à "
              f"{'/'.join(str(int(m)) for m in sorted(set(reactor_mw)))} MW = {p_nom:.0f} MW "
              f"synth (seed {seed}), CF={p_max.mean():.3f}, "
              f"kapital {cap_cost/n_years/1e3:.0f} €/kW/år + vom {vom} €/MWh")


def add_vre(
    n:            pypsa.Network,
    cfg:          dict,
    vre_profiles: pd.DataFrame,
    vre_noms:     dict,
    ccfg:         dict,
    r:            float,
    fom_fraction: float,
    n_years:      float,
) -> None:
    vre_types = [
        ("wind_onshore",  "wind_onshore_p_nom_mw",  "wind_onshore"),
        ("wind_offshore", "wind_offshore_p_nom_mw", "wind_offshore"),
        ("solar",         "solar_p_nom_mw",          "solar"),
    ]
    for zone in cfg["zones"]:
        for carrier, nom_key, cost_key in vre_types:
            tcfg       = ccfg[cost_key]
            mc         = tcfg["vom_eur_per_mwh"]
            extendable = tcfg["extendable"]
            # Per-zon diskontoränta (t.ex. --offwind-discount-rate SE-S:0.03) → egen
            # annualiserad kapitalkostnad för den zonen/tekniken. Default = global r.
            r_zone     = float((tcfg.get("discount_rate_by_zone") or {}).get(zone, r))
            cap_cost   = annualized_cost(
                tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r_zone, tcfg.get("fom_fraction", fom_fraction)
            ) * n_years

            p_nom = vre_noms.get(zone, {}).get(nom_key, 0)
            col   = f"{zone}_{carrier}"
            if col not in vre_profiles.columns:
                continue
            if p_nom == 0 and not extendable:
                continue

            p_nom_max = tcfg.get("p_nom_max_mw", np.inf)
            n.add(
                "Generator", f"{zone} {carrier}",
                bus=zone,
                carrier=carrier,
                p_nom=p_nom,
                p_nom_min=p_nom,
                p_nom_max=p_nom_max,
                p_nom_extendable=extendable,
                p_max_pu=vre_profiles[col],
                marginal_cost=mc,
                capital_cost=cap_cost if extendable else 0.0,
            )


def add_gas(
    n:            pypsa.Network,
    cfg:          dict,
    ccfg:         dict,
    r:            float,
    fom_fraction: float,
    n_years:      float,
) -> None:
    """Gasturbin som utbyggbar peaklast-resurs per zon."""
    tcfg       = ccfg["gas"]
    mc         = tcfg["vom_eur_per_mwh"]
    extendable = tcfg["extendable"]
    cap_cost   = annualized_cost(
        tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r, tcfg.get("fom_fraction", fom_fraction)
    ) * n_years

    p_nom_max = tcfg.get("p_nom_max_mw", np.inf)
    for zone in cfg["zones"]:
        n.add(
            "Generator", f"{zone} gas",
            bus=zone,
            carrier="gas",
            p_nom=0.0,
            p_nom_min=0.0,
            p_nom_max=p_nom_max,
            p_nom_extendable=extendable,
            marginal_cost=mc,
            capital_cost=cap_cost if extendable else 0.0,
        )
