"""
Bygger PyPSA-nätverket för NordPSA.

Nätverksstruktur:
  - 6 bussar (budzoner)
  - 8 Links (bidirektionella NTC-begränsningar)
  - Generatorer: hydro (StorageUnit), kärnkraft, vind on/offshore, sol, termisk must-run
  - Last per zon
  - Load shedding (slack) per zon med högt pris
"""
from pathlib import Path
from typing import Callable, Dict

import xarray as xr

import numpy as np
import pandas as pd
import pypsa
import yaml

from nordpsa.hydro import inflow_timeseries, load_nve_inflow, load_nve_ror
from nordpsa.nuclear_availability import availability_timeseries

# Load shedding pris (EUR/MWh)
MC_SLACK = 3000.0

# Nuclear: 1.0 = must-run låst till faktisk produktion (p_min_pu = p_max_pu).
# Sätt <1.0 för att tillåta load-following ned till den andelen av p_max_pu(t).
NUCLEAR_MIN_FRACTION = 1.0


# ---------------------------------------------------------------------------
# Kostnadsberäkning
# ---------------------------------------------------------------------------

def _crf(lifetime: int, r: float) -> float:
    """Capital Recovery Factor."""
    return r * (1 + r) ** lifetime / ((1 + r) ** lifetime - 1)


def _annualized_cost(overnight_eur_per_w: float, lifetime: int,
                     r: float, fom_fraction: float) -> float:
    """Annualiserad kapitalkostnad (EUR/MW/år).

    overnight_eur_per_w: t.ex. 7.0 för 7 EUR/W = 7 000 000 EUR/MW
    """
    oc_mw = overnight_eur_per_w * 1e6
    return oc_mw * (_crf(lifetime, r) + fom_fraction)


def build_network(
    cfg:                     dict,
    snapshots:               pd.DatetimeIndex,
    load:                    pd.DataFrame,
    vre_profiles:            pd.DataFrame,
    vre_noms:                dict,
    nuclear_profile:         pd.DataFrame,
    thermal_profile:         pd.DataFrame,
    hydro_params:            dict,
    market_prices:           Dict[str, pd.Series],
    actual_inflow:           bool = True,
    cyclic_soc:              bool = True,
    soc_initial_override:    dict | None = None,
    voll:                    bool = False,
    batteries:               list | None = None,
    extra_nuclear:           list | None = None,
    extra_wind:              list | None = None,
    synthetic_nuclear:       dict | None = None,
    hydrogen_overrides:      dict | None = None,
    heat_load:               pd.DataFrame | None = None,
    ev_profiles:             pd.DataFrame | None = None,
    ev_overrides:            dict | None = None,
) -> pypsa.Network:
    """
    Bygger och returnerar ett PyPSA Network.

    Termisk produktion modelleras som ett måste-köra Generator-objekt med
    p_min_pu = p_max_pu = faktisk profil. Lasten är oförändrad (bruttolast).

    Alla tidsserier måste ha samma index som `snapshots`.
    """
    n = pypsa.Network()
    n.set_snapshots(snapshots)

    # PyPSA 1.x sätter snapshot_weightings=1 per default; för 3h-tidssteg
    # måste vikterna sättas till dt_h så att rörliga kostnader (EUR/MWh) och
    # kapitalkostand (EUR/MW/år × n_år) är konsistenta i LP-objektet.
    dt_h  = (snapshots[1] - snapshots[0]).total_seconds() / 3600
    n.snapshot_weightings[:] = dt_h

    # Ytterligare fast last (t.ex. datacenter)
    extra = cfg.get("additional_load_mw", {})
    if extra:
        load = load.copy()
        for zone, mw in extra.items():
            if zone in load.columns:
                load[zone] += mw

    # Skalningsfaktor: capital_cost anges per år; modellen kan täcka fler år
    ccfg  = cfg["costs"]
    r     = ccfg["discount_rate"]
    fom   = ccfg["fom_fraction"]
    n_years = len(snapshots) * dt_h / 8760.0

    # Fjärrvärme: bygg FV-värmebehovsprofiler + DRA BORT dagens FV-el ur AC-lasten
    # (dubbelräkning) innan _add_loads. Värmekomponenterna byggs i _add_heat (slutet).
    heat_demand = _heat_demand_profiles(cfg, heat_load, snapshots, dt_h, n_years)
    if heat_demand:
        load = load.copy()
        hz = cfg["heat"]["zones"]
        for zone, dh in heat_demand.items():
            el_twh = float(hz.get(zone, {}).get("el_input_twh", 0.0))
            dh_e   = float(dh.sum() * dt_h)
            if el_twh > 0 and dh_e > 0 and zone in load.columns:
                load[zone] = load[zone] - dh * (el_twh * 1e6 * n_years / dh_e)

    zone_prices = {z: market_prices[z] for z in cfg["zones"] if z in market_prices}

    _add_buses(n, cfg)
    _add_links(n, cfg)
    _add_loads(n, load)
    _add_slack(n, cfg, all_zones=voll)
    _add_thermal(n, thermal_profile, cfg)
    _add_hydro(n, cfg, hydro_params, snapshots, ccfg,
               actual_inflow=actual_inflow,
               cyclic_soc=cyclic_soc, soc_initial_override=soc_initial_override,
               zone_prices=zone_prices)
    _add_nuclear(n, cfg, nuclear_profile, ccfg, r, fom, n_years,
                 snapshots, synthetic_nuclear)
    _add_vre(n, cfg, vre_profiles, vre_noms, ccfg, r, fom, n_years)
    _add_gas(n, cfg, ccfg, r, fom, n_years)
    _add_market_connections(n, cfg, market_prices)
    _add_batteries(n, batteries, ccfg, r, n_years)
    _add_extra_nuclear(n, extra_nuclear, ccfg, r, n_years, snapshots,
                       (synthetic_nuclear or {}).get("params"), fom)
    _add_extra_wind(n, extra_wind, vre_profiles, ccfg)
    _add_hydrogen(n, cfg, r, fom, n_years, hydrogen_overrides)
    _add_heat(n, cfg, heat_demand, r, n_years)
    _add_chp(n, cfg, heat_demand, r, n_years)
    _add_chp_fixed(n, cfg, thermal_profile)
    _add_ev(n, cfg, ev_profiles, ev_overrides, snapshots, dt_h, n_years)

    return n


# ---------------------------------------------------------------------------
# Interna byggfunktioner
# ---------------------------------------------------------------------------

def _add_buses(n: pypsa.Network, cfg: dict) -> None:
    for zone in cfg["zones"]:
        n.add("Bus", zone, carrier="AC")


def _add_links(n: pypsa.Network, cfg: dict) -> None:
    for z0, z1, p_nom in cfg["links"]:
        n.add(
            "Link", f"{z0}-{z1}",
            bus0=z0, bus1=z1,
            p_nom=p_nom,
            p_min_pu=-1.0,   # bidirektionell
            efficiency=1.0,
            marginal_cost=0.0,
        )


def _add_loads(n: pypsa.Network, load: pd.DataFrame) -> None:
    for zone in load.columns:
        n.add("Load", f"{zone} load", bus=zone, p_set=load[zone])


def _add_slack(n: pypsa.Network, cfg: dict, all_zones: bool = False) -> None:
    """Load shedding-generator per zon.

    all_zones=False (standard): bara zoner utan marknadsanslutning.
    all_zones=True (--voll): alla zoner, inklusive de med marknadsanslutning.
      Används som VOLL-mått: slack-dispatch × MC_SLACK = losskostnad i EUR.
      Priser toppas vid MC_SLACK istf att dualvariabler exploderar.
    """
    market_zones = {zone for _name, zone, *_ in cfg.get("market_connections", [])}
    for zone in cfg["zones"]:
        if not all_zones and zone in market_zones:
            continue
        n.add(
            "Generator", f"{zone} slack",
            bus=zone,
            p_nom=1e6,
            marginal_cost=MC_SLACK,
            carrier="slack",
        )


def _add_thermal(n: pypsa.Network, thermal_profile: pd.DataFrame, cfg: dict | None = None) -> None:
    """Termisk must-run som fast Generator (p_min_pu = p_max_pu = profil).

    Dispatch är helt given av data — optimeraren har inget val.
    Zoner utan termisk produktion (max = 0) hoppas över.

    Zoner med KVV-config (heat.zones[z].chp) får sin termisk-el reducerad med
    share_of_thermal → den delen produceras endogent av bakpress-KVV (se _add_chp),
    så att KVV-elen inte dubbelräknas. ENDAST när värmesektorn är aktiv (heat.enabled);
    annars behålls full must-run-termik (_add_chp lägger ju ej tillbaka KVV utan heat).
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


NVE_INFLOW_ZONES = {"NO-N", "NO-S", "SE-N", "SE-S"}


def _add_hydro(
    n:                    pypsa.Network,
    cfg:                  dict,
    hydro_params:         dict,
    snapshots:            pd.DatetimeIndex,
    ccfg:                 dict,
    actual_inflow:        bool = True,
    cyclic_soc:           bool = True,
    soc_initial_override: dict | None = None,
    zone_prices:          dict | None = None,
) -> None:
    mc_default = ccfg["hydro"]["vom_eur_per_mwh"]
    for zone, zcfg in cfg["zones"].items():
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0 or zone not in hydro_params:
            continue

        if actual_inflow and zone in NVE_INFLOW_ZONES:
            inflow = load_nve_inflow(zone, snapshots)
            # Run-of-river: separat must-run-generator. Reservoarinflödet
            # (inflow_nve) exkluderar redan B11. Reducera reservoar-p_nom med
            # RoR-turbinkapaciteten så total turbinkapacitet bevaras.
            ror = load_nve_ror(zone, snapshots)
            ror_p_nom = float(ror.max())
            if ror_p_nom > 1.0:
                pu = (ror / ror_p_nom).clip(0, 1)
                n.add(
                    "Generator", f"{zone} hydro_ror",
                    bus=zone,
                    carrier="hydro",
                    p_nom=ror_p_nom,
                    p_nom_extendable=False,
                    p_min_pu=pu,
                    p_max_pu=pu,
                    marginal_cost=ccfg["hydro"].get("vom_ror_eur_per_mwh", mc_default),
                )
                # Bevara reservoarens energikapacitet (p_nom × max_hours) när
                # turbineffekten reduceras med RoR-andelen.
                cap_mwh = p_nom * max_h
                p_nom   = max(p_nom - ror_p_nom, 1.0)
                max_h   = cap_mwh / p_nom
        else:
            params = hydro_params[zone]
            inflow = inflow_timeseries(params, snapshots)

        if cyclic_soc:
            soc_init = 0.0  # ignoreras när cyclic=True
        elif soc_initial_override and zone in soc_initial_override:
            soc_init = soc_initial_override[zone]
        else:
            frac = zcfg.get("hydro_soc_initial", 0.5)
            soc_init = frac * p_nom * max_h

        if zone_prices and zone in zone_prices:
            mc = zone_prices[zone].reindex(snapshots).ffill().clip(lower=mc_default)
        else:
            mc = mc_default

        n.add(
            "StorageUnit", f"{zone} hydro",
            bus=zone,
            carrier="hydro",
            p_nom=p_nom,
            max_hours=max_h,
            inflow=inflow,
            cyclic_state_of_charge=cyclic_soc,
            state_of_charge_initial=soc_init,
            spill_cost=ccfg["hydro"].get("spill_cost_eur_per_mwh", 0.1),  # lågt → tillåt spill vid full reservoar
            p_min_pu=0.0,          # förbjud pumpning (ej pumpad-lagringshydro)
            efficiency_dispatch=1.0,
            marginal_cost=mc,
        )


def _add_batteries(n: pypsa.Network, batteries: list | None,
                   ccfg: dict, r: float, n_years: float) -> None:
    """Lägger till batterier som StorageUnit (carrier 'battery').

    batteries: lista av (zon, p_nom_mw, max_hours, extendable). Round-trip ~90%
    (0.95×0.95), kan ladda (p_min_pu=-1), cyklisk SOC, litet marginalkostnad för
    att bryta degeneracy. Inget inflöde. Varaktigheten (max_hours) är fast — vid
    extendable optimeras bara effekten (p_nom), energi = p_nom × max_hours.

    Kostnad delas i effektdel (€/kW på p_nom) + energidel (€/kWh × max_hours).
    overnight €/MW = (power_eur_per_kw + max_hours × energy_eur_per_kwh) × 1000.
    Annualiseras med batteriets egen livslängd (kort, ~15 år) och FOM. Fast
    batteri (extendable=False) får capital_cost=0; extendable får finit p_nom_max.
    """
    if not batteries:
        return
    bc      = ccfg["battery"]
    e_kwh   = bc["energy_eur_per_kwh"]
    p_kw    = bc["power_eur_per_kw"]
    life    = bc["lifetime_years"]
    fom     = bc["fom_fraction"]
    pmax    = float(bc.get("p_nom_max_mw", 50000))
    for zone, p_nom, max_h, ext in batteries:
        if zone not in n.buses.index:
            print(f"  Varning: batteri-zon {zone} saknas — hoppar över")
            continue
        overnight_mw = (p_kw + max_h * e_kwh) * 1e3          # €/MW
        cap_cost     = overnight_mw * (_crf(life, r) + fom) * n_years
        n.add(
            "StorageUnit", f"{zone} battery",
            bus=zone,
            carrier="battery",
            p_nom=0.0 if ext else p_nom,
            p_nom_extendable=ext,
            p_nom_min=0.0 if ext else p_nom,
            p_nom_max=pmax if ext else float("inf"),
            max_hours=max_h,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=True,
            marginal_cost=0.01,
            capital_cost=cap_cost if ext else 0.0,
        )
        mode = f"extendable ≤{pmax:.0f} MW" if ext else f"fast {p_nom:.0f} MW"
        print(f"  → batteri {zone}: {mode} / {max_h:.0f}h, "
              f"overnight {overnight_mw/1e3:.0f} €/kW, "
              f"annual.kap {overnight_mw*(_crf(life, r)+fom)/1e3:.0f} €/kW/år")


def _add_extra_nuclear(n: pypsa.Network, extra_nuclear: list | None, ccfg: dict,
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
    min_frac = float(params.get("min_load_frac", 1.0))
    mw_each  = float(params.get("mw_per_reactor", 1500.0))
    for zone, n_react, seed in extra_nuclear:
        if zone not in n.buses.index:
            print(f"  Varning: kärnkrafts-zon {zone} saknas — hoppar över")
            continue
        r_zone   = float(disc_by_zone.get(zone, r))
        cap_cost = _annualized_cost(
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


def _add_extra_wind(n: pypsa.Network, extra_wind: list | None,
                    vre_profiles: pd.DataFrame, ccfg: dict) -> None:
    """Lägger till NY landbaserad vindkraft (Generator, carrier 'wind_onshore').

    extra_wind: lista av (zon, p_nom_mw). Använder samma CF-profil som zonens
    befintliga wind_onshore ({zon}_wind_onshore). Namnges '{zon} wind_new'.
    Icke-extendable (dispatch). MC = wind_onshore VOM. Energi-motsvarighet till
    --add-nuclear men variabel (curtailas vid överskott).
    """
    if not extra_wind:
        return
    mc = ccfg["wind_onshore"]["vom_eur_per_mwh"]
    for zone, p_nom in extra_wind:
        if zone not in n.buses.index:
            print(f"  Varning: vind-zon {zone} saknas — hoppar över")
            continue
        col = f"{zone}_wind_onshore"
        if col not in vre_profiles.columns:
            print(f"  Varning: ingen vindprofil {col} — hoppar över")
            continue
        n.add(
            "Generator", f"{zone} wind_new",
            bus=zone,
            carrier="wind_onshore",
            p_nom=p_nom,
            p_nom_extendable=False,
            p_max_pu=vre_profiles[col],
            marginal_cost=mc,
        )
        cf = vre_profiles[col].mean()
        print(f"  → ny vind {zone}: {p_nom:.0f} MW (CF {cf:.3f}, MC {mc} EUR/MWh)")


def _add_hydrogen(n: pypsa.Network, cfg: dict, r: float, fom: float,
                  n_years: float, overrides: dict | None = None) -> None:
    """Bygger valfria vätgassystem per zon (power-to-X):

        elbuss → Link(elektrolys, η_el) → Bus(H2) → Store(lager, e_cyclic)
        Bus(H2) → Load(baslast, fast MW) + Generator(slack, hög MC)
        Bus(H2) → Link(turbin, η_turb) → elbuss     (valfri)

    Zon-konfiguration från cfg['hydrogen'] sammanslaget med `overrides` (CLI, har
    företräde). Teknikkostnader/verkningsgrader från cfg['costs']['hydrogen'].
    Allt i MWh (LHV). Hoppar tyst över om inget H2 konfigurerats.

    Enheter: elektrolysör-p_nom i MW_el (bus0=el). Turbin specas i MW_el_ut men
    PyPSA-p_nom (bus0=H2) = p_el/η_turb, capital_cost skalas med η_turb → kostnad
    per kW_el ut. Icke-extendable komponenter får capital_cost=0.
    """
    h2_zones = dict(cfg.get("hydrogen") or {})
    if overrides:
        h2_zones.update(overrides)   # CLI har företräde
    if not h2_zones:
        return

    hc       = cfg["costs"]["hydrogen"]
    el_c, tb_c, st_c = hc["electrolyser"], hc["turbine"], hc["store"]
    mc_slack = hc.get("slack_eur_per_mwh", MC_SLACK)
    el_pmin  = hc.get("electrolyser_p_min_pu", 0.0)

    for car in ("H2", "electrolyser", "H2 turbine", "H2 store", "H2 slack"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    def ann_kw(c):   # annualiserad €/MW (overnight i €/kW)
        return c["overnight_eur_per_kw"] * 1e3 * (_crf(c["lifetime_years"], r) + c["fom_fraction"]) * n_years
    def ann_kwh(overnight, c):  # annualiserad €/MWh (overnight i €/kWh, per zon)
        return overnight * 1e3 * (_crf(c["lifetime_years"], r) + c["fom_fraction"]) * n_years

    # Lagerkostnad per zon: zon-block > geologiska undantag > default
    st_base    = st_c["overnight_eur_per_kwh"]
    st_by_zone = st_c.get("overnight_eur_per_kwh_by_zone") or {}

    for zone, zc in h2_zones.items():
        if zone not in n.buses.index:
            print(f"  Varning: H2-zon {zone} saknas — hoppar över")
            continue
        h2bus = f"{zone} H2"
        n.add("Bus", h2bus, carrier="H2")

        # Elektrolys: bus0=el, bus1=H2; p_nom i MW_el
        elc     = zc.get("electrolyser", {})
        el_ext  = bool(elc.get("extendable", False))
        el_pnom = float(elc.get("p_nom_mw", 0.0))
        el_pmax = float(elc.get("p_nom_max_mw", 50000.0))
        n.add("Link", f"{zone} electrolyser",
              bus0=zone, bus1=h2bus, carrier="electrolyser",
              efficiency=el_c["efficiency"],
              p_nom=el_pnom, p_nom_extendable=el_ext,
              p_nom_min=0.0 if el_ext else el_pnom,
              p_nom_max=el_pmax if el_ext else float("inf"),
              p_min_pu=el_pmin,
              capital_cost=ann_kw(el_c) if el_ext else 0.0)

        # Lager: Store på H2-bussen (energi fri från effekt; e_cyclic: start=slut)
        stc       = zc.get("store", {})
        st_ext    = bool(stc.get("extendable", False))
        e_nom     = float(stc.get("e_nom_mwh", 0.0))
        st_overn  = float(stc.get("overnight_eur_per_kwh", st_by_zone.get(zone, st_base)))
        st_emax   = float(stc.get("e_nom_max_mwh", st_c.get("e_nom_max_mwh", 1e7)))
        n.add("Store", f"{zone} H2 store",
              bus=h2bus, carrier="H2 store",
              e_nom=e_nom, e_nom_extendable=st_ext,
              e_nom_min=0.0 if st_ext else e_nom,
              e_nom_max=st_emax if st_ext else float("inf"),
              e_cyclic=True,
              capital_cost=ann_kwh(st_overn, st_c) if st_ext else 0.0)

        # Baslast (konstant MW) + slack (omött H2)
        demand = float(zc.get("demand_mw", 0.0))
        n.add("Load", f"{zone} H2 load", bus=h2bus, p_set=demand)
        n.add("Generator", f"{zone} H2 slack",
              bus=h2bus, carrier="H2 slack",
              p_nom=1e6, marginal_cost=mc_slack)

        # Valfri turbin: bus0=H2, bus1=el; config p_nom i MW_el_ut → p_nom(H2)=el/η
        tbc      = zc.get("turbine")
        turb_txt = "ingen turbin"
        if tbc:
            eta    = tb_c["efficiency"]
            tb_ext = bool(tbc.get("extendable", False))
            p_el   = float(tbc.get("p_nom_mw", 0.0))
            p_h2   = p_el / eta if eta > 0 else 0.0
            n.add("Link", f"{zone} H2 turbine",
                  bus0=h2bus, bus1=zone, carrier="H2 turbine",
                  efficiency=eta,
                  p_nom=p_h2, p_nom_extendable=tb_ext,
                  p_nom_min=0.0 if tb_ext else p_h2,
                  capital_cost=(ann_kw(tb_c) * eta) if tb_ext else 0.0)
            turb_txt = f"turbin {p_el:.0f} MW_el (η={eta})"

        print(f"  → H2 {zone}: last {demand:.0f} MW, "
              f"elektrolys {el_pnom:.0f} MW_el{' ext' if el_ext else ''} (η={el_c['efficiency']}), "
              f"lager {e_nom:.0f} MWh{' ext' if st_ext else ''} ({st_overn:g} €/kWh), {turb_txt}")


def _heat_demand_profiles(cfg, heat_load, snapshots, dt_h, n_years) -> dict:
    """FV-värmebehovsprofiler (termiskt MW) per zon = byggnadsvärmeprofilen
    (heat_load.parquet) skalad så zonens årsenergi = dh_demand_twh. Tom dict om
    värmesektorn ej aktiverad eller heat_load saknas."""
    hcfg = cfg.get("heat") or {}
    if not hcfg.get("enabled") or heat_load is None:
        return {}
    # KVV-fast-läge (--chp-fixed-gw): ingen värmebuss/lager/VP/panna byggs och dagens
    # FV-el dras INTE ur AC-lasten. KVV-elen återinförs av _add_chp_fixed i stället.
    if hcfg.get("chp_fixed_gw"):
        return {}
    out = {}
    for zone, zc in (hcfg.get("zones") or {}).items():
        if zone not in heat_load.columns:
            print(f"  Varning: heat-zon {zone} saknas i heat_load — hoppar över")
            continue
        prof = heat_load[zone].reindex(snapshots).ffill().clip(lower=0)
        ann  = float(prof.sum() * dt_h / 1e6 / n_years)   # TWh/år i profilen
        target = float(zc.get("dh_demand_twh", 0.0))
        out[zone] = prof * (target / ann) if ann > 0 else prof * 0.0
    return out


def _add_heat(n: pypsa.Network, cfg: dict, heat_demand: dict,
              r: float = 0.06, n_years: float = 1.0) -> None:
    """Fjärrvärmebuss per zon (v3 "1 buss + laster"):

        AC → Link(el-panna, η≈1)  ┐
        AC → Link(stor-VP, COP)   ┝→ Bus(heat) → Load(FV-behov) + Store(ackumulator)
        Generator(bio/KVV, MC)    ┘                              + Generator(slack)

    Flexen: optimeraren väljer el (panna/VP) när elen är billig vs bio annars, och
    laddar ackumulatorn. heat_demand = termiska behovsprofiler (från _heat_demand_
    profiles). Dagens FV-el är redan bortdragen ur AC-lasten i build_network.
    """
    hcfg = cfg.get("heat") or {}
    if not hcfg.get("enabled") or not heat_demand:
        return
    ccfg     = cfg["costs"]
    fom_g    = ccfg.get("fom_fraction", 0.02)
    cop      = float(hcfg.get("cop", 3.0))
    bio_vom  = float(hcfg.get("bio_vom_eur_per_mwh", 30.0))
    el_tax   = float(hcfg.get("el_tax_eur_per_mwh", 0.0))   # energiskatt per MWh el (på Link-p0)
    elb_vom  = float(hcfg.get("elboiler_vom_eur_per_mwh", 0.5))
    hp_vom   = float(hcfg.get("hp_vom_eur_per_mwh", 0.5))
    mr_share = float(hcfg.get("mustrun_share", 0.0))        # avfall/restvärme (~0 MC, must-run)
    mr_vom   = float(hcfg.get("mustrun_vom_eur_per_mwh", 0.0))
    st_hours = float(hcfg.get("store_hours", 6))
    # Värmelager: fast (store_hours×topplast) ELLER investerbart mot TES-kostnad
    # (--heat-store-ext → heat.store_extendable). Termiskt grop-/tanklager är billigt.
    st_ext   = bool(hcfg.get("store_extendable", False))
    st_overn = float(hcfg.get("store_overnight_eur_per_kwh", 0.5))   # €/kWh_th
    st_life  = int(hcfg.get("store_lifetime_years", 40))
    st_fom   = float(hcfg.get("store_fom_fraction", 0.01))
    st_emax  = float(hcfg.get("store_e_nom_max_mwh", 1e8))
    st_cap   = st_overn * 1e3 * (_crf(st_life, r) + st_fom) * n_years if st_ext else 0.0
    mc_slack = float(hcfg.get("slack_eur_per_mwh", MC_SLACK))
    zcfg     = hcfg.get("zones") or {}

    for car in ("heat", "heat mustrun", "heat chp", "heat elboiler", "heat hp", "heat store", "heat slack"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    for zone, dh in heat_demand.items():
        if zone not in n.buses.index:
            print(f"  Varning: heat-zon {zone} saknar AC-buss — hoppar över")
            continue
        zc       = zcfg.get(zone, {})
        zone_cop = float(zc.get("cop", cop))                 # per-zon COP (DK 4 ≠ SE 3)
        zone_mr  = float(zc.get("mustrun_share", mr_share))  # per-zon must-run-andel (DK 0.25 ≠ SE 0.41)
        hb   = f"{zone} heat"
        peak = float(dh.max())
        # Lagervolym (fast läge): per-zon store_gwh om angiven, annars store_hours × topplast.
        e_store = float(zc["store_gwh"]) * 1e3 if "store_gwh" in zc else st_hours * peak
        n.add("Bus", hb, carrier="heat")
        n.add("Load", f"{zone} heat load", bus=hb, p_set=dh)
        # Ackumulator (termiskt lager, e_cyclic): fast eller investerbart mot TES-kostnad
        if st_ext:
            n.add("Store", f"{zone} heat store", bus=hb, carrier="heat store",
                  e_nom=0.0, e_nom_extendable=True, e_nom_min=0.0, e_nom_max=st_emax,
                  e_cyclic=True, capital_cost=st_cap)
        else:
            n.add("Store", f"{zone} heat store", bus=hb, carrier="heat store",
                  e_nom=e_store, e_cyclic=True)
        # El-panna: AC → heat, η≈1.  MC = vom + elskatt (per MWh el, dvs på p0).
        # Extendable (costs.heat_elboiler) → capex per MW_el = ann(€/W_th)×η; golv = dagens MW.
        eb       = ccfg.get("heat_elboiler", {})
        eb_ext   = bool(eb.get("extendable", False))
        eb_pnom  = float(zc.get("elboiler_mw", 0.0))
        eb_cap   = (_annualized_cost(eb["overnight_eur_per_w"], eb["lifetime_years"], r,
                    eb.get("fom_fraction", fom_g)) * 0.99 * n_years) if (eb_ext and eb) else 0.0
        n.add("Link", f"{zone} heat elboiler", bus0=zone, bus1=hb, carrier="heat elboiler",
              efficiency=0.99, p_nom=eb_pnom, p_nom_extendable=eb_ext, p_nom_min=eb_pnom,
              p_nom_max=float(eb.get("p_nom_max_mw", np.inf)) if eb_ext else np.inf,
              marginal_cost=elb_vom + el_tax, capital_cost=eb_cap)
        # Stor-VP: AC → heat, COP (p_nom i MW_el).  MC = vom + elskatt (per MWh el).
        # Extendable (costs.heat_pump) → capex per MW_el = ann(€/W_th)×COP; golv = dagens MW.
        hp       = ccfg.get("heat_pump", {})
        hp_ext   = bool(hp.get("extendable", False))
        hp_pnom  = float(zc.get("hp_el_mw", 0.0))
        hp_cap   = (_annualized_cost(hp["overnight_eur_per_w"], hp["lifetime_years"], r,
                    hp.get("fom_fraction", fom_g)) * zone_cop * n_years) if (hp_ext and hp) else 0.0
        n.add("Link", f"{zone} heat hp", bus0=zone, bus1=hb, carrier="heat hp",
              efficiency=zone_cop, p_nom=hp_pnom, p_nom_extendable=hp_ext, p_nom_min=hp_pnom,
              p_nom_max=float(hp.get("p_nom_max_mw", np.inf)) if hp_ext else np.inf,
              marginal_cost=hp_vom + el_tax, capital_cost=hp_cap)
        # Must-run avfall/restvärme/rökgaskond (~0 MC): levererar zone_mr × behovet varje
        # timme (alltid först i meritordningen) → avlastar el-/bio-behovet och AC-lasten.
        if zone_mr > 0:
            pu = (zone_mr * dh / peak).clip(lower=0.0, upper=1.0) if peak > 0 else 0.0
            n.add("Generator", f"{zone} heat mustrun", bus=hb, carrier="heat mustrun",
                  p_nom=peak, p_min_pu=pu, p_max_pu=pu, marginal_cost=mr_vom)
        # Bio/KVV dispatchbar grundförsörjning. Om zonen har KVV-config (chp) ger
        # bakpress-KVV-länken (_add_chp) värmen i stället → hoppa över heat-only-gen.
        if not zc.get("chp"):
            bo     = ccfg.get("heat_bio", {})
            bo_ext = bool(bo.get("extendable", False))
            bo_cap = (_annualized_cost(bo["overnight_eur_per_w"], bo["lifetime_years"], r,
                      bo.get("fom_fraction", fom_g)) * n_years) if (bo_ext and bo) else 0.0
            n.add("Generator", f"{zone} heat chp", bus=hb, carrier="heat chp",
                  p_nom=0.0 if bo_ext else peak * 1.2, p_nom_extendable=bo_ext, p_nom_min=0.0,
                  p_nom_max=float(bo.get("p_nom_max_mw", np.inf)) if bo_ext else np.inf,
                  marginal_cost=bio_vom, capital_cost=bo_cap)
        # Slack (omött värme) för feasibility
        n.add("Generator", f"{zone} heat slack", bus=hb, carrier="heat slack",
              p_nom=1e6, marginal_cost=mc_slack)
        print(f"  → Heat {zone}: FV-behov topp {peak:.0f} MW_th, VP {zc.get('hp_el_mw',0):.0f} "
              f"MW_el (COP {zone_cop:g}), el-panna {zc.get('elboiler_mw',0):.0f} MW, must-run "
              f"{zone_mr:.0%}, bio MC {bio_vom}, lager {e_store:.0f} MWh"
              f"{' (store_gwh)' if 'store_gwh' in zc else f' ({st_hours:g}h×topp)'}")


def _add_chp(n: pypsa.Network, cfg: dict, heat_demand: dict,
             r: float = 0.06, n_years: float = 1.0) -> None:
    """Bakpress-KVV per zon (för zoner med heat.zones[z].chp):

        Bus(chp fuel) ─Generator(bränsle, MC=fuel_vom)─┐
                                                       └─Link(KVV)─► bus1=AC (el, η_el)
                                                                   └► bus2=heat (värme, η_heat)

    Fast el:värme-ratio (bakpress). Optimeraren kör KVV när elpris·η_el + värme-skuggpris·
    η_heat − fuel_vom > 0. Ersätter zonens heat-only `heat chp` (värmen kommer nu från bus2)
    OCH zonens must-run-termisk-el (reducerad i _add_thermal med share_of_thermal) — KVV-elen
    blir endogen. p_nom (MW_fuel) dimensioneras så KVV ensam kan täcka det dispatchbara
    värmebehovet (behov − must-run).
    """
    hcfg = cfg.get("heat") or {}
    if not hcfg.get("enabled") or not heat_demand:
        return
    zcfg = hcfg.get("zones") or {}
    cc      = cfg["costs"].get("heat_chp", {})
    chp_ext = bool(cc.get("extendable", False))
    fom_g   = cfg["costs"].get("fom_fraction", 0.02)
    if "chp fuel" not in n.carriers.index:
        n.add("Carrier", "chp fuel")
    for zone, dh in heat_demand.items():
        chp = (zcfg.get(zone, {}) or {}).get("chp")
        if not chp or zone not in n.buses.index:
            continue
        eta_el   = float(chp.get("eta_el", 0.28))
        eta_heat = float(chp.get("eta_heat", 0.52))
        fuel_vom = float(chp.get("fuel_vom", 22.0))
        mr       = float(zcfg[zone].get("mustrun_share", hcfg.get("mustrun_share", 0.0)))
        peak     = float(dh.max())
        # Bränsle-p_nom så η_heat × p_nom ≥ dispatchbart värmebehov (behov − must-run)
        p_nom_fuel = (peak * (1.0 - mr)) / eta_heat * 1.2 if eta_heat > 0 else 0.0
        fbus = f"{zone} chp fuel"
        n.add("Bus", fbus, carrier="chp fuel")
        n.add("Generator", f"{zone} chp fuel", bus=fbus, carrier="chp fuel",
              p_nom=1e7, marginal_cost=fuel_vom)
        # Extendable (costs.heat_chp): capex per MW_fuel = ann(€/W_el)×η_el; golv 0.
        chp_cap = (_annualized_cost(cc["overnight_eur_per_w"], cc["lifetime_years"], r,
                   cc.get("fom_fraction", fom_g)) * eta_el * n_years) if (chp_ext and cc) else 0.0
        n.add("Link", f"{zone} chp",
              bus0=fbus, bus1=zone, bus2=f"{zone} heat", carrier="heat chp",
              efficiency=eta_el, efficiency2=eta_heat,
              p_nom=0.0 if chp_ext else p_nom_fuel, p_nom_extendable=chp_ext, p_nom_min=0.0,
              p_nom_max=float(cc.get("p_nom_max_mw", np.inf)) if chp_ext else np.inf,
              capital_cost=chp_cap)
        print(f"  → CHP {zone}: bakpress η_el={eta_el} η_heat={eta_heat} (c={eta_el/eta_heat:.2f}), "
              f"fuel MC {fuel_vom}, p_nom {'ext≤'+str(int(cc.get('p_nom_max_mw',0)))+' MW_fuel' if chp_ext else f'{p_nom_fuel:.0f} MW_fuel'} "
              f"(→ ≤{p_nom_fuel*eta_el:.0f} MW_el / ≤{p_nom_fuel*eta_heat:.0f} MW_th vid fast)")


def _add_chp_fixed(n: pypsa.Network, cfg: dict, thermal_profile: pd.DataFrame) -> None:
    """Fast exogen KVV-EL utan värmebuss (--chp-fixed-gw). Snabbalternativ till hela
    värmesektorn (_add_heat/_add_chp): ingen heat-buss, lager, VP eller el-panna byggs.

    _add_thermal har redan reducerat must-run-termiken med share_of_thermal för KVV-
    zonerna (heat.enabled=True). Den borttagna termik-FORMEN (säsongsvintertung) återförs
    här som MUST-RUN KVV-generator på AC-bussen, omskalad så att totala toppen över alla
    KVV-zoner = chp_fixed_gw. Per-zon-vikt = borttagen termik (share_of_thermal × termik-
    topp) → FI:s låga share ger litet FI-bidrag (industri-KVV stannar i must-run-termiken).

    KVV är värmestyrd (går vintern oavsett elpris) → must-run, ej dispatchbar. Utan
    värmebussens skuggpris skulle en ren el-MC (fuel_vom/η_el ≈ 88 €/MWh) annars knappt köra.
    """
    hcfg = cfg.get("heat") or {}
    gw = hcfg.get("chp_fixed_gw")
    if not gw:
        return
    target_mw = float(gw) * 1e3
    zcfg = hcfg.get("zones") or {}
    # Borttagen termik-form per KVV-zon (share_of_thermal × termik-profil)
    removed = {}
    for zone, zc in zcfg.items():
        chp = (zc or {}).get("chp")
        if not chp or zone not in n.buses.index or zone not in thermal_profile.columns:
            continue
        share = float(chp.get("share_of_thermal", 1.0))
        prof = thermal_profile[zone].clip(lower=0) * share
        if float(prof.max()) > 0:
            removed[zone] = (prof, chp)
    tot_peak = sum(float(p.max()) for p, _ in removed.values())
    if tot_peak <= 0:
        print(f"  → CHP-fast: ingen reducerad termik att återföra (chp_fixed_gw={gw}) — hoppar över")
        return
    scale = target_mw / tot_peak     # omskala totala toppen → chp_fixed_gw
    if "heat chp" not in n.carriers.index:
        n.add("Carrier", "heat chp")
    dt_h = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600 if len(n.snapshots) > 1 else 1.0
    e_tot = 0.0
    for zone, (prof, chp) in removed.items():
        p = prof * scale
        p_nom = float(p.max())
        pu = (p / p_nom).clip(0, 1)
        n.add("Generator", f"{zone} chp", bus=zone, carrier="heat chp",
              p_nom=p_nom, p_nom_extendable=False, p_min_pu=pu, p_max_pu=pu,
              marginal_cost=float(chp.get("fuel_vom", 22.0)))
        e_z = float(p.sum() * dt_h / 1e6)
        e_tot += e_z
        print(f"  → CHP-fast {zone}: must-run topp {p_nom:.0f} MW_el, {e_z:.1f} TWh/period "
              f"(share {chp.get('share_of_thermal',1.0):g}, MC {chp.get('fuel_vom',22.0)})")
    print(f"  → CHP-fast TOTAL: {target_mw/1e3:.1f} GW topp / {e_tot:.1f} TWh (skala ×{scale:.2f} på borttagen termik)")


def _add_ev(n: pypsa.Network, cfg: dict, ev_profiles, ev_overrides,
            snapshots, dt_h: float, n_years: float) -> None:
    """Fordonsladdning per zon och fordonsklass. Två lägen (cfg['ev']['mode']):

      svk   — SvK-formulering: förbrukningen delas flex/oflex (flex_fraction). Oflexibel
              halva = fast AC-last (ladda-direkt-form). Flexibel halva = SvK-reservoar
              (batteri 12 GWh/flex-TWh, laddtak = batteri/charge_hours, fri tidsförflyttning).
      fleet — gammal per-fordon-modell (batteri/laddeffekt ur battery_kwh/charger_kw,
              hela flottan flexibel, hourly avail/minsoc-profiler).

    Bara smart laddning (ingen V2G). Antal fordon per zon från ev_overrides (CLI/scenario);
    E_tot = antal × annual_mwh. Flottan är exogen → Store ej extendable."""
    ecfg = cfg.get("ev") or {}
    if not ecfg.get("enabled") or ev_profiles is None or not ev_overrides:
        return
    if (ecfg.get("mode") or "svk") == "fleet":
        _add_ev_fleet(n, ecfg, ev_profiles, ev_overrides, snapshots, dt_h, n_years)
        return

    classes   = ecfg.get("classes") or {}
    mc_slack  = float(ecfg.get("slack_eur_per_mwh", MC_SLACK))
    flex_frac = float(ecfg.get("flex_fraction", 0.5))
    gwh_twh   = float(ecfg.get("flex_battery_gwh_per_twh", 12.0))
    chg_h     = float(ecfg.get("charge_hours", 10.0))

    for car in ("EV battery", "EV charger", "EV slack", "EV inflex"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    def _prof(col):
        return ev_profiles[col].reindex(snapshots).ffill().fillna(0.0)

    for zone, counts in ev_overrides.items():
        if zone not in n.buses.index:
            print(f"  Varning: EV-zon {zone} saknar AC-buss — hoppar över")
            continue
        for c, klass in classes.items():
            n_veh = float(counts.get(c, 0))
            if n_veh <= 0:
                continue
            e_tot    = n_veh * float(klass["annual_mwh"]) / 1e6     # TWh/år total konsumtion (grid-side)
            e_flex   = e_tot * flex_frac
            e_inflex = e_tot * (1.0 - flex_frac)

            # --- Oflexibel halva: fast AC-last, ladda-direkt-form (körprofil m. bryggad middag) ---
            ishape = _prof(f"{c}_inflex").clip(lower=0)
            ienerg = float(ishape.sum() * dt_h / 1e6 / n_years)    # TWh av råformen
            inflex = ishape * (e_inflex / ienerg) if ienerg > 0 else ishape * 0.0   # MW
            n.add("Load", f"{zone} EV {c} inflex", bus=zone, carrier="EV inflex", p_set=inflex)

            # --- Flexibel halva: SvK-reservoar (fri tidsförflyttning, batteri + 10h-laddtak) ---
            if e_flex <= 0:
                continue
            e_nom = gwh_twh * e_flex * 1e3        # GWh/flex-TWh × TWh × 1000 → MWh batteri
            p_chg = e_nom / chg_h                 # MW max laddeffekt (SvK 10h)
            drive = _prof(f"{c}_drive").clip(lower=0)
            denrg = float(drive.sum() * dt_h / 1e6 / n_years)
            drive = drive * (e_flex / denrg) if denrg > 0 else drive * 0.0

            evbus = f"{zone} EV {c}"
            n.add("Bus", evbus, carrier="EV battery")
            # SvK räknar grid-side förbrukning → η=1 (laddförluster ingår i annual_mwh);
            # ingen e_min_pu/avail → SOC fri [0,1], laddning flyttas till billigaste timmarna.
            n.add("Store", f"{zone} EV {c} store", bus=evbus, carrier="EV battery",
                  e_nom=e_nom, e_cyclic=True)
            n.add("Link", f"{zone} EV {c} charger", bus0=zone, bus1=evbus,
                  carrier="EV charger", efficiency=1.0, p_nom=p_chg, marginal_cost=0.01)
            n.add("Load", f"{zone} EV {c} drive", bus=evbus, p_set=drive)
            n.add("Generator", f"{zone} EV {c} slack", bus=evbus, carrier="EV slack",
                  p_nom=1e6, marginal_cost=mc_slack)
            print(f"  → EV {zone}/{c} (SvK): {n_veh:.0f} fordon, E={e_tot*1e3:.0f} GWh/år, "
                  f"flex {flex_frac:.0%} → batteri {e_nom/1e3:.0f} GWh, laddtak {p_chg:.0f} MW "
                  f"({chg_h:.0f}h); oflex-last {e_inflex*1e3:.0f} GWh/år")


def _add_ev_fleet(n: pypsa.Network, ecfg: dict, ev_profiles, ev_overrides,
                  snapshots, dt_h: float, n_years: float) -> None:
    """Gammal per-fordon-modell (mode: fleet). Hela flottan flexibel, batteri/laddeffekt
    ur battery_kwh/charger_kw, hourly avail (Link p_max_pu) + minsoc (Store e_min_pu)."""
    classes  = ecfg.get("classes") or {}
    mc_slack = float(ecfg.get("slack_eur_per_mwh", MC_SLACK))

    for car in ("EV battery", "EV charger", "EV slack"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    def _prof(col):
        return ev_profiles[col].reindex(snapshots).ffill().fillna(0.0)

    for zone, counts in ev_overrides.items():
        if zone not in n.buses.index:
            print(f"  Varning: EV-zon {zone} saknar AC-buss — hoppar över")
            continue
        for c, klass in classes.items():
            n_veh = float(counts.get(c, 0))
            if n_veh <= 0:
                continue
            eff   = float(klass.get("charge_efficiency", 0.9))
            e_nom = n_veh * float(klass["battery_kwh"]) / 1e3    # MWh (flottbatteri)
            p_chg = n_veh * float(klass["charger_kw"])  / 1e3    # MW (max samtidig laddeffekt)

            drive = _prof(f"{c}_drive").clip(lower=0)
            avail = _prof(f"{c}_avail").clip(0, 1)
            msoc  = _prof(f"{c}_minsoc").clip(0, 1)
            ann   = float(drive.sum() * dt_h / 1e6 / n_years)    # TWh av råformen
            tgt   = n_veh * float(klass["annual_mwh"]) / 1e6     # TWh-mål
            drive = drive * (tgt / ann) if ann > 0 else drive * 0.0

            evbus = f"{zone} EV {c}"
            n.add("Bus", evbus, carrier="EV battery")
            n.add("Store", f"{zone} EV {c} store", bus=evbus, carrier="EV battery",
                  e_nom=e_nom, e_cyclic=True, e_min_pu=msoc)
            n.add("Link", f"{zone} EV {c} charger", bus0=zone, bus1=evbus,
                  carrier="EV charger", efficiency=eff,
                  p_nom=p_chg, p_max_pu=avail, marginal_cost=0.01)
            n.add("Load", f"{zone} EV {c} drive", bus=evbus, p_set=drive)
            n.add("Generator", f"{zone} EV {c} slack", bus=evbus, carrier="EV slack",
                  p_nom=1e6, marginal_cost=mc_slack)
            print(f"  → EV {zone}/{c}: {n_veh:.0f} fordon, batteri {e_nom:.0f} MWh, "
                  f"laddeffekt {p_chg:.0f} MW (η={eff}), körbehov {tgt*1e3:.0f} GWh/år")


def _add_nuclear(
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
    cap_cost = _annualized_cost(
        tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r, tcfg.get("fom_fraction", fom_fraction)
    ) * n_years

    # Expansionsläge (--add-nuclear angivet → synthetic_nuclear['active']): befintlig
    # flotta blir FAST (ej extendable) + SYNTETISK profil; ny kärnkraft expanderas
    # separat i _add_extra_nuclear. Dispatchläge: faktisk profil, extendable per config.
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
            sc    = syn_existing[zone]
            p_max = availability_timeseries(
                syn_params, snapshots, int(sc["n_reactors"]), seed=int(sc["seed"]),
            )
            print(f"  → befintlig kärnkraft {zone}: {p_nom_existing:.0f} MW, "
                  f"{int(sc['n_reactors'])} reaktorer synth (seed {sc['seed']}), "
                  f"realiserad CF={p_max.mean():.3f}")
            p_min = (p_max * min_frac).clip(lower=0)
        else:
            p_max = nuclear_profile[zone]
            p_min = (p_max * NUCLEAR_MIN_FRACTION).clip(lower=0)

        p_nom_max = tcfg.get("p_nom_max_mw", np.inf)
        n.add(
            "Generator", f"{zone} nuclear",
            bus=zone,
            carrier="nuclear",
            p_nom=p_nom_existing,
            p_nom_min=p_nom_existing,
            p_nom_max=p_nom_max,
            p_nom_extendable=extendable,
            p_max_pu=p_max,
            p_min_pu=p_min,
            marginal_cost=mc,
            capital_cost=cap_cost if extendable else 0.0,
        )


def _add_vre(
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
            cap_cost   = _annualized_cost(
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


def _add_gas(
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
    cap_cost   = _annualized_cost(
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



def _add_market_connections(
    n:             pypsa.Network,
    cfg:           dict,
    market_prices: Dict[str, pd.Series],
) -> None:
    """Import/export-ventiler mot angränsande marknader.

    Standard: en Generator per kabel med p_min_pu=-1:
      p > 0 → import (zonen köper, kostnaden = price(t) × p)
      p < 0 → export (zonen säljer, intäkten = price(t) × |p|)
    Generatornamn: "<Nord zon> <motpart>", t.ex. "DK GB".
    price_bzn anger vilken kolumn i market_prices som används.

    Om cfg['market_elasticity']['enabled'] → pris-elastisk trappa istället
    (se _add_market_staircase).
    """
    me = cfg.get("market_elasticity") or {}
    if me.get("enabled", False):
        _add_market_staircase(n, cfg, market_prices, me)
        return
    for name, zone, p_nom, price_bzn in cfg.get("market_connections", []):
        mc = market_prices[price_bzn]
        n.add(
            "Generator", name,
            bus=zone,
            carrier="market",
            p_nom=p_nom,
            p_min_pu=-1.0,
            p_max_pu=1.0,
            marginal_cost=mc,
        )


def _add_market_staircase(
    n:             pypsa.Network,
    cfg:           dict,
    market_prices: Dict[str, pd.Series],
    me:            dict,
) -> None:
    """Pris-elastisk kontinentgräns som trappa (utbudselasticitet vid gränsen).

    Ersätter den enda ventilen med N import-tiers + N export-tiers per kabel,
    var och en med NTC/N kapacitet:
      import-tier k: p ≥ 0, marginal_cost = pris(t) + S_import · offsetₖ  (stigande)
      export-tier k: p ≤ 0, marginal_cost = pris(t) − S_export · offsetₖ  (fallande)
    Stora nordiska flöden flyttar då det effektiva gränspriset: import blir dyrare,
    export-intäkt sjunker med volymen. Eftersom import-pris > export-pris alltid
    körs aldrig båda samtidigt → NTC respekteras per riktning.

    S_export/S_import (€/MWh vid full NTC) och offset_profile från config; S är
    empiriskt förankrat i 2023-2025 prisspread per gräns. Namn: "<kabel> imp/exp k".
    """
    offs    = list(me.get("offset_profile", [0.1667, 0.5, 0.8333]))
    n_tiers = int(me.get("n_tiers", len(offs)))
    offs    = offs[:n_tiers] if n_tiers <= len(offs) else offs
    spreads = me.get("spread_eur_per_mwh") or {}
    n_steps = len(offs)
    for name, zone, p_nom, price_bzn in cfg.get("market_connections", []):
        mc       = market_prices[price_bzn]
        s        = spreads.get(name, {}) or {}
        s_exp    = float(s.get("export", 0.0))
        s_imp    = float(s.get("import", 0.0))
        tier_cap = p_nom / n_steps
        for k, off in enumerate(offs, 1):
            n.add(
                "Generator", f"{name} imp{k}",
                bus=zone, carrier="market",
                p_nom=tier_cap, p_min_pu=0.0, p_max_pu=1.0,
                marginal_cost=mc + s_imp * off,
            )
            n.add(
                "Generator", f"{name} exp{k}",
                bus=zone, carrier="market",
                p_nom=tier_cap, p_min_pu=-1.0, p_max_pu=0.0,
                marginal_cost=mc - s_exp * off,
            )
    print(f"  → pris-elastisk gräns (trappa, {n_steps} steg/riktning) för "
          f"{len(cfg.get('market_connections', []))} kablar")


# ---------------------------------------------------------------------------
# Extra LP-constraints
# ---------------------------------------------------------------------------

def oc_budget_constraint(cfg: dict, zones, budget_eur: float, equality: bool = False):
    """Returnerar en extra_functionality-callback som begränsar overnight-kostnaden
    för tillkommande VRE + batteri i givna zoner:

        Σ overnight_i × (p_nom_i − existing_i)  ≤  budget_eur   (equality=False)
        Σ overnight_i × (p_nom_i − existing_i)  =  budget_eur   (equality=True)

    equality=True TVINGAR fram exakt budget_eur i utgift (kombineras med kapital=0
    i målfunktionen) → ren dispatcheffekt, inte lönsamhetsval.

    Gäller extendable wind_onshore/wind_offshore/solar (existing = p_nom_min,
    den fasta golvkapaciteten) samt extendable batterier (existing = 0).
    Overnight för VRE = overnight_eur_per_w × 1e6 (€/MW); för batteri =
    (power_eur_per_kw + max_hours × energy_eur_per_kwh) × 1e3 (€/MW).
    """
    VRE   = {"wind_onshore", "wind_offshore", "solar"}
    zones = set(zones)

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        m     = n.model
        gp    = m.variables["Generator-p_nom"]
        gdim  = gp.dims[0]
        terms = []
        rhs   = float(budget_eur)

        gext = n.generators[n.generators.p_nom_extendable]
        for name, gen in gext.iterrows():
            if gen.bus in zones and gen.carrier in VRE:
                oc = cfg["costs"][gen.carrier]["overnight_eur_per_w"] * 1e6
                terms.append(oc * gp.sel({gdim: name}))
                rhs += oc * float(gen.p_nom_min)        # flytta existing till RHS

        if "StorageUnit-p_nom" in m.variables:
            sp   = m.variables["StorageUnit-p_nom"]
            sdim = sp.dims[0]
            bc   = cfg["costs"]["battery"]
            sext = n.storage_units[n.storage_units.p_nom_extendable]
            for name, su in sext.iterrows():
                if su.bus in zones and su.carrier == "battery":
                    oc = (bc["power_eur_per_kw"]
                          + su.max_hours * bc["energy_eur_per_kwh"]) * 1e3
                    terms.append(oc * sp.sel({sdim: name}))
                    rhs += oc * float(su.p_nom_min)     # = 0

        if not terms:
            print("  Varning: OC-budget — inga extendable VRE/batteri i zonerna, "
                  "ingen constraint lades till")
            return

        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        if equality:
            m.add_constraints(expr == rhs, name="vre_battery_oc_budget")
        else:
            m.add_constraints(expr <= rhs, name="vre_battery_oc_budget")
        op = "=" if equality else "≤"
        print(f"  → OC-budget aktiv: Σ overnight×Δp_nom {op} {budget_eur/1e9:.2f} mdr€ "
              f"över {len(terms)} enheter i {', '.join(sorted(zones))}")

    return _extra_functionality


def hydro_soc_initial_constraint(cfg: dict):
    """Returnerar en extra_functionality-callback som fixerar hydro SOC vid t=0.

    cyclic_state_of_charge=True ger SOC[0]==SOC[-1].
    Denna constraint lägger till SOC[0]==target, så att start=slut=target.

    Värdet hämtas från zones.yaml: hydro_soc_initial (fraktion av max kapacitet).
    """
    targets = {}  # {"{zone} hydro": target_mwh}
    for zone, zcfg in cfg["zones"].items():
        frac = zcfg.get("hydro_soc_initial", None)
        if frac is None:
            continue
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0:
            continue
        targets[f"{zone} hydro"] = frac * p_nom * max_h

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        if not targets:
            return
        m = n.model
        soc = m.variables["StorageUnit-state_of_charge"]
        t0 = snapshots[0]
        for su_name, target_mwh in targets.items():
            m.add_constraints(
                soc.sel(name=su_name, snapshot=t0) == target_mwh,
                name=f"soc_initial-{su_name}",
            )

    return _extra_functionality


def hydro_soc_terminal_pin_constraint(cfg: dict, end_fracs: dict):
    """Returnerar en extra_functionality-callback som pinnar slut-SOC till
    end_frac × kapacitet per zon (icke-cyklisk drift).

    Zoner utan angivet end_frac pinnas till config-mål (hydro_soc_initial), så att
    de förblir start=slut (cyklikt). Avsedd att kombineras med soc_initial_override
    så att BÅDA ändpunkterna låses till FAKTISKA reservoarnivåer (kalibrering mot
    observerad fyllnadsgrad). end_fracs: dict {zon: slut-fraktion av kapacitet}.
    """
    targets = {}
    for zone, zcfg in cfg["zones"].items():
        frac = zcfg.get("hydro_soc_initial", None)
        if frac is None:
            continue
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0:
            continue
        cap = p_nom * max_h
        targets[f"{zone} hydro"] = end_fracs.get(zone, frac) * cap

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        if not targets:
            return
        m = n.model
        soc = m.variables["StorageUnit-state_of_charge"]
        tT = snapshots[-1]
        for su_name, term_mwh in targets.items():
            m.add_constraints(
                soc.sel(name=su_name, snapshot=tT) == term_mwh,
                name=f"soc_terminal_pin-{su_name}",
            )

    return _extra_functionality
