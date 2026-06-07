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

from nordpsa.hydro import compute_annual_scales, inflow_timeseries, load_nve_inflow, load_nve_ror

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
    normalize_inflow:        bool = False,
    actual_inflow:           bool = False,
    cyclic_soc:              bool = True,
    soc_initial_override:    dict | None = None,
    voll:                    bool = False,
    batteries:               list | None = None,
    extra_nuclear:           list | None = None,
    hydrogen_overrides:      dict | None = None,
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

    zone_prices = {z: market_prices[z] for z in cfg["zones"] if z in market_prices}

    _add_buses(n, cfg)
    _add_links(n, cfg)
    _add_loads(n, load)
    _add_slack(n, cfg, all_zones=voll)
    _add_thermal(n, thermal_profile)
    _add_hydro(n, cfg, hydro_params, snapshots, ccfg, normalize_inflow,
               actual_inflow=actual_inflow,
               cyclic_soc=cyclic_soc, soc_initial_override=soc_initial_override,
               zone_prices=zone_prices)
    _add_nuclear(n, cfg, nuclear_profile, ccfg, r, fom, n_years)
    _add_vre(n, cfg, vre_profiles, vre_noms, ccfg, r, fom, n_years)
    _add_gas(n, cfg, ccfg, r, fom, n_years)
    _add_market_connections(n, cfg, market_prices)
    _add_batteries(n, batteries, ccfg, r, n_years)
    _add_extra_nuclear(n, extra_nuclear, ccfg)
    _add_hydrogen(n, cfg, r, fom, n_years, hydrogen_overrides)

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


def _add_thermal(n: pypsa.Network, thermal_profile: pd.DataFrame) -> None:
    """Termisk must-run som fast Generator (p_min_pu = p_max_pu = profil).

    Dispatch är helt given av data — optimeraren har inget val.
    Zoner utan termisk produktion (max = 0) hoppas över.
    """
    for zone in thermal_profile.columns:
        profile = thermal_profile[zone].clip(lower=0)
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
    normalize_inflow:     bool = False,
    actual_inflow:        bool = False,
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
            annual_scales = compute_annual_scales(zone, params) if normalize_inflow else None
            inflow = inflow_timeseries(params, snapshots, annual_scales=annual_scales)

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


def _add_extra_nuclear(n: pypsa.Network, extra_nuclear: list | None, ccfg: dict) -> None:
    """Lägger till NY kärnkraft (Generator, carrier 'nuclear').

    extra_nuclear: lista av (zon, p_nom_mw, p_min_pu). p_min_pu=1.0 → must-run baslast
    (full effekt varje timme); p_min_pu=0 → dispatchbar (kör när pris > MC, kan backa).
    Namnges '{zon} nuclear_new'. MC = nuclear VOM.
    """
    if not extra_nuclear:
        return
    mc = ccfg["nuclear"]["vom_eur_per_mwh"]
    for zone, p_nom, p_min in extra_nuclear:
        if zone not in n.buses.index:
            print(f"  Varning: kärnkrafts-zon {zone} saknas — hoppar över")
            continue
        n.add(
            "Generator", f"{zone} nuclear_new",
            bus=zone,
            carrier="nuclear",
            p_nom=p_nom,
            p_nom_extendable=False,
            p_max_pu=1.0,
            p_min_pu=p_min,
            marginal_cost=mc,
        )
        mode = "must-run baslast" if p_min >= 0.999 else f"dispatchbar (p_min={p_min})"
        print(f"  → ny kärnkraft {zone}: {p_nom:.0f} MW ({mode}, MC {mc} EUR/MWh)")


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
        st_emax   = float(stc.get("e_nom_max_mwh", 1e7))
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


def _add_nuclear(
    n:               pypsa.Network,
    cfg:             dict,
    nuclear_profile: pd.DataFrame,
    ccfg:            dict,
    r:               float,
    fom_fraction:    float,
    n_years:         float,
) -> None:
    tcfg       = ccfg["nuclear"]
    mc         = tcfg["vom_eur_per_mwh"]
    extendable = tcfg["extendable"]
    cap_cost   = _annualized_cost(
        tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r, fom_fraction
    ) * n_years

    for zone, zcfg in cfg["zones"].items():
        p_nom_existing = zcfg.get("nuclear_p_nom_mw", 0)
        if p_nom_existing == 0 and not extendable:
            continue

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
            cap_cost   = _annualized_cost(
                tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r, fom_fraction
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
        tcfg["overnight_eur_per_w"], tcfg["lifetime_years"], r, fom_fraction
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

    En Generator per kabel med p_min_pu=-1:
      p > 0 → import (zonen köper, kostnaden = price(t) × p)
      p < 0 → export (zonen säljer, intäkten = price(t) × |p|)
    Generatornamn: "<Nord zon> <motpart>", t.ex. "DK GB".
    price_bzn anger vilken kolumn i market_prices som används.
    """
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


def hydro_soc_band_constraint(cfg: dict, low_frac: float, high_frac: float):
    """Returnerar en extra_functionality-callback som tillåter SOC[t0] att flyta
    inom ett band [low_frac, high_frac] × kapacitet istället för en fast nivå.

    Med cyclic_state_of_charge=True gäller SOC[0]==SOC[-1], så bandet styr den
    gemensamma start/slut-nivån. Kapaciteten = config p_nom × max_hours (bevaras
    även när reservoar-p_nom reduceras av run-of-river-uppdelningen).
    """
    bands = {}  # {"{zone} hydro": (low_mwh, high_mwh)}
    for zone, zcfg in cfg["zones"].items():
        if zcfg.get("hydro_soc_initial", None) is None:
            continue
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0:
            continue
        cap = p_nom * max_h
        bands[f"{zone} hydro"] = (low_frac * cap, high_frac * cap)

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        if not bands:
            return
        m = n.model
        soc = m.variables["StorageUnit-state_of_charge"]
        t0 = snapshots[0]
        for su_name, (low_mwh, high_mwh) in bands.items():
            m.add_constraints(
                soc.sel(name=su_name, snapshot=t0) >= low_mwh,
                name=f"soc_band_lo-{su_name}",
            )
            m.add_constraints(
                soc.sel(name=su_name, snapshot=t0) <= high_mwh,
                name=f"soc_band_hi-{su_name}",
            )

    return _extra_functionality


def hydro_soc_terminal_band_constraint(cfg: dict, low_frac: float, high_frac: float):
    """Returnerar en extra_functionality-callback som binder SOC vid SISTA tidssteget
    till ett band [low_frac, high_frac] × kapacitet.

    Avsedd för icke-cyklisk drift (cyclic_state_of_charge=False): startnivån sätts av
    state_of_charge_initial (config hydro_soc_initial), och slutnivån tillåts flyta
    inom bandet — start ≠ slut tillåts. Modellen kan därmed netto-tömma (eller fylla)
    lagret en gång över horisonten, begränsat av bandet.
    """
    bands = {}
    for zone, zcfg in cfg["zones"].items():
        if zcfg.get("hydro_soc_initial", None) is None:
            continue
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0:
            continue
        cap = p_nom * max_h
        bands[f"{zone} hydro"] = (low_frac * cap, high_frac * cap)

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        if not bands:
            return
        m = n.model
        soc = m.variables["StorageUnit-state_of_charge"]
        tT = snapshots[-1]
        for su_name, (low_mwh, high_mwh) in bands.items():
            m.add_constraints(
                soc.sel(name=su_name, snapshot=tT) >= low_mwh,
                name=f"soc_terminal_lo-{su_name}",
            )
            m.add_constraints(
                soc.sel(name=su_name, snapshot=tT) <= high_mwh,
                name=f"soc_terminal_hi-{su_name}",
            )

    return _extra_functionality


def hydro_terminal_value(cfg: dict, lambda_per_zone: Dict[str, float]):
    """Returnerar en extra_functionality-callback som lägger till terminalvärde.

    Lägger till  -λ × SOC[T]  i LP-målfunktionen per hydro-zon.
    Belönar modellen för att hålla vatten vid fönstrets sista tidssteg,
    vilket ger ett realistiskt vattenvärde i rullande horisont-optimering.

    lambda_per_zone: {zone: EUR/MWh} — typiskt observerat marknadspris.
    """
    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        m = n.model
        soc = m.variables["StorageUnit-state_of_charge"]
        t_last = snapshots[-1]
        for zone in cfg["zones"]:
            lam = lambda_per_zone.get(zone, 0.0)
            if lam <= 0.0:
                continue
            su_name = f"{zone} hydro"
            if su_name not in n.storage_units.index:
                continue
            # Addera belöningsterm: hämta befintligt mål, lägg till -λ×SOC[T], sätt om
            term    = -lam * soc.sel(name=su_name, snapshot=t_last)
            new_obj = m.objective.expression + term
            m.add_objective(new_obj, overwrite=True)
    return _extra_functionality


def hydro_annual_production_constraints(
    cfg:               dict,
    annual_production: Dict[str, Dict[int, float]],
) -> Callable:
    """Returnerar en extra_functionality-callback som lägger till LP-caps:

    sum(p_dispatch[zone hydro, t] * dt_h  for t in year)  <=  actual_MWh[zone][year]

    Förhindrar att optimeraren omfördelar vatten mellan år.
    Används av --restricted_yearly_hydro.
    """
    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        if not annual_production:
            return
        m = n.model
        p = m.variables["StorageUnit-p_dispatch"]
        weights = n.snapshot_weightings["generators"]

        for zone, year_limits in annual_production.items():
            su_name = f"{zone} hydro"
            if su_name not in n.storage_units.index:
                continue
            for year, limit_mwh in year_limits.items():
                mask = snapshots.year == year
                if not mask.any():
                    continue
                year_snaps = snapshots[mask]
                # Vikter som DataArray (1D snapshot) — xarray broadcastar mot
                # linopy-variabelns 2D (snapshot × name) utan fel
                w_da = xr.DataArray(
                    weights.loc[year_snaps].values,
                    dims=["snapshot"],
                    coords={"snapshot": year_snaps},
                )
                dispatch = p.sel(name=su_name, snapshot=year_snaps)
                m.add_constraints(
                    (dispatch * w_da).sum() <= limit_mwh,
                    name=f"annual_hydro_cap-{su_name}-{year}",
                )

    return _extra_functionality
