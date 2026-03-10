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
from typing import Dict

import numpy as np
import pandas as pd
import pypsa
import yaml

from nordpsa.hydro import inflow_timeseries

# Load shedding pris (EUR/MWh)
MC_SLACK = 3000.0

# Nuclear load-following: kan gå ned till denna andel av p_max_pu(t)
NUCLEAR_MIN_FRACTION = 0.6


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
    cfg:              dict,
    snapshots:        pd.DatetimeIndex,
    load:             pd.DataFrame,
    vre_profiles:     pd.DataFrame,
    vre_noms:         dict,
    nuclear_profile:  pd.DataFrame,
    thermal_profile:  pd.DataFrame,
    hydro_params:     dict,
    market_price:     pd.Series,
) -> pypsa.Network:
    """
    Bygger och returnerar ett PyPSA Network.

    Termisk produktion modelleras som ett måste-köra Generator-objekt med
    p_min_pu = p_max_pu = faktisk profil. Lasten är oförändrad (bruttolast).

    Alla tidsserier måste ha samma index som `snapshots`.
    """
    n = pypsa.Network()
    n.set_snapshots(snapshots)

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
    dt_h  = (snapshots[1] - snapshots[0]).total_seconds() / 3600
    n_years = len(snapshots) * dt_h / 8760.0

    _add_buses(n, cfg)
    _add_links(n, cfg)
    _add_loads(n, load)
    _add_slack(n, cfg)
    _add_thermal(n, thermal_profile)
    _add_hydro(n, cfg, hydro_params, snapshots, ccfg)
    _add_nuclear(n, cfg, nuclear_profile, ccfg, r, fom, n_years)
    _add_vre(n, cfg, vre_profiles, vre_noms, ccfg, r, fom, n_years)
    _add_gas(n, cfg, ccfg, r, fom, n_years)
    _add_market_connections(n, cfg, market_price)

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


def _add_slack(n: pypsa.Network, cfg: dict) -> None:
    """Load shedding — bara för zoner utan marknadsanslutning.

    Zoner med marknadsventil (p_min_pu=-1 generator) klarar sig utan slack
    eftersom marknaden agerar säkerhetsventil. Slack behålls för isolerade
    zoner för att garantera LP-feasibilitet.
    """
    market_zones = {z for z, *_ in cfg.get("market_connections", [])}
    for zone in cfg["zones"]:
        if zone in market_zones:
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


def _add_hydro(
    n:            pypsa.Network,
    cfg:          dict,
    hydro_params: dict,
    snapshots:    pd.DatetimeIndex,
    ccfg:         dict,
) -> None:
    mc = ccfg["hydro"]["vom_eur_per_mwh"]
    for zone, zcfg in cfg["zones"].items():
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0 or zone not in hydro_params:
            continue

        inflow = inflow_timeseries(hydro_params[zone], snapshots)

        n.add(
            "StorageUnit", f"{zone} hydro",
            bus=zone,
            carrier="hydro",
            p_nom=p_nom,
            max_hours=max_h,
            inflow=inflow,
            cyclic_state_of_charge=True,
            spill_cost=0.0,        # tillåt fri spill vid reservoardumpning
            p_min_pu=0.0,          # förbjud pumpning (ej pumpad-lagringshydro)
            efficiency_dispatch=1.0,
            marginal_cost=mc,
        )


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
    n:            pypsa.Network,
    cfg:          dict,
    market_price: pd.Series,
) -> None:
    """Import/export-ventil mot kontinentala marknaden per zon.

    Modelleras som en Generator med p_min_pu=-1:
      p > 0 → import (zonen köper, kostnaden = price(t) × p)
      p < 0 → export (zonen säljer, intäkten = price(t) × |p|)
    marginal_cost är en tidsserie (DE-LU day-ahead) gemensam för alla zoner.
    Fallback till fast pris från config om market_price saknas.
    """
    for zone, p_nom, fallback_price in cfg.get("market_connections", []):
        mc = market_price if market_price is not None else fallback_price
        n.add(
            "Generator", f"{zone} market",
            bus=zone,
            carrier="market",
            p_nom=p_nom,
            p_min_pu=-1.0,
            p_max_pu=1.0,
            marginal_cost=mc,
        )
