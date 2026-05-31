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
            spill_cost=50.0,       # högt spillkostnad bryter LP-degeneracy (undviker artefakt-spill)
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
