"""build_network: sätter ihop hela PyPSA-nätverket ur config och indata."""
from typing import Dict

import pandas as pd
import pypsa

from nordpsa.network.core import add_buses, add_links, add_loads, add_slack
from nordpsa.network.dsr import add_industrial_dsr
from nordpsa.network.ev import add_ev
from nordpsa.network.generation import add_extra_nuclear, add_fixed_nuclear, add_gas, add_nuclear, add_thermal, add_vre
from nordpsa.network.heat import add_chp, add_heat, heat_demand_profiles
from nordpsa.network.hydrogen import add_hydrogen
from nordpsa.network.hydropower import add_hydro
from nordpsa.network.market import add_market_connections
from nordpsa.network.stability import add_synchronous_condensers
from nordpsa.network.storage import add_batteries, add_investable_batteries


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
    hydro_mc_override:       Dict[str, pd.Series] | None = None,
    voll:                    float | None = None,
    batteries:               list | None = None,
    extra_nuclear:           list | None = None,
    synthetic_nuclear:       dict | None = None,
    hydrogen_overrides:      dict | None = None,
    heat_load:               pd.DataFrame | None = None,
    ev_profiles:             pd.DataFrame | None = None,
    ev_overrides:            dict | None = None,
    ror_hifreq:              float = 0.0,
    ror_hifreq_seed:         int = 0,
    ror_hifreq_tau_days:     float = 3.5,
    battery_invest:          dict | None = None,
    syncon:                  dict | None = None,
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
    # (dubbelräkning) innan add_loads. Värmekomponenterna byggs i add_heat (slutet).
    heat_demand = heat_demand_profiles(cfg, heat_load, snapshots, dt_h, n_years)
    if heat_demand:
        load = load.copy()
        hz = cfg["heat"]["zones"]
        for zone, dh in heat_demand.items():
            el_twh = float(hz.get(zone, {}).get("el_input_twh", 0.0))
            dh_e   = float(dh.sum() * dt_h)
            if el_twh > 0 and dh_e > 0 and zone in load.columns:
                load[zone] = load[zone] - dh * (el_twh * 1e6 * n_years / dh_e)

    # Reservoarvattenkraftens marginal_cost: i expansion terminalkurvan längs normalbanan
    # (λ_bas·A(v), se terminal_curve.hydro_mc_from_curve); med frysta kapaciteter platt
    # VOM, och SOC-dualen bär vattenvärdet.
    zone_prices = hydro_mc_override or None

    add_buses(n, cfg)
    add_links(n, cfg)
    add_loads(n, load)
    add_slack(n, cfg, all_zones=(voll is not None), voll_price=voll)
    add_thermal(n, thermal_profile, cfg)
    add_hydro(n, cfg, hydro_params, snapshots, ccfg,
               zone_prices=zone_prices,
               ror_hifreq=ror_hifreq,
               ror_hifreq_seed=ror_hifreq_seed,
               ror_hifreq_tau_days=ror_hifreq_tau_days)
    add_nuclear(n, cfg, nuclear_profile, ccfg, r, fom, n_years,
                 snapshots, synthetic_nuclear)
    add_vre(n, cfg, vre_profiles, vre_noms, ccfg, r, fom, n_years)
    add_gas(n, cfg, ccfg, r, fom, n_years)
    add_market_connections(n, cfg, market_prices)
    add_batteries(n, batteries, ccfg)
    add_extra_nuclear(n, extra_nuclear, ccfg, r, n_years, snapshots,
                       (synthetic_nuclear or {}).get("params"), fom)
    add_fixed_nuclear(n, (synthetic_nuclear or {}).get("fixed"), cfg, r, n_years,
                       snapshots, (synthetic_nuclear or {}).get("params"))
    add_hydrogen(n, cfg, r, n_years, hydrogen_overrides)
    add_industrial_dsr(n, cfg)
    add_heat(n, cfg, heat_demand, r, n_years)
    add_chp(n, cfg, heat_demand, r, n_years)
    add_ev(n, cfg, ev_profiles, ev_overrides, snapshots, dt_h, n_years)
    zones = list(cfg["zones"])
    if battery_invest:        # {hours, extendable}
        add_investable_batteries(n, zones, ccfg, r, n_years, battery_invest["hours"],
                                 battery_invest["extendable"],
                                 battery_invest.get("cost_scale", 1.0),
                                 battery_invest.get("gfm_extra"))
    if syncon:                # {aux_loss_pu, extendable}
        add_synchronous_condensers(n, zones, ccfg, r, n_years, syncon["aux_loss_pu"],
                                   syncon["extendable"])

    return n
