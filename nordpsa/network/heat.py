"""Fjärrvärme: värmebuss, ackumulator, el-panna, stor-VP, bio och KVV."""

import numpy as np
import pypsa

from nordpsa.network.core import MC_SLACK
from nordpsa.network.costs import annualized_cost


def heat_demand_profiles(cfg, heat_load, snapshots, dt_h, n_years) -> dict:
    """FV-värmebehovsprofiler (termiskt MW) per zon = byggnadsvärmeprofilen
    (heat_load.parquet) skalad så zonens årsenergi = dh_demand_twh. Tom dict om
    värmesektorn ej aktiverad eller heat_load saknas."""
    hcfg = cfg.get("heat") or {}
    if not hcfg.get("enabled") or heat_load is None:
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


def add_heat(n: pypsa.Network, cfg: dict, heat_demand: dict,
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
    # ⚠️ PER ZON sedan 2026-08-16 (zc["el_tax_eur_per_mwh"] vinner över den globala).
    # Skatten avgör HELT om elpanna och stor-VP kan konkurrera: elpannan slås av över
    # (bio_vom − elb_vom) − skatt, alltså −7,5 €/MWh vid skatt 35 (= aldrig igång
    # opportunistiskt) mot +27,5 utan skatt (= mitt i LMA:s 15-50-intervall). VP:n
    # slås av över (bio_vom − hp_vom)·COP − skatt. Se hydrogen_split-blockets granne
    # i zones.yaml för de zonvisa värdena och deras motivering.
    elb_vom  = float(hcfg.get("elboiler_vom_eur_per_mwh", 0.5))
    hp_vom   = float(hcfg.get("hp_vom_eur_per_mwh", 0.5))
    mr_share = float(hcfg.get("mustrun_share", 0.0))        # avfall/restvärme (~0 MC, must-run)
    mr_vom   = float(hcfg.get("mustrun_vom_eur_per_mwh", 0.0))
    st_hours = float(hcfg.get("store_hours", 6))
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
        zone_tax = float(zc.get("el_tax_eur_per_mwh", el_tax))  # per-zon elskatt
        zone_mr  = float(zc.get("mustrun_share", mr_share))  # per-zon must-run-andel (DK 0.25 ≠ SE 0.41)
        hb   = f"{zone} heat"
        peak = float(dh.max())
        # Lagervolym: per-zon store_gwh om angiven, annars store_hours × topplast.
        e_store = float(zc["store_gwh"]) * 1e3 if "store_gwh" in zc else st_hours * peak
        n.add("Bus", hb, carrier="heat")
        n.add("Load", f"{zone} heat load", bus=hb, p_set=dh)
        # Ackumulator (termiskt lager, fast volym, e_cyclic)
        n.add("Store", f"{zone} heat store", bus=hb, carrier="heat store",
              e_nom=e_store, e_cyclic=True)
        # El-panna: AC → heat, η≈1.  MC = vom + elskatt (per MWh el, dvs på p0).
        # Extendable (costs.heat_elboiler) → capex per MW_el = ann(€/W_th)×η; golv = dagens MW.
        eb       = ccfg.get("heat_elboiler", {})
        eb_ext   = bool(eb.get("extendable", False))
        eb_pnom  = float(zc.get("elboiler_mw", 0.0))
        eb_cap   = (annualized_cost(eb["overnight_eur_per_w"], eb["lifetime_years"], r,
                    eb.get("fom_fraction", fom_g)) * 0.99 * n_years) if (eb_ext and eb) else 0.0
        n.add("Link", f"{zone} heat elboiler", bus0=zone, bus1=hb, carrier="heat elboiler",
              efficiency=0.99, p_nom=eb_pnom, p_nom_extendable=eb_ext, p_nom_min=eb_pnom,
              p_nom_max=float(eb.get("p_nom_max_mw", np.inf)) if eb_ext else np.inf,
              marginal_cost=elb_vom + zone_tax, capital_cost=eb_cap)
        # Stor-VP: AC → heat, COP (p_nom i MW_el).  MC = vom + elskatt (per MWh el).
        # Extendable (costs.heat_pump) → capex per MW_el = ann(€/W_th)×COP; golv = dagens MW.
        hp       = ccfg.get("heat_pump", {})
        hp_ext   = bool(hp.get("extendable", False))
        hp_pnom  = float(zc.get("hp_el_mw", 0.0))
        hp_cap   = (annualized_cost(hp["overnight_eur_per_w"], hp["lifetime_years"], r,
                    hp.get("fom_fraction", fom_g)) * zone_cop * n_years) if (hp_ext and hp) else 0.0
        n.add("Link", f"{zone} heat hp", bus0=zone, bus1=hb, carrier="heat hp",
              efficiency=zone_cop, p_nom=hp_pnom, p_nom_extendable=hp_ext, p_nom_min=hp_pnom,
              p_nom_max=float(hp.get("p_nom_max_mw", np.inf)) if hp_ext else np.inf,
              marginal_cost=hp_vom + zone_tax, capital_cost=hp_cap)
        # Must-run avfall/restvärme/rökgaskond (~0 MC): levererar zone_mr × behovet varje
        # timme (alltid först i meritordningen) → avlastar el-/bio-behovet och AC-lasten.
        if zone_mr > 0:
            pu = (zone_mr * dh / peak).clip(lower=0.0, upper=1.0) if peak > 0 else 0.0
            n.add("Generator", f"{zone} heat mustrun", bus=hb, carrier="heat mustrun",
                  p_nom=peak, p_min_pu=pu, p_max_pu=pu, marginal_cost=mr_vom)
        # Bio/KVV dispatchbar grundförsörjning. Om zonen har KVV-config (chp) ger
        # bakpress-KVV-länken (add_chp) värmen i stället → hoppa över heat-only-gen.
        if not zc.get("chp"):
            bo     = ccfg.get("heat_bio", {})
            bo_ext = bool(bo.get("extendable", False))
            bo_cap = (annualized_cost(bo["overnight_eur_per_w"], bo["lifetime_years"], r,
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


def add_chp(n: pypsa.Network, cfg: dict, heat_demand: dict,
             r: float = 0.06, n_years: float = 1.0) -> None:
    """Bakpress-KVV per zon (för zoner med heat.zones[z].chp):

        Bus(chp fuel) ─Generator(bränsle, MC=fuel_vom)─┐
                                                       └─Link(KVV)─► bus1=AC (el, η_el)
                                                                   └► bus2=heat (värme, η_heat)

    Fast el:värme-ratio (bakpress). Optimeraren kör KVV när elpris·η_el + värme-skuggpris·
    η_heat − fuel_vom > 0. Ersätter zonens heat-only `heat chp` (värmen kommer nu från bus2)
    OCH zonens must-run-termisk-el (reducerad i add_thermal med share_of_thermal) — KVV-elen
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
        chp_cap = (annualized_cost(cc["overnight_eur_per_w"], cc["lifetime_years"], r,
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
