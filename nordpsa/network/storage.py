"""Batterier (fasta, fria: dagens flotta eller demand-scenariots)."""

import pypsa


def add_batteries(n: pypsa.Network, batteries: list | None, ccfg: dict) -> None:
    """Lägger till batterier som StorageUnit (carrier 'battery').

    batteries: lista av (zon, p_nom_mw, max_hours). Alla är FASTA och fria (sunk,
    capital_cost 0): dagens baseline-batterier eller demand-scenariots exogena flotta.
    Round-trip ~90% (0.95×0.95), cyklisk SOC, litet marginalkostnad för att bryta
    degeneracy. Inget inflöde.
    """
    if not batteries:
        return
    bc = ccfg["battery"]
    for zone, p_nom, max_h in batteries:
        if zone not in n.buses.index:
            print(f"  Varning: batteri-zon {zone} saknas — hoppar över")
            continue
        overnight_mw = (bc["power_eur_per_kw"] + max_h * bc["energy_eur_per_kwh"]) * 1e3
        name = f"{zone} battery"
        n.add(
            "StorageUnit", name,
            bus=zone,
            carrier="battery",
            p_nom=p_nom,
            p_nom_extendable=False,
            p_nom_min=p_nom,
            p_nom_max=float("inf"),
            max_hours=max_h,
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=True,
            marginal_cost=0.01,
            capital_cost=0.0,
        )
        print(f"  → batteri {name}: fast {p_nom:.0f} MW (fri) / {max_h:.0f}h, "
              f"overnight {overnight_mw/1e3:.0f} €/kW, kapital=0 (sunk)")


def add_investable_batteries(n: pypsa.Network, zones: list, ccfg: dict, r: float,
                             n_years: float, hours: float, extendable: bool,
                             cost_scale: float = 1.0, gfm_extra: float | None = None) -> None:
    """battery.endogenous: två investerbara batterityper per zon, samma lagringstid.

      '{zon} battery exp'  carrier 'battery'      nätföljande (GFL): belastar nätstyrkan
      '{zon} battery gfm'  carrier 'battery_gfm'  nätbildande (GFM): ger tröghet och
                                                  felström, belastar inte nätstyrkan

    GFM kostar gfm_extra_eur_per_kw mer per kW (gfm_extra = None: zones.yaml); i övrigt är
    de identiska, så LP:t väljer GFM bara för dess stabilitetsvärde. Med merkostnad 0 byggs
    BARA GFM: GFL vore dominerad, och två identiskt prissatta typer gör LP:t degenererat.
    cost_scale skalar effekt- och energidelen (inte GFM-tillägget).
    capital_cost = overnight·(CRF + fom)·n_years på HELA p_nom_opt (byggs från noll). I dispatch (extendable=False) byggs de med p_nom 0
    och fryses till källkörningens p_nom_opt av freeze_capacities_from.
    """
    from nordpsa.network.costs import crf
    bc = ccfg["battery"]
    ann = crf(int(bc["lifetime_years"]), r) + float(bc.get("fom_fraction", 0.025))
    base = cost_scale * (float(bc["power_eur_per_kw"]) + hours * float(bc["energy_eur_per_kwh"]))
    extra_gfm = float(bc["gfm_extra_eur_per_kw"] if gfm_extra is None else gfm_extra)
    kinds = [("exp", "battery", 0.0)] if extra_gfm > 0 else []
    for kind, carrier, extra in kinds + [("gfm", "battery_gfm", extra_gfm)]:
        oc_mw = (base + extra) * 1e3
        for zone in zones:
            n.add(
                "StorageUnit", f"{zone} battery {kind}",
                bus=zone,
                carrier=carrier,
                p_nom=0.0,
                p_nom_extendable=extendable,
                p_nom_min=0.0,
                p_nom_max=float(bc["p_nom_max_mw"]),
                max_hours=float(hours),
                efficiency_store=0.95,
                efficiency_dispatch=0.95,
                cyclic_state_of_charge=True,
                marginal_cost=0.01,
                capital_cost=oc_mw * ann * n_years,
            )
        print(f"  → investerbara batterier '{kind}' ({carrier}), {hours:g}h, "
              f"{oc_mw/1e3:.0f} €/kW, {oc_mw*ann/1e3:.1f} k€/MW/år"
              + ("" if extendable else "  [dispatch: fryses till källan]"))


def add_fixed_gfm_batteries(n: pypsa.Network, mw_by_zone: dict, ccfg: dict, r: float,
                            n_years: float, hours: float, gfm_extra: float | None = None) -> None:
    """battery.gfm_fixed_mw: EXOGENA nätbildande batterier '{zon} battery gfm fixed'.

    Byggs i alla lägen, ovanpå övriga batterier, och fryses aldrig. Samma teknik som
    add_investable_batteries, men alltid till FULLT pris: battery.cost_scale gäller bara det
    modellen själv bygger, inte en exogen flotta. Årskostnaden (i objective_constant) skrivs
    ut för redovisningen.
    """
    from nordpsa.network.costs import crf
    bc = ccfg["battery"]
    ann = crf(int(bc["lifetime_years"]), r) + float(bc.get("fom_fraction", 0.025))
    extra = float(bc["gfm_extra_eur_per_kw"] if gfm_extra is None else gfm_extra)
    oc_mw = (float(bc["power_eur_per_kw"]) + hours * float(bc["energy_eur_per_kwh"]) + extra) * 1e3
    for zone, mw in mw_by_zone.items():
        n.add(
            "StorageUnit", f"{zone} battery gfm fixed",
            bus=zone,
            carrier="battery_gfm",
            p_nom=float(mw),
            p_nom_extendable=False,
            max_hours=float(hours),
            efficiency_store=0.95,
            efficiency_dispatch=0.95,
            cyclic_state_of_charge=True,
            marginal_cost=0.01,
            capital_cost=oc_mw * ann * n_years,
        )
        print(f"  → exogent nätbildande batteri {zone}: {mw:.0f} MW / {hours:g}h, "
              f"kapitalkostnad {mw * oc_mw * ann / 1e6:.1f} M€/år (utanför optimeringen)")
