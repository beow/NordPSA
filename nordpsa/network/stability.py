"""Synkronkompensatorer: roterande maskiner utan drivkälla, bara för tröghet och nätstyrka."""

import pypsa

from nordpsa.network.costs import crf


def add_synchronous_condensers(n: pypsa.Network, zones: list, ccfg: dict, r: float,
                               n_years: float, aux_loss_pu: float, extendable: bool) -> None:
    """'{zon} syncon' per zon, carrier 'syncon', p_nom tolkas som MVA.

    p_min_pu = p_max_pu = −aux_loss_pu: hjälpkraftförlusten är en äkta negativ generering
    som prissätts till zonens elpris. Bidraget till E_k och S_k räknas av
    constraints/stability.py ur zones.yaml:stability.tech.syncon. I dispatch
    (extendable=False) byggs de med p_nom 0 och fryses till källkörningens p_nom_opt.
    """
    sc = ccfg["syncon"]
    oc_mva = float(sc["overnight_eur_per_kva"]) * 1e3
    cap_cost = oc_mva * (crf(int(sc["lifetime_years"]), r) + float(sc["fom_fraction"]))
    for zone in zones:
        n.add(
            "Generator", f"{zone} syncon",
            bus=zone,
            carrier="syncon",
            p_nom=0.0,
            p_nom_extendable=extendable,
            p_nom_min=0.0,
            p_nom_max=float(sc["p_nom_max_mw"]),
            p_min_pu=-float(aux_loss_pu),
            p_max_pu=-float(aux_loss_pu),
            marginal_cost=0.0,
            capital_cost=cap_cost * n_years,
        )
    print(f"  → synkronkompensatorer i {len(zones)} zoner: {oc_mva/1e3:.0f} €/kVA, "
          f"{cap_cost:.0f} €/MVA/år, hjälpkraft {aux_loss_pu:.1%}"
          + ("" if extendable else "  [dispatch: fryses till källan]"))


def add_fixed_synchronous_condensers(n: pypsa.Network, mva_by_zone: dict, ccfg: dict, r: float,
                                     n_years: float, aux_loss_pu: float) -> None:
    """syncon.fixed_mva: EXOGENA synkronkompensatorer '{zon} syncon fixed' (fast p_nom = MVA).

    Byggs i alla lägen och fryses aldrig (namnet finns inte i en källkörning). capital_cost
    sätts för redovisningen; årskostnaden skrivs ut, eftersom den inte optimeras.
    """
    sc = ccfg["syncon"]
    cap_cost = (float(sc["overnight_eur_per_kva"]) * 1e3
                * (crf(int(sc["lifetime_years"]), r) + float(sc["fom_fraction"])))
    for zone, mva in mva_by_zone.items():
        n.add(
            "Generator", f"{zone} syncon fixed",
            bus=zone,
            carrier="syncon",
            p_nom=float(mva),
            p_nom_extendable=False,
            p_min_pu=-float(aux_loss_pu),
            p_max_pu=-float(aux_loss_pu),
            marginal_cost=0.0,
            capital_cost=cap_cost * n_years,
        )
        print(f"  → exogen synkronkompensator {zone}: {mva:.0f} MVA, "
              f"kapitalkostnad {mva * cap_cost / 1e6:.1f} M€/år (utanför optimeringen), "
              f"hjälpkraft {mva * aux_loss_pu:.0f} MW")
