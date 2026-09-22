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
