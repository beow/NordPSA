"""Cykliskt SOC-ankare för reservoarerna (start = slut = ankaret)."""

import pandas as pd
import pypsa


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
