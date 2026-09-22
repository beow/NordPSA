"""Bussar, interna NTC-länkar, last och slack (VOLL)."""

import pandas as pd
import pypsa


# Load shedding pris (EUR/MWh)
MC_SLACK = 3000.0


def add_buses(n: pypsa.Network, cfg: dict) -> None:
    for zone in cfg["zones"]:
        n.add("Bus", zone, carrier="AC")


def add_links(n: pypsa.Network, cfg: dict) -> None:
    for z0, z1, p_nom in cfg["links"]:
        n.add(
            "Link", f"{z0}-{z1}",
            bus0=z0, bus1=z1,
            p_nom=p_nom,
            p_min_pu=-1.0,   # bidirektionell
            efficiency=1.0,
            marginal_cost=0.0,
        )


def add_loads(n: pypsa.Network, load: pd.DataFrame) -> None:
    for zone in load.columns:
        n.add("Load", f"{zone} load", bus=zone, p_set=load[zone])


def add_slack(n: pypsa.Network, cfg: dict, all_zones: bool = False,
               voll_price: float | None = None) -> None:
    """Load shedding-generator per zon.

    all_zones=False (standard): bara zoner utan marknadsanslutning.
    all_zones=True (--voll): alla zoner, inklusive de med marknadsanslutning.
      Används som VOLL-mått: slack-dispatch × VOLL = losskostnad i EUR.
      Priser toppas vid VOLL istf att dualvariabler exploderar.
    voll_price: lossprislapp (EUR/MWh). None → MC_SLACK (3000). När satt
      gäller den UNIFORMT i alla slack-zoner (även de utan marknad).
    """
    mc = voll_price if voll_price is not None else MC_SLACK
    market_zones = {zone for _name, zone, *_ in cfg.get("market_connections", [])}
    for zone in cfg["zones"]:
        if not all_zones and zone in market_zones:
            continue
        n.add(
            "Generator", f"{zone} slack",
            bus=zone,
            p_nom=1e6,
            marginal_cost=mc,
            carrier="slack",
        )
