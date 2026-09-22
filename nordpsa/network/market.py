"""Kontinentkablar: prisventil mot grannmarknaden, med pristrappa."""
from typing import Dict

import pandas as pd
import pypsa


def add_market_connections(
    n:             pypsa.Network,
    cfg:           dict,
    market_prices: Dict[str, pd.Series],
) -> None:
    """Import/export-ventiler mot angränsande marknader.

    Standard: en Generator per kabel med p_min_pu=-1:
      p > 0 → import (zonen köper, kostnaden = price(t) × p)
      p < 0 → export (zonen säljer, intäkten = price(t) × |p|)
    Generatornamn: "<Nord zon> <motpart>", t.ex. "DK GB".
    price_bzn anger vilken kolumn i market_prices som används.

    Om cfg['market_elasticity']['enabled'] → pris-elastisk trappa istället
    (se _add_market_staircase).
    """
    me = cfg.get("market_elasticity") or {}
    if me.get("enabled", False):
        _add_market_staircase(n, cfg, market_prices, me)
        return
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


def _add_market_staircase(
    n:             pypsa.Network,
    cfg:           dict,
    market_prices: Dict[str, pd.Series],
    me:            dict,
) -> None:
    """Pris-elastisk kontinentgräns som trappa (utbudselasticitet vid gränsen).

    Ersätter den enda ventilen med N import-tiers + N export-tiers per kabel,
    var och en med NTC/N kapacitet:
      import-tier k: p ≥ 0, marginal_cost = pris(t) + S_import · offsetₖ  (stigande)
      export-tier k: p ≤ 0, marginal_cost = pris(t) − S_export · offsetₖ  (fallande)
    Stora nordiska flöden flyttar då det effektiva gränspriset: import blir dyrare,
    export-intäkt sjunker med volymen. Eftersom import-pris > export-pris alltid
    körs aldrig båda samtidigt → NTC respekteras per riktning.

    S_export/S_import (€/MWh vid full NTC) och offset_profile från config; S är
    empiriskt förankrat i 2023-2025 prisspread per gräns. Namn: "<kabel> imp/exp k".
    """
    offs    = list(me.get("offset_profile", [0.1667, 0.5, 0.8333]))
    n_tiers = int(me.get("n_tiers", len(offs)))
    offs    = offs[:n_tiers] if n_tiers <= len(offs) else offs
    spreads = me.get("spread_eur_per_mwh") or {}
    n_steps = len(offs)
    for name, zone, p_nom, price_bzn in cfg.get("market_connections", []):
        mc       = market_prices[price_bzn]
        s        = spreads.get(name, {}) or {}
        s_exp    = float(s.get("export", 0.0))
        s_imp    = float(s.get("import", 0.0))
        tier_cap = p_nom / n_steps
        for k, off in enumerate(offs, 1):
            n.add(
                "Generator", f"{name} imp{k}",
                bus=zone, carrier="market",
                p_nom=tier_cap, p_min_pu=0.0, p_max_pu=1.0,
                marginal_cost=mc + s_imp * off,
            )
            n.add(
                "Generator", f"{name} exp{k}",
                bus=zone, carrier="market",
                p_nom=tier_cap, p_min_pu=-1.0, p_max_pu=0.0,
                marginal_cost=mc - s_exp * off,
            )
    print(f"  → pris-elastisk gräns (trappa, {n_steps} steg/riktning) för "
          f"{len(cfg.get('market_connections', []))} kablar")
