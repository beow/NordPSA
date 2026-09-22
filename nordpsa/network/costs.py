"""Annualisering av kapitalkostnader (CRF, overnight → EUR/MW/år)."""



def crf(lifetime: int, r: float) -> float:
    """Capital Recovery Factor."""
    return r * (1 + r) ** lifetime / ((1 + r) ** lifetime - 1)


def annualized_cost(overnight_eur_per_w: float, lifetime: int,
                     r: float, fom_fraction: float) -> float:
    """Annualiserad kapitalkostnad (EUR/MW/år).

    overnight_eur_per_w: t.ex. 7.0 för 7 EUR/W = 7 000 000 EUR/MW
    """
    oc_mw = overnight_eur_per_w * 1e6
    return oc_mw * (crf(lifetime, r) + fom_fraction)


def scenario_overnight_mw(cfg: dict, tech: str, name: str = "svk_2040",
                           max_hours: float | None = None):
    """Overnight €/MW (INKL. byggränta IDC) + (lifetime, fom_fraction, vom) för en
    teknik ur cfg['cost_scenarios'][name]. Speglar IDC-matten i apply_cost_scenario
    (nordpsa/world.py) men returnerar råtal utan att mutera cfg['costs']. Används för att
    prissätta TILLAGD kapacitet (--add-battery / --add-nuclear-fixed) från SvK-2040
    OBEROENDE av run:ens --cost-scenario."""
    p   = cfg["cost_scenarios"][name][tech]
    r   = cfg["costs"]["discount_rate"]
    idc = 1.0 + p["build_years"] / 2 * r
    if tech == "battery":
        oc = (p["power_eur_per_kw"] + max_hours * p["energy_eur_per_kwh"]) * 1e3 * idc
        return oc, p["lifetime_years"], p.get("fom_fraction", 0.025), 0.0
    oc = p["oc_eur_per_kw"] * 1e3 * idc
    return oc, p["lifetime_years"], p["fom_eur_per_kw"] / (p["oc_eur_per_kw"] * idc), \
        p.get("vom_eur_per_mwh", 0.0)
