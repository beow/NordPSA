"""Vätgas: LMA:s tre kategorier (fast last, lagerflex, elektrobränslen)."""

import pypsa

from nordpsa.network.core import MC_SLACK
from nordpsa.network.costs import crf


def add_hydrogen(n: pypsa.Network, cfg: dict, r: float, fom: float,
                  n_years: float, overrides: dict | None = None) -> None:
    """Bygger valfria vätgassystem per zon (power-to-X):

        elbuss → Link(elektrolys, η_el) → Bus(H2) → Store(lager, e_cyclic)
        Bus(H2) → Load(baslast, fast MW) + Generator(slack, hög MC)
        Bus(H2) → Link(turbin, η_turb) → elbuss     (valfri)

    Zon-konfiguration från cfg['hydrogen'] sammanslaget med `overrides` (CLI, har
    företräde). Teknikkostnader/verkningsgrader från cfg['costs']['hydrogen'].
    Allt i MWh (LHV). Hoppar tyst över om inget H2 konfigurerats.

    Enheter: elektrolysör-p_nom i MW_el (bus0=el). Turbin specas i MW_el_ut men
    PyPSA-p_nom (bus0=H2) = p_el/η_turb, capital_cost skalas med η_turb → kostnad
    per kW_el ut. Icke-extendable komponenter får capital_cost=0.
    """
    h2_zones = dict(cfg.get("hydrogen") or {})
    if overrides:
        h2_zones.update(overrides)   # CLI har företräde
    if not h2_zones:
        return

    hc       = cfg["costs"]["hydrogen"]
    el_c, tb_c, st_c = hc["electrolyser"], hc["turbine"], hc["store"]
    mc_slack = hc.get("slack_eur_per_mwh", MC_SLACK)
    el_pmin  = hc.get("electrolyser_p_min_pu", 0.0)

    for car in ("H2", "electrolyser", "H2 turbine", "H2 store", "H2 slack"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    def ann_kw(c):   # annualiserad €/MW (overnight i €/kW)
        return c["overnight_eur_per_kw"] * 1e3 * (crf(c["lifetime_years"], r) + c["fom_fraction"]) * n_years
    def ann_kwh(overnight, c):  # annualiserad €/MWh (overnight i €/kWh, per zon)
        return overnight * 1e3 * (crf(c["lifetime_years"], r) + c["fom_fraction"]) * n_years

    # Lagerkostnad per zon: zon-block > geologiska undantag > default
    st_base    = st_c["overnight_eur_per_kwh"]
    st_by_zone = st_c.get("overnight_eur_per_kwh_by_zone") or {}

    for zone, zc in h2_zones.items():
        if zone not in n.buses.index:
            print(f"  Varning: H2-zon {zone} saknas — hoppar över")
            continue
        h2bus = f"{zone} H2"
        n.add("Bus", h2bus, carrier="H2")

        # Elektrolys: bus0=el, bus1=H2; p_nom i MW_el.
        # ⚠️ Sedan 2026-08-16 dimensioneras den ur cfg['electrolyser_overcapacity'] och
        # LÅSES (icke-extendable) → sex kapacitetsvariabler färre i LP:t. LMA:s 50 %
        # överkapacitet gäller BARA typ 2: typ 1 är oflexibel per definition och typ 3:s
        # last kan bara trappas NED, så extra kapacitet där kan aldrig användas.
        elc     = zc.get("electrolyser", {})
        el_ext  = bool(elc.get("extendable", False))
        el_pnom = float(elc.get("p_nom_mw", 0.0))
        _oc = cfg.get("electrolyser_overcapacity")
        if _oc:
            _sp = (cfg.get("hydrogen_split") or {}).get(zone, {"type2": 1.0})
            _d  = float(zc.get("demand_mw", 0.0))
            el_pnom = (float(_oc.get("type2", 1.5)) * _d * float(_sp.get("type2", 1.0))
                       / el_c["efficiency"])
            el_ext  = False
        el_pmax = float(elc.get("p_nom_max_mw", 50000.0))
        n.add("Link", f"{zone} electrolyser",
              bus0=zone, bus1=h2bus, carrier="electrolyser",
              efficiency=el_c["efficiency"],
              p_nom=el_pnom, p_nom_extendable=el_ext,
              p_nom_min=0.0 if el_ext else el_pnom,
              p_nom_max=el_pmax if el_ext else float("inf"),
              p_min_pu=el_pmin,
              capital_cost=ann_kw(el_c) if el_ext else 0.0)

        # Lager: Store på H2-bussen (energi fri från effekt; e_cyclic: start=slut)
        stc       = zc.get("store", {})
        st_ext    = bool(stc.get("extendable", False))
        e_nom     = float(stc.get("e_nom_mwh", 0.0))
        st_overn  = float(stc.get("overnight_eur_per_kwh", st_by_zone.get(zone, st_base)))
        st_emax   = float(stc.get("e_nom_max_mwh", st_c.get("e_nom_max_mwh", 1e7)))
        n.add("Store", f"{zone} H2 store",
              bus=h2bus, carrier="H2 store",
              e_nom=e_nom, e_nom_extendable=st_ext,
              e_nom_min=0.0 if st_ext else e_nom,
              e_nom_max=st_emax if st_ext else float("inf"),
              e_cyclic=True,
              capital_cost=ann_kwh(st_overn, st_c) if st_ext else 0.0)

        # ── LMA2026:s tre kategorier (cfg['hydrogen_split']) ──────────────────
        # typ 1 oflexibel · typ 2 lagerflexibel · typ 3 elektrobränslen med pristrappa.
        # Utan split-block faller allt på typ 2 = det gamla beteendet.
        demand = float(zc.get("demand_mw", 0.0))
        sp     = (cfg.get("hydrogen_split") or {}).get(zone, {"type1": 0.0, "type2": 1.0,
                                                              "type3": 0.0})
        tot_sh = sum(float(sp.get(k, 0.0)) for k in ("type1", "type2", "type3"))
        if abs(tot_sh - 1.0) > 1e-6:
            raise SystemExit(f"hydrogen_split[{zone}] summerar till {tot_sh:.4f}, ska vara 1,0 "
                             "— annars går H2-balansen inte att sluta")
        d1 = demand * float(sp.get("type1", 0.0))
        d2 = demand * float(sp.get("type2", 0.0))
        d3 = demand * float(sp.get("type3", 0.0))

        # TYP 1 — oflexibel: konstant EL-last direkt på elbussen. Ingen H2-buss behövs;
        # elektrolysören skulle ändå tvingas följa lasten exakt varje timme (inget lager,
        # inget alternativ), så en länk vore 26 000 låsta variabler utan frihetsgrad.
        if d1 > 0:
            n.add("Load", f"{zone} H2 inflex", bus=zone, p_set=d1 / el_c["efficiency"])

        # TYP 2 — lagerflexibel: last på H2-bussen, lagret ovan betjänar den.
        n.add("Load", f"{zone} H2 load", bus=h2bus, p_set=d2)
        n.add("Generator", f"{zone} H2 slack",
              bus=h2bus, carrier="H2 slack",
              p_nom=1e6, marginal_cost=mc_slack)

        # TYP 3 — elektrobränslen: EGEN buss utan lager. Måste vara separerad, annars kan
        # lagret laddas med "vätgas som inte producerades" och LP:t hittar arbitraget
        # shed → lagra → slipp H2-slacken; dessutom sätter trappan ett pristak som
        # dödar lagrets normala arbitrage. Ingen 3000-slack här: trancherna täcker per
        # definition 100 % av lasten, så en felsummering ska bli INFEASIBLE, inte döljas.
        if d3 > 0:
            _add_h2_electrofuel(n, cfg, zone, d3, el_c["efficiency"])

        # Valfri turbin: bus0=H2, bus1=el; config p_nom i MW_el_ut → p_nom(H2)=el/η
        tbc      = zc.get("turbine")
        turb_txt = "ingen turbin"
        if tbc:
            eta    = tb_c["efficiency"]
            tb_ext = bool(tbc.get("extendable", False))
            p_el   = float(tbc.get("p_nom_mw", 0.0))
            p_h2   = p_el / eta if eta > 0 else 0.0
            n.add("Link", f"{zone} H2 turbine",
                  bus0=h2bus, bus1=zone, carrier="H2 turbine",
                  efficiency=eta,
                  p_nom=p_h2, p_nom_extendable=tb_ext,
                  p_nom_min=0.0 if tb_ext else p_h2,
                  capital_cost=(ann_kw(tb_c) * eta) if tb_ext else 0.0)
            turb_txt = f"turbin {p_el:.0f} MW_el (η={eta})"

        print(f"  → H2 {zone}: last {demand:.0f} MW_H2 "
              f"(typ1 {d1:.0f} oflex / typ2 {d2:.0f} lager / typ3 {d3:.0f} elektrobränsle), "
              f"elektrolys typ2 {el_pnom:.0f} MW_el{' ext' if el_ext else ''} "
              f"(η={el_c['efficiency']}), lager {e_nom:.0f} MWh{' ext' if st_ext else ''} "
              f"({st_overn:g} €/kWh), {turb_txt}")


def _add_h2_electrofuel(n: pypsa.Network, cfg: dict, zone: str, demand_h2: float,
                        eta_el: float) -> None:
    """Elektrobränslen (LMA2026 typ 3): egen H2-buss med pristrappa i stället för lager.

    Trappan är en FÖRBRUKNINGSREDUKTION: när elpriset passerar tröskeln stängs en andel
    av produktionen av. I LP:t går det inte att göra p_nom prisberoende, så avstängningen
    uttrycks som ett ALTERNATIV till att köra — shed-generatorer på H2-bussen:

        elektrolysörens kostnad för 1 MWh_H2 = p_el / η
        shed-tranch k väljs när  p_el / η > tröskel_k / η  ⇔  p_el > tröskel_k

    Generatorerna ÄR slacken. Utan dem kan H2-balansen inte slutas när elektrolysören
    stängs av och LP:t blir infeasible. Ingen 3000-slack läggs till: andelarna summerar
    till 1,0, så balansen kan alltid slutas — och en felsummering ska då falla ut som
    infeasible i stället för att tyst absorberas till 3000 €/MWh.
    """
    he = cfg.get("hydrogen_elastic") or {}
    if not he.get("enabled", False):
        return
    steps  = list(he.get("price_steps_eur_per_mwh_el", [50, 100, 440]))
    shares = list(he.get("shed_shares", [0.30, 0.30, 0.40]))
    if len(steps) != len(shares):
        raise SystemExit("hydrogen_elastic: price_steps och shed_shares olika längd")
    if abs(sum(shares) - 1.0) > 1e-6:
        raise SystemExit(f"hydrogen_elastic.shed_shares summerar till {sum(shares):.4f}, "
                         "ska vara 1,0 — annars kan H2-balansen inte slutas vid höga priser")

    bus = f"{zone} H2ef"
    for car in ("H2", "electrolyser", "H2 shed"):
        if car not in n.carriers.index:
            n.add("Carrier", car)
    n.add("Bus", bus, carrier="H2")
    n.add("Load", f"{zone} H2ef load", bus=bus, p_set=demand_h2)

    oc = float((cfg.get("electrolyser_overcapacity") or {}).get("type3", 1.0))
    n.add("Link", f"{zone} electrolyser ef",
          bus0=zone, bus1=bus, carrier="electrolyser",
          efficiency=eta_el, p_nom=oc * demand_h2 / eta_el,
          p_nom_extendable=False, capital_cost=0.0)

    for k, (thr, sh) in enumerate(zip(steps, shares), start=1):
        n.add("Generator", f"{zone} H2ef shed{k}",
              bus=bus, carrier="H2 shed",
              p_nom=sh * demand_h2,
              marginal_cost=float(thr) / eta_el)
