"""Världen: omsätter körinställningarna i modellens config och nätverk.

`prepare_config` muterar zones.yaml-configen före bygget (scenarier, NTC, kostnader)
och returnerar det som bygget behöver utöver configen. `apply_post_build` gör de
ändringar som kräver det byggda nätverket (torrår, potentialtak, länkutbyggnad).

⚠️ Ordningen är en del av semantiken: scenariots ntc_overrides före grid.ntc_override,
market.ntc_scale före market.ntc_override, kostnadsscenariot före per-zon-räntor.
"""
from __future__ import annotations

from nordpsa.network.costs import annualized_cost


# --------------------------------------------------------------------------- scenarier
def apply_cost_scenario(cfg: dict, name: str) -> None:
    """Skriver över cfg['costs'] med ett kostnadsscenario ur cost_scenarios.

    Byggränta (IDC, ränta under byggtid): overnight' = OC·(1 + build_years/2·r).
    fom_fraction sätts = fast_DoU/overnight' så att absolut O&M-nivå bevaras (ej
    uppblåst av IDC, eftersom modellen beräknar annualiserat = overnight·(CRF+fom)).
    """
    scenarios = cfg.get("cost_scenarios", {})
    scen = scenarios.get(name)
    if scen is None:
        raise SystemExit(f"Okänt kostnadsscenario '{name}'. Finns: {list(scenarios)}")
    r = cfg["costs"]["discount_rate"]
    print(f"  → kostnadsscenario '{name}' (IDC med r={r}):")
    for tech, p in scen.items():
        if not isinstance(p, dict):
            continue
        tgt = cfg["costs"].setdefault(tech, {})
        idc = 1.0 + p["build_years"] / 2 * r
        tgt["lifetime_years"] = p["lifetime_years"]
        if tech == "battery":
            tgt["power_eur_per_kw"]   = p["power_eur_per_kw"]   * idc
            tgt["energy_eur_per_kwh"] = p["energy_eur_per_kwh"] * idc
            tgt["fom_fraction"]       = p.get("fom_fraction", 0.025)
            print(f"     {tech:<14} IDC×{idc:.3f}  power {tgt['power_eur_per_kw']:.0f} €/kW + "
                  f"energy {tgt['energy_eur_per_kwh']:.0f} €/kWh")
        else:
            oc_kw = p["oc_eur_per_kw"] * idc                       # OC' [EUR/kW]
            tgt["overnight_eur_per_w"] = oc_kw / 1000.0            # EUR/W
            tgt["fom_fraction"]        = p["fom_eur_per_kw"] / oc_kw
            tgt["vom_eur_per_mwh"]     = p["vom_eur_per_mwh"]
            print(f"     {tech:<14} IDC×{idc:.3f}  overnight {oc_kw/1000:.3f} €/W  "
                  f"fom {tgt['fom_fraction']*100:.2f}%  vom {p['vom_eur_per_mwh']}  L{p['lifetime_years']}")


def apply_demand_scenario(cfg: dict, name: str, battery_total=None) -> dict:
    """Adderar ett efterfrågescenario (demand_scenarios i zones.yaml) ovanpå eSett-basen.

    Muterar cfg: additional_load_mw (per zon), links (ntc_overrides), market_connections
    (market_ntc_overrides), zonernas befintliga kärnkraft och — om nuclear_exogenous —
    costs.nuclear.extendable. Returnerar det bygget behöver: hydrogen_overrides,
    ev_overrides, batteries [(zon, MW, h)] samt pnom_max/pnom_min {(zon, carrier): MW}
    (VRE-utbyggnadstak resp. exogena golv, appliceras efter bygget).

    battery_total = (GW, timmar): skala om scenariots batterier till den totalen,
    med bibehållen zonfördelning.
    """
    scenarios = cfg.get("demand_scenarios", {})
    scen = scenarios.get(name)
    if scen is None:
        raise SystemExit(f"Okänt efterfrågescenario '{name}'. Finns: {list(scenarios)}")

    print(f"  → efterfrågescenario '{name}' (additivt över eSett-bas):")
    if scen.get("nuclear_exogenous"):
        cfg["costs"]["nuclear"]["extendable"] = False
        print("     kärnkraft EXOGEN (costs.nuclear.extendable=False — ingen endogen expansion)")
    out = {"hydrogen_overrides": {}, "ev_overrides": {}, "batteries": [],
           "pnom_max": {}, "pnom_min": {}}
    pnom_max, pnom_min = out["pnom_max"], out["pnom_min"]
    scen_bats: list = []
    cfg.setdefault("additional_load_mw", {})
    for zone, zc in scen.get("zones", {}).items():
        extra = float(zc.get("extra_load_mw", 0.0))
        cfg["additional_load_mw"][zone] = cfg["additional_load_mw"].get(zone, 0.0) + extra

        h2 = zc.get("h2")
        if h2:
            # Elektrolysör ENDOGEN: kapaciteten optimeras (SvK-GW = startvärde, ej golv).
            # H2-last + lager förblir exogent fasta (SvK-plan). p_nom_max valfritt i config.
            out["hydrogen_overrides"][zone] = {
                "demand_mw":    float(h2["demand_mw"]),
                "electrolyser": {"p_nom_mw": float(h2["electrolyser_mw"]), "extendable": True,
                                 "p_nom_max_mw": float(h2.get("electrolyser_max_mw", 50000.0))},
                "store":        {"e_nom_mwh": float(h2["store_mwh"]),       "extendable": False},
            }

        cars = float(zc.get("ev_cars", 0.0))
        if cars > 0:
            out["ev_overrides"][zone] = {"car": cars, "heavy": 0.0}

        for carrier, pmax in (zc.get("pnom_max_mw") or {}).items():
            pnom_max[(zone, carrier)] = float(pmax)
        for carrier, pmin in (zc.get("pnom_min_mw") or {}).items():
            pnom_min[(zone, carrier)] = float(pmin)

        bat = zc.get("battery")
        bat_txt = ""
        if bat:                                              # exogent fast batteri (SvK storskaligt)
            scen_bats.append((zone, float(bat["p_nom_mw"]), float(bat.get("hours", 2))))
            if battery_total is None:
                bat_txt = f", batteri {bat['p_nom_mw']:.0f} MW/{bat.get('hours', 2)}h"

        nuc = zc.get("nuclear")
        nuc_txt = ""
        if nuc:                                              # befintlig kärnkraftsnivå i zonen
            cfg["zones"][zone]["nuclear_p_nom_mw"] = float(nuc["p_nom_mw"])
            nuc_txt = f", kärnkr {nuc['p_nom_mw']:.0f} MW (befintlig)"

        h2dem = (h2 or {}).get("demand_mw", 0)
        print(f"     {zone:<5} +{extra:6.0f} MW last, H2 {h2dem:.0f} MW, "
              f"EV {cars/1e6:.1f}M bilar, tak {{{', '.join(f'{c}:{int(p)}' for (z,c),p in pnom_max.items() if z==zone)}}}"
              f"{nuc_txt}{bat_txt}")

    if scen_bats:
        if battery_total is not None:
            gw, hours = float(battery_total[0]), float(battery_total[1])
            tot = sum(p for _, p, _ in scen_bats)
            scale = (gw * 1e3) / tot if tot > 0 else 0.0
            out["batteries"] = [(z, p * scale, hours) for z, p, _h in scen_bats]
            zsum = ", ".join(f"{z} {p*scale/1e3:.2f}" for z, p, _h in scen_bats)
            print(f"     batteri-OVERRIDE: {tot/1e3:.1f} GW → {gw:.1f} GW @ {hours:.0f}h "
                  f"(samma zonandel: {zsum} GW)")
        else:
            out["batteries"] = list(scen_bats)

    for z0, z1, mw in scen.get("ntc_overrides", []):
        for link in cfg.get("links", []):
            if link[0] == z0 and link[1] == z1 and link[2] != mw:
                # Ingen källetikett: av de fyra interna 2040-talen bär bara snitt 2 spår av
                # Tabell 10 (och även det är deratat 10700 → 9000); se zones.yaml.
                print(f"     NTC {z0}-{z1}: {link[2]} → {mw} MW")
                link[2] = mw

    # Kontinentkablar: matchas på connection-namn (market_connections = [namn, zon, mw, bzn])
    for cname, mw in scen.get("market_ntc_overrides", []):
        for mc in cfg.get("market_connections", []):
            if mc[0] == cname and mc[2] != mw:
                print(f"     Marknads-NTC {cname}: {mc[2]} → {mw} MW (Tabell 10)")
                mc[2] = mw
    return out


# --------------------------------------------------------------------------- före bygget
def prepare_config(cfg: dict, s: dict) -> dict:
    """Omsätter inställningarna `s` i `cfg` (muteras). Returnerar byggtilläggen:
    batteries, hydrogen_overrides, ev_overrides, pnom_max, pnom_min."""
    mode = s["run"]["mode"]
    sc   = s["scenario"]

    for k, v in (s["solver"] or {}).items():
        print(f"  → solver-option {k}: {cfg['solver'].get(k, '(ej satt)')} → {v!r}")
        cfg["solver"][k] = v

    cfg["additional_load_mw"] = {z: float(mw) for z, mw in (sc["extra_load_mw"] or {}).items()}

    if s["run"]["capacities"] == "config":          # dagens flotta: inget är investerbart
        for tech in cfg.get("costs", {}):
            if isinstance(cfg["costs"][tech], dict):
                cfg["costs"][tech]["extendable"] = False
    if sc["post_expansion_ntc"]:                     # framtida nät (t.ex. Aurora SE-N–FI)
        overrides = {(z0, z1): p for z0, z1, p in cfg.get("links_expansion_overrides", [])}
        for link in cfg.get("links", []):
            new = overrides.get((link[0], link[1]))
            if new is not None and link[2] != new:
                print(f"  → expansion-NTC: {link[0]}-{link[1]} {link[2]} → {new} MW")
                link[2] = new

    if sc["cost"]:
        apply_cost_scenario(cfg, sc["cost"])

    if sc["demand"]:
        extras = apply_demand_scenario(cfg, sc["demand"], sc["battery_total"])
    else:
        # Dagens värld: dagens FRIA baseline-batterier. Med ett demand-scenario
        # representeras flottan av scenariots batterier (undviker dubbelräkning).
        extras = {"hydrogen_overrides": {}, "ev_overrides": {}, "pnom_max": {}, "pnom_min": {},
                  "batteries": [(z, float(mw), float(h))
                                for z, mw, h in cfg.get("baseline_batteries", [])]}
        if extras["batteries"]:
            print("  Baseline-batterier (fria): "
                  + ", ".join(f"{z} {mw:.0f}MW/{h:.0f}h" for z, mw, h in extras["batteries"]))

    ext = mode == "expansion"
    extras["battery_invest"] = None
    if s["battery"]["endogenous"]:
        if extras["batteries"]:
            print("  → battery.endogenous: scenariots fria batterier tas bort ("
                  + ", ".join(f"{z} {mw:.0f}MW" for z, mw, _h in extras["batteries"]) + ")")
        extras["batteries"] = []
        ge = s["battery"]["gfm_extra_eur_per_kw"]
        extras["battery_invest"] = {"hours": float(s["battery"]["hours"]), "extendable": ext,
                                    "cost_scale": float(s["battery"]["cost_scale"]),
                                    "gfm_extra": None if ge is None else float(ge)}
    extras["syncon"] = ({"aux_loss_pu": float(cfg["stability"]["tech"]["syncon"]["aux_loss_pu"]),
                         "extendable": ext} if s["syncon"]["enabled"] else None)

    for pair, mw in (s["grid"]["ntc_override"] or {}).items():
        z0, z1 = pair.split(":")
        hit = [link for link in cfg.get("links", []) if link[0] == z0 and link[1] == z1]
        if not hit:
            raise SystemExit(f"grid.ntc_override: hittade ingen intern länk {z0}-{z1}")
        for link in hit:
            print(f"  → NTC-override {z0}-{z1}: {link[2]} → {float(mw):.0f} MW")
            link[2] = float(mw)

    for zone, frac in (s["hydro"]["soc_initial"] or {}).items():
        if zone not in cfg.get("zones", {}):
            raise SystemExit(f"hydro.soc_initial: okänd zon {zone!r}")
        print(f"  → SOC-ankare {zone}: {cfg['zones'][zone].get('hydro_soc_initial')} → {float(frac):.3f}")
        cfg["zones"][zone]["hydro_soc_initial"] = float(frac)

    mk = s["market"]
    if mk["ntc_scale"] is not None:
        f = float(mk["ntc_scale"])
        _mc = cfg.get("market_connections", []) or []
        _tot0 = sum(m[2] for m in _mc)
        print(f"  → Marknads-NTC skalas ×{f:g} "
              f"({len(_mc)} kablar, {_tot0:.0f} → {_tot0 * f:.0f} MW):")
        for mc in _mc:
            _old = mc[2]
            mc[2] = _old * f
            print(f"     {mc[0]:<12} {_old:6.0f} → {mc[2]:6.0f} MW")
    for cname, mw in (mk["ntc_override"] or {}).items():
        hit = [mc for mc in cfg.get("market_connections", []) if mc[0] == cname]
        if not hit:
            raise SystemExit(f"market.ntc_override: hittade ingen kontinentkabel '{cname}'")
        for mc in hit:
            print(f"  → Marknads-NTC-override {cname}: {mc[2]} → {float(mw):.0f} MW")
            mc[2] = float(mw)
    if not mk["enabled"]:
        cfg["market_connections"] = []

    spill = s[mode]["spill_cost"]
    cfg["costs"]["hydro"]["spill_cost_eur_per_mwh"] = spill
    print(f"  → hydro spill_cost = {spill} EUR/MWh")

    cfg.setdefault("market_elasticity", {})["enabled"] = bool(mk["elasticity"])

    nu, vre = s["nuclear"], s["vre"]
    if nu["discount_rate_by_zone"]:
        disc = {z: float(r) for z, r in nu["discount_rate_by_zone"].items()}
        cfg["costs"].setdefault("nuclear", {})["discount_rate_by_zone"] = disc
        print("  → kärnkrafts-diskontoränta (ny/expansion): "
              + ", ".join(f"{z} {r:.0%}" for z, r in disc.items()))
    if vre["offwind_discount_rate_by_zone"]:
        disc = {z: float(r) for z, r in vre["offwind_discount_rate_by_zone"].items()}
        cfg["costs"].setdefault("wind_offshore", {})["discount_rate_by_zone"] = disc
        print("  → havsvind-diskontoränta: " + ", ".join(f"{z} {r:.0%}" for z, r in disc.items()))

    if s["heat"]["enabled"]:
        cfg.setdefault("heat", {})["enabled"] = True
        if s["heat"]["el_tax_zero"]:
            old_tax = float(cfg["heat"].get("el_tax_eur_per_mwh", 0.0))
            cfg["heat"]["el_tax_eur_per_mwh"] = 0.0
            print(f"  → elskatt på VP/el-panna nollställd ({old_tax:g} → 0 €/MWh)")
        print("  → fjärrvärmesektor aktiv")

    if extras["ev_overrides"]:                    # fylls av demand-scenariots ev_cars
        cfg.setdefault("ev", {})["enabled"] = True
        print(f"  → fordonsladdning aktiv i {', '.join(extras['ev_overrides'])}")
    return extras


def scale_continent_prices(cfg: dict, s: dict, market_prices: dict) -> None:
    """Demand-scenariots continent_price_eur_mwh: multiplikativ omskalning av 2023-25-
    serien per budzon till scenariots nivå (timformen bevaras, nivån byts). Görs på HELA
    serien, före årsurval och resampling."""
    if not s["scenario"]["demand"]:
        return
    cps = (cfg.get("demand_scenarios", {}).get(s["scenario"]["demand"], {})
           .get("continent_price_eur_mwh") or {})
    for bzn, target in cps.items():
        if bzn in market_prices:
            m = market_prices[bzn].mean()
            if m > 0:
                market_prices[bzn] = market_prices[bzn] * (float(target) / m)
                print(f"  → kontinentpris {bzn}: snitt {m:.1f} → {float(target):.0f} €/MWh (×{float(target)/m:.2f})")


def nuclear_build_args(cfg: dict, s: dict) -> tuple[list, dict]:
    """(extra_nuclear, synthetic_nuclear) för build_network."""
    nu = s["nuclear"]
    extra = [(str(z), int(k), int(seed)) for z, k, seed in nu["add"]]
    fixed: dict = {}
    for spec in nu["add_fixed"]:
        z, k, mw = spec[:3]
        seed = int(spec[3]) if len(spec) > 3 else None
        fixed.setdefault(str(z), []).append((int(k), float(mw), seed))
    params = dict(cfg.get("nuclear_synth", {}))          # kopia: mutera ej configen
    if nu["new_min_load"] is not None:
        params["min_load_frac_exp"] = float(nu["new_min_load"])
        print(f"Ny kärnkraft LASTFÖLJANDE: p_min_pu = {nu['new_min_load']:g} × p_max_pu "
              f"(befintlig flotta oförändrat must-run)")
    synthetic = {"existing": cfg.get("nuclear_synth_existing", {}), "params": params,
                 "active": bool(extra), "fixed": fixed}
    if extra or fixed:
        ex = cfg.get("nuclear_synth_existing", {})
        print(f"Kärnkraft: befintlig flotta {'syntetisk ' + str(list(ex)) if extra else 'orörd (faktisk profil)'}; "
              f"nya extendable {extra}; nya FASTA (costed svk_2040) {fixed} "
              f"(target_cf={params.get('target_cf', 0.85)})")
    return extra, synthetic


# --------------------------------------------------------------------------- efter bygget
def apply_low_hydro(n, factor: float, year: int = 2024) -> None:
    """Torrår: skala ett års hydro NEDÅT med factor — både reservoar-inflöde
    (storage_units_t.inflow) och RoR (must-run-generatorer, carrier 'hydro': p_max_pu &
    p_min_pu). Övriga år orörda. factor<1 = torrare."""
    dt  = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600
    inf = n.storage_units_t.inflow
    m_inf = inf.index.year == year
    if not m_inf.any():
        print(f"  → low_hydro: år {year} ingår ej i perioden — ingen ändring")
        return
    before = float(inf.loc[m_inf].to_numpy().sum()) * dt / 1e6
    inf.loc[m_inf] *= factor
    ror = [g for g in n.generators.index if n.generators.at[g, "carrier"] == "hydro"]
    static = []
    for g in ror:
        hit = False
        for tbl in (n.generators_t.p_max_pu, n.generators_t.p_min_pu):
            if g in tbl.columns:
                tbl.loc[tbl.index.year == year, g] *= factor
                hit = True
        if not hit:
            static.append(g)
    msg = (f"  → low_hydro {factor:g}: {year} reservoar-inflöde {before:.1f} → "
           f"{before * factor:.1f} TWh + RoR ×{factor:g} ({len(ror) - len(static)} profil-gen)")
    if static:
        msg += f"  ⚠️ {len(static)} RoR-gen har statisk p_max_pu, ej skalade ({static[:3]})"
    print(msg)


def make_link_extendable(n, cfg: dict, link_name: str, overnight_eur_per_w: float,
                         n_years: float, p_nom_min=None, p_nom_max: float = 30000.0,
                         lifetime: int = 40):
    """Gör en intern NTC-länk kapacitetsexpanderbar med annualiserad overnight-kostnad.

    overnight_eur_per_w: t.ex. 2.0 för 2 M€/MW. Annualiseras som övriga tekniker
    (CRF(lifetime, r) + fom) × n_years och debiteras på p_nom_opt. p_min_pu=-1.0
    behålls → flödesgränsen blir ±p_nom_opt (symmetrisk bidirektionell utbyggnad).
    p_nom_min default = byggd p_nom (golvet)."""
    r   = cfg["costs"]["discount_rate"]
    fom = cfg["costs"]["fom_fraction"]
    if link_name not in n.links.index:
        raise SystemExit(f"grid.expand_link: länk '{link_name}' finns ej i nätverket")
    built = float(n.links.at[link_name, "p_nom"])
    floor = built if p_nom_min is None else float(p_nom_min)
    ann   = annualized_cost(overnight_eur_per_w, lifetime, r, fom)   # €/MW/år
    n.links.at[link_name, "p_nom_extendable"] = True
    n.links.at[link_name, "p_nom_min"]        = floor
    n.links.at[link_name, "p_nom_max"]        = p_nom_max
    n.links.at[link_name, "capital_cost"]     = ann * n_years
    print(f"  → expanderbar länk: {link_name} (golv {floor:.0f} MW, tak {p_nom_max:.0f} MW, "
          f"overnight {overnight_eur_per_w:.2f} €/W, annual.kap {ann/1e3:.0f} €/kW/år)")


def apply_post_build(n, cfg: dict, s: dict, n_years: float) -> None:
    """Ändringar som kräver det byggda nätverket, i fast ordning."""
    lh = s["scenario"]["low_hydro"]
    if lh is not None:
        apply_low_hydro(n, float(lh["factor"]), int(lh.get("year", 2024)))

    if s["grid"]["expand_link"]:
        print("Endogen länk-expansion:")
        for spec in s["grid"]["expand_link"]:
            z0, z1, meur = spec[0], spec[1], float(spec[2])
            floor = float(spec[3]) if len(spec) > 3 and spec[3] is not None else None
            pmax  = float(spec[4]) if len(spec) > 4 else 30000.0
            make_link_extendable(n, cfg, f"{z0}-{z1}", meur, n_years,
                                 p_nom_min=floor, p_nom_max=pmax)


def apply_potentials(n, s: dict, extras: dict) -> None:
    """Demand-scenariots utbyggnadstak (pnom_max_mw) och exogena golv (pnom_min_mw).
    Tak 0 (t.ex. kärnkraft i NO/DK) låser tekniken till befintlig nivå."""
    scale = float(s["vre"]["onshore_potential_scale"])
    for (zone, carrier), pmax in extras["pnom_max"].items():
        name = f"{zone} {carrier}"
        if name not in n.generators.index or not bool(n.generators.at[name, "p_nom_extendable"]):
            continue
        scaled = pmax * scale if carrier == "wind_onshore" else pmax
        existing = float(n.generators.at[name, "p_nom"])
        cap = max(scaled, existing)          # aldrig under redan installerat
        n.generators.at[name, "p_nom_min"] = existing
        n.generators.at[name, "p_nom_max"] = cap
        note = f" [×{scale:g}]" if (carrier == "wind_onshore" and scale != 1.0) else ""
        print(f"  → potential {zone} {carrier}: p_nom_max = {cap:.0f} MW (installerat {existing:.0f}){note}")

    for (zone, carrier), pmin in extras["pnom_min"].items():
        name = f"{zone} {carrier}"
        if name not in n.generators.index:
            continue
        if not bool(n.generators.at[name, "p_nom_extendable"]):
            print(f"  Varning: {name} ej extendable — kan ej sätta golv {pmin:.0f}, hoppar över")
            continue
        existing = float(n.generators.at[name, "p_nom"])
        floor = max(pmin, existing)
        n.generators.at[name, "p_nom_min"] = floor
        if float(n.generators.at[name, "p_nom_max"]) < floor:
            n.generators.at[name, "p_nom_max"] = floor
        print(f"  → golv {zone} {carrier}: p_nom_min = {floor:.0f} MW "
              f"(installerat {existing:.0f}{' — golv binder, +%.0f MW' % (floor-existing) if floor>existing else ''})")
