#!/usr/bin/env python
"""Mastersheet (Excel) för en batch körningar: per (run, zon) pris, installerad
effekt och produktion per kraftslag (inkl. hydro-magasin/RoR, KVV-el, batteri),
netto-export inom resp. utanför Norden — plus ett NTC-flödesblad (netto, brutto,
% bindande timmar).

Användning:
  python scripts/build_mastersheet.py --prefix 17 --out docs/batch17_mastersheet.xlsx
  python scripts/build_mastersheet.py --prefix 17 18 --out docs/master_17_18.xlsx   # flera batchar
"""
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]

# Kraftslag för kapacitet/produktion (speglar ebalance.py: Vattenkraft = magasin+RoR,
# VRE redovisas som TILLGÄNGLIGT, KVV/batteri egna rader).
CAP_CARRIERS = ["Vattenkraft", "Kärnkraft", "Vind onshore", "Vind offshore",
                "Sol", "KVV-el", "Termisk", "Gas", "Batteri"]
PROD_ROWS = ["Vattenkraft", "Kärnkraft", "Vind onshore", "Vind offshore",
             "Sol", "KVV-el", "Termisk", "Gas"]   # + PRODUKTION TOTAL (inkl slack)
# Flexibla last-kapaciteter (Link-p_nom_opt, MW_el → GW), placeras efter produktionsblocket
LOADCAP_COLS = ["Elektrolysör", "VP", "Elpanna", "EV-laddare"]
CONS_ROWS = ["Last", "H2-elektrolys", "Värme (VP+panna)", "EV-laddning",
             "Batteri (netto ut)", "Spill", "Kontinent-export",
             "Norden-intern export"]              # + KONSUMTION TOTAL


def discover_runs(prefix):
    """{label: path} där label = 'run170 baseline' (run-nr + scenario-slogan ur mappnamnet)."""
    out = {}
    for idx in range(7):          # run{prefix}0 .. run{prefix}6 (t.ex. batch 19: run190–196 inkl. notax)
        hits = sorted((ROOT / "results").glob(f"run{prefix}{idx}_*"))
        hits = [h for h in hits if (h / "network.nc").exists()]
        if hits:
            parts = hits[0].name.split("_")          # run170_baseline_3h / run175_batt25_4h_3h
            slogan = "_".join(parts[1:-1]) if len(parts) > 2 else ""   # behåller multi-_ namn
            label = f"{parts[0]} {slogan}".strip()
            out[label] = hits[0]
    return out


def extract(n):
    """Returnerar (zon→dict) med pris/kap/prod/handel samt NTC-flödeslista.
    Produktion och handel annualiseras (TWh/år); kapacitet i GW; pris objektivviktat."""
    w = n.snapshot_weightings.objective
    wsum = w.sum()
    nyears = float(wsum / 8760.0)          # antal år i simuleringen (≈3)
    g, su, lk = n.generators, n.storage_units, n.links

    def gen_cap(zone, carrier):
        m = (g.carrier == carrier) & (g.bus == zone)
        return g.loc[m, "p_nom_opt"].sum() / 1e3   # GW

    zero = pd.Series(0.0, index=w.index)
    twh = lambda s: float((s.mul(w, axis=0)).sum() / 1e6 / nyears)       # serie → TWh/år

    def gen_disp(zone, carrier):
        names = g.index[(g.carrier == carrier) & (g.bus == zone)]
        return n.generators_t.p[names].clip(lower=0).sum(axis=1) if len(names) else zero.copy()

    def vre_avail_spill(zone, carrier):
        """(tillgängligt TWh/år, spill TWh/år) = p_max_pu×p_nom_opt resp. tillg.−dispatchat."""
        names = [x for x in g.index[(g.carrier == carrier) & (g.bus == zone)]]
        if not names:
            return 0.0, 0.0
        pm = n.generators_t.p_max_pu
        avail = sum((pm[x] * g.at[x, "p_nom_opt"] for x in names if x in pm.columns), zero.copy())
        disp = n.generators_t.p[names].clip(lower=0).sum(axis=1)
        return twh(avail), twh((avail - disp).clip(lower=0))

    def su_cap(zone, carrier):
        m = (su.carrier == carrier) & (su.bus == zone)
        return su.loc[m, "p_nom_opt"].sum() / 1e3

    def su_disp(zone, carrier):
        names = su.index[(su.carrier == carrier) & (su.bus == zone)]
        return n.storage_units_t.p[names].clip(lower=0).sum(axis=1) if len(names) else zero.copy()

    def su_net(zone, carrier):                                          # signerat netto (+ = ut)
        names = su.index[(su.carrier == carrier) & (su.bus == zone)]
        return twh(n.storage_units_t.p[names].sum(axis=1)) if len(names) else 0.0

    def link_p0(name):
        return n.links_t.p0[name].clip(lower=0) if name in lk.index else zero.copy()

    def link_cap(name):
        return lk.at[name, "p_nom_opt"] / 1e3 if name in lk.index else 0.0   # GW_el

    def load_p(name):
        return n.loads_t.p_set[name] if name in n.loads_t.p_set.columns else zero.copy()

    def chp_cap_prod(zone):
        name = f"{zone} chp"
        if name not in lk.index:
            return 0.0, 0.0
        eta = float(lk.at[name, "efficiency"])
        cap = lk.at[name, "p_nom_opt"] * eta / 1e3                       # GW_el
        prod = twh((-n.links_t.p1[name]).clip(lower=0))                  # TWh_el/år
        return cap, prod

    # Marknadsgeneratorer per zon (kontinenthandel): netto-injektion till zonen
    mkt = g.index[g.carrier == "market"]
    cont_export = {z: 0.0 for z in ZONES}            # TWh ut ur Norden
    for gname in mkt:
        z = g.at[gname, "bus"]
        e = float((n.generators_t.p[gname] * w).sum() / 1e6 / nyears)   # >0 = import till zonen
        cont_export[z] = cont_export.get(z, 0.0) - e           # export = -import

    # Interna AC-länkar: netto-export inom Norden per zon + NTC-flödeslista
    ac = lk.index[lk.carrier == "AC"]
    intern_export = {z: 0.0 for z in ZONES}
    ntc_rows = []
    for l in ac:
        b0, b1 = lk.at[l, "bus0"], lk.at[l, "bus1"]
        cap = float(lk.at[l, "p_nom"])                          # NTC (fryst), MW
        f = n.links_t.p0[l]                                     # + = b0→b1
        net = float((f * w).sum() / 1e6 / nyears)               # TWh/år b0→b1
        gross_fwd = float(f.clip(lower=0).mul(w).sum() / 1e6 / nyears)
        gross_rev = float((-f).clip(lower=0).mul(w).sum() / 1e6 / nyears)
        bind = float(w[(f.abs() >= 0.99 * cap)].sum() / wsum * 100) if cap > 0 else 0.0
        ntc_rows.append({"länk": l, "typ": "Intern", "NTC_MW": cap, "netto_TWh/år": net,
                         "brutto→_TWh/år": gross_fwd, "brutto←_TWh/år": gross_rev,
                         "%bindande": bind})
        if b0 in intern_export:
            intern_export[b0] += net
        if b1 in intern_export:
            intern_export[b1] -= net

    # Marknadslänkar (kontinentkablar) = imp/exp-generatorer per anslutning. netto/→ = EXPORT
    # (zon→kontinent, läses vänster→höger som AC-raderna); ← = import. NTC = Σ imp-stegens p_nom.
    conns = {}
    for gname in mkt:
        conns.setdefault(gname.rsplit(" ", 1)[0], []).append(gname)   # 'DK DE imp1' → 'DK DE'
    for conn, gens in sorted(conns.items()):
        s = n.generators_t.p[gens].sum(axis=1)                        # + = nettoimport till zonen
        cap = float(g.loc[[x for x in gens if "imp" in x], "p_nom"].sum())
        netto = float((-s * w).sum() / 1e6 / nyears)                  # + = export
        gross_exp = float((-s).clip(lower=0).mul(w).sum() / 1e6 / nyears)
        gross_imp = float(s.clip(lower=0).mul(w).sum() / 1e6 / nyears)
        bind = float(w[(s.abs() >= 0.99 * cap)].sum() / wsum * 100) if cap > 0 else 0.0
        ntc_rows.append({"länk": conn, "typ": "Marknad", "NTC_MW": cap, "netto_TWh/år": netto,
                         "brutto→_TWh/år": gross_exp, "brutto←_TWh/år": gross_imp,
                         "%bindande": bind})

    # Pris (objektivviktat snitt)
    price = {z: float((n.buses_t.marginal_price[z] * w).sum() / wsum) for z in ZONES}

    # Tak-bindning: ligger ett kraftslags p_nom_opt vid sitt p_nom_max? (Bara EXTENDABLE
    # komponenter kan binda; fast flotta hoppas över.) zone=None → Norden-aggregat.
    def _binds(comp, carrier, zone):
        m = (comp.carrier == carrier) & comp.p_nom_extendable
        m &= (comp.bus == zone) if zone is not None else comp.bus.isin(ZONES)
        if not m.any():
            return False
        opt, mx = comp.loc[m, "p_nom_opt"].sum(), comp.loc[m, "p_nom_max"].sum()
        return bool(np.isfinite(mx) and mx > 0 and opt >= 0.999 * mx)

    def binds_dict(zone):
        return {
            "Vattenkraft": _binds(su, "hydro", zone) or _binds(g, "hydro", zone),
            "Kärnkraft": _binds(g, "nuclear", zone),
            "Vind onshore": _binds(g, "wind_onshore", zone),
            "Vind offshore": _binds(g, "wind_offshore", zone),
            "Sol": _binds(g, "solar", zone), "KVV-el": False, "Termisk": False,
            "Gas": _binds(g, "gas", zone), "Batteri": _binds(su, "battery", zone),
        }

    zone_data = {}
    for z in ZONES:
        chp_c, chp_p = chp_cap_prod(z)
        cap = {
            "Vattenkraft": su_cap(z, "hydro") + gen_cap(z, "hydro"),    # magasin + RoR
            "Kärnkraft": gen_cap(z, "nuclear"), "Vind onshore": gen_cap(z, "wind_onshore"),
            "Vind offshore": gen_cap(z, "wind_offshore"), "Sol": gen_cap(z, "solar"),
            "KVV-el": chp_c, "Termisk": gen_cap(z, "thermal"), "Gas": gen_cap(z, "gas"),
            "Batteri": su_cap(z, "battery"),
        }
        # Produktion (VRE = TILLGÄNGLIGT; spill ackumuleras till konsumtionsblocket)
        on_a, on_s = vre_avail_spill(z, "wind_onshore")
        off_a, off_s = vre_avail_spill(z, "wind_offshore")
        sol_a, sol_s = vre_avail_spill(z, "solar")
        spill = on_s + off_s + sol_s
        prod = {
            "Vattenkraft": twh(su_disp(z, "hydro")) + twh(gen_disp(z, "hydro")),
            "Kärnkraft": twh(gen_disp(z, "nuclear")), "Vind onshore": on_a,
            "Vind offshore": off_a, "Sol": sol_a, "KVV-el": chp_p,
            "Termisk": twh(gen_disp(z, "thermal")), "Gas": twh(gen_disp(z, "gas")),
        }
        slack = twh(gen_disp(z, "slack"))
        prod_total = sum(prod.values()) + slack

        # Konsumtion (sänkor + nettoexport). batt_net = signerat netto ut (≈ −laddförluster).
        load = twh(load_p(f"{z} load"))
        h2 = twh(link_p0(f"{z} electrolyser"))
        heat = twh(link_p0(f"{z} heat hp") + link_p0(f"{z} heat elboiler"))
        ev = twh(sum((link_p0(f"{z} EV {c} charger") for c in ("car", "heavy")), zero.copy())
                 + sum((load_p(f"{z} EV {c} inflex") for c in ("car", "heavy")), zero.copy()))
        batt_net = su_net(z, "battery")
        cons = {
            "Last": load, "H2-elektrolys": h2, "Värme (VP+panna)": heat, "EV-laddning": ev,
            "Batteri (netto ut)": batt_net, "Spill": spill,
            "Kontinent-export": cont_export[z], "Norden-intern export": intern_export[z],
        }
        kons_total = (load + h2 + heat + ev + cont_export[z] + intern_export[z]
                      + spill - batt_net)
        balans = prod_total + batt_net - load - h2 - heat - ev - spill \
                 - cont_export[z] - intern_export[z]

        loadcap = {
            "Elektrolysör": link_cap(f"{z} electrolyser"),
            "VP": link_cap(f"{z} heat hp"),
            "Elpanna": link_cap(f"{z} heat elboiler"),
            "EV-laddare": sum(link_cap(f"{z} EV {c} charger") for c in ("car", "heavy")),
        }
        zone_data[z] = {"pris": price[z], "cap": cap, "loadcap": loadcap,
                        "prod": prod, "prod_total": prod_total, "binds": binds_dict(z),
                        "cons": cons, "kons_total": kons_total, "balans": balans}
    return zone_data, ntc_rows, binds_dict(None)


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, nargs="+",
                    help="Ett eller flera batch-prefix, t.ex. '17' eller '17 18' "
                         "(flera batchar i samma blad; run-nr skiljer dem)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = {}
    for p in args.prefix:
        runs.update(discover_runs(p))
    if not runs:
        raise SystemExit(f"Inga körningar funna för prefix {args.prefix}")
    print("Körningar:", {k: v.name for k, v in runs.items()})

    def make_row(zd_z):
        """En master-rad (kolumn-tuple → värde) ur en zons data."""
        row = {("Pris", "€/MWh"): zd_z["pris"]}
        for c in CAP_CARRIERS:
            row[("Kapacitet GW", c)] = zd_z["cap"][c]
        for c in PROD_ROWS:
            row[("Produktion TWh/år", c)] = zd_z["prod"][c]
        row[("Produktion TWh/år", "PRODUKTION TOTAL")] = zd_z["prod_total"]
        for c in LOADCAP_COLS:
            row[("Last-kapacitet GW", c)] = zd_z["loadcap"][c]
        for c in CONS_ROWS:
            row[("Konsumtion TWh/år", c)] = zd_z["cons"][c]
        row[("Konsumtion TWh/år", "KONSUMTION TOTAL")] = zd_z["kons_total"]
        row[("Kontroll", "BALANS")] = zd_z["balans"]
        return row

    master_rows, ntc_all, binds_list = [], [], []
    for run, d in runs.items():
        n = pypsa.Network(str(d / "network.nc"))
        zd, ntc, binds_norden = extract(n)
        for z in ZONES:
            master_rows.append(((run, z), make_row(zd[z])))
            binds_list.append(zd[z]["binds"])
        # Norden-aggregat (summa över zoner; pris = NaN)
        agg = {"pris": np.nan,
               "cap": {c: sum(zd[z]["cap"][c] for z in ZONES) for c in CAP_CARRIERS},
               "loadcap": {c: sum(zd[z]["loadcap"][c] for z in ZONES) for c in LOADCAP_COLS},
               "prod": {c: sum(zd[z]["prod"][c] for z in ZONES) for c in PROD_ROWS},
               "prod_total": sum(zd[z]["prod_total"] for z in ZONES),
               "cons": {c: sum(zd[z]["cons"][c] for z in ZONES) for c in CONS_ROWS},
               "kons_total": sum(zd[z]["kons_total"] for z in ZONES),
               "balans": sum(zd[z]["balans"] for z in ZONES)}
        master_rows.append(((run, "Norden"), make_row(agg)))
        binds_list.append(binds_norden)
        for r in ntc:
            ntc_all.append({"run": run, **r})

    idx = pd.MultiIndex.from_tuples([k for k, _ in master_rows], names=["run", "zon"])
    df = pd.DataFrame([v for _, v in master_rows], index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df = df.round(2)

    ntc_df = pd.DataFrame(ntc_all).set_index(["run", "länk"]).round(2)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        df.to_excel(xl, sheet_name="Master")
        ntc_df.to_excel(xl, sheet_name="NTC")
        # Röd siffra där ett kraftslags kapacitet ligger vid sitt tak (p_nom_opt ≈ p_nom_max).
        # Datarad 1 = headernivåer (2) + index-namnsrad (1) + 1 = rad 4 (1-based); datakol börjar
        # efter index-kolumnerna (run, zon).
        from openpyxl.styles import Font
        ws = xl.book["Master"]
        red = Font(color="FFCC0000")
        row0 = df.columns.nlevels + 2
        col0 = df.index.nlevels
        cap_pos = {c: df.columns.get_loc(("Kapacitet GW", c)) for c in CAP_CARRIERS}
        nred = 0
        for i, bd in enumerate(binds_list):
            for c, b in bd.items():
                if b:
                    ws.cell(row=row0 + i, column=col0 + cap_pos[c] + 1).font = red
                    nred += 1
    print(f"Skrev {out}  ({df.shape[0]} rader master, {ntc_df.shape[0]} NTC-rader, "
          f"{nred} tak-bindande kapaciteter markerade rött)")


if __name__ == "__main__":
    build()
