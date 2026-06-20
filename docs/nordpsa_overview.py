"""NordPSA översiktsfigur: sektorkopplad modell-schematik (vänster) + högerkolumn med
6-zons NTC-karta (topp) och energibalans-tabell (under). Tabellen = ebalance-cellens
country_balance, per land (TWh/år, medel över körningens år). Alla siffror i schematiken
(TWh-flöden, GW-kapaciteter) extraheras ur körningens network.nc → auto-uppdateras per run.

Användning:
    python docs/nordpsa_overview.py [RUN_LABEL] [utfil.png]
Default: RUN_LABEL=run142_svk2040mm_cont2040_3h, utfil=docs/nordpsa_overview.png
Körs cwd-oberoende (results/ + config/ hittas relativt repo-roten)."""
import sys
from pathlib import Path
import yaml
import pandas as pd
import pypsa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parent.parent
RUN  = sys.argv[1] if len(sys.argv) > 1 else "run142_svk2040mm_cont2040_3h"
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("nordpsa_overview.png")

ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]
cfg   = yaml.safe_load(open(ROOT / "config" / "zones.yaml"))

# ───────────────────────── energibalans + kapaciteter ─────────────────────────
COUNTRY_MAP = {"SE-N": "SE", "SE-S": "SE", "NO-N": "NO", "NO-S": "NO", "DK": "DK", "FI": "FI"}
COUNTRIES   = ["SE", "NO", "DK", "FI", "Norden"]
SOURCES     = ["hydro", "nuclear", "wind_onshore", "wind_offshore", "solar", "thermal", "gas", "slack"]
SHOW_ROWS   = SOURCES + ["prod_twh", "load_twh", "h2_elec", "heat_elec", "ev_elec",
                         "batt_net", "kont_export", "intern_export"]
ROW_LABELS  = {
    "hydro": "Vattenkraft", "nuclear": "Kärnkraft", "wind_onshore": "Vind onshore",
    "wind_offshore": "Vind offshore", "solar": "Sol", "thermal": "Termisk (KVV-el)",
    "gas": "Gas", "slack": "Slack (lastsk.)",
    "prod_twh": "PRODUKTION", "load_twh": "Last", "h2_elec": "H2-elektrolys",
    "heat_elec": "Värme-el (VP+panna)", "ev_elec": "EV-laddning", "batt_net": "Batteri (netto ut)",
    "kont_export": "Kont. export", "intern_export": "Intern export (NTC)",
}
BALANCE_SIGNS = {"prod_twh": +1, "batt_net": +1, "load_twh": -1, "h2_elec": -1,
                 "heat_elec": -1, "ev_elec": -1, "kont_export": -1, "intern_export": -1}


def load_run(res_label):
    """Returnerar (cyr, caps): per-lands-balans (DataFrame) + dict med schematik-annoteringar."""
    RES_ = ROOT / "results" / res_label
    nn = pypsa.Network(); nn.import_from_netcdf(RES_ / "network.nc")
    disp = pd.read_csv(RES_ / "dispatch_generators.csv", index_col=0, parse_dates=True)
    hyd  = pd.read_csv(RES_ / "dispatch_hydro.csv",      index_col=0, parse_dates=True)
    flw  = pd.read_csv(RES_ / "flows.csv",               index_col=0, parse_dates=True)
    nl = nn.loads_t.p_set.copy(); nl.index = pd.to_datetime(nl.index).tz_localize(None)
    nl = nl.reindex(disp.index)
    ld = pd.DataFrame({z: nl[f"{z} load"] for z in ZONES if f"{z} load" in nl.columns})
    dt_h    = (disp.index[1] - disp.index[0]).total_seconds() / 3600
    n_years = len(disp) * dt_h / 8760.0
    twh     = lambda s: float(s.sum()) * dt_h / 1e6 / n_years
    zero    = pd.Series(0.0, index=disp.index)

    ntc = [(l, nn.links.at[l, "bus0"], nn.links.at[l, "bus1"]) for l in nn.links.index
           if nn.links.at[l, "bus0"] in ZONES and nn.links.at[l, "bus1"] in ZONES]
    batt = nn.storage_units[nn.storage_units.carrier != "hydro"] if len(nn.storage_units) else nn.storage_units

    def zmkt(zone):
        cols = [g for g in disp.columns if g in nn.generators.index
                and nn.generators.at[g, "bus"] == zone and nn.generators.at[g, "carrier"] == "market"]
        return disp[cols].sum(axis=1) if cols else zero

    rows = []
    for zone in ZONES:
        r = {"country": COUNTRY_MAP[zone]}
        for c in SOURCES:
            if c == "hydro":
                rcols = [g for g in disp.columns if g in nn.generators.index
                         and nn.generators.at[g, "bus"] == zone and nn.generators.at[g, "carrier"] == "hydro"]
                s = hyd.get(f"{zone} hydro", zero).clip(lower=0) \
                    + (disp[rcols].clip(lower=0).sum(axis=1) if rcols else zero)
            else:
                # Summera ALLA generatorer med carrier c i zonen (t.ex. 'nuclear' +
                # 'nuclear exp', 'wind_onshore' + 'wind_new') — ej bara exakt '{zon} {c}'.
                ccols = [g for g in disp.columns if g in nn.generators.index
                         and nn.generators.at[g, "bus"] == zone and nn.generators.at[g, "carrier"] == c]
                s = disp[ccols].clip(lower=0).sum(axis=1) if ccols else zero
            r[c] = twh(s)
        eta_el = float(((((cfg.get("heat") or {}).get("zones") or {}).get(zone, {}).get("chp")) or {}).get("eta_el", 0.0))
        r["thermal"] += twh(flw.get(f"{zone} chp", zero).clip(lower=0)) * eta_el
        r["heat_elec"] = twh(flw.get(f"{zone} heat elboiler", zero).clip(lower=0)
                             + flw.get(f"{zone} heat hp", zero).clip(lower=0))
        r["h2_elec"]   = twh(flw.get(f"{zone} electrolyser", zero).clip(lower=0))
        r["ev_elec"]   = twh(sum((flw.get(f"{zone} EV {c} charger", zero).clip(lower=0)
                                  for c in ("car", "heavy")), zero))
        r["kont_export"] = -twh(zmkt(zone))
        ie = zero.copy()
        for l, b0, b1 in ntc:
            f = flw.get(l, zero)
            if   b0 == zone: ie = ie + f
            elif b1 == zone: ie = ie - f
        r["intern_export"] = twh(ie)
        bz = [s for s in batt.index if nn.storage_units.at[s, "bus"] == zone]
        r["batt_net"] = (float(nn.storage_units_t.p[bz].to_numpy().sum()) * dt_h / 1e6 / n_years) if bz else 0.0
        r["load_twh"] = twh(ld[zone]) if zone in ld.columns else 0.0
        rows.append(r)

    d = pd.DataFrame(rows)
    d["prod_twh"] = d[SOURCES].sum(axis=1)
    cyr = d.groupby("country")[SHOW_ROWS].sum()
    cyr.loc["Norden"] = cyr.sum()

    # ── kapaciteter / flöden för schematik-annoteringar (nordiska totaler) ──
    G = nn.generators; L = nn.links; SU = nn.storage_units
    pn = lambda idx: float(L.loc[idx, "p_nom_opt"].sum()) / 1e3  # GW
    lk = lambda suf: pn([l for l in L.index if l.endswith(suf)])
    hy = SU[SU.carrier == "hydro"]; ba = SU[SU.carrier != "hydro"]
    chp_el = sum(L.at[l, "p_nom_opt"]
                 * float(((((cfg.get("heat") or {}).get("zones") or {}).get(l.rsplit(" ", 1)[0], {}).get("chp")) or {}).get("eta_el", 0.0))
                 for l in L.index if l.endswith("chp")) / 1e3
    def sumload(suf):  # energi/år för laster med givet suffix (tidsvar. i loads_t, annars statiskt p_set)
        tot = 0.0
        for ln in nn.loads.index:
            if not ln.endswith(suf):
                continue
            tot += twh(nl[ln]) if ln in nl.columns else float(nn.loads.at[ln, "p_set"]) * 8760 / 1e6
        return tot
    ror = sum(twh(disp.get(f"{z} hydro", zero).clip(lower=0)) for z in ZONES)
    thermal_pure = sum(twh(disp.get(f"{z} thermal", zero).clip(lower=0)) for z in ZONES)
    heat_mustrun = sum(twh(disp.get(g, zero).clip(lower=0)) for g in G.index
                       if G.at[g, "carrier"] == "heat mustrun")
    St = nn.stores
    store_gwh = lambda suf: float(St[St.bus.str.endswith(suf)].e_nom_opt.sum()) / 1e3
    emp = nn.stores_t.get("e_min_pu")
    ev_cols = [c for c in (emp.columns if emp is not None else []) if "EV" in c]
    ev_floor = (float(emp[ev_cols].min().min()), float(emp[ev_cols].max().max())) if ev_cols else (0.0, 0.0)

    # ── F/E/ET-klassning: F=fast (ej extendable), E=expanderbar m. luft,
    #    ET=expanderbar & potentialtaket binder i ≥1 zon (opt≈p_nom_max/e_nom_max).
    def _cls(df, ocol, ccol, ecol):
        if df.empty:
            return ""
        ext = df[df[ecol]]
        if ext.empty:
            return "F"
        binds = (ext[ocol] >= 0.99 * ext[ccol]) & (ext[ccol] < 1e11)
        return "ET" if bool(binds.any()) else "E"
    cg = lambda c: _cls(G[G.carrier == c], "p_nom_opt", "p_nom_max", "p_nom_extendable")
    cs = lambda c: _cls(SU[SU.carrier == c], "p_nom_opt", "p_nom_max", "p_nom_extendable")
    cl = lambda suf: _cls(L.loc[[l for l in L.index if l.endswith(suf)]], "p_nom_opt", "p_nom_max", "p_nom_extendable")
    cst = lambda suf: _cls(St[St.bus.str.endswith(suf)], "e_nom_opt", "e_nom_max", "e_nom_extendable")
    tags = dict(
        nuclear=cg("nuclear"), onw=cg("wind_onshore"), offw=cg("wind_offshore"), sol=cg("solar"),
        thermal=cg("thermal"), gas=cg("gas"), market=cg("market"), heat_mustrun=cg("heat mustrun"),
        ror=cg("hydro") or "F", hydro=cs("hydro"), batt=cs(ba.carrier.iloc[0]) if len(ba) else "F",
        electrolyser=cl("electrolyser"), hp=cl("heat hp"), elboiler=cl("heat elboiler"),
        chp=cl("chp"), ev=cl("EV car charger"),
        h2_store=cst(" H2"), heat_store=cst(" heat"), ev_store=cst(" EV car"), load="F",
    )
    N = cyr.loc["Norden"]
    caps = dict(
        nuclear=N["nuclear"], onw=N["wind_onshore"], offw=N["wind_offshore"], sol=N["solar"],
        gas=N["gas"], netexp=N["kont_export"], thermal_pure=thermal_pure, ror=ror,
        hydro_gen=N["hydro"], ellast=N["load_twh"],
        magasin_vol=float((hy.p_nom_opt * hy.max_hours).sum()) / 1e6,  # TWh
        magasin_gw=float(hy.p_nom_opt.sum()) / 1e3,
        batt_gw=float(ba.p_nom_opt.sum()) / 1e3,
        batt_gwh=float((ba.p_nom_opt * ba.max_hours).sum()) / 1e3,
        h2_demand=sumload(" H2 load"), heat_demand=sumload(" heat load"),
        ev_demand=sumload(" EV car drive") + sumload(" EV heavy drive"),
        gw_electrolyser=lk("electrolyser"), gw_turbine=lk("h2 turbine") + lk("turbine"),
        gw_hp=lk("heat hp"), gw_elboiler=lk("heat elboiler"), gw_chp_el=chp_el,
        gw_ev=lk("EV car charger") + lk("EV heavy charger"),
        heat_mustrun=heat_mustrun, h2_store=store_gwh(" H2"),
        heat_store=store_gwh(" heat"), ev_store=store_gwh(" EV car"),
        ev_floor_lo=ev_floor[0], ev_floor_hi=ev_floor[1], tags=tags,
    )
    return cyr, caps


print(f"Läser {RUN} …")
bal, K = load_run(RUN)
T = K["tags"]
tag = lambda k: f" ({T[k]})" if T.get(k) else ""

# ───────────────────────────── figur ─────────────────────────────
C = dict(
    nuc="#8e44ad", onw="#27ae60", offw="#16a085", sol="#f1c40f", hyd="#2980b9",
    ror="#5dade2", therm="#7f5539", gas="#7f8c8d", mkt="#34495e", batt="#e67e22",
    h2="#1abc9c", heat="#e74c3c", ev="#2c5f2d", boil="#d35400", bus="#1b2631",
    h2bus="#0e6655", heatbus="#922b21", evbus="#1e5631", load="#c0392b", text="#1b2631",
    bio="#6aa84f",
)
XMAX = 142.0
fig, ax = plt.subplots(figsize=(29.5, 13.0))
ax.set_xlim(0, XMAX); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, label, color, val=None, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8",
                 linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.16, zorder=3))
    txt = label if val is None else f"{label}\n{val}"
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=C["text"], zorder=4, weight="bold")

def busbar(x, y, w, h, label, color):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0, facecolor=color, zorder=3))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=12.5,
            color="white", rotation=90, weight="bold", zorder=4)

def arrow(x0, y0, x1, y1, color, bidir=False, lw=2.2):
    style = "<|-|>" if bidir else "-|>"
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=16, linewidth=lw, color=color, zorder=2,
                 shrinkA=1, shrinkB=1))

# ---- ELBUSS (spine) ----
SP = 37.0
busbar(SP, 12, 3.2, 82, "ELBUSS  (AC)", C["bus"])
SPR = SP + 3.2
ax.text(SP + 1.6, 13.3, "×6", ha="center", fontsize=11, style="italic",
        color="white", weight="bold", zorder=5)

# ---- GENERERING (ingen RoR-box: i run141 är all vattenkraft magasin) ----
gx, gw, gh = 1.5, 14.5, 7.5
gens = [
    ("Kärnkraft (must-run)", C["nuc"], f"{K['nuclear']:.0f} TWh{tag('nuclear')}"),
    ("Vind, land", C["onw"], f"{K['onw']:.0f} TWh{tag('onw')}"),
    ("Vind, hav", C["offw"], f"{K['offw']:.0f} TWh{tag('offw')}"),
    ("Sol-PV", C["sol"], f"{K['sol']:.0f} TWh{tag('sol')}"),
    ("Termisk must-run (industri)", C["therm"], f"{K['thermal_pure']:.0f} TWh{tag('thermal')}"),
    ("Gas CCGT-CCS (topp)", C["gas"], f"{K['gas']:.0f} TWh{tag('gas')}"),
    ("Marknad (kontinentventil)", C["mkt"], f"netto-exp {K['netexp']:.0f} TWh{tag('market')}"),
]
if K["ror"] > 0.5:   # lägg in RoR-box bara om körningen har älvkraft
    gens.insert(4, ("Vattenkraft, älv (RoR)", C["ror"], f"{K['ror']:.0f} TWh{tag('ror')}"))
gy_top = 88.0
gstep = (gy_top - 19.0) / (len(gens) - 1)
for i, (lab, col, val) in enumerate(gens):
    gy = gy_top - i * gstep
    box(gx, gy, gw, gh, lab, col, val, fs=9.2)
    arrow(gx + gw, gy + gh/2, SP, gy + gh/2, col, bidir=(i == len(gens) - 1))

# ---- LAGRING (magasin = vattenkraftkällan i run141) ----
box(26.0, 0.6, 12.5, 8.8, "Vattenmagasin", C["hyd"],
    f"{K['hydro_gen']:.0f} TWh/år · {K['magasin_vol']:.0f} TWh vol{tag('hydro')}", fs=7.4)
arrow(32.2, 9.4, 32.2, 12.0, C["hyd"], bidir=True)
box(40.0, 0.6, 12.0, 8.8, "Batteri (exogen)", C["batt"],
    f"{K['batt_gw']:.0f} GW · {K['batt_gwh']:.0f} GWh{tag('batt')}", fs=7.6)
arrow(46.0, 9.4, 46.0, 12.0, C["batt"], bidir=True)

# ---- EL-LAST ----
box(44.0, 86.5, 16.0, 8.0, "El-last", C["load"],
    f"{K['ellast']:.0f} TWh{tag('load')}\n(inflex. + industri/datacenter)", fs=8.6)
arrow(SPR, 90.5, 44.0, 90.5, C["load"])

# ---- OMVANDLING (mellan elbuss & sektorbussar → GW-kapacitet) ----
cx, cw, ch = 52.0, 13.0, 6.6
conv = [("Elektrolysör", C["h2"], 79.0, "to_h2", f"{K['gw_electrolyser']:.1f} GW{tag('electrolyser')}")]
if K["gw_turbine"] > 0.05:
    conv.append(("H2-turbin", C["h2"], 70.0, "from_h2", f"{K['gw_turbine']:.1f} GW"))
conv += [
    ("Värmepump", C["heat"], 53.0, "to_heat", f"{K['gw_hp']:.1f} GW{tag('hp')}"),
    ("El-panna", C["boil"], 44.5, "to_heat", f"{K['gw_elboiler']:.1f} GW{tag('elboiler')}"),
    ("KVV bakpress (el+värme)", C["therm"], 35.5, "chp", f"{K['gw_chp_el']:.1f} GW el{tag('chp')}"),
    ("EV-laddare", C["ev"], 23.0, "to_ev", f"{K['gw_ev']:.0f} GW{tag('ev')}"),
]
conv_y = {}
for lab, col, yc, kind, val in conv:
    conv_y[lab] = yc
    box(cx, yc - ch/2, cw, ch, lab, col, val, fs=8.4)
    if kind == "from_h2":
        arrow(cx, yc, SPR, yc, col)
    elif kind == "chp":
        arrow(cx, yc + 1.4, SPR, yc + 1.4, col)
    else:
        arrow(SPR, yc, cx, yc, col)

# ---- SEKTORBUSSAR ----
BB = 70.0; BBR = 73.2
busbar(BB, 66, 3.2, 21, "VÄTGASBUSS", C["h2bus"])
busbar(BB, 30, 3.2, 28, "VÄRMEBUSS", C["heatbus"])
busbar(BB, 17, 3.2, 12, "EV-BUSS", C["evbus"])
arrow(cx + cw, conv_y["Elektrolysör"], BB, conv_y["Elektrolysör"], C["h2"])
if "H2-turbin" in conv_y:
    arrow(BB, conv_y["H2-turbin"], cx + cw, conv_y["H2-turbin"], C["h2"])
arrow(cx + cw, conv_y["Värmepump"], BB, conv_y["Värmepump"], C["heat"])
arrow(cx + cw, conv_y["El-panna"], BB, conv_y["El-panna"], C["boil"])
arrow(cx + cw, conv_y["KVV bakpress (el+värme)"] - 1.4, BB,
      conv_y["KVV bakpress (el+värme)"] - 1.4, C["therm"])
arrow(cx + cw, conv_y["EV-laddare"], BB, conv_y["EV-laddare"], C["ev"])

# ---- SEKTOR-LAGER & LASTER ----
sx, sw, sh = 78.5, 15.0, 6.2
box(sx, 80.0, sw, sh, "H2-lager", C["h2"], f"{K['h2_store']:.0f} GWh{tag('h2_store')}", fs=9.0)
arrow(BBR, 83.1, sx, 83.1, C["h2"], bidir=True)
box(sx, 68.0, sw, sh, "H2-last (P2X)", C["h2"], f"{K['h2_demand']:.0f} TWh{tag('load')}", fs=9.0)
arrow(BBR, 71.1, sx, 71.1, C["h2"])
_hh = float((cfg.get("heat") or {}).get("store_hours", 6))
box(sx, 50.0, sw, sh, "Värmelager", C["heat"], f"{K['heat_store']:.0f} GWh{tag('heat_store')} ({_hh:.0f}h·topp)", fs=7.4)
arrow(BBR, 53.1, sx, 53.1, C["heat"], bidir=True)
box(sx, 40.0, sw, sh, "Värme must-run", C["bio"], f"bio/avfall · {K['heat_mustrun']:.0f} TWh{tag('heat_mustrun')}", fs=8.2)
arrow(sx, 43.1, BBR, 43.1, C["bio"])
box(sx, 30.0, sw, sh, "Värme-last (fjärrvärme)", C["heatbus"], f"{K['heat_demand']:.0f} TWh{tag('load')}", fs=9.0)
arrow(BBR, 33.1, sx, 33.1, C["heatbus"])
box(sx, 21.0, sw, sh, "EV-lager", C["ev"],
    f"{K['ev_store']:.0f} GWh{tag('ev_store')} · golv {K['ev_floor_lo']*100:.0f}–{K['ev_floor_hi']*100:.0f}%", fs=7.6)
arrow(BBR, 24.1, sx, 24.1, C["ev"], bidir=True)
box(sx, 12.5, sw, sh, "EV-körlast", C["evbus"], f"{K['ev_demand']:.0f} TWh{tag('load')}", fs=9.0)
arrow(BBR, 18.5, sx, 18.5, C["evbus"])

# ================= HÖGERKOLUMN: zonkarta (topp) + tabell (under) =================
ax.plot([99.5, 99.5], [2, 98], color="#cccccc", lw=1.0, zorder=1)

# --- zonkarta ---
ax.text((103.0 + 138.0) / 2, 96.5, "6 zoner · NTC-länkar", ha="center",
        fontsize=12, weight="bold", color=C["text"])
zpos = {"NO-N": (110, 90), "SE-N": (124, 90), "FI": (137, 87),
        "NO-S": (110, 78), "SE-S": (124, 76), "DK": (119, 70)}
links = [("NO-N","SE-N"), ("NO-N","NO-S"), ("SE-N","SE-S"), ("SE-N","FI"),
         ("NO-S","SE-S"), ("SE-S","FI"), ("SE-S","DK"), ("NO-S","DK")]
for a, b in links:
    (xa, ya), (xb, yb) = zpos[a], zpos[b]
    ax.plot([xa, xb], [ya, yb], color="#888", lw=2.0, zorder=2)
for z in ["SE-S", "NO-S", "DK", "FI"]:
    xz, yz = zpos[z]
    ax.plot([xz, xz + 3.2], [yz - 2.8, yz - 5.2], color=C["mkt"], lw=1.8, ls=":", zorder=2)
for z, (xz, yz) in zpos.items():
    ax.add_patch(Circle((xz, yz), 3.0, facecolor=C["bus"], edgecolor="white",
                 linewidth=1.4, zorder=4))
    ax.text(xz, yz, z, ha="center", va="center", fontsize=8.5, color="white",
            weight="bold", zorder=5)
ax.text((103.0 + 138.0) / 2, 65.5, "··· = kontinentventil (DE-LU/NL/GB/EE/LT/PL)",
        ha="center", fontsize=9, style="italic", color=C["text"])

# --- energibalans-tabell ---
ax.text((103.0 + 138.0) / 2, 62.8, f"Energibalans  (TWh/år)  —  {RUN}",
        ha="center", fontsize=12, weight="bold", color=C["text"])
rows_order = SHOW_ROWS + ["__balance__"]
balance_row = {c: sum(sgn * bal.loc[c, row] for row, sgn in BALANCE_SIGNS.items())
               for c in COUNTRIES}
table_rows = []
for row in rows_order:
    if row == "__balance__":
        vals = [f"{(0.0 if abs(balance_row[c]) < 1e-6 else balance_row[c]):.1f}" for c in COUNTRIES]
        table_rows.append(["BALANS (≈0)", *vals])
    else:
        table_rows.append([ROW_LABELS[row], *[f"{bal.loc[c, row]:.1f}" for c in COUNTRIES]])

tbl = ax.table(cellText=table_rows, colLabels=["", *COUNTRIES],
               colWidths=[0.40, 0.115, 0.115, 0.115, 0.115, 0.14],
               cellLoc="right", bbox=[0.706, 0.035, 0.285, 0.56], transform=ax.transAxes)
tbl.auto_set_font_size(False); tbl.set_fontsize(9.0)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor("#1b2631"); cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    if c == 0 and r > 0:
        cell.get_text().set_ha("left")
    rn = table_rows[r - 1][0] if r > 0 else ""
    if rn == "PRODUKTION":
        cell.set_facecolor("#eaf2f8"); cell.get_text().set_weight("bold")
    if rn == "BALANS (≈0)":
        cell.set_facecolor("#fdebd0"); cell.get_text().set_weight("bold")

# ---- titel ----
ax.text(XMAX / 2, 99.3, "NordPSA — sektorkopplad expansionsmodell  (struktur per zon, "
        "nordisk översikt)", ha="center", fontsize=16, weight="bold", color=C["text"])
ax.text(50, 97.2, f"Årsvärden: {RUN} · SvK 2040 MM · 3h · nordiska totaler",
        ha="center", fontsize=11, style="italic", color="#555")
ax.text(50, 95.0, "(F) fast   ·   (E) expanderbar (luft kvar)   ·   "
        "(ET) expanderbar – potentialtaket binder i ≥1 zon",
        ha="center", fontsize=9.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#bbb"))

plt.savefig(OUT, dpi=155, bbox_inches="tight", facecolor="white")
print("sparad:", OUT)
