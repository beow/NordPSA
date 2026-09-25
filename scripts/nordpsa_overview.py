"""NordPSA översiktsfigur: sektorkopplad modell-schematik (vänster) + högerkolumn med
6-zons NTC-karta (topp) och energibalans-tabell (under). Tabellen =
nordpsa.analysis.energy_balance.country_balance, per land (TWh/år, medel över körningens år). Alla siffror i schematiken
(TWh-flöden, GW-kapaciteter) extraheras ur körningens network.nc → auto-uppdateras per run.

Användning:
    python scripts/nordpsa_overview.py RUN [utfil.png] [--exp KÖRNING | --no-exp]

⭐ DISPATCH-KÖRNINGAR: kapaciteterna är frysta (p_nom_extendable = False på allt), så
expanderbarheten går inte att läsa ur körningen själv — fast/utbyggt-uppdelningen hamnar
helt på "fast", alla taggar blir (F) och inget potentialtak kan flaggas. Figuren visar då
RÄTT kapaciteter men kan inte säga vad som BYGGDES. Skriptet slår därför automatiskt upp
källexpansionen ur run_config.yaml (`run.capacities`; äldre körningar: `--dispatch X` i
run_meta.txt) och läser expanderbarheten därifrån, medan energi, flöden och priser alltid
kommer från körningen som visas.
    --exp KÖRNING   hämta expanderbarheten från en annan körning än källkörningen
    --no-exp        stäng av; läs allt ur körningen själv (gamla beteendet)
RUN kan vara hela katalognamnet (run143_svk2040mm_nucexp_3h) ELLER bara prefixet
(run143 / run143_) — då matchas det entydigt mot results/.
Default-utfil: docs/nordpsa_overview_<runNNN>.png
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


def resolve_run(arg):
    """Tar hela katalognamnet eller bara prefixet (run143) → entydig results/-katalog."""
    results = ROOT / "results"
    if (results / arg).is_dir():
        return arg
    hits = sorted(p.name for p in results.glob(f"{arg}*") if p.is_dir())
    if not hits:
        sys.exit(f"Ingen results/-katalog matchar '{arg}'. Finns: "
                 + ", ".join(sorted(p.name for p in results.glob('run*') if p.is_dir())))
    if len(hits) > 1:
        sys.exit(f"'{arg}' är tvetydig — matchar: {', '.join(hits)}")
    return hits[0]


def expansion_source(res_label):
    """Källkörningen bakom en dispatch: `run.capacities` i run_config.yaml, eller för
    körningar gjorda före run_config.yaml, `--dispatch X` i run_meta.txt.

    ⭐ VARFÖR: i en dispatch är kapaciteterna FRYSTA — `freeze_capacities_from` sätter
    p_nom_extendable = False på allt. Då blir p_nom_min/p_nom_max meningslösa och de tre
    ställen som läser dem tappar sin information: fast/utbyggt-uppdelningen (allt hamnar
    på 'fast'), (F)/(E)/(ET)-taggarna (allt blir 'F') och takbindningen (aldrig röd).
    Figuren visar alltså RÄTT kapaciteter men kan inte längre säga vad som BYGGDES.
    Strukturen hämtas därför från expansionen, energin från dispatchen.
    """
    rc = ROOT / "results" / res_label / "run_config.yaml"
    if rc.exists():
        cap = ((yaml.safe_load(rc.read_text()) or {}).get("run") or {}).get("capacities")
        if cap and cap not in ("optimized", "config"):
            return cap if (ROOT / "results" / cap / "network.nc").exists() else None
        return None
    meta = ROOT / "results" / res_label / "run_meta.txt"
    if not meta.exists():
        return None
    for line in meta.read_text().splitlines():
        if not line.startswith("argv:"):
            continue
        # Sök på TOKEN, inte delsträng: run343:s --desc innehåller ordet "dispatch", vilket
        # matchade `"--dispatch" in line` men saknade token → ValueError i .index().
        parts = line.split()
        if "--dispatch" not in parts:
            continue
        i = parts.index("--dispatch")
        if i + 1 < len(parts):
            cand = parts[i + 1].strip("'\"")
            return cand if (ROOT / "results" / cand / "network.nc").exists() else None
    return None


argv = [a for a in sys.argv[1:]]
EXP_OVERRIDE = None
if "--exp" in argv:                       # explicit: hämta strukturen från denna körning
    i = argv.index("--exp")
    EXP_OVERRIDE = argv[i + 1] if i + 1 < len(argv) else None
    del argv[i:i + 2]
NO_EXP = "--no-exp" in argv               # tvinga: läs strukturen ur körningen själv
if NO_EXP:
    argv.remove("--no-exp")

if not argv:
    sys.exit(__doc__)
RUN  = resolve_run(argv[0])
# Default-utfil får run-prefix-suffix (nordpsa_overview_run145.png) → skriver ej över andra runs.
OUT  = Path(argv[1]) if len(argv) > 1 else \
       ROOT / "docs" / f"nordpsa_overview_{RUN.split('_')[0]}.png"

ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]
cfg   = yaml.safe_load(open(ROOT / "config" / "zones.yaml"))

# ───────────────────────── energibalans + kapaciteter ─────────────────────────
# Raddefinitioner OCH balanslogiken ligger i paketet (nordpsa/analysis/energy_balance.py).
sys.path.insert(0, str(ROOT))
from nordpsa.analysis.energy_balance import (country_balance, COUNTRIES, SOURCES,   # noqa: E402
                                             SHOW_ROWS, ROW_LABELS, DISPLAY_NEG, BALANCE_SIGNS)


def load_run(res_label, struct_label=None):
    """Returnerar (cyr, caps): per-lands-balans (DataFrame) + dict med schematik-annoteringar.

    `struct_label` = körningen som EXPANDERBARHETEN läses ur (se expansion_source()).
    Energi, flöden och priser kommer alltid från `res_label`. None = läs allt ur körningen.
    """
    RES_ = ROOT / "results" / res_label
    nn = pypsa.Network(); nn.import_from_netcdf(RES_ / "network.nc")
    if struct_label:
        ns = pypsa.Network(); ns.import_from_netcdf(ROOT / "results" / struct_label / "network.nc")
    else:
        ns = nn
    disp = pd.read_csv(RES_ / "dispatch_generators.csv", index_col=0, parse_dates=True)
    flw  = pd.read_csv(RES_ / "flows.csv",               index_col=0, parse_dates=True)
    nl = nn.loads_t.p_set.copy(); nl.index = pd.to_datetime(nl.index).tz_localize(None)
    nl = nl.reindex(disp.index)
    dt_h    = (disp.index[1] - disp.index[0]).total_seconds() / 3600
    n_years = len(disp) * dt_h / 8760.0
    twh     = lambda s: float(s.sum()) * dt_h / 1e6 / n_years
    zero    = pd.Series(0.0, index=disp.index)

    ntc = [(l, nn.links.at[l, "bus0"], nn.links.at[l, "bus1"]) for l in nn.links.index
           if nn.links.at[l, "bus0"] in ZONES and nn.links.at[l, "bus1"] in ZONES]
    # VRE-tillgänglighet (p_max_pu × p_nom_opt) → curtailment (skillnad mot dispatchat).
    pmax = nn.generators_t.p_max_pu.copy()
    pmax.index = pd.to_datetime(pmax.index).tz_localize(None)
    pmax = pmax.reindex(disp.index)

    def zmkt(zone):
        cols = [g for g in disp.columns if g in nn.generators.index
                and nn.generators.at[g, "bus"] == zone and nn.generators.at[g, "carrier"] == "market"]
        return disp[cols].sum(axis=1) if cols else zero

    # Balanstabellen kommer från nordpsa.analysis.energy_balance, inte från en kopia här.
    # En kopia fanns och hade ärvt exakt de buggar som rättades i originalet 2026-08-16: loads_t.p_set
    # missar statiskt satta laster (H2 typ 1), namnuppslag missade "electrolyser ef",
    # och DSR saknades som källa — tillsammans 35,1 TWh/år fel i run361. Två kopior av
    # samma identitet är en garanti för att bara den ena blir rättad.
    cyr = country_balance(res_label)

    # ── kapaciteter / flöden för schematik-annoteringar (nordiska totaler) ──
    G = nn.generators; L = nn.links; SU = nn.storage_units
    # Strukturramar: expanderbarhet + potentialtak. Samma index som G/L/SU när
    # dispatchen byggts ur samma konfiguration; okända namn faller tillbaka på nn.
    GS = ns.generators; LS = ns.links; SUS = ns.storage_units
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
    # RoR (älvkraft) = hydro-CARRIER-generatorer ({zon} hydro_ror), skild från magasin
    # (StorageUnit). Matcha på carrier, ej namn (generatorerna heter hydro_ror).
    ror_cols = [g for g in disp.columns if g in nn.generators.index
                and nn.generators.at[g, "carrier"] == "hydro"]
    ror = twh(disp[ror_cols].clip(lower=0).sum(axis=1)) if ror_cols else 0.0

    def split_fixed_exp(carrier):
        """(fast_TWh, exp_TWh) för ett kraftslag: fast = befintlig flotta (ej-extendable
        gen + p_nom_min-andelen av extendable), exp = utbyggnaden över p_nom_min.
        Befintlig & nybyggd kapacitet delar CF-profil/curtailment → fördela produktionen
        proportionellt mot p_nom_min / p_nom_opt per generator (separata kärnkr-gen → 0/1)."""
        fixed = expn = 0.0
        for g in G.index[G.carrier == carrier]:
            if g not in disp.columns:
                continue
            e = twh(disp[g].clip(lower=0))
            # Expanderbarheten ur STRUKTURkörningen; energin ur den som visas.
            gs = GS if g in GS.index else G
            po = float(gs.at[g, "p_nom_opt"])
            if gs.at[g, "p_nom_extendable"] and po > 1e-6:
                fr = min(max(float(gs.at[g, "p_nom_min"]) / po, 0.0), 1.0)
                fixed += e * fr; expn += e * (1.0 - fr)
            else:
                fixed += e
        return fixed, expn

    def cap_fixed_exp(carrier):
        """(fast_GW, exp_GW) för ett kraftslag, samma uppdelning som split_fixed_exp:
        fast = ej-extendable p_nom_opt + p_nom_min för extendable, exp = p_nom_opt − p_nom_min."""
        fixed = expn = 0.0
        for g in G.index[G.carrier == carrier]:
            gs = GS if g in GS.index else G
            po = float(G.at[g, "p_nom_opt"])
            if gs.at[g, "p_nom_extendable"]:
                pmin = min(float(gs.at[g, "p_nom_min"]), po)
                fixed += pmin; expn += po - pmin
            else:
                fixed += po
        return fixed / 1e3, expn / 1e3

    gw_split = {k: cap_fixed_exp(c) for k, c in (
        ("nuclear", "nuclear"), ("onw", "wind_onshore"), ("offw", "wind_offshore"),
        ("sol", "solar"), ("thermal", "thermal"), ("gas", "gas"), ("ror", "hydro"))}

    nuclear_fixed, nuclear_exp = split_fixed_exp("nuclear")
    onw_fixed,  onw_exp  = split_fixed_exp("wind_onshore")
    offw_fixed, offw_exp = split_fixed_exp("wind_offshore")
    sol_fixed,  sol_exp  = split_fixed_exp("solar")
    thermal_pure = sum(twh(disp.get(f"{z} thermal", zero).clip(lower=0)) for z in ZONES)
    heat_mustrun = sum(twh(disp.get(g, zero).clip(lower=0)) for g in G.index
                       if G.at[g, "carrier"] == "heat mustrun")
    St = nn.stores; StS = ns.stores
    store_gwh = lambda suf: float(St[St.bus.str.endswith(suf)].e_nom_opt.sum()) / 1e3
    emp = nn.stores_t.get("e_min_pu")
    ev_cols = [c for c in (emp.columns if emp is not None else []) if "EV" in c]
    ev_floor = (float(emp[ev_cols].min().min()), float(emp[ev_cols].max().max())) if ev_cols else (0.0, 0.0)

    # Årlig energi genom omvandlingslänkarna (TWh/år, länk-p0 = AC-sidans uttag/insättning)
    lke = lambda suf: twh(sum((flw[c].clip(lower=0) for c in flw.columns if c.endswith(suf)), zero))
    e_chp_el = sum(twh(flw[c].clip(lower=0))     # KVV-EL = bränsleflöde × eta_el
                   * float(((((cfg.get("heat") or {}).get("zones") or {}).get(c.rsplit(" ", 1)[0], {}).get("chp")) or {}).get("eta_el", 0.0))
                   for c in flw.columns if c.endswith("chp"))

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
    cg = lambda c: _cls(GS[GS.carrier == c], "p_nom_opt", "p_nom_max", "p_nom_extendable")
    cs = lambda c: _cls(SUS[SUS.carrier == c], "p_nom_opt", "p_nom_max", "p_nom_extendable")
    cl = lambda suf: _cls(LS.loc[[l for l in LS.index if l.endswith(suf)]], "p_nom_opt", "p_nom_max", "p_nom_extendable")
    cst = lambda suf: _cls(StS[StS.bus.str.endswith(suf)], "e_nom_opt", "e_nom_max", "e_nom_extendable")
    tags = dict(
        nuclear=cg("nuclear"), onw=cg("wind_onshore"), offw=cg("wind_offshore"), sol=cg("solar"),
        thermal=cg("thermal"), gas=cg("gas"), market=cg("market"), heat_mustrun=cg("heat mustrun"),
        ror=cg("hydro") or "F", hydro=cs("hydro"), batt=cs(ba.carrier.iloc[0]) if len(ba) else "F",
        electrolyser=cl("electrolyser"), hp=cl("heat hp"), elboiler=cl("heat elboiler"),
        chp=cl("chp"), ev=cl("EV car charger"),
        h2_store=cst(" H2"), heat_store=cst(" heat"), ev_store=cst(" EV car"), load="F",
    )
    N = cyr.loc["Norden"]
    # Kontinent-NTC per zon → {zon: {land: MW}} (summa över imp-stegen i elasticitetsstegen;
    # speglar körningens faktiska p_nom inkl. ev. demand-scenario-override, t.ex. DK-DE 4000).
    mkt_ntc = {}
    Gm = nn.generators
    for gname in Gm.index[Gm.carrier == "market"]:
        if "imp" not in gname:
            continue
        conn = gname.rsplit(" ", 1)[0]              # "DK DE imp1" → "DK DE"
        country = conn.split()[-1]                  # "DE"
        zone = Gm.at[gname, "bus"]
        mkt_ntc.setdefault(zone, {}).setdefault(country, 0.0)
        mkt_ntc[zone][country] += float(Gm.at[gname, "p_nom"])
    # %bindande per kontinentanslutning = andel timmar då hela kabeln går på ±NTC
    # (|nettoflöde Σ(imp+exp)| ≥ 0.99 × Σimp-steg). Mellansteg = ej bindande (headroom kvar).
    mkt_bind = {}
    mkt_net  = {}
    mconns = {}
    for gname in Gm.index[Gm.carrier == "market"]:
        if gname in disp.columns:
            mconns.setdefault(gname.rsplit(" ", 1)[0], []).append(gname)
    for conn, gens in mconns.items():
        cap = sum(float(Gm.at[g, "p_nom"]) for g in gens if "imp" in g)   # = full NTC
        net = disp[gens].sum(axis=1)
        bp = float((net.abs() >= 0.99 * cap).mean() * 100) if cap > 0 else 0.0
        mkt_bind.setdefault(Gm.at[gens[0], "bus"], {})[conn.split()[-1]] = bp
        # NETTOIMPORT per anslutning, TWh/år. market-generatorn har p_min_pu = −1, så
        # p > 0 = inflöde till zonen (import) och p < 0 = utflöde (export). Tecknet är
        # alltså redan rätt: + = nettoimport, − = nettoexport.
        mkt_net.setdefault(Gm.at[gens[0], "bus"], {})[conn.split()[-1]] = twh(net)
    # Tak-bindning per land×kraftslag: p_nom_opt ≈ p_nom_max (bara EXTENDABLE komponenter →
    # VRE + kärnkraft + gas; hydro/RoR/termik/slack är ej extendable → aldrig). Rödmarkeras i
    # balanstabellens produktionsrader. Logik = "NÅGON ZON vid taket" (landet flaggas om minst
    # en av dess zoner är maxad i kraftslaget), så t.ex. SE Sol blir röd när SE-S sitter på taket.
    country_zones = {"SE": ["SE-N", "SE-S"], "NO": ["NO-N", "NO-S"],
                     "DK": ["DK"], "FI": ["FI"], "Norden": ZONES}
    def _capbinds(carrier, zs):
        for z in zs:
            m = (GS.carrier == carrier) & GS.p_nom_extendable & (GS.bus == z)
            if not m.any():
                continue
            opt, mx = GS.loc[m, "p_nom_opt"].sum(), GS.loc[m, "p_nom_max"].sum()
            if 0 < mx < 1e12 and opt >= 0.999 * mx:
                return True
        return False
    cap_binds = {(co, c): _capbinds(c, zs) for co, zs in country_zones.items() for c in SOURCES}
    caps = dict(
        nuclear=N["nuclear"], onw=N["wind_onshore"], offw=N["wind_offshore"], sol=N["solar"],
        nuclear_fixed=nuclear_fixed, nuclear_exp=nuclear_exp,
        onw_fixed=onw_fixed, onw_exp=onw_exp, offw_fixed=offw_fixed, offw_exp=offw_exp,
        sol_fixed=sol_fixed, sol_exp=sol_exp, gw_split=gw_split,
        gas=N["gas"], netexp=N["kont_export"], thermal_pure=thermal_pure, ror=ror,
        hydro_gen=N["hydro"], hydro_reservoir=N["hydro"], ellast=N["load_twh"],
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
        e_electrolyser=lke("electrolyser"), e_turbine=lke("h2 turbine") + lke("turbine"),
        e_hp=lke("heat hp"), e_elboiler=lke("heat elboiler"), e_chp_el=e_chp_el,
        e_ev=lke("EV car charger") + lke("EV heavy charger"),
        ntc_mw={frozenset((b0, b1)): float(nn.links.at[l, "p_nom"]) for l, b0, b1 in ntc},
        # %bindande timmar per intern länk (|flöde| ≥ 0.99 × NTC), som mastersheet.
        ntc_bind={frozenset((b0, b1)): (float((flw.get(l, zero).abs()
                  >= 0.99 * float(nn.links.at[l, "p_nom"])).mean() * 100)
                  if float(nn.links.at[l, "p_nom"]) > 0 else 0.0) for l, b0, b1 in ntc},
        mkt_ntc=mkt_ntc, mkt_bind=mkt_bind, mkt_net=mkt_net, cap_binds=cap_binds,
        # zon-snittpris: rakt tidsmedel (= LMA-troget per zon, = LMA:s årsmedelpris per
        # elområde). Kapat vid NordPool-taket 4000 EUR/MWh för att utesluta ofysikaliska
        # scarcity-spikar (FI 15195-artefakten); påverkar bara FI (övriga zoner ≤1980).
        zprice={z: float(nn.buses_t.marginal_price[z].clip(upper=4000).mean())
                for z in ZONES if z in nn.buses_t.marginal_price.columns},
    )
    return cyr, caps


STRUCT = None if NO_EXP else (EXP_OVERRIDE or expansion_source(RUN))
print(f"Läser {RUN} …")
if STRUCT:
    print(f"  expanderbarhet (F/E/ET, fast+utbyggt, takbindning) från {STRUCT}")
bal, K = load_run(RUN, STRUCT)
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

# ---- GENERERING (RoR-box läggs in nedan om körningen har älvkraft) ----
gx, gw, gh = 1.5, 14.5, 7.5

def split_val(key, total, fixed, expn):
    """Etikett efter var produktionen kommer ifrån (ej bara komponentens expanderbarhet):
       fast & exp → 'fast (F) + exp (E/ET)'; nästan allt exp → 'exp (E/ET)';
       nästan allt befintligt (t.ex. ej-utbyggd DK-havsvind) → 'total (F)'."""
    if expn > 0.5 and fixed > 0.5:
        return f"{fixed:.0f} TWh (F)  +  {expn:.0f} TWh{tag(key)}"
    if expn > 0.5:
        return f"{expn:.0f} TWh{tag(key)}"
    return f"{total:.0f} TWh (F)"

def gw_val(key):
    """Kapacitetsrad: 'fast GW (F) + utbyggt GW (E/ET)', eller bara det som finns."""
    fixed, expn = K["gw_split"][key]
    if expn > 0.05 and fixed > 0.05:
        return f"{fixed:.1f} GW (F)  +  {expn:.1f} GW{tag(key)}"
    if expn > 0.05 or fixed <= 0.05:          # ev. inget byggt alls: visa ändå E/ET-taggen
        return f"{expn:.1f} GW{tag(key)}"
    return f"{fixed:.1f} GW (F)"

gens = [
    ("Kärnkraft (must-run)", C["nuc"], split_val("nuclear", K['nuclear'], K['nuclear_fixed'], K['nuclear_exp']) + "\n" + gw_val("nuclear")),
    ("Vind, land", C["onw"], split_val("onw", K['onw'], K['onw_fixed'], K['onw_exp']) + "\n" + gw_val("onw")),
    ("Vind, hav", C["offw"], split_val("offw", K['offw'], K['offw_fixed'], K['offw_exp']) + "\n" + gw_val("offw")),
    ("Sol-PV", C["sol"], split_val("sol", K['sol'], K['sol_fixed'], K['sol_exp']) + "\n" + gw_val("sol")),
    ("Termisk must-run", C["therm"], f"{K['thermal_pure']:.0f} TWh{tag('thermal')}\n" + gw_val("thermal")),
    ("Gas CCGT-CCS", C["gas"], f"{K['gas']:.0f} TWh{tag('gas')}\n" + gw_val("gas")),
    ("Kontinent exp/imp", C["mkt"], f"netto-exp {K['netexp']:.0f} TWh{tag('market')}"),
]
if K["ror"] > 0.5:   # lägg in RoR-box bara om körningen har älvkraft
    gens.insert(4, ("Vattenkraft, älv (RoR)", C["ror"], f"{K['ror']:.0f} TWh{tag('ror')}\n" + gw_val("ror")))
gy_top = 88.0
gstep = (gy_top - 19.0) / (len(gens) - 1)
for i, (lab, col, val) in enumerate(gens):
    gy = gy_top - i * gstep
    box(gx, gy, gw, gh, lab, col, val, fs=9.2)
    arrow(gx + gw, gy + gh/2, SP, gy + gh/2, col, bidir=(i == len(gens) - 1))

# ---- LAGRING (magasin = reservoardelen; RoR redovisas separat i genereringskolumnen) ----
box(26.0, 0.6, 12.5, 8.8, "Vattenmagasin", C["hyd"],
    f"{K['hydro_reservoir']:.0f} TWh/år · {K['magasin_vol']:.0f} TWh vol{tag('hydro')}", fs=7.4)
arrow(32.2, 9.4, 32.2, 12.0, C["hyd"], bidir=True)
box(40.0, 0.6, 12.0, 8.8, "Batteri (exogen)", C["batt"],
    f"{K['batt_gw']:.0f} GW · {K['batt_gwh']:.0f} GWh{tag('batt')}", fs=7.6)
arrow(46.0, 9.4, 46.0, 12.0, C["batt"], bidir=True)

# ---- EL-LAST ----
box(44.0, 86.5, 16.0, 8.0, "El-last", C["load"],
    f"{K['ellast']:.0f} TWh{tag('load')}\n(hushåll/industri/DC/EV/förluster)", fs=8.6)
arrow(SPR, 90.5, 44.0, 90.5, C["load"])

# ---- OMVANDLING (mellan elbuss & sektorbussar → GW-kapacitet) ----
cx, cw, ch = 52.0, 13.0, 7.4
conv = [("Elektrolysör", C["h2"], 79.0, "to_h2",
         f"{K['gw_electrolyser']:.1f} GW{tag('electrolyser')}\n{K['e_electrolyser']:.0f} TWh")]
if K["gw_turbine"] > 0.05:
    conv.append(("H2-turbin", C["h2"], 70.0, "from_h2",
                 f"{K['gw_turbine']:.1f} GW\n{K['e_turbine']:.0f} TWh"))
conv += [
    ("Värmepump", C["heat"], 53.0, "to_heat",
     f"{K['gw_hp']:.1f} GW{tag('hp')}\n{K['e_hp']:.0f} TWh"),
    ("El-panna", C["boil"], 44.5, "to_heat",
     f"{K['gw_elboiler']:.1f} GW{tag('elboiler')}\n{K['e_elboiler']:.0f} TWh"),
    ("Kraftvärme (el+värme)", C["therm"], 35.5, "chp",
     f"{K['gw_chp_el']:.1f} GW el{tag('chp')}\n{K['e_chp_el']:.0f} TWh el"),
    ("EV-laddare", C["ev"], 23.0, "to_ev",
     f"{K['gw_ev']:.0f} GW{tag('ev')}\n{K['e_ev']:.0f} TWh"),
]
conv_y = {}
for lab, col, yc, kind, val in conv:
    conv_y[lab] = yc
    box(cx, yc - ch/2, cw, ch, lab, col, val, fs=7.4)
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
arrow(cx + cw, conv_y["Kraftvärme (el+värme)"] - 1.4, BB,
      conv_y["Kraftvärme (el+värme)"] - 1.4, C["therm"])
arrow(cx + cw, conv_y["EV-laddare"], BB, conv_y["EV-laddare"], C["ev"])

# ---- SEKTOR-LAGER & LASTER ----
sx, sw, sh = 78.5, 15.0, 6.2
box(sx, 80.0, sw, sh, "H2-lager", C["h2"], f"{K['h2_store']:.0f} GWh{tag('h2_store')}", fs=9.0)
arrow(BBR, 83.1, sx, 83.1, C["h2"], bidir=True)
box(sx, 68.0, sw, sh, "H2-last (P2X)", C["h2"], f"{K['h2_demand']:.0f} TWh{tag('load')}", fs=9.0)
arrow(BBR, 71.1, sx, 71.1, C["h2"])
box(sx, 50.0, sw, sh, "Värmelager", C["heat"], f"{K['heat_store']:.0f} GWh{tag('heat_store')}", fs=7.4)
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
ax.text((103.0 + 138.0) / 2, 96.5, "6 zoner · NTC-länkar & Exp/Imp-ventiler (GW · %bindande · nettoimport TWh/år)", ha="center",
        fontsize=12, weight="bold", color=C["text"])
zpos = {"NO-N": (110, 90), "SE-N": (124, 90), "FI": (137, 87),
        "NO-S": (110, 78), "SE-S": (124, 76), "DK": (119, 70)}
links = [("NO-N","SE-N"), ("NO-N","NO-S"), ("SE-N","SE-S"), ("SE-N","FI"),
         ("NO-S","SE-S"), ("SE-S","FI"), ("SE-S","DK"), ("NO-S","DK")]
ntc_mw = K.get("ntc_mw", {})
ntc_bind = K.get("ntc_bind", {})
# Etikett-position = länkens mittpunkt, utom där noder trängs (SE-S–DK: DK ligger rakt
# under SE-S → flytta etiketten till den fria fickan nedanför SE-S / höger om DK).
mid_override = {frozenset(("SE-S", "DK")): (123.0, 70.6)}
for a, b in links:
    (xa, ya), (xb, yb) = zpos[a], zpos[b]
    ax.plot([xa, xb], [ya, yb], color="#888", lw=2.0, zorder=2)
    mw = ntc_mw.get(frozenset((a, b)))
    if mw:
        mx, my = mid_override.get(frozenset((a, b)), ((xa + xb) / 2, (ya + yb) / 2))
        ax.text(mx, my, f"{mw/1e3:.1f}",
                ha="center", va="center", fontsize=6.8, color="#444", zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        bp = ntc_bind.get(frozenset((a, b)))   # %bindande i rött, bredvid (höger om) kapacitetssiffran
        if bp is not None:
            ax.text(mx + 0.8, my, f"{bp:.0f}%", ha="left", va="center",
                    fontsize=6.0, color="#c0392b", weight="bold", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75))
# Per-zon kontinentventil (prickad stubbe) + NTC-kapacitet (GW) till varje exportland noden
# når. Riktning/etikettläge tunade mot zonkartans trånga layout (NO-S pekar vänster mot öppen yta).
mkt_ntc = K.get("mkt_ntc", {})
mkt_bind = K.get("mkt_bind", {})
mkt_net  = K.get("mkt_net", {})
mkt_dir = {"SE-S": +1, "NO-S": -1, "DK": +1, "FI": +1}          # x-riktning på ventilstubben
mkt_lbl_pos = {  # (x_vänster, y_topp) för det staplade per-land-blocket (boxad GW + rött %)
    "NO-S": (100.0, 73.6), "SE-S": (129.0, 70.4),
    # DK har bara fönstret 122,0 .. 129,0: noden ligger på x = 119 med radie 3,0 och
    # SE-S:s block börjar på 129,0. Blocket är lx .. lx+MKT_NET_DX+1,6 brett (5,8), så
    # 122,7 är det enda som ryms med marginal åt båda håll. Mätt, inte gissat — 121,0
    # lade rutorna bakom noden och 123,0 (värdet före nettoimport-kolumnen) sköt in i SE-S.
    "DK":   (122.7, 68.8), "FI":   (135.0, 80.2),
}
MKT_ROW_H = 1.7
MKT_NET_DX = 4.2      # x-offset för nettoimport-kolumnen, se mkt_lbl_pos['DK']
for z in ["SE-S", "NO-S", "DK", "FI"]:
    xz, yz = zpos[z]
    ax.plot([xz, xz + 3.2 * mkt_dir[z]], [yz - 2.8, yz - 5.2], color=C["mkt"], lw=1.8, ls=":", zorder=2)
    d = mkt_ntc.get(z, {})
    db = mkt_bind.get(z, {})
    dn = mkt_net.get(z, {})
    lx, ly = mkt_lbl_pos[z]
    for i, (c, mw) in enumerate(sorted(d.items(), key=lambda kv: -kv[1])):
        y = ly - i * MKT_ROW_H
        ax.text(lx, y, f"{c} {mw/1e3:.1f}", ha="left", va="center", fontsize=5.6,
                color=C["mkt"], zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=C["mkt"], lw=0.4, alpha=0.9))
        bp = db.get(c)
        if bp is not None:
            ax.text(lx + 2.5, y, f"{bp:.0f}%", ha="left", va="center", fontsize=5.6,
                    color="#c0392b", weight="bold", zorder=3)
        # Nettoimport TWh/år: + in i Norden, − ut. Blått för import, grått för export, så
        # riktningen syns utan att läsa tecknet.
        ne = dn.get(c)
        if ne is not None:
            ax.text(lx + MKT_NET_DX, y, f"{ne:+.1f}", ha="left", va="center", fontsize=5.6,
                    color=("#1f6f9c" if ne >= 0 else "#7f8c8d"), weight="bold", zorder=3)
zprice = K.get("zprice", {})
price_off = {  # (dx, dy) finjustering per zon; default (0, -3.9). Knuffa enskilda undan krock.
    "NO-N": (-5, 0), "SE-N": (5, 1), "FI": (-5,0),
    "NO-S": (-5, 0), "SE-S": (5, 0), "DK": (-5,0),
}
for z, (xz, yz) in zpos.items():
    ax.add_patch(Circle((xz, yz), 3.0, facecolor=C["bus"], edgecolor="white",
                 linewidth=1.4, zorder=4))
    ax.text(xz, yz, z, ha="center", va="center", fontsize=8.5, color="white",
            weight="bold", zorder=5)
    if z in zprice:
        dx, dy = price_off.get(z, (0, -3.9))
        ax.text(xz + dx, yz + dy, f"{zprice[z]:.0f} EUR/MWh", ha="center", va="top",
                fontsize=6.4, color=C["text"], weight="bold", zorder=5)

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
        sgn = -1 if row in DISPLAY_NEG else 1   # sänkor visas negativa (kosmetiskt)
        table_rows.append([ROW_LABELS[row], *[f"{sgn * bal.loc[c, row]:.1f}" for c in COUNTRIES]])

tbl = ax.table(cellText=table_rows, colLabels=["", *COUNTRIES],
               colWidths=[0.40, 0.115, 0.115, 0.115, 0.115, 0.14],
               cellLoc="right", bbox=[0.706, 0.035, 0.285, 0.56], transform=ax.transAxes)
tbl.auto_set_font_size(False); tbl.set_fontsize(9.0)
cap_binds = K.get("cap_binds", {})
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor("#1b2631"); cell.get_text().set_color("white")
        cell.get_text().set_weight("bold")
    if c == 0 and r > 0:
        cell.get_text().set_ha("left")
    rn = table_rows[r - 1][0] if r > 0 else ""
    # Bind mot NYCKLARNA, inte mot etiketterna: raden hette "PRODUKTION" i den lokala
    # kopian och "PRODUKTION TOTAL" i energy_balance, så en literal sträng slutade
    # matcha TYST när raddefinitionerna delades.
    if rn in (ROW_LABELS["prod_twh"], ROW_LABELS["kons_total"]):
        cell.set_facecolor("#eaf2f8"); cell.get_text().set_weight("bold")
    if rn == ROW_LABELS["curt"]:      # upplysningsrad utanför balansen → kursiv
        cell.get_text().set_style("italic"); cell.get_text().set_color("#6b7b85")
    if rn == "BALANS (≈0)":
        cell.set_facecolor("#fdebd0"); cell.get_text().set_weight("bold")
    # Rödmarkera produktionssiffror där kraftslagets kapacitet ligger vid sitt tak (p_nom_opt≈max).
    if r > 0 and c > 0 and (r - 1) < len(SHOW_ROWS):
        if cap_binds.get((COUNTRIES[c - 1], SHOW_ROWS[r - 1])):
            cell.get_text().set_color("#c0392b"); cell.get_text().set_weight("bold")

# ---- titel ----
ax.text(40, 99.3, "NordPSA — sektorkopplad expansionsmodell  (struktur per zon, "
        "nordisk översikt)", ha="center", fontsize=16, weight="bold", color=C["text"])
ax.text(40, 97.2, f"Årsvärden: {RUN} · SvK 2040 MM · nordiska totaler",
        ha="center", fontsize=11, style="italic", color="#555")
ax.text(75, 5.0, "(F) fast   ·   (E) expanderbar (luft kvar)   ·   "
        "(ET) expanderbar – potentialtaket binder i ≥1 zon",
        ha="center", fontsize=9.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f4f4", edgecolor="#bbb"))

plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("sparad:", OUT)
