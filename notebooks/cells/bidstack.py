"""bidstack — UTBUDSKURVAN i en prisö vid EN given timme: hur zonen når sitt marginalpris.

Systercell till marginal.py. Där svarar `marginal_source()` på VEM som sätter priset;
här ritas HELA budstapeln bakom det svaret, så man ser hur långt in i merit order
efterfrågan skär och hur brant kurvan är just där.

Metod (samma prisö-logik som marginal.py):
  1. Prisön = zoner som delar pris via trängselfria AC-länkar. Utbudet är öns samlade,
     efterfrågan öns samlade — inom ön finns per definition ingen trängsel.
  2. Varje enhet i ön ger ett BLOCK: bredd = tillgänglig effekt, höjd = budet.
       generator      bud = marginal_cost[t],  bredd = p_nom·(p_max_pu − p_min_pu)
       hydro          bud = mc + WV/η + trappsteg,  ETT block per steg à p_nom/K
       batteri        bud = mc + WV/η,  bredd = p_nom
     Den OFLEXIBLA delen (p_min_pu > 0: kärnkraft, termisk, RoR, must-run värme, plus
     hydrons timgolv) ligger som en grå sockel längst till vänster — den produceras
     oavsett pris. Marknadsventilens exportben har p_min_pu < 0 och blir i stället
     TVUNGEN LAST, dvs. den flyttas till efterfrågesidan.
  3. Efterfrågan = last + nettoexport över öns yttre länkar (inkl. sektorlänkar:
     elektrolysör/värmepump/elpanna/EV drar effekt, H2-turbin och KVV matar in) +
     lagerladdning. Det är exakt nodbalansen för ön, så skärningen MÅSTE ligga där
     LP:t hamnade — avviker den är det ett tecken på att en restriktion utanför
     budstapeln binder (veckotak, dygnsgolv, SOC-gräns), och det skrivs ut.

Förutsätter bootstrap.py (n, cfg, plt, pd, ZONES, LABEL) och läser budtrappan ur
run_meta.txt precis som marginal.py/objcost.py.
"""

# Cellen kan importeras fristående (från explore_results.py) ELLER klistras in
# i en notebook-cell efter att bootstrap.py redan körts där (explore.ipynb).
if 'LABEL' not in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bootstrap import *  # noqa: F401,F403
import collections
import numpy as np
import pandas as pd
import re as _re
from matplotlib.patches import Patch as _Patch

# Kaskadförklaringen lånas från marginal.py — samma ö-logik, samma toleranser, ett
# enda ställe att underhålla.
# ⚠️ Laddningen MÅSTE ske här på cellens toppnivå. `exec(kod)` utan uttryckliga
# namnrum använder den ANROPANDE ramens faktiska globals; samma exec INUTI en funktion
# skulle i stället träffa funktionens `__globals__`, fastfruset vid def-tillfället —
# precis den fälla `_cellcode()`:s docstring i explore_results.py varnar för, och som
# en gång gav tyst fel LABEL.
if 'cascade_line' not in globals():
    MARGINAL_TS = None                    # hindrar marginal.py:s egen exempelutskrift
    exec(compile((ROOT / 'notebooks' / 'cells' / 'marginal.py').read_text(encoding='utf-8'),
                 str(ROOT / 'notebooks' / 'cells' / 'marginal.py'), 'exec'))

# Budtrappan: samma parser som marginal.py (default 3:36 syns bara som bidladder-3_36).
def _bs_ladder(label):
    f = ROOT / 'results' / label / 'run_meta.txt'
    if not f.is_file():
        return None
    txt = f.read_text(encoding='utf-8')
    m = _re.search(r'--hydro-bid-ladder\s+(\d+):(\d+(?:\.\d+)?)', txt)
    if m is None:
        if _re.search(r'\bno-bidladder\b', txt):
            return None
        m = _re.search(r'\bbidladder-(\d+)_(\d+(?:\.\d+)?)', txt)
    return (int(m.group(1)), float(m.group(2))) if m else None


BS_LADDER = _bs_ladder(LABEL)
BS_OFFSETS = [BS_LADDER[1] * ((k + 0.5) / BS_LADDER[0] - 0.5)
              for k in range(BS_LADDER[0])] if BS_LADDER else [0.0]
# Hydrons timgolv är en custom-restriktion (inte p_min_pu), så den måste läsas ur config.
BS_HYDRO_FLOOR = float(cfg.get('hydro_operation', {}).get('min_hourly_frac', 0.0) or 0.0)

BS_COLORS = {
    'nuclear': '#7e57c2', 'thermal': '#8d6e63',
    'wind_onshore': '#4fc3f7', 'wind_offshore': '#0277bd', 'solar': '#fdd835',
    'hydro': '#1e88e5', 'battery': '#26a69a', 'market': '#ef6c00',
    'gas': '#795548', 'DSR': '#ab47bc', 'slack': '#c62828', 'chp fuel': '#6d4c41',
}
BS_NAMES = {
    'nuclear': 'kärnkraft', 'thermal': 'termiskt',
    'wind_onshore': 'landvind', 'wind_offshore': 'havsvind', 'solar': 'sol',
    'hydro': 'vattenkraft', 'battery': 'batteri', 'market': 'marknadsventil',
    'gas': 'gas', 'DSR': 'industri-DSR', 'slack': 'lastbortkoppling (VOLL)',
}


def bid_stack(ts_str, zone='SE-S', ymax=None, xmax=None, tol_couple=0.01, ax=None,
              quiet=False):
    """Rita utbudskurvan för prisön som `zone` tillhör vid tidpunkten `ts_str`.

    `quiet=True` hoppar över graf och utskrift och returnerar bara en sammanfattning
    (dict) — för svep över många timmar."""
    req = pd.Timestamp(ts_str)
    snaps = pd.DatetimeIndex(n.snapshots)
    if req < snaps[0] or req > snaps[-1]:
        print(f"{req} ligger utanför körningens period ({snaps[0]} – {snaps[-1]}).")
        return
    ts = snaps[snaps.get_indexer([req], method='ffill')[0]]
    if ts != req:
        print(f"OBS: {req} faller i steget som börjar {ts} — använder den snapshoten.")
    mp = n.buses_t.marginal_price.loc[ts]

    # ── 1. Prisöar (identiskt med marginal.py) ────────────────────────────────
    parent = {z: z for z in ZONES}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for l in n.links.index:
        b0, b1 = n.links.at[l, 'bus0'], n.links.at[l, 'bus1']
        if b0 not in ZONES or b1 not in ZONES:
            continue
        sat = abs(abs(n.links_t.p0.at[ts, l]) - n.links.at[l, 'p_nom']) < 1e-3
        if not sat and abs(mp[b1] - mp[b0]) < tol_couple:
            parent[find(b0)] = find(b1)
    isl = sorted([z for z in ZONES if find(z) == find(zone)])
    price = mp[zone]

    def _pu(df, col, default):
        return df.at[ts, col] if col in df.columns else default

    # ── 2. Utbudsblock ────────────────────────────────────────────────────────
    # must = enheter som producerar OAVSETT pris (p_min_pu > 0, plus hydrons timgolv).
    # De ritas i sitt EGET kraftslags färg längst till vänster, inte i en grå klump:
    # kärnkraft är delvis must-run och delvis fri, och den delningen ska synas som
    # samma färg på båda ställena. Strömkraft och timgolv är vattenkraft och blir blå.
    must, blocks, fixed_dem = [], [], 0.0
    for g in n.generators[n.generators.bus.isin(isl)].index:
        pn = n.generators.at[g, 'p_nom_opt'] or n.generators.at[g, 'p_nom']
        lo = pn * _pu(n.generators_t.p_min_pu, g, n.generators.at[g, 'p_min_pu'])
        hi = pn * _pu(n.generators_t.p_max_pu, g, n.generators.at[g, 'p_max_pu'])
        car = n.generators.at[g, 'carrier']
        if lo > 1e-6:
            must.append((lo, f'{g} (must-run)', car))
        elif lo < -1e-6:                    # exportben: tvungen last, ej utbud
            fixed_dem += -lo
        if hi - lo > 1e-6:
            blocks.append((_pu(n.generators_t.marginal_cost, g,
                               n.generators.at[g, 'marginal_cost']), hi - lo, g, car, True))
    # Hur många timmar ett tidssteg är — urladdningen begränsas av ENERGI, inte bara
    # av effekt, och omräkningen MWh → MW går via steglängden.
    w_h = float(n.snapshot_weightings.objective.at[ts]) \
        if hasattr(n.snapshot_weightings, 'objective') else 1.0
    i_ts = list(snaps).index(ts)
    for s in n.storage_units[n.storage_units.bus.isin(isl)].index:
        pn = n.storage_units.at[s, 'p_nom']
        eff = n.storage_units.at[s, 'efficiency_dispatch']
        bud = _pu(n.storage_units_t.marginal_cost, s,
                  n.storage_units.at[s, 'marginal_cost']) \
            + n.storage_units_t.mu_energy_balance.at[ts, s] / eff
        # ⛔ Blockets bredd är min(effekt, det ENERGIN räcker till). Ett tomt batteri
        # har p_nom i namnplåt men kan inte leverera en enda MWh, och att rita det som
        # tillgängligt utbud gör kurvan fysiskt fel. Lagret som är kvar vid ingången av
        # steget är SOC i FÖREGÅENDE snapshot plus periodens tillrinning (noll för
        # batterier); uttaget kostar 1/eta MWh ur lagret per levererad MWh.
        soc_in = (n.storage_units_t.state_of_charge.iloc[i_ts - 1][s] if i_ts > 0
                  else float(n.storage_units.at[s, 'state_of_charge_initial']))
        inflow = (n.storage_units_t.inflow.at[ts, s]
                  if s in n.storage_units_t.inflow.columns else 0.0)
        cap = min(pn, max(0.0, (soc_in + inflow * w_h) * eff / w_h))
        # ⛔ Duger lagrets bud som PRISSÄTTARE? Bara om SOC är interiört. Vid taket
        # eller golvet binder energikapacitetsvillkoret och dess skuggpris läcker in i
        # mu_energy_balance, så budet mc + μ/η ANPASSAS till priset i stället för att
        # bestämma det. Kapaciteten är däremot verklig — blocket ritas, det får bara
        # inte utses till sättare. Samma villkor som i marginal.py. Mätt: 6,6 % av
        # zon-timmarna i run461 pekade annars ut ett lager vid en SOC-gräns.
        smax_s = pn * n.storage_units.at[s, 'max_hours']
        soc_ts = n.storage_units_t.state_of_charge.at[ts, s]
        soc_ok = 1.0 < soc_ts < smax_s * 0.999
        if n.storage_units.at[s, 'carrier'] == 'hydro':
            floor = min(pn * BS_HYDRO_FLOOR, cap)
            if floor > 1e-6:
                must.append((floor, f'{s} (timgolv)', 'hydro'))
            left = cap - floor                      # kvar att fördela på trappstegen
            for k, off in enumerate(BS_OFFSETS):
                wdt = pn / len(BS_OFFSETS) - (floor if k == 0 else 0.0)
                wdt = max(0.0, min(wdt, left)); left -= wdt
                nm = f'{s} [nivå {k + 1}]' if BS_LADDER else s
                if wdt > 1e-6:
                    blocks.append((bud + off, wdt, nm, 'hydro', soc_ok))
        else:
            # ⭐ Lager med laddningssida (p_min_pu < 0) hanteras som marknadsventilen:
            # laddningsbenet är TVUNGEN LAST i bredden och ett prisberoende block i
            # höjden. Buden skiljer sig — urladdning ger mc + μ/η_dis (man tar 1/η ur
            # lagret per levererad MWh), laddning mc + μ·η_sto (man får bara η kvar av
            # varje köpt MWh) — så de är två separata block, inte ett.
            # ⛔ Utan laddningsbenet försvinner sättaren i låglasttimmar: mätt på
            # SE-N/SE-S 2024-06-15 03:00, där batteriet är tomt (ingen urladdning alls
            # möjlig) men laddar — kurvan skar då 58,91 mot dualens 45,94.
            if n.storage_units.at[s, 'p_min_pu'] < 0:
                smax = pn * n.storage_units.at[s, 'max_hours']
                e_sto = n.storage_units.at[s, 'efficiency_store']
                chg = min(pn, max(0.0, (smax - soc_in) / (e_sto * w_h)))
                if chg > 1e-6:
                    fixed_dem += chg
                    blocks.append((_pu(n.storage_units_t.marginal_cost, s,
                                       n.storage_units.at[s, 'marginal_cost'])
                                   + n.storage_units_t.mu_energy_balance.at[ts, s] * e_sto,
                                   chg, f'{s} (laddning)', 'battery', soc_ok))
            if cap > 1e-6:
                blocks.append((bud, cap, f'{s} (urladdning)', 'battery', soc_ok))
    must.sort(key=lambda m: (m[2], -m[0]))
    fixed_sup = sum(m[0] for m in must)

    # ── 3. Efterfrågan = nodbalansen för ön ───────────────────────────────────
    load_isl = sum(n.loads_t.p.at[ts, ld] for ld in n.loads.index
                   if n.loads.at[ld, 'bus'] in isl)
    net_exp, sector = 0.0, collections.Counter()
    for l in n.links.index:
        for j, pcol in ((0, n.links_t.p0), (1, n.links_t.p1), (2, getattr(n.links_t, 'p2', None))):
            bcol = f'bus{j}' if j else 'bus0'
            if bcol not in n.links.columns or pcol is None or l not in pcol.columns:
                continue
            if n.links.at[l, bcol] in isl:
                v = pcol.at[ts, l]
                net_exp += v
                if n.links.at[l, 'carrier'] != 'AC':
                    sector[n.links.at[l, 'carrier']] += v
    charge = sum(max(0.0, -n.storage_units_t.p.at[ts, s])
                 for s in n.storage_units[n.storage_units.bus.isin(isl)].index)
    demand = load_isl + net_exp + charge + fixed_dem

    # ── 4. Skärning ───────────────────────────────────────────────────────────
    # ⭐ Efterfrågan kan landa på TVÅ sätt, och skillnaden ÄR marginal.py:s två utfall:
    #   INUTI ett block  → den enheten är delvis lastad, alltså lokalt marginell, och
    #                      öpriset MÅSTE vara dess bud.
    #   PÅ en blockgräns → kurvan är VERTIKAL där; varje pris mellan blocket under och
    #                      blocket över klarerar ön lika bra. Ön har ingen lokal sättare
    #                      och priset kommer utifrån (trängselränta över en mättad länk).
    blocks.sort(key=lambda b: b[0])
    # Must-run har inget meningsfullt bud — de körs vid vilket pris som helst, alltså
    # ligger kurvan formellt vid −∞ över deras bredd. De ritas därför på en egen nivå
    # strax UNDER billigaste riktiga bud, tydligt åtskild från merit order.
    lo_bid = min([b[0] for b in blocks] + [0.0])
    y_top = ymax if ymax else max(price * 1.6, 60)
    band = 0.06 * max(y_top - lo_bid, 1.0)      # bandhöjd i % av y-omfånget, ej fast tal
    Y_MUST = lo_bid - band
    x, cum = 0.0, []
    for w, nm, car in must:
        cum.append((x, x + w, Y_MUST, nm, car, True, False)); x += w
    for bid, w, nm, car, ok in blocks:
        cum.append((x, x + w, bid, nm, car, False, ok)); x += w
    EPS = 1.0                                   # MW — LP:ts egen feasibility-slack
    # ⭐ SLÅ IHOP ANGRÄNSANDE BLOCK MED SAMMA BUD innan skärningen prövas. Två enheter
    # som budar lika — t.ex. SE-N och SE-S hydro på steg 1, båda 46,91 — blir två
    # intilliggande block, och LP:t fördelar produktionen mellan dem godtyckligt. Utan
    # ihopslagning läses efterfrågan på skarven som "blockgräns", fast paret som helhet
    # är marginellt och priset ÄR deras gemensamma bud. Mätt i run461: 31,2 % av alla
    # vertikala fall var sådana skarvar, med gapets median bara 2,27 EUR/MWh.
    # Segmentet räknas som användbart om NÅGON medlem har ett brukbart bud.
    TOL_MERGE = 0.005                           # halva TOL_PRICE
    seg = []
    for x0, x1, bid, nm, car, mr, ok in cum:
        if seg and abs(seg[-1][2] - bid) < TOL_MERGE and seg[-1][5] == mr:
            g = seg[-1]
            seg[-1] = (g[0], x1, g[2], g[3] + [(nm, ok)], g[4] if g[4] == car else 'flera',
                       mr, g[6] or ok)
        else:
            seg.append((x0, x1, bid, [(nm, ok)], car, mr, ok))

    def _lbl(members):
        good = [m for m, o in members if o] or [m for m, _ in members]
        return good[0] if len(good) == 1 else f"{good[0]} + {len(members) - 1} till (samma bud)"

    setter, span, unreliable = None, None, None
    for x0, x1, bid, mem, car, mr, ok in seg:
        if x0 + EPS < demand < x1 - EPS:
            tup = (bid, _lbl(mem), car, mr, {m for m, _ in mem})
            if ok:
                setter = tup
            else:                               # must-run eller lager vid SOC-gräns
                unreliable = tup
            break
    edge = None
    if setter is None:
        # Vertikalt (eller landat i ett segment vars bud inte duger): intervallet spänns
        # av de närmaste segmenten vars bud FÅR användas.
        below = [b for b in seg if b[1] <= demand + EPS and b[6]]
        above = [b for b in seg if b[0] >= demand - EPS and b[6]]
        lo_seg = below[-1] if below else None
        hi_seg = above[0] if above else None
        span = (lo_seg[2] if lo_seg else -np.inf, hi_seg[2] if hi_seg else np.inf)
        # ⭐ EXAKT PÅ EN STEGKANT är det NORMALFALLET, inte en anomali: en enhet fyller
        # sitt steg precis och stannar, och priset är då NÄSTA stegs bud — vad en MWh
        # till skulle kosta. Mätt i run461: 122 av 518 vertikala fall, 115 av dem
        # vattenkraft som fyller ett trappsteg jämnt (t.ex. NO-N hydro nivå 1 full,
        # nivå 2 tom, pris = nivå 2:s bud på fyra decimaler).
        # ⚠️ Vilket av de två stegen som gäller avgörs med DUALEN, eftersom primalen är
        # likgiltig mellan dem. Det är alltså ingen oberoende validering av priset —
        # därför redovisas de här fallen separat, inte som "kurvan träffar dualen".
        for cand, lbl in ((hi_seg, 'nästa steg i merit order'),
                          (lo_seg, 'sista använda steget')):
            if cand is not None and abs(price - cand[2]) < 0.01:
                setter = (cand[2], _lbl(cand[3]), cand[4], False,
                          {m for m, _ in cand[3]})
                edge = lbl
                break
    p_hat = setter[0] if setter else np.nan
    # Saknar ön lokal sättare: hämta kaskadförklaringen från marginal.py, som skrivs
    # sist i textrapporten. Har ön en egen sättare blir cl None och inget skrivs.
    # ⛔ Prövad som text under x-axeln också — blev för rörigt, grafen ska bara bära
    # krysset (pris · effekt).
    # Fråga marginal.py när budstapeln INTE räcker: antingen för att den är vertikal,
    # eller för att skärningen inte matchar dualen (då binder något utanför stapeln).
    _explain = (not setter) or (not edge and abs(p_hat - price) >= 0.05)
    cl = cascade_line(str(ts), zone, tol_couple=tol_couple) if _explain else None

    # ── 5. Graf ───────────────────────────────────────────────────────────────
    if quiet:
        return dict(ts=ts, isl=isl, price=price, p_hat=p_hat, demand=demand,
                    setter=setter, span=span, cl=cl, cum=cum, seg=seg, must=must,
                    fixed_sup=fixed_sup, fixed_dem=fixed_dem,
                    unreliable=unreliable,
                    edge=edge,
                    kind=('stegkant' if edge else 'block') if setter
                         else ('obrukbart' if unreliable else 'vertikal'))
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(13, 6))
    ybase = Y_MUST - band                       # golv att fylla blocken från
    seen = set()
    for x0, x1, bid, nm, car, mr, ok in cum:
        ax.fill_between([x0 / 1e3, x1 / 1e3], bid, ybase, lw=0,
                        color=BS_COLORS.get(car, '#bdbdbd'),
                        hatch='///' if mr else None, edgecolor='white' if mr else None)
        ax.plot([x0 / 1e3, x1 / 1e3], [bid, bid], color='k', lw=0.6)
        seen.add(car)
    if fixed_sup > 0:
        ax.plot([0, fixed_sup / 1e3], [Y_MUST, Y_MUST], color='k', lw=0.8, ls='--')
        ax.text(fixed_sup / 2e3, Y_MUST - band / 2,
                f'must-run {fixed_sup/1e3:.1f} GW — körs oavsett pris, budet saknar betydelse',
                ha='center', va='center', fontsize=8, color='#222',
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    ax.axvline(demand / 1e3, color='k', ls='--', lw=1.6)
    ax.axhline(price, color='crimson', ls=':', lw=1.6)
    if setter:
        ax.annotate(f'marginell: {setter[1]}\n({BS_NAMES.get(setter[2], setter[2])}, bud {setter[0]:.1f})',
                    xy=(demand / 1e3, setter[0]), xytext=(12, 26), textcoords='offset points',
                    fontsize=9, arrowprops=dict(arrowstyle='->', lw=1))
    else:
        # Ingen lokal sättare → märk bara SJÄLVA KRYSSET med pris och mött effekt.
        # Förklaringen till varifrån priset kommer skrivs som text under grafen.
        ax.plot([demand / 1e3], [price], marker='x', ms=9, mew=2, color='crimson')
        ax.annotate(f'{price:.1f} EUR/MWh · {demand/1e3:.1f} GW',
                    xy=(demand / 1e3, price), xytext=(12, 10), textcoords='offset points',
                    fontsize=9, color='crimson', fontweight='bold',
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
    ax.set_xlim(0, xmax if xmax else max(demand * 1.45, fixed_sup * 1.1) / 1e3)
    ax.set_ylim(ybase, ymax if ymax else max(price * 1.6, 60))
    # ⚠️ Etiketterna måste sättas EFTER set_ylim/set_xlim — de ankras mot axelgränserna.
    y0, y1 = ax.get_ylim(); x1lim = ax.get_xlim()[1]
    _bb = dict(fc='white', ec='none', alpha=0.8, pad=1.5)
    ax.text(demand / 1e3, y1 - 0.02 * (y1 - y0), f' efterfrågan {demand/1e3:.1f} GW ',
            va='top', ha='left', fontsize=9, rotation=90, bbox=_bb)
    ax.text(x1lim, price, f' zonpris {price:.1f} ', color='crimson', va='bottom',
            ha='right', fontsize=9, fontweight='bold', bbox=_bb)
    ax.set_xlabel('kumulativ tillgänglig effekt (GW)')
    ax.set_ylabel('bud (EUR/MWh)')
    ax.set_title(f'Utbudskurva {{{", ".join(isl)}}} @ {ts}   ({LABEL})')
    # Egen legend: färgrutorna ska vara SOLIDA (kraftslaget), och streckningen ska ha
    # sin egen rad — annars ser det ut som om t.ex. all vattenkraft vore must-run.
    order = [c for c in BS_COLORS if c in seen] + sorted(c for c in seen if c not in BS_COLORS)
    handles = [_Patch(facecolor=BS_COLORS.get(c, '#bdbdbd'), label=BS_NAMES.get(c, c))
               for c in order]
    if fixed_sup > 0:
        handles.append(_Patch(facecolor='white', edgecolor='#333', hatch='///',
                              label='därav must-run'))
    ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=2)
    if own:
        plt.tight_layout(); plt.show()

    # ── 6. Textrapport ────────────────────────────────────────────────────────
    print(f"\n{'='*74}\nUTBUDSKURVA  prisö {{{', '.join(isl)}}}  @ {ts}   ({LABEL})\n{'='*74}")
    print(f"Zonpris (dual)          {price:9.2f} EUR/MWh")
    if setter and edge:
        print(f"Efterfrågan tar slut EXAKT på en stegkant vid {demand/1e3:.2f} GW.")
        print(f"   → priset = {edge}: {setter[1]} (bud {setter[0]:.2f})."
              f"  Intervallet var [{span[0]:.2f}, {span[1]:.2f}] — vilket av de två"
              f"\n     stegen som gäller avgörs av dualen, så detta är ingen oberoende kontroll.")
    elif setter:
        print(f"Kurvans skärning        {p_hat:9.2f} EUR/MWh   (inuti {setter[1]})"
              f"   → {'✔ stämmer' if abs(p_hat - price) < 0.05 else '⚠ AVVIKER'}")
    else:
        if unreliable:
            why = ('öns tvingade produktion räcker mer än väl' if unreliable[3]
                   else 'lagret står vid en SOC-gräns, så dess bud bär ingen information')
            bud_s = 'must-run' if unreliable[3] else f'bud {unreliable[0]:.2f}'
            print(f"⚠ Efterfrågan faller i {unreliable[1]} ({bud_s}) — {why};"
                  f" budet får inte utses till sättare.")
        lo_s = '−∞' if span[0] == -np.inf else f'{span[0]:.2f}'
        hi_s = '+∞' if span[1] == np.inf else f'{span[1]:.2f}'
        ok = (span[0] - 0.05) <= price <= (span[1] + 0.05)
        print(f"Kurvan är VERTIKAL vid efterfrågan: varje pris i [{lo_s}, {hi_s}] klarerar ön.")
        print(f"   → inget kraftslag är marginellt; sättaren ligger på EFTERFRÅGESIDAN"
              f" (sektorlänk) eller utanför ön — se noten sist."
              f"  {'✔ dualen ligger i intervallet' if ok else '⚠ dualen ligger UTANFÖR'}")
    print(f"\nEfterfrågan {demand/1e3:7.2f} GW  =  last {load_isl/1e3:.2f}"
          f"  + nettoexport {net_exp/1e3:+.2f}  + lagerladdning {charge/1e3:+.2f}"
          f"  + tvungen export {fixed_dem/1e3:+.2f}")
    if sector:
        print("   varav sektorlänkar (+ = drar el): "
              + ", ".join(f"{k} {v/1e3:+.2f}" for k, v in sorted(sector.items())))
    mr_by_car = collections.Counter()
    for w, nm, car in must:
        mr_by_car[car] += w
    print(f"Must-run {fixed_sup/1e3:6.2f} GW (körs oavsett pris): "
          + ", ".join(f"{BS_NAMES.get(k, k)} {v/1e3:.2f}" for k, v in mr_by_car.most_common()))
    flex_by_car, voll = collections.Counter(), 0.0
    for bid, w, nm, car, ok in blocks:
        # VOLL-generatorerna har p_nom 1e6 MW var — en modellteknisk oändlighet som
        # skulle dränka summan. De redovisas separat och ritas ändå (utanför xlim).
        if car == 'slack':
            voll += w
        else:
            flex_by_car[car] += w
    print(f"Prisberoende {sum(flex_by_car.values())/1e3:6.2f} GW: "
          + ", ".join(f"{BS_NAMES.get(k, k)} {v/1e3:.2f}" for k, v in flex_by_car.most_common())
          + (f"   [+ VOLL {voll/1e3:.0f} GW, obegränsad backstop]" if voll else ""))
    print(f"\nMerit order kring skärningen (bud | bredd | kumulativt):")
    idx = next((i for i, c in enumerate(cum) if c[1] >= demand - EPS), len(cum) - 1)
    for x0, x1, bid, nm, car, mr, ok in cum[max(0, idx - 4):idx + 4]:
        mark = (' ⟸ MARGINELL' if setter and nm in setter[4]
                else ('  ← efterfrågan faller på denna gräns'
                      if abs(x1 - demand) < EPS else ''))
        bid_s = 'must-run' if mr else f'{bid:9.2f}'
        print(f"  {bid_s:>9} | {(x1-x0)/1e3:6.2f} GW | {x1/1e3:7.2f} GW  {nm}{mark}")
    if setter and abs(p_hat - price) >= 0.05:
        print("\n⚠ Skärningen matchar inte dualen trots att efterfrågan hamnar INUTI ett block."
              "\n  Budstapeln beskriver bara kostnader och effektgränser — binder en restriktion"
              "\n  UTANFÖR den (hydrons dygnsgolv eller veckotak, en SOC-gräns) hamnar LP:t"
              "\n  någon annanstans. Den verkliga sättaren står i så fall nedan.")
    if _explain:
        if cl:
            print()
            for row in cl:
                print(f"  ⟹ {row}" if row is cl[0] else f"     {row}")
        else:
            print("\n⚠ marginal.py hittar en LOKAL sättare i ön trots att budstapeln är"
                  "\n  vertikal vid efterfrågan — de två kriterierna är inte samma sak"
                  "\n  (bud≈pris mot block-interiöritet). Kör marginal_source() för detaljer.")
    return None


BIDSTACK_TS = globals().get('BIDSTACK_TS', "2024-12-12 17:00")
BIDSTACK_ZONE = globals().get('BIDSTACK_ZONE', 'SE-S')
if BIDSTACK_TS:
    bid_stack(BIDSTACK_TS, BIDSTACK_ZONE)
