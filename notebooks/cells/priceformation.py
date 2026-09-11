"""priceformation — vad sätter priset över HELA körningen, per PRISÖ.

Skiljer sig från marginal.py (EN given timme) genom att aggregera över samtliga
snapshots. Sedan 2026-09-09 använder den samma ö-logik som marginal.py:

  1. PRISÖAR. Zoner som är förbundna med en icke-mättad AC-länk och har samma pris
     (inom TOL_COUPLE) delar ETT pris och därmed EN marginalenhet. Öarna byggs per
     snapshot med union-find över de interna länkarna.
  2. SÄTTARE PER Ö. Bland ALLA enheter i öns zoner söks den som är interiör (strikt
     mellan sina gränser) och vars bud ligger närmast öpriset. Hittas ingen inom
     TOL_PRICE bokförs timmen som 'sätts utanför prisön' — ön är då trängselinstängd
     och priset kommer från en annan ö över en mättad länk (kaskaden dit följs inte
     här; se marginal.py). ⚠️ Etiketten hette tidigare 'ingen LOKAL sättare', vilket
     var fel: sökningen är ö-vid, inte zon-lokal — DK kan få sin sättare ur NO-S.
  3. Sättarens carrier tilldelas SAMTLIGA zoner i ön.

⛔ Varför detta behövdes: den gamla versionen sökte bara i zonens EGNA enheter och
hade ingen avståndströskel — den utsåg alltid en vinnare, även när bästa lokala bud
låg 8-40 EUR/MWh från priset. Mätt på run450: bara 27-40 % av timmarna hade en lokal
kandidat inom 0,5 EUR/MWh, och 35-45 % låg över 10. Tabellen svarade alltså på
"vilken lokal teknik ligger närmast marginalen", inte på "vad sätter priset". Det gör
stor skillnad här, eftersom SE-N/SE-S är priskopplade ~99 % av timmarna och Sverige
därmed har EN marginalenhet, inte två.

Bud per enhetstyp:
  generator          marginal_cost (tidsvarierande om sådan finns)
  lager (hydro/batt) egen marginal_cost + vattenvärde/verkningsgrad (water_value.csv)
  sektorlänk         härlett ur motpartsbussens pris, se _link_bid()

Extraherad ur notebooks/explore_results.py (f.d. cell 4), 2026-08-21.

Förutsätter bootstrap.py (globala: n, ZONES, LABEL, dispatch, hydro_d, prices,
water_value, dt_h, pd, plt).
"""

# Cellen kan importeras fristående (från explore_results.py) ELLER klistras in
# i en notebook-cell efter att bootstrap.py redan körts där (explore.ipynb).
# Guard: skippar importen om bootstrap redan satt sina globaler i detta namnrum.
if 'LABEL' not in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bootstrap import *  # noqa: F401,F403

import numpy as np
import re as _re


def _ladder(label):
    """(K, BREDD) för --hydro-bid-ladder ur run_meta.txt, eller None om trappan var av.
    Samma parser som notebooks/cells/objcost.py: sedan trappan blev DEFAULT står den
    normalt INTE i `argv:`, bara i `flaggor:` som `bidladder-K_BREDD`."""
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


LADDER = _ladder(LABEL)
# Budtrappan delar reservoarens uttag i K nivåer med offset_k = BREDD·((k+½)/K − ½).
# ⛔ Utan den termen jämförs hydro mot FEL nivå: mätt på run450 (3:36, offsetter
# −12/0/+12) var medianavståndet 6,2-18,4 EUR/MWh, och med offsetterna 0,00 i NO-N
# och NO-S. Hydrons fall från 96 % till 43 % i den första ö-versionen var alltså till
# stor del rekonstruktionsfel, inte ett resultat.
OFFSETS = [LADDER[1] * ((k + 0.5) / LADDER[0] - 0.5)
           for k in range(LADDER[0])] if LADDER else [0.0]

REF_ZONE   = 'SE-S'    # <-- zon för vattenvärdesgrafen (tabellen tar alla)
# ⭐ Skärpta 2026-09-09 från 0,6 (ärvt från marginal.py, = hydrons VOM). I ett LP har
# kopplade zoner EXAKT samma pris och den marginella enhetens bud ÄR priset — 0,6 var
# ett slaskutrymme för rekonstruktionsfel, inte en egenskap hos modellen. Det slasket
# behövs inte längre sedan hydrons bud tar med budtrappans offsetter (se nedan).
TOL_COUPLE = 0.01      # prisskillnad under detta = samma prisö
TOL_PRICE  = 0.01      # bud längre än detta från öpriset räknas INTE som sättare

snap = dispatch.index

mc_t = n.generators_t.marginal_cost.reindex(snap) \
    if not n.generators_t.marginal_cost.empty else pd.DataFrame(index=snap)
smc_t = n.storage_units_t.marginal_cost.reindex(snap) \
    if not n.storage_units_t.marginal_cost.empty else pd.DataFrame(index=snap)
mcl_t = n.links_t.marginal_cost.reindex(snap) \
    if not n.links_t.marginal_cost.empty else pd.DataFrame(index=snap)


def _pnom(df, name):
    """Kapacitet att mäta marginalitet mot. p_nom_opt i expansion, p_nom annars —
    p_nom ensamt gör on_margin FEL i en expansionskörning, där den utbyggda
    kapaciteten är p_nom_opt och p_nom bara är golvet."""
    v = df.at[name, 'p_nom_opt'] if 'p_nom_opt' in df.columns else 0.0
    return float(v) if v and v > 0 else float(df.at[name, 'p_nom'])


def _link_bid(lnk, zone):
    """Effektivt elbud för en sektorkopplingslänk, eller None om den inte rör zonen.

    Länkens optimalitetsvillkor i interiören är
        pris(bus0) + mc  =  eta1·pris(bus1) + eta2·pris(bus2)
    Löser man ut elbussen får man dess bud:
      elbussen är bus0 (FÖRBRUKARE — elektrolysör, VP, elpanna, EV-laddare):
        bud = eta1·pris(bus1) + eta2·pris(bus2) − mc        (betalningsvilja)
      elbussen är bus1 (PRODUCENT — KVV, H2-turbin):
        bud = (pris(bus0) + mc − eta2·pris(bus2)) / eta1     (offert)
    ⚠️ Kräver motpartsbussens pris; prices.csv skrivs för ALLA bussar, så värme-,
    H2-, bränsle- och EV-bussarna finns.
    """
    L = n.links
    b0, b1 = L.at[lnk, 'bus0'], L.at[lnk, 'bus1']
    b2 = L.at[lnk, 'bus2'] if 'bus2' in L.columns else ''
    e1 = float(L.at[lnk, 'efficiency'])
    e2 = float(L.at[lnk, 'efficiency2']) if 'efficiency2' in L.columns else 0.0
    mc = mcl_t[lnk] if lnk in mcl_t.columns else \
        pd.Series(float(L.at[lnk, 'marginal_cost']), index=snap)

    def px(b):
        return prices[b].reindex(snap) if b and b in prices.columns else 0.0

    if b0 == zone:
        return e1 * px(b1) + (e2 * px(b2) if b2 else 0.0) - mc
    if b1 == zone and e1 > 0:
        return (px(b0) + mc - (e2 * px(b2) if b2 else 0.0)) / e1
    return None


def _bus_independent(b):
    """Har bussen `b` en EGEN prissättare, oberoende av länken vi vill värdera?

    ⛔ Motivet är mätt. En sektorlänks bud är härlett ur motpartsbussens pris. Är
    länken ENDA prissättaren där blir villkoret pris(el) = eta·pris(motpart) en
    IDENTITET — avståndet blir exakt 0,0000 i varje interiör timme och länken vinner
    "närmast pris" mot allt, utan att bära någon information.

    Mätt i run450: `{zon} H2ef` (typ 3, elektrobränslen) saknar lager, och dess pris
    ÄR pris(el)/eta i 99,9-100 % av timmarna — shed-trappan är marginell i bara
    1,4-6,5 %. `{zon} H2` (typ 2) har däremot en Store med e_cyclic, vars SOC-dual är
    ett oberoende lagervärde: cirkulärt bara 0-31 % av timmarna. Samma sak för
    värmebussen (lager + must-run + KVV) och EV-bussen (bilbatteriet).

    Oberoende = något av:
      • en Store på bussen är INTERIÖR (0 < e < e_nom) → dess dual sätter priset
      • en Generator på bussen är interiör och matchar busspriset (exogen mc)
    """
    ok = pd.Series(False, index=snap)
    for st in n.stores.index:
        if n.stores.at[st, 'bus'] != b:
            continue
        e = n.stores_t.e[st].reindex(snap)
        en = float(n.stores.at[st, 'e_nom_opt'] if 'e_nom_opt' in n.stores.columns
                   and n.stores.at[st, 'e_nom_opt'] else n.stores.at[st, 'e_nom'])
        if en > 0:
            ok |= (e > 1) & (e < en * 0.999)
    for gg in n.generators.index:
        if n.generators.at[gg, 'bus'] != b:
            continue
        gmc = mc_t[gg] if gg in mc_t.columns else \
            pd.Series(float(n.generators.at[gg, 'marginal_cost']), index=snap)
        pg = n.generators_t.p[gg].reindex(snap)
        pn = _pnom(n.generators, gg)
        ok |= (pg > 1) & (pg < pn * 0.999) & ((prices[b] - gmc).abs() < TOL_PRICE)
    return ok


# ── 1. Prisöar per snapshot ────────────────────────────────────────────────────
# Interna AC-länkar. En länk KOPPLAR om den varken är mättad eller bär prisskillnad.
ac = [(l, n.links.at[l, 'bus0'], n.links.at[l, 'bus1']) for l in n.links.index
      if n.links.at[l, 'bus0'] in ZONES and n.links.at[l, 'bus1'] in ZONES]
coupled = pd.DataFrame(index=snap)
for l, b0, b1 in ac:
    p0 = n.links_t.p0[l].reindex(snap)
    cap = _pnom(n.links, l)
    sat = (p0.abs() - cap).abs() < 1e-3
    coupled[l] = (~sat) & ((prices[b1] - prices[b0]).abs() < TOL_COUPLE)

# Kopplingsmönstret kodas som ett heltal → som mest 2^len(ac) unika mönster, i
# praktiken några tiotal. Union-find körs en gång per MÖNSTER, inte per timme.
key = (coupled.values.astype(int) * (1 << np.arange(len(ac)))).sum(axis=1)
isl = pd.DataFrame(0, index=snap, columns=ZONES, dtype=int)
for k in np.unique(key):
    parent = {z: z for z in ZONES}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, (l, b0, b1) in enumerate(ac):
        if k >> i & 1:
            parent[find(b0)] = find(b1)
    isl.loc[key == k, :] = [ZONES.index(find(z)) for z in ZONES]

same_isl = {z: {w: (isl[z] == isl[w]) for w in ZONES} for z in ZONES}

# ── 2. Kandidater (zon, bud, interiör, carrier) ────────────────────────────────
exo, lnkc = {}, {}
for g in n.generators.index:
    z = n.generators.at[g, 'bus']
    if z not in ZONES:
        continue
    # ALLA generatorer på bussen, inte en vitlista: DSR (statisk mc 100/250/500) och
    # VRE (vom − curtailment-cost) föll tidigare bort, och båda KAN vara marginella —
    # VRE när den är delvis nedreglerad, vilket är enda vägen till negativa priser.
    bid = mc_t[g] if g in mc_t.columns else \
        pd.Series(float(n.generators.at[g, 'marginal_cost']), index=snap)
    p_ = n.generators_t.p[g].reindex(snap)
    pn = _pnom(n.generators, g)
    # ⛔ Marknadsventilen har p_min_pu = −1 och kan gå ÅT BÅDA HÅLL. Ett ensidigt
    # test (p > 1) dömer bort varje EXPORTtimme som "ej interiör", trots att
    # ventilen då har fullt utrymme uppåt och mycket väl kan vara marginell. Mätt
    # på run450: 94-100 % av de oförklarade timmarna i SE-N/SE-S/DK/FI hade en
    # marknadsventil i prisön. Undre gränsen är p_min_pu·p_nom, inte 0.
    lo = float(n.generators.at[g, 'p_min_pu']) * pn if 'p_min_pu' in n.generators.columns else 0.0
    exo[g] = (z, bid, (p_ > lo + 1) & (p_ < pn * 0.999), n.generators.at[g, 'carrier'])
if water_value is not None:
    for u in water_value.columns:
        if u not in n.storage_units.index:
            continue
        z = n.storage_units.at[u, 'bus']
        if z not in ZONES:
            continue
        base = smc_t[u] if u in smc_t.columns else \
            pd.Series(float(n.storage_units.at[u, 'marginal_cost']), index=snap)
        eff = n.storage_units.at[u, 'efficiency_dispatch']
        p_ = n.storage_units_t.p[u].reindex(snap)
        pn = _pnom(n.storage_units, u)
        car = n.storage_units.at[u, 'carrier']
        bid_u = base + water_value[u].reindex(snap) / eff
        interior_u = (p_.abs() > 1) & (p_ < pn * 0.999)
        if car == 'hydro' and LADDER:
            # En kandidat PER TRAPPSTEG — LP:t fyller billigaste nivån först, så vilken
            # som är marginell varierar per timme.
            for k, off in enumerate(OFFSETS):
                exo[f'{u} [nivå {k + 1}]'] = (z, bid_u + off, interior_u, car)
        else:
            exo[u] = (z, bid_u, interior_u, car)
for lnk in n.links.index:
    if n.links.at[lnk, 'carrier'] == 'AC':
        continue
    for z in ZONES:
        bid = _link_bid(lnk, z)
        if bid is None:
            continue
        p_ = n.links_t.p0[lnk].reindex(snap)
        pn = _pnom(n.links, lnk)
        interior = (p_.abs() > 1) & (p_.abs() < pn * 0.999)
        # Länken duger som sättare bara i timmar då motpartsbussen har en EGEN
        # prissättare — annars är budet en identitet (se _bus_independent).
        for _b in {n.links.at[lnk, 'bus0'], n.links.at[lnk, 'bus1'],
                   (n.links.at[lnk, 'bus2'] if 'bus2' in n.links.columns else '')}:
            if _b and _b != z and _b in prices.columns:
                interior &= _bus_independent(_b)
        lnkc[lnk] = (z, bid, interior, n.links.at[lnk, 'carrier'])
        break

# ── 3. Sättare per ö, tilldelad alla zoner i ön ────────────────────────────────
# ⛔ Länkarna körs i ett ANDRA pass och får bara fylla timmar utan exogen sättare.
# Deras bud är härlett ur en annan buss, och när länken är enda prissättaren där
# (elektrolysören på H2-bussen) är villkoret pris(el) = eta·pris(H2) en IDENTITET:
# avståndet blir exakt 0,0000 i varje interiör timme och länken vinner mot allt.
# Mätt i run450/NO-N: elektrolysören interiör 79,4 % av h med |bud−pris| max 0,0000,
# medan hydrons avstånd är 7,95 i medel.
UNSET = 'sätts utanför prisön'
shares, coupling, marg_by_zone, dist_by_zone = {}, {}, {}, {}
for z in ZONES:
    marg = pd.Series(UNSET, index=snap, dtype=object)
    best = pd.Series(np.inf, index=snap)

    def _scan(items, only_unassigned):
        for name, (zc, bid, on_margin, carrier) in items.items():
            elig = on_margin & same_isl[z][zc]   # ⭐ enheten måste ligga i SAMMA prisö
            d = (prices[z] - bid.reindex(snap)).abs()
            hit = elig & (d < best) & (d < TOL_PRICE)
            if only_unassigned:
                hit &= (marg == UNSET)
            marg[hit] = carrier
            best[hit] = d[hit]

    _scan(exo, False)
    _scan(lnkc, True)
    marg_by_zone[z] = marg
    dist_by_zone[z] = best.replace(np.inf, np.nan)
    shares[z] = marg.value_counts(normalize=True).mul(100)
    kopplad = pd.Series(False, index=snap)
    for w in ZONES:
        if w != z:
            kopplad |= same_isl[z][w]
    coupling[z] = {'i prisö med granne %': 100 * kopplad.mean(),
                   'ensam prisö %': 100 * (~kopplad).mean(),
                   'östorlek medel': float(sum(same_isl[z][w].astype(int)
                                               for w in ZONES).mean())}

tab = pd.DataFrame(shares).reindex(columns=ZONES)
_spec = [r for r in (UNSET,) if r in tab.index]
_rest = tab.drop(index=_spec).sum(axis=1).sort_values(ascending=False).index
tab = tab.reindex(list(_rest) + _spec).fillna(0.0)

print(f'PRISSÄTTANDE TEKNIK PER PRISÖ — % av timmarna  ({LABEL})')
print(f'  öar av trängselfria AC-länkar, tol_couple={TOL_COUPLE}; '
      f'sättare krävs inom {TOL_PRICE} EUR/MWh av öpriset')
print(tab.round(1).to_string())
print(f'\nSUMMA {tab.sum().round(1).to_dict()}')

print('\nPRISÖARNAS STORLEK')
print(pd.DataFrame(coupling).reindex(columns=ZONES).astype(float).round(1).to_string())

print('\n|bud − pris| för sättaren, EUR/MWh (NaN-andel = sätts utanför prisön)')
_d = pd.DataFrame({z: dist_by_zone[z] for z in ZONES})
print(pd.concat([_d.mean().rename('medel'), _d.median().rename('median'),
                 _d.max().rename('max'), _d.isna().mean().mul(100).rename('NaN %')],
                axis=1).T.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
ax = axes[0]
for z in ZONES:
    ax.hist(prices[z], bins=80, histtype='step', label=z, lw=1.1)
ax.set_xlabel('EUR/MWh'); ax.set_ylabel('antal timmar')
ax.set_title('Prisfördelning per zon'); ax.legend(fontsize=8)

ax = axes[1]
win = max(int(24 * 7 / dt_h), 1)
if water_value is not None:
    for u in [c for c in water_value.columns if c.endswith('hydro')]:
        ax.plot(snap, water_value[u].rolling(win, min_periods=1).mean(), lw=1.0, label=u)
ax.plot(snap, prices[REF_ZONE].rolling(win, min_periods=1).mean(), lw=1.4, color='black',
        label=f'{REF_ZONE} pris')
ax.set_ylabel('EUR/MWh'); ax.set_title('Vattenvärde vs pris (veckomedel)')
ax.legend(fontsize=8)
plt.show()
