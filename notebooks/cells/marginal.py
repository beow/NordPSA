"""marginal — given ETT datum+timme, vilken källa är marginell (sätter priset) och varför.
Förutsätter bootstrap.py kört (globala: n, ZONES, LABEL). Anropa marginal_source("2024-12-12 17:00").

Metod: zonpriset = dualen på nodbalansen. I ett trängselfritt delnät delar zoner ETT pris
(prisö). Inom en prisö sätts priset av den marginella enheten = en dispatchad enhet med
uppåt-headroom vars marginalkostnad ≈ öpriset (för hydro: vattenvärdet/WV, dualen på
lagringsbalansen). Hittas ingen sådan prövas SEKTORLÄNKARNA — elektrolysör, EV-laddare,
värmepump, elpanna, KVV — som är elastisk last/produktion och mycket väl kan vara
marginella; deras bud är länkens optimalitetsvillkor löst för elbussen, och de godtas
bara när motpartsbussen har ett eget prisankare (annars är villkoret en identitet).
Saknas även det är ön TRÄNGSELKOPPLAD: priset = en grann-ös pris ∓
trängselränta (= prisskillnaden över den mättade länken). Då följs kaskaden (minsta totala
ränta) till den ö som HAR en lokal marginalenhet → den är systemets enda äkta prissättare.
"""

# Cellen kan importeras fristående (från explore_results.py) ELLER klistras in
# i en notebook-cell efter att bootstrap.py redan körts där (explore.ipynb).
# Guard: skippar importen om bootstrap redan satt sina globaler i detta namnrum.
if 'LABEL' not in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bootstrap import *  # noqa: F401,F403
import collections
import pandas as pd
import heapq
import re as _re


def _ladder(label):
    """(K, BREDD) för --hydro-bid-ladder ur run_meta.txt, eller None om trappan var av.
    Samma parser som notebooks/cells/objcost.py och priceformation.py: sedan trappan
    blev DEFAULT står den normalt INTE i `argv:`, bara i `flaggor:` som
    `bidladder-K_BREDD`."""
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
# ⛔ Utan den termen jämförs hydro mot FEL nivå: default 3:36 ger −12/0/+12, så två
# tredjedelar av budet ligger 12 EUR/MWh från vad cellen räknade fram och hydron föll
# ut som sättare i de timmarna. Mätt i priceformation.py på run450: medianavstånd
# 6,2-18,4 EUR/MWh utan offsetterna, 0,00 i NO-N/NO-S med dem.
OFFSETS = [LADDER[1] * ((k + 0.5) / LADDER[0] - 0.5)
           for k in range(LADDER[0])] if LADDER else [0.0]


# ⭐ Toleranserna skärptes 2026-09-10 från 0,6 (= hydrons VOM, aldrig motiverat) till
# 0,01. I ett LP har prisöns zoner EXAKT samma pris och den marginella enhetens bud ÄR
# priset; 0,6 var slaskutrymme för rekonstruktionsfel — främst den saknade budtrappan
# ovan — inte en egenskap hos modellen. Uppmätt sättaravstånd i priceformation.py med
# rätt bud: medel 0,000, max 0,010.
# Etiketter för sektorlänkar som kan vara marginella på ELSIDAN.
LINK_LABELS = {
    'electrolyser': 'elektrolysör (flexibel last)',
    'EV charger': 'EV-laddning (flexibel last)',
    'heat hp': 'värmepump (flexibel last)',
    'heat elboiler': 'elpanna (flexibel last)',
    'heat chp': 'kraftvärme (producent)',
}


def _link_bid_at(ts, lnk, zone, mp):
    """Effektivt elbud för en sektorlänk vid ts, plus dess motpartsbussar.

    Länkens optimalitetsvillkor i interiören är
        pris(bus0) + mc = eta1·pris(bus1) + eta2·pris(bus2)
    Elbussen är bus0 för en FÖRBRUKARE (elektrolysör, VP, elpanna, EV-laddare) och
    bus1 för en PRODUCENT (KVV, H2-turbin); budet är det villkoret löst för elbussen.
    Returnerar (None, []) om länken inte rör zonen."""
    L = n.links
    b0, b1 = L.at[lnk, 'bus0'], L.at[lnk, 'bus1']
    b2 = L.at[lnk, 'bus2'] if 'bus2' in L.columns else ''
    e1 = float(L.at[lnk, 'efficiency'])
    e2 = float(L.at[lnk, 'efficiency2']) if 'efficiency2' in L.columns else 0.0
    mc = (n.links_t.marginal_cost.at[ts, lnk]
          if lnk in n.links_t.marginal_cost.columns else float(L.at[lnk, 'marginal_cost']))
    px = lambda b: (mp[b] if b and b in mp.index else 0.0)
    if b0 == zone:
        return e1 * px(b1) + (e2 * px(b2) if b2 else 0.0) - mc, [b for b in (b1, b2) if b]
    if b1 == zone and e1 > 0:
        return (px(b0) + mc - (e2 * px(b2) if b2 else 0.0)) / e1, [b for b in (b0, b2) if b]
    return None, []


def _bus_independent_at(ts, b, mp, tol):
    """Har bussen `b` en EGEN prissättare, oberoende av länken vi vill värdera?

    ⛔ Motivet är mätt (priceformation.py, run450). En sektorlänks bud är HÄRLETT ur
    motpartsbussens pris. Är länken enda prissättaren där blir villkoret
    pris(el) = eta·pris(motpart) en IDENTITET — avståndet blir exakt 0,0000 i varje
    interiör timme och länken vinner "närmast pris" mot allt utan att bära någon
    information. `{zon} H2ef` saknar lager och är cirkulär i 99,9-100 % av timmarna;
    `{zon} H2` (Store med e_cyclic), värmebussen och EV-bussen har egna ankare."""
    for st in n.stores.index[n.stores.bus == b]:
        en = n.stores.at[st, 'e_nom_opt'] if 'e_nom_opt' in n.stores.columns else 0.0
        en = float(en) if en and en > 0 else float(n.stores.at[st, 'e_nom'])
        if en > 0 and 1 < n.stores_t.e.at[ts, st] < en * 0.999:
            return True
    for gg in n.generators.index[n.generators.bus == b]:
        pn = n.generators.at[gg, 'p_nom_opt'] or n.generators.at[gg, 'p_nom']
        pg = n.generators_t.p.at[ts, gg]
        gmc = (n.generators_t.marginal_cost.at[ts, gg]
               if gg in n.generators_t.marginal_cost.columns
               else n.generators.at[gg, 'marginal_cost'])
        if pn > 0 and 1 < pg < pn * 0.999 and abs(mp[b] - gmc) < tol:
            return True
    return False


def _snap(ts_str, quiet=False):
    """Snäpp till tidssteget som innehåller den begärda timmen (3h: 17:00 → 15:00-steget)."""
    req = pd.Timestamp(ts_str)
    snaps = pd.DatetimeIndex(n.snapshots)
    if req < snaps[0] or req > snaps[-1]:
        if not quiet:
            print(f"{req} ligger utanför körningens period ({snaps[0]} – {snaps[-1]}).")
        return None
    ts = snaps[snaps.get_indexer([req], method='ffill')[0]]
    if ts != req and not quiet:
        step = snaps[1] - snaps[0]
        print(f"OBS: {req} faller i {step}-steget som börjar {ts} — använder den snapshoten.")
    return ts


def _analyse(ts, tol_price=0.01, tol_couple=0.01):
    """Prisöar, lokal sättare per ö och trängselkaskaden — allt utom utskriften.
    Bruten ur marginal_source() så att ANDRA celler (bidstack.py) kan förklara var ett
    pris kommer ifrån utan att skriva ut hela rapporten, och utan att duplicera logiken."""
    mp = n.buses_t.marginal_price.loc[ts]

    # --- generator-merit @ ts ---
    gp = n.generators_t.p.loc[ts]
    pmaxpu = pd.Series(1.0, index=n.generators.index)
    for c in n.generators_t.p_max_pu.columns:
        pmaxpu[c] = n.generators_t.p_max_pu.at[ts, c]
    avail_up = n.generators.p_nom_opt * pmaxpu                  # max möjlig produktion
    mc = n.generators.marginal_cost.copy()
    for c in n.generators_t.marginal_cost.columns:
        mc[c] = n.generators_t.marginal_cost.at[ts, c]
    head_up = avail_up - gp                                     # rum att producera 1 MW mer

    # --- hydro-lager @ ts (WV = mu_energy_balance) ---
    sp = n.storage_units_t.p.loc[ts]
    wv = n.storage_units_t.mu_energy_balance.loc[ts]
    spn = n.storage_units.p_nom

    # --- prisöar: union over trängselfria länkar ---
    parent = {z: z for z in ZONES}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    links = []
    for l in n.links.index:
        b0, b1 = n.links.at[l, 'bus0'], n.links.at[l, 'bus1']
        # bara transmissionslänkar mellan AC-zoner; hoppa över sektor-länkar
        # (elektrolysör/värmepump/turbin → H2-/värmebussar som ej är i ZONES)
        if b0 not in ZONES or b1 not in ZONES:
            continue
        p0 = n.links_t.p0.at[ts, l]; cap = n.links.at[l, 'p_nom']
        sat = abs(abs(p0) - cap) < 1e-3
        dpr = mp[b1] - mp[b0]
        if not sat and abs(dpr) < tol_couple:
            parent[find(b0)] = find(b1)
        links.append((l, b0, b1, p0, cap, sat, dpr))
    islands = collections.defaultdict(list)
    for z in ZONES:
        islands[find(z)].append(z)
    pri = lambda r: mp[islands[r][0]]

    # --- lokal marginalenhet per ö ---
    # Marginell = har plats att producera 1 MW MER (headroom upp) OCH mc ≈ öpriset.
    # OBS: kravet är INTE dispatch>0 — en enhet på 0 MW vars mc≈pris är nästa steg i
    # merit-order och sätter priset (t.ex. en importtranche som precis ska tas i bruk).
    def carlabel(c):
        return {'market': 'marknadsventil'}.get(c, c)
    setter = {}     # root -> (True, name, value, carrier) eller (False,)
    for root, zs in islands.items():
        pr = pri(root); cands = []
        for z in zs:
            for g in n.generators[n.generators.bus == z].index:
                if head_up[g] > 1.0:
                    cands.append((abs(mc[g] - pr), g, mc[g], carlabel(n.generators.at[g, 'carrier'])))
            # mu_energy_balance värderar LAGRAD energi. För att leverera 1 MWh till
            # nätet tas 1/eff MWh ur lagret, så urladdningsbudet är WV/eff. Hydro har
            # eff = 1.0 och påverkas inte; batterier (0.95) låg 5 % fel — vid ett
            # scarcity-pris på 1782 EUR/MWh blev det 89 EUR och batteriet missades
            # som prissättare trots att WV/eff träffade priset på decimalen.
            # ...och lagrets EGNA marginalkostnad måste med. Vattenkraften får
            # marginal_cost = zonens FAKTISKA historiska pris (network.py:410-411,
            # zone_prices ur market_prices.parquet, golvat på VOM 0.6), så budet är
            # historiskt pris + vattenvärde. Med enbart WV träffade hydro priset i
            # ~1 % av de snapshots där magasinet är interiört; med mc+WV i 64-76 %
            # (NO-N/NO-S) och då på decimalen.
            for s in n.storage_units[n.storage_units.bus == z].index:
                # ⛔ SOC MÅSTE VARA INTERIÖRT (2026-09-10). Headroom uppåt räcker inte:
                # ett stillastående batteri har alltid p_nom − p > 1. Vid SOC-taket
                # eller -golvet binder energikapacitetsvillkoret, och dess skuggpris
                # läcker in i mu_energy_balance — budet mc + μ/η ANPASSAS då till
                # priset i stället för att bestämma det, och bär ingen information.
                # Mätt i run462: 6,2 % av alla prisöar utsågs till "satt av ett lager
                # som står stilla vid en SOC-gräns". Exempel SE-S 2024-01-06 10:00 —
                # båda batterierna på p=0 och SOC 100 % budade 181,931680, exakt
                # zonpriset, medan den verkliga sättaren var SE-S LT imp2 (interiör,
                # 192,4 av 233,3 MW). ⚠️ Gäller BARA lager. För en GENERATOR är p=0
                # med mc≈pris fortfarande giltigt — den är nästa steg i merit order.
                soc = n.storage_units_t.state_of_charge.at[ts, s]
                smax = spn[s] * n.storage_units.at[s, 'max_hours']
                if (spn[s] - sp[s]) > 1.0 and 1.0 < soc < smax * 0.999:
                    eff = n.storage_units.at[s, 'efficiency_dispatch']
                    smc = (n.storage_units_t.marginal_cost.at[ts, s]
                           if s in n.storage_units_t.marginal_cost.columns
                           else n.storage_units.at[s, 'marginal_cost'])
                    bud = smc + wv[s] / eff
                    if n.storage_units.at[s, 'carrier'] == 'hydro':
                        # En kandidat PER TRAPPSTEG — LP:t fyller billigaste nivån
                        # först, så vilken nivå som är marginell varierar per timme.
                        lab = ('hydro (hist. pris + vattenvärde + trappa)' if LADDER
                               else 'hydro (hist. pris + vattenvärde)')
                        for k, off in enumerate(OFFSETS):
                            nm = f'{s} [nivå {k + 1}]' if LADDER else s
                            cands.append((abs(bud + off - pr), nm, bud + off, lab))
                    else:
                        cands.append((abs(bud - pr), s, bud, 'batteri (urladdningsbud)'))
        cands.sort()
        if cands and cands[0][0] < tol_price:
            setter[root] = (True, cands[0][1], cands[0][2], cands[0][3], 'exo')
            continue
        # ── ANDRA PASSET: sektorlänkar (elastisk efterfrågan/produktion) ──────────
        # ⛔ Utan detta pass fick varje ö vars enda marginella enhet är en länk
        #    etiketten "trängselkopplad", och kaskaden pekade ut en GRANNZONS enhet
        #    som sättare — topologiskt sant men sakligt fel. Mätt i run461, FI @
        #    2024-12-12 10:00: FI:s EV-laddare interiör på 1266/2400 MW med
        #    pris(FI)+mc = eta·pris(FI EV car) på fyra decimaler, medan rapporten
        #    påstod att SE-N:s batteri satte priset.
        # Länkarna körs SIST och bara om inget kraftslag matchar, eftersom deras bud
        # är härlett ur en annan buss (se _bus_independent_at).
        lcands = []
        for z in zs:
            for lnk in n.links.index:
                if n.links.at[lnk, 'carrier'] == 'AC':
                    continue
                bud, others = _link_bid_at(ts, lnk, z, mp)
                if bud is None:
                    continue
                p0 = n.links_t.p0.at[ts, lnk]
                cap = n.links.at[lnk, 'p_nom']
                if not (1.0 < abs(p0) < cap - 1.0):        # måste vara INTERIÖR
                    continue
                if not all(_bus_independent_at(ts, b, mp, tol_price) for b in others):
                    continue
                car = n.links.at[lnk, 'carrier']
                lcands.append((abs(bud - pr), lnk, bud, LINK_LABELS.get(car, car)))
        lcands.sort()
        setter[root] = ((True, lcands[0][1], lcands[0][2], lcands[0][3], 'link')
                        if (lcands and lcands[0][0] < tol_price) else (False,))

    # --- ö-graf över mättade länkar; Dijkstra till närmaste lokal-sättar-ö (minsta ränta) ---
    adj = collections.defaultdict(list)
    for l, b0, b1, p0, cap, sat, dpr in links:
        if sat and find(b0) != find(b1):
            adj[find(b0)].append((find(b1), abs(dpr)))
            adj[find(b1)].append((find(b0), abs(dpr)))
    def trace(root):
        pq = [(0.0, root, [])]; seen = set()
        while pq:
            cost, cur, path = heapq.heappop(pq)
            if cur in seen: continue
            seen.add(cur)
            if setter[cur][0]:
                return cur, path, cost
            for nb, rent in adj[cur]:
                if nb not in seen:
                    heapq.heappush(pq, (cost + rent, nb, path + [(cur, nb, rent)]))
        return None, [], 0.0

    return dict(ts=ts, mp=mp, sp=sp, wv=wv, islands=islands, pri=pri,
                setter=setter, links=links, trace=trace, find=find)


def _chain(A, path):
    """Prisledet från sättar-ön till denna ö, hopp för hopp (faktiska öpriser →
    alltid teckenkorrekt: +ränta uppströms en dyrare granne, −ränta nedströms)."""
    islands, pri = A['islands'], A['pri']
    nodes = [path[0][0]] + [b for _, b, _ in path]   # denna ö → … → sättare
    seq = nodes[::-1]                                 # sättare → … → denna ö
    return " → ".join(
        f"{islands[seq[i]][0]}={pri(seq[i]):.0f}"
        + (f" {'+' if pri(seq[i+1])>pri(seq[i]) else '−'}{abs(pri(seq[i+1])-pri(seq[i])):.0f}"
           if i < len(seq)-1 else "")
        for i in range(len(seq)))


def cascade_line(ts_str, zone, tol_price=0.01, tol_couple=0.01):
    """De två raderna som förklarar var `zone`:s pris kommer ifrån NÄR öns pris sätts
    utifrån. Returnerar None om ön har en egen lokal marginalenhet — den som anropar
    ska då inte skriva något alls."""
    ts = _snap(ts_str, quiet=True)
    if ts is None:
        return None
    A = _analyse(ts, tol_price, tol_couple)
    root = A['find'](zone)
    st = A['setter'][root]
    if st[0] and st[4] == 'exo':
        return None                       # budstapeln visar redan sättaren
    if st[0]:                             # en sektorlänk sätter priset lokalt
        return (f"LOKAL SÄTTARE UTANFÖR BUDSTAPELN: {st[1]} ({st[3]}, bud={st[2]:.2f}) "
                f"— elastisk sektorlast, ligger på efterfrågesidan i kurvan ovan",)
    dst, path, tot = A['trace'](root)
    if dst is None:
        return ("ingen sättare (degenererat LP) — priset går inte att spåra till en enhet",)
    ss, pri, islands = A['setter'][dst], A['pri'], A['islands']
    rel = "över" if pri(root) > pri(dst) else "under"
    return (f"TRÄNGSELKOPPLAD ({rel} sättar-ön): satt av {ss[1]} ({ss[3]}, mc={ss[2]:.1f}) "
            f"i {{{', '.join(islands[dst])}}}={pri(dst):.1f}",
            f"kaskad (öpris ± trängselränta): {_chain(A, path)}")


def marginal_source(ts_str, tol_price=0.01, tol_couple=0.01):
    ts = _snap(ts_str)
    if ts is None:
        return
    A = _analyse(ts, tol_price, tol_couple)
    mp, sp, wv = A['mp'], A['sp'], A['wv']
    islands, pri, setter, links, trace = (A['islands'], A['pri'], A['setter'],
                                          A['links'], A['trace'])

    # --- rapport ---
    print(f"\n{'='*74}\nMARGINAL KÄLLA @ {ts}   ({LABEL})\n{'='*74}")
    print("Zonpriser (EUR/MWh):  " + "   ".join(f"{z}={mp[z]:.1f}" for z in ZONES))
    print("\nPrisöar (zoner som delar pris via trängselfri länk) → vad som sätter priset:")
    for root, zs in sorted(islands.items(), key=lambda kv: -pri(kv[0])):
        s = setter[root]
        hyd = [z for z in zs if sp.get(f'{z} hydro', 0) > 1e-3]
        floor = min((wv[f'{z} hydro'] for z in hyd), default=None)
        fnote = f"  [hydrogolv WV≈{floor:.1f}]" if floor is not None else ""
        if s[0]:
            what = 'LOKAL marginalenhet' if s[4] == 'exo' else 'LOKAL marginalenhet (SEKTORLÄNK)'
            print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}")
            print(f"            ⟹ {what}: {s[1]} ({s[3]}, mc={s[2]:.1f})")
        else:
            dst, path, tot = trace(root)
            if dst is None:
                print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}  → ingen sättare (degenererat)")
                continue
            ss = setter[dst]
            chain = _chain(A, path)
            rel = "över" if pri(root) > pri(dst) else "under"
            print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}")
            print(f"            ⟹ TRÄNGSELKOPPLAD ({rel} sättar-ön): satt av {ss[1]} "
                  f"({ss[3]}, mc={ss[2]:.1f}) i {{{', '.join(islands[dst])}}}={pri(dst):.1f}")
            print(f"            kaskad (öpris ± trängselränta): {chain}")

    print("\nLänkar (flöde / kapacitet, priser i ändarna, trängselränta):")
    for l, b0, b1, p0, cap, sat, dpr in links:
        dirn = f"{b0}→{b1}" if p0 >= 0 else f"{b1}→{b0}"
        flag = f"MÄTTAD  ränta={abs(dpr):.1f}" if sat else ("kopplad (1 pris)" if abs(dpr) < tol_couple else f"Δp={dpr:.1f}")
        print(f"  {l:14} {dirn:13} |{abs(p0):6.0f}|/{cap:5.0f}   {mp[b0]:6.1f}|{mp[b1]:6.1f}   {flag}")
    print("\nTolk: en ö med LOKAL enhet sätter sitt eget pris (enheten där mc≈pris — kan vara "
          "en enhet på 0 MW som är nästa steg i merit-order). En trängselkopplad ö ärver en "
          "sättar-ös pris ± trängselränta: NEDströms (exportträngd mot dyrare granne) → lägre; "
          "UPPströms (importträngd, billig granne avskuren) → högre. Hydroöar har vattenvärdet "
          "(WV) som GOLV men prisas högre när de är export-/importträngda.")


# Vilken timme cellen kör vid exec. Samma globals().get-mönster som bootstrap.py:s LABEL —
# sätt MARGINAL_TS före exec för att välja timme, eller MARGINAL_TS = None för att BARA
# definiera funktionen och ropa själv (annars skrivs exempeltimmen ut i onödan först).
# Default-exemplet är en Dunkelflaute-spik: den kontinentala ventilen i DK sätter priset och
# kaskaden faller norrut genom mättade länkar till det billiga hydro-norr.
MARGINAL_TS = globals().get('MARGINAL_TS', "2024-12-12 17:00")
if MARGINAL_TS:
    marginal_source(MARGINAL_TS)
