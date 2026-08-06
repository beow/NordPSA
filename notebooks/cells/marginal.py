"""marginal — given ETT datum+timme, vilken källa är marginell (sätter priset) och varför.
Förutsätter bootstrap.py kört (globala: n, ZONES, LABEL). Anropa marginal_source("2024-12-12 17:00").

Metod: zonpriset = dualen på nodbalansen. I ett trängselfritt delnät delar zoner ETT pris
(prisö). Inom en prisö sätts priset av den marginella enheten = en dispatchad enhet med
uppåt-headroom vars marginalkostnad ≈ öpriset (för hydro: vattenvärdet/WV, dualen på
lagringsbalansen). Saknas sådan enhet är ön TRÄNGSELKOPPLAD: priset = en grann-ös pris ∓
trängselränta (= prisskillnaden över den mättade länken). Då följs kaskaden (minsta totala
ränta) till den ö som HAR en lokal marginalenhet → den är systemets enda äkta prissättare.
"""
import collections
import pandas as pd
import heapq


def marginal_source(ts_str, tol_price=0.6, tol_couple=0.6):
    req = pd.Timestamp(ts_str)
    snaps = pd.DatetimeIndex(n.snapshots)
    if req < snaps[0] or req > snaps[-1]:
        print(f"{req} ligger utanför körningens period ({snaps[0]} – {snaps[-1]}).")
        return
    # Snäpp till tidssteget som innehåller den begärda timmen (3h: 17:00 → 15:00-steget).
    ts = snaps[snaps.get_indexer([req], method='ffill')[0]]
    if ts != req:
        step = snaps[1] - snaps[0]
        print(f"OBS: {req} faller i {step}-steget som börjar {ts} — använder den snapshoten.")
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
                if (spn[s] - sp[s]) > 1.0:
                    eff = n.storage_units.at[s, 'efficiency_dispatch']
                    smc = (n.storage_units_t.marginal_cost.at[ts, s]
                           if s in n.storage_units_t.marginal_cost.columns
                           else n.storage_units.at[s, 'marginal_cost'])
                    bud = smc + wv[s] / eff
                    lab = ('hydro (hist. pris + vattenvärde)'
                           if n.storage_units.at[s, 'carrier'] == 'hydro'
                           else 'batteri (urladdningsbud)')
                    cands.append((abs(bud - pr), s, bud, lab))
        cands.sort()
        setter[root] = (True, cands[0][1], cands[0][2], cands[0][3]) if (cands and cands[0][0] < tol_price) else (False,)

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
            print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}")
            print(f"            ⟹ LOKAL marginalenhet: {s[1]} ({s[3]}, mc={s[2]:.1f})")
        else:
            dst, path, tot = trace(root)
            if dst is None:
                print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}  → ingen sättare (degenererat)")
                continue
            ss = setter[dst]
            # Prisledet från sättar-ön till denna ö, hopp för hopp (faktiska öpriser →
            # alltid teckenkorrekt: +ränta uppströms en dyrare granne, −ränta nedströms).
            nodes = [path[0][0]] + [b for _, b, _ in path]   # denna ö → … → sättare
            seq = nodes[::-1]                                 # sättare → … → denna ö
            chain = " → ".join(
                f"{islands[seq[i]][0]}={pri(seq[i]):.0f}"
                + (f" {'+' if pri(seq[i+1])>pri(seq[i]) else '−'}{abs(pri(seq[i+1])-pri(seq[i])):.0f}" if i < len(seq)-1 else "")
                for i in range(len(seq)))
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


# Exempel (Dunkelflaute-spik): kontinental DE-export-ventil i DK sätter 666; kaskaden faller
# norrut genom mättade länkar till det billiga hydro-norr (~61).
marginal_source("2024-12-12 17:00")
