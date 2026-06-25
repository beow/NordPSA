"""ebalance — energibalans per land (medel TWh/år) för LABEL och (om satt) LABEL2.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, pd, pypsa, ...).

ENDAST modellkörningar — ingen faktisk-jämförelse (borttagen; den var bara meningsfull
för dispatch-körningar som överlappar verkligheten, och krånglade till cellen).

BALANSEN ÄR EN ÄKTA KONTROLL. Varje term mäts OBEROENDE direkt ur körningens CSV:er /
nätverk:
  + produktion (alla källor INKL slack/lastskärning)
  + batteri (netto urladdning)
  − last  − H2-elektrolys  − värme-el
  − kontinent-export (market-generatorer)
  − intern export (NTC-länkflöden ur flows.csv; summerar till 0 på Norden-nivå)
Identiteten ska bli ≈ 0. Eftersom exporten räknas från market-gen + länkflöden (INTE som
residualen prod−last), avslöjar BALANS-raden ≠ 0 en glömd/felräknad post — till skillnad
från tidigare residual-brutto som gav 0 per definition.
"""

REF_LABEL = globals().get('LABEL2', None)   # referenskörning (delad global), annars ingen

COUNTRY_MAP = {'SE-N': 'SE', 'SE-S': 'SE', 'NO-N': 'NO', 'NO-S': 'NO', 'DK': 'DK', 'FI': 'FI'}
COUNTRIES   = ['SE', 'NO', 'DK', 'FI', 'Norden']
# SOURCES = produktionsposter (mäts var för sig; 'hydro'+'ror' slås ihop till 'vatten'
# i visningen, 'slack' ingår i prod_twh men visas ej som egen rad).
SOURCES     = ['hydro', 'ror', 'nuclear', 'wind_onshore', 'wind_offshore', 'solar', 'thermal', 'gas', 'slack']
# Visningsrader: vatten (= hydro+ror), produktion, demand, KONSUMTION TOTAL (= demand-summa), balans.
SHOW_ROWS   = ['vatten', 'nuclear', 'wind_onshore', 'wind_offshore', 'solar', 'thermal', 'gas',
               'prod_twh', 'load_twh', 'h2_elec', 'heat_elec', 'ev_elec',
               'batt_net', 'spill', 'kont_export', 'intern_export', 'kons_total']
ROW_LABELS  = {
    'vatten': 'Vattenkraft',
    'nuclear': 'Kärnkraft', 'wind_onshore': 'Vind onshore',
    'wind_offshore': 'Vind offshore', 'solar': 'Sol',
    'thermal': 'Termisk',
    'gas': 'Gas',
    'prod_twh': 'PRODUKTION TOTAL', 'load_twh': 'Last', 'h2_elec': 'H2-elektrolys',
    'heat_elec': 'Värme (VP+panna)', 'ev_elec': 'EV-laddning', 'batt_net': 'Batteri (netto ut)',
    'spill': 'Spill',
    'kont_export': 'Kontinental export', 'intern_export': 'Norden-intern export',
    'kons_total': 'KONSUMTION TOTAL',
}
# Sänk-/förbruknings-rader visas NEGATIVA (utflöde) i tabellen. Rent KOSMETISKT —
# balanskontrollen nedan använder de RÅA (positiva) värdena via BALANCE_SIGNS.
DISPLAY_NEG = {'load_twh', 'h2_elec', 'heat_elec', 'ev_elec',
               'kont_export', 'intern_export', 'spill', 'kons_total'}


def country_balance(res_label):
    """Energibalans per land (medel TWh/år) för en given körning — alla poster oberoende."""
    RES_ = ROOT / 'results' / res_label
    nn = pypsa.Network(); nn.import_from_netcdf(RES_ / 'network.nc')
    disp = pd.read_csv(RES_ / 'dispatch_generators.csv', index_col=0, parse_dates=True)
    hyd  = pd.read_csv(RES_ / 'dispatch_hydro.csv',      index_col=0, parse_dates=True)
    flw  = pd.read_csv(RES_ / 'flows.csv',               index_col=0, parse_dates=True)
    nl = nn.loads_t.p_set.copy(); nl.index = pd.to_datetime(nl.index).tz_localize(None)
    ld = pd.DataFrame({z: nl[f'{z} load'] for z in ZONES if f'{z} load' in nl.columns}).reindex(disp.index)
    dt_h    = (disp.index[1] - disp.index[0]).total_seconds() / 3600
    n_years = len(disp) * dt_h / 8760.0          # → medel per år
    twh     = lambda s: float(s.sum()) * dt_h / 1e6 / n_years
    zero    = pd.Series(0.0, index=disp.index)

    # NTC-länkar = länkar mellan två AC-zoner (elektrolysör-länkar går till H2-buss → exkl).
    ntc = [(l, nn.links.at[l, 'bus0'], nn.links.at[l, 'bus1']) for l in nn.links.index
           if nn.links.at[l, 'bus0'] in ZONES and nn.links.at[l, 'bus1'] in ZONES]
    # Batteri-storage (ej hydro) per buss
    batt = nn.storage_units[nn.storage_units.carrier != 'hydro'] if len(nn.storage_units) else nn.storage_units
    # VRE-tillgänglighet (p_max_pu × p_nom_opt) för curtailment/spill-beräkning.
    VRE_CARRIERS = {'wind_onshore', 'wind_offshore', 'solar'}
    pmax = nn.generators_t.p_max_pu.copy()
    pmax.index = pd.to_datetime(pmax.index).tz_localize(None)
    pmax = pmax.reindex(disp.index)

    def zmkt(zone):
        cols = [g for g in disp.columns if g in nn.generators.index
                and nn.generators.at[g, 'bus'] == zone and nn.generators.at[g, 'carrier'] == 'market']
        return disp[cols].sum(axis=1) if cols else zero

    rows = []
    for zone in ZONES:
        r = {'country': COUNTRY_MAP[zone]}
        curt = zero.copy()                       # ackumulerad VRE-curtailment (→ spill-raden)
        for c in SOURCES:
            if c == 'hydro':
                # Magasin (StorageUnit); RoR-generatorerna redovisas separat som 'ror'.
                s = hyd.get(f'{zone} hydro', zero).clip(lower=0)
            elif c == 'ror':
                rcols = [g for g in disp.columns if g in nn.generators.index
                         and nn.generators.at[g, 'bus'] == zone and nn.generators.at[g, 'carrier'] == 'hydro']
                s = disp[rcols].clip(lower=0).sum(axis=1) if rcols else zero
            else:
                # Summera ALLA generatorer med carrier c i zonen (t.ex. 'nuclear' +
                # 'nuclear exp', wind_onshore + wind_new) — ej bara exakt '{zon} {c}'.
                ccols = [g for g in disp.columns if g in nn.generators.index
                         and nn.generators.at[g, 'bus'] == zone and nn.generators.at[g, 'carrier'] == c]
                disp_c = disp[ccols].clip(lower=0).sum(axis=1) if ccols else zero   # slack ingår här
                if c in VRE_CARRIERS and ccols:
                    # VRE redovisas som TILLGÄNGLIGT (p_max_pu × p_nom_opt); skillnaden
                    # mot dispatchat = curtailment → ackumuleras till spill-raden.
                    avail = sum((pmax[g] * nn.generators.at[g, 'p_nom_opt']
                                 for g in ccols if g in pmax.columns), zero)
                    curt = curt + (avail - disp_c).clip(lower=0)
                    s = avail
                else:
                    s = disp_c
            r[c] = twh(s)
        r['spill']     = twh(curt)
        # KVV-el (bakpress-länk: bränsle×η_el → AC) läggs på thermal-raden (matchar eSett).
        eta_el = float(((((cfg.get('heat') or {}).get('zones') or {}).get(zone, {}).get('chp')) or {}).get('eta_el', 0.0))
        r['thermal'] += twh(flw.get(f'{zone} chp', zero).clip(lower=0)) * eta_el
        # Värme-el (AC → värmebuss via VP + el-panna, länk-p0) och H2-elektrolys (länk-p0)
        r['heat_elec'] = twh(flw.get(f'{zone} heat elboiler', zero).clip(lower=0)
                             + flw.get(f'{zone} heat hp', zero).clip(lower=0))
        r['h2_elec']   = twh(flw.get(f'{zone} electrolyser', zero).clip(lower=0))
        # EV-laddning = flexibel (charger-länk p0) + oflexibel (fast AC-last, svk-läget).
        # Oflex-lasten ligger på AC-bussen men EJ i ld (bara '{z} load') → måste in här.
        ev_inflex = sum((nl[f'{zone} EV {c} inflex'].reindex(disp.index).fillna(0.0)
                         for c in ('car', 'heavy') if f'{zone} EV {c} inflex' in nl.columns), zero)
        r['ev_elec']   = twh(sum((flw.get(f'{zone} EV {c} charger', zero).clip(lower=0)
                                  for c in ('car', 'heavy')), zero) + ev_inflex)
        # Kontinent-export (market-gen: p>0 import, p<0 export → netto export = −Σp)
        r['kont_export'] = -twh(zmkt(zone))
        # Intern export = netto NTC-flöde UT ur zonen (p0 = bus0→bus1)
        ie = zero.copy()
        for l, b0, b1 in ntc:
            f = flw.get(l, zero)
            if   b0 == zone: ie = ie + f
            elif b1 == zone: ie = ie - f
        r['intern_export'] = twh(ie)
        # Batteri netto urladdning (positiv = ut på nätet); index per position (samma snapshots)
        bz = [s for s in batt.index if nn.storage_units.at[s, 'bus'] == zone]
        r['batt_net'] = (float(nn.storage_units_t.p[bz].to_numpy().sum()) * dt_h / 1e6 / n_years) if bz else 0.0
        r['load_twh'] = twh(ld[zone]) if zone in ld.columns else 0.0
        rows.append(r)

    d = pd.DataFrame(rows)
    d['prod_twh']   = d[SOURCES].sum(axis=1)        # produktion inkl slack
    d['vatten']     = d['hydro'] + d['ror']         # magasin + älv (RoR) i en post
    # KONSUMTION TOTAL = inhemsk last + nettoexport (export = utflöde = last) − batteri (netto ut).
    # = PRODUKTION TOTALT vid balans → de två totalraderna möts.
    d['kons_total'] = (d['load_twh'] + d['h2_elec'] + d['heat_elec'] + d['ev_elec']
                       + d['kont_export'] + d['intern_export'] + d['spill'] - d['batt_net'])
    cyr = d.groupby('country')[SHOW_ROWS].sum()
    cyr.loc['Norden'] = cyr.sum()
    return cyr


# ── Tabell: LABEL och (om satt) LABEL2 ───────────────────────────────────────
labels = [LABEL] + ([REF_LABEL] if REF_LABEL else [])
bal    = {lab: country_balance(lab) for lab in labels}

cols = pd.MultiIndex.from_tuples([(lab, c) for lab in labels for c in COUNTRIES],
                                 names=['Körning', 'Land'])
data = {(lab, c): {row: (bal[lab].loc[c, row] if c in bal[lab].index else 0.0) for row in SHOW_ROWS}
        for lab in labels for c in COUNTRIES}

tbl = pd.DataFrame(data, columns=cols)
# Visa förbruknings-/sänk-rader negativa (utflöde); råvärdena i `data` (positiva)
# används oförändrade i balanskontrollen nedan.
tbl = tbl.mul(pd.Series({r: (-1 if r in DISPLAY_NEG else 1) for r in SHOW_ROWS}), axis=0)
tbl.index = [ROW_LABELS[r] for r in SHOW_ROWS]
tbl.index.name = 'TWh/år (medel)'

# Balanskontroll: ÄKTA — varje post oberoende mätt. Identitet:
#   produktion + batteri − last − H2 − värme-el − EV-laddning − kont.export − intern export = 0
BALANCE_SIGNS = {'prod_twh': +1, 'batt_net': +1, 'load_twh': -1, 'h2_elec': -1,
                 'heat_elec': -1, 'ev_elec': -1, 'spill': -1, 'kont_export': -1, 'intern_export': -1}
bal_vals = [sum(sgn * data[col][row] for row, sgn in BALANCE_SIGNS.items()) for col in cols]
bal_vals = [0.0 if abs(v) < 1e-6 else v for v in bal_vals]
balance  = pd.DataFrame([bal_vals], index=['BALANS (bör ≈ 0)'], columns=cols)
tbl_full = pd.concat([tbl, balance])
tbl_full.index.name = tbl.index.name

with pd.option_context('display.float_format', '{:.1f}'.format,
                       'display.max_columns', None, 'display.width', 250):
    out_lines = tbl_full.to_string().split('\n')
w = max(len(l) for l in out_lines)
out_lines.insert(len(out_lines) - 1, '-' * w)   # streckad linje före BALANS-raden
print('\n'.join(out_lines))
