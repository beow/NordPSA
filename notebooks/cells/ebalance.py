"""ebalance — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Energibalans per land — medel TWh/år (2023-2025). Vänster = LABEL (laddad körning).
# Höger referensblock = LABEL2-körning om satt i bootstrap, annars FAKTISK eSett-dispatch.
REF_LABEL = globals().get('LABEL2', None)   # referenskörning = delad LABEL2-global

COUNTRY_MAP = {'SE-N': 'SE', 'SE-S': 'SE', 'NO-N': 'NO', 'NO-S': 'NO', 'DK': 'DK', 'FI': 'FI'}
COUNTRIES   = ['SE', 'NO', 'DK', 'FI', 'Norden']
SOURCES     = ['hydro', 'nuclear', 'wind_onshore', 'wind_offshore', 'solar', 'thermal', 'gas']
SHOW_ROWS   = SOURCES + ['prod_twh', 'load_twh', 'h2_elec', 'heat_elec', 'brutto_export', 'netto_export']
ROW_LABELS  = {
    'hydro': 'Vattenkraft', 'nuclear': 'Kärnkraft', 'wind_onshore': 'Vind onshore',
    'wind_offshore': 'Vind offshore', 'solar': 'Sol', 'thermal': 'Termisk (inkl KVV-el)', 'gas': 'Gas',
    'prod_twh': 'PRODUKTION TOTALT', 'load_twh': 'Last (konsumtion)', 'h2_elec': 'H2-elektrolys',
    'heat_elec': 'Värme-el (VP+panna)',
    'brutto_export': 'Total export', 'netto_export': 'Kont. export',
}

def country_balance(res_label):
    """Energibalans per land (medel TWh/år) för en given körning."""
    RES_ = ROOT / 'results' / res_label
    nn = pypsa.Network(); nn.import_from_netcdf(RES_ / 'network.nc')
    disp = pd.read_csv(RES_ / 'dispatch_generators.csv', index_col=0, parse_dates=True)
    hyd  = pd.read_csv(RES_ / 'dispatch_hydro.csv',      index_col=0, parse_dates=True)
    flw  = pd.read_csv(RES_ / 'flows.csv',               index_col=0, parse_dates=True)
    nl = nn.loads_t.p_set.copy(); nl.index = pd.to_datetime(nl.index).tz_localize(None)
    ld = pd.DataFrame({z: nl[f'{z} load'] for z in ZONES if f'{z} load' in nl.columns}).reindex(disp.index)
    dt_h    = (disp.index[1] - disp.index[0]).total_seconds() / 3600
    n_years = len(disp) * dt_h / 8760.0   # → medel per år

    def zmkt(zone):
        cols = [g for g in disp.columns if g in nn.generators.index
                and nn.generators.at[g, 'bus'] == zone and nn.generators.at[g, 'carrier'] == 'market']
        return disp[cols].sum(axis=1) if cols else pd.Series(0.0, index=disp.index)

    rows = []
    for zone in ZONES:
        r = {'country': COUNTRY_MAP[zone]}
        for c in SOURCES:
            if c == 'hydro':
                rcols = [g for g in disp.columns if g in nn.generators.index
                         and nn.generators.at[g, 'bus'] == zone and nn.generators.at[g, 'carrier'] == 'hydro']
                s = hyd.get(f'{zone} hydro', pd.Series(0, index=disp.index)).clip(lower=0) \
                    + (disp[rcols].clip(lower=0).sum(axis=1) if rcols else 0)
            else:
                s = disp.get(f'{zone} {c}', pd.Series(0, index=disp.index)).clip(lower=0)
            r[c] = s.sum() * dt_h / 1e6 / n_years
        # KVV-el (bakpress-länk: bränsle×η_el → AC) är endogen termisk elproduktion som
        # ersätter den reducerade must-run-termiken → läggs på thermal-raden (matchar att
        # faktisk eSett-thermal redan inkluderar KVV-el).
        chp_cfg = ((cfg.get('heat') or {}).get('zones') or {}).get(zone, {}).get('chp') or {}
        eta_el  = float(chp_cfg.get('eta_el', 0.0))
        chp_p0  = flw.get(f'{zone} chp', pd.Series(0.0, index=disp.index)).clip(lower=0)
        r['thermal'] += chp_p0.sum() * dt_h / 1e6 / n_years * eta_el
        # Värme-el: AC-bussen levererar el till värmebussen via VP + el-panna (länk-p0).
        elb = flw.get(f'{zone} heat elboiler', pd.Series(0.0, index=disp.index)).clip(lower=0)
        hp  = flw.get(f'{zone} heat hp',       pd.Series(0.0, index=disp.index)).clip(lower=0)
        r['heat_elec'] = (elb + hp).sum() * dt_h / 1e6 / n_years
        h2 = flw.get(f'{zone} electrolyser', pd.Series(0.0, index=disp.index)).clip(lower=0)
        r['h2_elec'] = h2.sum() * dt_h / 1e6 / n_years
        mk = zmkt(zone)
        r['import']   = mk.clip(lower=0).sum() * dt_h / 1e6 / n_years
        r['export']   = mk.clip(upper=0).sum() * dt_h / 1e6 / n_years
        r['load_twh'] = (ld[zone].sum() * dt_h / 1e6 / n_years) if zone in ld.columns else 0.0
        rows.append(r)
    d = pd.DataFrame(rows)
    d['prod_twh']      = d[SOURCES].sum(axis=1)
    d['netto_export']  = -d['export'] - d['import']
    d['brutto_export'] = d['prod_twh'] - d['load_twh'] - d['h2_elec'] - d['heat_elec']
    cyr = d.groupby('country')[SHOW_ROWS].sum()
    cyr.loc['Norden'] = cyr.sum()
    return cyr

# eSett-carrier → modellens SOURCES (gas saknas i eSett → 0, ingår i thermal i verkligheten)
ESETT_MAP = {'hydro': 'hydro', 'nuclear': 'nuclear', 'wind': 'wind_onshore',
             'windOffshore': 'wind_offshore', 'solar': 'solar', 'thermal': 'thermal'}

# DK: eSett samlar all produktion i 'other' → använd Energy Charts (production_DK_ec)
# som build_inputs. Thermal = coal+gas+biomass+waste+fossil_oil; last = EC 'load'.
DK_EC_THERMAL = ['coal', 'gas', 'biomass', 'waste', 'fossil_oil']

def actual_balance(years=(2023, 2024, 2025)):
    """Faktisk energibalans per land (eSett produktion + konsumtion; DK via Energy Charts),
    medel TWh/år. netto_export = brutto_export för faktiskt (ingen kont/intern-uppdelning;
    på Norden-nivå sammanfaller de ändå när interna flöden tar ut varandra)."""
    rows = []
    for zone in ZONES:
        r = {'country': COUNTRY_MAP[zone]}
        if zone == 'DK':
            ec = pd.concat([pd.read_parquet(ROOT / 'data/raw' / f'production_DK_ec_{y}.parquet')
                            for y in years])
            n_years = len(ec) / 8760.0
            tw = lambda s: s.clip(lower=0).sum() / 1e6 / n_years
            r.update({'hydro': 0.0, 'nuclear': 0.0, 'gas': 0.0,
                      'wind_onshore':  tw(ec['wind']),
                      'wind_offshore': tw(ec.get('windOffshore', pd.Series(0.0, index=ec.index))),
                      'solar':         tw(ec['solar']),
                      'thermal':       tw(sum(ec[c].fillna(0) for c in DK_EC_THERMAL if c in ec.columns)),
                      'h2_elec':       0.0, 'heat_elec': 0.0,
                      'load_twh':      ec['load'].abs().sum() / 1e6 / n_years})
            rows.append(r)
            continue
        prod = pd.concat([pd.read_parquet(ROOT / 'data/raw' / f'production_{zone}_{y}.parquet')
                          for y in years])
        cons = pd.concat([pd.read_parquet(ROOT / 'data/raw' / f'consumption_{zone}_{y}.parquet')
                          for y in years])
        n_years = len(prod) / 8760.0
        for c in SOURCES:
            esc = next((k for k, v in ESETT_MAP.items() if v == c), None)
            r[c] = (prod[esc].clip(lower=0).sum() / 1e6 / n_years) if esc and esc in prod.columns else 0.0
        r['h2_elec']  = 0.0
        r['heat_elec'] = 0.0
        r['load_twh'] = cons['total'].abs().sum() / 1e6 / n_years
        rows.append(r)
    d = pd.DataFrame(rows)
    d['prod_twh']      = d[SOURCES].sum(axis=1)
    d['brutto_export'] = d['prod_twh'] - d['load_twh'] - d['h2_elec'] - d['heat_elec']
    d['netto_export']  = d['brutto_export']
    cyr = d.groupby('country')[SHOW_ROWS].sum()
    cyr.loc['Norden'] = cyr.sum()
    return cyr

# Referensblock: LABEL2-körning om satt, annars FAKTISK eSett-dispatch 2023-2025.
if REF_LABEL:
    ref_name, ref_bal = REF_LABEL, country_balance(REF_LABEL)
else:
    ref_name, ref_bal = 'faktiskt (eSett)', actual_balance()

labels = [LABEL, ref_name]
bal = {LABEL: country_balance(LABEL), ref_name: ref_bal}

col_tuples = [(lab, c) for lab in labels for c in COUNTRIES]
cols = pd.MultiIndex.from_tuples(col_tuples, names=['Körning', 'Land'])
data = {}
for lab in labels:
    for c in COUNTRIES:
        data[(lab, c)] = {row: bal[lab].loc[c, row] if c in bal[lab].index else 0.0 for row in SHOW_ROWS}

tbl = pd.DataFrame(data, columns=cols)
tbl.index = [ROW_LABELS[r] for r in SHOW_ROWS]
tbl.index.name = 'TWh/år (medel 23-25)'

# Balanskontroll: summera VISADE rader med rätt tecken → ska bli ≈ 0 per kolumn.
# Identitet: produktion − last − H2 − värme-el − Total export = 0. Summeras oberoende
# av brutto-formeln, så raden fångar om en konsumtions-/produktionsrad glöms bort framöver.
BALANCE_SIGNS = {'prod_twh': +1, 'load_twh': -1, 'h2_elec': -1, 'heat_elec': -1, 'brutto_export': -1}
bal_vals = []
for col in cols:
    v = sum(sgn * data[col][row] for row, sgn in BALANCE_SIGNS.items())
    bal_vals.append(0.0 if abs(v) < 1e-9 else v)
balance = pd.DataFrame([bal_vals], index=['BALANS (bör ≈ 0)'], columns=cols)
tbl_full = pd.concat([tbl, balance])
tbl_full.index.name = tbl.index.name

with pd.option_context('display.float_format', '{:.1f}'.format,
                       'display.max_columns', None, 'display.width', 250):
    out_lines = tbl_full.to_string().split('\n')
w = max(len(l) for l in out_lines)
out_lines.insert(len(out_lines) - 1, '-' * w)   # streckad linje före BALANS-raden
print('\n'.join(out_lines))
