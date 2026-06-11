"""ebalance — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Energibalans per land — medel TWh/år (2023-2025), aktuell körning vs referenskörning
# Vänster = LABEL (laddad körning), höger = REF_LABEL (referens). En kolumn/land = 3-årsmedel.
REF_LABEL = "run68_baseline_1h"   # referenskörning (höger blocket); byt vid behov

COUNTRY_MAP = {'SE-N': 'SE', 'SE-S': 'SE', 'NO-N': 'NO', 'NO-S': 'NO', 'DK': 'DK', 'FI': 'FI'}
COUNTRIES   = ['SE', 'NO', 'DK', 'FI', 'Norden']
SOURCES     = ['hydro', 'nuclear', 'wind_onshore', 'wind_offshore', 'solar', 'thermal', 'gas']
SHOW_ROWS   = SOURCES + ['prod_twh', 'load_twh', 'h2_elec', 'brutto_export', 'netto_export']
ROW_LABELS  = {
    'hydro': 'Vattenkraft', 'nuclear': 'Kärnkraft', 'wind_onshore': 'Vind onshore',
    'wind_offshore': 'Vind offshore', 'solar': 'Sol', 'thermal': 'Termisk', 'gas': 'Gas',
    'prod_twh': 'PRODUKTION TOTALT', 'load_twh': 'Last (konsumtion)', 'h2_elec': 'H2-elektrolys',
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
    d['brutto_export'] = d['prod_twh'] - d['load_twh'] - d['h2_elec']
    cyr = d.groupby('country')[SHOW_ROWS].sum()
    cyr.loc['Norden'] = cyr.sum()
    return cyr

bal = {LABEL: country_balance(LABEL), REF_LABEL: country_balance(REF_LABEL)}

col_tuples = [(LABEL, c) for c in COUNTRIES] + [(REF_LABEL, c) for c in COUNTRIES]
cols = pd.MultiIndex.from_tuples(col_tuples, names=['Körning', 'Land'])
data = {}
for lab in (LABEL, REF_LABEL):
    for c in COUNTRIES:
        data[(lab, c)] = {row: bal[lab].loc[c, row] if c in bal[lab].index else 0.0 for row in SHOW_ROWS}

tbl = pd.DataFrame(data, columns=cols)
tbl.index = [ROW_LABELS[r] for r in SHOW_ROWS]
tbl.index.name = 'TWh/år (medel 23-25)'

with pd.option_context('display.float_format', '{:.1f}'.format,
                       'display.max_columns', None, 'display.width', 250):
    print(tbl.to_string())
