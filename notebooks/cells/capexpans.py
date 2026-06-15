"""capexpans — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Optimal installerad kapacitet per teknik och zon
ext = n.generators[n.generators.p_nom_extendable].copy()
ext['p_nom_opt_mw'] = n.generators.p_nom_opt[ext.index]
ext['added_mw']     = (ext['p_nom_opt_mw'] - ext['p_nom_min']).clip(lower=0)

# Pivottabell: zoner × carriers (generatorer; gas = CCGT-peaker)
tbl = ext.pivot_table(values='p_nom_opt_mw', index='bus', columns='carrier', aggfunc='sum', fill_value=0)
tbl.index.name = 'zon'

# Konverterings-/flexeffekt utanför Generator-tabellen: elektrolysör (Link) + CHP-el (Link)
ely = n.links[n.links.carrier == 'electrolyser'] if 'carrier' in n.links.columns else n.links.iloc[:0]
ely_mw = ely.groupby('bus0')['p_nom_opt'].sum() if not ely.empty else pd.Series(dtype=float)
# CHP-eleffekt: bakpress-KVV-länk (carrier 'heat chp'), MW_el = p_nom_opt × η_el, på el-bussen (bus1)
chp = n.links[n.links.carrier == 'heat chp'] if 'carrier' in n.links.columns else n.links.iloc[:0]
chp_el = (chp['p_nom_opt'] * chp['efficiency']).groupby(chp['bus1']).sum() if not chp.empty else pd.Series(dtype=float)

tbl['electrolyser'] = ely_mw.reindex(tbl.index).fillna(0)   # elektrolysör MW_el
tbl['chp_el']       = chp_el.reindex(tbl.index).fillna(0)   # CHP max el-effekt MW_el (η_el × p_nom)

print('Optimal kapacitet (MW):')
print(tbl.round(0).to_string())
print()
added = ext[ext['added_mw'] > 1][['carrier','p_nom_min','p_nom_opt_mw','added_mw']]
if added.empty:
    print('Ingen ny kapacitet byggd utöver befintlig.')
else:
    print('Ny kapacitet (utöver befintlig):')
    print(added.round(0).to_string())

# Lagerstorlekar (energi, GWh) per zon: e_nom (utgång) → e_nom_opt (optimerat)
# Gridbatteri (StorageUnit): energi = p_nom × max_hours (flyttat hit från effekt-tabellen)
bat = n.storage_units[n.storage_units.carrier == 'battery']
def _su_gwh(col):
    if bat.empty:
        return pd.Series(dtype=float)
    return ((bat[col] * bat['max_hours']).groupby(bat['bus']).sum() / 1e3)   # MWh → GWh
def _store_gwh(carrier, suffix, col):
    s = n.stores[n.stores.carrier == carrier]
    if s.empty:
        return pd.Series(dtype=float)
    zon = s['bus'].str.replace(f' {suffix}', '', regex=False)
    return (s[col].groupby(zon).sum() / 1e3)   # MWh → GWh
stor = pd.DataFrame({
    'batt_e_nom':     _su_gwh('p_nom'),
    'batt_e_nom_opt': _su_gwh('p_nom_opt'),
    'H2_e_nom':       _store_gwh('H2 store',   'H2',   'e_nom'),
    'H2_e_nom_opt':   _store_gwh('H2 store',   'H2',   'e_nom_opt'),
    'heat_e_nom':     _store_gwh('heat store', 'heat', 'e_nom'),
    'heat_e_nom_opt': _store_gwh('heat store', 'heat', 'e_nom_opt'),
}).fillna(0)
if not stor.empty:
    stor.index.name = 'zon'
    print()
    print('Lagerstorlekar (GWh, e_nom → e_nom_opt):')
    print(stor.round(1).to_string())
