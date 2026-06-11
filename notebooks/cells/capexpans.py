"""capexpans — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Optimal installerad kapacitet per teknik och zon
ext = n.generators[n.generators.p_nom_extendable].copy()
ext['p_nom_opt_mw'] = n.generators.p_nom_opt[ext.index]
ext['added_mw']     = (ext['p_nom_opt_mw'] - ext['p_nom_min']).clip(lower=0)

# Pivottabell: zoner × carriers
tbl = ext.pivot_table(values='p_nom_opt_mw', index='bus', columns='carrier', aggfunc='sum', fill_value=0)
tbl.index.name = 'zon'
print('Optimal kapacitet (MW):')
print(tbl.round(0).to_string())
print()
added = ext[ext['added_mw'] > 1][['carrier','p_nom_min','p_nom_opt_mw','added_mw']]
if added.empty:
    print('Ingen ny kapacitet byggd utöver befintlig.')
else:
    print('Ny kapacitet (utöver befintlig):')
    print(added.round(0).to_string())
