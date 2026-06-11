"""bootstrap — laddar en körnings resultat till modulvariabler för analys-
cellerna i notebooks/cells/. Sätt LABEL nedan. Extraherad ur explore_results.ipynb.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pypsa
import yaml
from pathlib import Path
# LABEL = "run68_baseline_1h"
#LABEL = "run67_battery_5gw4h_1h"
# LABEL = "run69_battery_10gw10h_1h"
# LABEL = "run82_wind_ses_9893_1h"
# LABEL = "run84_socpin_senose_1h"
LABEL = "run94_elasticity_1h"          # 1h-baseline: elasticitet + hydrogolv-komp
ROOT = Path('..').resolve()
sys.path.insert(0, str(ROOT))

RES  = ROOT / 'results' / LABEL
PROC = ROOT / 'data' / 'processed'

plt.rcParams['figure.figsize'] = (15, 4)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

ZONES = ['SE-N', 'SE-S', 'NO-N', 'NO-S', 'DK', 'FI']

with open(ROOT / 'config' / 'zones.yaml') as f:
    cfg = yaml.safe_load(f)

# Ladda nätverk
n = pypsa.Network()
n.import_from_netcdf(RES / 'network.nc')

# Ladda CSV-resultat
dispatch  = pd.read_csv(RES / 'dispatch_generators.csv', index_col=0, parse_dates=True)
hydro_d   = pd.read_csv(RES / 'dispatch_hydro.csv',      index_col=0, parse_dates=True)
hydro     = hydro_d   # alias

# Run-of-river (hydro_ror) ligger i dispatch_generators.csv (carrier hydro).
# Hjälpserie: total hydro per zon = reservoar (hydro_d) + RoR (must-run gen).
def zone_hydro_total(zone):
    res = hydro_d.get(f'{zone} hydro', pd.Series(0.0, index=dispatch.index)).clip(lower=0)
    ror_cols = [g for g in dispatch.columns if g in n.generators.index
                and n.generators.at[g, 'bus'] == zone and n.generators.at[g, 'carrier'] == 'hydro']
    ror = dispatch[ror_cols].clip(lower=0).sum(axis=1) if ror_cols else pd.Series(0.0, index=dispatch.index)
    return res.add(ror, fill_value=0.0)

soc       = pd.read_csv(RES / 'hydro_soc.csv',           index_col=0, parse_dates=True)
spill     = pd.read_csv(RES / 'hydro_spill.csv',         index_col=0, parse_dates=True)
flows     = pd.read_csv(RES / 'flows.csv',               index_col=0, parse_dates=True)
prices    = pd.read_csv(RES / 'prices.csv',              index_col=0, parse_dates=True)
# Vattenvärde (dual på lagringsbalansen, EUR/MWh) — finns om körningen sparat det
_wv_path = RES / 'water_value.csv'
water_value = pd.read_csv(_wv_path, index_col=0, parse_dates=True) if _wv_path.exists() else None
mkt_prices = pd.read_parquet(PROC / 'market_prices.parquet')
mkt_price  = mkt_prices['DE-LU']   # DE-LU som referens i prisplottar

# Last från nätverk — inkl. extra-load om körningen använde --extra-load
_nloads = n.loads_t.p_set.copy()
_nloads.index = pd.to_datetime(_nloads.index).tz_localize(None)
load = pd.DataFrame({
    z: _nloads[f'{z} load']
    for z in ZONES if f'{z} load' in _nloads.columns
}).reindex(dispatch.index)

# Hjälpfunktion: summera alla market-generatorer för en zon
def zone_market(zone):
    cols = [g for g in dispatch.columns
            if g in n.generators.index
            and n.generators.loc[g, 'bus'] == zone
            and n.generators.loc[g, 'carrier'] == 'market']
    return dispatch[cols].sum(axis=1) if cols else pd.Series(0.0, index=dispatch.index)

print(f'Tidssteg: {len(n.snapshots)}  ({n.snapshots[0]} – {n.snapshots[-1]})')
print(f'Zoner: {ZONES}')
print(f'Slack totalt: {dispatch.filter(like="slack").sum().sum()/1e3:.1f} GWh  (bör vara 0)')
print(f'Last SE-N medel: {load["SE-N"].mean():.0f} MW')
