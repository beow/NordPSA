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
# LABEL = "run94_elasticity_1h"          # 1h-baseline: elasticitet + hydrogolv-komp
# globals().get(...): notebooks/explore_results.py kan sätta LABEL/LABEL2 FÖRE
# `import bootstrap` för att byta körning utan att röra den här filen.
LABEL  = globals().get('LABEL',  "run400_expansion")
LABEL2 = globals().get('LABEL2', "run360_baseline_2h")   # None = jämför bara mot faktiskt
# Om explore_results.py redan räknat ut ROOT (walk-up från cwd) och exec:ar den här
# filen i sitt eget namnrum (run_cell()), återanvänd DEN — annars pekar __file__ fel
# (mot bootstrap.py:s egen plats i cells/, inte mot den exec:ande filens plats).
if 'ROOT' not in globals():
    ROOT = Path(__file__).resolve().parents[2] if '__file__' in globals() else Path('..').resolve()
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

# Snapshot-viktning i timmar (2h-körning → 2.0). Används för MW→TWh-konvertering.
dt_h = float(n.snapshot_weightings.objective.iloc[0])

# Faktiska spotpriser för ALLA sex zoner (mkt_prices har fler kolumner, t.ex.
# DE-LU/NL/PL/GB) — facit för dispatchkörningar, referensnivå för 2040-scenarier.
_act = mkt_prices[ZONES].copy()
_act.index = pd.to_datetime(_act.index).tz_localize(None)
act_price = _act.reindex(dispatch.index, method='ffill')


def twh(x):
    """MW-serie/DataFrame → TWh över körperioden (summerar kolumner om DataFrame)."""
    s = x.sum(axis=1) if isinstance(x, pd.DataFrame) else x
    return float((s * dt_h).sum()) / 1e6


def in_zone(carrier, zones=None):
    """Generatornamn (ur dispatch.columns) med given carrier i angivna zoner
    (default alla sex elzoner)."""
    zs = ZONES if zones is None else zones
    return [g for g in dispatch.columns if g in n.generators.index
            and n.generators.at[g, 'bus'] in zs
            and n.generators.at[g, 'carrier'] == carrier]


def by_carrier(zones=None, df=None):
    """Summerar generatordispatch per carrier-ATTRIBUT (inte per namn)."""
    d = dispatch if df is None else df
    cols = [g for g in d.columns if g in n.generators.index
            and n.generators.at[g, 'bus'] in (ZONES if zones is None else zones)]
    return d[cols].T.groupby(n.generators['carrier'][cols]).sum().T


print(f'Körning : {LABEL}' + (f'  (LABEL2 = {LABEL2})' if LABEL2 else ''))
print(f'Tidssteg: {len(n.snapshots)}  ({n.snapshots[0]} – {n.snapshots[-1]}) à {dt_h:.0f}h')
print(f'Zoner   : {ZONES}')
print(f'Modell  : {len(n.generators)} generatorer, {len(n.storage_units)} lager, '
      f'{len(n.stores)} stores, {len(n.links)} länkar')
print(f"\n{'zon':6s} {'pris':>7} {'last TWh':>9} {'nettoimport TWh':>16}")
for _z in ZONES:
    print(f"{_z:6s} {prices[_z].mean():>7.1f} "
          f"{twh(load[_z]):>9.1f} {twh(zone_market(_z)):>16.2f}")
_slack = twh(dispatch[in_zone('slack')].clip(lower=0)) if in_zone('slack') else 0.0
print(f"\nLastbortkoppling (el-slack): {_slack:.4f} TWh   (bör vara ≈ 0)")
