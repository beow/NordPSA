#!/usr/bin/env python
"""Verifiering av hydro-driftrestriktionerna (--hydro-restrictions).

Leksaksnät (1 zon, 3 veckor, 3h) där varje villkor bevisas STYRA optimum, inte bara
finnas i modellen. Kör:  python scripts/test_hydro_operation.py

Testfall:
  A  villkoren ändrar driftmönstret och kostar mer än referensen
  B  skärpt dygnsgolv binder exakt och höjer kostnaden
  C  bindande veckotak kapar produktionen och tvingar fram spill
  D  per-zon-override slår igenom
  E  förhandskontrollen fångar omöjliga parametrar FÖRE solve
  F  bypass-spill-gångjärnet uppfylls exakt vecka för vecka
"""
import sys
from pathlib import Path

import pandas as pd
pd.set_option('future.infer_string', False)
import numpy as np, pypsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.network import hydro_operation_constraints, hydro_operation_feasibility_report

RES, DAYS, P_NOM = 3, 21, 1000.0

def build(inflow_mw, cheap_mw=600, cheap_cost=0.5):
    sn = pd.date_range('2023-01-02', periods=DAYS*24//RES, freq=f'{RES}h')
    n = pypsa.Network(); n.set_snapshots(sn); n.snapshot_weightings.loc[:, :] = float(RES)
    n.add('Bus','b'); n.add('Carrier','hydro'); n.add('Carrier','AC'); n.buses['carrier']='AC'
    load = 400 + 900*np.sin(np.arange(len(sn))*2*np.pi/(24/RES))**2
    n.add('Load','l',bus='b',p_set=pd.Series(load,index=sn))
    n.add('StorageUnit','Z hydro',bus='b',carrier='hydro',p_nom=P_NOM,max_hours=500,
          inflow=pd.Series(float(inflow_mw),index=sn),cyclic_state_of_charge=True,
          p_min_pu=0.0,spill_cost=0.1,marginal_cost=1.0)
    n.add('Generator','g',bus='b',p_nom=5000,marginal_cost=40)
    n.add('Generator','cheap',bus='b',p_nom=cheap_mw,marginal_cost=cheap_cost)
    return n

def run(n, oc):
    if oc:
        cb = hydro_operation_constraints(oc)
        n.optimize.create_model(); cb(n, n.snapshots)
        st,_ = n.optimize.solve_model(solver_name='highs')
    else:
        st,_ = n.optimize(solver_name='highs')
    assert st=='ok', f'solver {st}'
    d = n.storage_units_t.p_dispatch['Z hydro']; wts = n.snapshot_weightings.stores
    dh = wts.groupby(d.index.normalize()).sum(); wk = d.index.to_period('W-SUN').start_time
    wh = wts.groupby(wk).sum()
    return dict(obj=n.objective, hmin=d.min(),
                dmin=((d*wts).groupby(d.index.normalize()).sum()/(P_NOM*dh)).min(),
                wmax=((d*wts).groupby(wk).sum()/(P_NOM*wh)).max(),
                spill=float((n.storage_units_t.spill['Z hydro']*wts).sum()) if 'Z hydro' in n.storage_units_t.spill else 0.0,
                n=n)

OC = {'min_hourly_frac':0.10,'min_daily_frac':0.20,'max_weekly_frac':0.77}
print(f"{'scenario':38s}{'obj':>12s}{'min tim':>9s}{'min dygn':>10s}{'max vecka':>11s}{'spill MWh':>11s}")
def line(lbl,r): print(f"  {lbl:36s}{r['obj']:12,.0f}{r['hmin']:9.1f}{r['dmin']:10.4f}{r['wmax']:11.4f}{r['spill']:11,.0f}")

# A: låg tillrinning -> golven ska binda
base = run(build(450), None); line('A ref (utan restriktioner)', base)
withc = run(build(450), OC);  line('A med restriktioner', withc)
assert base['hmin'] < 0.10*P_NOM, 'referensen uppfyllde redan timgolvet — svagt test'
assert withc['hmin'] >= 0.10*P_NOM-1e-4 and withc['dmin'] >= 0.20-1e-6
assert withc['obj'] >= base['obj']-1e-6, 'restriktioner gjorde det BILLIGARE — fel'

# B: skärpta golv -> dygnsgolvet ska binda och kosta mer
oc_b = {**OC, 'min_daily_frac':0.45}
b = run(build(450), oc_b); line('B min_daily 0.45 (skärpt)', b)
assert b['dmin'] >= 0.45-1e-6, 'dygnsgolvet ej uppfyllt'
assert b['obj'] > withc['obj'], 'skärpt dygnsgolv borde kosta mer'

# C: dyr alternativkraft -> LP vill köra hydro högt; veckotaket ska binda
cref = run(build(900, cheap_mw=0), None); line('C ref (dyr alt.kraft)', cref)
c = run(build(900, cheap_mw=0), {**OC, 'max_weekly_frac':0.40}); line('C max_weekly 0.40', c)
assert cref['wmax'] > 0.40, f"referensen låg redan under taket ({cref['wmax']:.3f}) - svagt test"
assert c['wmax'] <= 0.40+1e-6, 'veckotaket ej uppfyllt'
assert c['obj'] > cref['obj'], 'bindande veckotak borde kosta mer'
assert c['spill'] > cref['spill']+1e-6, 'kapad produktion borde ge mer spill'

# D: per-zon-override ska slå igenom
d = run(build(900), {**OC, 'max_weekly_frac':0.77, 'max_weekly_frac_by_zone':{'Z':0.35}})
line('D per-zon-override Z=0.35', d)
assert d['wmax'] <= 0.35+1e-6, 'per-zon-override slog inte igenom'

# E: förhandskontrollen ska FÅNGA ett omöjligt fall före solve
warn = hydro_operation_feasibility_report(build(450), {'min_daily_frac':0.60,'max_weekly_frac':0.77})
print('\n  E förhandskontroll (min_daily 0.60 vs tillrinning 45%):')
for w in warn: print('     ', w)
assert warn and 'INFEASIBLE' in warn[0], 'förhandskontrollen missade omöjligt fall'
warn2 = hydro_operation_feasibility_report(build(900), {'min_daily_frac':0.20,'max_weekly_frac':0.50})
print('  E2 (veckotak 0.50 vs tillrinning 90%):')
for w in warn2: print('     ', w)
assert warn2 and 'spill' in warn2[0]

# F: bypass_spill-gångjärnet
KAP, THR, KOEF = 0.60, 0.10, 0.15
f = run(build(600, cheap_mw=0), {**OC,'max_weekly_frac':KAP,
        'bypass_spill':{'active':True,'threshold_below_max':THR,'coefficient':KOEF}})
line(f'F bypass_spill k={KOEF}', f)
nf = f['n']; dd = nf.storage_units_t.p_dispatch['Z hydro']; ww = nf.snapshot_weightings.stores
wk = dd.index.to_period('W-SUN').start_time
prod_w = (dd*ww).groupby(wk).sum(); spill_w = (nf.storage_units_t.spill['Z hydro']*ww).groupby(wk).sum()
hrs = ww.groupby(wk).sum(); thr_w = (KAP-THR)*P_NOM*hrs
print('     vecka   prod MWh   troskel   kravd spill   faktisk spill')
okall = True
for k in prod_w.index:
    krav = max(0.0, KOEF*(prod_w[k]-thr_w[k])); ok = spill_w[k] >= krav-1e-4; okall &= ok
    print(f'     {str(k)[:10]} {prod_w[k]:10,.0f} {thr_w[k]:9,.0f} {krav:13,.0f} {spill_w[k]:15,.0f}  {"OK" if ok else "FEL"}')
assert okall, 'bypass_spill-gangjarnet uppfylldes inte'

print('\nALLA TESTER GODKÄNDA')
