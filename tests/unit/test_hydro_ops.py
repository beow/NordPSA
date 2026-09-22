"""Hydro-driftrestriktionerna på ett leksaksnät (1 zon, 3 veckor, 3h): varje villkor ska
bevisligen STYRA optimum, inte bara finnas i modellen."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from nordpsa.constraints import hydro_operation_constraints, hydro_operation_feasibility_report

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


OC = {'min_hourly_frac': 0.10, 'min_daily_frac': 0.20, 'max_weekly_frac': 0.77}


@pytest.fixture(scope="module")
def low_inflow():
    return run(build(450), None), run(build(450), OC)


def test_floors_bind_and_cost_more(low_inflow):
    base, withc = low_inflow
    assert base['hmin'] < 0.10 * P_NOM, 'referensen uppfyllde redan timgolvet — svagt test'
    assert withc['hmin'] >= 0.10 * P_NOM - 1e-4 and withc['dmin'] >= 0.20 - 1e-6
    assert withc['obj'] >= base['obj'] - 1e-6


def test_tighter_daily_floor_binds_and_costs_more(low_inflow):
    _, withc = low_inflow
    b = run(build(450), {**OC, 'min_daily_frac': 0.45})
    assert b['dmin'] >= 0.45 - 1e-6
    assert b['obj'] > withc['obj']


def test_weekly_cap_binds_and_forces_spill():
    cref = run(build(900, cheap_mw=0), None)
    c = run(build(900, cheap_mw=0), {**OC, 'max_weekly_frac': 0.40})
    assert cref['wmax'] > 0.40, 'referensen låg redan under taket — svagt test'
    assert c['wmax'] <= 0.40 + 1e-6
    assert c['obj'] > cref['obj']
    assert c['spill'] > cref['spill'] + 1e-6


def test_per_zone_weekly_cap_overrides_global():
    d = run(build(900), {**OC, 'max_weekly_frac': 0.77, 'max_weekly_frac_by_zone': {'Z': 0.35}})
    assert d['wmax'] <= 0.35 + 1e-6


def test_feasibility_report_catches_impossible_parameters():
    warn = hydro_operation_feasibility_report(build(450), {'min_daily_frac': 0.60, 'max_weekly_frac': 0.77})
    assert warn and 'INFEASIBLE' in warn[0]
    warn2 = hydro_operation_feasibility_report(build(900), {'min_daily_frac': 0.20, 'max_weekly_frac': 0.50})
    assert warn2 and 'spill' in warn2[0]


def test_bypass_spill_hinge_holds_week_by_week():
    KAP, THR, KOEF = 0.60, 0.10, 0.15
    f = run(build(600, cheap_mw=0), {**OC, 'max_weekly_frac': KAP,
            'bypass_spill': {'active': True, 'threshold_below_max': THR, 'coefficient': KOEF}})
    nf = f['n']
    dd = nf.storage_units_t.p_dispatch['Z hydro']; ww = nf.snapshot_weightings.stores
    wk = dd.index.to_period('W-SUN').start_time
    prod_w = (dd * ww).groupby(wk).sum()
    spill_w = (nf.storage_units_t.spill['Z hydro'] * ww).groupby(wk).sum()
    thr_w = (KAP - THR) * P_NOM * ww.groupby(wk).sum()
    for k in prod_w.index:
        assert spill_w[k] >= max(0.0, KOEF * (prod_w[k] - thr_w[k])) - 1e-4
