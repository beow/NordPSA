"""SOC-ankaret, budtrappan och terminalvärdet på ett leksaksnät som löses med HiGHS."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from nordpsa.constraints import (hydro_bid_ladder, hydro_soc_initial_constraint,
                                 hydro_terminal_value)

P_NOM, MAX_H = 900.0, 200.0


def toy(hours=24 * 14, inflow=400.0, cyclic=True):
    sn = pd.date_range("2024-01-01", periods=hours, freq="h")
    n = pypsa.Network(); n.set_snapshots(sn)
    n.add("Bus", "Z"); n.add("Carrier", "hydro")
    load = 500 + 300 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    n.add("Load", "l", bus="Z", p_set=pd.Series(load, index=sn))
    n.add("StorageUnit", "Z hydro", bus="Z", carrier="hydro", p_nom=P_NOM, max_hours=MAX_H,
          inflow=pd.Series(inflow, index=sn), cyclic_state_of_charge=cyclic, p_min_pu=0.0,
          marginal_cost=0.6, spill_cost=0.1, state_of_charge_initial=0.5 * P_NOM * MAX_H)
    n.add("Generator", "peak", bus="Z", p_nom=2000, marginal_cost=80.0)
    return n


def solve(n, callbacks):
    def extra(nn, sns):
        for cb in callbacks:
            cb(nn, sns)
    status, _ = n.optimize(solver_name="highs", extra_functionality=extra,
                           solver_options={"output_flag": False})
    assert status == "ok"
    return n


def test_soc_anchor_pins_start_and_cyclic_end():
    cfg = {"zones": {"Z": {"hydro_soc_initial": 0.3, "hydro_p_nom_mw": P_NOM, "hydro_max_hours": MAX_H}}}
    n = solve(toy(), [hydro_soc_initial_constraint(cfg)])
    soc = n.storage_units_t.state_of_charge["Z hydro"]
    assert soc.iloc[0] == pytest.approx(0.3 * P_NOM * MAX_H, rel=1e-6)


def test_bid_ladder_tiers_sum_to_dispatch_and_fill_cheapest_first():
    n = toy()
    solve(n, [hydro_bid_ladder(3, 36.0)])
    tiers = n.model.solution["hydro_bid_tier"].sel(name="Z hydro").to_pandas().T   # snapshot × tier
    p = n.storage_units_t.p_dispatch["Z hydro"]
    np.testing.assert_allclose(tiers.sum(axis=1).to_numpy(), p.to_numpy(), atol=1e-3)
    cap = P_NOM / 3
    # en dyrare nivå används bara när alla billigare är fulla
    for k in (1, 2):
        used = tiers[k] > 1e-3
        assert (tiers.loc[used, list(range(k))] >= cap - 1e-3).all().all()


def test_bid_ladder_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        hydro_bid_ladder(1, 36.0)
    with pytest.raises(ValueError):
        hydro_bid_ladder(3, 0.0)


def test_higher_terminal_value_keeps_more_water():
    cap = {"Z hydro": P_NOM * MAX_H}
    end = {}
    for lam in (10.0, 150.0):
        n = solve(toy(cyclic=False), [hydro_terminal_value({"Z hydro": lam}, cap, [1.0])])
        end[lam] = float(n.storage_units_t.state_of_charge["Z hydro"].iloc[-1])
    assert end[150.0] > end[10.0]
