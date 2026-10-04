"""Numerisk städning (nordpsa/numerics.py) och objektivkonstanten (solve.solve) på leksaksnät."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from nordpsa import numerics, solve


def toy(hours=48):
    sn = pd.date_range("2024-01-01", periods=hours, freq="h")
    n = pypsa.Network(); n.set_snapshots(sn)
    n.add("Bus", "Z")
    load = 500 + 200 * np.sin(np.arange(hours) * 2 * np.pi / 24)
    n.add("Load", "l", bus="Z", p_set=pd.Series(load, index=sn))
    cf = np.clip(np.sin((np.arange(hours) % 24 - 6) * np.pi / 12), 0, None)
    cf[cf == 0] = 4e-6                                   # nattbrus som i data
    n.add("Generator", "sol", bus="Z", p_nom=300, p_nom_extendable=True, p_nom_min=300, p_nom_max=2000,
          capital_cost=50.0, marginal_cost=0.0, p_max_pu=pd.Series(cf, index=sn))
    ror = pd.Series(np.where(np.arange(hours) % 7 == 0, 5e-4, 0.4), index=sn)   # must-run med brus
    n.add("Generator", "ror", bus="Z", p_nom=100, p_max_pu=ror, p_min_pu=ror)
    mc = pd.Series(np.where(np.arange(hours) % 5 == 0, 3.7e-4, 60.0), index=sn)
    n.add("Generator", "import", bus="Z", p_nom=2000, marginal_cost=mc)
    n.add("Generator", "tie", bus="Z", p_nom=1, marginal_cost=0.01)          # avsiktlig skiljekostnad
    return n


def test_tidy_zeroes_noise_and_keeps_pmin_below_pmax():
    n = toy()
    out = numerics.tidy(n, pmax_eps=1e-3, cost_eps=5e-3)
    pmax = n.get_switchable_as_dense("Generator", "p_max_pu")
    pmin = n.get_switchable_as_dense("Generator", "p_min_pu")
    assert ((pmax["sol"] == 0) | (pmax["sol"] >= 1e-3)).all()
    assert (pmin <= pmax + 1e-12).all().all()
    assert (pmax["ror"].min() == 0) and (pmin["ror"].min() == 0)
    mc = n.get_switchable_as_dense("Generator", "marginal_cost")
    assert ((mc["import"] == 0) | (mc["import"] == 60)).all()
    assert n.generators.at["tie", "marginal_cost"] == pytest.approx(0.01)     # skiljekostnaden orörd
    assert out["Generator p_max_pu"] > 0 and out["Generator marginal_cost"] > 0


def test_tidy_with_zero_eps_changes_nothing():
    n, ref = toy(), toy()
    numerics.tidy(n, pmax_eps=0.0, cost_eps=0.0)
    for attr in ("p_max_pu", "p_min_pu", "marginal_cost"):
        pd.testing.assert_frame_equal(n.get_switchable_as_dense("Generator", attr),
                                      ref.get_switchable_as_dense("Generator", attr))


def test_objective_constant_var_does_not_change_total_cost():
    cfg = {"solver": {"name": "highs", "output_flag": False}, "zones": {}}
    tot = {}
    for ocv in (True, False):
        n = toy()
        assert solve.solve(n, cfg, extra_callbacks=[], objective_constant_var=ocv)
        tot[ocv] = float(n.objective) + float(n.objective_constant or 0.0)
        if not ocv:
            assert float(n.objective_constant or 0.0) == 0.0
    assert tot[False] == pytest.approx(tot[True], rel=1e-7)
