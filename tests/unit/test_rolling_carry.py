"""Rullande horisont: batterier och Stores bärs över mellan fönstren (dispatch.carry_storage)."""
import numpy as np
import pandas as pd
import pypsa

from nordpsa.solve import solve_rolling_horizon

HOURS = 24 * 7 * 3                                   # tre fönster à en vecka


def toy():
    sn = pd.date_range("2024-01-01", periods=HOURS, freq="h")
    n = pypsa.Network(); n.set_snapshots(sn)
    for c in ("hydro", "battery", "EV"):
        n.add("Carrier", c)
    n.add("Bus", "SE-N"); n.add("Bus", "SE-N EV")
    load = 600 + 300 * np.sin(np.arange(HOURS) * 2 * np.pi / 24)
    n.add("Load", "l", bus="SE-N", p_set=pd.Series(load, index=sn))
    n.add("StorageUnit", "SE-N hydro", bus="SE-N", carrier="hydro", p_nom=900, max_hours=200,
          inflow=pd.Series(400.0, index=sn), cyclic_state_of_charge=True, marginal_cost=0.6)
    n.add("Generator", "peak", bus="SE-N", p_nom=2000, marginal_cost=80.0)
    n.add("Generator", "wind", bus="SE-N", p_nom=800, marginal_cost=0.0,
          p_max_pu=pd.Series(0.5 + 0.5 * np.sin(np.arange(HOURS) * 2 * np.pi / 70), index=sn))
    n.add("StorageUnit", "SE-N battery", bus="SE-N", carrier="battery", p_nom=200, max_hours=2,
          cyclic_state_of_charge=True, efficiency_store=0.95, efficiency_dispatch=0.95)
    # EV: laddare → lager → körlast som varierar över veckan
    n.add("Link", "SE-N EV charger", bus0="SE-N", bus1="SE-N EV", p_nom=300, carrier="EV")
    n.add("Store", "SE-N EV store", bus="SE-N EV", e_nom=5000, e_cyclic=True, carrier="EV")
    drive = 100 + 80 * np.sin(np.arange(HOURS) * 2 * np.pi / (24 * 7))
    n.add("Load", "SE-N EV drive", bus="SE-N EV", p_set=pd.Series(drive, index=sn))
    return n


def run(carry):
    n = toy()
    cfg = {"solver": {"name": "highs", "output_flag": False},
           "zones": {"SE-N": {"hydro_soc_initial": 0.5}}}
    d = {"rolling_weeks": 1, "lookahead_weeks": 1, "terminal_segments": 5,
         "terminal_curve": "config/terminal_curves/terminal_curve_2040_gemini_v12.yaml",
         "terminal_anchor": None, "carry_storage": carry, "storage_initial_frac": 0.5}
    ok, res = solve_rolling_horizon(n, cfg, d, res=1)
    assert ok
    return n, res


def jumps(e, outflow):
    """Lagerbalansens rest e[t] − e[t−1] + utflöde[t]; 0 om lagret är kontinuerligt."""
    return (e - e.shift(1) + outflow).iloc[1:].abs()


def test_carry_storage_removes_jumps_at_window_boundaries():
    n, res = run(True)
    e = res["h2_store_soc"]["SE-N EV store"]
    ch = res["flows"]["SE-N EV charger"]
    drive = n.loads_t.p_set["SE-N EV drive"]
    assert jumps(e, drive - ch).max() < 1e-3          # EV-lagret: inga hopp
    soc = res["hydro_soc"]["SE-N battery"]
    p = res["dispatch_hydro"]["SE-N battery"]
    exp = soc.shift(1) - p.clip(lower=0) / 0.95 + (-p).clip(lower=0) * 0.95
    assert (soc - exp).iloc[1:].abs().max() < 1e-3    # batteriet: inga hopp
    assert e.iloc[0] > 0                              # startnivån (50 %) används


def test_old_behaviour_jumps_without_carry():
    n, res = run(False)
    e = res["h2_store_soc"]["SE-N EV store"]
    ch = res["flows"]["SE-N EV charger"]
    drive = n.loads_t.p_set["SE-N EV drive"]
    j = jumps(e, drive - ch)
    boundaries = [n.snapshots.get_loc(t) - 1 for t in n.snapshots[::24 * 7][1:]]
    assert j.iloc[boundaries].max() > 1.0             # hopp vid fönstergränserna
