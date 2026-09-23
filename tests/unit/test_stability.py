"""Stabilitetsmätningen (nordpsa/analysis/stability.py) på ett olöst leksaksnät med
handsatt dispatch: klassningen per (komponent, carrier), KVV-länken och lo/hi/max-gränserna."""
import numpy as np
import pandas as pd
import pypsa
import pytest

from nordpsa.analysis.stability import stability_data, stability_metrics, unit_table

SD = stability_data()
X_T = SD["x_t"]


def build():
    sn = pd.date_range("2023-01-02", periods=4, freq="h")
    n = pypsa.Network()
    n.set_snapshots(sn)
    for b, c in [("A", "AC"), ("B", "AC"), ("A heat", "heat"), ("A chp fuel", "chp fuel")]:
        n.add("Bus", b, carrier=c)
    n.add("StorageUnit", "A hydro", bus="A", carrier="hydro", p_nom=1000, max_hours=100)
    n.add("Generator", "A hydro_ror", bus="A", carrier="hydro", p_nom=200,
          p_min_pu=0.5, p_max_pu=0.5)
    n.add("Generator", "A nuclear exp", bus="A", carrier="nuclear", p_nom=0,
          p_nom_extendable=True)
    n.add("Generator", "A wind", bus="A", carrier="wind_onshore", p_nom=3000)
    n.add("Generator", "A market", bus="A", carrier="market", p_nom=2000, p_min_pu=-1)
    n.add("Generator", "A heat mustrun", bus="A heat", carrier="thermal", p_nom=500,
          p_min_pu=1, p_max_pu=1)
    n.add("Link", "A chp", bus0="A chp fuel", bus1="A", carrier="heat chp", p_nom=400,
          efficiency=0.25)
    n.add("Link", "A elboiler", bus0="A", bus1="A heat", carrier="heat chp", p_nom=100)
    n.add("Generator", "B thermal", bus="B", carrier="thermal", p_nom=300,
          p_min_pu=1, p_max_pu=1)
    # handsatt "lösning"
    n.storage_units_t.p_dispatch = pd.DataFrame({"A hydro": [0, 150, 300, 1000.0]}, index=sn)
    n.generators_t.p = pd.DataFrame({"A hydro_ror": 100.0, "B thermal": 300.0,
                                     "A wind": [0, 500, 1000, 2000.0], "A market": 800.0},
                                    index=sn)
    n.links_t.p0 = pd.DataFrame({"A chp": [0, 400, 200, 400.0]}, index=sn)
    return n


@pytest.fixture(scope="module")
def net():
    n = build()
    return n, unit_table(n, SD)


def test_classification(net):
    _, u = net
    assert u.loc["A hydro", "tech"] == "hydro_res"
    assert u.loc["A hydro_ror", "tech"] == "hydro_ror"
    assert u.loc["A nuclear exp", "tech"] == "nuclear_new"          # namnsuffix före carrier
    assert u.loc["A wind", "tech"] == "ibr"
    assert u.loc["A market", "tech"] == "hvdc"
    assert u.loc["A hydro_ror", "fixed"] and not u.loc["A hydro", "fixed"]
    # värmebussen räknas inte, inte heller en länk vars bus1 är en värmebuss
    assert "A heat mustrun" not in u.index and "A elboiler" not in u.index


def test_chp_link_counts_electric_side(net):
    _, u = net
    t = SD["tech"]["chp"]
    r = u.loc["A chp"]
    assert r.zone == "A" and r.component == "Link"
    assert r.e_coef == pytest.approx(0.25 * t["H"] / t["cos_phi"])
    assert r.s_coef == pytest.approx(0.25 / ((t["xd2"] + X_T) * t["cos_phi"]))


def test_online_bounds(net):
    n, u = net
    ts = stability_metrics(n, u, {"B": 0.0})
    res, ror, chp = (SD["tech"][k] for k in ("hydro_res", "hydro_ror", "chp"))
    e_res = res["H"] / res["cos_phi"]
    e_ror = 100 * ror["H"] / ror["cos_phi"]                        # fast: u = 0,5·200
    e_chp = 0.25 * chp["H"] / chp["cos_phi"]
    p_res = np.array([0, 150, 300, 1000.0])
    p_chp = np.array([0, 400, 200, 400.0])
    lo = (e_res * p_res + e_ror + e_chp * p_chp) / 1e3
    hi = (e_res * np.minimum(p_res / res["m_min"], 1000)
          + e_ror + e_chp * np.minimum(p_chp / chp["m_min"], 400)) / 1e3
    mx = np.full(4, (e_res * 1000 + e_ror + e_chp * 400) / 1e3)
    np.testing.assert_allclose(ts[("Ek_lo", "A")], lo)
    np.testing.assert_allclose(ts[("Ek_hi", "A")], hi)
    np.testing.assert_allclose(ts[("Ek_max", "A")], mx)
    assert (ts["Ek_lo"]["A"] <= ts["Ek_hi"]["A"] + 1e-12).all()
    # sync_weight 0 utesluter B ur systemet men inte ur zonvärdet
    assert ts[("Ek_lo", "B")].gt(0).all()
    np.testing.assert_allclose(ts[("Ek_lo", "SYSTEM")], ts[("Ek_lo", "A")])
    # SCR mot inmatad effekt: bara vinden räknas (hvdc har ibr_w 0), NaN utan inmatning
    np.testing.assert_allclose(ts[("P_ibr", "A")], 3.0)
    wind = np.array([np.nan, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(ts[("P_ibr_out", "A")], np.nan_to_num(wind))
    np.testing.assert_allclose(ts[("SCR_max", "A")], ts[("Sk_max", "A")] / wind)
