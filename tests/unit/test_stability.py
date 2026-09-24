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
    assert u.loc["A wind", "tech"] == "ibr_wind"
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


# ---- villkoren (constraints/stability.py) på ett löst leksaksnät --------------------------
from nordpsa.constraints.stability import (stability_constraints,  # noqa: E402
                                           stability_feasibility_report, stability_results)

E_RES = SD["tech"]["hydro_res"]["H"] / SD["tech"]["hydro_res"]["cos_phi"]   # MWs/MW
E_GAS = SD["tech"]["gas"]["H"] / SD["tech"]["gas"]["cos_phi"]
M_RES, M_GAS = SD["tech"]["hydro_res"]["m_min"], SD["tech"]["gas"]["m_min"]


def toy():
    """A: magasin 1000 MW + vind, B: gas 500 MW + vind. Vinden täcker lasten, så utan krav
    står magasin och gas still och systemet har ingen rotationsenergi alls."""
    sn = pd.date_range("2023-01-02", periods=6, freq="h")
    n = pypsa.Network()
    n.set_snapshots(sn)
    for b in ("A", "B"):
        n.add("Bus", b, carrier="AC")
    n.add("Link", "A-B", bus0="A", bus1="B", carrier="AC", p_nom=2000, p_min_pu=-1)
    n.add("StorageUnit", "A hydro", bus="A", carrier="hydro", p_nom=1000, max_hours=100,
          state_of_charge_initial=50000, marginal_cost=20)
    n.add("Generator", "A wind", bus="A", carrier="wind_onshore", p_nom=2000, p_max_pu=0.9)
    n.add("Generator", "B wind", bus="B", carrier="wind_onshore", p_nom=1000, p_max_pu=0.9)
    n.add("Generator", "B gas", bus="B", carrier="gas", p_nom=500, marginal_cost=100)
    n.add("Load", "A load", bus="A", p_set=800)
    n.add("Load", "B load", bus="B", p_set=300)
    return n


def solve_toy(sys_gws=None, floors=None, weights=None, penalty=None, scr=None, exempt=None):
    n = toy()
    sd = stability_data(sync_weight=weights or {})
    if exempt is not None:
        sd["scr_exempt"] = exempt
    cb = (stability_constraints(sd, sys_gws, floors or {}, penalty, scr)
          if (sys_gws or floors or scr) else None)
    status, _ = n.optimize(solver_name="highs", extra_functionality=cb)
    return n, status, stability_results(n) if status == "ok" else {}


def ek_gws(n, res):
    on = res["stability_online"]
    return pd.DataFrame({"A": on["A hydro"] * E_RES, "B": on["B gas"] * E_GAS}) / 1e3


@pytest.fixture(scope="module")
def reference():
    n, status, res = solve_toy()
    assert status == "ok"
    return n, res


def test_disabled_adds_nothing(reference):
    n, res = reference
    assert res == {}                                      # extract_results oförändrat
    assert not [c for c in n.model.constraints if "stability" in c]


def test_system_requirement_binds_and_costs_more(reference):
    ref, _ = reference
    assert ref.storage_units_t.p_dispatch["A hydro"].max() < 1e-6, "referensen körde hydro — svagt test"
    n, status, res = solve_toy(sys_gws=2.0)
    assert status == "ok"
    ek = ek_gws(n, res)
    assert (ek.sum(axis=1) >= 2.0 - 1e-6).all()
    assert n.objective > ref.objective + 1.0
    # billigast per MWs: magasinet (20 €/MWh · m_min/e) före gasen (100 · m_min/e)
    assert n.generators_t.p["B gas"].max() < 1e-6
    np.testing.assert_allclose(n.storage_units_t.p_dispatch["A hydro"], M_RES * 2000 / E_RES,
                               rtol=1e-6)
    assert (res["stability_dual"]["SYSTEM"] > 0).all()


def test_commitment_bounds_hold():
    n, _, res = solve_toy(sys_gws=2.5, floors={"B": 0.5})
    on = res["stability_online"]
    p = pd.concat([n.storage_units_t.p_dispatch["A hydro"], n.generators_t.p["B gas"]], axis=1)
    cap = pd.Series({"A hydro": 1000.0, "B gas": 500.0})
    mmin = pd.Series({"A hydro": M_RES, "B gas": M_GAS})
    on = on[p.columns]
    assert (p <= on + 1e-6).all().all()
    assert (p >= on * mmin - 1e-6).all().all()
    assert (on <= cap + 1e-6).all().all()


def test_zone_floor_is_local():
    n, status, res = solve_toy(floors={"B": 1.0})
    assert status == "ok"
    assert (ek_gws(n, res)["B"] >= 1.0 - 1e-6).all()
    assert n.generators_t.p["B gas"].min() > 0            # A:s magasin kan inte hjälpa B


def test_sync_weight_zero_excludes_zone_from_system():
    n, status, res = solve_toy(sys_gws=2.0, weights={"B": 0.0})
    assert status == "ok"
    assert (ek_gws(n, res)["A"] >= 2.0 - 1e-6).all()      # hela kravet på A


def test_hard_infeasible_soft_prices_the_shortfall():
    ek_max_a = 1000 * E_RES / 1e3                         # A:s tak; B räknas inte
    _, status, _ = solve_toy(sys_gws=5.0, weights={"B": 0.0})
    assert status != "ok"
    n, status, res = solve_toy(sys_gws=5.0, weights={"B": 0.0}, penalty=100.0)
    assert status == "ok"
    np.testing.assert_allclose(res["stability_slack"]["SYSTEM"], (5.0 - ek_max_a) * 1e3,
                               rtol=1e-6)
    lines = stability_feasibility_report(toy(), stability_data(sync_weight={"B": 0.0}), 5.0, {})
    assert "INFEASIBLE" in lines[0]
    assert not "INFEASIBLE" in stability_feasibility_report(toy(), SD, 2.0, {})[0]


S_RES = 1 / ((SD["tech"]["hydro_res"]["xd2"] + X_T) * SD["tech"]["hydro_res"]["cos_phi"])
S_GAS = 1 / ((SD["tech"]["gas"]["xd2"] + X_T) * SD["tech"]["gas"]["cos_phi"])


def test_scr_floor_holds_against_infeed(reference):
    ref, _ = reference
    n, status, res = solve_toy(scr=1.5, exempt=[])
    assert status == "ok"
    on = res["stability_online"]
    sk = pd.DataFrame({"A": on["A hydro"] * S_RES, "B": on["B gas"] * S_GAS})
    ibr = pd.DataFrame({"A": n.generators_t.p["A wind"], "B": n.generators_t.p["B wind"]})
    assert (sk >= 1.5 * ibr - 1e-4).all().all()
    ref_ibr = ref.generators_t.p[["A wind", "B wind"]].sum(axis=1)
    assert (ref_ibr > 0).all(), "referensen hade ingen vind — svagt test"
    assert n.objective > ref.objective + 1.0
    assert {"SCR_A", "SCR_B"} <= set(res["stability_dual"])


def test_scr_exempt_zone_has_no_requirement():
    n, status, res = solve_toy(scr=1.5, exempt=["A", "B"])
    assert status == "ok"
    assert not [c for c in n.model.constraints if "scr" in c]
    n2, _, res2 = solve_toy(scr=1.5, exempt=["A"])
    assert "SCR_B" in res2["stability_dual"] and "SCR_A" not in res2["stability_dual"]


def test_gfm_share_reduces_scr_load(net):
    n, _ = net
    sd = stability_data(tech={"ibr_wind": {"ibr_w": 0.7}})       # 30 % av vinden nätbildande
    ts = stability_metrics(n, unit_table(n, sd))
    np.testing.assert_allclose(ts[("P_ibr_out", "A")], 0.7 * np.array([0, 0.5, 1.0, 2.0]))


# ---- expansion: investerbar kapacitet i villkoren -----------------------------------------
S_SC = 1 / ((SD["tech"]["syncon"]["xd2"] + X_T) * SD["tech"]["syncon"]["cos_phi"])
GFM = SD["tech"]["gfm"]


def toy_expansion(syncon_cost=None, gfm_cost=None, gas_ext=False):
    """En zon: vind 1000 MW (CF 0,8) täcker lasten 500; gas som dyr reserv. Att spilla
    vind kräver gas ⇒ nätstyrkan byggs hellre än att vinden spills."""
    sn = pd.date_range("2023-01-02", periods=4, freq="h")
    n = pypsa.Network()
    n.set_snapshots(sn)
    n.add("Bus", "Z", carrier="AC")
    n.add("Load", "Z load", bus="Z", p_set=500)
    n.add("Generator", "Z wind", bus="Z", carrier="wind_onshore", p_nom=1000, p_max_pu=0.8)
    n.add("Generator", "Z gas", bus="Z", carrier="gas", p_nom=0 if gas_ext else 1000,
          p_nom_extendable=gas_ext, capital_cost=1.0, marginal_cost=100)
    if syncon_cost is not None:
        n.add("Generator", "Z syncon", bus="Z", carrier="syncon", p_nom_extendable=True,
              p_min_pu=-0.01, p_max_pu=-0.01, capital_cost=syncon_cost)
    if gfm_cost is not None:
        n.add("StorageUnit", "Z battery gfm", bus="Z", carrier="battery_gfm",
              p_nom_extendable=True, max_hours=4, capital_cost=gfm_cost,
              cyclic_state_of_charge=True)
    return n


def solve_exp(n, **kw):
    cb = stability_constraints(dict(SD, scr_exempt=[]), None, {}, None, **kw)
    status, _ = n.optimize(solver_name="highs", extra_functionality=cb)
    assert status == "ok"
    return n


def test_expansion_builds_syncon_for_scr():
    n = solve_exp(toy_expansion(syncon_cost=1.0), scr_min=1.5)
    wind = n.generators_t.p["Z wind"]
    assert wind.min() > 490                                # vinden spills inte bort
    gas_on = stability_results(n)["stability_online"]["Z gas"]
    sk = n.generators.at["Z syncon", "p_nom_opt"] * S_SC + gas_on * (
        1 / ((SD["tech"]["gas"]["xd2"] + X_T) * SD["tech"]["gas"]["cos_phi"]))
    assert (sk >= 1.5 * wind - 1e-3).all()
    assert n.generators.at["Z syncon", "p_nom_opt"] > 100
    np.testing.assert_allclose(n.generators_t.p["Z syncon"],
                               -0.01 * n.generators.at["Z syncon", "p_nom_opt"], rtol=1e-6)


def test_expansion_gfm_battery_provides_strength_without_being_load():
    n = solve_exp(toy_expansion(gfm_cost=1.0), scr_min=1.5)
    cap = n.storage_units.at["Z battery gfm", "p_nom_opt"]
    wind = n.generators_t.p["Z wind"]
    assert n.generators_t.p["Z gas"].max() < 1e-6          # gasen är dyrare än batteriet
    # nätbildande batteri ger sk_pu·avail per MW oavsett drift och räknas inte som last
    np.testing.assert_allclose(cap * GFM["sk_pu"] * GFM["avail"], 1.5 * wind.max(), rtol=1e-6)


def test_expansion_commitment_capped_by_built_capacity():
    n = solve_exp(toy_expansion(gas_ext=True), scr_min=1.5)
    on = stability_results(n)["stability_online"]["Z gas"]
    assert (on <= n.generators.at["Z gas", "p_nom_opt"] + 1e-6).all()
    assert n.generators.at["Z gas", "p_nom_opt"] > 0       # utan syncon: gas byggs för styvhet


def test_investable_batteries_cost_scale_and_gfm_only():
    from nordpsa.inputs import load_config
    from nordpsa.network.costs import crf
    from nordpsa.network.storage import add_investable_batteries
    ccfg = load_config()["costs"]
    bc = ccfg["battery"]
    ann = crf(int(bc["lifetime_years"]), 0.06) + float(bc.get("fom_fraction", 0.025))
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2023-01-02", periods=2, freq="h"))
    n.add("Bus", "Z", carrier="AC")
    add_investable_batteries(n, ["Z"], ccfg, 0.06, 1.0, 4, True, cost_scale=0.5, gfm_extra=0.0)
    assert list(n.storage_units.index) == ["Z battery gfm"]          # bara nätbildande
    oc = 0.5 * (bc["power_eur_per_kw"] + 4 * bc["energy_eur_per_kwh"]) * 1e3
    assert n.storage_units.at["Z battery gfm", "capital_cost"] == pytest.approx(oc * ann)
    n2 = pypsa.Network()
    n2.set_snapshots(n.snapshots)
    n2.add("Bus", "Z", carrier="AC")
    add_investable_batteries(n2, ["Z"], ccfg, 0.06, 1.0, 4, True)
    assert set(n2.storage_units.index) == {"Z battery exp", "Z battery gfm"}
