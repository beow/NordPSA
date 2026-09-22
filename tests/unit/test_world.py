"""world.prepare_config mot den riktiga zones.yaml: scenarier, NTC och deras ordning."""
import pytest

from nordpsa import settings, world
from nordpsa.inputs import load_config


def _prepare(mode="expansion", world_name="2040_svk_mm", sets=(), experiments=(),
             capacities="optimized"):
    s = settings.resolve(mode, world_name, experiments, list(sets), capacities=capacities)
    cfg = load_config()
    extras = world.prepare_config(cfg, s)
    return cfg, extras


def _link(cfg, z0, z1):
    return next(l[2] for l in cfg["links"] if l[0] == z0 and l[1] == z1)


def _cable(cfg, name):
    return next(m[2] for m in cfg["market_connections"] if m[0] == name)


def test_today_freezes_every_technology_and_keeps_todays_grid():
    base = load_config()
    cfg, extras = _prepare("dispatch", "today", capacities="config")
    assert all(not t.get("extendable", False)
               for t in cfg["costs"].values() if isinstance(t, dict))
    assert cfg["links"] == base["links"]                           # inget framtida nät
    assert [b[0] for b in extras["batteries"]] == [b[0] for b in base["baseline_batteries"]]
    assert cfg["costs"]["hydro"]["spill_cost_eur_per_mwh"] == 0.1  # dispatch-värdet


def test_2040_grid_applies_expansion_then_scenario_overrides():
    cfg, _ = _prepare()
    assert _link(cfg, "SE-N", "FI") == 2000        # links_expansion_overrides (Aurora)
    assert _link(cfg, "SE-N", "SE-S") == 9000      # demand-scenariots snitt 2
    assert _link(cfg, "NO-N", "NO-S") == 1500      # scenariot vinner över expansionsvärdet
    assert cfg["costs"]["hydro"]["spill_cost_eur_per_mwh"] == 50


def test_cli_ntc_override_wins_over_scenario():
    cfg, _ = _prepare(sets=["grid.ntc_override.SE-N:SE-S=5000"])
    assert _link(cfg, "SE-N", "SE-S") == 5000


def test_market_scale_then_named_override():
    cfg, _ = _prepare(sets=["market.ntc_scale=0.5", "market.ntc_override.FI EE=1234"])
    assert _cable(cfg, "FI EE") == 1234            # namngiven kabel vinner
    assert _cable(cfg, "DK DE") == pytest.approx(4500 * 0.5)   # scenariots värde, halverat


def test_unknown_link_or_cable_is_an_error():
    with pytest.raises(SystemExit, match="ntc_override"):
        _prepare(sets=["grid.ntc_override.SE-N:DK=1"])
    with pytest.raises(SystemExit, match="kontinentkabel"):
        _prepare(sets=["market.ntc_override.XX YY=1"])


def test_no_market_removes_all_cables():
    cfg, _ = _prepare(sets=["market.enabled=false"])
    assert cfg["market_connections"] == []


def test_cost_scenario_adds_interest_during_construction():
    cfg, _ = _prepare()
    base = load_config()
    p = base["cost_scenarios"]["svk_2040"]["nuclear"]
    idc = 1 + p["build_years"] / 2 * base["costs"]["discount_rate"]
    assert cfg["costs"]["nuclear"]["overnight_eur_per_w"] == pytest.approx(p["oc_eur_per_kw"] * idc / 1000)
    # fom_fraction räknas mot OC inkl. byggränta så att absolut O&M bevaras
    assert cfg["costs"]["nuclear"]["fom_fraction"] * p["oc_eur_per_kw"] * idc == pytest.approx(p["fom_eur_per_kw"])


def test_battery_total_rescales_scenario_batteries():
    _, extras = _prepare(experiments=["batt25_4h"])
    assert sum(p for _, p, _ in extras["batteries"]) == pytest.approx(25_000)
    assert {h for _, _, h in extras["batteries"]} == {4.0}


def test_heat_tax_zero():
    cfg, _ = _prepare(experiments=["notax"])
    assert cfg["heat"]["enabled"] and cfg["heat"]["el_tax_eur_per_mwh"] == 0.0


def test_discount_rates_by_zone():
    cfg, _ = _prepare(experiments=["se_disc3"])
    assert cfg["costs"]["nuclear"]["discount_rate_by_zone"] == {"SE-N": 0.03, "SE-S": 0.03}
    assert cfg["costs"]["wind_offshore"]["discount_rate_by_zone"] == {"SE-N": 0.03, "SE-S": 0.03}


def test_extra_load_and_soc_anchor_override():
    cfg, _ = _prepare(sets=["scenario.extra_load_mw.SE-S=1000", "hydro.soc_initial.FI=0.4"])
    assert cfg["additional_load_mw"]["SE-S"] >= 1000
    assert cfg["zones"]["FI"]["hydro_soc_initial"] == 0.4
