"""Inställningslagren: defaults ← värld ← experiment ← --set, och arvet i dispatch."""
import pytest
import yaml

from nordpsa import settings


def test_expand_default_world_is_2040():
    s = settings.resolve("expansion", "2040_svk_mm")
    assert s["run"] == {"mode": "expansion", "world": "2040_svk_mm", "capacities": "optimized",
                        "experiments": [], "set": []}
    assert s["scenario"]["cost"] == "svk_2040"
    assert s["scenario"]["demand"] == "svk_2040_mm"
    assert s["heat"]["enabled"] is True
    assert [z for z, _, _ in s["nuclear"]["add"]] == ["SE-S", "SE-N", "FI"]


def test_today_world_has_todays_system_and_own_anchors():
    s = settings.resolve("dispatch", "today", capacities="config")
    assert s["scenario"]["cost"] is None and s["scenario"]["demand"] is None
    assert s["nuclear"]["add"] == [] and s["heat"]["enabled"] is False
    assert s["dispatch"]["terminal_anchor"]["SE-N"] == pytest.approx(27.1)


def test_today_world_refuses_expansion():
    with pytest.raises(SystemExit, match="tillåter inte"):
        settings.resolve("expansion", "today")


def test_unknown_world_is_reported():
    with pytest.raises(SystemExit, match="okänd värld"):
        settings.resolve("expansion", "nope")


def test_set_parses_yaml_values():
    assert settings.parse_set("market.ntc_scale=0.5") == {"market": {"ntc_scale": 0.5}}
    assert settings.parse_set("hydro.bid_ladder=[5, 34.6]") == {"hydro": {"bid_ladder": [5, 34.6]}}
    assert settings.parse_set("voll=null") == {"voll": None}
    # on/off ska förbli strängar (HiGHS vill ha strängen, YAML 1.1 gör dem till bool)
    assert settings.parse_set("solver.run_crossover=on") == {"solver": {"run_crossover": "on"}}
    with pytest.raises(SystemExit):
        settings.parse_set("utan_likhetstecken")


@pytest.mark.parametrize("bad", ["hydro.bid_ladders=[3,36]", "scenario.lowhydro=0.6",
                                 "dispatch.terminal_ankare=70"])
def test_unknown_key_is_an_error(bad):
    with pytest.raises(SystemExit, match="okänd inställning"):
        settings.resolve("expansion", "2040_svk_mm", sets=[bad])


def test_free_tables_merge_key_by_key():
    s = settings.resolve("expansion", "2040_svk_mm", sets=[
        "grid.ntc_override.SE-N:SE-S=7600", "grid.ntc_override.NO-N:NO-S=900",
        "hydro.operation.min_hourly_frac=0.1", "solver.user_bound_scale=-6"])
    assert s["grid"]["ntc_override"] == {"SE-N:SE-S": 7600, "NO-N:NO-S": 900}
    assert s["hydro"]["operation"] == {"min_hourly_frac": 0.1}
    assert s["solver"] == {"user_bound_scale": -6}


def test_later_layer_wins_and_none_replaces_table():
    s = settings.resolve("expansion", "2040_svk_mm", experiments=["lowhydro06"],
                         sets=["scenario.low_hydro.factor=0.7", "hydro.ror_hifreq=null"])
    assert s["scenario"]["low_hydro"] == {"factor": 0.7, "year": 2024}
    assert s["hydro"]["ror_hifreq"] is None


def test_every_experiment_file_resolves():
    from nordpsa.settings import CONFIG_DIR
    for f in sorted((CONFIG_DIR / "experiments").glob("*.yaml")):
        settings.resolve("expansion", "2040_svk_mm", experiments=[f.stem])


def test_validation_rules():
    with pytest.raises(SystemExit, match="battery_total"):
        settings.resolve("dispatch", "today", sets=["scenario.battery_total=[25, 4]"],
                         capacities="config")
    with pytest.raises(SystemExit, match="el_tax_zero"):
        settings.resolve("dispatch", "today", sets=["heat.el_tax_zero=true"], capacities="config")


def _write_source(results_dir, label, **sets):
    s = settings.resolve("expansion", "2040_svk_mm",
                         sets=[f"{k}={v}" for k, v in sets.items()])
    (results_dir / label).mkdir()
    (results_dir / label / "run_config.yaml").write_text(yaml.safe_dump(s))
    return s


def test_dispatch_inherits_world_but_not_per_command_sections(results_dir):
    _write_source(results_dir, "run001", **{"scenario.low_hydro": "{factor: 0.6, year: 2024}",
                                            "market.ntc_scale": 0.5,
                                            "period.resolution_hours": 3,
                                            "dispatch.terminal_segments": 7})
    d = settings.resolve_from_run("run001")
    assert d["run"]["mode"] == "dispatch" and d["run"]["capacities"] == "run001"
    assert d["scenario"]["low_hydro"] == {"factor": 0.6, "year": 2024}   # världen ärvs
    assert d["market"]["ntc_scale"] == 0.5
    assert d["period"]["resolution_hours"] == 2          # kommandobundet: från defaults
    assert d["dispatch"]["terminal_segments"] == 20      # likaså


def test_dispatch_from_run_without_config_is_explained(results_dir):
    (results_dir / "old_run").mkdir()
    with pytest.raises(SystemExit, match="run_config.yaml saknas"):
        settings.resolve_from_run("old_run")
