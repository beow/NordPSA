"""Kommandoraden: de tre kommandona och --print-config (ingen lösning, inga data)."""
import pytest
import yaml

from nordpsa import cli


def _print_config(capsys, *argv):
    cli.main([*argv, "--print-config"])
    return yaml.safe_load(capsys.readouterr().out)


def test_expand_print_config(capsys):
    s = _print_config(capsys, "expand", "--year", "2024", "--resolution", "3")
    assert s["run"]["mode"] == "expansion"
    assert s["period"] == {"year": 2024, "resolution_hours": 3}


def test_today_print_config(capsys):
    s = _print_config(capsys, "today")
    assert s["run"]["world"] == "today" and s["run"]["capacities"] == "config"


def test_experiment_and_set(capsys):
    s = _print_config(capsys, "expand", "--experiment", "market50", "--set", "voll=null")
    assert s["market"]["ntc_scale"] == 0.5 and s["voll"] is None
    assert s["run"]["experiments"] == ["market50"]


def test_output_is_required_for_a_run():
    with pytest.raises(SystemExit, match="--output"):
        cli.main(["expand"])


def test_dispatch_requires_from():
    with pytest.raises(SystemExit):
        cli.main(["dispatch", "--print-config"])
