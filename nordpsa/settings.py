"""Körinställningar: laddning i lager, --set, validering och sparning.

    defaults.yaml  ←  worlds/<värld>.yaml  ←  experiment …  ←  --set …

`resolve()` ger ett fullständigt upplöst dict som är det ENDA som styr en körning, och
som sparas som results/<run>/run_config.yaml. En dispatch ärver sin källkörnings värld
ur den filen (se `resolve_from_run`).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT         = Path(__file__).resolve().parents[1]
CONFIG_DIR   = ROOT / "config"
RESULTS_DIR  = ROOT / "results"
DEFAULTS     = CONFIG_DIR / "defaults.yaml"
WORLDS_DIR   = CONFIG_DIR / "worlds"
MODES        = ("expansion", "dispatch")

# Sektioner som hör till KOMMANDOT, inte till världen: en dispatch tar dem från
# defaults + källans världsfil i stället för från källkörningen. Kalibreras
# terminalkurvan om får alltså en omdispatch av en gammal expansion den nya kurvan.
PER_COMMAND = ("period", "solver", "dispatch", "expansion")


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _free_paths(tree: dict, prefix: str = "") -> set:
    """Sökvägar vars default är en TOM tabell: deras nycklar är zon-, kabel- eller
    parameternamn och kontrolleras inte, och lager slås ihop nyckel för nyckel."""
    out = set()
    for k, v in tree.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out |= {path} if not v else _free_paths(v, path)
    return out


_FREE: set | None = None


def _is_free(path: str) -> bool:
    global _FREE
    if _FREE is None:
        _FREE = _free_paths(_load_yaml(DEFAULTS))
    return any(path == f or path.startswith(f + ".") for f in _FREE)


def merge(base: dict, over: dict, where: str, path: str = "", strict: bool = True) -> dict:
    """Djup sammanslagning av `over` in i en KOPIA av `base`.

    Okända nycklar är fel (fångar felstavningar), utom under fria tabeller (se
    _free_paths), som slås ihop nyckel för nyckel. Är base-värdet None eller en
    lista ersätts hela värdet (t.ex. `low_hydro: {factor: …}`, `nuclear.add`).
    `strict=False` stänger av schemat helt (används mot zones.yaml-tabeller)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        key = f"{path}.{k}" if path else str(k)
        if k not in out and strict and not _is_free(path):
            raise SystemExit(f"{where}: okänd inställning '{key}'")
        cur = out.get(k)
        if isinstance(cur, dict) and isinstance(v, dict):
            out[k] = merge(cur, v, where, key, strict=strict)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_set(spec: str) -> dict:
    """'a.b.c=VÄRDE' → {'a': {'b': {'c': VÄRDE}}}. Värdet tolkas som YAML ('0.5' → 0.5,
    'null' → None, '[3, 36]' → lista), utom att on/off förblir strängar (HiGHS vill ha
    strängen, YAML 1.1 gör dem till bool)."""
    if "=" not in spec:
        raise SystemExit(f"--set: förväntade NYCKEL=VÄRDE, fick '{spec}'")
    key, raw = spec.split("=", 1)
    val = yaml.safe_load(raw.strip())
    if isinstance(val, bool) and raw.strip().lower() not in ("true", "false"):
        val = raw.strip()
    out: dict = {}
    node = out
    parts = [p for p in key.strip().split(".") if p]
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = val
    return out


def _world_file(world: str) -> tuple[dict, list]:
    path = WORLDS_DIR / f"{world}.yaml"
    if not path.exists():
        have = sorted(p.stem for p in WORLDS_DIR.glob("*.yaml"))
        raise SystemExit(f"okänd värld '{world}' (finns: {', '.join(have)})")
    doc = _load_yaml(path)
    modes = doc.pop("modes", list(MODES))
    return doc, modes


def _layers(settings: dict, experiments: list[str], sets: list[str]) -> dict:
    for exp in experiments:
        p = Path(exp)
        if not p.is_absolute() and not p.exists():
            p = CONFIG_DIR / "experiments" / exp
            if p.suffix != ".yaml":
                p = p.with_suffix(".yaml")
        if not p.exists():
            raise SystemExit(f"experimentfil saknas: {exp}")
        settings = merge(settings, _load_yaml(p), f"experiment {p.name}")
    for spec in sets:
        settings = merge(settings, parse_set(spec), f"--set {spec}")
    return settings


def resolve(mode: str, world: str, experiments=(), sets=(), capacities: str = "optimized") -> dict:
    """Upplösta inställningar för en ny körning i värld `world`."""
    defaults = _load_yaml(DEFAULTS)
    wdoc, modes = _world_file(world)
    if mode not in modes:
        raise SystemExit(f"världen '{world}' tillåter inte läget '{mode}' (bara {modes})")
    s = merge(defaults, wdoc, f"värld {world}")
    s = _layers(s, list(experiments), list(sets))
    return _stamp(s, mode, world, capacities, experiments, sets)


def resolve_from_run(label: str, experiments=(), sets=()) -> dict:
    """Dispatch av en tidigare körning: världen ärvs ur källans run_config.yaml,
    de kommandobundna sektionerna (PER_COMMAND) tas från defaults + källans världsfil."""
    src_path = RESULTS_DIR / label / "run_config.yaml"
    if not src_path.exists():
        raise SystemExit(f"--from {label}: {src_path} saknas — källkörningen måste vara "
                         "gjord med nordpsa expand/dispatch/today")
    src = _load_yaml(src_path)
    world = src["run"]["world"]
    wdoc, modes = _world_file(world)
    if "dispatch" not in modes:
        raise SystemExit(f"världen '{world}' tillåter inte dispatch")
    fresh = merge(_load_yaml(DEFAULTS), wdoc, f"värld {world}")
    s = {k: v for k, v in src.items() if k != "run"}
    for k in PER_COMMAND:
        s[k] = fresh[k]
    s = merge(fresh, s, f"{label}/run_config.yaml")   # kontrollerar att nycklarna finns kvar
    s = _layers(s, list(experiments), list(sets))
    return _stamp(s, "dispatch", world, label, experiments, sets)


def _stamp(s: dict, mode, world, capacities, experiments, sets) -> dict:
    s["run"] = {"mode": mode, "world": world, "capacities": capacities,
                "experiments": list(experiments), "set": list(sets)}
    validate(s)
    return s


def validate(s: dict) -> None:
    """Regler som gäller den färdigupplösta kombinationen."""
    mode = s["run"]["mode"]
    if s["scenario"]["battery_total"] is not None and not s["scenario"]["demand"]:
        raise SystemExit("scenario.battery_total kräver scenario.demand")
    if s["heat"]["el_tax_zero"] and not s["heat"]["enabled"]:
        raise SystemExit("heat.el_tax_zero kräver heat.enabled (skatten sitter på värmebussen)")
    if mode == "dispatch":
        d = s["dispatch"]
        if not d["terminal_curve"]:
            raise SystemExit("dispatch.terminal_curve krävs: den rullande horisonten har "
                             "inget annat terminalvärde")
        if d["rolling_weeks"] < 1 or d["lookahead_weeks"] < 0:
            raise SystemExit("dispatch.rolling_weeks ≥ 1 och lookahead_weeks ≥ 0")
    lh = s["scenario"]["low_hydro"]
    if lh is not None and set(lh) - {"factor", "year"}:
        raise SystemExit("scenario.low_hydro: bara nycklarna factor och year")
    bl = s["hydro"]["bid_ladder"]
    if bl is not None and (len(bl) != 2 or int(bl[0]) < 1):
        raise SystemExit("hydro.bid_ladder: [K, BREDD] med K ≥ 1, eller null")
    st = s["stability"]
    if st["enabled"]:
        if mode == "expansion":
            raise SystemExit("stability.enabled: bara dispatch än så länge (expansionens "
                             "kapacitetsvillkor är inte implementerat)")
        if not st["ek_system_gws"] and not st["ek_zone_floor_gws"]:
            raise SystemExit("stability.enabled kräver ek_system_gws eller ek_zone_floor_gws")
    if st["ek_system_gws"] is not None and float(st["ek_system_gws"]) <= 0:
        raise SystemExit("stability.ek_system_gws: > 0 eller null")
    if any(float(v) < 0 for v in st["ek_zone_floor_gws"].values()):
        raise SystemExit("stability.ek_zone_floor_gws: golven måste vara ≥ 0")
    if any(not 0 <= float(v) <= 1 for v in st["sync_weight"].values()):
        raise SystemExit("stability.sync_weight: vikterna måste ligga i [0, 1]")
    pen = s["dispatch"]["stability_slack_penalty"]
    if pen is not None and float(pen) <= 0:
        raise SystemExit("dispatch.stability_slack_penalty: > 0, eller null för hårt krav")


def save(s: dict, label: str) -> Path:
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    p = out / "run_config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(s, f, sort_keys=False, allow_unicode=True)
    return p


def dump(s: dict) -> str:
    return yaml.safe_dump(s, sort_keys=False, allow_unicode=True)
