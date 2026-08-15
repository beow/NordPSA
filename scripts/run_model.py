"""
Bygger och löser NordPSA-modellen.

Läser:  data/processed/  + config/zones.yaml
Skriver: results/

Användning:
    python scripts/run_model.py
    python scripts/run_model.py --resolution 3    # kör på 3h upplösning
    python scripts/run_model.py --year 2024        # kör ett enstaka år
"""
import argparse
import shlex
import sys
from pathlib import Path

import pandas as pd
import pypsa
import yaml

# pandas 2.x använder Arrow-strängar som standard; PyPSA/xarray stöder inte det
pd.options.future.infer_string = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.network import (
    build_network,
    hydro_soc_initial_constraint,
    hydro_soc_terminal_pin_constraint,
    hydro_terminal_value,
    DEFAULT_TERMINAL_PROFILE,
    soc_terminal_pin_mwh,
    oc_budget_constraint,
    grid_cost_objective,
    hydro_operation_bounds,
    hydro_operation_constraints,
    hydro_operation_feasibility_report,
    _annualized_cost,
    _grid_capital_cost,
)

USD_TO_EUR = 0.926   # 1 USD ≈ 0.926 EUR (1 EUR ≈ 1.08 USD), 2026
VRE_CARRIERS = ("wind_onshore", "wind_offshore", "solar")

PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "zones.yaml"

# --- Kanonisk expansions-baseline (run250-konfen) ---------------------------
# Optionerna nedan är DEFAULT sedan 2026-08-05, så `python scripts/run_model.py`
# utan flaggor kör baselinen. Varje post har en avstängningsväg (se --help).
# Motsvarande config-default: snapshots.resolution_hours = 2.
DEFAULT_ADD_NUCLEAR = ["SE-S:10:201", "SE-N:10:202", "FI:10:203"]

# --vre-curtailment-cost: default BARA i dispatchläge (--dispatch eller --no-expansion,
# dvs frysta kapaciteter). I expansion måste den vara 0 — annars blir den en
# produktionssubvention och spärren i apply_vre_curtailment_cost slår till på varje
# baseline-körning. Nivån är kalibrerad mot vad som faktiskt går förlorat per MWh vid
# avkortning i 2040: ursprungsgarantier (GO). Elcertifikaten är noll (systemet stängt
# för nya anläggningar efter 2021, avslutas senast 2035, utreds för tidigare avslut) och
# CfD:er betalar inte i negativa timmar (CEEAG suspenderar stöd vid negativa priser, och
# tyska EEG §51 går till 1-timmesregel 2027). Kvar: GO, prognos 2-5 EUR/MWh.
# 5 = ÖVRE delen av det intervallet, valt av användaren; central skattning vore 2.
DEFAULT_VRE_CURTAILMENT_COST = 5.0

# --dispatch utan --resolution. Var 1h när omdispatchen var ett finupplöst komplement
# till en grov expansion; nu är dispatchmallen (run340) 2h och kurvan kalibrerad där.
DEFAULT_DISPATCH_RESOLUTION = 2

# --spill-cost: LÄGESBEROENDE, spegelbilden av --vre-curtailment-cost.
#
# EXPANSION 50: ett modelleringsräcke, ingen fysisk kostnad. Med gratis spill kan
# optimeraren bygga överskott av vind/sol och dumpa den undanträngda vattenkraften
# nästan gratis — systemets verkliga förmåga att absorbera VRE döljs och den
# överinvesterar. Mätt 2026-05-31: run43 (spill 1) gav 63 TWh fantomspill och NO-S vind
# 20,8 GW; run44 (spill 50) gav 0 spill och 13,1 GW, alltså −37 %.
#
# DISPATCH 0,1: med frysta kapaciteter finns inget investeringsbeslut att snedvrida, och
# då DUBBELRÄKNAR 50 vattnets värde. Kostnaden för att spilla ÄR det förlorade vattnets
# värde, och det bär LP:t redan som skuggpriset på SOC (λ ≈ 73-80). Vid taket har
# marginellt vatten noll lagringsvärde, så valet står mellan att producera till priset p
# eller betala c; modellen producerar så länge p > −c. Med c = 50 kör den alltså hydro
# ned till −50 EUR/MWh hellre än att spilla, vilket ingen verklig operatör gör — den
# förbileder. ⚠️ Verkningslöst i praktiken i dagens körningar (spill = 0 och priset ligger
# på 52-74 i alla timmar över 90 % fyllnad), så ändringen är principiell, inte numerisk.
DEFAULT_SPILL_COST_EXPANSION = 50.0
DEFAULT_SPILL_COST_DISPATCH  = 0.1

# Skrivs som 'defaults:'-rad i run_meta.txt. Körningar UTAN raden är gjorda före
# omläggningen och måste replayas mot dåtidens defaults (PRE_BASELINE_DEFAULTS).
BASELINE_DEFAULTS_TAG = "baseline-v1 (run250-konfen)"

# Defaultvärden som gällde FÖRE omläggningen. En --dispatch-replay av en körning
# som gjordes innan dess ska återge KÄLLANS värld, inte dagens defaults: körde
# run240 utan --hydro-restrictions ska omdispatchen också göra det. Nycklarna är
# dest-namn, värdena de gamla defaultarna, och tokens de flaggor som i källans
# argv innebär att användaren valde värdet MEDVETET (då rörs det inte).
PRE_BASELINE_DEFAULTS = {
    "hydro_restrictions":      (False, ("--hydro-restrictions", "--no-hydro-restrictions")),
    "add_heat":                (False, ("--add-heat", "--no-add-heat")),
    # spill_cost hörde hit när defaulten var 50 i båda lägena. Sedan den blev
    # LÄGESBEROENDE tas den alltid från det nya kommandot (se apply_dispatch_replay), och
    # en oangiven flagga löses till DEFAULT_SPILL_COST_DISPATCH = 0,1 — exakt det värde
    # den gamla återställningen gav via network.py:s fallback. Posten kunde alltså aldrig
    # längre fälla ut och är borttagen.
    "cost_scenario":           (None,  ("--cost-scenario",)),
    "demand_scenario":         (None,  ("--demand-scenario",)),
    "onwind_capfac_increase":  (0.0,   ("--onwind-capfac-increase",)),
    "offwind_capfac_increase": (0.0,   ("--offwind-capfac-increase",)),
    "nuclear_min_load":        (None,  ("--nuclear-min-load",)),
    "add_nuclear":             ([],    ("--add-nuclear", "--no-add-nuclear")),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_vre_extendable(n, cfg: dict, zones: list, n_years: float,
                        capital_in_objective: bool = True) -> list:
    """Gör wind_onshore/wind_offshore/solar i givna zoner investerbara, oavsett
    global expansion-toggle. Befintlig kapacitet låses som golv (p_nom_min).

    capital_in_objective=True  → capital_cost = annualiserad × n_years (lönsamhets-
                                 optimering, modellen bygger så mycket som lönar sig).
    capital_in_objective=False → capital_cost = 0 (sunk): kombineras med en
                                 likhets-budget som TVINGAR fram en given utgift;
                                 modellen optimerar bara mix + dispatch (dispatcheffekt).
    Returnerar lista av berörda namn."""
    r   = cfg["costs"]["discount_rate"]
    fom = cfg["costs"]["fom_fraction"]
    zones = set(zones)
    touched = []
    for name, gen in n.generators.iterrows():
        if gen.bus in zones and gen.carrier in VRE_CARRIERS:
            tcfg     = cfg["costs"][gen.carrier]
            existing = float(gen.p_nom)
            ann      = _annualized_cost(tcfg["overnight_eur_per_w"],
                                        tcfg["lifetime_years"], r, fom)
            n.generators.at[name, "p_nom_extendable"] = True
            n.generators.at[name, "p_nom_min"]        = existing
            n.generators.at[name, "p_nom_max"]        = existing + tcfg.get("p_nom_max_mw", 50000)
            n.generators.at[name, "capital_cost"]     = ann * n_years if capital_in_objective else 0.0
            touched.append(name)
            cc = f"annual.kap {ann/1e3:.0f} €/kW/år" if capital_in_objective else "kapital=0 (sunk)"
            print(f"  → expanderbar: {name} (existing {existing:.0f} MW, "
                  f"overnight {tcfg['overnight_eur_per_w']:.2f} €/W, {cc})")
    return touched


def make_link_extendable(n, cfg: dict, link_name: str, overnight_eur_per_w: float,
                         n_years: float, p_nom_min=None, p_nom_max: float = 30000.0,
                         lifetime: int = 40):
    """Gör en intern NTC-länk kapacitetsexpanderbar med annualiserad overnight-kostnad.

    overnight_eur_per_w: t.ex. 2.0 för 2 M€/MW. Annualiseras som övriga tekniker
    (CRF(lifetime, r) + fom) × n_years och debiteras på p_nom_opt. p_min_pu=-1.0
    behålls → flödesgränsen blir ±p_nom_opt (symmetrisk bidirektionell utbyggnad).
    p_nom_min default = byggd p_nom (golvet). Returnerar (floor, ann €/MW/år)."""
    r   = cfg["costs"]["discount_rate"]
    fom = cfg["costs"]["fom_fraction"]
    if link_name not in n.links.index:
        raise SystemExit(f"--expand-link: länk '{link_name}' finns ej i nätverket")
    built = float(n.links.at[link_name, "p_nom"])
    floor = built if p_nom_min is None else float(p_nom_min)
    ann   = _annualized_cost(overnight_eur_per_w, lifetime, r, fom)   # €/MW/år
    n.links.at[link_name, "p_nom_extendable"] = True
    n.links.at[link_name, "p_nom_min"]        = floor
    n.links.at[link_name, "p_nom_max"]        = p_nom_max
    n.links.at[link_name, "capital_cost"]     = ann * n_years
    print(f"  → expanderbar länk: {link_name} (golv {floor:.0f} MW, tak {p_nom_max:.0f} MW, "
          f"overnight {overnight_eur_per_w:.2f} €/W, annual.kap {ann/1e3:.0f} €/kW/år)")
    return floor, ann


def report_grid_cost(n, cfg: dict, n_years: float) -> None:
    """Post-solve nätkostnads-bokföring (--grid-cost): annualiserad nät-adder ×
    tillförd kapacitet (p_nom_opt − p_nom_min) per kraftslag × zon. Ren
    rapporteringspost ("egen buss" = bokföring), ingår redan i objektivet."""
    ccfg = cfg["costs"]
    r    = ccfg["discount_rate"]
    fom  = ccfg["fom_fraction"]
    rows = []
    comps = [(n.generators, "carrier", "bus"), (n.storage_units, "carrier", "bus")]
    for df, ckey, bkey in comps:
        ext = df[df.p_nom_extendable] if "p_nom_extendable" in df else df.iloc[0:0]
        for name, row in ext.iterrows():
            ann_mw = _grid_capital_cost(row[ckey], row[bkey], ccfg, r, fom, n_years)
            if ann_mw <= 0:
                continue
            added = float(row.p_nom_opt - row.p_nom_min)
            if added <= 1e-6:
                continue
            rows.append((row[bkey], row[ckey], added, ann_mw * added))
    print("\n=== NÄTKOSTNAD (kapital-adder, ⚠️ PLATSHÅLLAR-siffror) ===")
    if not rows:
        print("  (ingen tillförd extendable-kapacitet med nät-adder)")
        return
    rows.sort(key=lambda x: -x[3])
    tot = 0.0
    for zone, carrier, added, cost in rows:
        tot += cost
        print(f"  {zone:5s} {carrier:13s}  +{added/1e3:7.2f} GW  "
              f"nätkostnad {cost/1e6:8.1f} M€/{n_years:.1f}år")
    print(f"  {'TOTALT':5s} {'':13s}  {'':10s}  nätkostnad {tot/1e6:8.1f} M€/{n_years:.1f}år")


def boost_onshore_capfac(profiles: "pd.DataFrame", increase: float) -> "pd.DataFrame":
    """Höjer landbaserad vinds kapacitetsfaktor med en relativ andel `increase`
    (0.1 = +10%) via en olinjär potens-transform per zon:

        x   = cf / cf_max            (cf_max = profilens max per zon, ~0.8 pga
                                      geografisk spridning — INTE installerad effekt)
        cf' = cf_max · x^γ,  γ < 1

    Konkav (γ<1) → lyfter låga/mellan effektnivåer mest (relativt; "mer effektiv
    vid lätta vindar"), fixerar bägge ändar: cf=0→0 (vindstilla) och cf=cf_max→cf_max
    (märkeffekt oförändrad). cf' överskrider aldrig cf_max (geografisk envelopp).

    γ löses per zon med bisektion (mean är monotont avtagande i γ) så att
    mean(cf') = (1+increase)·mean(cf). Endast `*_wind_onshore`-kolumner berörs;
    offshore och sol lämnas orörda.
    """
    if increase <= 0:
        return profiles
    out = profiles.copy()
    print(f"Höjer landbaserad vind-CF med {increase*100:.0f}% (olinjär potens-transform):")
    for col in [c for c in profiles.columns if c.endswith("_wind_onshore")]:
        cf     = profiles[col].astype(float)
        cf_max = float(cf.max())
        if cf_max <= 0:
            continue
        x      = (cf / cf_max).clip(0.0, 1.0)
        mean0  = float(cf.mean())
        target = mean0 * (1.0 + increase)
        # Tak: γ→0 ger cf'→cf_max för alla cf>0
        max_mean = cf_max * float((cf > 0).mean())
        zone = col.replace("_wind_onshore", "")
        if target > max_mean:
            print(f"  Varning: {zone} — mål {target:.3f} > tak {max_mean:.3f}; klampar till taket")
            target = max_mean
        # Bisektion i γ ∈ (eps, 1]; mean(γ) avtagande → m>target ⇒ höj γ
        lo, hi = 1e-4, 1.0
        g = 1.0
        for _ in range(60):
            g = 0.5 * (lo + hi)
            m = float((cf_max * x.pow(g)).mean())
            if m > target:
                lo = g
            else:
                hi = g
        cf1 = cf_max * x.pow(g)
        out[col] = cf1
        print(f"  {zone:6s} CF {mean0:.3f}→{float(cf1.mean()):.3f} "
              f"(+{100*(cf1.mean()/mean0-1):.1f}%), γ={g:.3f}, cf_max={cf_max:.3f} (oförändrad)")
    return out


def boost_offshore_capfac(profiles: "pd.DataFrame", increase: float) -> "pd.DataFrame":
    """Som boost_onshore_capfac men för `*_wind_offshore`-kolumner. Höjer havsbaserad
    vinds fleet-CF med relativ andel `increase` via samma olinjära potens-transform
    (cf' = cf_max·(cf/cf_max)^γ, mean(cf')=(1+increase)·mean(cf)). Speglar att 2040:s
    nybyggnadsflotta (moderna 15 MW-turbiner, bästa sitsar) slår dagens kalibrering."""
    if increase <= 0:
        return profiles
    out = profiles.copy()
    print(f"Höjer havsbaserad vind-CF med {increase*100:.0f}% (olinjär potens-transform):")
    for col in [c for c in profiles.columns if c.endswith("_wind_offshore")]:
        cf     = profiles[col].astype(float)
        cf_max = float(cf.max())
        if cf_max <= 0:
            continue
        x      = (cf / cf_max).clip(0.0, 1.0)
        mean0  = float(cf.mean())
        target = mean0 * (1.0 + increase)
        max_mean = cf_max * float((cf > 0).mean())
        zone = col.replace("_wind_offshore", "")
        if target > max_mean:
            print(f"  Varning: {zone} — mål {target:.3f} > tak {max_mean:.3f}; klampar till taket")
            target = max_mean
        lo, hi, g = 1e-4, 1.0, 1.0
        for _ in range(60):
            g = 0.5 * (lo + hi)
            m = float((cf_max * x.pow(g)).mean())
            if m > target:
                lo = g
            else:
                hi = g
        cf1 = cf_max * x.pow(g)
        out[col] = cf1
        print(f"  {zone:6s} CF {mean0:.3f}→{float(cf1.mean()):.3f} "
              f"(+{100*(cf1.mean()/mean0-1):.1f}%), γ={g:.3f}, cf_max={cf_max:.3f} (oförändrad)")
    return out


def load_inputs(cfg: dict) -> dict:
    """Laddar alla förberedda indata från data/processed/."""
    load_df       = pd.read_parquet(PROC_DIR / "load.parquet")
    vre           = pd.read_parquet(PROC_DIR / "vre_profiles.parquet")
    nuclear       = pd.read_parquet(PROC_DIR / "nuclear_profile.parquet")
    thermal       = pd.read_parquet(PROC_DIR / "thermal_profile.parquet")
    prices_df     = pd.read_parquet(PROC_DIR / "market_prices.parquet")

    with open(PROC_DIR / "vre_pnom.yaml") as f:
        vre_noms = yaml.safe_load(f)
    with open(CONFIG_PATH.parent / "hydro_params.yaml") as f:
        hydro_params = yaml.safe_load(f)

    # Fjärrvärme-värmebehov (valfritt; gitignorerad, byggs av scripts/build_heat.py)
    heat_path = PROC_DIR / "heat_load.parquet"
    heat_load = pd.read_parquet(heat_path) if heat_path.exists() else None

    # EV-laddningsprofiler (valfritt; byggs av build_ev_profiles i build_inputs.py)
    ev_path     = PROC_DIR / "ev_profiles.parquet"
    ev_profiles = pd.read_parquet(ev_path) if ev_path.exists() else None

    # Sätt UTC-index och ta bort timezone (PyPSA kräver tz-naivt)
    dfs = [load_df, vre, nuclear, thermal, prices_df]
    if heat_load is not None:
        dfs.append(heat_load)
    if ev_profiles is not None:
        dfs.append(ev_profiles)
    for df in dfs:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    market_prices = {col: prices_df[col] for col in prices_df.columns}

    return dict(
        load=load_df, vre_profiles=vre, vre_noms=vre_noms,
        nuclear_profile=nuclear, thermal_profile=thermal,
        hydro_params=hydro_params, market_prices=market_prices,
        heat_load=heat_load, ev_profiles=ev_profiles,
    )


def apply_cost_scenario(cfg: dict, name: str) -> None:
    """Skriver över cfg['costs'] med ett kostnadsscenario ur cost_scenarios.

    Byggränta (IDC, ränta under byggtid): overnight' = OC·(1 + build_years/2·r).
    fom_fraction sätts = fast_DoU/overnight' så att absolut O&M-nivå bevaras (ej
    uppblåst av IDC, eftersom modellen beräknar annualiserat = overnight·(CRF+fom)).
    """
    scenarios = cfg.get("cost_scenarios", {})
    scen = scenarios.get(name)
    if scen is None:
        raise SystemExit(f"Okänt --cost-scenario '{name}'. Finns: {list(scenarios)}")
    r = cfg["costs"]["discount_rate"]
    print(f"  → kostnadsscenario '{name}' (IDC med r={r}):")
    for tech, p in scen.items():
        if not isinstance(p, dict):
            continue
        tgt = cfg["costs"].setdefault(tech, {})
        idc = 1.0 + p["build_years"] / 2 * r
        tgt["lifetime_years"] = p["lifetime_years"]
        if tech == "battery":
            tgt["power_eur_per_kw"]   = p["power_eur_per_kw"]   * idc
            tgt["energy_eur_per_kwh"] = p["energy_eur_per_kwh"] * idc
            tgt["fom_fraction"]       = p.get("fom_fraction", 0.025)
            print(f"     {tech:<14} IDC×{idc:.3f}  power {tgt['power_eur_per_kw']:.0f} €/kW + "
                  f"energy {tgt['energy_eur_per_kwh']:.0f} €/kWh")
        else:
            oc_kw = p["oc_eur_per_kw"] * idc                       # OC' [EUR/kW]
            tgt["overnight_eur_per_w"] = oc_kw / 1000.0            # EUR/W
            tgt["fom_fraction"]        = p["fom_eur_per_kw"] / oc_kw
            tgt["vom_eur_per_mwh"]     = p["vom_eur_per_mwh"]
            print(f"     {tech:<14} IDC×{idc:.3f}  overnight {oc_kw/1000:.3f} €/W  "
                  f"fom {tgt['fom_fraction']*100:.2f}%  vom {p['vom_eur_per_mwh']}  L{p['lifetime_years']}")


def apply_demand_scenario(cfg: dict, name: str,
                          hydrogen_overrides: dict, ev_overrides: dict,
                          batteries: list, scenario_battery: tuple | None = None) -> dict:
    """Adderar ett efterfrågescenario (demand_scenarios i zones.yaml) ovanpå eSett-basen.

    Muterar in-place: cfg['additional_load_mw'] (per zon), hydrogen_overrides,
    ev_overrides och batteries. Applicerar ntc_overrides på cfg['links'] och
    (om nuclear_exogenous) låser kärnkraften (costs.nuclear.extendable=False) till
    nivåerna i per-zon-nuclear-block. Returnerar (pnom_max, pnom_min): två dictar
    {(zon, carrier): MW} för VRE-utbyggnadstak resp. exogena golv (pnom_min_mw, t.ex.
    SE-basflotta), båda applicerade EFTER build_network.

    Princip: additiv (eSett-bas bevaras), se zones.yaml-kommentar. CLI-givna H2/EV
    (--add-h2/--add-ev) har företräde och skrivs INTE över.
    """
    scenarios = cfg.get("demand_scenarios", {})
    scen = scenarios.get(name)
    if scen is None:
        raise SystemExit(f"Okänt --demand-scenario '{name}'. Finns: {list(scenarios)}")

    print(f"  → efterfrågescenario '{name}' (additivt över eSett-bas):")
    if scen.get("nuclear_exogenous"):
        cfg["costs"]["nuclear"]["extendable"] = False
        print("     kärnkraft EXOGEN (costs.nuclear.extendable=False — ingen endogen expansion)")
    pnom_max: dict = {}
    pnom_min: dict = {}
    scen_bats: list = []          # (zon, p_nom_mw, hours) — appendas (ev. omskalat) efter loopen
    cfg.setdefault("additional_load_mw", {})
    for zone, zc in scen.get("zones", {}).items():
        extra = float(zc.get("extra_load_mw", 0.0))
        cfg["additional_load_mw"][zone] = cfg["additional_load_mw"].get(zone, 0.0) + extra

        h2 = zc.get("h2")
        if h2 and zone not in hydrogen_overrides:           # CLI har företräde
            # Elektrolysör ENDOGEN: kapaciteten optimeras (SvK-GW = startvärde, ej golv).
            # H2-last + lager förblir exogent fasta (SvK-plan). p_nom_max valfritt i config.
            hydrogen_overrides[zone] = {
                "demand_mw":    float(h2["demand_mw"]),
                "electrolyser": {"p_nom_mw": float(h2["electrolyser_mw"]), "extendable": True,
                                 "p_nom_max_mw": float(h2.get("electrolyser_max_mw", 50000.0))},
                "store":        {"e_nom_mwh": float(h2["store_mwh"]),       "extendable": False},
            }

        cars = float(zc.get("ev_cars", 0.0))
        if cars > 0 and zone not in ev_overrides:
            ev_overrides[zone] = {"car": cars, "heavy": 0.0}

        for carrier, pmax in (zc.get("pnom_max_mw") or {}).items():
            pnom_max[(zone, carrier)] = float(pmax)

        for carrier, pmin in (zc.get("pnom_min_mw") or {}).items():
            pnom_min[(zone, carrier)] = float(pmin)

        bat = zc.get("battery")
        bat_txt = ""
        if bat:                                              # exogent fast batteri (SvK storskaligt)
            scen_bats.append((zone, float(bat["p_nom_mw"]), float(bat.get("hours", 2))))
            if scenario_battery is None:                     # annars visas override-summa efter loopen
                bat_txt = f", batteri {bat['p_nom_mw']:.0f} MW/{bat.get('hours', 2)}h"

        nuc = zc.get("nuclear")
        nuc_txt = ""
        if nuc:                                              # befintlig kärnkraftsnivå i zonen (override)
            cfg["zones"][zone]["nuclear_p_nom_mw"] = float(nuc["p_nom_mw"])
            nuc_txt = f", kärnkr {nuc['p_nom_mw']:.0f} MW (befintlig)"

        h2dem = (h2 or {}).get("demand_mw", 0)
        print(f"     {zone:<5} +{extra:6.0f} MW last, H2 {h2dem:.0f} MW, "
              f"EV {cars/1e6:.1f}M bilar, tak {{{', '.join(f'{c}:{int(p)}' for (z,c),p in pnom_max.items() if z==zone)}}}"
              f"{nuc_txt}{bat_txt}")

    # Exogena scenariobatterier: as-is, eller omskalade till total GW @ HOURS (--scenario-battery)
    if scen_bats:
        if scenario_battery is not None:
            gw, hours = scenario_battery
            tot = sum(p for _, p, _ in scen_bats)
            scale = (gw * 1e3) / tot if tot > 0 else 0.0
            for z, p, _h in scen_bats:
                batteries.append((z, p * scale, hours, False))
            zsum = ", ".join(f"{z} {p*scale/1e3:.2f}" for z, p, _h in scen_bats)
            print(f"     batteri-OVERRIDE: {tot/1e3:.1f} GW → {gw:.1f} GW @ {hours:.0f}h "
                  f"(samma zonandel: {zsum} GW)")
        else:
            for z, p, h in scen_bats:
                batteries.append((z, p, h, False))

    for z0, z1, mw in scen.get("ntc_overrides", []):
        for link in cfg.get("links", []):
            if link[0] == z0 and link[1] == z1 and link[2] != mw:
                print(f"     NTC {z0}-{z1}: {link[2]} → {mw} MW (Tabell 10)")
                link[2] = mw

    # Kontinentkablar: matchas på connection-namn (market_connections = [namn, zon, mw, bzn])
    for name, mw in scen.get("market_ntc_overrides", []):
        for mc in cfg.get("market_connections", []):
            if mc[0] == name and mc[2] != mw:
                print(f"     Marknads-NTC {name}: {mc[2]} → {mw} MW (Tabell 10)")
                mc[2] = mw
    return pnom_max, pnom_min


def make_snapshots(cfg: dict, resolution: int, year: int | None) -> pd.DatetimeIndex:
    """Genererar snapshot-index utifrån konfiguration och CLI-parametrar."""
    start = pd.Timestamp(cfg["snapshots"]["start"], tz="UTC")
    end   = pd.Timestamp(cfg["snapshots"]["end"],   tz="UTC") - pd.Timedelta(hours=1)

    if year is not None:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end   = pd.Timestamp(f"{year}-12-31 23:00", tz="UTC")

    freq = f"{resolution}h"
    idx  = pd.date_range(start, end, freq=freq)
    return idx.tz_localize(None)  # PyPSA kräver timezone-naiva snapshots


def resample_inputs(inputs: dict, snapshots: pd.DatetimeIndex, resolution: int) -> dict:
    """Resamplar alla tidsserier till snapshot-frekvensen (medelvärde)."""
    freq = f"{resolution}h"
    out  = {}
    for key in ("load", "vre_profiles", "nuclear_profile", "thermal_profile"):
        out[key] = inputs[key].resample(freq).mean().reindex(snapshots).ffill()
    out["market_prices"] = {
        bzn: s.resample(freq).mean().reindex(snapshots).ffill()
        for bzn, s in inputs["market_prices"].items()
    }
    out["vre_noms"]     = inputs["vre_noms"]
    out["hydro_params"] = inputs["hydro_params"]
    hl = inputs.get("heat_load")
    out["heat_load"] = (hl.resample(freq).mean().reindex(snapshots).ffill()
                        if hl is not None else None)
    ev = inputs.get("ev_profiles")
    out["ev_profiles"] = (ev.resample(freq).mean().reindex(snapshots).ffill()
                          if ev is not None else None)
    return out


# ---------------------------------------------------------------------------
# Lösning och sparning
# ---------------------------------------------------------------------------

def solve(n, cfg: dict, log_path: Path | None = None,
          soc_pin_end: dict | None = None,
          extra_callbacks: list | None = None) -> bool:
    scfg    = cfg["solver"]
    solver  = scfg["name"]
    options = {k: v for k, v in scfg.items() if k != "name"}

    if log_path is not None:
        options["log_file"] = str(log_path)

    if soc_pin_end:
        # Icke-cyklisk: start = soc_initial_override (faktisk), slut pinnas till faktisk fraktion.
        callbacks = [hydro_soc_terminal_pin_constraint(cfg, soc_pin_end)]
        print("  → SOC-pin (icke-cyklisk, faktiska ändpunkter): "
              + ", ".join(f"{z} slut {f*100:.1f}%" for z, f in soc_pin_end.items()))
    else:
        callbacks = [hydro_soc_initial_constraint(cfg)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    def extra_func(n, snapshots):
        for cb in callbacks:
            cb(n, snapshots)

    print(f"Löser med {solver} ({len(n.snapshots)} tidssteg, "
          f"{len(n.generators) + len(n.storage_units)} generatorer) ...")
    if log_path:
        print(f"  HiGHS-logg: {log_path}")

    status, condition = n.optimize(
        solver_name=solver,
        solver_options=options,
        extra_functionality=extra_func,
        assign_all_duals=True,   # behövs för att få vattenvärdet (mu_energy_balance)
    )

    print(f"  Status: {status} / {condition}")
    return status == "ok"


def extract_results(n) -> dict:
    """Alla resultat-tidsserier ur ett löst nätverk → {namn: DataFrame}.

    EN sanningskälla för BÅDA körvägarna (standard + rullande horisont): lägg till
    en rad här så dyker resultatet upp i båda automatiskt. Nyckel = filnamn (utan
    .csv). None/tomma DataFrames hoppas tyst över vid sparning.

    Obs: hydro_soc/dispatch_hydro innehåller ALLA storage units (även batteri).
    water_value (dual på lagringsbalansen) kräver assign_all_duals=True i optimize.
    """
    out = {
        "dispatch_generators": n.generators_t.p,
        "dispatch_hydro":      n.storage_units_t.p,
        "hydro_soc":           n.storage_units_t.state_of_charge,          # inkl. batteri-SOC
        "hydro_spill":         n.storage_units_t.spill,
        "flows":               n.links_t.p0,
        "prices":              n.buses_t.marginal_price,
        "water_value":         n.storage_units_t.get("mu_energy_balance"),  # dual = vattenvärde
    }
    if len(n.stores) > 0:
        out["h2_store_soc"] = n.stores_t.e
    return out


def save_results_dict(results: dict, label: str) -> None:
    """Sparar {namn: DataFrame} som platta CSV:er (namn.csv). Hoppar över None/tomma.
    Delas av standard- och rullande-vägen."""
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if df is not None and getattr(df, "shape", (0, 0))[1] > 0:
            df.to_csv(out / f"{name}.csv")
    print(f"  → resultat sparade i {out}/")


def save_results(n, label: str) -> None:
    """Standard-vägen: nätverk (.nc) + alla CSV:er via extract_results-registryt."""
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(out / "network.nc")
    save_results_dict(extract_results(n), label)


# ---------------------------------------------------------------------------
# Tvåpass: fönstervis dispatch med hydro-SOC pinnad mot en källkörnings lagerbana
# ---------------------------------------------------------------------------

def load_soc_levels(label: str) -> pd.DataFrame:
    """results/LABEL/hydro_soc.csv → lagernivåer (MWh) indexerade på VÄGGKLOCKAN.

    PyPSA:s soc[t] är nivån vid SLUTET av tidssteget som börjar i t, alltså nivån vid
    väggklockan t+dt. Indexet skiftas därför fram ett källtidssteg, så att linjär
    interpolation i tid ger rätt nivå vid en godtycklig fönstergräns (källan kan ha
    annan upplösning än dispatchkörningen).
    """
    path = RESULTS_DIR / label / "hydro_soc.csv"
    if not path.exists():
        raise SystemExit(f"--soc-pin-from: hittar inte {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if len(df) < 2:
        raise SystemExit(f"--soc-pin-from: {path} har färre än 2 rader")
    df.index = df.index + (df.index[1] - df.index[0])
    return df


def soc_levels_at(levels: pd.DataFrame, t: pd.Timestamp, units: list) -> dict:
    """Lagernivå (MWh) för givna enheter vid väggklockan t, linjärt interpolerad."""
    if t not in levels.index:
        levels = levels.reindex(levels.index.union([t])).interpolate(method="time")
    row = levels.ffill().bfill().loc[t]
    return {u: float(row[u]) for u in units}


def pin_windows(snapshots: pd.DatetimeIndex, freq: str) -> list:
    """Delar snapshots i sammanhängande fönster per kalenderperiod (MS=månad, W=vecka)."""
    s = pd.Series(0, index=snapshots)
    return [g.index for _, g in s.groupby(pd.Grouper(freq=freq)) if len(g)]


def solve_soc_pinned(n, cfg: dict, src_label: str, freq: str,
                     band_frac: float = 0.0,
                     log_path: Path | None = None,
                     extra_callbacks: list | None = None) -> tuple[bool, dict | None]:
    """Tvåpass-dispatch: lös perioden fönster för fönster (freq) med hydro-SOC pinnad
    i BÅDA ändar till källkörningens (src_label) lagerbana.

    Ersätter cyklisk SOC + perfekt framsyn över hela perioden med källans säsongsbana
    (exogen) + fritt optimerad dispatch inom fönstret. Säsongssignalen ärvs alltså från
    källan; det fönstret tillför är intra-fönster-dynamik på finare upplösning.

    Övriga lager (batteri, EV/värme-Stores) förblir cykliska PER FÖNSTER — ofarligt så
    länge fönstret är långt mot deras cykeltid (månad ≫ 4h batteri / dygns-EV).
    """
    scfg    = cfg["solver"]
    solver  = scfg["name"]
    options = {k: v for k, v in scfg.items() if k != "name"}
    if log_path is not None:
        options["log_file"] = str(log_path)

    levels = load_soc_levels(src_label)
    units  = [u for u in n.storage_units.index
              if n.storage_units.at[u, "carrier"] == "hydro" and u in levels.columns]
    if not units:
        raise SystemExit(f"--soc-pin-from: inga hydrolager i {src_label}/hydro_soc.csv "
                         f"matchar nätverkets storage units")
    missing = [u for u in n.storage_units.index
               if n.storage_units.at[u, "carrier"] == "hydro" and u not in levels.columns]
    n.storage_units.loc[units, "cyclic_state_of_charge"] = False
    cap = {u: float(n.storage_units.at[u, "p_nom"]) * float(n.storage_units.at[u, "max_hours"])
           for u in units}

    band = {u: max(band_frac, 0.0) * cap[u] for u in units}
    windows = pin_windows(n.snapshots, freq)
    dt = n.snapshots[1] - n.snapshots[0]
    print(f"  → SOC-pin från {src_label}: {len(windows)} fönster ({freq}), "
          f"{len(units)} hydrolager pinnade i båda ändar ({', '.join(units)}); "
          + (f"band ±{band_frac:.1%} av kapaciteten (start = FÖREGÅENDE FÖNSTERS UPPNÅDDA nivå)"
             if band_frac > 0 else "HÅRD likhet (band 0)")
          + (f"  ⚠️ {len(missing)} utan källdata, förblir cykliska: {missing}" if missing else ""))
    if log_path is not None:
        print(f"  HiGHS-logg: {log_path} (skrivs över per fönster — sista fönstret kvarstår)")

    parts, keys, achieved = [], [], None
    for i, sns in enumerate(windows, 1):
        # Med band gäller inte achieved == target, så nästa fönster startar på den
        # UPPNÅDDA nivån. Målet hämtas fortfarande ur källan → avvikelsen kan inte
        # ackumulera (varje fönster re-ankras mot källbanan), den är bunden av bandet.
        init = achieved if achieved is not None else soc_levels_at(levels, sns[0], units)
        term = soc_levels_at(levels, sns[-1] + dt, units)
        # Skyddsklipp mot [0, cap]. Ska normalt inte bita — om det gör det avviker
        # dispatchnätets reservoarvolym från källans och pinnen blir inte längre källans bana.
        for tag, d in (("start", init), ("slut", term)):
            for u, v in d.items():
                c = min(max(v, 0.0), cap[u])
                if abs(c - v) > 1e-4 * cap[u]:
                    print(f"  ⚠️ fönster {i} {tag}-pin {u} klippt {v/1e6:.3f} → {c/1e6:.3f} TWh "
                          f"(volym {cap[u]/1e6:.3f} TWh ≠ källans) — pinnen följer EJ källbanan")
                d[u] = c
        for u in units:
            n.storage_units.at[u, "state_of_charge_initial"] = init[u]

        callbacks = [soc_terminal_pin_mwh(term, band)] + list(extra_callbacks or [])

        def extra_func(nn, snapshots, _cbs=callbacks):
            for cb in _cbs:
                cb(nn, snapshots)

        status, condition = n.optimize(
            snapshots=sns,
            solver_name=solver,
            solver_options=options,
            extra_functionality=extra_func,
            assign_all_duals=True,
        )
        span = " ".join(f"{u.split()[0]} {init[u]/cap[u]:.0%}→{term[u]/cap[u]:.0%}" for u in units)
        dev = ""
        if status == "ok":
            achieved = {u: float(n.storage_units_t.state_of_charge.at[sns[-1], u]) for u in units}
            worst = max(units, key=lambda u: abs(achieved[u] - term[u]) / cap[u])
            d = (achieved[worst] - term[worst]) / cap[worst]
            dev = f"   maxavvik {worst.split()[0]} {d:+.2%}"
        print(f"  fönster {i:3d}/{len(windows)} {sns[0]:%Y-%m-%d}–{sns[-1]:%Y-%m-%d} "
              f"({len(sns)} steg): {status}/{condition}   {span}{dev}")
        if status != "ok":
            print(f"  ✖ fönster {i} misslyckades ({status}/{condition}) — avbryter. "
                  f"Vanligaste orsaken: pinnad ΔSOC ej nåbar på denna upplösning.")
            return False, None

        part = {k: v.loc[sns] for k, v in extract_results(n).items()
                if v is not None and getattr(v, "shape", (0, 0))[1] > 0}
        for k in part:
            if k not in keys:
                keys.append(k)
        parts.append(part)

    results = {k: pd.concat([p[k] for p in parts if k in p]).sort_index() for k in keys}
    return True, results


# ---------------------------------------------------------------------------
# Rullande horisont: sekventiella fönster + terminalvärde på slut-SOC
# ---------------------------------------------------------------------------

def rolling_windows(snapshots: pd.DatetimeIndex, window_steps: int,
                    lookahead_steps: int = 0):
    """Delar snapshots i sekventiella fönster och ger (BEHÅLL, LÖS) per fönster.

    lookahead_steps=0 → klassiskt icke-överlappande: LÖS == BEHÅLL, varje fönster
    ser noll framåt och hela säsongssignalen måste bäras av terminalvärdet λ.

    lookahead_steps>0 → ÄKTA RECEDING HORIZON (väg E i docs/vattenvarde_plan.md):
    fönstret LÖSES med extra look-ahead men bara den första delen BEHÅLLS, och
    SOC bärs över från slutet av BEHÅLL-delen. Poängen är att look-ahead gör det
    mesta av jobbet i stället för terminalkurvan, vilket i sin tur krymper
    cirkularitetsproblemet (λ kalibreras mot observerade priser).

    Sista fönstret får ingen look-ahead att hämta — det finns inget efter
    periodens slut — så där sammanfaller LÖS och BEHÅLL igen.
    """
    n = len(snapshots)
    for s in range(0, n, window_steps):
        keep = snapshots[s: s + window_steps]
        solve = snapshots[s: min(s + window_steps + lookahead_steps, n)]
        yield keep, solve


def terminal_lambdas(args, cfg: dict, market_prices: dict, units: list,
                     t_last: pd.Timestamp, lookahead_steps: int) -> dict:
    """λ per hydrolager (EUR/MWh) för terminalvärdet −λ×SOC[T].

    Default: framåtriktat medelvärde av DE-LU över NÄSTA fönster — alltså
    alternativkostnaden för att exportera vattnet till kontinenten. Det är exogen
    indata (modellen prissätter redan marknadsventilen med den), så till skillnad
    från zonens EGET observerade pris är det inte cirkulärt.

    ⚠️ DE-LU övervärderar ändå vattnet i trängselinlåsta norra zoner, vars pris
    inte kan följa kontinenten. Använd --terminal-lambda ZON:VÄRDE för att sätta
    per zon manuellt.
    """
    # λ_zon(t) = α_zon × DE-LU_framåt(t): behåll kontinentprisets TIDSFORM, skala NIVÅN
    # per zon. Ett KONSTANT λ per zon fungerar inte — run271 tömde SE-N till 9 % och
    # NO-N till 0 % i februari 2024 och blev infeasible, eftersom vattnets
    # alternativkostnad är hög före vårfloden och låg efter, inte lika året runt.
    # --terminal-seasonal: byt λ:s TIDSFORM från DE-LU:s framåtpris till en handsatt
    # säsongstabell. Motivet är cirkularitet — default hämtar formen ur en prisserie,
    # alltså ur det modellen ska förutsäga. Tabellen är ett antagande i stället, vilket
    # är sämre grundat men inte avläst ur måldata. ⚠️ NIVÅN (base_level_eur_mwh) är
    # fortfarande zonens observerade medelpris, så bara halva cirkulariteten försvinner.
    tvcfg = cfg.get("terminal_value", {}) or {}
    if getattr(args, "terminal_seasonal", False):
        sf = tvcfg.get("seasonal_factors") or {}
        if not sf:
            raise SystemExit("--terminal-seasonal: terminal_value.seasonal_factors "
                             "saknas i config/zones.yaml")
        base = tvcfg.get("base_level_eur_mwh") or {}
        if not base:
            raise SystemExit("--terminal-seasonal: terminal_value.base_level_eur_mwh "
                             "saknas i config/zones.yaml")
        s = float(sf[int(t_last.month)])
        if tvcfg.get("normalize_seasonal", True):
            # Håll årsnivån oförändrad så att A/B-testet isolerar FORMEN.
            s /= (sum(float(v) for v in sf.values()) / len(sf))
        # Skalflaggan får fortfarande verka, så känslighetstestet fungerar likadant.
        pert = None
        if args.terminal_lambda_scale:
            nominal = tvcfg.get("lambda_scale") or {}
            given = {z.strip(): float(v) for z, v in
                     (pair.split(":") for pair in args.terminal_lambda_scale.split(","))}
            pert = {z: given[z] / nominal[z] for z in given if nominal.get(z)}
        return {u: float(base.get(u.split()[0], 0.0)) * s
                   * (pert.get(u.split()[0], 1.0) if pert else 1.0)
                for u in units}

    # Flaggan åsidosätter config-värdena (terminal_value.lambda_scale).
    alpha = None
    if args.terminal_lambda_scale:
        alpha = {z.strip(): float(v) for z, v in
                 (pair.split(":") for pair in args.terminal_lambda_scale.split(","))}
    elif not args.terminal_lambda:
        alpha = (cfg.get("terminal_value", {}) or {}).get("lambda_scale") or None
    if alpha is not None:
        de = market_prices.get("DE-LU")
        if de is None:
            base = 60.0
        else:
            idx   = de.index.get_indexer([t_last], method="nearest")[0]
            ahead = de.iloc[idx: idx + lookahead_steps]
            base  = float(ahead.mean()) if len(ahead) else float(de.mean())
        return {u: base * alpha.get(u.split()[0], 1.0) for u in units}

    if args.terminal_lambda:
        if ":" in args.terminal_lambda:
            per_zone = {z.strip(): float(v) for z, v in
                        (pair.split(":") for pair in args.terminal_lambda.split(","))}
        else:
            per_zone = {u.split()[0]: float(args.terminal_lambda) for u in units}
        return {u: per_zone.get(u.split()[0], 0.0) for u in units}

    de = market_prices.get("DE-LU")
    if de is None:
        lam = 60.0
    else:
        idx   = de.index.get_indexer([t_last], method="nearest")[0]
        ahead = de.iloc[idx: idx + lookahead_steps]
        lam   = float(ahead.mean()) if len(ahead) else float(de.mean())
    return {u: lam for u in units}


def solve_rolling_horizon(n, cfg: dict, args, market_prices: dict, res: int,
                          log_path: Path | None = None,
                          extra_callbacks: list | None = None) -> tuple[bool, dict | None]:
    """Rullande horisont: lös perioden fönster för fönster med icke-cyklisk SOC,
    carry-over av slut-SOC, och ett terminalvärde −λ×SOC[T] per fönster.

    Syftet är att bryta den perfekta framsynen över hela perioden, som gör det
    endogena vattenvärdet nästan konstant (1–6 unika värden per zon över tre år).

    Fönstren är som default icke-överlappande, så varje fönster ser NOLL framåt och
    hela säsongssignalen måste bäras av λ. `--rolling-lookahead-weeks N` ger äkta
    receding horizon: fönstret löses med N veckors extra look-ahead men bara första
    delen behålls, och SOC bärs över från behåll-delens slut.
    """
    scfg    = cfg["solver"]
    options = {k: v for k, v in scfg.items() if k != "name"}
    if log_path is not None:
        options["log_file"] = str(log_path)

    units = [u for u in n.storage_units.index
             if n.storage_units.at[u, "carrier"] == "hydro"]
    if not units:
        raise SystemExit("--rolling-horizon: inga hydrolager i nätverket")
    n.storage_units.loc[units, "cyclic_state_of_charge"] = False

    cap = {u: float(n.storage_units.at[u, "p_nom"]) * float(n.storage_units.at[u, "max_hours"])
           for u in units}
    # Start-SOC. I rullande horisont finns INGET cykliskt villkor — nivån bärs över från
    # fönster till fönster i hela perioden, så startvärdet är ett äkta begynnelsevillkor
    # som propagerar i stället för att tvättas bort. Använd därför den FAKTISKA nivån
    # (hydro_soc_start, ur Energy Charts) när den finns; hydro_soc_initial är cykliska
    # körningars ankare för start = slut och är en annan sak.
    start_cfg = cfg.get("hydro_soc_start", {}) or {}
    soc_carry, start_src = {}, []
    for u in units:
        zone = u.split()[0]
        if zone in start_cfg:
            frac = float(start_cfg[zone]); src = "faktisk"
        else:
            frac = cfg["zones"].get(zone, {}).get("hydro_soc_initial", 0.5); src = "cyklisk-ankare"
        soc_carry[u] = frac * cap[u]
        start_src.append(f"{zone} {frac:.0%}({src})")
    print("  → start-SOC: " + ", ".join(start_src))

    # Terminalkurvan: --terminal-lambda-profile åsidosätter ALLA zoner; annars per zon ur
    # config (terminal_value.profiles), med default_profile för zoner som saknas.
    tvcfg = cfg.get("terminal_value", {}) or {}
    if args.terminal_lambda_profile:
        profile = [float(x) for x in args.terminal_lambda_profile.split(",") if x.strip()]
    else:
        dflt    = list(tvcfg.get("default_profile") or DEFAULT_TERMINAL_PROFILE)
        byzone  = tvcfg.get("profiles") or {}
        profile = {u: list(byzone.get(u.split()[0], dflt)) for u in units}

    # --terminal-curve: analytisk λ_k(vecka, zon) ersätter BÅDE λ och profilen och
    # räknas om PER FÖNSTER, eftersom varje fönster slutar i en annan vecka. Utan den
    # är profilen årskonstant och hela säsongssignalen måste bäras av λ ensamt.
    curve = None
    if args.terminal_curve is not None:
        from nordpsa.wv import terminal_curve as tc
        cparams, canchor = tc.load_params(args.terminal_curve or None)
        # Kurvan genererar sina EGNA segmentvärden, så av `profile` används bara
        # LÄNGDEN. Den kopplingen var en artefakt: antalet segment styrdes av en
        # profil vars värden ändå kastades. --terminal-segments gör det explicit.
        # En uttrycklig --terminal-lambda-profile vinner ändå, så gamla kommandon
        # (run340 kördes med en 20-värdesprofil) reproduceras exakt.
        segments = (len(profile) if args.terminal_lambda_profile
                    else args.terminal_segments)
        curve = (tc, cparams, canchor, segments)
        print(f"  → TERMINALKURVA λ_k(vecka, zon), {segments} segment. λ_bas: "
              + ", ".join(f"{z} {v:.1f}" for z, v in sorted(canchor.items())))
        for z in sorted(cparams):
            q = cparams[z]
            print(f"       {z:6s} a_amp={q.a_amp:.2f} a_peak=v{q.a_peak:.0f} "
                  f"b_mean={q.b_mean:.2f} b_amp={q.b_amp:.2f} b_peak=v{q.b_peak:.0f}")

    steps_per_week  = max(1, (7 * 24) // res)
    window_steps    = args.rolling_weeks * steps_per_week
    lookahead_steps = args.rolling_lookahead_weeks * steps_per_week
    windows         = list(rolling_windows(n.snapshots, window_steps, lookahead_steps))
    print(f"  → rullande horisont: {args.rolling_weeks} veckor/fönster "
          f"({window_steps} tidssteg), {len(windows)} fönster, {len(units)} hydrolager")
    if lookahead_steps:
        print(f"  → RECEDING HORIZON: +{args.rolling_lookahead_weeks} veckors look-ahead "
              f"({lookahead_steps} steg) löses men KASTAS; SOC bärs över från behåll-delen. "
              f"Terminalvärdet hamnar {args.rolling_lookahead_weeks} veckor bort och styr "
              f"därmed mindre.")
    if curve is not None:
        pass          # kurvan ersätter profilen helt; dess egen rad är redan utskriven
    elif isinstance(profile, dict):
        K = len(next(iter(profile.values())))
        print(f"  → terminalvärde KONKAVT per zon, {K} segment à {100.0/K:.0f} % av volymen:")
        for u in units:
            print(f"       {u.split()[0]:6} λ×[{', '.join(f'{p:g}' for p in profile[u])}]")
    elif len(profile) == 1:
        print(f"  ⚠️ terminalvärde LINJÄRT (profil {profile[0]:g}) — konstant marginalvärde "
              f"oavsett fyllnadsgrad ger bang-bang: magasin i taket + priskollaps till VOM "
              f"(run268). Använd flersegmentsprofil.")
    else:
        band = 100.0 / len(profile)
        print(f"  → terminalvärde KONKAVT (global), {len(profile)} segment à {band:.0f} %: "
              f"λ×[{', '.join(f'{p:g}' for p in profile)}] (tomt→fullt)")
    if log_path is not None:
        print(f"  HiGHS-logg: {log_path} (skrivs över per fönster — sista kvarstår)")

    parts, keys = [], []
    for i, (keep, sns) in enumerate(windows, 1):
        for u in units:
            n.storage_units.at[u, "state_of_charge_initial"] = soc_carry[u]

        if curve is not None:
            tc, cparams, canchor, segments = curve
            wk   = tc.week_of(sns[-1])
            lam  = tc.lambdas_for_week(wk, units, canchor, cparams)
            prof = tc.profiles_for_week(wk, units, cparams, segments)
        else:
            wk   = None
            lam  = terminal_lambdas(args, cfg, market_prices, units, sns[-1], window_steps)
            prof = profile
        callbacks = [hydro_terminal_value(lam, cap, prof)] + list(extra_callbacks or [])

        def extra_func(nn, snapshots, _cbs=callbacks):
            for cb in _cbs:
                cb(nn, snapshots)

        status, condition = n.optimize(
            snapshots=sns,
            solver_name=scfg["name"],
            solver_options=options,
            extra_functionality=extra_func,
            assign_all_duals=True,
        )
        start_txt = " ".join(f"{u.split()[0]} {soc_carry[u]/cap[u]:.0%}" for u in units)
        if status == "ok":
            # SOC bärs över från slutet av BEHÅLL-delen, inte från look-ahead-svansen:
            # svansen är bara en framtidsbild och kastas.
            soc_carry = {u: float(n.storage_units_t.state_of_charge.at[keep[-1], u])
                         for u in units}
            slut_txt = " ".join(f"→{soc_carry[u]/cap[u]:.0%}" for u in units)
        else:
            slut_txt = ""
        tail = f"+{len(sns)-len(keep)}" if len(sns) > len(keep) else ""
        print(f"  fönster {i:3d}/{len(windows)} {keep[0]:%Y-%m-%d}–{keep[-1]:%Y-%m-%d} "
              f"({len(keep)}{tail} steg): {status}/{condition}  "f"{('v%d ' % wk) if wk else ''}λ={list(lam.values())[0]:.1f}  "
              f"{start_txt} {slut_txt}")
        if status != "ok":
            print(f"  ✖ fönster {i} misslyckades ({status}/{condition}) — avbryter.")
            return False, None

        part = {k: v.loc[keep] for k, v in extract_results(n).items()
                if v is not None and getattr(v, "shape", (0, 0))[1] > 0}
        for k in part:
            if k not in keys:
                keys.append(k)
        parts.append(part)

    fill = {u: soc_carry[u] / cap[u] for u in units}
    print("  slut-SOC: " + " ".join(f"{u.split()[0]} {f:.0%}" for u, f in fill.items()))
    if all(f > 0.95 for f in fill.values()):
        print("  ⚠️ ALLA reservoarer >95 % vid periodens slut — hamstring. "
              "Terminal-λ är för högt mot släppmarginalen (jfr run91–93).")

    results = {k: pd.concat([p[k] for p in parts if k in p]).sort_index() for k in keys}
    return True, results


def _git_commit() -> str:
    """Aktuell git-commit (kort hash + ev. 'dirty'). Tom sträng om ej git."""
    import subprocess
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=str(Path(__file__).resolve().parents[1])) != 0
        return h + (" (dirty)" if dirty else "")
    except Exception:
        return ""


def write_run_meta(label: str, args, res: int, year, flag_str: str) -> None:
    """Skriver results/<label>/run_meta.txt: syfte, konfig, full argv, git, tid.

    Görs så fort mappen finns (före lösning) så att även avbrutna/misslyckade
    körningar är självbeskrivande.
    """
    import datetime
    out = RESULTS_DIR / label / "run_meta.txt"
    desc = args.desc or "(ingen --desc angiven)"
    lines = [
        f"output:      {label}",
        f"syfte:       {desc}",
        f"tid:         {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"git:         {_git_commit() or '(ej git)'}",
        f"upplösning:  {res}h",
        f"år:          {year or '2023-2025'}",
        f"flaggor:     {flag_str.strip().strip('[]') or '(inga)'}",
        # Markör för att argv nedan skrevs mot den KANONISKA baselinens defaults.
        # --dispatch-replayen använder den för att avgöra om utelämnade optioner ska
        # tolkas som dagens defaults (raden finns) eller dåtidens (raden saknas).
        f"defaults:    {BASELINE_DEFAULTS_TAG}",
        # shlex.join → argument som innehåller mellanslag (t.ex. --market-ntc-override
        # "SE-S DE:1315") överlever round-trip via --dispatch-replayen.
        f"argv:        {shlex.join(sys.argv)}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → run_meta.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def apply_dispatch_replay(parser, args):
    """--dispatch LABEL: återspela LABEL:s scenario-argv (ur run_meta.txt) så att ett
    IDENTISKT system byggs (samma komponentnamn), men låt den nya körningens
    --resolution/--output/--desc/--year gälla. Kapaciteterna fryses sedan till LABEL:s
    p_nom_opt efter build_network (freeze_capacities_from). Expansionsläget behålls med
    flit — frysningen (extendable=False) gör körningen till ren dispatch."""
    label = args.dispatch
    meta = RESULTS_DIR / label / "run_meta.txt"
    if not meta.exists():
        parser.error(f"--dispatch: hittar inte {meta}")
    argv_line = next((l for l in meta.read_text().splitlines() if l.startswith("argv:")), None)
    if argv_line is None:
        parser.error(f"--dispatch: ingen argv-rad i {meta}")
    raw = argv_line.split(":", 1)[1]
    try:
        toks = shlex.split(raw)          # nya run_meta är shlex-citerade
    except ValueError:                   # äldre okiterade rader kan ha obalanserade '
        toks = raw.split()
    if toks and toks[0].endswith(".py"):
        toks = toks[1:]
    if "--desc" in toks:                      # ociterad fritext sist → klipp bort
        toks = toks[:toks.index("--desc")]
    drop = {"--output", "--resolution", "--dispatch"}   # enkelvärdes-flaggor som ersätts
    cleaned, i = [], 0
    while i < len(toks):
        if toks[i] in drop:
            i += 2
            continue
        cleaned.append(toks[i]); i += 1
    base = parser.parse_args(cleaned)
    # Källans argv skrevs mot DÅTIDENS defaults. Sedan 2026-08-05 är run250-konfen
    # default, så en oförändrad parse skulle smyga in t.ex. hydro-restriktioner i
    # omdispatchen av en körning som aldrig hade dem. Körningar gjorda EFTER
    # omläggningen bär en 'defaults:'-rad i run_meta och ska tvärtom behålla dagens
    # defaults — annars skulle en replay av run260 tappa hela baselinen.
    if not any(l.startswith("defaults:") for l in meta.read_text().splitlines()):
        restored = []
        for dest, (old, tokens) in PRE_BASELINE_DEFAULTS.items():
            if any(t in cleaned for t in tokens):
                continue                  # källan valde värdet medvetet → rör inte
            if getattr(base, dest) != old:
                setattr(base, dest, list(old) if isinstance(old, list) else old)
                restored.append(dest)
        if restored:
            print(f"--dispatch: {label} saknar 'defaults:'-rad (gjord före "
                  f"baseline-omläggningen) — återställer {', '.join(sorted(restored))} "
                  f"till dåtidens default")
    base.resolution = args.resolution or DEFAULT_DISPATCH_RESOLUTION
    base.output     = args.output
    base.desc       = args.desc or f"omdispatch av {label} @ {base.resolution}h (frysta p_nom_opt)"
    base.dry_run    = args.dry_run            # "hur"-flaggor från nya kommandot vinner
    base.low_hydro  = args.low_hydro          # scenario-modifierare på NYA körningen
    base.vre_curtailment_cost = args.vre_curtailment_cost   # hör till OMDISPATCHEN, inte källan:
                                              # källans argv har den aldrig (expansion förbjuder
                                              # den), så utan denna rad blir flaggan tyst
                                              # verkningslös — jfr --voll/--low-hydro-fällan.
                                              # None = orörd; sentineln löses ut efter denna
                                              # funktion, då base.dispatch är satt.
    base.voll       = args.voll               # VOLL-slack appliceras på dispatch-replayen
    base.no_voll    = args.no_voll            # ...och dess av-knapp måste följa med, annars
                                              # läses den ur KÄLLANS argv och --dispatch X
                                              # --no-voll blir tyst verkningslös
    base.soc_pin_from = args.soc_pin_from     # tvåpass-pin styrs av NYA kommandot
    base.soc_pin_freq = args.soc_pin_freq
    base.soc_pin_band = args.soc_pin_band
    base.solver_option = args.solver_option   # numerik-flaggor hör till NYA körningen
    # Hydro-driftgolven hör till OMDISPATCHEN: de är ett scenarioval om hur hårt
    # vattenkraften måste gå, precis som --low-hydro. Utan dessa rader läses de ur
    # KÄLLANS argv och en ny --hydro-min-hourly blir tyst verkningslös.
    if args.hydro_min_hourly is not None:
        base.hydro_min_hourly = args.hydro_min_hourly
    if args.hydro_min_daily is not None:
        base.hydro_min_daily = args.hydro_min_daily
    # --spill-cost betyder OLIKA SAKER i de två lägena (räcke mot fantomabsorption i
    # expansion, fysisk kostnad i dispatch), så källans värde får inte läcka in och ett
    # nytt värde måste bita. Utan raden läses den ur KÄLLANS argv och `--dispatch X
    # --spill-cost 0.1` blir tyst verkningslöst — samma fällklass som run319. None här
    # betyder "orörd" och löses ut lägesberoende efter denna funktion.
    base.spill_cost = args.spill_cost
    # Rullande horisont och terminalvärdet hör HELT till omdispatchen. Källans argv kan
    # aldrig innehålla dem — rullande är dispatch-only, så en expansionskörning förbjuder
    # dem. Utan dessa rader blir de tyst verkningslösa och körningen faller tillbaka på
    # en vanlig cyklisk LP MED prisproxyn, alltså tvärtemot vad kommandot bad om.
    # ⚠️ Det är en tystare fälla än --voll/--low-hydro: eftersom även --terminal-curve
    # nollas hinner dess egna räcken aldrig utlösa. run319 gick i den 2026-08-13.
    base.rolling_horizon         = args.rolling_horizon
    base.rolling_weeks           = args.rolling_weeks
    base.rolling_lookahead_weeks = args.rolling_lookahead_weeks
    base.terminal_curve          = args.terminal_curve
    base.terminal_segments       = args.terminal_segments
    base.no_rolling_horizon      = args.no_rolling_horizon
    base.no_terminal_curve       = args.no_terminal_curve
    base.hydro_price_proxy       = args.hydro_price_proxy
    base.terminal_seasonal       = args.terminal_seasonal
    base.terminal_lambda         = args.terminal_lambda
    base.terminal_lambda_scale   = args.terminal_lambda_scale
    base.terminal_lambda_profile = args.terminal_lambda_profile
    base.no_hydro_price_proxy    = args.no_hydro_price_proxy
    if args.year is not None:
        base.year = args.year
    base.dispatch = label
    print(f"--dispatch: återspelar {label}:s argv → {base.resolution}h; "
          f"kapaciteter fryses till {label}:s p_nom_opt efter bygget")
    return base


def freeze_capacities_from(n, label):
    """Sätt p_nom = p_nom_opt (och extendable=False) på alla komponenter (gen/lager/länkar)
    som matchar namn i results/LABEL/network.nc → fryser kapaciteten till den körningens
    optimum. Flexibilitet (dispatch, lager-SOC, handel) optimeras fortfarande fritt.

    För lager kopieras även max_hours. Annars blandas källans p_nom med den NYA
    upplösningens max_hours, och eftersom RoR-splitten (som bevarar reservoarvolymen
    genom max_h = cap_mwh/p_nom) faller ut olika vid olika upplösning blir produkten
    p_nom×max_hours — reservoarvolymen — fel. Konkret: FI fick 5,27 i stället för
    5,50 TWh vid 2h-omdispatch av en 3h-körning (−4%).
    """
    src = pypsa.Network()
    src.import_from_netcdf(RESULTS_DIR / label / "network.nc")
    print(f"  → fryser kapaciteter till {label}:s p_nom_opt:")
    for cname, ndf, sdf in (("generatorer", n.generators, src.generators),
                            ("lager",       n.storage_units, src.storage_units),
                            ("länkar",      n.links, src.links)):
        if "p_nom_opt" not in sdf.columns or ndf.empty:
            continue
        common = [x for x in ndf.index if x in sdf.index]
        skipped = []
        if cname == "generatorer" and "p_nom_extendable" in sdf.columns:
            # p_nom_opt bär BARA information för extendable komponenter. För databestämda
            # must-run-generatorer (thermal, hydro_ror) är p_nom = profilens max i KÄLLANS
            # snapshot-fönster, medan p_min_pu = p_max_pu = profil/max normaliseras mot den
            # NYA körningens fönster. Att frysa p_nom skalar då om produktionen med
            # (källans max / nya fönstrets max) — och för must-run ÄR p_nom × pu dispatchen.
            # Konkret: 1h-dispatch per år av en 3-årig källa gav NO-N termik ×2,6 (2023) och
            # ×2,8 (2025), NO-S ×2,5/×3,1, eftersom årsmaxen ligger långt under 3-årsmaxet.
            fixed = [x for x in common if not bool(sdf.at[x, "p_nom_extendable"])]
            if fixed:
                skipped = fixed
                common = [x for x in common if x not in set(fixed)]
        ndf.loc[common, "p_nom"] = sdf.loc[common, "p_nom_opt"].astype(float)
        if "p_nom_extendable" in ndf.columns:
            ndf.loc[common, "p_nom_extendable"] = False
        if "max_hours" in ndf.columns and "max_hours" in sdf.columns:
            before = (ndf.loc[common, "p_nom"] * ndf.loc[common, "max_hours"]).sum()
            ndf.loc[common, "max_hours"] = sdf.loc[common, "max_hours"].astype(float)
            after = (ndf.loc[common, "p_nom"] * ndf.loc[common, "max_hours"]).sum()
            if abs(after - before) > 1e-6 * max(after, 1.0):
                print(f"      lagervolym korrigerad: {before/1e6:.2f} → {after/1e6:.2f} TWh "
                      f"(max_hours ärvs från {label})")
        miss = [x for x in ndf.index if x not in sdf.index]
        msg = f"      {cname}: {len(common)} frysta"
        if skipped:
            msg += (f", {len(skipped)} databestämda ej frysta (behåller egen p_nom: "
                    f"{', '.join(skipped[:3])}{' …' if len(skipped) > 3 else ''})")
        if miss:
            msg += f"  ⚠️ {len(miss)} saknas i {label} ({miss[:3]})"
        print(msg)


VRE_CARRIERS = ("wind_onshore", "wind_offshore", "solar")


def apply_vre_curtailment_cost(n, cost: float) -> None:
    """--vre-curtailment-cost C: låt VRE bjuda vom − C i stället för vom.

    En kostnad C per MWh AVKORTAD energi ger objektivtermen

        mc·p + C·(A − p)  =  C·A + (mc − C)·p

    där A = p_max_pu·p_nom är den tillgängliga energin. Med FAST kapacitet är C·A
    en konstant som faller ur optimeringen — en avkortningskostnad är därför exakt
    ekvivalent med ett bud på mc − C, och implementeras enklast så. Det är den
    mekanism som ger negativa priser på riktiga marknader: subventionen (CfD,
    elcertifikat) sätter golvet på −C, eftersom producenten hellre betalar upp till
    C för att bli av med kraften än tappar stödet.

    Utan detta kan modellens zonpriser aldrig gå under billigaste budet (VRE:s VOM
    0,1 €/MWh): avkortning är gratis, så avdisposition kostar aldrig något och en
    extra MWh last kan aldrig sänka systemkostnaden.

    ⚠️ Kräver FRYSTA kapaciteter. Med extendable VRE är A ∝ p_nom_opt, C·A är då
    INTE konstant utan en produktionssubvention som växer med byggd kapacitet —
    optimeraren bygger till p_nom_max för att skörda den. Subventionen är en
    transferering, inte en resurskostnad, och hör hemma i prisbildningen men inte i
    investeringskalkylen. Därav tvåpass: expansion med sanna kostnader → --dispatch
    med negativa bud.
    """
    if not cost:
        return
    g = n.generators
    targets = [x for x in g.index if g.at[x, "carrier"] in VRE_CARRIERS]
    if not targets:
        print(f"  → --vre-curtailment-cost {cost:g}: inga VRE-generatorer — ingen ändring")
        return

    ext = [x for x in targets if bool(g.at[x, "p_nom_extendable"])]
    if ext:
        raise SystemExit(
            f"\n--vre-curtailment-cost {cost:g} kräver frysta kapaciteter, men "
            f"{len(ext)} VRE-generatorer är fortfarande extendable\n"
            f"  (t.ex. {', '.join(ext[:3])}{' …' if len(ext) > 3 else ''}).\n\n"
            "Med extendable VRE blir avkortningskostnaden en produktionssubvention som\n"
            "växer med byggd kapacitet — modellen bygger till p_nom_max för att skörda\n"
            "den och investeringssvaret blir meningslöst. Kör i stället tvåpass:\n"
            "  1) expansion med sanna kostnader  → results/<label>/network.nc\n"
            "  2) python scripts/run_model.py --dispatch <label> "
            f"--vre-curtailment-cost {cost:g} --output <nytt>\n"
        )

    print(f"  → --vre-curtailment-cost {cost:g} EUR/MWh: VRE bjuder vom − {cost:g}")
    tv = n.generators_t.marginal_cost
    for x in targets:
        g.at[x, "marginal_cost"] = float(g.at[x, "marginal_cost"]) - cost
        if x in tv.columns:                      # tidsberoende mc vinner över den statiska
            tv[x] = tv[x] - cost
    lo = g.loc[targets, "marginal_cost"]
    print(f"     {len(targets)} generatorer, bud {lo.min():.2f} … {lo.max():.2f} EUR/MWh "
          f"(prisgolv i överskottstimmar ≈ {lo.min():.2f})")


def apply_low_hydro(n, factor, year=2024):
    """Torrårs-scenario: skala ett års hydro NEDÅT med factor — både reservoar-inflöde
    (storage_units_t.inflow) och RoR (must-run-generatorer, carrier 'hydro': p_max_pu &
    p_min_pu). Övriga år orörda. factor<1 = torrare."""
    dt  = (n.snapshots[1] - n.snapshots[0]).total_seconds() / 3600
    inf = n.storage_units_t.inflow
    m_inf = inf.index.year == year
    if not m_inf.any():
        print(f"  → --low-hydro: år {year} ingår ej i perioden — ingen ändring")
        return
    before = float(inf.loc[m_inf].to_numpy().sum()) * dt / 1e6
    inf.loc[m_inf] *= factor
    ror = [g for g in n.generators.index if n.generators.at[g, "carrier"] == "hydro"]
    static = []
    for g in ror:
        hit = False
        for tbl in (n.generators_t.p_max_pu, n.generators_t.p_min_pu):
            if g in tbl.columns:
                tbl.loc[tbl.index.year == year, g] *= factor
                hit = True
        if not hit:
            static.append(g)
    msg = (f"  → --low-hydro {factor:g}: {year} reservoar-inflöde {before:.1f} → "
           f"{before * factor:.1f} TWh + RoR ×{factor:g} ({len(ror) - len(static)} profil-gen)")
    if static:
        msg += f"  ⚠️ {len(static)} RoR-gen har statisk p_max_pu, ej skalade ({static[:3]})"
    print(msg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=None,
                        help="Tidsupp. i timmar (åsidosätter config)")
    parser.add_argument("--year", type=int, default=None,
                        help="Kör ett enstaka år (t.ex. 2024)")
    parser.add_argument("--output", default=None,
                        help="Resultatmapp under results/ (t.ex. 'run_v2_spring_flood'). "
                             "Standard: automatiskt namn baserat på upplösning och år.")
    parser.add_argument("--desc", default=None,
                        help="Kort fritext om körningens SYFTE (t.ex. 'kandidat för "
                             "dispatch-baseline med värme och vätgaslast'). Sparas i "
                             "results/<output>/run_meta.txt.")
    parser.add_argument("--dispatch", default=None, metavar="LABEL",
                        help="OMDISPATCH-läge: bygg om en tidigare körnings system vid annan "
                             "upplösning och LÅS alla kapaciteter till dess p_nom_opt. Återspelar "
                             "LABEL:s scenario-argv (results/LABEL/run_meta.txt) men låter den NYA "
                             "körningens --resolution/--output/--desc/--year gälla. Flexibiliteter "
                             "(hydro, batteri, handel) optimeras fritt. T.ex. expansion 3h → "
                             "omdispatch 1h: '--dispatch run164_... --resolution 1 --output run167_...'.")
    parser.add_argument("--low-hydro", type=float, default=None, metavar="FACTOR",
                        help="Torrårs-scenario: skala 2024 års hydro NEDÅT med FACTOR (t.ex. 0.6 "
                             "= 60%% av normalt = −40%%). Påverkar både reservoar-inflöde och RoR "
                             "(must-run). Andra år orörda. Tydligast effekt i en --year 2024-körning "
                             "(cyklisk SOC buffrar i fleråriga). Realistiskt torrår ≈ 0.6–0.7.")
    parser.add_argument("--extra-load", type=float, default=0.0,
                        help="Extra flat last i MW per zon (utöver faktisk last, standard: 0)")
    parser.add_argument("--extra-load-zone", action="append", default=[], metavar="ZON:MW",
                        help="Extra flat last i MW i EN zon (t.ex. 'SE-S:1000'). Additivt över "
                             "eSett-basen och över --extra-load. Kan anges flera gånger. Avsett "
                             "för marginalkostnads-/LRMC-experiment (ΔObjektiv/ΔKonsumtion).")
    parser.add_argument("--no-expansion", action="store_true",
                        help="Lås alla teknologier som non-extendable — ren dispatch-körning")
    parser.add_argument("--cost-scenario", default="svk_2040", metavar="NAMN",
                        help="Skriv över cfg['costs'] med ett kostnadsscenario ur "
                             "cost_scenarios i zones.yaml (t.ex. svk_2040, svk_2050). "
                             "Inkl. byggränta (IDC). DEFAULT svk_2040 (kanonisk baseline); "
                             "'none' använder dagens kostnader i config.")
    parser.add_argument("--demand-scenario", default="svk_2040_mm", metavar="NAMN",
                        help="Addera ett efterfrågescenario ur demand_scenarios i "
                             "zones.yaml (t.ex. svk_2040_mm): per-zon extra-last, H2, EV, "
                             "utbyggnadstak (p_nom_max) och NTC-höjningar. Additivt över "
                             "eSett-basen. DEFAULT svk_2040_mm (kanonisk baseline); "
                             "'none' kör dagens efterfrågan.")
    parser.add_argument("--continent-diurnal-scale", type=float, default=1.0, metavar="FAKTOR",
                        help="Komprimera kontinent-ventilprisets DYGNSSVÄNG (hour-of-day-komponent) "
                             "med FAKTOR (1.0=oförändrat, 0.5=halverad). Behåller nivå + dag-till-dag-"
                             "väder-volatilitet. Modellerar att 2040-kontinentlagring arbitrerar bort "
                             "den sol-formade dygnsspreaden Nordens hydro annars utnyttjar. Bara "
                             "ventil-bzn (DE-LU/EE/LT/PL/NL/GB), ej zon-priser. Se project_solar_overbuild_continent_spread.")
    parser.add_argument("--no-market", action="store_true",
                        help="Stäng ned alla externa marknadsanslutningar (p_nom=0)")
    parser.add_argument("--voll", nargs="?", type=float, const=3000.0, default=3000.0, metavar="EUR",
                        help="Lägg VOLL-slack i ALLA zoner vid EUR/MWh (bart --voll = 3000, EI ~8000). "
                             "Ger LOLE/EENS-mått (slack-dispatch = osåld energi), cappar priser vid VOLL och "
                             "förhindrar dualexplosion. DEFAULT 3000 (kanonisk baseline); --no-voll ger det "
                             "gamla beteendet: slack bara i icke-marknadszoner (SE-N, NO-N) @ 3000.")
    parser.add_argument("--no-voll", action="store_true",
                        help="Ingen VOLL-slack i marknadszonerna (bara SE-N/NO-N får slack, som före 2026-08-05).")
    parser.add_argument("--soc-pin", action="append", default=[], metavar="ZON:START:END",
                        help="Icke-cyklisk: lås BÅDA ändpunkterna till faktiska fyllnadsfraktioner per zon, "
                             "t.ex. 'SE-N:0.577:0.709' (start 57.7%%, slut 70.9%% av kapacitet). Kalibrering mot "
                             "observerad reservoarnivå. Komma-separera flera eller upprepa flaggan.")
    parser.add_argument("--soc-pin-from", default=None, metavar="LABEL",
                        help="TVÅPASS: lös perioden fönstervis med hydro-SOC pinnad i BÅDA ändar "
                             "till LABEL:s lagerbana (results/LABEL/hydro_soc.csv, interpoleras till "
                             "denna körnings upplösning). Ersätter cyklisk SOC + perfekt framsyn över "
                             "hela perioden med källans säsongsbana + fri dispatch inom fönstret. "
                             "Avsedd ihop med --dispatch LABEL (frysta kapaciteter). Fönsterlängd: "
                             "--soc-pin-freq.")
    parser.add_argument("--soc-pin-band", type=float, default=0.0, metavar="FRAC",
                        help="Band kring SOC-pinnen: målet blir target ± FRAC × kapaciteten. "
                             "Default 0 = hård likhet (oförändrat beteende) — men den blir lätt "
                             "INFEASIBLE när dispatchen körs på annan upplösning än källan "
                             "(uttappningen ryms ej i timbalansen). Med band > 0 startar varje "
                             "fönster på föregående fönsters UPPNÅDDA nivå; målet re-ankras "
                             "mot källan varje fönster så avvikelsen ackumuleras ej. "
                             "OBS: bandgrenen är ännu inte verifierad i en lyckad körning.")
    parser.add_argument("--soc-pin-freq", default="MS", metavar="FREQ",
                        help="Fönsterlängd för --soc-pin-from (pandas-frekvens). Default 'MS' = "
                             "kalendermånad; 'W' = vecka, 'QS' = kvartal. Fönstret måste vara långt "
                             "mot batteri/EV-cykeln (de förblir cykliska per fönster).")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="Skriv över/lägg till en solver-option ur config/zones.yaml:solver, "
                             "t.ex. 'user_bound_scale=-6' (HiGHS: skala modellens gränser med "
                             "2^VALUE — mot 'excessively large row bounds' när lager i MWh ger "
                             "RHS ~6e7 och IPM:s dual divergerar). Typ tolkas automatiskt "
                             "(int/float/bool/sträng). Kan anges flera gånger.")
    parser.add_argument("--spill-cost", type=float, default=None, metavar="EUR",
                        help="Hydro-spillkostnad (EUR/MWh). LÄGESBEROENDE default: 50 i "
                             "EXPANSION (räcke — gratis spill låter modellen dumpa undanträngd "
                             "vattenkraft och överinvestera i VRE: run43 63 TWh fantomspill och "
                             "NO-S vind 20,8 GW, mot run44:s 0 och 13,1) och 0.1 med FRYSTA "
                             "kapaciteter (--dispatch/--no-expansion), där 50 skulle dubbelräkna "
                             "vattnets värde — det bärs redan av SOC-skuggpriset. Explicit värde "
                             "vinner i båda lägena.")
    parser.add_argument("--rolling-horizon", action="store_true",
                        help="Lös perioden i sekventiella fönster (--rolling-weeks) med "
                             "icke-cyklisk SOC, carry-over av slut-SOC och terminalvärde "
                             "−λ×SOC[T]. Bryter den perfekta framsynen som gör det endogena "
                             "vattenvärdet nästan konstant. DISPATCH-ONLY: varje fönster är en "
                             "egen LP, så kapaciteter måste vara frysta (--no-expansion/--dispatch). "
                             "⚠️ Kör med --no-hydro-price-proxy — annars är λ på bruttoprisnivå "
                             "medan marginalen är nettovattenvärdet, och reservoarerna hamstrar "
                             "(felet i run91–93).")
    parser.add_argument("--rolling-weeks", type=int, default=1, metavar="N",
                        help="Fönsterlängd i veckor för --rolling-horizon (default 4).")
    parser.add_argument("--terminal-seasonal", action="store_true",
                        help="Byt λ:s TIDSFORM från DE-LU:s framåtpris till den handsatta "
                             "säsongstabellen terminal_value.seasonal_factors. Motivet är "
                             "cirkularitet: default läser formen ur en prisserie, alltså ur "
                             "det modellen ska förutsäga. ⚠️ NIVÅN (base_level_eur_mwh) är "
                             "fortfarande zonens observerade medelpris — bara halva "
                             "cirkulariteten försvinner.")
    parser.add_argument("--rolling-lookahead-weeks", type=int, default=None, metavar="N",
                        help="ÄKTA RECEDING HORIZON (väg E): lös varje fönster med N "
                             "veckors extra look-ahead men behåll bara den första delen. "
                             "Default 0 = icke-överlappande fönster (nuvarande beteende), "
                             "där varje fönster ser noll framåt och terminalvärdet ensamt "
                             "måste bära säsongssignalen. Med look-ahead gör framsynen det "
                             "mesta av jobbet och känsligheten för terminalkalibreringen "
                             "bör falla — det är själva testet.")
    parser.add_argument("--expansion", action="store_true",
                        help="Kanonisk EXPANSION. Är redan default när --dispatch "
                             "utelämnas; flaggan finns för att skript ska kunna säga "
                             "det uttryckligen. Kan inte kombineras med --dispatch.")
    parser.add_argument("--no-rolling-horizon", action="store_true",
                        help="Stäng av den rullande horisonten som --dispatch slår på.")
    parser.add_argument("--no-terminal-curve", action="store_true",
                        help="Stäng av terminalkurvan som --dispatch slår på.")
    parser.add_argument("--hydro-price-proxy", action="store_true",
                        help="Sätt TILLBAKA prisproxyn, som --dispatch stänger av. "
                             "⚠️ Går inte att kombinera med terminalkurvan.")
    parser.add_argument("--terminal-segments", type=int, default=20, metavar="N",
                        help="Antal lika stora SOC-segment i terminalvärdeskurvan "
                             "(--terminal-curve). DEFAULT 20. Kurvan sätter segmentens "
                             "VÄRDEN själv; det här styr bara upplösningen i "
                             "fyllnadsgrad. ⚠️ Finare indelning ger en jämnare budtrappa "
                             "men löser INTE att en zon bara har ETT vattenvärde åt "
                             "gången — magasinet rör sig ~20 procentenheter per månad "
                             "och korsar därför få segment oavsett indelning (run339).")
    parser.add_argument("--terminal-lambda-profile", default=None, metavar="M1,M2,...",
                        help="Styckvis linjär KONKAV terminalvärdeskurva: multiplikatorer på "
                             "bas-λ per lika stort SOC-segment, TOMT→FULLT. Måste vara "
                             "icke-växande. DEFAULT "
                             f"{','.join(f'{p:g}' for p in DEFAULT_TERMINAL_PROFILE)} "
                             "(marginalvärde = bas-λ vid 60-80 %% fyllnad, ~config-ankaret; "
                             "2× nära tomt, 0,2× i toppbandet). '1.0' ger det gamla LINJÄRA "
                             "beteendet, som ger bang-bang: magasin i taket och priskollaps "
                             "till VOM (run268). ⚠️ Formen är ett ANTAGANDE, ej kalibrerad.")
    parser.add_argument("--terminal-curve", nargs="?", const="", default=None, metavar="FIL",
                        help="Analytisk terminalvärdeskurva λ_k(vecka, zon) = λ_bas · A(w) · "
                             "P_k(B(w)) ur FIL (default config/terminal_curve.yaml). Ersätter "
                             "BÅDE --terminal-lambda* och --terminal-lambda-profile och räknas "
                             "om per fönster, så profilen blir SÄSONGSBEROENDE i stället för "
                             "årskonstant. Efterföljare till --terminal-seasonal: A(w) är "
                             "samma idé som S(m) men parametrisk, B(w) ger dessutom "
                             "fyllnadsberoendet en säsong, och λ_bas hämtas ur expansionens "
                             "EFFEKTIVA hydrobud (proxy + μ/η) i stället för zonens "
                             "observerade pris — S(m):s kvarvarande halva cirkularitet. "
                             "Kalibreras med scripts/calibrate_terminal_curve.py mot eSetts "
                             "fysiska hydrosäsong och EC:s magasinband, båda MÄTDATA. "
                             "⚠️ Kräver --rolling-horizon och --no-hydro-price-proxy.")
    parser.add_argument("--terminal-lambda-scale", default=None, metavar="ZON:α,...",
                        help="λ_zon(t) = α_zon × framåtblickande DE-LU. Behåller kontinent"
                             "prisets TIDSFORM men skalar NIVÅN per zon — en inlåst zon kan "
                             "inte värdera sitt vatten till kontinentens pris. Föredras framför "
                             "--terminal-lambda: ett KONSTANT λ tömmer magasinen före vårfloden "
                             "(run271 blev infeasible, SE-N 9 %% / NO-N 0 %% i februari 2024), "
                             "eftersom vattnets alternativkostnad är säsongsberoende.")
    parser.add_argument("--terminal-lambda", default=None, metavar="VÄRDE|ZON:V,...",
                        help="Fast terminal-λ (EUR/MWh) i stället för framåtblickande DE-LU. "
                             "Skalär för alla hydrozoner, eller per zon "
                             "('SE-N:30,NO-S:55'). Zoner som utelämnas får λ=0.")
    parser.add_argument("--vre-curtailment-cost", type=float, default=None, metavar="EUR",
                        help="Kostnad (EUR/MWh) för AVKORTAD vind/sol — ekvivalent med att VRE "
                             "bjuder vom − EUR. Enda vägen till NEGATIVA zonpriser: utan detta är "
                             "avkortning gratis och priset kan aldrig gå under billigaste budet "
                             "(VOM 0,1). KRÄVER frysta kapaciteter — med extendable VRE blir det "
                             "en produktionssubvention som bygger till p_nom_max. "
                             f"DEFAULT {DEFAULT_VRE_CURTAILMENT_COST:g} i DISPATCHLÄGE "
                             "(--dispatch eller --no-expansion), annars 0. Kalibrerad mot "
                             "ursprungsgarantier (GO, prognos 2-5) — elcert är noll i 2040 och "
                             "CfD betalar inte i negativa timmar. Sätt 0 för att stänga av.")
    parser.add_argument("--add-battery", action="append", default=[], metavar="ZON:MW:HOURS",
                        help="Lägg till batteri (StorageUnit) i en zon, t.ex. 'SE-S:5000:4'. BÄR "
                             "annualiserad svk_2040-kapex (inkl. IDC) även fast i dispatch (konstant "
                             "i objektivet). Med --battery-extendable optimeras effekten i stället. "
                             "Kan anges flera gånger.")
    parser.add_argument("--battery-extendable", action="store_true",
                        help="Gör --add-battery investerbart: modellen optimerar effekten "
                             "0..MW (varaktighet fast) mot batterikostnad i config. "
                             "Utan flaggan är batteriet fast (dispatch-tillägg).")
    parser.add_argument("--add-solar", action="append", default=[], metavar="ZON:MW",
                        help="Lägg till COSTAD sol (Generator '{zon} solar add') i en zon, t.ex. "
                             "'SE-S:3895'. Bär annualiserad svk_2040-kapex (inkl. IDC) i objektivet "
                             "(extendable-pinnat), zonens solprofil. Kan anges flera gånger.")
    parser.add_argument("--add-onshore", action="append", default=[], metavar="ZON:MW[:UPLIFT]",
                        help="Lägg till COSTAD landbaserad vind (Generator '{zon} onshore add') "
                             "i en zon, t.ex. 'SE-S:5720'. Bär annualiserad svk_2040-kapex "
                             "(inkl. IDC) i objektivet (extendable-pinnat), zonens onshore-profil. "
                             "Valfri CF-uplift som 3:e fält (0.20 = +20%% medel-CF, potens-transform) "
                             "boostar BARA detta tillägg — flottan orörd. Kan anges flera gånger.")
    parser.add_argument("--add-offshore", action="append", default=[], metavar="ZON:MW[:UPLIFT]",
                        help="Som --add-onshore men HAVSbaserad vind (Generator '{zon} offshore add', "
                             "carrier wind_offshore, svk_2040-kapex). Zonens offshore-profil, valfri "
                             "CF-uplift som 3:e fält. Kan anges flera gånger.")
    parser.add_argument("--add-cost-scenario", default="svk_2040", metavar="NAMN",
                        help="Kostnadsscenario (cost_scenarios i config) som COSTADE tillägg "
                             "(--add-battery/--add-nuclear-fixed/--add-solar/--add-onshore/"
                             "--add-offshore) prissätts från. Default svk_2040. T.ex. svk_2025 "
                             "för dagens kostnadsnivå. Oberoende av --cost-scenario (driftsblocket).")
    parser.add_argument("--add-oc-scale", nargs="+", default=None, metavar="TECH:FAKTOR",
                        help="Skala overnight-kostnaden för COSTADE tillägg per teknik, t.ex. "
                             "'battery:0.5 nuclear:1.5'. Påverkar bara --add-battery / "
                             "--add-nuclear-fixed (svk_2040-kapexen), ej befintlig flotta. "
                             "Sensitivitetsanalys.")
    parser.add_argument("--battery", nargs="+", default=None, metavar="DURATION [ZON:MW ...]",
                        help="Batterier med given varaktighet (t.ex. '4h'). Utan zoner: "
                             "expanderbart 4h-batteri i VARJE zon (modellen optimerar effekten). "
                             "Med zoner (t.ex. '4h SE-S:5000 SE-N:2000'): fasta storlekar i de "
                             "angivna zonerna, 0 i övriga. Varaktigheten är fast (StorageUnit).")
    parser.add_argument("--scenario-battery", default=None, metavar="GW:HOURS",
                        help="Skala om efterfrågescenariots exogena batterier till total GW @ HOURS "
                             "(behåller zonfördelningen). T.ex. '25:4' = 25 GW 4h i stället för "
                             "scenariots 12 GW 2h. Kräver --demand-scenario.")
    parser.add_argument("--expand-vre", action="append", default=[], metavar="ZON",
                        help="Gör wind_onshore/wind_offshore/solar i ZON investerbara "
                             "(oavsett --no-expansion). Kan anges flera gånger. Kombinera "
                             "med --expand-budget-musd för riktad budgetbegränsad expansion.")
    parser.add_argument("--expand-budget-musd", type=float, default=None, metavar="MUSD",
                        help="Tak på overnight-kostnaden (miljoner USD) för Σ tillkommande "
                             "VRE+batteri i --expand-vre-zonerna. T.ex. 25000 = 25 GUSD. "
                             f"Konverteras till EUR med {USD_TO_EUR} USD/EUR.")
    parser.add_argument("--expand-budget-meur", type=float, default=None, metavar="MEUR",
                        help="Som --expand-budget-musd men direkt i miljoner EUR. "
                             "T.ex. 20000 = 20 mdr€. Har företräde om båda anges.")
    parser.add_argument("--onwind-capfac-increase", type=float, default=0.30, metavar="FRAC",
                        help="Höj landbaserad vinds kapacitetsfaktor med denna relativa "
                             "andel (0.1 = +10%%). Olinjär potens-transform per zon: lyfter "
                             "låga effektnivåer mest, märkeffekt (cf_max) oförändrad. "
                             "DEFAULT 0.30 (kanonisk baseline); 0 = dagens flotta.")
    parser.add_argument("--offwind-capfac-increase", type=float, default=0.10, metavar="FRAC",
                        help="Som --onwind-capfac-increase men för HAVSbaserad vind "
                             "(0.1 = +10%%). Speglar 2040:s nybyggnadsflotta (moderna 15 MW-"
                             "turbiner). Påverkar genereringen; potential-MW i config förutsätter "
                             "matchande CF. DEFAULT 0.10 (kanonisk baseline); 0 = dagens flotta.")
    parser.add_argument("--offwind-discount-rate", nargs="+", action="extend", default=[], metavar="ZON:RATE",
                        help="Egen diskontoränta för HAVSbaserad vind i en zon, t.ex. "
                             "'SE-N:0.03 SE-S:0.03'. Påverkar bara den annualiserade kapitalkostnaden "
                             "(samma subventioneringsmekanik som --nuclear-discount-rate). Default = "
                             "global costs.discount_rate. Kan anges flera gånger / som lista.")
    parser.add_argument("--onshore-cap", action="append", default=[], metavar="ZON:MW",
                        help="Sätt expansionstak (p_nom_max, MW) för landbaserad vind per zon, "
                             "t.ex. 'SE-N:20000'. Override på default-taket (50 GW/zon). "
                             "Kräver expansion (onshore extendable). Kan anges flera gånger.")
    parser.add_argument("--onshore-lower", type=float, default=1.0, metavar="FAKTOR",
                        help="Skala ALLA zoners landvind-potentialtak (--demand-scenario "
                             "pnom_max_mw) med FAKTOR. Default 1.0; 0.8 = −20%% onshore-tak. "
                             "Golv (pnom_min_mw) lämnas orört. Sensitivitet på utbyggnadspotentialen.")
    parser.add_argument("--solar-cap", type=float, default=None, metavar="MW",
                        help="Sätt expansionstak (p_nom_max, MW) för sol i ALLA zoner, "
                             "t.ex. '10000'. Override på default-taket (50 GW/zon). "
                             "Kräver expansion (sol extendable).")
    parser.add_argument("--grid-cost", action="store_true",
                        help="Aktivera nätkostnad som kapital-adder per kraftslag × zon "
                             "(costs.grid i zones.yaml). Internaliserar att vind/sol långt "
                             "från lasten drar mer nätutbyggnad per MW än kärnkraft nära. "
                             "Default AV (capital_cost identisk med idag). ⚠️ grid-siffrorna "
                             "är PLATSHÅLLARE tills de förankrats i källor (ENTSO-E TYNDP/NREL).")
    # Driftrestriktionerna är PÅ som default; --no-hydro-restrictions stänger av.
    # --hydro-restrictions behålls (no-op) för bakåtkompatibilitet med äldre kommandon.
    parser.add_argument("--hydro-restrictions", action="store_true", default=True,
                        dest="hydro_restrictions",
                        help="(default PÅ) Driftrestriktioner på RESERVOARvattenkraften "
                             "(hydro_operation i zones.yaml): min timproduktion, min "
                             "dygnsproduktion och max veckoproduktion som andel av "
                             "installerad reservoareffekt. Hindrar både total avstängning "
                             "under lågprisperioder och maxeffekt vecka efter vecka. "
                             "⚠️ veckotaken för NO-N/NO-S/FI är ANTAGANDEN — Ek Fälth m.fl. "
                             "(2025) täcker bara Sverige.")
    parser.add_argument("--no-hydro-restrictions", action="store_false",
                        dest="hydro_restrictions",
                        help="Stäng av hydro-driftrestriktionerna (fri reservoardrift).")
    parser.add_argument("--no-hydro-price-proxy", action="store_true",
                        help="Ge reservoarvattenkraften platt VOM som marginal_cost i "
                             "stället för zonens FAKTISKA historiska day-ahead-pris. "
                             "Proxyn (network.py, 'water value proxy' från run23) ligger "
                             "sedan run57 STAPLAD ovanpå det äkta endogena vattenvärdet "
                             "och står för HELA tidsvariationen i hydrons bud. Denna "
                             "flagga isolerar dess effekt. Default: proxyn PÅ (oförändrat).")
    parser.add_argument("--hydro-min-hourly", type=float, default=None, metavar="FRAC",
                        help="Override på min timproduktion (andel av reservoar-p_nom). "
                             "Implicerar --hydro-restrictions. 0 = av.")
    parser.add_argument("--hydro-min-daily", type=float, default=None, metavar="FRAC",
                        help="Override på min dygnsproduktion (andel av max dygnsproduktion). "
                             "Implicerar --hydro-restrictions. 0 = av.")
    parser.add_argument("--hydro-max-weekly", nargs="+", action="extend", default=None,
                        metavar="FRAC|ZON:FRAC",
                        help="Override på max veckoproduktion (andel av max veckoproduktion). "
                             "Ett ensamt tal gäller alla zoner; 'ZON:FRAC' sätter en zon, "
                             "t.ex. '--hydro-max-weekly 0.77 NO-S:0.80'. "
                             "Implicerar --hydro-restrictions. 0 = av.")
    parser.add_argument("--hydro-bypass-spill", type=float, default=None, metavar="KOEF",
                        help="Aktivera spill förbi mindre stationer när veckoproduktionen "
                             "ligger inom threshold_below_max under veckotaket. KOEF = MWh "
                             "spill per MWh produktion över tröskeln. Implicerar "
                             "--hydro-restrictions. ⚠️ kan ge INFEASIBLE: PyPSA:s spill är "
                             "begränsad av tillrinningen i samma snapshot.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Bygg nätverket och skriv en komponent-sammanfattning "
                             "(kärnkraft m.m.), men SOLVE:a inte. För verifiering före körning.")
    # OBS: default=None är en sentinel, inte "tom lista". Med action="extend" och en
    # icke-tom default skulle ett eget --add-nuclear EXTENDA baslistan i stället för att
    # ersätta den. Den kanoniska listan sätts därför efter parse_args.
    parser.add_argument("--add-nuclear", nargs="+", action="extend", default=None, metavar="ZON:N:SEED",
                        help="Lägg till N NYA kärnkraftsreaktorer i en zon med syntetisk "
                             "stokastisk tillgänglighet (RNG-seed), t.ex. "
                             "'SE-S:10:101'. EXTENDABLE — kapaciteten optimeras (implicit "
                             "reaktorstorlek ≈ p_nom_opt/N), tak N×1500 MW. Befintlig flotta "
                             "finns med by default och byter då till syntetisk profil "
                             "(config.nuclear_synth_existing). Kan anges flera gånger. "
                             f"DEFAULT {' '.join(DEFAULT_ADD_NUCLEAR)} (kanonisk baseline); "
                             "--no-add-nuclear bygger ingen ny kärnkraft.")
    parser.add_argument("--no-add-nuclear", action="store_true",
                        help="Stäng av den kanoniska nya kärnkraften (tom --add-nuclear-lista).")
    parser.add_argument("--add-nuclear-fixed", nargs="+", action="extend", default=[], metavar="ZON:N:MW[:SEED]",
                        help="Lägg till N NYA EXOGENA (fasta, ej extendable) reaktorer à MW i en zon "
                             "som EGEN must-run-generator '{zon} nuclear fixed', t.ex. 'SE-S:1:1000' = "
                             "+1 GW. Syntetisk stokastisk tillgänglighet (valfri SEED, annars härledd). "
                             "BÄR verklig annualiserad svk_2040-kapex (inkl. IDC) + VOM — laddas ÄVEN i "
                             "dispatch (konstant i objektivet → synliggör kostnaden). Befintliga flottan "
                             "lämnas orörd. Kan anges flera gånger.")
    parser.add_argument("--nuclear-discount-rate", nargs="+", action="extend", default=[], metavar="ZON:RATE",
                        help="Egen diskontoränta för NY kärnkraft (--add-nuclear) i en zon, "
                             "t.ex. 'SE-N:0.03 SE-S:0.03'. Påverkar bara den extendable expansionens "
                             "annualiserade kapitalkostnad (befintlig flotta är fast). Default = global "
                             "costs.discount_rate. Kan anges flera gånger / som lista.")
    parser.add_argument("--nuclear-min-load", type=float, default=0.6, metavar="FRAC",
                        help="Lastföljande NY kärnkraft: p_min_pu = FRAC × p_max_pu. "
                             "Gäller både expanderbar (--add-nuclear) och exogen fast "
                             "(--add-nuclear-fixed) NY kärnkraft. Befintliga flottan förblir ren "
                             "must-run (nuclear_synth.min_load_frac = 1.0). DEFAULT 0.6 "
                             "(kanonisk baseline); 1.0 gör även ny kärnkraft till ren must-run.")
    parser.add_argument("--add-wind", action="append", default=[], metavar="ZON:MW",
                        help="Lägg till fast landbaserad vindkraft (dispatch, ej extendable), "
                             "t.ex. 'SE-S:9893'. Samma CF-profil som zonens befintliga wind_onshore. "
                             "Energi-motsvarighet till --add-nuclear men variabel. Kan anges flera gånger.")
    parser.add_argument("--add-h2", action="append", default=[], metavar="ZON:DEMAND:EL:STORE[:TURB]",
                        help="Lägg till vätgassystem i en zon (fasta storlekar): baslast DEMAND MW_H2, "
                             "elektrolysör EL MW_el, lager STORE MWh, valfri turbin TURB MW_el. "
                             "T.ex. 'SE-S:500:1000:50000:300'. Kostnader från costs.hydrogen. "
                             "Kan anges flera gånger.")
    parser.add_argument("--add-h2-ext", action="append", default=[], metavar="ZON:DEMAND[:STORE_MAX_MWH]",
                        help="Investerbart vätgassystem (extendable elektrolys + lager, INGEN turbin): "
                             "baslast DEMAND MW_H2, modellen dimensionerar elektrolysör och lager mot "
                             "costs.hydrogen. Valfritt lagertak STORE_MAX_MWH (e_nom_max). "
                             "T.ex. 'SE-N:1000' eller 'DK:1000:1000000' (≤1 TWh). Kan anges flera gånger.")
    parser.add_argument("--add-ev", action="append", default=[], metavar="ZON:N_CARS:N_HEAVY",
                        help="Fordonsladdning (smart charging) i ZON: antal personbilar + tunga "
                             "fordon, t.ex. 'SE-S:2000000:50000'. Per-fordon-tal + profiler från "
                             "config.ev / ev_profiles.parquet. Kan anges flera gånger.")
    parser.add_argument("--effective-ntc", action="store_true",
                        help="Använd effektiv kontinentkapacitet (P80 av faktiska flöden, "
                             "market_connections_effective_mw) istället för märkkapacitet")
    parser.add_argument("--ntc-override", nargs="+", default=None, metavar="Z0:Z1:MW",
                        help="Sätt intern NTC-länk Z0-Z1 till MW (t.ex. SE-S:DK:2007). "
                             "Appliceras EFTER demand-scenariots ntc_overrides → pinnar/ersätter "
                             "en enskild länk. Kan anges flera gånger.")
    parser.add_argument("--market-ntc-override", nargs="+", default=None, metavar="NAMN:MW",
                        help="Sätt kontinentkabel (market_connection) NAMN till MW (t.ex. "
                             "'SE-S DE:1315' för Hansa Power Bridge +700). Appliceras EFTER "
                             "demand-scenariots market_ntc_overrides. Kan anges flera gånger.")
    parser.add_argument("--expand-link", nargs="+", default=None, metavar="Z0:Z1:MEUR[:FLOOR_MW[:MAX_MW]]",
                        help="Gör intern länk Z0-Z1 kapacitetsexpanderbar med overnight-kostnad "
                             "MEUR (M€/MW), annualiseras (40 år). FLOOR_MW=p_nom_min (default = "
                             "byggd p_nom), MAX_MW=p_nom_max (default 30000). T.ex. SE-N:SE-S:2.0:7600. "
                             "Kan anges flera gånger.")
    parser.add_argument("--move-load", nargs="+", default=None, metavar="ZFROM:ZTO:FRAC",
                        help="Flytta en ANDEL (0-1) av ZFROM:s last-timserie till ZTO, profil-troget "
                             "(t.ex. SE-S:SE-N:0.30 för Stockholm-remsan). Bevarar lastform + total "
                             "nordisk last. Kan anges flera gånger.")
    parser.add_argument("--move-nuclear", nargs="+", default=None, metavar="ZFROM:ZTO:MW",
                        help="Flytta MW installerad kärnkraft ZFROM→ZTO (t.ex. SE-S:SE-N:3300 för "
                             "Forsmark). I dispatch-läge kopieras ZFROM:s tillgänglighetsprofil till "
                             "ZTO. Kan anges flera gånger.")
    parser.add_argument("--move-link", nargs="+", default=None, metavar="ZFROM:ZOTHER:ZTO",
                        help="Relokera intern länk ZFROM-ZOTHER så den ansluter ZTO-ZOTHER (t.ex. "
                             "SE-S:FI:SE-N för Fenno-Skan). Slås ihop med ev. befintlig ZTO-ZOTHER-länk. "
                             "Kan anges flera gånger.")
    # Pris-elastisk kontinentgräns är PÅ som default; --no-market-elast stänger av.
    # --market-elasticity behålls (no-op) för bakåtkompatibilitet med äldre kommandon.
    parser.add_argument("--market-elasticity", action="store_true", default=True,
                        dest="market_elasticity",
                        help="(default PÅ) Pris-elastisk kontinentgräns (trappa): stora "
                             "nordiska flöden flyttar gränspriset. S_export/S_import per "
                             "gräns från config market_elasticity.")
    parser.add_argument("--no-market-elast", action="store_false",
                        dest="market_elasticity",
                        help="Stäng av den pris-elastiska kontinentgränsen (fast gränspris).")
    # Fjärrvärmesektorn är PÅ som default; --no-add-heat stänger av.
    # --add-heat behålls (no-op) för bakåtkompatibilitet med äldre kommandon.
    parser.add_argument("--add-heat", action="store_true", default=True,
                        dest="add_heat",
                        help="(default PÅ) Fjärrvärmesektorn (config heat): per-zon heat-buss "
                             "(FV-behov + ackumulator + el-panna + stor-VP + bio/KVV). Drar bort "
                             "dagens FV-el ur AC-lasten. Kräver data/processed/heat_load.parquet.")
    parser.add_argument("--no-add-heat", action="store_false",
                        dest="add_heat",
                        help="Kör utan fjärrvärmesektorn (ren elmodell).")
    parser.add_argument("--heat-store-ext", action="store_true",
                        help="Gör värmeackumulatorn investerbar: modellen dimensionerar lagret "
                             "fritt mot TES-kostnad (config heat.store_overnight_eur_per_kwh) i "
                             "stället för fast store_hours×topplast. Kräver --add-heat.")
    parser.add_argument("--chp-fixed-gw", type=float, default=None, metavar="GW",
                        help="Snabbalternativ till --add-heat: koppla bort hela värmebussen "
                             "(ingen heat-buss/lager/VP/panna) och återinför KVV-elen som ett "
                             "FAST must-run-block om GW (totala toppen). Must-run-termiken reduceras "
                             "med share_of_thermal som i heat-läget; borttagen termik-form skalas till "
                             "GW. Snabbare LP. Ömsesidigt uteslutande med --add-heat.")
    parser.add_argument("--no-tax-heatpower", action="store_true",
                        help="Nollställ energiskatten (heat.el_tax_eur_per_mwh) på FV-elektrifieringen "
                             "(VP + el-panna) → marginalkostnad = bara VOM. Kräver --add-heat. "
                             "Isolerar skattens effekt på elektrifieringsgrad/priser mot baseline.")
    parser.add_argument("--market-scale", default=None, metavar="FACTOR|ZON:F,...",
                        help="Skala kontinentkablars kapacitet. Enskild faktor för alla "
                             "(t.ex. '0.7') eller per zon (t.ex. 'FI:0.5,NO-S:0.8,SE-S:0.6,DK:0.7'). "
                             "Appliceras efter --effective-ntc om båda anges.")
    args = parser.parse_args()
    if args.expansion and args.dispatch:
        parser.error("--expansion och --dispatch är varandras motsatser: den ena "
                     "optimerar kapaciteter, den andra fryser dem.")

    if args.dispatch:                          # återspela källkörningens system-argv
        args = apply_dispatch_replay(parser, args)

    # --cost-scenario/--demand-scenario har numera scenarionamn som default. 'none'
    # (eller tom sträng) är avstängningsvägen tillbaka till dagens kostnader/last.
    for _attr in ("cost_scenario", "demand_scenario"):
        if str(getattr(args, _attr) or "").strip().lower() in ("none", "-", ""):
            setattr(args, _attr, None)

    # --add-nuclear: None = "orörd" → kanonisk lista. Se sentinel-kommentaren vid
    # add_argument (action="extend" gör en icke-tom default farlig).
    if args.no_add_nuclear:
        args.add_nuclear = []
    elif args.add_nuclear is None:
        args.add_nuclear = list(DEFAULT_ADD_NUCLEAR)

    # --vre-curtailment-cost: None = "orörd" → default bara när kapaciteterna är frysta.
    # Måste ligga EFTER apply_dispatch_replay (som sätter base.dispatch/no_expansion) och
    # kan inte vara en argparse-default, eftersom en icke-noll default i expansionsläge
    # skulle få spärren att fälla varje baseline-körning.
    if args.vre_curtailment_cost is None:
        frozen = bool(args.dispatch) or bool(args.no_expansion)
        args.vre_curtailment_cost = DEFAULT_VRE_CURTAILMENT_COST if frozen else 0.0

    # --spill-cost: None = orörd → lägesberoende default (samma sentinel-mönster och samma
    # skäl att den inte kan vara en argparse-default: värdet beror på om kapaciteterna är
    # frysta, vilket avgörs först efter apply_dispatch_replay).
    if args.spill_cost is None:
        frozen = bool(args.dispatch) or bool(args.no_expansion)
        args.spill_cost = (DEFAULT_SPILL_COST_DISPATCH if frozen
                           else DEFAULT_SPILL_COST_EXPANSION)

    # --dispatch = KANONISK DISPATCH (run340:s mall). Att skriva ut --rolling-horizon,
    # --terminal-curve och --no-hydro-price-proxy på varje omdispatch var både ordrikt och
    # felbenäget: glömdes någon av dem föll körningen TYST tillbaka på en cyklisk LP med
    # prisproxyn — tvärtemot avsikten, vilket run319 gick i. Varje del har en av-knapp.
    # Måste ligga EFTER apply_dispatch_replay (som sätter base.dispatch) och FÖRE räckena
    # nedan, så att de validerar den färdigupplösta kombinationen.
    if args.dispatch:
        if not args.no_rolling_horizon:
            args.rolling_horizon = True
        if (not args.no_terminal_curve and args.terminal_curve is None
                and not args.terminal_seasonal and args.terminal_lambda is None):
            args.terminal_curve = ""      # "" → DEFAULT_PARAM_FILE = kalibrerade kurvan
        if not args.hydro_price_proxy:
            args.no_hydro_price_proxy = True

    # None = orörd. Kan inte vara en argparse-default: ett nollskilt värde läses även
    # när rullande horisont är AV, och räcket nedan skulle då fälla varje expansion.
    if args.rolling_lookahead_weeks is None:
        args.rolling_lookahead_weeks = 3 if args.rolling_horizon else 0

    if args.rolling_lookahead_weeks and not args.rolling_horizon:
        parser.error("--rolling-lookahead-weeks kräver --rolling-horizon.")
    if args.rolling_lookahead_weeks < 0:
        parser.error("--rolling-lookahead-weeks måste vara ≥ 0.")

    if args.terminal_curve is not None:
        # Utan rullande horisont finns inget fönsterslut att sätta kurvan i: en enda
        # cyklisk LP har bara ETT slut, och där är SOC dessutom bunden av cykliciteten.
        if not args.rolling_horizon:
            parser.error("--terminal-curve kräver --rolling-horizon: kurvan verkar i "
                         "FÖNSTERSLUT, och en enda cyklisk LP har inget fritt sådant.")
        if not args.no_hydro_price_proxy:
            parser.error("--terminal-curve kräver --no-hydro-price-proxy: med proxyn på är "
                         "hydrons marginal_cost det historiska zonpriset, marginalen krymper "
                         "till nettovattenvärdet, och λ på bruttoprisnivå övervärderar lagring.")
        if args.terminal_seasonal:
            parser.error("--terminal-curve och --terminal-seasonal sätter BÅDA λ:s "
                         "säsongsform (A(w) respektive S(m)) och skulle multipliceras på "
                         "varandra. Välj en — kurvan är efterföljaren.")

    if args.rolling_horizon:
        # Varje fönster är en EGEN LP → investeringsbesluten skulle fattas oberoende
        # per fönster (och dimensioneras mot fönstrets eget väder). Inkoherent.
        if not (args.no_expansion or args.dispatch):
            parser.error("--rolling-horizon är dispatch-only: varje fönster löses som en egen "
                         "LP, så extendable kapaciteter skulle optimeras oberoende per fönster. "
                         "Kör med --no-expansion, eller tvåpass: expansion först, sedan "
                         "--dispatch <label> --rolling-horizon.")
        if not args.no_hydro_price_proxy:
            print("⚠️  --rolling-horizon MED vattenvärdes-proxyn påslagen: hydros marginal_cost "
                  "är det historiska zonpriset, så släppmarginalen är nettovattenvärdet (~18–47) "
                  "medan terminal-λ ligger på bruttoprisnivå (~73). Reservoarerna väntas hamstra "
                  "mot 100 % — det var felet i run91–93. Lägg till --no-hydro-price-proxy.")

    if args.no_voll:                      # tillbaka till slack enbart i icke-marknadszoner
        args.voll = None

    if args.soc_pin_from:
        if args.soc_pin:
            parser.error("--soc-pin-from och --soc-pin är ömsesidigt uteslutande "
                         "(fönstervis bana vs fasta ändpunkter för hela körningen)")
        if not args.dispatch:
            print("  ⚠️ --soc-pin-from utan --dispatch: kapaciteterna fryses INTE mot "
                  f"{args.soc_pin_from} — pinnad SOC mot en annan flotta kan bli infeasible")

    soc_pin_start = {}   # zon → start-fraktion (→ soc_initial_override)
    soc_pin_end = {}     # zon → slut-fraktion (→ terminal-pin callback)
    for spec in args.soc_pin:
        for item in spec.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 3:
                print(f"Ogiltigt --soc-pin format: '{item}' (förväntat ZON:START:END)")
                sys.exit(1)
            z = parts[0].strip()
            soc_pin_start[z] = float(parts[1])
            soc_pin_end[z] = float(parts[2])

    # Batterier samlas som tupler (zon, p_nom_mw, max_hours, extendable[, costed]).
    # --add-battery: costed=True → bär svk_2040+IDC-kapex även fast (dispatch). Baseline/
    # scenario/--battery lämnas 4-tupler (costed default False).
    batteries = []
    for spec in args.add_battery:   # bakåtkompatibel: --add-battery + --battery-extendable
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Ogiltigt --add-battery format: '{spec}' (förväntat ZON:MW:HOURS)")
            sys.exit(1)
        batteries.append((parts[0].strip(), float(parts[1]), float(parts[2]),
                          bool(args.battery_extendable), True))
    # --battery DURATION [ZON:MW ...]: utan zoner = expanderbart i ALLA zoner;
    # med zoner = fasta storlekar i de angivna, 0 i övriga. (Zoner expanderas
    # efter att cfg lästs in.)
    battery_hours = None
    battery_fixed = None   # None = global expandable; list = fasta (zon, mw)
    if args.battery:
        battery_hours = float(str(args.battery[0]).rstrip("hH"))
        rest = ":".join(args.battery[1:])
        if rest:
            toks = [t for t in rest.split(":") if t != ""]
            if len(toks) % 2 != 0:
                print(f"Ogiltigt --battery zonformat: '{rest}' (förväntat ZON:MW:ZON:MW ...)")
                sys.exit(1)
            battery_fixed = [(toks[i].strip(), float(toks[i + 1])) for i in range(0, len(toks), 2)]

    # --add-nuclear ZON:N:SEED → N nya reaktorer, syntetisk tillgänglighet (seed),
    # EXTENDABLE (kapacitet optimeras). Befintlig flotta finns med by default; när
    # --add-nuclear anges går befintlig flotta också över till syntetisk profil.
    extra_nuclear = []
    for spec in args.add_nuclear:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Ogiltigt --add-nuclear format: '{spec}' (förväntat ZON:N:SEED — "
                  f"N nya reaktorer + RNG-seed; syntetisk, extendable)")
            sys.exit(1)
        extra_nuclear.append((parts[0].strip(), int(parts[1]), int(parts[2])))

    # --add-nuclear-fixed ZON:N:MW → N fasta (exogena) reaktorer à MW, must-run, sammanvävda
    # med befintliga flottans syntetiska profil (heterogen reaktorlista). {zon: [(N, MW), ...]}.
    fixed_nuclear: dict = {}
    for spec in args.add_nuclear_fixed:
        parts = spec.split(":")
        if len(parts) not in (3, 4):
            print(f"Ogiltigt --add-nuclear-fixed format: '{spec}' (förväntat ZON:N:MW[:SEED] — "
                  f"N fasta reaktorer à MW, exogen must-run, costed svk_2040)")
            sys.exit(1)
        seed = int(parts[3]) if len(parts) == 4 else None
        fixed_nuclear.setdefault(parts[0].strip(), []).append((int(parts[1]), float(parts[2]), seed))

    extra_wind = []
    for spec in args.add_wind:
        parts = spec.split(":")
        if len(parts) != 2:
            print(f"Ogiltigt --add-wind format: '{spec}' (förväntat ZON:MW)")
            sys.exit(1)
        extra_wind.append((parts[0].strip(), float(parts[1])))

    hydrogen_overrides = {}
    for spec in args.add_h2:
        parts = spec.split(":")
        if len(parts) not in (4, 5):
            print(f"Ogiltigt --add-h2 format: '{spec}' (förväntat ZON:DEMAND:EL:STORE[:TURB])")
            sys.exit(1)
        zone = parts[0].strip()
        hz = {
            "demand_mw":    float(parts[1]),
            "electrolyser": {"p_nom_mw": float(parts[2]), "extendable": False},
            "store":        {"e_nom_mwh": float(parts[3]), "extendable": False},
        }
        if len(parts) == 5 and float(parts[4]) > 0:
            hz["turbine"] = {"p_nom_mw": float(parts[4]), "extendable": False}
        hydrogen_overrides[zone] = hz

    for spec in args.add_h2_ext:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            print(f"Ogiltigt --add-h2-ext format: '{spec}' (förväntat ZON:DEMAND[:STORE_MAX_MWH])")
            sys.exit(1)
        store = {"e_nom_mwh": 0.0, "extendable": True}
        if len(parts) == 3:                       # valfritt lagertak (e_nom_max)
            store["e_nom_max_mwh"] = float(parts[2])
        hydrogen_overrides[parts[0].strip()] = {
            "demand_mw":    float(parts[1]),
            "electrolyser": {"p_nom_mw": 0.0, "extendable": True},
            "store":        store,
        }

    ev_overrides = {}
    for spec in args.add_ev:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Ogiltigt --add-ev format: '{spec}' (förväntat ZON:N_CARS:N_HEAVY)")
            sys.exit(1)
        ev_overrides[parts[0].strip()] = {"car": float(parts[1]), "heavy": float(parts[2])}

    onshore_caps = {}
    for spec in args.onshore_cap:
        parts = spec.split(":")
        if len(parts) != 2:
            print(f"Ogiltigt --onshore-cap format: '{spec}' (förväntat ZON:MW)")
            sys.exit(1)
        onshore_caps[parts[0].strip()] = float(parts[1])

    cfg = load_config()
    res = args.resolution or cfg["snapshots"].get("resolution_hours", 1)

    # --solver-option KEY=VALUE: skriv över cfg["solver"] (allt utom "name" går rakt
    # till solver_options i solve()). Typ tolkas som i YAML så att t.ex. -6 blir int.
    for spec in args.solver_option:
        if "=" not in spec:
            print(f"Ogiltigt --solver-option format: '{spec}' (förväntat KEY=VALUE)")
            sys.exit(1)
        k, v = spec.split("=", 1)
        k = k.strip()
        raw = v.strip()
        val = yaml.safe_load(raw)                # '-6' → int, '1e-6' → float
        if isinstance(val, bool) and raw.lower() not in ("true", "false"):
            val = raw                            # YAML 1.1 gör 'on'/'off' till bool — HiGHS vill ha strängen
        old = cfg["solver"].get(k, "(ej satt)")
        cfg["solver"][k] = val
        print(f"  → solver-option {k}: {old} → {val!r}")

    # --soc-pin: start-fraktion → MWh (soc_initial_override förväntas i MWh, ej fraktion)
    soc_pin_start = {z: f * cfg["zones"][z]["hydro_p_nom_mw"] * cfg["zones"][z]["hydro_max_hours"]
                     for z, f in soc_pin_start.items()}

    # Expandera --battery nu när zonlistan är känd
    if battery_hours is not None:
        if battery_fixed is None:          # global: expanderbart i varje zon
            batteries += [(z, 0.0, battery_hours, True) for z in cfg["zones"]]
        else:                              # fasta storlekar i angivna zoner
            batteries += [(z, mw, battery_hours, False) for z, mw in battery_fixed]

    # Extra last: nollställ alltid config-värden; applicera --extra-load om givet
    cfg["additional_load_mw"] = {}
    if args.extra_load:
        for z in cfg["zones"]:
            cfg["additional_load_mw"][z] = args.extra_load
    for tok in args.extra_load_zone:
        z, mw = tok.split(":")
        cfg["additional_load_mw"][z] = cfg["additional_load_mw"].get(z, 0.0) + float(mw)

    if args.no_expansion:
        for tech in cfg.get("costs", {}):
            if isinstance(cfg["costs"][tech], dict):
                cfg["costs"][tech]["extendable"] = False
    else:
        # Expansionskörningar modellerar framtida system → post-utbyggnads-NTC
        # (t.ex. Aurora SE-N–FI). Dispatch behåller historiska värden i cfg["links"].
        overrides = {(z0, z1): p for z0, z1, p in cfg.get("links_expansion_overrides", [])}
        for link in cfg.get("links", []):
            new = overrides.get((link[0], link[1]))
            if new is not None and link[2] != new:
                print(f"  → expansion-NTC: {link[0]}-{link[1]} {link[2]} → {new} MW")
                link[2] = new

    if args.cost_scenario:
        apply_cost_scenario(cfg, args.cost_scenario)

    scenario_battery = None
    if args.scenario_battery:
        try:
            _gw, _h = args.scenario_battery.split(":")
            scenario_battery = (float(_gw), float(_h))
        except ValueError:
            raise SystemExit(f"Ogiltigt --scenario-battery '{args.scenario_battery}' (förväntat GW:HOURS)")
        if not args.demand_scenario:
            raise SystemExit("--scenario-battery kräver --demand-scenario")

    demand_pnom_max, demand_pnom_min = {}, {}
    if args.demand_scenario:
        demand_pnom_max, demand_pnom_min = apply_demand_scenario(
            cfg, args.demand_scenario, hydrogen_overrides, ev_overrides, batteries,
            scenario_battery=scenario_battery)
    else:
        # Dagens värld (inget demand-scenario): ladda dagens FRIA baseline-batterier
        # (costed=False). I demand-scenario-läge representeras flottan av scenario-
        # batterierna → baseline hoppas över (undviker dubbelräkning).
        base_bats = cfg.get("baseline_batteries", [])
        for z, mw, h in base_bats:
            batteries.append((z, float(mw), float(h), False, False))
        if base_bats:
            print("  Baseline-batterier (fria): "
                  + ", ".join(f"{z} {mw:.0f}MW/{h:.0f}h" for z, mw, h in base_bats))

    if args.ntc_override:
        for spec in args.ntc_override:
            z0, z1, mw = spec.split(":")
            mw = float(mw)
            hit = False
            for link in cfg.get("links", []):
                if link[0] == z0 and link[1] == z1:
                    print(f"  → NTC-override {z0}-{z1}: {link[2]} → {mw:.0f} MW (CLI)")
                    link[2] = mw
                    hit = True
            if not hit:
                raise SystemExit(f"--ntc-override: hittade ingen intern länk {z0}-{z1} i cfg['links']")

    if args.market_ntc_override:
        for spec in args.market_ntc_override:
            name, mw = spec.rsplit(":", 1)
            mw = float(mw)
            hit = False
            for mc in cfg.get("market_connections", []):
                if mc[0] == name:
                    print(f"  → Marknads-NTC-override {name}: {mc[2]} → {mw:.0f} MW (CLI)")
                    mc[2] = mw
                    hit = True
            if not hit:
                raise SystemExit(f"--market-ntc-override: hittade ingen kontinentkabel '{name}' "
                                 f"i cfg['market_connections']")

    if args.no_market:
        cfg["market_connections"] = []

    if args.spill_cost is not None:
        cfg["costs"]["hydro"]["spill_cost_eur_per_mwh"] = args.spill_cost
        print(f"  → hydro spill_cost = {args.spill_cost} EUR/MWh")

    if args.market_elasticity:
        cfg.setdefault("market_elasticity", {})["enabled"] = True

    if args.grid_cost:
        grd = cfg["costs"].setdefault("grid", {})
        grd["active"] = True
        zf = grd.get("zone_factor") or {}
        print("  → NÄTKOSTNAD aktiv: djup nätförstärkning per kraftslag × zon, "
              "INKREMENT-only (bara tillkommande effekt), exkl. KVV. €/W:")
        for c, oc in (grd.get("cost_eur_per_w") or {}).items():
            print(f"      {c:13s} {oc:.2f} €/W  (× zon-faktor {zf})")

    if args.nuclear_discount_rate:
        disc = {}
        for spec in args.nuclear_discount_rate:
            parts = spec.split(":")
            if len(parts) != 2:
                parser.error(f"Ogiltigt --nuclear-discount-rate format: '{spec}' (förväntat ZON:RATE)")
            disc[parts[0].strip()] = float(parts[1])
        cfg["costs"].setdefault("nuclear", {})["discount_rate_by_zone"] = disc
        print("  → kärnkrafts-diskontoränta (ny/expansion): "
              + ", ".join(f"{z} {r:.0%}" for z, r in disc.items()))

    if args.offwind_discount_rate:
        disc = {}
        for spec in args.offwind_discount_rate:
            parts = spec.split(":")
            if len(parts) != 2:
                parser.error(f"Ogiltigt --offwind-discount-rate format: '{spec}' (förväntat ZON:RATE)")
            disc[parts[0].strip()] = float(parts[1])
        cfg["costs"].setdefault("wind_offshore", {})["discount_rate_by_zone"] = disc
        print("  → havsvind-diskontoränta: "
              + ", ".join(f"{z} {r:.0%}" for z, r in disc.items()))

    if args.add_heat and args.chp_fixed_gw is not None:
        parser.error("--chp-fixed-gw och --add-heat är ömsesidigt uteslutande "
                     "(KVV-fast kopplar bort värmebussen).")

    if args.no_tax_heatpower and not args.add_heat:
        parser.error("--no-tax-heatpower kräver --add-heat (skatten sitter på värmebussens VP/el-panna).")

    if args.add_heat:
        cfg.setdefault("heat", {})["enabled"] = True
        if args.heat_store_ext:
            cfg["heat"]["store_extendable"] = True
        if args.no_tax_heatpower:
            old_tax = float(cfg["heat"].get("el_tax_eur_per_mwh", 0.0))
            cfg["heat"]["el_tax_eur_per_mwh"] = 0.0
            print(f"  → elskatt på VP/el-panna nollställd ({old_tax:g} → 0 €/MWh)")
        print(f"  → fjärrvärmesektor aktiv"
              + (" (värmelager extendable mot TES-kostnad)" if args.heat_store_ext else ""))

    if args.chp_fixed_gw is not None:
        # KVV-fast: aktivera heat.enabled (→ _add_thermal reducerar termik med share_of_thermal)
        # men ingen värmebuss byggs (_heat_demand_profiles returnerar {} när chp_fixed_gw satt).
        cfg.setdefault("heat", {})["enabled"] = True
        cfg["heat"]["chp_fixed_gw"] = args.chp_fixed_gw
        print(f"  → KVV-fast aktiv: {args.chp_fixed_gw:g} GW must-run-el, värmebuss bortkopplad")

    if ev_overrides:
        cfg.setdefault("ev", {})["enabled"] = True
        print(f"  → fordonsladdning aktiv i {', '.join(ev_overrides)}")

    if args.effective_ntc:
        eff = cfg.get("market_connections_effective_mw", {})
        for mc in cfg.get("market_connections", []):
            if mc[0] in eff:
                mc[2] = eff[mc[0]]
        print(f"  → effektiv kontinentkapacitet (P80) aktiv: {len(eff)} kablar")

    if args.market_scale is not None:
        if ":" in args.market_scale:
            # Per zon: "FI:0.5,NO-S:0.8,..."
            zone_factors = {}
            for pair in args.market_scale.split(","):
                z, f = pair.split(":")
                zone_factors[z.strip()] = float(f)
            for mc in cfg.get("market_connections", []):
                if mc[1] in zone_factors:
                    mc[2] = mc[2] * zone_factors[mc[1]]
            print(f"  → kontinentkablar skalade per zon: "
                  + ", ".join(f"{z}×{f}" for z, f in zone_factors.items()))
        else:
            factor = float(args.market_scale)
            for mc in cfg.get("market_connections", []):
                mc[2] = mc[2] * factor
            print(f"  → kontinentkablar skalade ×{factor}: "
                  f"{len(cfg.get('market_connections', []))} kablar")

    flags = []
    if args.dispatch:                   flags.append(f"dispatch-{args.dispatch}")
    if args.low_hydro is not None:      flags.append(f"lowhydro-{args.low_hydro:g}")
    if args.vre_curtailment_cost:       flags.append(f"vrecurt-{args.vre_curtailment_cost:g}")
    # Alltid med: defaulten är lägesberoende, så utan raden går en körnings spillkostnad
    # bara att läsa ur console.log eller gissa ur källans ålder.
    flags.append(f"spill-{args.spill_cost:g}")
    if args.rolling_horizon:            flags.append(f"rolling-{args.rolling_weeks}w")
    if args.rolling_lookahead_weeks:    flags.append(f"lookahead-{args.rolling_lookahead_weeks}w")
    if args.terminal_seasonal:          flags.append("term-seasonal")
    if args.terminal_curve is not None: flags.append("termkurva")
    if args.terminal_lambda:            flags.append(f"termlambda-{args.terminal_lambda.replace(':','_')}")
    if args.terminal_lambda_scale:      flags.append(f"termscale-{args.terminal_lambda_scale.replace(':','_')}")
    if args.rolling_horizon and args.terminal_curve is None:
        _p = (args.terminal_lambda_profile.split(",") if args.terminal_lambda_profile
              else [f"{p:g}" for p in DEFAULT_TERMINAL_PROFILE])
        flags.append("termprofil-" + "_".join(x.strip() for x in _p))
    if args.extra_load:                 flags.append(f"extra-load-{args.extra_load:.0f}mw")
    for tok in args.extra_load_zone:    flags.append("xload-" + tok.replace(":", "-") + "mw")
    if args.cost_scenario:              flags.append(f"cost-{args.cost_scenario}")
    if args.demand_scenario:            flags.append(f"demand-{args.demand_scenario}")
    if args.no_expansion:               flags.append("no-expansion")
    if args.no_market:                  flags.append("no-market")
    if args.effective_ntc:              flags.append("effective-ntc")
    if args.ntc_override:               flags.append("ntc-override-" + "_".join(args.ntc_override))
    if args.market_ntc_override:        flags.append("market-ntc-" + "_".join(args.market_ntc_override))
    if args.expand_link:                flags.append("expand-link-" + "_".join(args.expand_link).replace(":", "_"))
    if args.add_oc_scale:               flags.append("ocscale-" + "_".join(args.add_oc_scale).replace(":", ""))
    for spec in args.add_solar:         flags.append(f"solar-{spec.replace(':','_')}")
    for spec in args.add_onshore:       flags.append(f"onshore-{spec.replace(':','_')}")
    for spec in args.add_offshore:      flags.append(f"offshore-{spec.replace(':','_')}")
    if args.add_cost_scenario != "svk_2040": flags.append(f"addcost-{args.add_cost_scenario}")
    if args.move_load:                  flags.append("move-load-" + "_".join(args.move_load))
    if args.move_nuclear:               flags.append("move-nuclear-" + "_".join(args.move_nuclear))
    if args.move_link:                  flags.append("move-link-" + "_".join(args.move_link))
    if not args.market_elasticity:      flags.append("no-market-elast")
    if args.add_heat:                   flags.append("heat-store-ext" if args.heat_store_ext else "heat")
    if args.no_tax_heatpower:           flags.append("no-tax-heatpower")
    if args.chp_fixed_gw is not None:   flags.append(f"chpfixed-{args.chp_fixed_gw:g}gw")
    if args.market_scale is not None:   flags.append("market-scale-" + args.market_scale.replace(":", "").replace(",", "_"))
    if args.voll is not None:           flags.append(f"voll{int(args.voll)}")
    if args.no_hydro_price_proxy:       flags.append("no-hydro-price-proxy")
    for spec in args.add_battery:       flags.append(f"battery-{spec.replace(':','_')}")
    if args.battery_extendable:         flags.append("battery-ext")
    if args.battery:                    flags.append("battery-" + "_".join(args.battery).replace(":", "_"))
    if args.scenario_battery:           flags.append(f"scenbatt-{args.scenario_battery.replace(':','_')}")
    for z in args.expand_vre:           flags.append(f"expand-vre-{z}")
    if args.expand_budget_musd:         flags.append(f"oc-budget-{args.expand_budget_musd:.0f}musd")
    if args.expand_budget_meur:         flags.append(f"oc-budget-{args.expand_budget_meur:.0f}meur")
    if args.onwind_capfac_increase:     flags.append(f"onwind-cf+{args.onwind_capfac_increase:.2f}")
    if args.offwind_capfac_increase:    flags.append(f"offwind-cf+{args.offwind_capfac_increase:.2f}")
    for spec in args.add_nuclear:       flags.append(f"nuclear-{spec.replace(':','_')}")
    for spec in args.add_nuclear_fixed: flags.append(f"nucfix-{spec.replace(':','_')}")
    for spec in args.nuclear_discount_rate: flags.append(f"nucdisc-{spec.replace(':','_')}")
    for spec in args.offwind_discount_rate: flags.append(f"offdisc-{spec.replace(':','_')}")
    for spec in args.add_wind:          flags.append(f"wind-{spec.replace(':','_')}")
    for spec in args.add_h2:            flags.append(f"h2-{spec.replace(':','_')}")
    for spec in args.add_h2_ext:        flags.append(f"h2ext-{spec.replace(':','_')}")
    for spec in args.add_ev:            flags.append(f"ev-{spec.replace(':','_')}")
    for spec in args.onshore_cap:       flags.append(f"onshorecap-{spec.replace(':','_')}")
    if args.onshore_lower != 1.0:       flags.append(f"onshorelow-{args.onshore_lower:g}")
    if args.solar_cap is not None:      flags.append(f"solarcap-{args.solar_cap:.0f}")
    if args.grid_cost:                  flags.append("grid-cost")
    if (args.hydro_restrictions or args.hydro_min_hourly is not None
            or args.hydro_min_daily is not None or args.hydro_max_weekly is not None
            or args.hydro_bypass_spill is not None):
        flags.append("hydro-restrictions")
    if soc_pin_end:                     flags.append("soc-pin-" + "_".join(soc_pin_end.keys()))
    if args.soc_pin_from:               flags.append(f"soc-pin-from-{args.soc_pin_from}@{args.soc_pin_freq}")
    if args.nuclear_min_load is not None: flags.append(f"nucminload-{args.nuclear_min_load:g}")
    for spec in args.solver_option:     flags.append(f"solveropt-{spec.replace('=','')}")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"Konfiguration: upplösning={res}h, år={args.year or '2023-2025'}{flag_str}")

    inputs    = load_inputs(cfg)
    if args.onwind_capfac_increase:
        inputs["vre_profiles"] = boost_onshore_capfac(
            inputs["vre_profiles"], args.onwind_capfac_increase)
    if args.offwind_capfac_increase:
        inputs["vre_profiles"] = boost_offshore_capfac(
            inputs["vre_profiles"], args.offwind_capfac_increase)

    # Kontinentpris-skift (--demand-scenario ...continent_price_eur_mwh): multiplikativ
    # omskalning av 2023-25-serien per bzn → SvK 2040-nivå (timform bevaras, nivå byts).
    if args.demand_scenario:
        cps = (cfg.get("demand_scenarios", {}).get(args.demand_scenario, {})
               .get("continent_price_eur_mwh") or {})
        for bzn, target in cps.items():
            if bzn in inputs["market_prices"]:
                m = inputs["market_prices"][bzn].mean()
                if m > 0:
                    inputs["market_prices"][bzn] = inputs["market_prices"][bzn] * (float(target) / m)
                    print(f"  → kontinentpris {bzn}: snitt {m:.1f} → {float(target):.0f} €/MWh (×{float(target)/m:.2f})")

    # Dygnssväng-komprimering (--continent-diurnal-scale): 2040-kontinentlagring plattar den
    # deterministiska sol-formade dygnsspreaden. Dela varje ventil-bzn i nivå + hour-of-day-
    # komponent + residual (väder); skala BARA hour-of-day-komponenten. Nivå + residual bevaras.
    dsc = float(args.continent_diurnal_scale)
    if dsc != 1.0:
        valve_bzn = {mc[3] for mc in cfg.get("market_connections", [])}   # ej zon-priser (hydro)
        for bzn in valve_bzn:
            s = inputs["market_prices"].get(bzn)
            if s is None:
                continue
            om   = s.mean()
            hodm = s.groupby(s.index.hour).transform("mean")   # hour-of-day-medel alignat
            sw0  = float(hodm.max() - hodm.min())
            inputs["market_prices"][bzn] = om + dsc * (hodm - om) + (s - hodm)
            print(f"  → kontinent dygnssväng {bzn}: {sw0:.0f} → {sw0*dsc:.0f} €/MWh (×{dsc}), nivå {om:.1f} bevarad")

    snapshots = make_snapshots(cfg, res, args.year)
    inputs    = resample_inputs(inputs, snapshots, res)

    if args.output:
        label = args.output
    else:
        label = f"res{res}h_{'_'.join(str(s.year) for s in [snapshots[0], snapshots[-1]])}"
        if args.year:
            label = f"res{res}h_{args.year}"

    # Icke-cyklisk om SOC-ändpunkterna pinnas (--soc-pin); annars cyklisk
    cyclic_soc = not soc_pin_end

    # Kärnkraft: befintlig flotta (zones.nuclear_p_nom_mw) + ev. ny via --add-nuclear.
    # active=True (--add-nuclear angivet) → expansionsläge: befintlig flotta blir fast
    # + syntetisk (nuclear_synth_existing), nya reaktorer expanderas (_add_extra_nuclear).
    # active = expansionsläge (--add-nuclear): befintlig flotta blir fast+syntetisk.
    # --add-nuclear-fixed triggar INTE detta längre — den byggs som egen costed generator
    # (_add_fixed_nuclear) och lämnar befintliga flottan orörd (faktisk profil).
    synth_params = dict(cfg.get("nuclear_synth", {}))   # kopia: mutera ej configen
    if args.nuclear_min_load is not None:
        synth_params["min_load_frac_exp"] = args.nuclear_min_load
        print(f"Ny kärnkraft LASTFÖLJANDE: p_min_pu = {args.nuclear_min_load:g} × p_max_pu "
              f"(befintlig flotta oförändrat must-run)")
    synthetic_nuclear = {
        "existing": cfg.get("nuclear_synth_existing", {}),
        "params":   synth_params,
        "active":   bool(extra_nuclear),
        "fixed":    fixed_nuclear,
    }
    if extra_nuclear or fixed_nuclear:
        ex = cfg.get("nuclear_synth_existing", {})
        print(f"Kärnkraft: befintlig flotta {'syntetisk ' + str(list(ex)) if extra_nuclear else 'orörd (faktisk profil)'}; "
              f"nya extendable {[(z, n, s) for z, n, s in extra_nuclear]}; "
              f"nya FASTA (costed svk_2040) {dict(fixed_nuclear)} "
              f"(target_cf={synthetic_nuclear['params'].get('target_cf', 0.85)})")

    # Zon-omdragning (komposerbara spakar; appliceras på inputs/cfg före build_network).
    # Bevarar total nordisk last/kärnkraft — flyttar bara MW/andel mellan zon-etiketter.
    if args.move_load:
        ld = inputs["load"]
        for spec in args.move_load:
            zf, zt, frac = spec.split(":"); frac = float(frac)
            if zf not in ld.columns or zt not in ld.columns:
                raise SystemExit(f"--move-load: okänd zon i '{spec}' (finns: {list(ld.columns)})")
            moved = frac * ld[zf]
            ld[zt] = ld[zt] + moved          # ZTO först (från ursprunglig ZFROM-serie)
            ld[zf] = ld[zf] * (1.0 - frac)   # sedan skala ned ZFROM
            print(f"  → move-load {zf}→{zt}: {frac:.0%} av {zf}-last "
                  f"({moved.mean():.0f} MW snitt) profil-troget flyttad")

    if args.move_nuclear:
        if synthetic_nuclear["active"]:
            raise SystemExit("--move-nuclear stöds bara i dispatch-läge (ej --add-nuclear/synth); "
                             "synth-läget kräver en nuclear_synth_existing-post för målzonen.")
        nprof = inputs["nuclear_profile"]
        for spec in args.move_nuclear:
            zf, zt, mw = spec.split(":"); mw = float(mw)
            cur = cfg["zones"].get(zf, {}).get("nuclear_p_nom_mw", 0)
            if mw > cur:
                raise SystemExit(f"--move-nuclear: {zf} har bara {cur} MW, kan ej flytta {mw:.0f}")
            prev_zt = cfg["zones"].setdefault(zt, {}).get("nuclear_p_nom_mw", 0)
            cfg["zones"][zt]["nuclear_p_nom_mw"] = prev_zt + mw
            cfg["zones"][zf]["nuclear_p_nom_mw"] = cur - mw
            # Tillgänglighetsprofil (0-1) för målzonen. Obs: kolumnen kan redan finnas
            # (t.ex. SE-N = nollor) → sätt den ALLTID, inte bara vid saknad kolumn.
            if prev_zt <= 0 or zt not in nprof.columns:
                nprof[zt] = nprof[zf]                              # målet ärver källans form
            else:
                nprof[zt] = (nprof[zt] * prev_zt + nprof[zf] * mw) / (prev_zt + mw)  # kap-viktad
            print(f"  → move-nuclear {zf}→{zt}: {mw:.0f} MW ({zf} {cur}→{cur-mw}, "
                  f"{zt} {prev_zt}→{cfg['zones'][zt]['nuclear_p_nom_mw']:.0f}); "
                  f"profil {'ärvd från' if prev_zt <= 0 else 'blandad med'} {zf}")

    if args.move_link:
        for spec in args.move_link:
            zf, zo, zt = spec.split(":")
            src = next((l for l in cfg["links"] if {l[0], l[1]} == {zf, zo}), None)
            if src is None:
                raise SystemExit(f"--move-link: hittade ingen länk {zf}-{zo} i cfg['links']")
            mw = src[2]; cfg["links"].remove(src)
            dst = next((l for l in cfg["links"] if {l[0], l[1]} == {zt, zo}), None)
            if dst is not None:
                dst[2] += mw
                print(f"  → move-link {zf}-{zo} → {zt}-{zo}: {mw:.0f} MW hopslagen "
                      f"(→ {zt}-{zo} = {dst[2]:.0f} MW)")
            else:
                cfg["links"].append([zt, zo, mw])
                print(f"  → move-link {zf}-{zo} → {zt}-{zo}: {mw:.0f} MW (ny länk)")

    add_oc_scale = {}
    for spec in (args.add_oc_scale or []):
        tech, fac = spec.split(":")
        add_oc_scale[tech.strip()] = float(fac)
    if add_oc_scale:
        print(f"  OC-skala för costade tillägg: {add_oc_scale}")

    onshore_adds = []
    for spec in args.add_onshore:
        parts = spec.split(":")
        z, mw = parts[0], parts[1]
        uplift = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        onshore_adds.append((z.strip(), float(mw), uplift))

    offshore_adds = []
    for spec in args.add_offshore:
        parts = spec.split(":")
        z, mw = parts[0], parts[1]
        uplift = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        offshore_adds.append((z.strip(), float(mw), uplift))

    solar_adds = []
    for spec in args.add_solar:
        z, mw = spec.split(":")
        solar_adds.append((z.strip(), float(mw)))

    print(f"Bygger nätverk ({len(snapshots)} tidssteg) ...")
    n = build_network(cfg, snapshots, **inputs,
                      cyclic_soc=cyclic_soc,
                      voll=args.voll,
                      batteries=batteries,
                      extra_nuclear=extra_nuclear,
                      extra_wind=extra_wind,
                      synthetic_nuclear=synthetic_nuclear,
                      soc_initial_override=soc_pin_start or None,
                      hydrogen_overrides=hydrogen_overrides or None,
                      ev_overrides=ev_overrides or None,
                      add_oc_scale=add_oc_scale or None,
                      solar_adds=solar_adds or None,
                      onshore_adds=onshore_adds or None,
                      offshore_adds=offshore_adds or None,
                      hydro_price_proxy=not args.no_hydro_price_proxy,
                      add_cost_scenario=args.add_cost_scenario)

    if args.low_hydro is not None:             # torrårs-scenario: skala 2024 hydro nedåt
        apply_low_hydro(n, args.low_hydro)

    # Riktad VRE-expansion + OC-budget (bara angivna zoner, oavsett --no-expansion)
    extra_callbacks = []
    if args.expand_vre:
        n_years = len(snapshots) * res / 8760.0
        forced  = (args.expand_budget_musd is not None
                   or args.expand_budget_meur is not None)   # likhets-budget → dispatcheffekt
        print(f"Riktad VRE-expansion i: {', '.join(args.expand_vre)}"
              + (" (TVINGAD budget, kapital=0)" if forced else ""))
        make_vre_extendable(n, cfg, args.expand_vre, n_years,
                            capital_in_objective=not forced)
        if forced:
            # batteriets kapitalkostnad nollas också (sunk) i de berörda zonerna
            for bname, su in n.storage_units.iterrows():
                if su.bus in set(args.expand_vre) and su.carrier == "battery":
                    n.storage_units.at[bname, "capital_cost"] = 0.0
            if args.expand_budget_meur is not None:
                budget_eur = args.expand_budget_meur * 1e6
                print(f"  OC-budget: {args.expand_budget_meur:.0f} MEUR "
                      f"= {budget_eur/1e9:.2f} mdr€ (likhet)")
            else:
                budget_eur = args.expand_budget_musd * 1e6 * USD_TO_EUR
                print(f"  OC-budget: {args.expand_budget_musd:.0f} MUSD "
                      f"= {budget_eur/1e9:.2f} mdr€ (vid {USD_TO_EUR} USD/EUR, likhet)")
            extra_callbacks.append(
                oc_budget_constraint(cfg, args.expand_vre, budget_eur, equality=True))

    # Endogen snitt-expansion (--expand-link): gör intern NTC-länk kapacitetsexpanderbar
    if args.expand_link:
        n_years_link = len(snapshots) * res / 8760.0
        print("Endogen länk-expansion:")
        for spec in args.expand_link:
            parts = spec.split(":")
            if len(parts) < 3:
                raise SystemExit(f"--expand-link: ogiltig spec '{spec}' (kräver Z0:Z1:MEUR[:FLOOR[:MAX]])")
            z0, z1, meur = parts[0], parts[1], float(parts[2])
            floor = float(parts[3]) if len(parts) > 3 else None
            pmax  = float(parts[4]) if len(parts) > 4 else 30000.0
            make_link_extendable(n, cfg, f"{z0}-{z1}", meur, n_years_link,
                                 p_nom_min=floor, p_nom_max=pmax)

    # Nätkostnad (--grid-cost): egen objektiv-term på INKREMENTET för alla extendable
    # enheter (inkl. ny kärnkraft/vind), exkl. KVV. I tvingat budgetläge går grid via
    # OC-budgeten istället (kapital sunk i objektivet) → lägg ej till termen då.
    _forced = bool(args.expand_vre) and (args.expand_budget_musd is not None
                                         or args.expand_budget_meur is not None)
    if args.grid_cost and not _forced:
        n_years_grid = len(snapshots) * res / 8760.0
        extra_callbacks.append(grid_cost_objective(cfg, n_years_grid))

    # Driftrestriktioner på reservoarvattenkraften (--hydro-restrictions).
    ocfg = dict(cfg.get("hydro_operation") or {})
    _hydro_flags = (args.hydro_min_hourly is not None or args.hydro_min_daily is not None
                    or args.hydro_max_weekly is not None
                    or args.hydro_bypass_spill is not None)
    if args.hydro_restrictions or _hydro_flags:
        ocfg["active"] = True
        if args.hydro_min_hourly is not None:
            ocfg["min_hourly_frac"] = args.hydro_min_hourly
        if args.hydro_min_daily is not None:
            ocfg["min_daily_frac"] = args.hydro_min_daily
        if args.hydro_max_weekly is not None:
            by_zone = dict(ocfg.get("max_weekly_frac_by_zone") or {})
            for spec in args.hydro_max_weekly:
                if ":" in spec:
                    zone, val = spec.split(":", 1)
                    if zone not in cfg["zones"]:
                        raise SystemExit(f"--hydro-max-weekly: okänd zon '{zone}'")
                    by_zone[zone] = float(val)
                else:
                    ocfg["max_weekly_frac"] = float(spec)
            ocfg["max_weekly_frac_by_zone"] = by_zone
        if args.hydro_bypass_spill is not None:
            bs = dict(ocfg.get("bypass_spill") or {})
            bs["active"] = True
            bs["coefficient"] = args.hydro_bypass_spill
            ocfg["bypass_spill"] = bs

    if ocfg.get("active"):
        by_zone = ocfg.get("max_weekly_frac_by_zone") or {}
        print("Hydro-driftrestriktioner (reservoardelen):")
        _gw = float(ocfg.get("max_weekly_frac", 0) or 0)
        _wk = (f"max vecka {_gw:.2f} × max vecka" if _gw > 0
               else "max vecka: enbart per zon")
        print(f"  min tim {ocfg.get('min_hourly_frac', 0) or 0:.2f} × p_nom, "
              f"min dygn {ocfg.get('min_daily_frac', 0) or 0:.2f} × max dygn, "
              + _wk + (f" ({by_zone})" if by_zone else ""))
        if (ocfg.get("bypass_spill") or {}).get("active"):
            bs = ocfg["bypass_spill"]
            print(f"  bypass-spill PÅ: κ={bs.get('coefficient')} över "
                  f"(veckotak − {bs.get('threshold_below_max', 0.10)})")
        bounds = hydro_operation_bounds(n)
        if not bounds.empty:
            print(f"  {'zon':7s}{'p_nom MW':>10s}{'tillrinn TWh':>14s}"
                  f"{'andel av max':>14s}   förenligt med [min dygn, max vecka]")
            lo = float(ocfg.get("min_daily_frac", 0) or 0)
            for zone, row in bounds.iterrows():
                hi = float((ocfg.get("max_weekly_frac_by_zone") or {}).get(
                    zone, ocfg.get("max_weekly_frac", 0)) or 0)
                frac = row["inflow_frac"]
                ok = (frac >= lo) and (hi <= 0 or frac <= hi)
                print(f"  {zone:7s}{row['p_nom_mw']:10.0f}{row['inflow_mwh']/1e6:14.2f}"
                      f"{frac:14.3f}   {'ja' if ok else 'NEJ'}")
        for w in hydro_operation_feasibility_report(n, ocfg):
            print(f"  ⚠️  {w}")
        extra_callbacks.append(hydro_operation_constraints(ocfg))

    # Onshore-expansionstak per zon (override på default p_nom_max)
    for zone, cap in onshore_caps.items():
        name = f"{zone} wind_onshore"
        if name not in n.generators.index:
            print(f"  Varning: --onshore-cap zon {zone} saknas — hoppar över")
            continue
        existing = float(n.generators.at[name, "p_nom"])
        if not bool(n.generators.at[name, "p_nom_extendable"]):
            print(f"  Varning: {name} ej extendable (kör utan --no-expansion) — hoppar över cap")
            continue
        if cap < existing:
            print(f"  Varning: onshore-cap {zone} {cap:.0f} < installerat {existing:.0f} — sätter till installerat")
            cap = existing
        n.generators.at[name, "p_nom_min"] = existing
        n.generators.at[name, "p_nom_max"] = cap
        print(f"  → onshore-cap {zone}: p_nom_max = {cap:.0f} MW (installerat {existing:.0f})")

    # Sol-expansionstak i alla zoner (override på default p_nom_max)
    if args.solar_cap is not None:
        for zone in cfg["zones"]:
            name = f"{zone} solar"
            if name not in n.generators.index:
                continue
            existing = float(n.generators.at[name, "p_nom"])
            if not bool(n.generators.at[name, "p_nom_extendable"]):
                continue
            cap = max(args.solar_cap, existing)
            n.generators.at[name, "p_nom_min"] = existing
            n.generators.at[name, "p_nom_max"] = cap
            print(f"  → solar-cap {zone}: p_nom_max = {cap:.0f} MW (installerat {existing:.0f})")

    # Utbyggnadstak per zon/teknik från --demand-scenario (Bilaga B-potentialer).
    # cap=0 (t.ex. kärnkraft i NO/DK) låser tekniken till befintlig nivå.
    for (zone, carrier), pmax in demand_pnom_max.items():
        name = f"{zone} {carrier}"
        if name not in n.generators.index:
            continue
        if not bool(n.generators.at[name, "p_nom_extendable"]):
            continue
        # --onshore-lower skalar landvind-potentialen (golvet hanteras separat nedan).
        scaled = pmax * args.onshore_lower if carrier == "wind_onshore" else pmax
        existing = float(n.generators.at[name, "p_nom"])
        cap = max(scaled, existing)          # aldrig under redan installerat
        n.generators.at[name, "p_nom_min"] = existing
        n.generators.at[name, "p_nom_max"] = cap
        note = f" [×{args.onshore_lower:g}]" if (carrier == "wind_onshore" and args.onshore_lower != 1.0) else ""
        print(f"  → potential {zone} {carrier}: p_nom_max = {cap:.0f} MW (installerat {existing:.0f}){note}")

    # Exogena GOLV per zon/teknik från --demand-scenario (pnom_min_mw, t.ex. SE-basflotta
    # LMA2026). Sätter p_nom_min = max(golv, installerat); höjer p_nom_max om golvet
    # skulle hamna över taket. Tvingar fram minst golv-MW (ej curtailbart bort).
    for (zone, carrier), pmin in demand_pnom_min.items():
        name = f"{zone} {carrier}"
        if name not in n.generators.index:
            continue
        if not bool(n.generators.at[name, "p_nom_extendable"]):
            print(f"  Varning: {name} ej extendable — kan ej sätta golv {pmin:.0f}, hoppar över")
            continue
        existing = float(n.generators.at[name, "p_nom"])
        floor = max(pmin, existing)
        n.generators.at[name, "p_nom_min"] = floor
        if float(n.generators.at[name, "p_nom_max"]) < floor:
            n.generators.at[name, "p_nom_max"] = floor
        print(f"  → golv {zone} {carrier}: p_nom_min = {floor:.0f} MW "
              f"(installerat {existing:.0f}{' — golv binder, +%.0f MW' % (floor-existing) if floor>existing else ''})")

    if args.dispatch:                          # frys alla kapaciteter till källkörningens p_nom_opt
        freeze_capacities_from(n, args.dispatch)

    # EFTER frysningen: kontrollen av extendable måste se den frysta världen.
    apply_vre_curtailment_cost(n, args.vre_curtailment_cost)

    if args.dry_run:
        nuc = n.generators[n.generators.carrier == "nuclear"]
        print("\n=== DRY-RUN: kärnkraftsgeneratorer ===")
        for g, row in nuc.iterrows():
            kind = "EXPANSION" if g.endswith("nuclear exp") else "befintlig"
            print(f"  {g:18s} [{kind:9s}] p_nom={row.p_nom:7.0f}  "
                  f"p_nom_max={row.p_nom_max:8.0f}  ext={bool(row.p_nom_extendable)!s:5s}  "
                  f"CF(p_max)={float(n.generators_t.p_max_pu[g].mean()) if g in n.generators_t.p_max_pu else row.p_max_pu:.3f}  "
                  f"must-run={float(n.generators_t.p_min_pu[g].mean()) if g in n.generators_t.p_min_pu else row.p_min_pu:.3f}")
        print(f"  Σ befintlig p_nom = {nuc[~nuc.index.str.endswith('exp')].p_nom.sum():.0f} MW, "
              f"Σ expansion-tak p_nom_max = {nuc[nuc.index.str.endswith('exp')].p_nom_max.sum():.0f} MW")
        print("=== DRY-RUN klar (ingen solve) ===")
        return

    # Skapa resultatmappen i förväg så att loggfilen kan skrivas dit
    log_path = RESULTS_DIR / label / "highs.log"
    (RESULTS_DIR / label).mkdir(parents=True, exist_ok=True)
    write_run_meta(label, args, res, args.year, flag_str)

    n.sanitize()
    if args.rolling_horizon:
        ok, pinned_results = solve_rolling_horizon(
            n, cfg, args, inputs["market_prices"], res,
            log_path=log_path, extra_callbacks=extra_callbacks)
    elif args.soc_pin_from:
        ok, pinned_results = solve_soc_pinned(
            n, cfg, args.soc_pin_from, args.soc_pin_freq,
            band_frac=args.soc_pin_band,
            log_path=log_path, extra_callbacks=extra_callbacks)
    else:
        ok = solve(n, cfg, log_path=log_path,
                   soc_pin_end=soc_pin_end or None,
                   extra_callbacks=extra_callbacks)
    if not ok:
        print("Lösning misslyckades — kontrollera nätverket")
        sys.exit(1)

    if args.soc_pin_from or args.rolling_horizon:
        # CSV:erna byggs av de fönstervisa lösningarna (sanningskälla); network.nc
        # exporteras också — PyPSA ackumulerar fönstren i n.*_t via update().
        (RESULTS_DIR / label).mkdir(parents=True, exist_ok=True)
        n.export_to_netcdf(RESULTS_DIR / label / "network.nc")
        save_results_dict(pinned_results, label)
    else:
        save_results(n, label)

    if args.grid_cost:
        report_grid_cost(n, cfg, len(snapshots) * res / 8760.0)

    print("Klart!")


if __name__ == "__main__":
    main()
