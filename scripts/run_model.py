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
import sys
from pathlib import Path

import pandas as pd
import yaml

# pandas 2.x använder Arrow-strängar som standard; PyPSA/xarray stöder inte det
pd.options.future.infer_string = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nordpsa.network import (
    build_network,
    hydro_soc_initial_constraint,
    hydro_soc_terminal_pin_constraint,
    oc_budget_constraint,
    _annualized_cost,
)

USD_TO_EUR = 0.926   # 1 USD ≈ 0.926 EUR (1 EUR ≈ 1.08 USD), 2026
VRE_CARRIERS = ("wind_onshore", "wind_offshore", "solar")

PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "zones.yaml"


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

    # Sätt UTC-index och ta bort timezone (PyPSA kräver tz-naivt)
    dfs = [load_df, vre, nuclear, thermal, prices_df]
    if heat_load is not None:
        dfs.append(heat_load)
    for df in dfs:
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    market_prices = {col: prices_df[col] for col in prices_df.columns}

    return dict(
        load=load_df, vre_profiles=vre, vre_noms=vre_noms,
        nuclear_profile=nuclear, thermal_profile=thermal,
        hydro_params=hydro_params, market_prices=market_prices,
        heat_load=heat_load,
    )


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=None,
                        help="Tidsupp. i timmar (åsidosätter config)")
    parser.add_argument("--year", type=int, default=None,
                        help="Kör ett enstaka år (t.ex. 2024)")
    parser.add_argument("--output", default=None,
                        help="Resultatmapp under results/ (t.ex. 'run_v2_spring_flood'). "
                             "Standard: automatiskt namn baserat på upplösning och år.")
    parser.add_argument("--extra-load", type=float, default=0.0,
                        help="Extra flat last i MW per zon (utöver faktisk last, standard: 0)")
    parser.add_argument("--no-expansion", action="store_true",
                        help="Lås alla teknologier som non-extendable — ren dispatch-körning")
    parser.add_argument("--no-market", action="store_true",
                        help="Stäng ned alla externa marknadsanslutningar (p_nom=0)")
    parser.add_argument("--voll", action="store_true",
                        help="Lägg till VOLL-slack (3000 EUR/MWh) i ALLA zoner — ger losslastmått och förhindrar dualexplosion")
    parser.add_argument("--soc-pin", action="append", default=[], metavar="ZON:START:END",
                        help="Icke-cyklisk: lås BÅDA ändpunkterna till faktiska fyllnadsfraktioner per zon, "
                             "t.ex. 'SE-N:0.577:0.709' (start 57.7%%, slut 70.9%% av kapacitet). Kalibrering mot "
                             "observerad reservoarnivå. Komma-separera flera eller upprepa flaggan.")
    parser.add_argument("--spill-cost", type=float, default=None, metavar="EUR",
                        help="Hydro-spillkostnad (EUR/MWh). Default 0.1 (tillåter spill vid full reservoar). "
                             "Högt värde (t.ex. 50) bryter LP-degeneracy i expansionskörningar.")
    parser.add_argument("--add-battery", action="append", default=[], metavar="ZON:MW:HOURS",
                        help="Lägg till batteri (StorageUnit) i en zon, t.ex. 'SE-S:1000:4'. "
                             "Kan anges flera gånger.")
    parser.add_argument("--battery-extendable", action="store_true",
                        help="Gör --add-battery investerbart: modellen optimerar effekten "
                             "0..MW (varaktighet fast) mot batterikostnad i config. "
                             "Utan flaggan är batteriet fast (dispatch-tillägg).")
    parser.add_argument("--battery", nargs="+", default=None, metavar="DURATION [ZON:MW ...]",
                        help="Batterier med given varaktighet (t.ex. '4h'). Utan zoner: "
                             "expanderbart 4h-batteri i VARJE zon (modellen optimerar effekten). "
                             "Med zoner (t.ex. '4h SE-S:5000 SE-N:2000'): fasta storlekar i de "
                             "angivna zonerna, 0 i övriga. Varaktigheten är fast (StorageUnit).")
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
    parser.add_argument("--onwind-capfac-increase", type=float, default=0.0, metavar="FRAC",
                        help="Höj landbaserad vinds kapacitetsfaktor med denna relativa "
                             "andel (0.1 = +10%%). Olinjär potens-transform per zon: lyfter "
                             "låga effektnivåer mest, märkeffekt (cf_max) oförändrad.")
    parser.add_argument("--add-nuclear", action="append", default=[], metavar="ZON:MW[:PMIN]",
                        help="Lägg till ny kärnkraft i en zon, t.ex. 'SE-S:2500' (must-run baslast) "
                             "eller 'SE-S:2500:0' (dispatchbar). Kan anges flera gånger.")
    parser.add_argument("--add-wind", action="append", default=[], metavar="ZON:MW",
                        help="Lägg till fast landbaserad vindkraft (dispatch, ej extendable), "
                             "t.ex. 'SE-S:9893'. Samma CF-profil som zonens befintliga wind_onshore. "
                             "Energi-motsvarighet till --add-nuclear men variabel. Kan anges flera gånger.")
    parser.add_argument("--add-h2", action="append", default=[], metavar="ZON:DEMAND:EL:STORE[:TURB]",
                        help="Lägg till vätgassystem i en zon (fasta storlekar): baslast DEMAND MW_H2, "
                             "elektrolysör EL MW_el, lager STORE MWh, valfri turbin TURB MW_el. "
                             "T.ex. 'SE-S:500:1000:50000:300'. Kostnader från costs.hydrogen. "
                             "Kan anges flera gånger.")
    parser.add_argument("--add-h2-ext", action="append", default=[], metavar="ZON:DEMAND",
                        help="Investerbart vätgassystem (extendable elektrolys + lager, INGEN turbin): "
                             "baslast DEMAND MW_H2, modellen dimensionerar elektrolysör och lager mot "
                             "costs.hydrogen. T.ex. 'SE-N:1000'. Kan anges flera gånger.")
    parser.add_argument("--effective-ntc", action="store_true",
                        help="Använd effektiv kontinentkapacitet (P80 av faktiska flöden, "
                             "market_connections_effective_mw) istället för märkkapacitet")
    parser.add_argument("--market-elasticity", action="store_true",
                        help="Pris-elastisk kontinentgräns (trappa): stora nordiska flöden "
                             "flyttar gränspriset. S_export/S_import per gräns från "
                             "config market_elasticity. Rekommenderas för expansionskörningar.")
    parser.add_argument("--add-heat", action="store_true",
                        help="Aktivera fjärrvärmesektorn (config heat): per-zon heat-buss "
                             "(FV-behov + ackumulator + el-panna + stor-VP + bio/KVV). Drar bort "
                             "dagens FV-el ur AC-lasten. Kräver data/processed/heat_load.parquet.")
    parser.add_argument("--market-scale", default=None, metavar="FACTOR|ZON:F,...",
                        help="Skala kontinentkablars kapacitet. Enskild faktor för alla "
                             "(t.ex. '0.7') eller per zon (t.ex. 'FI:0.5,NO-S:0.8,SE-S:0.6,DK:0.7'). "
                             "Appliceras efter --effective-ntc om båda anges.")
    args = parser.parse_args()

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

    # Batterier samlas som 4-tupler (zon, p_nom_mw, max_hours, extendable).
    batteries = []
    for spec in args.add_battery:   # bakåtkompatibel: --add-battery + --battery-extendable
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Ogiltigt --add-battery format: '{spec}' (förväntat ZON:MW:HOURS)")
            sys.exit(1)
        batteries.append((parts[0].strip(), float(parts[1]), float(parts[2]),
                          bool(args.battery_extendable)))
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

    extra_nuclear = []
    for spec in args.add_nuclear:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            print(f"Ogiltigt --add-nuclear format: '{spec}' (förväntat ZON:MW eller ZON:MW:PMIN)")
            sys.exit(1)
        pmin = float(parts[2]) if len(parts) == 3 else 1.0
        extra_nuclear.append((parts[0].strip(), float(parts[1]), pmin))

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
        if len(parts) != 2:
            print(f"Ogiltigt --add-h2-ext format: '{spec}' (förväntat ZON:DEMAND)")
            sys.exit(1)
        hydrogen_overrides[parts[0].strip()] = {
            "demand_mw":    float(parts[1]),
            "electrolyser": {"p_nom_mw": 0.0, "extendable": True},
            "store":        {"e_nom_mwh": 0.0, "extendable": True},
        }

    cfg = load_config()
    res = args.resolution or cfg["snapshots"].get("resolution_hours", 1)

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

    if args.no_expansion:
        for tech in cfg.get("costs", {}):
            if isinstance(cfg["costs"][tech], dict):
                cfg["costs"][tech]["extendable"] = False

    if args.no_market:
        cfg["market_connections"] = []

    if args.spill_cost is not None:
        cfg["costs"]["hydro"]["spill_cost_eur_per_mwh"] = args.spill_cost
        print(f"  → hydro spill_cost = {args.spill_cost} EUR/MWh")

    if args.market_elasticity:
        cfg.setdefault("market_elasticity", {})["enabled"] = True

    if args.add_heat:
        cfg.setdefault("heat", {})["enabled"] = True
        print("  → pris-elastisk kontinentgräns (trappa) aktiv")

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
    if args.extra_load:                 flags.append(f"extra-load-{args.extra_load:.0f}mw")
    if args.no_expansion:               flags.append("no-expansion")
    if args.no_market:                  flags.append("no-market")
    if args.effective_ntc:              flags.append("effective-ntc")
    if args.market_elasticity:          flags.append("market-elasticity")
    if args.add_heat:                   flags.append("heat")
    if args.market_scale is not None:   flags.append("market-scale-" + args.market_scale.replace(":", "").replace(",", "_"))
    if args.voll:                       flags.append("voll")
    for spec in args.add_battery:       flags.append(f"battery-{spec.replace(':','_')}")
    if args.battery_extendable:         flags.append("battery-ext")
    if args.battery:                    flags.append("battery-" + "_".join(args.battery).replace(":", "_"))
    for z in args.expand_vre:           flags.append(f"expand-vre-{z}")
    if args.expand_budget_musd:         flags.append(f"oc-budget-{args.expand_budget_musd:.0f}musd")
    if args.expand_budget_meur:         flags.append(f"oc-budget-{args.expand_budget_meur:.0f}meur")
    if args.onwind_capfac_increase:     flags.append(f"onwind-cf+{args.onwind_capfac_increase:.2f}")
    for spec in args.add_nuclear:       flags.append(f"nuclear-{spec.replace(':','_')}")
    for spec in args.add_wind:          flags.append(f"wind-{spec.replace(':','_')}")
    for spec in args.add_h2:            flags.append(f"h2-{spec.replace(':','_')}")
    for spec in args.add_h2_ext:        flags.append(f"h2ext-{spec.replace(':','_')}")
    if soc_pin_end:                     flags.append("soc-pin-" + "_".join(soc_pin_end.keys()))
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"Konfiguration: upplösning={res}h, år={args.year or '2023-2025'}{flag_str}")

    inputs    = load_inputs(cfg)
    if args.onwind_capfac_increase:
        inputs["vre_profiles"] = boost_onshore_capfac(
            inputs["vre_profiles"], args.onwind_capfac_increase)
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

    print(f"Bygger nätverk ({len(snapshots)} tidssteg) ...")
    n = build_network(cfg, snapshots, **inputs,
                      cyclic_soc=cyclic_soc,
                      voll=args.voll,
                      batteries=batteries,
                      extra_nuclear=extra_nuclear,
                      extra_wind=extra_wind,
                      soc_initial_override=soc_pin_start or None,
                      hydrogen_overrides=hydrogen_overrides or None)

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

    # Skapa resultatmappen i förväg så att loggfilen kan skrivas dit
    log_path = RESULTS_DIR / label / "highs.log"
    (RESULTS_DIR / label).mkdir(parents=True, exist_ok=True)

    n.sanitize()
    ok = solve(n, cfg, log_path=log_path,
               soc_pin_end=soc_pin_end or None,
               extra_callbacks=extra_callbacks)
    if not ok:
        print("Lösning misslyckades — kontrollera nätverket")
        sys.exit(1)

    save_results(n, label)
    print("Klart!")


if __name__ == "__main__":
    main()
