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
                          batteries: list) -> dict:
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
            batteries.append((zone, float(bat["p_nom_mw"]), float(bat.get("hours", 2)), False))
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

    for z0, z1, mw in scen.get("ntc_overrides", []):
        for link in cfg.get("links", []):
            if link[0] == z0 and link[1] == z1 and link[2] != mw:
                print(f"     NTC {z0}-{z1}: {link[2]} → {mw} MW (Tabell 10)")
                link[2] = mw
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
        f"argv:        {' '.join(sys.argv)}",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"  → run_meta.txt")


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
    parser.add_argument("--desc", default=None,
                        help="Kort fritext om körningens SYFTE (t.ex. 'kandidat för "
                             "dispatch-baseline med värme och vätgaslast'). Sparas i "
                             "results/<output>/run_meta.txt.")
    parser.add_argument("--extra-load", type=float, default=0.0,
                        help="Extra flat last i MW per zon (utöver faktisk last, standard: 0)")
    parser.add_argument("--no-expansion", action="store_true",
                        help="Lås alla teknologier som non-extendable — ren dispatch-körning")
    parser.add_argument("--cost-scenario", default=None, metavar="NAMN",
                        help="Skriv över cfg['costs'] med ett kostnadsscenario ur "
                             "cost_scenarios i zones.yaml (t.ex. svk_2040, svk_2050). "
                             "Inkl. byggränta (IDC). Avsett för expansionskörningar.")
    parser.add_argument("--demand-scenario", default=None, metavar="NAMN",
                        help="Addera ett efterfrågescenario ur demand_scenarios i "
                             "zones.yaml (t.ex. svk_2040_mm): per-zon extra-last, H2, EV, "
                             "utbyggnadstak (p_nom_max) och NTC-höjningar. Additivt över "
                             "eSett-basen. Avsett för expansionskörningar.")
    parser.add_argument("--continent-diurnal-scale", type=float, default=1.0, metavar="FAKTOR",
                        help="Komprimera kontinent-ventilprisets DYGNSSVÄNG (hour-of-day-komponent) "
                             "med FAKTOR (1.0=oförändrat, 0.5=halverad). Behåller nivå + dag-till-dag-"
                             "väder-volatilitet. Modellerar att 2040-kontinentlagring arbitrerar bort "
                             "den sol-formade dygnsspreaden Nordens hydro annars utnyttjar. Bara "
                             "ventil-bzn (DE-LU/EE/LT/PL/NL/GB), ej zon-priser. Se project_solar_overbuild_continent_spread.")
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
    parser.add_argument("--offwind-capfac-increase", type=float, default=0.0, metavar="FRAC",
                        help="Som --onwind-capfac-increase men för HAVSbaserad vind "
                             "(0.1 = +10%%). Speglar 2040:s nybyggnadsflotta (moderna 15 MW-"
                             "turbiner). Påverkar genereringen; potential-MW i config förutsätter "
                             "matchande CF.")
    parser.add_argument("--onshore-cap", action="append", default=[], metavar="ZON:MW",
                        help="Sätt expansionstak (p_nom_max, MW) för landbaserad vind per zon, "
                             "t.ex. 'SE-N:20000'. Override på default-taket (50 GW/zon). "
                             "Kräver expansion (onshore extendable). Kan anges flera gånger.")
    parser.add_argument("--solar-cap", type=float, default=None, metavar="MW",
                        help="Sätt expansionstak (p_nom_max, MW) för sol i ALLA zoner, "
                             "t.ex. '10000'. Override på default-taket (50 GW/zon). "
                             "Kräver expansion (sol extendable).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Bygg nätverket och skriv en komponent-sammanfattning "
                             "(kärnkraft m.m.), men SOLVE:a inte. För verifiering före körning.")
    parser.add_argument("--add-nuclear", nargs="+", action="extend", default=[], metavar="ZON:N:SEED",
                        help="Lägg till N NYA kärnkraftsreaktorer i en zon med syntetisk "
                             "stokastisk tillgänglighet (RNG-seed), t.ex. "
                             "'SE-S:10:101'. EXTENDABLE — kapaciteten optimeras (implicit "
                             "reaktorstorlek ≈ p_nom_opt/N), tak N×1500 MW. Befintlig flotta "
                             "finns med by default och byter då till syntetisk profil "
                             "(config.nuclear_synth_existing). Kan anges flera gånger.")
    parser.add_argument("--nuclear-discount-rate", nargs="+", action="extend", default=[], metavar="ZON:RATE",
                        help="Egen diskontoränta för NY kärnkraft (--add-nuclear) i en zon, "
                             "t.ex. 'SE-N:0.03 SE-S:0.03'. Påverkar bara den extendable expansionens "
                             "annualiserade kapitalkostnad (befintlig flotta är fast). Default = global "
                             "costs.discount_rate. Kan anges flera gånger / som lista.")
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
    parser.add_argument("--add-heat", action="store_true",
                        help="Aktivera fjärrvärmesektorn (config heat): per-zon heat-buss "
                             "(FV-behov + ackumulator + el-panna + stor-VP + bio/KVV). Drar bort "
                             "dagens FV-el ur AC-lasten. Kräver data/processed/heat_load.parquet.")
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

    demand_pnom_max, demand_pnom_min = {}, {}
    if args.demand_scenario:
        demand_pnom_max, demand_pnom_min = apply_demand_scenario(
            cfg, args.demand_scenario, hydrogen_overrides, ev_overrides, batteries)

    if args.no_market:
        cfg["market_connections"] = []

    if args.spill_cost is not None:
        cfg["costs"]["hydro"]["spill_cost_eur_per_mwh"] = args.spill_cost
        print(f"  → hydro spill_cost = {args.spill_cost} EUR/MWh")

    if args.market_elasticity:
        cfg.setdefault("market_elasticity", {})["enabled"] = True

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

    if args.add_heat and args.chp_fixed_gw is not None:
        parser.error("--chp-fixed-gw och --add-heat är ömsesidigt uteslutande "
                     "(KVV-fast kopplar bort värmebussen).")

    if args.add_heat:
        cfg.setdefault("heat", {})["enabled"] = True
        if args.heat_store_ext:
            cfg["heat"]["store_extendable"] = True
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
    if args.extra_load:                 flags.append(f"extra-load-{args.extra_load:.0f}mw")
    if args.cost_scenario:              flags.append(f"cost-{args.cost_scenario}")
    if args.demand_scenario:            flags.append(f"demand-{args.demand_scenario}")
    if args.no_expansion:               flags.append("no-expansion")
    if args.no_market:                  flags.append("no-market")
    if args.effective_ntc:              flags.append("effective-ntc")
    if not args.market_elasticity:      flags.append("no-market-elast")
    if args.add_heat:                   flags.append("heat-store-ext" if args.heat_store_ext else "heat")
    if args.chp_fixed_gw is not None:   flags.append(f"chpfixed-{args.chp_fixed_gw:g}gw")
    if args.market_scale is not None:   flags.append("market-scale-" + args.market_scale.replace(":", "").replace(",", "_"))
    if args.voll:                       flags.append("voll")
    for spec in args.add_battery:       flags.append(f"battery-{spec.replace(':','_')}")
    if args.battery_extendable:         flags.append("battery-ext")
    if args.battery:                    flags.append("battery-" + "_".join(args.battery).replace(":", "_"))
    for z in args.expand_vre:           flags.append(f"expand-vre-{z}")
    if args.expand_budget_musd:         flags.append(f"oc-budget-{args.expand_budget_musd:.0f}musd")
    if args.expand_budget_meur:         flags.append(f"oc-budget-{args.expand_budget_meur:.0f}meur")
    if args.onwind_capfac_increase:     flags.append(f"onwind-cf+{args.onwind_capfac_increase:.2f}")
    if args.offwind_capfac_increase:    flags.append(f"offwind-cf+{args.offwind_capfac_increase:.2f}")
    for spec in args.add_nuclear:       flags.append(f"nuclear-{spec.replace(':','_')}")
    for spec in args.nuclear_discount_rate: flags.append(f"nucdisc-{spec.replace(':','_')}")
    for spec in args.add_wind:          flags.append(f"wind-{spec.replace(':','_')}")
    for spec in args.add_h2:            flags.append(f"h2-{spec.replace(':','_')}")
    for spec in args.add_h2_ext:        flags.append(f"h2ext-{spec.replace(':','_')}")
    for spec in args.add_ev:            flags.append(f"ev-{spec.replace(':','_')}")
    for spec in args.onshore_cap:       flags.append(f"onshorecap-{spec.replace(':','_')}")
    if args.solar_cap is not None:      flags.append(f"solarcap-{args.solar_cap:.0f}")
    if soc_pin_end:                     flags.append("soc-pin-" + "_".join(soc_pin_end.keys()))
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
    synthetic_nuclear = {
        "existing": cfg.get("nuclear_synth_existing", {}),
        "params":   cfg.get("nuclear_synth", {}),
        "active":   bool(extra_nuclear),
    }
    if extra_nuclear:
        ex = cfg.get("nuclear_synth_existing", {})
        print(f"Kärnkraft EXPANSIONSLÄGE: befintlig flotta syntetisk {list(ex)}; "
              f"nya reaktorer {[(z, n, s) for z, n, s in extra_nuclear]} "
              f"(target_cf={synthetic_nuclear['params'].get('target_cf', 0.85)})")

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
                      ev_overrides=ev_overrides or None)

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
        existing = float(n.generators.at[name, "p_nom"])
        cap = max(pmax, existing)          # aldrig under redan installerat
        n.generators.at[name, "p_nom_min"] = existing
        n.generators.at[name, "p_nom_max"] = cap
        print(f"  → potential {zone} {carrier}: p_nom_max = {cap:.0f} MW (installerat {existing:.0f})")

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
