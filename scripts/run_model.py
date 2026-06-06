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

from nordpsa.hydro import load_annual_hydro_production
from nordpsa.network import (
    build_network,
    hydro_annual_production_constraints,
    hydro_soc_initial_constraint,
    hydro_soc_band_constraint,
    hydro_soc_terminal_band_constraint,
    hydro_terminal_value,
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

    # Sätt UTC-index och ta bort timezone (PyPSA kräver tz-naivt)
    for df in (load_df, vre, nuclear, thermal, prices_df):
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    market_prices = {col: prices_df[col] for col in prices_df.columns}

    return dict(
        load=load_df, vre_profiles=vre, vre_noms=vre_noms,
        nuclear_profile=nuclear, thermal_profile=thermal,
        hydro_params=hydro_params, market_prices=market_prices,
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
    return out


# ---------------------------------------------------------------------------
# Lösning och sparning
# ---------------------------------------------------------------------------

def solve(n, cfg: dict, log_path: Path | None = None,
          restricted_yearly_hydro: bool = False,
          lambda_per_zone: dict | None = None,
          soc_band: tuple[float, float] | None = None,
          soc_terminal_band: tuple[float, float] | None = None,
          extra_callbacks: list | None = None) -> bool:
    scfg    = cfg["solver"]
    solver  = scfg["name"]
    options = {k: v for k, v in scfg.items() if k != "name"}

    if log_path is not None:
        options["log_file"] = str(log_path)

    if soc_terminal_band is not None:
        # Icke-cyklisk: start sätts av state_of_charge_initial, slut binds till band.
        callbacks = [hydro_soc_terminal_band_constraint(cfg, soc_terminal_band[0], soc_terminal_band[1])]
        print(f"  → SOC terminal-band: start=config, slut∈[{soc_terminal_band[0]*100:.0f}%, "
              f"{soc_terminal_band[1]*100:.0f}%] (icke-cyklisk)")
    elif soc_band is not None:
        callbacks = [hydro_soc_band_constraint(cfg, soc_band[0], soc_band[1])]
        print(f"  → SOC-band aktivt: [{soc_band[0]*100:.0f}%, {soc_band[1]*100:.0f}%] "
              f"av kapacitet (flytande start=slut)")
    else:
        callbacks = [hydro_soc_initial_constraint(cfg)]
    if restricted_yearly_hydro:
        hydro_zones = [z for z, zc in cfg["zones"].items()
                       if zc.get("hydro_p_nom_mw", 0) > 0]
        annual_prod = load_annual_hydro_production(hydro_zones)
        callbacks.append(hydro_annual_production_constraints(cfg, annual_prod))
        print(f"  → årsvis hydrocap aktiv för: {', '.join(annual_prod)}")
    if lambda_per_zone:
        callbacks.append(hydro_terminal_value(cfg, lambda_per_zone))
        for zone, lam in lambda_per_zone.items():
            print(f"  → terminalvärde {zone}: λ={lam:.1f} EUR/MWh")
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


def save_results(n, label: str) -> None:
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)

    n.export_to_netcdf(out / "network.nc")

    # Platta csv-filer för enkel inspektion
    n.generators_t.p.to_csv(out / "dispatch_generators.csv")
    n.storage_units_t.p.to_csv(out / "dispatch_hydro.csv")
    n.storage_units_t.state_of_charge.to_csv(out / "hydro_soc.csv")
    n.storage_units_t.spill.to_csv(out / "hydro_spill.csv")
    n.links_t.p0.to_csv(out / "flows.csv")
    n.buses_t.marginal_price.to_csv(out / "prices.csv")

    # Vattenvärde = dual på lagringsbalansen (EUR/MWh), om assignad
    wv = n.storage_units_t.get("mu_energy_balance")
    if wv is not None and wv.shape[1] > 0:
        wv.to_csv(out / "water_value.csv")

    # thermal dispatch finns nu i dispatch_generators.csv (carrier="thermal")

    print(f"  → resultat sparade i {out}/")


# ---------------------------------------------------------------------------
# Rullande horisont
# ---------------------------------------------------------------------------

def slice_inputs(inputs: dict, snapshots: pd.DatetimeIndex) -> dict:
    """Skär alla tidsserier i inputs till givna snapshots."""
    out = {}
    for key in ("load", "vre_profiles", "nuclear_profile", "thermal_profile"):
        out[key] = inputs[key].reindex(snapshots)
    out["market_prices"] = {
        bzn: s.reindex(snapshots) for bzn, s in inputs["market_prices"].items()
    }
    out["vre_noms"]     = inputs["vre_noms"]
    out["hydro_params"] = inputs["hydro_params"]
    return out


def rolling_windows(snapshots: pd.DatetimeIndex, window_steps: int):
    """Delar upp snapshots i sekventiella fönster av window_steps tidssteg."""
    n = len(snapshots)
    for s in range(0, n, window_steps):
        yield snapshots[s : min(s + window_steps, n)]


def get_terminal_lambdas(
    cfg:             dict,
    market_prices:   dict,
    t_last:          pd.Timestamp,
    lookahead_steps: int,
) -> dict:
    """λ per hydro-zon = framåtriktat medelpris av DE-LU nästa fönster."""
    de_lu = market_prices.get("DE-LU")
    if de_lu is None:
        lam = 60.0
    else:
        idx   = de_lu.index.get_indexer([t_last], method="nearest")[0]
        ahead = de_lu.iloc[idx : idx + lookahead_steps]
        lam   = float(ahead.mean()) if len(ahead) > 0 else float(de_lu.mean())
    return {
        zone: lam
        for zone, zcfg in cfg["zones"].items()
        if zcfg.get("hydro_p_nom_mw", 0) > 0
    }


def save_rolling_results(results: dict, label: str) -> None:
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    results["gen"].to_csv(out / "dispatch_generators.csv")
    results["hydro_p"].to_csv(out / "dispatch_hydro.csv")
    results["soc"].to_csv(out / "hydro_soc.csv")
    results["spill"].to_csv(out / "hydro_spill.csv")
    results["flows"].to_csv(out / "flows.csv")
    results["prices"].to_csv(out / "prices.csv")
    print(f"  → resultat sparade i {out}/")


def rolling_horizon_solve(
    cfg:        dict,
    inputs:     dict,
    snapshots:  pd.DatetimeIndex,
    args,
    resolution: int,
    label:      str,
) -> bool:
    steps_per_week = (7 * 24) // resolution
    window_steps   = args.rolling_weeks * steps_per_week
    windows        = list(rolling_windows(snapshots, window_steps))
    n_win          = len(windows)
    scfg           = cfg["solver"]
    solver_opts    = {k: v for k, v in scfg.items() if k != "name"}

    print(f"Rullande horisont: {args.rolling_weeks} veckor/fönster"
          f" ({window_steps} tidssteg), {n_win} fönster totalt")

    # Fast terminalvärde λ (override) om angivet, annars framåtblickande DE-LU
    lambda_override = None
    if getattr(args, "terminal_lambda", None):
        hydro_zones = [z for z, zc in cfg["zones"].items() if zc.get("hydro_p_nom_mw", 0) > 0]
        if ":" in args.terminal_lambda:
            lambda_override = {z.strip(): float(v) for z, v in
                               (pair.split(":") for pair in args.terminal_lambda.split(","))}
        else:
            lambda_override = {z: float(args.terminal_lambda) for z in hydro_zones}
        print(f"  → fast terminal-λ: {lambda_override}")

    # Initial SOC från zones.yaml
    soc_carry: dict = {}
    for zone, zcfg in cfg["zones"].items():
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom > 0:
            frac = zcfg.get("hydro_soc_initial", 0.5)
            soc_carry[zone] = frac * p_nom * max_h

    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: dict = {k: [] for k in ("prices", "gen", "hydro_p", "soc", "spill", "flows")}

    for i, win_snaps in enumerate(windows):
        print(f"\nFönster {i+1}/{n_win}: {win_snaps[0].date()} – {win_snaps[-1].date()}"
              f"  ({len(win_snaps)} tidssteg)")

        win_inputs = slice_inputs(inputs, win_snaps)
        n = build_network(
            cfg, win_snaps, **win_inputs,
            normalize_inflow=args.normalized_inflow_profiles,
            actual_inflow=not args.parametric_inflow,
            cyclic_soc=False,
            soc_initial_override=soc_carry,
        )

        if lambda_override is not None:
            lam_per_zone = lambda_override
        else:
            lam_per_zone = get_terminal_lambdas(
                cfg, inputs["market_prices"], win_snaps[-1], window_steps
            )
        tv_cb = hydro_terminal_value(cfg, lam_per_zone)

        def extra_func(n, snaps, _cb=tv_cb):
            _cb(n, snaps)

        log_path = out_dir / f"highs_w{i+1:02d}.log"
        status, condition = n.optimize(
            solver_name=scfg["name"],
            solver_options=dict(solver_opts, log_file=str(log_path)),
            extra_functionality=extra_func,
        )
        print(f"  Status: {status} / {condition}")

        if status != "ok":
            print("  Lösning misslyckades — avbryter")
            return False

        # SOC carry-over till nästa fönster
        soc_t = n.storage_units_t.state_of_charge
        for zone in list(soc_carry):
            su = f"{zone} hydro"
            if su in soc_t.columns:
                soc_carry[zone] = float(soc_t[su].iloc[-1])

        all_dfs["prices"].append(n.buses_t.marginal_price)
        all_dfs["gen"].append(n.generators_t.p)
        all_dfs["hydro_p"].append(n.storage_units_t.p)
        all_dfs["soc"].append(soc_t)
        all_dfs["spill"].append(n.storage_units_t.spill)
        all_dfs["flows"].append(n.links_t.p0)

    save_rolling_results({k: pd.concat(v) for k, v in all_dfs.items()}, label)
    return True


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
    parser.add_argument("--normalized-inflow-profiles", action="store_true",
                        help="Normera inflödesprofilen per år mot faktisk vattenkraftproduktion")
    parser.add_argument("--parametric-inflow", action="store_true",
                        help="Använd parametrisk inflödesfunktion (Gauss+cosinus) istf faktiska "
                             "NVE/ENTSO-E-profiler. Standard: faktiska profiler används.")
    parser.add_argument("--restricted-yearly-hydro", action="store_true",
                        help="LP-caps: begränsa hydrodispatch per zon och år till faktisk nivå")
    parser.add_argument("--no-market", action="store_true",
                        help="Stäng ned alla externa marknadsanslutningar (p_nom=0)")
    parser.add_argument("--rolling-horizon", action="store_true",
                        help="Lös med rullande horisont + terminalvärde (fixar konstant vattenvärde)")
    parser.add_argument("--rolling-weeks", type=int, default=4,
                        help="Fönsterstorlek i veckor för rullande horisont (standard: 4)")
    parser.add_argument("--terminal-lambda", default=None, metavar="VAL|Z:val,...",
                        help="Fast terminalvärde λ (EUR/MWh) i rullande horisont istället för "
                             "framåtblickande DE-LU-medel. Enskilt ('11') eller per zon "
                             "('NO-N:11,NO-S:2.8,SE-N:12,SE-S:1.6,FI:11'). Sätt till extraherade vattenvärden.")
    parser.add_argument("--hydro-terminal-value", action="store_true",
                        help="Vattenvärde via terminalvillkor -λ*SOC[T] (λ=medel zonpris) istf cyklisk SOC")
    parser.add_argument("--link-scale", action="append", default=[], metavar="Z0,Z1,FACTOR",
                        help="Skala NTC för länk Z0↔Z1 med FACTOR (t.ex. 'SE-N,SE-S,0.5'). Kan anges flera gånger.")
    parser.add_argument("--voll", action="store_true",
                        help="Lägg till VOLL-slack (3000 EUR/MWh) i ALLA zoner — ger losslastmått och förhindrar dualexplosion")
    parser.add_argument("--soc-band", default=None, metavar="LOW,HIGH",
                        help="Tillåt cyklisk SOC start=slut att flyta inom [LOW,HIGH] av kapacitet "
                             "(t.ex. '0.6,0.8') istället för fast hydro_soc_initial-nivå")
    parser.add_argument("--soc-terminal-band", default=None, metavar="LOW,HIGH",
                        help="Icke-cyklisk: start = config hydro_soc_initial, slut flyter i [LOW,HIGH] "
                             "av kapacitet (t.ex. '0.6,0.8'). Tillåter netto-uttömning över horisonten.")
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
    parser.add_argument("--effective-ntc", action="store_true",
                        help="Använd effektiv kontinentkapacitet (P80 av faktiska flöden, "
                             "market_connections_effective_mw) istället för märkkapacitet")
    parser.add_argument("--market-scale", default=None, metavar="FACTOR|ZON:F,...",
                        help="Skala kontinentkablars kapacitet. Enskild faktor för alla "
                             "(t.ex. '0.7') eller per zon (t.ex. 'FI:0.5,NO-S:0.8,SE-S:0.6,DK:0.7'). "
                             "Appliceras efter --effective-ntc om båda anges.")
    args = parser.parse_args()

    def _parse_band(spec, name):
        parts = spec.split(",")
        if len(parts) != 2:
            print(f"Ogiltigt {name} format: '{spec}' (förväntat LOW,HIGH)")
            sys.exit(1)
        return (float(parts[0]), float(parts[1]))

    soc_band = _parse_band(args.soc_band, "--soc-band") if args.soc_band else None
    soc_terminal_band = _parse_band(args.soc_terminal_band, "--soc-terminal-band") if args.soc_terminal_band else None

    batteries = []
    for spec in args.add_battery:
        parts = spec.split(":")
        if len(parts) != 3:
            print(f"Ogiltigt --add-battery format: '{spec}' (förväntat ZON:MW:HOURS)")
            sys.exit(1)
        batteries.append((parts[0].strip(), float(parts[1]), float(parts[2])))

    extra_nuclear = []
    for spec in args.add_nuclear:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            print(f"Ogiltigt --add-nuclear format: '{spec}' (förväntat ZON:MW eller ZON:MW:PMIN)")
            sys.exit(1)
        pmin = float(parts[2]) if len(parts) == 3 else 1.0
        extra_nuclear.append((parts[0].strip(), float(parts[1]), pmin))

    cfg = load_config()
    res = args.resolution or cfg["snapshots"].get("resolution_hours", 1)

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

    for spec in args.link_scale:
        parts = spec.split(",")
        if len(parts) != 3:
            print(f"Ogiltigt --link-scale format: '{spec}' (förväntat Z0,Z1,FACTOR)")
            sys.exit(1)
        z0, z1, factor = parts[0].strip(), parts[1].strip(), float(parts[2])
        scaled = False
        for link in cfg["links"]:
            if (link[0] == z0 and link[1] == z1) or (link[0] == z1 and link[1] == z0):
                link[2] = link[2] * factor
                print(f"  → NTC {link[0]}↔{link[1]}: {link[2]/factor:.0f} → {link[2]:.0f} MW (×{factor})")
                scaled = True
        if not scaled:
            print(f"Varning: ingen länk hittades för {z0}↔{z1}")

    flags = []
    if args.extra_load:                 flags.append(f"extra-load-{args.extra_load:.0f}mw")
    if args.no_expansion:               flags.append("no-expansion")
    if args.normalized_inflow_profiles: flags.append("normalized-inflow")
    if args.parametric_inflow:          flags.append("parametric-inflow")
    if args.restricted_yearly_hydro:    flags.append("restricted-hydro")
    if args.no_market:                  flags.append("no-market")
    if args.effective_ntc:              flags.append("effective-ntc")
    if args.market_scale is not None:   flags.append("market-scale-" + args.market_scale.replace(":", "").replace(",", "_"))
    if args.hydro_terminal_value:       flags.append("tv-hydro")
    if args.rolling_horizon:            flags.append(f"rolling-{args.rolling_weeks}w")
    for spec in args.link_scale:        flags.append(f"link-scale-{spec.replace(',','_')}")
    if args.voll:                       flags.append("voll")
    for spec in args.add_battery:       flags.append(f"battery-{spec.replace(':','_')}")
    if args.battery_extendable:         flags.append("battery-ext")
    for z in args.expand_vre:           flags.append(f"expand-vre-{z}")
    if args.expand_budget_musd:         flags.append(f"oc-budget-{args.expand_budget_musd:.0f}musd")
    if args.expand_budget_meur:         flags.append(f"oc-budget-{args.expand_budget_meur:.0f}meur")
    if args.onwind_capfac_increase:     flags.append(f"onwind-cf+{args.onwind_capfac_increase:.2f}")
    for spec in args.add_nuclear:       flags.append(f"nuclear-{spec.replace(':','_')}")
    if soc_band:                        flags.append(f"soc-band-{soc_band[0]:.2f}-{soc_band[1]:.2f}")
    if soc_terminal_band:               flags.append(f"soc-term-band-{soc_terminal_band[0]:.2f}-{soc_terminal_band[1]:.2f}")
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

    if args.rolling_horizon:
        ok = rolling_horizon_solve(cfg, inputs, snapshots, args, res, label)
        if not ok:
            print("Rullande horisont misslyckades")
            sys.exit(1)
        print("Klart!")
        return

    # Icke-cyklisk om terminalvärde ELLER terminal-band används (start≠slut tillåts)
    cyclic_soc = not (args.hydro_terminal_value or soc_terminal_band is not None)

    print(f"Bygger nätverk ({len(snapshots)} tidssteg) ...")
    n = build_network(cfg, snapshots, **inputs,
                      normalize_inflow=args.normalized_inflow_profiles,
                      actual_inflow=not args.parametric_inflow,
                      cyclic_soc=cyclic_soc,
                      voll=args.voll,
                      batteries=batteries,
                      battery_extendable=args.battery_extendable,
                      extra_nuclear=extra_nuclear)

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

    # Terminalvärde: λ per hydro-zon = medel zonpris över hela perioden
    lambda_per_zone = None
    if args.hydro_terminal_value:
        lambda_per_zone = {
            zone: float(inputs["market_prices"][zone].mean())
            for zone, zcfg in cfg["zones"].items()
            if zcfg.get("hydro_p_nom_mw", 0) > 0 and zone in inputs["market_prices"]
        }

    # Skapa resultatmappen i förväg så att loggfilen kan skrivas dit
    log_path = RESULTS_DIR / label / "highs.log"
    (RESULTS_DIR / label).mkdir(parents=True, exist_ok=True)

    n.sanitize()
    ok = solve(n, cfg, log_path=log_path,
               restricted_yearly_hydro=args.restricted_yearly_hydro,
               lambda_per_zone=lambda_per_zone,
               soc_band=soc_band,
               soc_terminal_band=soc_terminal_band,
               extra_callbacks=extra_callbacks)
    if not ok:
        print("Lösning misslyckades — kontrollera nätverket")
        sys.exit(1)

    save_results(n, label)
    print("Klart!")


if __name__ == "__main__":
    main()
