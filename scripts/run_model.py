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
    hydro_terminal_value,
)

PROC_DIR    = Path(__file__).resolve().parents[1] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "zones.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
          restricted_yearly_hydro: bool = False) -> bool:
    scfg    = cfg["solver"]
    solver  = scfg["name"]
    options = {k: v for k, v in scfg.items() if k != "name"}

    if log_path is not None:
        options["log_file"] = str(log_path)

    callbacks = [hydro_soc_initial_constraint(cfg)]
    if restricted_yearly_hydro:
        hydro_zones = [z for z, zc in cfg["zones"].items()
                       if zc.get("hydro_p_nom_mw", 0) > 0]
        annual_prod = load_annual_hydro_production(hydro_zones)
        callbacks.append(hydro_annual_production_constraints(cfg, annual_prod))
        print(f"  → årsvis hydrocap aktiv för: {', '.join(annual_prod)}")

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
            cyclic_soc=False,
            soc_initial_override=soc_carry,
        )

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
    parser.add_argument("--no-extra-load", action="store_true",
                        help="Nollställ additional_load_mw — använd faktisk last utan tillägg")
    parser.add_argument("--no-expansion", action="store_true",
                        help="Lås alla teknologier som non-extendable — ren dispatch-körning")
    parser.add_argument("--normalized-inflow-profiles", action="store_true",
                        help="Normera inflödesprofilen per år mot faktisk vattenkraftproduktion")
    parser.add_argument("--restricted-yearly-hydro", action="store_true",
                        help="LP-caps: begränsa hydrodispatch per zon och år till faktisk nivå")
    parser.add_argument("--no-market", action="store_true",
                        help="Stäng ned alla externa marknadsanslutningar (p_nom=0)")
    parser.add_argument("--rolling-horizon", action="store_true",
                        help="Lös med rullande horisont + terminalvärde (fixar konstant vattenvärde)")
    parser.add_argument("--rolling-weeks", type=int, default=4,
                        help="Fönsterstorlek i veckor för rullande horisont (standard: 4)")
    args = parser.parse_args()

    cfg = load_config()
    res = args.resolution or cfg["snapshots"].get("resolution_hours", 1)

    if args.no_extra_load:
        cfg["additional_load_mw"] = {}

    if args.no_expansion:
        for tech in cfg.get("costs", {}):
            if isinstance(cfg["costs"][tech], dict):
                cfg["costs"][tech]["extendable"] = False

    if args.no_market:
        cfg["market_connections"] = []

    flags = []
    if args.no_extra_load:              flags.append("no-extra-load")
    if args.no_expansion:               flags.append("no-expansion")
    if args.normalized_inflow_profiles: flags.append("normalized-inflow")
    if args.restricted_yearly_hydro:    flags.append("restricted-hydro")
    if args.no_market:                  flags.append("no-market")
    if args.rolling_horizon:            flags.append(f"rolling-{args.rolling_weeks}w")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"Konfiguration: upplösning={res}h, år={args.year or '2023-2025'}{flag_str}")

    inputs    = load_inputs(cfg)
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

    print(f"Bygger nätverk ({len(snapshots)} tidssteg) ...")
    n = build_network(cfg, snapshots, **inputs,
                      normalize_inflow=args.normalized_inflow_profiles)

    # Skapa resultatmappen i förväg så att loggfilen kan skrivas dit
    log_path = RESULTS_DIR / label / "highs.log"
    (RESULTS_DIR / label).mkdir(parents=True, exist_ok=True)

    n.sanitize()
    ok = solve(n, cfg, log_path=log_path,
               restricted_yearly_hydro=args.restricted_yearly_hydro)
    if not ok:
        print("Lösning misslyckades — kontrollera nätverket")
        sys.exit(1)

    save_results(n, label)
    print("Klart!")


if __name__ == "__main__":
    main()
