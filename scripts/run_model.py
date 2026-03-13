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

from nordpsa.network import build_network, hydro_soc_initial_constraint

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

def solve(n, cfg: dict, log_path: Path | None = None) -> bool:
    scfg    = cfg["solver"]
    solver  = scfg["name"]
    options = {k: v for k, v in scfg.items() if k != "name"}

    if log_path is not None:
        options["log_file"] = str(log_path)

    extra_func = hydro_soc_initial_constraint(cfg)

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
    args = parser.parse_args()

    cfg = load_config()
    res = args.resolution or cfg["snapshots"].get("resolution_hours", 1)

    if args.no_extra_load:
        cfg["additional_load_mw"] = {}

    if args.no_expansion:
        for tech in cfg.get("costs", {}):
            if isinstance(cfg["costs"][tech], dict):
                cfg["costs"][tech]["extendable"] = False

    flags = []
    if args.no_extra_load: flags.append("no-extra-load")
    if args.no_expansion:  flags.append("no-expansion")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""
    print(f"Konfiguration: upplösning={res}h, år={args.year or '2023-2025'}{flag_str}")

    inputs    = load_inputs(cfg)
    snapshots = make_snapshots(cfg, res, args.year)
    inputs    = resample_inputs(inputs, snapshots, res)

    print(f"Bygger nätverk ({len(snapshots)} tidssteg) ...")
    n = build_network(cfg, snapshots, **inputs)

    if args.output:
        label = args.output
    else:
        label = f"res{res}h_{'_'.join(str(s.year) for s in [snapshots[0], snapshots[-1]])}"
        if args.year:
            label = f"res{res}h_{args.year}"

    # Skapa resultatmappen i förväg så att loggfilen kan skrivas dit
    log_path = RESULTS_DIR / label / "highs.log"
    (RESULTS_DIR / label).mkdir(parents=True, exist_ok=True)

    n.sanitize()
    ok = solve(n, cfg, log_path=log_path)
    if not ok:
        print("Lösning misslyckades — kontrollera nätverket")
        sys.exit(1)

    save_results(n, label)
    print("Klart!")


if __name__ == "__main__":
    main()
