"""
Skapar stackade area-grafer för elproduktion per zon och för Norden totalt.

Färger och ordning (nere→upp): export (market+länk), import (market+länk),
kärnkraft, hydro, onshore vind, offshore vind, sol, gas, termisk must-run.
Efterfrågan visas som svart heldragen linje.
Efterfrågan visas som svart heldragen linje.

Användning:
    python scripts/plot_dispatch.py results/res6h_2024/
    python scripts/plot_dispatch.py results/res3h_2023_2025/ --resample 3D
    python scripts/plot_dispatch.py results/res3h_2023_2025/ --zone SE-S --resample 7D
"""
import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ---------------------------------------------------------------------------
# Kända zoner (för tolkning av länkkolumnnamn)
# ---------------------------------------------------------------------------

KNOWN_ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]

# ---------------------------------------------------------------------------
# Färger och stapelordning
# ---------------------------------------------------------------------------

COLORS = {
    "nuclear":        "#7B2FBE",   # lila
    "hydro":          "#2196F3",   # blå
    "wind_onshore":   "#4CAF50",   # grön
    "wind_offshore":  "#1B5E20",   # mörkgrön
    "solar":          "#FF9800",   # orange
    "gas":            "#9E9E9E",   # grå (dispatchable gas)
    "thermal":        "#795548",   # brun (must-run thermal)
    "import":         "#F48FB1",   # rosa (marknad)
    "export":         "#F48FB1",   # rosa (marknad, under noll)
    "link_import":    "#FF7043",   # korall/orange-röd (transmissionslänkar)
    "link_export":    "#FF7043",   # korall/orange-röd (transmissionslänkar, under noll)
    "slack":          "#FF0000",   # röd (load shedding, bör vara noll)
}

LABELS = {
    "nuclear":        "Kärnkraft",
    "hydro":          "Hydro",
    "wind_onshore":   "Vind onshore",
    "wind_offshore":  "Vind offshore",
    "solar":          "Sol",
    "gas":            "Gas",
    "thermal":        "Termisk",
    "import":         "Import (marknad)",
    "export":         "Export (marknad)",
    "link_import":    "Import (länk)",
    "link_export":    "Export (länk)",
    "slack":          "Last-avlastning",
}

# Ordning nere→upp i positiv stapel (export hanteras separat under noll)
STACK_ORDER = [
    "import", "link_import", "nuclear", "thermal", "hydro", "wind_onshore",
    "wind_offshore", "solar", "gas",
]


# ---------------------------------------------------------------------------
# Inläsning
# ---------------------------------------------------------------------------

def load_results(res_dir: Path) -> dict:
    gen   = pd.read_csv(res_dir / "dispatch_generators.csv",
                        index_col=0, parse_dates=True)
    hydro = pd.read_csv(res_dir / "dispatch_hydro.csv",
                        index_col=0, parse_dates=True)
    flows_path = res_dir / "flows.csv"
    flows = pd.read_csv(flows_path, index_col=0, parse_dates=True) if flows_path.exists() else None
    return dict(gen=gen, hydro=hydro, flows=flows)


def load_demand(snapshots: pd.DatetimeIndex, res_hours: float) -> pd.DataFrame:
    """Laddar faktisk förbrukning (inkl. must-run thermal) och resamplar."""
    freq = f"{int(res_hours)}h" if res_hours >= 1 else "h"
    df = pd.read_parquet(PROC_DIR / "load.parquet")
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df.resample(freq).mean().reindex(snapshots)


# ---------------------------------------------------------------------------
# Länkflöden
# ---------------------------------------------------------------------------

def _parse_link_col(col: str) -> tuple[str, str] | tuple[None, None]:
    """Tolkar kolumnnamn som '{z0}-{z1}' och returnerar (z0, z1) eller (None, None)."""
    for z0 in KNOWN_ZONES:
        if col.startswith(z0 + "-"):
            z1 = col[len(z0) + 1:]
            if z1 in KNOWN_ZONES:
                return z0, z1
    return None, None


def _zone_link_gross(zone: str, flows: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Brutto-inflöde och brutto-utflöde för zonen via transmissionslänkar.

    Varje länk behandlas per riktning: en länk kan flöda in till zonen
    medan en annan flödar ut — det nettas INTE ihop.
    Returnerar (inflow, outflow), båda positiva.
    """
    inflow  = pd.Series(0.0, index=flows.index)
    outflow = pd.Series(0.0, index=flows.index)
    for col in flows.columns:
        z0, z1 = _parse_link_col(col)
        if z0 == zone:
            # Positivt flöde = z0→z1 (utförsel från zonen)
            outflow += flows[col].clip(lower=0)
            inflow  += (-flows[col]).clip(lower=0)
        elif z1 == zone:
            # Positivt flöde = z0→z1 (inflöde till zonen)
            inflow  += flows[col].clip(lower=0)
            outflow += (-flows[col]).clip(lower=0)
    return inflow, outflow


# ---------------------------------------------------------------------------
# Aggregering per zon och bärare
# ---------------------------------------------------------------------------

def extract_zone_carrier(gen: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    """Returnerar {zone: {carrier: pd.Series}} för alla generatorer."""
    result: dict[str, dict[str, pd.Series]] = {}
    for col in gen.columns:
        # Namnformat: "{zon} {carrier med understreck}"
        parts = col.split(" ", 1)
        if len(parts) != 2:
            continue
        zone, carrier = parts[0], parts[1].replace(" ", "_")
        result.setdefault(zone, {})[carrier] = gen[col]
    return result


def build_zone_df(zone: str, data: dict, demand: pd.DataFrame) -> pd.DataFrame:
    """Bygger en DataFrame med en kolumn per bärare (MW) för en zon.

    Kolumner:
      export         — marknadsgenerator, negativa värden (under noll)
      link_export    — netto-utförsel via NTC-länkar, negativa värden
      import         — marknadsgenerator, positiva värden
      link_import    — netto-inflöde via NTC-länkar, positiva värden
      nuclear/hydro/… — lokal produktion, positiva värden
      demand         — faktisk last (MW)
      demand_net     — last + nettoutförsel = effektiv last på lokala generatorer
                       (ska sammanfalla med toppen av produktionsstapeln)
    """
    gen_by_zone = extract_zone_carrier(data["gen"])
    zone_gen    = gen_by_zone.get(zone, {})
    idx         = data["gen"].index

    cols = {}

    # Marknadsgenerator: import (pos) och export (neg)
    market_series = zone_gen.get("market", pd.Series(0.0, index=idx))
    if "market" in zone_gen:
        cols["import"] = market_series.clip(lower=0)
        cols["export"] = market_series.clip(upper=0)

    # Övriga generatorer
    for carrier, series in zone_gen.items():
        if carrier not in ("market",) and carrier in COLORS:
            cols[carrier] = series.clip(lower=0)

    # Hydro från dispatch_hydro.csv
    hydro_col = f"{zone} hydro"
    if hydro_col in data["hydro"].columns:
        cols["hydro"] = data["hydro"][hydro_col].clip(lower=0)

    # Transmissionslänkar: brutto inflow/outflow per zon
    flows = data.get("flows")
    link_inflow  = pd.Series(0.0, index=idx)
    link_outflow = pd.Series(0.0, index=idx)
    if flows is not None:
        li, lo = _zone_link_gross(zone, flows.reindex(idx).fillna(0))
        link_inflow, link_outflow = li, lo
        cols["link_import"] = link_inflow           # positiv, staplas ovan noll
        cols["link_export"] = -link_outflow         # negativ, visas under noll

    # Efterfrågan och streckad kurva
    if demand is not None and zone in demand.columns:
        cols["demand"] = demand[zone]
        # Visuell stacktop = G + Li⁺ + Lm⁺
        # Från energibalansen: G + Li⁺ + Lm⁺ = D + Li⁻ + Lm⁻
        # → stacktop = D + brutto-utflöde (länk) + |export market|
        cols["demand_net"] = (
            demand[zone]
            + link_outflow
            + (-market_series).clip(lower=0)   # marknadens export (0 vid import)
        )

    return pd.DataFrame(cols, index=idx).fillna(0)


# ---------------------------------------------------------------------------
# Ritning
# ---------------------------------------------------------------------------

def _detect_resolution(index: pd.DatetimeIndex) -> float:
    """Returnerar tidssteg i timmar."""
    if len(index) < 2:
        return 1.0
    return (index[1] - index[0]).total_seconds() / 3600


def plot_zone(ax: plt.Axes, df: pd.DataFrame, zone: str,
              resample: str | None) -> None:
    if resample:
        df = df.resample(resample).mean()

    t = df.index

    # --- Negativa ytor: export (marknad + länk), staplade under noll ---
    neg_bottom = np.zeros(len(df))
    for key in ("export", "link_export"):
        if key in df.columns:
            vals = df[key].values   # ≤ 0
            ax.fill_between(t, neg_bottom + vals, neg_bottom,
                            color=COLORS[key], alpha=0.8,
                            label=LABELS[key], step="post")
            neg_bottom += vals

    # --- Positiva staplar ---
    bottoms = np.zeros(len(df))
    for carrier in STACK_ORDER:
        if carrier not in df.columns:
            continue
        vals = df[carrier].values
        ax.fill_between(t, bottoms, bottoms + vals,
                        color=COLORS[carrier], alpha=0.85,
                        label=LABELS.get(carrier, carrier),
                        step="post")
        bottoms += vals

    # --- Efterfrågekurva (heldragen) ---
    if "demand" in df.columns:
        ax.step(t, df["demand"], color="black", linewidth=1.0,
                where="post", label="Efterfrågan")

    # --- Formatering ---
    ax.set_title(zone, fontsize=11, fontweight="bold")
    ax.set_ylabel("MW")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f} GW" if abs(x) >= 1000 else f"{x:.0f} MW")
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.5)


def make_legend(fig: plt.Figure, carriers_present: set) -> None:
    handles = []
    seen = set()

    # Export (marknad + länk), under noll
    for key in ("export", "link_export"):
        if key in carriers_present:
            patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS[key], alpha=0.8)
            handles.append((patch, LABELS[key]))
            seen.add(key)

    # Positiva staplar i STACK_ORDER
    for carrier in STACK_ORDER:
        if carrier in carriers_present and carrier not in seen:
            patch = plt.Rectangle((0, 0), 1, 1, fc=COLORS[carrier], alpha=0.85)
            handles.append((patch, LABELS.get(carrier, carrier)))
            seen.add(carrier)

    # Kurvor
    line_demand = plt.Line2D([0], [0], color="black", linewidth=1.0)
    handles.append((line_demand, "Efterfrågan"))

    patches, labels = zip(*handles)
    fig.legend(patches, labels,
               loc="lower center", ncol=len(handles),
               frameon=True, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path,
                        help="Mapp med körningsresultat, t.ex. results/res3h_2023_2025/")
    parser.add_argument("--resample", default=None,
                        help="Resample-frekvens, t.ex. '1D', '7D'. Standard: ingen (rådata)")
    parser.add_argument("--zone", default=None,
                        help="Rita bara en zon (t.ex. SE-S). Standard: alla + Nordic total")
    args = parser.parse_args()

    res_dir = args.results_dir
    if not res_dir.exists():
        print(f"Mapp saknas: {res_dir}")
        sys.exit(1)

    print(f"Laddar resultat från {res_dir} ...")
    data = load_results(res_dir)

    snapshots = data["gen"].index
    res_hours = _detect_resolution(snapshots)
    demand    = load_demand(snapshots, res_hours)

    zones = sorted({c.split(" ", 1)[0] for c in data["gen"].columns})
    if args.zone:
        zones = [args.zone]

    plots_dir = res_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- En fil per zon ---
    for zone in zones:
        df = build_zone_df(zone, data, demand)
        carriers_present = set(df.columns)

        fig, ax = plt.subplots(figsize=(14, 4))
        plot_zone(ax, df, zone, args.resample)
        make_legend(fig, carriers_present)
        fig.tight_layout(rect=[0, 0.08, 1, 1])

        out = plots_dir / f"dispatch_{zone}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")

    # --- Nordic total ---
    if not args.zone:
        all_zones_df = []
        for zone in zones:
            df = build_zone_df(zone, data, demand)
            all_zones_df.append(df)

        nordic = pd.concat(all_zones_df).groupby(level=0).sum()

        # Netto import/export: summera over alla zoner (interna flöden tar ut varandra)
        carriers_present = set(nordic.columns)

        fig, ax = plt.subplots(figsize=(14, 4))
        plot_zone(ax, nordic, "Norden totalt", args.resample)
        make_legend(fig, carriers_present)
        fig.tight_layout(rect=[0, 0.08, 1, 1])

        out = plots_dir / "dispatch_nordic_total.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out}")

    print("Klart!")


if __name__ == "__main__":
    main()
