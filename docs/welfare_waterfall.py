#!/usr/bin/env python
"""Välfärds-vattenfall per kraftslag (en graf var). Stänger exakt på ΔTotal/år.

  Netto = ΣKonsumentnytta + Producentnytta + Trängselrent + Förluster/handel − Kapex
        ≈ −ΔTotal (systemkostnadsändring)

Alla texter och tal är justerbara i KONFIG-blocket nedan. Kör:  python temp/welfare_waterfall.py
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================ KONFIG — JUSTERA HÄR ============================
YLABEL   = "Δ mot baseline, M€/år"
SUPTITLE = "Välfärdsuppdelning — vem vinner/förlorar när billig energi adderas\n(2025-kostnader, 781 M€/år)"
OUTDIR   = "temp"          # sparas som temp/welfare_<nyckel>.png
YLIM     = (-560, 2820)    # (min, max) på y-axeln

# Färger
C_UP   = "#2e7d32"   # vinst (+)
C_DOWN = "#c62828"   # förlust (−)
C_NET  = "#37474f"   # nettovälfärd

# Namn på legenden
LEG_UP, LEG_DOWN, LEG_NET = "Vinst (+)", "Förlust (−)", "Nettovälfärd"

# Ett block per kraftslag → en graf var. 'bars' = (etikett, värde) i ordning;
# sista "Netto"-stapeln ritas automatiskt som summan av bars.
CASES = {
    "onshore": dict(
        title="Landvind (onshore +20 %)",
        subtitle="ΔTotal system −286 M€/år",
        bars=[
            ("Konsument-\nnytta",  2571),
            ("Producent-\nnytta", -2377),
            ("Trängsel-\nrent",     328),
            ("Förluster &\nhandel", -28),   # curtail≈0, round-trip≈0 → domineras av handelselasticitet
            ("Kapex",              -780),
        ],
        net_label="Netto",
    ),
    "nuclear": dict(
        title="Kärnkraft (@8000)",
        subtitle="ΔTotal system −439 M€/år",
        bars=[
            ("Konsument-\nnytta",  1883),
            ("Producent-\nnytta", -1769),
            ("Trängsel-\nrent",     216),
            ("Förluster &\nhandel", -56),
            ("Kapex",              -713),
        ],
        net_label="Netto",
    ),
    "offshore": dict(
        title="Havsvind (offshore +10 %)",
        subtitle="ΔTotal system −470 M€/år",
        bars=[
            ("Konsument-\nnytta",  1823),
            ("Producent-\nnytta", -1686),
            ("Trängsel-\nrent",     255),
            ("Förluster &\nhandel", -81),
            ("Kapex",              -781),
        ],
        net_label="Netto",
    ),
    "solonly": dict(
        title="Sol-only (10,47 GW)",
        subtitle="ΔTotal system −494 M€/år",
        bars=[
            ("Konsument-\nnytta",  2085),
            ("Producent-\nnytta", -1974),
            ("Trängsel-\nrent",     337),
            ("Förluster &\nhandel",-161),
            ("Kapex",              -781),
        ],
        net_label="Netto",
    ),
    "solbatteri": dict(
        title="Sol + batteri (2,65 GW ×2)",
        subtitle="ΔTotal system −590 M€/år",
        bars=[
            ("Konsument-\nnytta",   706),
            ("Producent-\nnytta",  -597),
            ("Trängsel-\nrent",      35),
            ("Förluster &\nhandel",  47),
            ("Kapex",              -781),
        ],
        net_label="Netto",
    ),
    "purbatteri": dict(
        title="Pur batteri (3,55 GW/4h)",
        subtitle="ΔTotal system −643 M€/år",
        bars=[
            ("Konsument-\nnytta",    62),
            ("Producent-\nnytta",    20),
            ("Trängsel-\nrent",     -73),
            ("Förluster &\nhandel", 130),
            ("Kapex",              -782),
        ],
        net_label="Netto",
    ),
}
# ============================================================================


def draw(key, cfg):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    vals   = [v for _, v in cfg["bars"]]
    xlabs  = [l for l, _ in cfg["bars"]] + [cfg["net_label"]]

    starts, cum = [], 0.0
    for v in vals:
        starts.append(cum); cum += v
    net = cum

    for i, v in enumerate(vals):
        bottom = starts[i] + (v if v < 0 else 0)
        ax.bar(i, abs(v), bottom=bottom, width=0.62,
               color=(C_UP if v > 0 else C_DOWN), edgecolor="black", linewidth=0.8, zorder=3)
        ytxt = (starts[i] + v + 70) if v > 0 else (starts[i] + v - 75)
        ax.text(i, ytxt, f"{v:+.0f}", ha="center",
                va=("bottom" if v > 0 else "top"), fontsize=10, fontweight="bold")

    n = len(vals)
    ax.bar(n, net, width=0.62, color=C_NET, edgecolor="black", linewidth=0.8, zorder=3)
    ax.text(n, net - 75, f"{net:+.0f}", ha="center", va="top",
            fontsize=10, fontweight="bold", color=C_NET)

    conn = [0]
    for i in range(n):
        conn.append(conn[-1] + vals[i])
    for i in range(n):
        ax.plot([i + 0.31, i + 1 - 0.31], [conn[i + 1], conn[i + 1]],
                color="grey", lw=0.9, ls="--", zorder=2)

    ax.axhline(0, color="black", lw=1.1, zorder=1)
    ax.set_title(f"{cfg['title']}\n{cfg['subtitle']}", fontsize=12, fontweight="bold")
    ax.set_xticks(range(n + 1)); ax.set_xticklabels(xlabs, fontsize=10)
    ax.set_ylabel(YLABEL, fontsize=11)
    ax.set_ylim(*YLIM)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)

    leg = [Patch(fc=C_UP, ec="black", label=LEG_UP),
           Patch(fc=C_DOWN, ec="black", label=LEG_DOWN),
           Patch(fc=C_NET, ec="black", label=LEG_NET)]
    ax.legend(handles=leg, loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(SUPTITLE, fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = f"{OUTDIR}/welfare_{key}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Sparat {path}  (netto {net:+.0f} M€/år)")


if __name__ == "__main__":
    for key, cfg in CASES.items():
        draw(key, cfg)
