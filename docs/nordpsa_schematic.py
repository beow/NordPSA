"""NordPSA sektor-kopplad modell-schematik (à la Adelsfors Ireland-2050-figuren).
Ritar alla komponenttyper: generering, lagring, omvandling, sektorbussar, laster,
plus 6-zons NTC-sammankoppling. Årsvärden = run141 (SvK 2040 MM, 3h) nordiska totaler.

Användning:
    python docs/nordpsa_schematic.py [utfil.png]
Default-utfil = docs/nordpsa_schematic.png (bredvid skriptet, cwd-oberoende)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

C = dict(
    nuc="#8e44ad", onw="#27ae60", offw="#16a085", sol="#f1c40f", hyd="#2980b9",
    ror="#5dade2", therm="#7f5539", gas="#7f8c8d", mkt="#34495e", batt="#e67e22",
    h2="#1abc9c", heat="#e74c3c", ev="#2c5f2d", boil="#d35400", bus="#1b2631",
    h2bus="#0e6655", heatbus="#922b21", evbus="#1e5631", load="#c0392b", text="#1b2631",
)

fig, ax = plt.subplots(figsize=(21, 12.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def box(x, y, w, h, label, color, val=None, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8",
                 linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.16, zorder=3))
    txt = label if val is None else f"{label}\n{val}"
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=fs,
            color=C["text"], zorder=4, weight="bold")

def busbar(x, y, w, h, label, color):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=0, facecolor=color, zorder=3))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=12.5,
            color="white", rotation=90, weight="bold", zorder=4)

def arrow(x0, y0, x1, y1, color, bidir=False, lw=2.2, ls="-"):
    style = "<|-|>" if bidir else "-|>"
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                 mutation_scale=16, linewidth=lw, color=color, zorder=2,
                 linestyle=ls, shrinkA=1, shrinkB=1))

# ---- ELBUSS (spine) ----
SP = 37.0  # bus left x
busbar(SP, 12, 3.2, 82, "ELBUSS  (AC)", C["bus"])
SPR = SP + 3.2  # right edge
ax.text(SP + 1.6, 13.3, "×6", ha="center", fontsize=11, style="italic",
        color="white", weight="bold", zorder=5)

# ---- GENERERING (vänster, → elbuss) ----
gx, gw, gh = 1.5, 14.5, 7.0
gens = [
    ("Kärnkraft (must-run)", C["nuc"], "94 TWh"),
    ("Vind, land", C["onw"], "176 TWh"),
    ("Vind, hav", C["offw"], "10 TWh"),
    ("Sol-PV", C["sol"], "91 TWh"),
    ("Vattenkraft, älv (RoR)", C["ror"], "42 TWh"),
    ("Termisk must-run (industri)", C["therm"], None),
    ("Gas CCGT-CCS (topp)", C["gas"], "7 TWh"),
    ("Marknad (kontinentventil)", C["mkt"], "netto-exp 13 TWh"),
]
gy_top = 86.5
gstep = (gy_top - 20.5) / (len(gens) - 1)
for i, (lab, col, val) in enumerate(gens):
    gy = gy_top - i * gstep
    box(gx, gy, gw, gh, lab, col, val, fs=9.3)
    bd = (i == len(gens) - 1)  # market = bidirectional
    arrow(gx + gw, gy + gh/2, SP, gy + gh/2, col, bidir=bd)

# ---- LAGRING (under elbuss, bidirektionellt) ----
box(26.5, 1.0, 11.5, 8.0, "Vattenmagasin", C["hyd"], "SOC cykliskt", fs=9.3)
arrow(32.2, 9.0, 32.2, 12.0, C["hyd"], bidir=True)
box(40.0, 1.0, 11.5, 8.0, "Batteri (exogen)", C["batt"], "12 GW · 2h", fs=9.3)
arrow(45.7, 9.0, 45.7, 12.0, C["batt"], bidir=True)

# ---- EL-LAST (höger topp, ut) ----
box(44.0, 86.5, 16.0, 8.0, "El-last", C["load"], "inflexibel + industri/datacenter", fs=9.3)
arrow(SPR, 90.5, 44.0, 90.5, C["load"])

# ---- OMVANDLING (mitten, elbuss ↔ sektorbussar) ----
cx, cw, ch = 52.0, 13.0, 6.2
conv = [
    ("Elektrolysör", C["h2"], 80.0, "to_h2"),
    ("H2-turbin", C["h2"], 70.5, "from_h2"),
    ("Värmepump", C["heat"], 53.0, "to_heat"),
    ("El-panna", C["boil"], 44.5, "to_heat"),
    ("KVV bakpress (el+värme)", C["therm"], 35.5, "chp"),
    ("EV-laddare", C["ev"], 23.0, "to_ev"),
]
conv_y = {}
for lab, col, yc, kind in conv:
    cy = yc - ch/2
    conv_y[lab] = yc
    box(cx, cy, cw, ch, lab, col, fs=9.0)
    if kind == "from_h2":
        arrow(cx, yc, SPR, yc, col)            # turbine → elbuss (power back)
    elif kind == "chp":
        arrow(cx, yc + 1.2, SPR, yc + 1.2, col) # CHP → elbuss (el)
    else:
        arrow(SPR, yc, cx, yc, col)            # elbuss → converter

# ---- SEKTORBUSSAR ----
BB = 70.0; BBR = 73.2
busbar(BB, 66, 3.2, 21, "VÄTGASBUSS", C["h2bus"])
busbar(BB, 30, 3.2, 28, "VÄRMEBUSS", C["heatbus"])
busbar(BB, 17, 3.2, 12, "EV-BUSS", C["evbus"])

# converter ↔ sektorbuss
arrow(cx + cw, conv_y["Elektrolysör"], BB, conv_y["Elektrolysör"], C["h2"])
arrow(BB, conv_y["H2-turbin"], cx + cw, conv_y["H2-turbin"], C["h2"])
arrow(cx + cw, conv_y["Värmepump"], BB, conv_y["Värmepump"], C["heat"])
arrow(cx + cw, conv_y["El-panna"], BB, conv_y["El-panna"], C["boil"])
arrow(cx + cw, conv_y["KVV bakpress (el+värme)"] - 1.2, BB,
      conv_y["KVV bakpress (el+värme)"] - 1.2, C["therm"])   # CHP → värmebuss
arrow(cx + cw, conv_y["EV-laddare"], BB, conv_y["EV-laddare"], C["ev"])

# ---- SEKTOR-LAGER & LASTER (höger) ----
sx, sw, sh = 78.5, 15.0, 6.2
# Vätgas
box(sx, 80.0, sw, sh, "H2-lager", C["h2"], "Store", fs=9.0)
arrow(BBR, 83.1, sx, 83.1, C["h2"], bidir=True)
box(sx, 68.0, sw, sh, "H2-last (P2X)", C["h2"], "28 TWh", fs=9.0)
arrow(BBR, 71.1, sx, 71.1, C["h2"])
# Värme
box(sx, 50.0, sw, sh, "Värmelager", C["heat"], "Store (tak 20 GWh)", fs=9.0)
arrow(BBR, 53.1, sx, 53.1, C["heat"], bidir=True)
box(sx, 40.0, sw, sh, "Bio-panna", C["boil"], "(extendable)", fs=9.0)
arrow(sx, 43.1, BBR, 43.1, C["boil"])            # bio → värmebuss (källa)
box(sx, 30.0, sw, sh, "Värme-last (fjärrvärme)", C["heatbus"], "~108 TWh", fs=9.0)
arrow(BBR, 33.1, sx, 33.1, C["heatbus"])
# EV
box(sx, 21.0, sw, sh, "EV-lager", C["ev"], "Store (avgångsgolv)", fs=9.0)
arrow(BBR, 24.1, sx, 24.1, C["ev"], bidir=True)
box(sx, 12.5, sw, sh, "EV-körlast", C["evbus"], "44 TWh", fs=9.0)
arrow(BBR, 18.5, sx, 18.5, C["evbus"])

# ================= INSET: 6-zons NTC-topologi =================
ix, iy, iw, ih = 1.5, 0.5, 22.0, 17.5
ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0.2",
             linewidth=1.2, edgecolor="#999", facecolor="#f7f7f7", alpha=0.95, zorder=1))
ax.text(ix + iw/2, iy + ih - 1.2, "6 zoner · NTC-länkar", ha="center",
        fontsize=9.5, weight="bold", color=C["text"], zorder=5)
zpos = {"NO-N": (5.5, 12.0), "SE-N": (12.5, 12.0), "FI": (19.5, 11.2),
        "NO-S": (5.5, 6.2),  "SE-S": (12.5, 5.5),  "DK": (10.0, 2.0)}
links = [("NO-N","SE-N"), ("NO-N","NO-S"), ("SE-N","SE-S"), ("SE-N","FI"),
         ("NO-S","SE-S"), ("SE-S","FI"), ("SE-S","DK"), ("NO-S","DK")]
for a, b in links:
    (xa, ya), (xb, yb) = zpos[a], zpos[b]
    ax.plot([xa, xb], [ya, yb], color="#888", lw=1.6, zorder=2)
# market-ventil (kontinent) på SE-S, NO-S, DK, FI
for z in ["SE-S", "NO-S", "DK", "FI"]:
    xz, yz = zpos[z]
    ax.plot([xz, xz + 1.6], [yz - 1.4, yz - 2.6], color=C["mkt"], lw=1.4, ls=":", zorder=2)
for z, (xz, yz) in zpos.items():
    ax.add_patch(Circle((xz, yz), 1.5, facecolor=C["bus"], edgecolor="white",
                 linewidth=1.2, zorder=4))
    ax.text(xz, yz, z, ha="center", va="center", fontsize=6.6, color="white",
            weight="bold", zorder=5)
ax.text(ix + iw/2, iy + 0.4, "··· = kontinentventil (DE-LU/NL/GB/EE/LT/PL)",
        ha="center", fontsize=6.6, style="italic", color=C["text"], zorder=5)

# ---- titel ----
ax.text(50, 99.3, "NordPSA — sektorkopplad expansionsmodell (struktur per zon)",
        ha="center", fontsize=16, weight="bold", color=C["text"])
ax.text(50, 97.0, "Årsvärden: run141 · SvK 2040 MM · 3h · nordiska totaler",
        ha="center", fontsize=11, style="italic", color="#555")

plt.tight_layout()
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_suffix(".png")
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("sparad:", out)
