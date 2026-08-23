"""trading — handel: interna NTC-flöden mellan de sex zonerna och kontinent-
ventilen (market-generatorer). Veckomedel-tidsserier + tabell per länk/zon
(kapacitet, nettoflöde, trängselfrekvens, prisdifferens) och import/export/
netto per zon mot kontinenten.

Extraherad ur notebooks/explore_results.py (f.d. cell 7), 2026-08-21 — den cellen
saknade motsvarighet i notebooks/cells/ innan den här filen skapades.

Förutsätter bootstrap.py (globala: n, ZONES, LABEL, dispatch, flows, prices,
zone_market, twh, plt).
"""

# Cellen kan importeras fristående (från explore_results.py) ELLER klistras in
# i en notebook-cell efter att bootstrap.py redan körts där (explore.ipynb).
# Guard: skippar importen om bootstrap redan satt sina globaler i detta namnrum.
if 'LABEL' not in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bootstrap import *  # noqa: F401,F403

import numpy as np

snap = dispatch.index
ac = [lnk for lnk in n.links.index if n.links.at[lnk, 'carrier'] == 'AC']
win = max(int(24 * 7 / dt_h), 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), constrained_layout=True)
ax = axes[0]
for lnk in ac:
    if lnk in flows.columns:
        ax.plot(snap, flows[lnk].rolling(win, min_periods=1).mean(), lw=0.9, label=lnk)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('MW'); ax.set_title('Interna NTC-flöden, veckomedel'); ax.legend(fontsize=7)

ax = axes[1]
net = pd.DataFrame({z: zone_market(z) for z in ZONES})
ax.plot(snap, net.sum(axis=1).rolling(win, min_periods=1).mean(), color='#e07a5f',
        label='Norden netto')
for z in ZONES:
    if net[z].abs().sum() > 0:
        ax.plot(snap, net[z].rolling(win, min_periods=1).mean(), lw=0.8, label=z)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('MW (+ = import)'); ax.set_title('Kontinentventilen, veckomedel')
ax.legend(fontsize=7)
plt.show()

print(f"{'länk':14s} {'p_nom MW':>9} {'netto TWh':>10} {'trängsel %':>11} {'prisdiff':>9}")
for lnk in ac:
    if lnk not in flows.columns:
        continue
    p_nom = n.links.at[lnk, 'p_nom']
    z0, z1 = n.links.at[lnk, 'bus0'], n.links.at[lnk, 'bus1']
    trang = 100 * (flows[lnk].abs() > p_nom * 0.99).mean()
    diff = (prices[z0] - prices[z1]).abs().mean() if z0 in prices and z1 in prices else np.nan
    print(f"{lnk:14s} {p_nom:>9.0f} {twh(flows[lnk]):>10.2f} {trang:>10.1f}% {diff:>9.1f}")

print(f"\n{'zon':6s} {'import TWh':>11} {'export TWh':>11} {'netto TWh':>10}")
for z in ZONES:
    s = net[z]
    print(f"{z:6s} {twh(s.clip(lower=0)):>11.2f} {twh(-s.clip(upper=0)):>11.2f} "
          f"{twh(s):>10.2f}")
