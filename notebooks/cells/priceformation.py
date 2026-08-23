"""priceformation — vad sätter priset över HELA körningen, och hur mycket är
NTC-kopplat. Skiljer sig från marginal.py (EN given timme) genom att aggregera
över samtliga snapshots: för varje timme letas den dispatchade enhet som är PÅ
MARGINALEN (strikt mellan sina gränser) och vars marginalkostnad ligger närmast
zonpriset. För lager är budet lagrets EGNA marginalkostnad + vattenvärdet
(dualen på lagringsbalansen, water_value.csv) delat på verkningsgraden.

Extraherad ur notebooks/explore_results.py (f.d. cell 4), 2026-08-21 — den cellen
saknade motsvarighet i notebooks/cells/ innan den här filen skapades.

Förutsätter bootstrap.py (globala: n, ZONES, LABEL, dispatch, hydro_d, prices,
water_value, plt).
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

ZONE = 'SE-N'      # <-- zon att granska
snap = dispatch.index

mc_t = n.generators_t.marginal_cost.reindex(snap) \
    if not n.generators_t.marginal_cost.empty else pd.DataFrame(index=snap)

cand = {}
for g in dispatch.columns:
    if g not in n.generators.index or n.generators.at[g, 'bus'] != ZONE:
        continue
    if g in mc_t.columns:
        cand[g] = mc_t[g]
    elif n.generators.at[g, 'carrier'] in ('gas', 'slack', 'nuclear', 'thermal'):
        cand[g] = pd.Series(n.generators.at[g, 'marginal_cost'], index=snap)
if water_value is not None:
    smc_t = n.storage_units_t.marginal_cost.reindex(snap) \
        if not n.storage_units_t.marginal_cost.empty else pd.DataFrame(index=snap)
    for u in water_value.columns:
        if u in n.storage_units.index and n.storage_units.at[u, 'bus'] == ZONE:
            base = smc_t[u] if u in smc_t.columns else \
                pd.Series(n.storage_units.at[u, 'marginal_cost'], index=snap)
            eff = n.storage_units.at[u, 'efficiency_dispatch']
            cand[u] = base + water_value[u].reindex(snap) / eff

marg = pd.Series('okänd', index=snap, dtype=object)
best = pd.Series(np.inf, index=snap)
for name, mc in cand.items():
    if name in dispatch.columns:
        p_, p_nom = dispatch[name], n.generators.at[name, 'p_nom']
        on_margin = (p_ > 1) & (p_ < p_nom * 0.999)
        carrier = n.generators.at[name, 'carrier']
    else:
        p_, p_nom = hydro_d[name], n.storage_units.at[name, 'p_nom']
        on_margin = (p_.abs() > 1) & (p_ < p_nom * 0.999)
        carrier = n.storage_units.at[name, 'carrier']
    d = (prices[ZONE] - mc.reindex(snap)).abs()
    hit = on_margin & (d < best)
    marg[hit] = carrier
    best[hit] = d[hit]

# NTC-koppling: identiskt pris med en granne = priset sätts där, inte lokalt.
# Timmar utan lokal marginalenhet OCH med prislikhet mot en granne bokförs som
# NTC-kopplade i stället för 'okänd' — annars ser hälften av timmarna oförklarade ut.
partners = sorted({z for lnk in n.links.index if n.links.at[lnk, 'carrier'] == 'AC'
                   for z in (n.links.at[lnk, 'bus0'], n.links.at[lnk, 'bus1'])
                   if z in ZONES
                   and ZONE in (n.links.at[lnk, 'bus0'], n.links.at[lnk, 'bus1'])}
                  - {ZONE})
kopplad = pd.Series(False, index=snap)
for z in partners:
    kopplad |= (prices[ZONE] - prices[z]).abs() < 0.01
marg[(marg == 'okänd') & kopplad] = 'NTC-kopplad'

print(f'{ZONE}: prissättande teknik, % av timmarna  ({LABEL})')
print(marg.value_counts(normalize=True).mul(100).round(1).to_string())
print(f'\nNTC-koppling mot {partners}:')
print(f'  identiskt pris med minst en granne : {100*kopplad.mean():5.1f} % av timmarna')
print(f'  lokalt prissatt (trängsel åt alla håll) : {100*(~kopplad).mean():5.1f} %')

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
ax = axes[0]
for z in ZONES:
    ax.hist(prices[z], bins=80, histtype='step', label=z, lw=1.1)
ax.set_xlabel('EUR/MWh'); ax.set_ylabel('antal timmar')
ax.set_title('Prisfördelning per zon'); ax.legend(fontsize=8)

ax = axes[1]
win = max(int(24 * 7 / dt_h), 1)
if water_value is not None:
    for u in [c for c in water_value.columns if c.endswith('hydro')]:
        ax.plot(snap, water_value[u].rolling(win, min_periods=1).mean(), lw=1.0, label=u)
ax.plot(snap, prices[ZONE].rolling(win, min_periods=1).mean(), lw=1.4, color='black',
        label=f'{ZONE} pris')
ax.set_ylabel('EUR/MWh'); ax.set_title('Vattenvärde vs pris (veckomedel)')
ax.legend(fontsize=8)
plt.show()
