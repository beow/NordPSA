"""flexassets — flexibilitetstillgångarnas EGEN drift: volym, urladdad energi,
cykler/år och SoC-spann per lager (batteri, H2, värme, EV), plus batteriets
dygnsmönster och elektrolysörens pris-respons.

Skiljer sig från flexprov.py (balansbidrag per tidsskala, bandpass-dekomponerat)
genom att titta på TILLGÅNGARNA snarare än vilken roll de spelar i residuallasten.

Extraherad ur notebooks/explore_results.py (f.d. cell 6), 2026-08-21 — den cellen
saknade motsvarighet i notebooks/cells/ innan den här filen skapades.

Förutsätter bootstrap.py (globala: n, ZONES, LABEL, dispatch, hydro_d, soc, prices,
plt).
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
store_soc = n.stores_t.e.reindex(snap) if not n.stores_t.e.empty else pd.DataFrame(index=snap)
link_p = n.links_t.p0.reindex(snap) if not n.links_t.p0.empty else pd.DataFrame(index=snap)

_ar = len(snap) * dt_h / 8760
print(f"{'lager':18s} {'volym GWh':>10} {'urladdat TWh':>13} {'cykler/år':>10} {'spann %':>8}")
for u in [x for x in hydro_d.columns if x in n.storage_units.index
          and n.storage_units.at[x, 'carrier'] == 'battery']:
    vol = n.storage_units.at[u, 'p_nom'] * n.storage_units.at[u, 'max_hours']
    ut = twh(hydro_d[u].clip(lower=0))
    cyk = (ut * 1e6 / vol) / _ar if vol else np.nan
    sv = 100 * (soc[u].max() - soc[u].min()) / vol if vol and u in soc.columns else np.nan
    print(f"{u:18s} {vol/1e3:>10.1f} {ut:>13.2f} {cyk:>10.1f} {sv:>8.0f}")
for s in [x for x in store_soc.columns if x in n.stores.index
          and n.stores.at[x, 'carrier'] in ('H2 store', 'EV battery', 'heat store')]:
    vol = n.stores.at[s, 'e_nom']
    if vol <= 0:
        print(f"{s:18s} {0.0:>10.1f} {'—':>13s} {'—':>10s} {'—':>8s}")
        continue
    sv = 100 * (store_soc[s].max() - store_soc[s].min()) / vol
    # Stores har ingen effektriktning i e; omsatt energi ≈ summan av positiva ändringar
    oms = float(store_soc[s].diff().clip(lower=0).sum()) / 1e6
    print(f"{s:18s} {vol/1e3:>10.1f} {oms:>13.2f} {(oms*1e6/vol)/_ar:>10.1f} {sv:>8.0f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
ax = axes[0]
batt = [u for u in hydro_d.columns if u in n.storage_units.index
        and n.storage_units.at[u, 'carrier'] == 'battery']
if batt:
    h = snap.hour
    ax.plot(range(24), hydro_d[batt].sum(axis=1).groupby(h).mean().reindex(range(24)).values,
            color='#1f78b4')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('timme'); ax.set_ylabel('MW (+ = urladdning)')
ax.set_title('Batteriets dygnsmönster')

ax = axes[1]
for s in [x for x in store_soc.columns if x.endswith(('H2 store', 'heat store'))][:6]:
    ax.plot(snap, store_soc[s] / 1e3, lw=0.7, label=s)
ax.set_ylabel('GWh'); ax.set_title('Sektorlagrens SOC'); ax.legend(fontsize=7)

ax = axes[2]
elyser = [lnk for lnk in link_p.columns if lnk in n.links.index
          and n.links.at[lnk, 'carrier'] == 'electrolyser']
if elyser:
    z0 = n.links.at[elyser[0], 'bus0']
    ax.scatter(prices[z0], link_p[elyser[0]], s=2, alpha=0.2, color='#1f78b4')
    ax.set_xlabel(f'{z0} pris EUR/MWh'); ax.set_ylabel('elektrolys MW')
    ax.set_title('Elektrolys vs pris')
plt.show()
