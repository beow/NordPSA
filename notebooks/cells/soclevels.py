"""soclevels — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# SoC i % av max kapacitet
import yaml
with open(ROOT / 'config' / 'zones.yaml') as f:
    cfg = yaml.safe_load(f)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
hydro_zones = [z for z in ZONES if cfg['zones'][z]['hydro_p_nom_mw'] > 0]

for ax, zone in zip(axes.flat, hydro_zones):
    p_nom  = cfg['zones'][zone]['hydro_p_nom_mw']
    max_h  = cfg['zones'][zone]['hydro_max_hours']
    max_e  = p_nom * max_h  # MWh
    col    = f'{zone} hydro'
    if col in soc.columns:
        soc_pct = soc[col] / max_e * 100
        ax.plot(soc_pct.index, soc_pct.values, lw=1, color='royalblue')
        ax.axhline(100, color='red', ls='--', lw=0.8, label='Max')
        ax.axhline(0,   color='orange', ls='--', lw=0.8, label='Min')
        ax.set_ylim(-5, 110)
        ax.set_title(f'{zone}  (max {max_e/1e6:.0f} TWh)')
        ax.set_ylabel('Fyllnadsgrad %')

for ax in axes.flat[len(hydro_zones):]:
    ax.set_visible(False)

plt.suptitle('Reservoarnivåer 2024', y=1.01)
plt.tight_layout()

# --- spill (infogad) ---

# Spill — hur mycket vatten spills?
total_spill = spill.sum()
print('Total spill (MWh):')
for col, val in total_spill.items():
    if val > 0:
        print(f'  {col}: {val/1e6:.2f} TWh')
if (total_spill == 0).all():
    print('  Inget spill — modellen klarar sig utan!')
