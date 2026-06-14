"""soclevels — reservoarernas fyllnadsgrad (SoC i % av max) per hydrozon. Plottar LABEL
och, om LABEL2 är satt i bootstrap, även LABEL2 som jämförelse — med korrekta rubriker.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, ZONES, cfg, soc, spill, plt).
"""
import yaml
with open(ROOT / 'config' / 'zones.yaml') as f:
    cfg = yaml.safe_load(f)

LABEL2 = globals().get('LABEL2', None)

# SoC-serier att plotta: (rubrik, DataFrame, färg, linjestil)
soc2 = None
if LABEL2:
    _p = ROOT / 'results' / LABEL2 / 'hydro_soc.csv'
    if _p.exists():
        soc2 = pd.read_csv(_p, index_col=0, parse_dates=True)
    else:
        print(f"OBS: hydro_soc.csv saknas för LABEL2 = {LABEL2}.")
series = [(LABEL, soc, 'royalblue', '-')]
if soc2 is not None:
    series.append((LABEL2, soc2, 'darkorange', '-'))

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
hydro_zones = [z for z in ZONES if cfg['zones'][z]['hydro_p_nom_mw'] > 0]

for ax, zone in zip(axes.flat, hydro_zones):
    p_nom  = cfg['zones'][zone]['hydro_p_nom_mw']
    max_h  = cfg['zones'][zone]['hydro_max_hours']
    max_e  = p_nom * max_h  # MWh
    col    = f'{zone} hydro'
    for run_label, s, color, ls in series:
        if col in s.columns:
            soc_pct = s[col] / max_e * 100
            ax.plot(soc_pct.index, soc_pct.values, lw=1, color=color, ls=ls, label=run_label)
    ax.axhline(100, color='red',    ls='--', lw=0.8, label='Max')
    ax.axhline(0,   color='orange', ls='--', lw=0.8, label='Min')
    ax.set_ylim(-5, 110)
    ax.set_title(f'{zone}  (max {max_e/1e6:.0f} TWh)')
    ax.set_ylabel('Fyllnadsgrad %')
    # Färre x-ticks så datumen inte skriver över varandra
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# Legend i första tomma rutan (till höger om sista zonen, t.ex. FI); dölj ev. resten
n_z = len(hydro_zones)
if n_z < len(axes.flat):
    leg_ax = axes.flat[n_z]
    leg_ax.axis('off')
    h, l = axes.flat[0].get_legend_handles_labels()
    leg_ax.legend(h, l, loc='center', fontsize=12, frameon=True, title='Körning / gränser')
for ax in axes.flat[n_z + 1:]:
    ax.set_visible(False)

_sub = f'Reservoarnivåer — {LABEL}' + (f'  vs  {LABEL2}' if soc2 is not None else '')
plt.suptitle(_sub, y=1.01)
plt.tight_layout()

# --- Spill per körning ---
for run_label, s, _c, _ls in series:
    sp = spill if run_label == LABEL else \
         pd.read_csv(ROOT / 'results' / run_label / 'hydro_spill.csv', index_col=0, parse_dates=True)
    total_spill = sp.sum()
    print(f'Total spill — {run_label} (TWh):')
    if (total_spill == 0).all():
        print('  Inget spill — modellen klarar sig utan!')
    else:
        for c, val in total_spill.items():
            if val > 0:
                print(f'  {c}: {val/1e6:.2f} TWh')
