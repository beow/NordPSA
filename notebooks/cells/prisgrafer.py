"""prisgrafer — modellpris vs referens per zon (veckomedel, 2×3). Referens = faktiska
spotpriser, eller LABEL2-körning om satt i bootstrap.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, ZONES, prices, mkt_prices, plt, mdates).
Priserna är RÅA, ingen hydrogolv-kompensation — se [[project_wv_comp_validity]].
"""

# Cellen kan importeras fristående (från explore_results.py) ELLER klistras in
# i en notebook-cell efter att bootstrap.py redan körts där (explore.ipynb).
# Guard: skippar importen om bootstrap redan satt sina globaler i detta namnrum.
if 'LABEL' not in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bootstrap import *  # noqa: F401,F403

LABEL2 = globals().get('LABEL2', None)

model = prices

# Referenssidan
if LABEL2:
    ref = pd.read_csv(ROOT / 'results' / LABEL2 / 'prices.csv',
                      index_col=0, parse_dates=True).reindex(model.index)
    ref_name = LABEL2
else:
    ref = mkt_prices.copy()
    ref.index = pd.to_datetime(ref.index).tz_localize(None)
    ref = ref.reindex(model.index, method='ffill')
    ref_name = 'faktiskt'

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)
for ax, zone in zip(axes.flat, ZONES):
    p = model[zone].resample('W').mean()
    ax.plot(p.index, p.values, lw=1.5, color='steelblue', label=f'Modell ({LABEL})')
    if zone in ref.columns:
        r = ref[zone].resample('W').mean()
        ax.plot(r.index, r.values, lw=1, color='tomato', ls='--', alpha=0.7, label=ref_name)
    ax.set_title(zone, fontweight='bold')
    ax.set_ylabel('EUR/MWh')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
axes[0, 0].legend()
plt.suptitle(f'Modellpris vs {ref_name} (veckomedel)', fontsize=13)
plt.tight_layout()
