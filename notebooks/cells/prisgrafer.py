"""prisgrafer — modellpris vs referens per zon (veckomedel, 2×3). Referens = faktiska
spotpriser, eller LABEL2-körning om satt i bootstrap.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, ZONES, prices, mkt_prices, plt, mdates)
och — för kompenserade priser — hydrocomp-cellen (calw/calw2).

USE_CALW (default True): samma logik som pricetable-cellen — plotta hydrogolv-
kompenserade priser (calw/calw2) i stället för råpriser. Modellsidan kompenseras alltid;
LABEL2-referensen kompenseras också (calw2). Faktiska spotpriser lämnas alltid OKOMP.
"""

# ── Inledande val (samma som pricetable) ─────────────────────────────────────
USE_CALW = True    # True: hydrogolv-kompenserade priser (calw/calw2); False: råpriser
# ─────────────────────────────────────────────────────────────────────────────

LABEL2 = globals().get('LABEL2', None)
_calw  = globals().get('calw', None)
_calw2 = globals().get('calw2', None)

if USE_CALW and _calw is None:
    print("OBS: USE_CALW=True men calw saknas — kör hydrocomp-cellen först. Faller tillbaka på råpriser.")
    USE_CALW = False

# Modellsidan: kompenserat eller rått
model = _calw if USE_CALW else prices
mtag  = 'komp' if USE_CALW else 'rå'

# Referenssidan
if LABEL2:
    if USE_CALW and _calw2 is not None:
        ref = _calw2.reindex(model.index)
        ref_name = f'{LABEL2} (komp)'
    else:
        ref = pd.read_csv(ROOT / 'results' / LABEL2 / 'prices.csv',
                          index_col=0, parse_dates=True).reindex(model.index)
        ref_name = f'{LABEL2} (rå)'
else:
    ref = mkt_prices.copy()
    ref.index = pd.to_datetime(ref.index).tz_localize(None)
    ref = ref.reindex(model.index, method='ffill')
    ref_name = 'faktiskt'

fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=False)
for ax, zone in zip(axes.flat, ZONES):
    p = model[zone].resample('W').mean()
    ax.plot(p.index, p.values, lw=1.5, color='steelblue', label=f'Modell {mtag} ({LABEL})')
    if zone in ref.columns:
        r = ref[zone].resample('W').mean()
        ax.plot(r.index, r.values, lw=1, color='tomato', ls='--', alpha=0.7, label=ref_name)
    ax.set_title(zone, fontweight='bold')
    ax.set_ylabel('EUR/MWh')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
axes[0, 0].legend()
plt.suptitle(f'Modellpris [{mtag}] vs {ref_name} (veckomedel)', fontsize=13)
plt.tight_layout()
