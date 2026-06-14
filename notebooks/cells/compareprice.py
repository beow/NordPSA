"""compareprice — zoomad prisjämförelse för EN zon över ett datumintervall: tidsserie
(plot) + statistiktabell (medel/P5/P95/SD). Jämför LABEL mot referens (LABEL2-körning om
satt, annars faktiska spotpriser) — samma input-val-logik som pricetable/prisgrafer.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, ZONES, prices, mkt_prices, plt, mdates, np)
och — för kompenserade priser — hydrocomp-cellen (calw/calw2).
"""

# ── Inledande val ────────────────────────────────────────────────────────────
ZONE       = 'SE-S'                          # zon att jämföra
USE_CALW   = True                            # True: kompenserade priser (calw/calw2); False: rå
DATE_RANGE = ('2024-12-01', '2024-12-31')    # zoomfönster (YYYY-MM-DD); ändra fritt
# ─────────────────────────────────────────────────────────────────────────────

LABEL2 = globals().get('LABEL2', None)
_calw  = globals().get('calw', None)
_calw2 = globals().get('calw2', None)

if USE_CALW and _calw is None:
    print("OBS: USE_CALW=True men calw saknas — kör hydrocomp-cellen först. Faller tillbaka på råpriser.")
    USE_CALW = False

# Modellsidan (LABEL): kompenserat eller rått
model = _calw if USE_CALW else prices
mtag  = 'komp' if USE_CALW else 'rå'

# Referenssidan: LABEL2-körning (komp/rå) eller faktiskt (alltid okomp.)
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

sl   = slice(*DATE_RANGE)
runs = [(f'{LABEL} [{mtag}]', model, 'steelblue'),
        (ref_name,            ref,   'tomato')]

fig, ax = plt.subplots(figsize=(16, 5))
for label, d, color in runs:
    if ZONE in d.columns:
        s = d.loc[sl, ZONE]
        ax.plot(s.index, s.values, lw=1.3, color=color, label=label)
ax.set_ylabel('EUR/MWh')
ax.set_title(f'{ZONE} pris {DATE_RANGE[0]} – {DATE_RANGE[1]}  —  {LABEL} [{mtag}] vs {ref_name}')
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.tight_layout()

# Tabell: medel / P5 / P95 / SD över den plottade perioden
print(f'{ZONE} pris {DATE_RANGE[0]} – {DATE_RANGE[1]} (EUR/MWh):')
print(f"{'Fall':<32}{'Medel':>8}{'P5':>8}{'P95':>8}{'SD':>8}")
print('-' * 64)
for label, d, _ in runs:
    if ZONE not in d.columns:
        continue
    s = d.loc[sl, ZONE].dropna()
    if s.empty:
        print(f"{label:<32}{'(tomt intervall)':>32}")
        continue
    print(f"{label:<32}{s.mean():>8.1f}{np.percentile(s, 5):>8.1f}{np.percentile(s, 95):>8.1f}{np.std(s):>8.1f}")
