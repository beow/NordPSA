"""compareprice — zoomad prisjämförelse för EN zon över ett datumintervall: tidsserie
(plot) + statistiktabell (medel/P5/P95/SD). Jämför LABEL mot referens (LABEL2-körning om
satt, annars faktiska spotpriser) — samma input-val-logik som pricetable/prisgrafer.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, ZONES, prices, mkt_prices, plt, mdates, np).
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

# ── Inledande val ────────────────────────────────────────────────────────────
ZONE       = 'SE-S'                          # zon att jämföra
DATE_RANGE = ('2024-12-01', '2024-12-31')    # zoomfönster (YYYY-MM-DD); ändra fritt
# ─────────────────────────────────────────────────────────────────────────────

LABEL2 = globals().get('LABEL2', None)

model = prices

# Referenssidan: LABEL2-körning eller faktiskt
if LABEL2:
    ref = pd.read_csv(ROOT / 'results' / LABEL2 / 'prices.csv',
                      index_col=0, parse_dates=True).reindex(model.index)
    ref_name = LABEL2
else:
    ref = mkt_prices.copy()
    ref.index = pd.to_datetime(ref.index).tz_localize(None)
    ref = ref.reindex(model.index, method='ffill')
    ref_name = 'faktiskt'

sl   = slice(*DATE_RANGE)
runs = [(LABEL,    model, 'steelblue'),
        (ref_name, ref,   'tomato')]

fig, ax = plt.subplots(figsize=(16, 5))
for label, d, color in runs:
    if ZONE in d.columns:
        s = d.loc[sl, ZONE]
        ax.plot(s.index, s.values, lw=1.3, color=color, label=label)
ax.set_ylabel('EUR/MWh')
ax.set_title(f'{ZONE} pris {DATE_RANGE[0]} – {DATE_RANGE[1]}  —  {LABEL} vs {ref_name}')
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
