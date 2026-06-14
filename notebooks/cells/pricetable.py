"""pricetable — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, prices,
mkt_prices, ...) och — för kompenserade priser — att hydrocomp-cellen körts (globala
calw/calw2).

Tabellerar LABEL-körningens zonpriser mot en referens:
  - default: FAKTISKA zonpriser (data/processed/market_prices.parquet)
  - om global LABEL2 är satt (i bootstrap): mot den körningens priser i stället
Per zon: medel, referensmedel, bias, MAE, RMSE, korrelation; samt bias per år.

USE_CALW (default True): använd hydrogolv-kompenserade priser (calw/calw2) i stället för
råpriser. Modellsidan kompenseras alltid; LABEL2-referensen kompenseras också (calw2 →
äpplen-mot-äpplen). Faktiska spotpriser är verkliga och lämnas alltid OKOMPENSERADE.
"""

# ── Inledande val ───────────────────────────────────────────────────────────
USE_CALW = True    # True: hydrogolv-kompenserade priser (calw/calw2); False: råpriser
# ─────────────────────────────────────────────────────────────────────────────

LABEL2 = globals().get('LABEL2', None)
_calw  = globals().get('calw', None)
_calw2 = globals().get('calw2', None)

if USE_CALW and _calw is None:
    print("OBS: USE_CALW=True men calw saknas — kör hydrocomp-cellen först. Faller tillbaka på råpriser.\n")
    USE_CALW = False

# Modellsidan: kompenserat eller rått
model = _calw if USE_CALW else prices
mtag  = 'komp' if USE_CALW else 'rå'

# Referenssidan
if LABEL2:
    if USE_CALW and _calw2 is not None:
        ref = _calw2.reindex(model.index)
        ref_src = 'körning, komp'
    else:
        ref = pd.read_csv(ROOT / 'results' / LABEL2 / 'prices.csv',
                          index_col=0, parse_dates=True).reindex(model.index)
        ref_src = 'körning, rå'
    ref_name = LABEL2
else:
    ref = mkt_prices.copy()
    ref.index = pd.to_datetime(ref.index).tz_localize(None)
    ref = ref.reindex(model.index, method='ffill')
    ref_name, ref_src = 'faktiskt', 'faktiska spotpriser (alltid okomp.)'

def _stats(m, a):
    d = (m - a).dropna()
    return d.mean(), d.abs().mean(), (d ** 2).mean() ** 0.5, m.corr(a)

print(f'Pristabell: {LABEL} [{mtag}]  vs  {ref_name}  ({ref_src})')
print(f"{'Zon':<7}{'modell':>9}{'ref':>9}{'bias':>8}{'MAE':>8}{'RMSE':>8}{'korr':>7}")
print('-' * 56)
for z in ZONES:
    if z not in ref.columns:
        continue
    b, mae, rmse, corr = _stats(model[z], ref[z])
    print(f"{z:<7}{model[z].mean():>9.1f}{ref[z].mean():>9.1f}"
          f"{b:>+8.1f}{mae:>8.1f}{rmse:>8.1f}{corr:>7.2f}")

# Bias per år (modell − referens)
years = sorted({d.year for d in model.index})
if len(years) > 1:
    print(f"\nBias per år (modell [{mtag}] − {ref_name}, EUR/MWh):")
    print(f"{'Zon':<7}" + "".join(f"{y:>8}" for y in years))
    print('-' * (7 + 8 * len(years)))
    for z in ZONES:
        if z not in ref.columns:
            continue
        row = []
        for y in years:
            m = model.loc[model.index.year == y, z]
            a = ref.loc[ref.index.year == y, z]
            row.append((m - a).dropna().mean())
        print(f"{z:<7}" + "".join(f"{v:>+8.1f}" for v in row))
