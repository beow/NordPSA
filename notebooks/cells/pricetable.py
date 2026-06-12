"""pricetable — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).

Tabellerar LABEL-körningens zonpriser mot en referens:
  - default: FAKTISKA zonpriser (data/processed/market_prices.parquet)
  - om global LABEL2 är satt (i bootstrap): mot den körningens prices.csv i stället
Per zon: medel, referensmedel, bias, MAE, RMSE, korrelation; samt bias per år.
"""

# Referens: LABEL2-körning om satt, annars faktiska spotpriser.
LABEL2 = globals().get('LABEL2', None)
if LABEL2:
    ref = pd.read_csv(ROOT / 'results' / LABEL2 / 'prices.csv',
                      index_col=0, parse_dates=True).reindex(prices.index)
    ref_name, ref_src = LABEL2, 'körning'
else:
    ref = mkt_prices.copy()
    ref.index = pd.to_datetime(ref.index).tz_localize(None)
    ref = ref.reindex(prices.index, method='ffill')
    ref_name, ref_src = 'faktiskt', 'faktiska spotpriser'

def _stats(m, a):
    d = (m - a).dropna()
    return d.mean(), d.abs().mean(), (d ** 2).mean() ** 0.5, m.corr(a)

print(f'Pristabell: {LABEL}  vs  {ref_name}  ({ref_src})')
print(f"{'Zon':<7}{'modell':>9}{'ref':>9}{'bias':>8}{'MAE':>8}{'RMSE':>8}{'korr':>7}")
print('-' * 56)
for z in ZONES:
    if z not in ref.columns:
        continue
    b, mae, rmse, corr = _stats(prices[z], ref[z])
    print(f"{z:<7}{prices[z].mean():>9.1f}{ref[z].mean():>9.1f}"
          f"{b:>+8.1f}{mae:>8.1f}{rmse:>8.1f}{corr:>7.2f}")

# Bias per år (modell − referens)
years = sorted({d.year for d in prices.index})
if len(years) > 1:
    print(f"\nBias per år (modell − {ref_name}, EUR/MWh):")
    print(f"{'Zon':<7}" + "".join(f"{y:>8}" for y in years))
    print('-' * (7 + 8 * len(years)))
    for z in ZONES:
        if z not in ref.columns:
            continue
        row = []
        for y in years:
            m = prices.loc[prices.index.year == y, z]
            a = ref.loc[ref.index.year == y, z]
            row.append((m - a).dropna().mean())
        print(f"{z:<7}" + "".join(f"{v:>+8.1f}" for v in row))
