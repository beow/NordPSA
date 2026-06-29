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
# calw/calw2 är RÅpriser om hydrocomp kördes med NO_WV_COMP=True → spegla i taggen
mtag  = ('rå' if globals().get('NO_WV_COMP', False) else 'komp') if USE_CALW else 'rå'

# Referenssidan
if LABEL2:
    if USE_CALW and _calw2 is not None:
        ref = _calw2                       # behåll ref på SIN egen upplösning (ej reindex)
        ref_src = 'körning, komp'
    else:
        ref = pd.read_csv(ROOT / 'results' / LABEL2 / 'prices.csv',
                          index_col=0, parse_dates=True)
        ref_src = 'körning, rå'
    ref_name = LABEL2
else:
    ref = mkt_prices.copy()
    ref.index = pd.to_datetime(ref.index).tz_localize(None)
    ref = ref.reindex(model.index, method='ffill')
    ref_name, ref_src = 'faktiskt', 'faktiska spotpriser (alltid okomp.)'

# Harmonisera upplösning: om modell och ref har olika tidssteg (t.ex. 2h-run vs 3h-run)
# ger ren reindex glesa NaN-överlapp (var 6:e h) → bias/MAE/RMSE räknas på ett partiskt
# delurval och de visade medlen blir inkonsistenta med bias. Resampla då BÅDA till det
# grövre steget (medel) innan jämförelse. Samma-upplösning-fallet är oförändrat.
def _step(ix):
    return (ix[1] - ix[0]) if len(ix) > 1 else None
_ms, _rs = _step(model.index), _step(ref.index)
if _ms is not None and _rs is not None and _ms != _rs:
    _freq = max(_ms, _rs)
    model_cmp = model.resample(_freq).mean()
    ref_cmp = ref.resample(_freq).mean().reindex(model_cmp.index)
    print(f"OBS: modell-steg {_ms} ≠ ref-steg {_rs} → båda resamplade till {_freq} (medel) "
          f"för jämförbar bias/MAE/RMSE.\n")
else:
    model_cmp = model
    ref_cmp = ref.reindex(model.index)

def _stats(m, a):
    d = (m - a).dropna()
    return d.mean(), d.abs().mean(), (d ** 2).mean() ** 0.5, m.corr(a)

print(f'Pristabell: {LABEL} [{mtag}]  vs  {ref_name}  ({ref_src})')
print(f"{'Zon':<7}{'modell':>9}{'ref':>9}{'bias':>8}{'MAE':>8}{'RMSE':>8}{'korr':>7}")
print('-' * 56)
for z in ZONES:
    if z not in ref_cmp.columns:
        continue
    b, mae, rmse, corr = _stats(model_cmp[z], ref_cmp[z])
    print(f"{z:<7}{model_cmp[z].mean():>9.1f}{ref_cmp[z].mean():>9.1f}"
          f"{b:>+8.1f}{mae:>8.1f}{rmse:>8.1f}{corr:>7.2f}")

# Bias per år (modell − referens)
years = sorted({d.year for d in model_cmp.index})
if len(years) > 1:
    print(f"\nBias per år (modell [{mtag}] − {ref_name}, EUR/MWh):")
    print(f"{'Zon':<7}" + "".join(f"{y:>8}" for y in years))
    print('-' * (7 + 8 * len(years)))
    for z in ZONES:
        if z not in ref_cmp.columns:
            continue
        row = []
        for y in years:
            m = model_cmp.loc[model_cmp.index.year == y, z]
            a = ref_cmp.loc[ref_cmp.index.year == y, z]
            row.append((m - a).dropna().mean())
        print(f"{z:<7}" + "".join(f"{v:>+8.1f}" for v in row))

# ── Landaggregat: tre viktningsvyer ─────────────────────────────────────────
# LMA26 (SvK) räknar landpriser som elområdespriserna VOLYMVIKTADE med årlig
# elanvändning (spatialt mellan zoner; temporalt = rakt årsmedel per område).
# → kolumn 'energivikt' är den LMA-jämförbara siffran. För enzons-land = tidsmedel.
# 'tidsmedel' = oviktat medel av zonernas tidsmedel; 'lastvikt(h)' = timvis Σpris·last/Σlast
# (lyfter medlet via pris↔last-korrelation inom zonen — EJ LMA:s metod, men visas för kontext).
_n = globals().get('n', None)
if _n is not None:
    _w = _n.snapshot_weightings.objective.reindex(model_cmp.index).fillna(1.0)
    _ps = _n.loads_t.p_set
    _lbus = _n.loads.bus
    _E, _LZ = {}, {}
    for z in ZONES:
        _cols = [c for c in _lbus[_lbus == z].index if c in _ps.columns]
        _lz = (_ps[_cols].sum(axis=1).reindex(model_cmp.index).fillna(0.0)
               if _cols else pd.Series(0.0, index=model_cmp.index))
        _LZ[z] = _lz
        _E[z] = float((_lz * _w).sum())
    _country = {}
    for z in ZONES:                       # gruppera via prefix före '-' (SE-N→SE, DK→DK)
        _country.setdefault(z.split('-')[0], []).append(z)
    print(f"\nLandaggregat [{mtag}] (energivikt = LMA26-trogen):")
    print(f"{'Land':<8}{'tidsmedel':>11}{'energivikt':>12}{'lastvikt(h)':>13}")
    print('-' * 44)
    for c, zs in _country.items():
        zs = [z for z in zs if z in model_cmp.columns]
        if not zs:
            continue
        simple = float(pd.Series({z: model_cmp[z].mean() for z in zs}).mean())
        _Es = sum(_E[z] for z in zs)
        ew = (sum(_E[z] * model_cmp[z].mean() for z in zs) / _Es) if _Es > 0 else float('nan')
        _ln = sum(float((model_cmp[z] * _LZ[z] * _w).sum()) for z in zs)
        lw = (_ln / _Es) if _Es > 0 else float('nan')
        print(f"{c:<8}{simple:>11.1f}{ew:>12.1f}{lw:>13.1f}")
else:
    print("\n(Landaggregat hoppat över: global 'n' saknas — kör bootstrap-cellen.)")
