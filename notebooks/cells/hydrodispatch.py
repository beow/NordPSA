"""hydrodispatch — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Vattenkraft modell (reservoar + RoR) vs eSett faktisk — hela perioden
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

hydro_zones = [z for z in ZONES if cfg['zones'][z]['hydro_p_nom_mw'] > 0]

for ax, zone in zip(axes.flat, hydro_zones):
    # Modell = reservoar + run-of-river
    mod = zone_hydro_total(zone).loc['2023':'2025'].resample('ME').mean()
    ax.plot(mod.index, mod.values, label='Modell (res+RoR)', lw=2)

    # Faktisk (eSett) — alla år
    frames = []
    for year in [2023,2024,2025]:
        raw = ROOT / 'data' / 'raw' / f'production_{zone}_{year}.parquet'
        if raw.exists():
            df = pd.read_parquet(raw)
            df['timestampUTC'] = pd.to_datetime(df['timestampUTC'], utc=True).dt.tz_localize(None)
            frames.append(df.set_index('timestampUTC')['hydro'])
    if frames:
        actual = pd.concat(frames).sort_index().resample('ME').mean()
        ax.plot(actual.index, actual.values, label='eSett faktisk', lw=1.5, ls='--')

    ax.set_title(zone)
    ax.set_ylabel('MW')
    ax.legend(fontsize=8)

for ax in axes.flat[len(hydro_zones):]:
    ax.set_visible(False)

plt.suptitle('Vattenkraft: modell (reservoar+RoR) vs eSett faktisk 2023–2025 (månadsmedel)', y=1.01)
plt.tight_layout()
