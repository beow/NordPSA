"""hydrodispatch — vattenkraftsdispatch (reservoar + RoR) per hydrozon, månadsmedel.
Visar LABEL och, om LABEL2 är satt i bootstrap, även LABEL2. eSett faktisk produktion
är ett VAL (SHOW_ESETT). Samma formattering som soclevels: legend i tom ruta (höger om
sista zonen), färre x-ticks.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, ZONES, cfg, plt, mdates, pd).
"""

# ── Inledande val ────────────────────────────────────────────────────────────
SHOW_ESETT = True    # visa eSett faktisk produktion som referens
# ─────────────────────────────────────────────────────────────────────────────

LABEL2 = globals().get('LABEL2', None)


def _hydro_total_df(label):
    """res (dispatch_hydro: '{zon} hydro') + RoR (dispatch_generators: '{zon} hydro_ror'),
    per zon, för EN körning — direkt ur dess results-CSV:er (oberoende av bootstrap)."""
    base = ROOT / 'results' / label
    res = pd.read_csv(base / 'dispatch_hydro.csv',      index_col=0, parse_dates=True)
    gen = pd.read_csv(base / 'dispatch_generators.csv', index_col=0, parse_dates=True)
    out = {}
    for z in ZONES:
        s = res[f'{z} hydro'].clip(lower=0) if f'{z} hydro' in res.columns else None
        rc = f'{z} hydro_ror'
        if rc in gen.columns:
            r = gen[rc].clip(lower=0)
            s = r if s is None else s.add(r, fill_value=0.0)
        if s is not None:
            out[z] = s
    return pd.DataFrame(out)


# Modell-serier: (rubrik, DataFrame, färg)
series = [(LABEL, _hydro_total_df(LABEL), 'royalblue')]
if LABEL2:
    series.append((LABEL2, _hydro_total_df(LABEL2), 'darkorange'))

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
hydro_zones = [z for z in ZONES if cfg['zones'][z]['hydro_p_nom_mw'] > 0]

for ax, zone in zip(axes.flat, hydro_zones):
    for run_label, df, color in series:
        if zone in df.columns:
            mod = df[zone].resample('ME').mean()
            ax.plot(mod.index, mod.values, label=f'{run_label} (res+RoR)', lw=1.8, color=color)

    if SHOW_ESETT:
        frames = []
        for year in [2023, 2024, 2025]:
            raw = ROOT / 'data' / 'raw' / f'production_{zone}_{year}.parquet'
            if raw.exists():
                d = pd.read_parquet(raw)
                d['timestampUTC'] = pd.to_datetime(d['timestampUTC'], utc=True).dt.tz_localize(None)
                frames.append(d.set_index('timestampUTC')['hydro'])
        if frames:
            actual = pd.concat(frames).sort_index().resample('ME').mean()
            ax.plot(actual.index, actual.values, label='eSett faktisk', lw=1.4, ls='--', color='dimgray')

    ax.set_title(zone, fontweight='bold')
    ax.set_ylabel('MW')
    # Färre x-ticks så datumen inte skriver över varandra (som soclevels)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

# Legend i första tomma rutan (till höger om sista zonen); dölj ev. resten
n_z = len(hydro_zones)
if n_z < len(axes.flat):
    leg_ax = axes.flat[n_z]
    leg_ax.axis('off')
    h, l = axes.flat[0].get_legend_handles_labels()
    leg_ax.legend(h, l, loc='center', fontsize=11, frameon=True, title='Körning / referens')
for ax in axes.flat[n_z + 1:]:
    ax.set_visible(False)

_runs = ' vs '.join(s[0] for s in series) + (' vs eSett' if SHOW_ESETT else '')
plt.suptitle(f'Vattenkraft (reservoar+RoR) — {_runs} (månadsmedel)', y=1.01)
plt.tight_layout()
