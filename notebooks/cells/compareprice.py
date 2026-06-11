"""compareprice — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# SE-S pris: basfall / batteri / kärnkraft (1h) — ändra DATE_RANGE för att zooma
DATE_RANGE = ('2025-01-01', '2025-01-31')   # ← ändra här (YYYY-MM-DD)
#DATE_RANGE = ('2023-01-01', '2025-12-31')   # ← ändra här (YYYY-MM-DD)
BAT_ZONE   = 'SE-S'

p_base = pd.read_csv(ROOT / 'results' / 'run68_baseline_1h'        / 'prices.csv', index_col=0, parse_dates=True)
p_bat  = pd.read_csv(ROOT / 'results' / 'run82_wind_ses_9893_1h' / 'prices.csv', index_col=0, parse_dates=True)
#p_bat  = pd.read_csv(ROOT / 'results' / 'run72_oc_budget_vre_ses_1h' / 'prices.csv', index_col=0, parse_dates=True)
#p_nuc  = pd.read_csv(ROOT / 'results' / 'run70_nuclear_2500_1h'    / 'prices.csv', index_col=0, parse_dates=True)
#p_nuc  = pd.read_csv(ROOT / 'results' / 'run71_nuclear_disp_1h'    / 'prices.csv', index_col=0, parse_dates=True)

sl   = slice(*DATE_RANGE)
runs = [('Basfall (run68)',           p_base, 'red'),
        ('Vindkraft +10GW (run82)', p_bat,  'green')]
#        ('+2.5 GW kärnkraft (run70)',   p_nuc,  'purple')]

fig, ax = plt.subplots(figsize=(16, 5))
for label, d, color in runs:
    ax.plot(d.loc[sl, BAT_ZONE].index, d.loc[sl, BAT_ZONE], lw=1.3, color=color, label=label)
ax.set_ylabel('EUR/MWh')
ax.set_title(f'{BAT_ZONE} pris {DATE_RANGE[0]} – {DATE_RANGE[1]} (1h) — basfall vs batteri vs kärnkraft')
ax.legend(); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
plt.tight_layout()

# Tabell: medel / P5 / P95 över den plottade perioden
print(f'{BAT_ZONE} pris {DATE_RANGE[0]} – {DATE_RANGE[1]} (EUR/MWh):')
print(f"{'Fall':<28}{'Medel':>8}{'P5':>8}{'P95':>8}{'SD':>8}")
print('-' * 60)
for label, d, _ in runs:
    s = d.loc[sl, BAT_ZONE]
    print(f"{label:<28}{s.mean():>8.1f}{np.percentile(s, 5):>8.1f}{np.percentile(s, 95):>8.1f}{np.std(s):>8.1f}")
