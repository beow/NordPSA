"""hydrocomp — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Per-scenario HYDROGOLV-KOMPENSATION (self-contained — kräver INGET faktiskt pris, funkar
# på expansionskörningar). Modellens prisgolv i en hydrozon = reservoarens vattenvärde (WV,
# dual på lagringsbalansen, sparas i water_value.csv med assign_all_duals). Pga 3-årig cyklisk
# SOC + perfekt framsyn lyfter WV HELA priskurvan (biaset är FLATT över alla pristerciler, ej
# bara golvtimmar) → kompensera genom att dra bort WV i varje timme.
#   Auto-selektiv: höga-WV inlåsta nordzoner (SE-N/NO-N/FI) kompenseras ~fullt; kontinent-
#   kopplade låga-WV-zoner (SE-S/DK, WV~0) lämnas orörda (deras bias = ventilen, ej hydrogolvet).
#   ADAPTERAR per scenario: WV ur körningen → faller automatiskt när VRE-expansion sänker
#   golvet (run82: WV 8.9→0.7 → ingen överkompensering). Detta löser att den FASTA dispatch-
#   offseten ej överförs till expansion (golvet är merit-order-styrt). Se project_offset_calibration.
FLOOR_TARGET = 0.0   # reell vattenvärdesnivå i inlåsta överskottszoner (~0 EUR/MWh)

if water_value is None:
    print("Ingen water_value.csv — kör med assign_all_duals för hydrogolv-kompensation.")
else:
    hz = [z for z in ZONES if f'{z} hydro' in water_value.columns]
    calw = prices.copy()
    print(f"Hydrogolv-kompensation ({LABEL}) — drar bort vattenvärdet per hydrozon (EUR/MWh)")
    print(f"{'Zon':<7}{'WV-medel':>10}{'WV-std':>8}{'pris rå':>10}{'→ komp':>9}{'Δ':>8}")
    print('-' * 52)
    for z in hz:
        wv = water_value[f'{z} hydro'].reindex(prices.index).fillna(0.0)
        calw[z] = prices[z] - (wv - FLOOR_TARGET)
        print(f"{z:<7}{wv.mean():>10.1f}{wv.std():>8.2f}{prices[z].mean():>10.1f}"
              f"{calw[z].mean():>9.1f}{calw[z].mean()-prices[z].mean():>+8.1f}")

    # --- Jämförelse mot FAKTISKA priser (meningsfullt för dispatch-körningar vars period
    #     överlappar verkligheten; för rena framtidsscenarier finns inget faktiskt att jämföra) ---
    act_h = mkt_prices.copy()
    act_h.index = pd.to_datetime(act_h.index).tz_localize(None)
    act_h = act_h.reindex(prices.index, method='ffill')

    def _cmp(m, a):
        d = (m - a).dropna()
        return d.mean(), d.abs().mean(), (d ** 2).mean() ** 0.5

    print()
    print("Kompenserat modellpris vs FAKTISKT:", LABEL)
    print(f"{'Zon':<7}{'komp':>8}{'faktiskt':>10}{'bias':>8}{'MAE':>7}{'RMSE':>7}{'rå bias':>9}")
    print('-' * 56)
    for z in ZONES:
        if z not in act_h.columns:
            continue
        b, mae, rmse = _cmp(calw[z], act_h[z])
        b_raw = (prices[z] - act_h[z]).mean()
        print(f"{z:<7}{calw[z].mean():>8.1f}{act_h[z].mean():>10.1f}{b:>+8.1f}{mae:>7.1f}{rmse:>7.1f}{b_raw:>+9.1f}")
    print()
    print("Inlåsta nordzoner (SE-N/NO-N/NO-S/FI): bias → ~0, låg MAE = WV-subtraktionen löser dem.")
    print("SE-S/DK kvar (WV~0, orörda) — deras bias = ventilen/kontinentkoppling, separat problem.")
    print("OBS: korrigerar NIVÅN, ej sub-golv-spridningen (negativa tim) — RMSE därför kvar.")

    # Plott: rå modell / hydrogolv-kompenserat / faktiskt (veckomedel) per zon
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for ax, z in zip(axes.flat, ZONES):
        ax.plot(prices[z].resample('W').mean(), lw=1.0, color='lightsteelblue', label='Rå modell')
        ax.plot(calw[z].resample('W').mean(),   lw=1.5, color='steelblue',     label='Hydrogolv-komp.')
        if z in act_h.columns:
            ax.plot(act_h[z].resample('W').mean(), lw=1.0, color='tomato', ls='--', alpha=0.75, label='Faktiskt')
        tag = f"  (WV {water_value[f'{z} hydro'].mean():.1f})" if f'{z} hydro' in water_value.columns else "  (ej hydro)"
        ax.set_title(z + tag, fontweight='bold')
        ax.set_ylabel('EUR/MWh')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    axes[0, 0].legend(fontsize=8)
    plt.suptitle(f'Hydrogolv-kompenserat modellpris vs faktiskt ({LABEL}, veckomedel)', fontsize=13)
    plt.tight_layout()
