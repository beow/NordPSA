"""compdiff — extraherad nyckelcell ur notebooks/explore_results.ipynb.
Förutsätter att bootstrap.py körts (globala: LABEL, ROOT, ZONES, cfg, n, dispatch,
hydro_d, soc, spill, flows, prices, water_value, mkt_prices, load + hjälpfunktioner).
"""

# Diff: hydrogolv-kompenserat modellpris − faktiskt pris, per zon (veckomedel).
# Använder calw (kompenserat) + act_h från cellen ovan. ~0 = bra kalibrering; kvarvarande
# struktur = icke-hydro-fel (SE-S/DK = ventilen/kontinentkoppling) eller sub-golv-spridning.
if water_value is None:
    print("Ingen water_value.csv — hydrogolv-kompensation ej körd, ingen diff att visa.")
else:
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
    for ax, z in zip(axes.flat, ZONES):
        if z not in act_h.columns:
            ax.set_visible(False); continue
        d    = (calw[z] - act_h[z])
        dw   = d.resample('W').mean()
        ax.plot(dw.index, dw.values, lw=1.2, color='purple')
        ax.axhline(0, color='black', lw=0.8, ls='--')
        ax.set_title(f"{z}  (bias {d.mean():+.1f}, MAE {d.abs().mean():.1f})", fontweight='bold')
        ax.set_ylabel('EUR/MWh')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    plt.suptitle(f'Diff: hydrogolv-kompenserat modellpris − faktiskt ({LABEL}, veckomedel)', fontsize=13)
    plt.tight_layout()
