"""hydrocomp — hydrogolv-kompensation. Räknar WV-golvet per hydrozon, korrigerar
zonpriserna (korrigerat = orginalpris − WV-golv) och tabellerar org/korrigerad
prisstatistik plus hur stor andel av tiden vattenvärdesgolvet är marginellt (sätter
priset) i varje zon.

INGEN jämförelse mot faktiska priser längre — den (+ de snygga model/komp/faktiskt-
graferna + compdiff) är arkiverad i minnet project_hydrocomp_faktiskt_archive och kan
klistras tillbaka. Skälet: faktiskt-jämförelsen är bara meningsfull för dispatch-
körningar vars period överlappar verkligheten; för expansion/kontrafaktiska system är
den missvisande (project_offset_calibration run106).

Förutsätter bootstrap.py (globala: prices, water_value, ZONES, LABEL).
Producerar global `calw` = DataFrame med korrigerade zonpriser (alla zoner).
"""
import numpy as np

# Reservoarens BUDPRIS = vattenvärde + rörlig kostnad (VOM). När reservoaren är marginell blir
# alltså zonpriset = WV + VOM (ej WV exakt). Vi ankrar metriken på det budpriset; då räcker en
# minimal tolerans (toleranskänsligheten försvinner: ankrat på WV+VOM ger SE-N run121 6.7/7.2/8.5%
# för ±0.1/0.2/0.5 — mot 1/9/12% när man felaktigt ankrar på WV). VOM läses ur config.
MARG_TOL = 0.2   # EUR/MWh — tight, runt budpriset WV+VOM
VOM_HYDRO = cfg.get('costs', {}).get('hydro', {}).get('vom_eur_per_mwh', 0.0)

if water_value is None:
    print("Ingen water_value.csv — kör med assign_all_duals för hydrogolv-kompensation.")
    calw = prices.copy()
else:
    calw = prices.copy()
    rows = []
    for z in ZONES:
        col = f'{z} hydro'
        has_h = col in water_value.columns
        wv = (water_value[col].reindex(prices.index).fillna(0.0)
              if has_h else pd.Series(0.0, index=prices.index))
        calw[z] = prices[z] - wv                              # korrigerat = org − momentant WV
        # Partitionering efter pris relativt reservoarens BUDPRIS = WV + VOM (momentant, per timme):
        #   pris < budpris − tol  ⇒ VRE-marg  (under golvet → sol/vind sätter, vattnet lagras)
        #   |pris − budpris| ≤ tol ⇒ golv-marg (reservoaren är marginell prissättare)
        #   pris > budpris + tol   ⇒ (resten)  (kopplad uppåt över granne/kontinent)
        gap = prices[z] - (wv + VOM_HYDRO)
        golv = float(gap.abs().le(MARG_TOL).mean() * 100) if has_h else np.nan
        vre  = float((gap < -MARG_TOL).mean() * 100) if has_h else np.nan
        rows.append({
            'WV-golv':     wv.mean() if has_h else np.nan,
            'org medel':   prices[z].mean(),
            'org std':     prices[z].std(),
            'korr medel':  calw[z].mean(),
            'korr std':    calw[z].std(),
            'Δ medel':     calw[z].mean() - prices[z].mean(),
            '% VRE-marg':  vre,
            '% golv-marg': golv,
        })
    wvtab = pd.DataFrame(rows, index=ZONES)
    wvtab.index.name = 'zon'

    print(f"Hydrogolv-kompensation ({LABEL}) — korrigerat pris = orginalpris − momentant WV (EUR/MWh)")
    print(f"  WV-golv     = reservoarens vattenvärde (dual på lagringsbalansen, water_value.csv); budpris = WV+VOM ({VOM_HYDRO})")
    print(f"  % VRE-marg  = andel tidssteg där pris < (WV+VOM) − {MARG_TOL} (under golvet → sol/vind sätter, vattnet lagras)")
    print(f"  % golv-marg = andel tidssteg där |pris − (WV+VOM)| < {MARG_TOL} (reservoaren sätter priset)")
    print(f"  (resten = pris > (WV+VOM) + {MARG_TOL}: kopplad uppåt över granne/kontinent)")
    print(f"  zoner utan reservoar (t.ex. DK) lämnas orörda (WV-golv = saknas)\n")
    print(wvtab.to_string(float_format=lambda v: f'{v:8.2f}', na_rep='   –'))
