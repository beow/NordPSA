"""hydrocomp — hydrogolv-kompensation. Räknar WV-golvet per hydrozon, korrigerar
zonpriserna (korrigerat = orginalpris − momentant WV) och tabellerar org/korrigerad
prisstatistik plus hur stor andel av tiden vattenvärdesgolvet är marginellt (sätter
priset) i varje zon.

INGEN jämförelse mot faktiska priser längre — den (+ de snygga model/komp/faktiskt-
graferna + compdiff) är arkiverad i minnet project_hydrocomp_faktiskt_archive och kan
klistras tillbaka. Skälet: faktiskt-jämförelsen är bara meningsfull för dispatch-
körningar vars period överlappar verkligheten; för expansion/kontrafaktiska system är
den missvisande (project_offset_calibration run106).

Förutsätter bootstrap.py (globala: prices, water_value, ZONES, LABEL, LABEL2, ROOT, cfg).
Producerar globala:
  calw  = korrigerade zonpriser för LABEL  (DataFrame, alla zoner)
  calw2 = korrigerade zonpriser för LABEL2 (DataFrame) om LABEL2 är satt, annars None
Helpern wv_adjusted(label) räknar calw för VILKEN körning som helst direkt ur dess
results-CSV:er → andra celler (t.ex. compareprice) kan kalla den för valfri körning.
"""
import numpy as np

# ── Inställning ──────────────────────────────────────────────────────────────
# WV-kompensationen är giltig BARA när vattenvärdet är ett lågt/platt/icke-bindande
# golv (baseline-regim). I knapphets-/importberoende-/expansionsscenarier stiger WV
# legitimt (hydro blir marginell prissättare mot dyr import) och då RADERAR
# kompensationen den verkliga prissignalen (ΔWV ≈ Δpris, se project_wv_comp_validity).
# Sätt NO_WV_COMP=True i sådana scenarier → wv_adjusted/calw/calw2 returnerar RÅpriser
# oförändrade, så alla downstream-celler (pricetable, compareprice, ...) jobbar på
# råpris utan egna ändringar.
NO_WV_COMP = False
# ─────────────────────────────────────────────────────────────────────────────

# Reservoarens BUDPRIS = vattenvärde + rörlig kostnad (VOM). När reservoaren är marginell blir
# alltså zonpriset = WV + VOM (ej WV exakt). Vi ankrar metriken på det budpriset; då räcker en
# minimal tolerans (toleranskänsligheten försvinner: ankrat på WV+VOM ger SE-N run121 6.7/7.2/8.5%
# för ±0.1/0.2/0.5 — mot 1/9/12% när man felaktigt ankrar på WV). VOM läses ur config.
MARG_TOL = 0.2   # EUR/MWh — tight, runt budpriset WV+VOM
VOM_HYDRO = cfg.get('costs', {}).get('hydro', {}).get('vom_eur_per_mwh', 0.0)


def wv_adjusted(label):
    """Korrigerat zonpris (org − momentant WV) för EN körning, läst direkt ur dess
    results-CSV:er (prices.csv + water_value.csv). Oberoende av bootstrap → funkar för
    vilken körning som helst, även LABEL2 som inte är inladdad. Hydrozoner korrigeras
    per timme; zoner utan reservoar (t.ex. DK) lämnas orörda. Saknas water_value.csv
    returneras råpriserna."""
    base = ROOT / 'results' / label
    pr = pd.read_csv(base / 'prices.csv', index_col=0, parse_dates=True)
    if NO_WV_COMP:                       # kompensation avstängd → råpriser
        return pr.copy()
    wv_path = base / 'water_value.csv'
    if not wv_path.exists():
        return pr.copy()
    wvv = pd.read_csv(wv_path, index_col=0, parse_dates=True)
    out = pr.copy()
    for z in ZONES:
        c = f'{z} hydro'
        if c in wvv.columns:
            out[z] = pr[z] - wvv[c].reindex(pr.index).fillna(0.0)
    return out


# Korrigerade priser för LABEL (och LABEL2 om satt) — tillgängliga för andra celler.
calw  = wv_adjusted(LABEL)
calw2 = wv_adjusted(LABEL2) if LABEL2 else None

if NO_WV_COMP:
    print("⚠️  NO_WV_COMP=True — WV-kompensation AVSTÄNGD. calw/calw2 = RÅpriser "
          "(downstream-celler visar råpris).")
    print("    Använd detta i knapphets-/import-/expansionsscenarier där WV stiger "
          "legitimt (se project_wv_comp_validity).")
    print("    WV-diagnostiken nedan visar ändå golvets storlek — det är den som "
          "motiverar avstängningen.\n")

if water_value is None:
    print("Ingen water_value.csv — kör med assign_all_duals för hydrogolv-kompensation.")
else:
    rows = []
    for z in ZONES:
        col = f'{z} hydro'
        has_h = col in water_value.columns
        wv = (water_value[col].reindex(prices.index).fillna(0.0)
              if has_h else pd.Series(0.0, index=prices.index))
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

if calw2 is not None:
    print(f"\ncalw2 beräknad för LABEL2 = {LABEL2} (korrigerade priser, för pris-jämförelse).")
