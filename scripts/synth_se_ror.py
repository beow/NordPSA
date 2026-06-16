"""
Syntetiserar run-of-river (RoR) för de svenska zonerna SE-N och SE-S.

Bakgrund
--------
Svenska kraftnät rapporterar INGEN separat run-of-river-kategori (ENTSO-E B11)
till ENTSO-E — all svensk vattenkraft läggs under B12 ("Hydro Water Reservoir").
Därför är `ror_entsoe_SE*`-filerna tomma och SE saknar den must-run-strömkraft
som NO får via faktisk B11-data.

Detta skript konstruerar en syntetisk RoR-profil:

  1. Formen lånas från det naturliga reservoarinflödet (`build_inflow`), som har
     den klassiska vårflodssäsongen. Äkta RoR följer den oreglerade avrinningen,
     vars säsong ligger nära inflödet (om än något skarpare/tidigare topp).
  2. Nivån skalas till en antagen RoR-årssumma per zon (SE_ROR_TARGET_TWH).
     ⚠️ PLATSHÅLLARE — kalibreras mot SVK/Energiföretagen. SE har relativt lite
     ren RoR (~3–5 TWh totalt), mest i SE2 (mellersta Norrland) och något i SE3.

Energibalans
------------
Reservoarinflödet (`inflow_nve_SE-*`) inkluderar idag hela vattenkraften (B11=0
drogs av). För att inte dubbelräkna när RoR-generatorn läggs till skrivs
`inflow_nve_SE-*` om med RoR-energin avdragen vecka för vecka — exakt som NO
(där inflödet = total − B11). Skriptet är idempotent: inflödet räknas alltid om
från rådata (reservoir_entsoe + hydro_entsoe), aldrig från en redan reducerad fil.

Utdata
------
  data/raw/ror_nve_{zone}_{year}.parquet     syntetisk RoR must-run-profil (timme)
  data/raw/inflow_nve_{zone}_{year}.parquet  reservoarinflöde (vecka), RoR avdraget

Användning:
    python scripts/synth_se_ror.py            # skriv alla SE-filer
    python scripts/synth_se_ror.py --plot     # spara även kontrollgraf i temp/
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.fetch_nve as fetch_nve
from scripts.fetch_nve import (
    SE_ZONE_MAP,
    YEARS,
    HOURS_PER_WEEK,
    RAW_DIR,
    build_inflow,
    build_reservoir_weekly_entsoe,
    raw_path,
    ror_profile_path,
)

# RoR-andel av total vattenkraftsproduktion. NVE-modellering (via SvK) anger 32 %
# för Sverige som helhet, men det överdränerar den lilla SE-S-zonen (uniform 32 %
# tömmer SE-S-magasinet mot 0). 20 % som kompromiss för båda SE-zonerna.
SE_ROR_FRACTION = 0.20

# Mål-kapacitetsfaktor för RoR-generatorn. p_nom = ror.max() (sätts i network.py),
# så profilen glättas tills topp/medel ≤ 1/CF → p_nom ≈ medeleffekt/CF. Undviker att
# den peakiga råinflödesformen (topp/medel ~4,4) skär ut orimligt mycket turbineffekt.
SE_ROR_CF = 0.5


def _smooth_to_cf(weekly_mw: pd.Series, cf: float) -> tuple[pd.Series, int]:
    """
    Glättar en veckovis RoR-effektprofil tills topp/medel ≤ 1/cf (energibevarande).
    Returnerar (glättad profil, använt fönster). Med p_nom = topp blir CF ≈ cf.
    """
    target_ratio = 1.0 / cf
    base_sum = float(weekly_mw.sum())
    prof = weekly_mw.copy()
    w = 1
    while prof.max() / prof.mean() > target_ratio and w < len(weekly_mw) // 2:
        w += 2
        prof = weekly_mw.rolling(w, center=True, min_periods=1).mean()
        prof = prof * (base_sum / float(prof.sum()))   # bevara energin
    return prof, w


def synth_zone(zone: str, fraction: float, ror_cf: float, plot_ax=None) -> None:
    bzns = SE_ZONE_MAP[zone]["bzns"]
    reservoir_wkly = build_reservoir_weekly_entsoe(bzns)

    for year in YEARS:
        # 1. Naturligt reservoarinflöde (vecka) — full vattenkraft, ger RoR-formen.
        infl = build_inflow(zone, year, reservoir_wkly)
        if infl.empty:
            print(f"  {zone} {year}: inget inflöde, hoppar")
            continue

        shape = infl["inflow_gwh"].clip(lower=0.0)
        if shape.sum() <= 0:
            print(f"  {zone} {year}: platt inflöde, hoppar")
            continue

        # 2. RoR-energi = andel av total vattenkraftsproduktion (≈ inflöde), formad
        #    efter inflödet. Veckovis effekt (MW), index på veckostart.
        ror_gwh0 = shape / shape.sum() * fraction * shape.sum()  # = shape*fraction
        weekly = pd.Series(
            (ror_gwh0 * 1000.0 / HOURS_PER_WEEK).values,
            index=pd.to_datetime(infl["week_start"]).dt.tz_convert("UTC"),
        ).sort_index()

        # 3. Glätta profilen till mål-CF (topp/medel ≤ 1/CF) → realistisk p_nom.
        #    Energibevarande, så RoR-andelen hålls = fraction.
        weekly, win = _smooth_to_cf(weekly, ror_cf)
        ror_gwh = weekly * HOURS_PER_WEEK / 1000.0   # tillbaka till GWh/vecka (glättad)

        # 4. Broadcasta veckoprofilen till timupplösning (konstant inom vecka).
        hourly_idx = pd.date_range(
            f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h", tz="UTC"
        )
        hourly = weekly.reindex(weekly.index.union(hourly_idx)).ffill().bfill()
        hourly = hourly.reindex(hourly_idx).ffill().bfill()

        # 5. Skriv RoR-timprofil (samma format som NO: timestampUTC, ror_mw).
        ror_df = hourly.rename("ror_mw").reset_index()
        ror_df.columns = ["timestampUTC", "ror_mw"]
        ror_df.to_parquet(ror_profile_path(zone, year), index=False)

        # 6. Dra av (glättad) RoR-energi från reservoarinflödet (undvik dubbelräkning).
        infl_out = infl.copy()
        infl_out["inflow_gwh"] = (infl_out["inflow_gwh"] - ror_gwh.values).clip(lower=0.0)
        infl_out["inflow_mw"] = infl_out["inflow_gwh"] * 1000.0 / HOURS_PER_WEEK
        infl_out.to_parquet(raw_path(zone, year), index=False)

        cf_act = hourly.mean() / hourly.max() if hourly.max() > 0 else 0
        print(
            f"  {zone} {year}: RoR {ror_gwh.sum()/1000:.2f} TWh ({100*fraction:.0f}%) "
            f"p_nom={hourly.max():.0f} MW (CF {cf_act:.2f}, glättn {win}v) → "
            f"reservoarinflöde {infl_out['inflow_gwh'].sum()/1000:.1f} TWh"
        )

        if plot_ax is not None:
            doy = pd.to_datetime(infl["week_start"]).dt.dayofyear
            plot_ax.plot(doy, weekly.values, lw=1.5, marker="o", ms=2, label=str(year))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Spara kontrollgraf i temp/")
    parser.add_argument("--smooth-weeks", type=int, default=1,
                        help="Glättningsfönster (veckor) på RESERVOARinflödet via fetch_nve.build_inflow. "
                             "1 = ingen glättning (default, skarp vårflod). Emulerar rumslig "
                             "diversitet i ett lumpat magasin (flodutjämning).")
    parser.add_argument("--ror-fraction", type=float, default=SE_ROR_FRACTION,
                        help=f"RoR-andel av total vattenkraftsproduktion (default {SE_ROR_FRACTION}, NVE).")
    parser.add_argument("--ror-cf", type=float, default=SE_ROR_CF,
                        help=f"Mål-CF för RoR-generatorn → p_nom≈medel/CF (default {SE_ROR_CF}).")
    args = parser.parse_args()

    # Sätt glättningsfönstret som build_inflow läser (modul-global i fetch_nve).
    fetch_nve.SMOOTH_WEEKS = args.smooth_weeks
    if args.smooth_weeks > 1:
        print(f"Flodutjämning: {args.smooth_weeks}-veckors glidande medel (energibevarande)\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    axes = None
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for i, zone in enumerate(["SE-N", "SE-S"]):
        print(f"\n=== {zone} (RoR {100*args.ror_fraction:.0f}% av prod, mål-CF {args.ror_cf}) ===")
        ax = axes[i] if axes is not None else None
        synth_zone(zone, args.ror_fraction, args.ror_cf, plot_ax=ax)
        if ax is not None:
            ax.set_title(f"{zone} — syntetisk RoR ({100*args.ror_fraction:.0f}% av prod)")
            ax.set_ylabel("RoR [MW]"); ax.set_xlabel("Dag på året")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

    if axes is not None:
        import matplotlib.pyplot as plt
        out = Path(__file__).resolve().parents[1] / "temp" / "se_ror_synth.png"
        out.parent.mkdir(exist_ok=True)
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print(f"\nGraf: {out}")

    print("\nKlar.")


if __name__ == "__main__":
    main()
