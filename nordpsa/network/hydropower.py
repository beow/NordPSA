"""Vattenkraft: reservoar (StorageUnit) och strömkraft (must-run)."""

import pandas as pd
import pypsa

from nordpsa.profiles.hydro_inflow import (add_ror_hifreq, inflow_timeseries, load_nve_inflow,
                                          load_nve_ror)


NVE_INFLOW_ZONES = {"NO-N", "NO-S", "SE-N", "SE-S"}


# Zoner vars STRÖMKRAFT är syntetisk och därför saknar högfrekvent struktur.
# SE: Svenska kraftnät rapporterar ingen B11 till ENTSO-E, serierna byggs av
# scripts/synth_se_ror.py. FI: splittas av _synth_ror_profile ur den analytiska
# inflödeskurvan. ⛔ NO-N/NO-S saknas medvetet — deras B11 är RAPPORTERAD timvis
# (7 943/8 476 unika värden per år mot SE:s 52) och ska inte röras.
SYNTHETIC_ROR_ZONES = {"SE-N", "SE-S", "FI"}


def _synth_ror_profile(inflow: pd.Series, frac: float, target_cf: float,
                       alpha: float = 0.0) -> pd.Series:
    """Syntetisk run-of-river must-run-profil (MW) för zoner utan separat RoR-data.

    RoR-energin = andel `frac` av inflödesenergin, formad som en BLANDNING av
    oreglerad avrinning och jämn avtappning från uppströms magasin:

        RoR = (1 - alpha)·inflöde + alpha·platt          (energibevarande)

    ⛔ `alpha` tillkom 2026-08-19. Ren inflödesform (alpha=0) är MÄTT FEL: Norges
    uppmätta B11 har v/s 0,62-0,81 mot dess naturliga inflödes 0,09-0,10, dvs.
    verklig strömkraft är 7-8x flackare än avrinningen — den är till största delen
    vatten som redan passerat ett magasin. Se scripts/synth_se_ror.SE_ROR_ALPHA för
    anpassningen. Sätts per zon med `hydro_ror_regulated_frac` i zones.yaml.

    Profilen GLÄTTAS därefter tills topp/medel ≤ 1/target_cf (energibevarande) så att
    p_nom = profil.max() ger en realistisk RoR-CF ≈ target_cf. Speglar metoden i
    scripts/synth_se_ror._smooth_to_cf (där fast på veckodata; här på snapshot-data).
    Utan glättning skulle p_nom sättas av en enstaka vårflodstopp → orimligt låg CF
    och för stor reduktion av reservoarturbinen.
    """
    base  = inflow.clip(lower=0.0) * frac
    total = float(base.sum())
    if total <= 0:
        return base
    if alpha > 0:
        base = (1.0 - alpha) * base + alpha * (total / len(base))
    ratio = 1.0 / target_cf
    prof  = base
    w, n_ = 1, len(base)
    while prof.max() / max(prof.mean(), 1e-9) > ratio and w < n_ // 2:
        w    = w * 2 + 1
        prof = base.rolling(w, center=True, min_periods=1).mean()
        prof = prof * (total / float(prof.sum()))   # bevara energin
    return prof


def add_hydro(
    n:                    pypsa.Network,
    cfg:                  dict,
    hydro_params:         dict,
    snapshots:            pd.DatetimeIndex,
    ccfg:                 dict,
    actual_inflow:        bool = True,
    zone_prices:          dict | None = None,
    ror_hifreq:           float = 0.0,
    ror_hifreq_seed:      int = 0,
    ror_hifreq_tau_days:  float = 3.5,
) -> None:
    mc_default = ccfg["hydro"]["vom_eur_per_mwh"]
    for zone, zcfg in cfg["zones"].items():
        p_nom = zcfg.get("hydro_p_nom_mw", 0)
        max_h = zcfg.get("hydro_max_hours", 0)
        if p_nom == 0 or zone not in hydro_params:
            continue

        used_nve = bool(actual_inflow and zone in NVE_INFLOW_ZONES)
        if used_nve:
            inflow = load_nve_inflow(zone, snapshots)
            # Run-of-river: separat must-run-generator. Reservoarinflödet
            # (inflow_nve) exkluderar redan B11. Reducera reservoar-p_nom med
            # RoR-turbinkapaciteten så total turbinkapacitet bevaras.
            ror = load_nve_ror(zone, snapshots)
            ror_p_nom = float(ror.max())
            if (ror_hifreq > 0 and zone in SYNTHETIC_ROR_ZONES):
                _z = ror_hifreq_seed + 100 * (sorted(cfg["zones"]).index(zone) + 1)
                ror = add_ror_hifreq(ror, ror_p_nom, sigma=ror_hifreq,
                                     tau_days=ror_hifreq_tau_days, seed=_z)
                print(f"  → RoR-högfrekvens {zone}: sigma={ror_hifreq:g} "
                      f"tau={ror_hifreq_tau_days:g}d frö={_z}  "
                      f"p_nom {ror_p_nom:.0f} MW LÅST, veckoenergi bevarad")
            if ror_p_nom > 1.0:
                pu = (ror / ror_p_nom).clip(0, 1)
                n.add(
                    "Generator", f"{zone} hydro_ror",
                    bus=zone,
                    carrier="hydro",
                    p_nom=ror_p_nom,
                    p_nom_extendable=False,
                    p_min_pu=pu,
                    p_max_pu=pu,
                    marginal_cost=ccfg["hydro"].get("vom_ror_eur_per_mwh", mc_default),
                )
                # Bevara reservoarens energikapacitet (p_nom × max_hours) när
                # turbineffekten reduceras med RoR-andelen.
                cap_mwh = p_nom * max_h
                p_nom   = max(p_nom - ror_p_nom, 1.0)
                max_h   = cap_mwh / p_nom
        else:
            params       = hydro_params[zone]
            peak_gamma   = params.get("peak_gamma")
            target_annual = params.get("target_annual_twh")
            inflow = inflow_timeseries(
                params, snapshots,
                peak_gamma=peak_gamma,
                target_annual_twh=target_annual,
            )

        # Syntetisk run-of-river-split för zoner utan separat NVE/synth-RoR-fil
        # (t.ex. FI, som går via den parametriska grenen och därför aldrig nås av
        # NVE-RoR-grenen ovan). hydro_ror_fraction = andel av ENERGIN (produktionen)
        # som är icke-reglerbar strömkraft; formad efter avrinningen + glättad till
        # realistisk CF (se _synth_ror_profile). Reservoaren får resten av energin
        # ((1−frac)×inflöde) PLUS hela lagervolymen (p_nom×max_h bevaras när
        # turbineffekten reduceras med RoR:ns p_nom). Speglar scripts/synth_se_ror.
        ror_frac = zcfg.get("hydro_ror_fraction", 0.0)
        if ror_frac > 0 and f"{zone} hydro_ror" not in n.generators.index:
            ror_cf     = ccfg["hydro"].get("ror_target_cf", 0.5)
            ror_series = _synth_ror_profile(
                inflow, ror_frac, ror_cf,
                alpha=zcfg.get("hydro_ror_regulated_frac", 0.0))
            ror_p_nom  = float(ror_series.max())
            # ⚠️ Moduleras FÖRE avdraget nedan, så att TOTALVATTNET bevaras: får
            # strömkraften mer variation får reservoarinflödet komplementär variation.
            # Det är rätt fysik (summan är given) och gör dessutom FI jämförbar med
            # SE, där inflow_nve redan har RoR avdraget vecka för vecka.
            if (ror_hifreq > 0 and zone in SYNTHETIC_ROR_ZONES):
                _z = ror_hifreq_seed + 100 * (sorted(cfg["zones"]).index(zone) + 1)
                ror_series = add_ror_hifreq(ror_series, ror_p_nom, sigma=ror_hifreq,
                                            tau_days=ror_hifreq_tau_days, seed=_z)
                print(f"  → RoR-högfrekvens {zone}: sigma={ror_hifreq:g} "
                      f"tau={ror_hifreq_tau_days:g}d frö={_z}  "
                      f"p_nom {ror_p_nom:.0f} MW LÅST, veckoenergi bevarad")
            if ror_p_nom > 1.0:
                pu = (ror_series / ror_p_nom).clip(0, 1)
                n.add(
                    "Generator", f"{zone} hydro_ror",
                    bus=zone,
                    carrier="hydro",
                    p_nom=ror_p_nom,
                    p_nom_extendable=False,
                    p_min_pu=pu,
                    p_max_pu=pu,
                    marginal_cost=ccfg["hydro"].get("vom_ror_eur_per_mwh", mc_default),
                )
                dt_h    = (snapshots[1] - snapshots[0]).total_seconds() / 3600
                ror_twh = float(ror_series.sum()) * dt_h / 1e6
                inflow  = (inflow - ror_series).clip(lower=0.0)  # reservoaren får resten
                res_twh = float(inflow.sum()) * dt_h / 1e6
                cap_mwh = p_nom * max_h
                p_nom   = max(p_nom - ror_p_nom, 1.0)
                max_h   = cap_mwh / p_nom
                print(f"  → RoR-split {zone}: must-run {ror_p_nom:.0f} MW "
                      f"(CF={float(ror_series.mean())/ror_p_nom:.2f}, {ror_twh:.1f} TWh = "
                      f"{100*ror_twh/(ror_twh+res_twh):.0f}%); "
                      f"reservoar {p_nom:.0f} MW / {p_nom*max_h/1e6:.1f} TWh, "
                      f"inflöde {res_twh:.1f} TWh")

        # Reservoarens marginal_cost: hydro-mc-kurvan (expansion) eller platt VOM, där
        # SOC-dualen bär vattenvärdet (frysta kapaciteter).
        if zone_prices and zone in zone_prices:
            mc = zone_prices[zone].reindex(snapshots).ffill().clip(lower=mc_default)
        else:
            mc = mc_default

        n.add(
            "StorageUnit", f"{zone} hydro",
            bus=zone,
            carrier="hydro",
            p_nom=p_nom,
            max_hours=max_h,
            inflow=inflow,
            cyclic_state_of_charge=True,
            spill_cost=ccfg["hydro"].get("spill_cost_eur_per_mwh", 0.1),  # lågt → tillåt spill vid full reservoar
            p_min_pu=0.0,          # förbjud pumpning (ej pumpad-lagringshydro)
            efficiency_dispatch=1.0,
            marginal_cost=mc,
        )
