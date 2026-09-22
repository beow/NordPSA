"""Elbilsladdning: flexibel och oflexibel laddning."""

import pandas as pd
import pypsa

from nordpsa.network.core import MC_SLACK


def add_ev(n: pypsa.Network, cfg: dict, ev_profiles, ev_overrides,
            snapshots, dt_h: float, n_years: float) -> None:
    """Fordonsladdning per zon och fordonsklass. Två lägen (cfg['ev']['mode']):

      svk   — SvK-formulering: förbrukningen delas flex/oflex (flex_fraction). Oflexibel
              halva = fast AC-last (ladda-direkt-form). Flexibel halva = SvK-reservoar
              (batteri 12 GWh/flex-TWh, laddtak = batteri/charge_hours, fri tidsförflyttning).
      fleet — gammal per-fordon-modell (batteri/laddeffekt ur battery_kwh/charger_kw,
              hela flottan flexibel, hourly avail/minsoc-profiler).

    Bara smart laddning (ingen V2G). Antal fordon per zon från ev_overrides (CLI/scenario);
    E_tot = antal × annual_mwh. Flottan är exogen → Store ej extendable."""
    ecfg = cfg.get("ev") or {}
    if not ecfg.get("enabled") or ev_profiles is None or not ev_overrides:
        return
    if (ecfg.get("mode") or "svk") == "fleet":
        _add_ev_fleet(n, ecfg, ev_profiles, ev_overrides, snapshots, dt_h, n_years)
        return

    classes   = ecfg.get("classes") or {}
    mc_slack  = float(ecfg.get("slack_eur_per_mwh", MC_SLACK))
    flex_frac = float(ecfg.get("flex_fraction", 0.5))
    gwh_twh   = float(ecfg.get("flex_battery_gwh_per_twh", 12.0))
    chg_h     = float(ecfg.get("charge_hours", 10.0))
    soc_floor = float(ecfg.get("morning_soc_floor", 0.0))   # SOC-golv kl morning_soc_hour
    floor_h   = int(ecfg.get("morning_soc_hour", 6))

    # Morgon-SOC-golv: e_min_pu = soc_floor på snapshots där timmen = floor_h, annars 0.
    # Tvingar reservoaren körklar varje morgon → bryter perfekt-framsyn-flerdygnscoast
    # (med 10h-laddtak måste påfyllnaden börja kvällen innan; nattligt mönster faller ut).
    # Ingen V2G → SOC faller bara via körning, så inget golv övriga timmar behövs.
    emin_morning = None
    if soc_floor > 0:
        hours = pd.DatetimeIndex(snapshots).hour
        emin_morning = pd.Series(0.0, index=snapshots)
        emin_morning[hours == floor_h] = soc_floor
        if not (hours == floor_h).any():
            print(f"  Varning: EV morgon-SOC-golv {soc_floor:g} satt men ingen snapshot "
                  f"vid kl {floor_h:02d} (upplösning?) — golvet binder aldrig")

    for car in ("EV battery", "EV charger", "EV slack", "EV inflex"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    def _prof(col):
        return ev_profiles[col].reindex(snapshots).ffill().fillna(0.0)

    for zone, counts in ev_overrides.items():
        if zone not in n.buses.index:
            print(f"  Varning: EV-zon {zone} saknar AC-buss — hoppar över")
            continue
        for c, klass in classes.items():
            n_veh = float(counts.get(c, 0))
            if n_veh <= 0:
                continue
            e_tot    = n_veh * float(klass["annual_mwh"]) / 1e6     # TWh/år total konsumtion (grid-side)
            e_flex   = e_tot * flex_frac
            e_inflex = e_tot * (1.0 - flex_frac)

            # --- Oflexibel halva: fast AC-last, ladda-direkt-form (körprofil m. bryggad middag) ---
            ishape = _prof(f"{c}_inflex").clip(lower=0)
            ienerg = float(ishape.sum() * dt_h / 1e6 / n_years)    # TWh av råformen
            inflex = ishape * (e_inflex / ienerg) if ienerg > 0 else ishape * 0.0   # MW
            n.add("Load", f"{zone} EV {c} inflex", bus=zone, carrier="EV inflex", p_set=inflex)

            # --- Flexibel halva: SvK-reservoar (fri tidsförflyttning, batteri + 10h-laddtak) ---
            if e_flex <= 0:
                continue
            e_nom = gwh_twh * e_flex * 1e3        # GWh/flex-TWh × TWh × 1000 → MWh batteri
            p_chg = e_nom / chg_h                 # MW max laddeffekt (SvK 10h)
            drive = _prof(f"{c}_drive").clip(lower=0)
            denrg = float(drive.sum() * dt_h / 1e6 / n_years)
            drive = drive * (e_flex / denrg) if denrg > 0 else drive * 0.0

            evbus = f"{zone} EV {c}"
            n.add("Bus", evbus, carrier="EV battery")
            # SvK räknar grid-side förbrukning → η=1 (laddförluster ingår i annual_mwh).
            # SOC fri [0,1] utom morgon-SOC-golvet (e_min_pu kl floor_h) som tvingar
            # körklar reservoar varje morgon; laddning flyttas annars till billigaste timmar.
            store_kw = {} if emin_morning is None else {"e_min_pu": emin_morning}
            n.add("Store", f"{zone} EV {c} store", bus=evbus, carrier="EV battery",
                  e_nom=e_nom, e_cyclic=True, **store_kw)
            n.add("Link", f"{zone} EV {c} charger", bus0=zone, bus1=evbus,
                  carrier="EV charger", efficiency=1.0, p_nom=p_chg, marginal_cost=0.01)
            n.add("Load", f"{zone} EV {c} drive", bus=evbus, p_set=drive)
            n.add("Generator", f"{zone} EV {c} slack", bus=evbus, carrier="EV slack",
                  p_nom=1e6, marginal_cost=mc_slack)
            floor_txt = f", morgongolv {soc_floor:.0%}@{floor_h:02d}h" if emin_morning is not None else ""
            print(f"  → EV {zone}/{c} (SvK): {n_veh:.0f} fordon, E={e_tot*1e3:.0f} GWh/år, "
                  f"flex {flex_frac:.0%} → batteri {e_nom/1e3:.0f} GWh, laddtak {p_chg:.0f} MW "
                  f"({chg_h:.0f}h){floor_txt}; oflex-last {e_inflex*1e3:.0f} GWh/år")


def _add_ev_fleet(n: pypsa.Network, ecfg: dict, ev_profiles, ev_overrides,
                  snapshots, dt_h: float, n_years: float) -> None:
    """Gammal per-fordon-modell (mode: fleet). Hela flottan flexibel, batteri/laddeffekt
    ur battery_kwh/charger_kw, hourly avail (Link p_max_pu) + minsoc (Store e_min_pu)."""
    classes  = ecfg.get("classes") or {}
    mc_slack = float(ecfg.get("slack_eur_per_mwh", MC_SLACK))

    for car in ("EV battery", "EV charger", "EV slack"):
        if car not in n.carriers.index:
            n.add("Carrier", car)

    def _prof(col):
        return ev_profiles[col].reindex(snapshots).ffill().fillna(0.0)

    for zone, counts in ev_overrides.items():
        if zone not in n.buses.index:
            print(f"  Varning: EV-zon {zone} saknar AC-buss — hoppar över")
            continue
        for c, klass in classes.items():
            n_veh = float(counts.get(c, 0))
            if n_veh <= 0:
                continue
            eff   = float(klass.get("charge_efficiency", 0.9))
            e_nom = n_veh * float(klass["battery_kwh"]) / 1e3    # MWh (flottbatteri)
            p_chg = n_veh * float(klass["charger_kw"])  / 1e3    # MW (max samtidig laddeffekt)

            drive = _prof(f"{c}_drive").clip(lower=0)
            avail = _prof(f"{c}_avail").clip(0, 1)
            msoc  = _prof(f"{c}_minsoc").clip(0, 1)
            ann   = float(drive.sum() * dt_h / 1e6 / n_years)    # TWh av råformen
            tgt   = n_veh * float(klass["annual_mwh"]) / 1e6     # TWh-mål
            drive = drive * (tgt / ann) if ann > 0 else drive * 0.0

            evbus = f"{zone} EV {c}"
            n.add("Bus", evbus, carrier="EV battery")
            n.add("Store", f"{zone} EV {c} store", bus=evbus, carrier="EV battery",
                  e_nom=e_nom, e_cyclic=True, e_min_pu=msoc)
            n.add("Link", f"{zone} EV {c} charger", bus0=zone, bus1=evbus,
                  carrier="EV charger", efficiency=eff,
                  p_nom=p_chg, p_max_pu=avail, marginal_cost=0.01)
            n.add("Load", f"{zone} EV {c} drive", bus=evbus, p_set=drive)
            n.add("Generator", f"{zone} EV {c} slack", bus=evbus, carrier="EV slack",
                  p_nom=1e6, marginal_cost=mc_slack)
            print(f"  → EV {zone}/{c}: {n_veh:.0f} fordon, batteri {e_nom:.0f} MWh, "
                  f"laddeffekt {p_chg:.0f} MW (η={eff}), körbehov {tgt*1e3:.0f} GWh/år")
