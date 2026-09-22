"""Driftrestriktioner på reservoaren: tim-/dygnsgolv, veckotak, bypass."""

import numpy as np
import pandas as pd
import pypsa
import xarray as xr


def _window_keys(snapshots: pd.DatetimeIndex, freq: str) -> "np.ndarray":
    """Fönsternyckel per snapshot: 'D' = kalenderdygn, 'W' = ISO-vecka (mån–sön).

    Nyckeln är fönstrets starttidpunkt, så partiella fönster i början/slutet av
    serien får en egen nyckel och behandlas separat (se _hours_per_window).
    """
    idx = pd.DatetimeIndex(snapshots)
    if freq == "D":
        return idx.normalize().to_numpy()
    if freq == "W":
        return idx.to_period("W-SUN").start_time.to_numpy()
    raise ValueError(f"okänd fönsterfrekvens: {freq}")


def hydro_operation_constraints(ocfg: dict):
    """extra_functionality-callback med driftrestriktioner för reservoarvattenkraft.

    Syftet är att hindra LP:n från de två orealistiska ytterligheterna: att stänga
    av vattenkraften helt under långa lågprisperioder (små magasin i älvsystemen
    skulle svämma över), och att köra på maxeffekt vecka efter vecka (vanligt
    utfall i ELLI-liknande modeller).

    Fyra villkor, alla valfria (0/None = av). Referenseffekten är StorageUnitens
    p_nom, dvs. RESERVOARDELEN efter en ev. RoR-split — inte hela flottan.

      min_hourly_frac   p_dispatch[t] ≥ f × p_nom                       (t.ex. 0.10)
      min_daily_frac    Σ_dygn p·w    ≥ f × p_nom × H_dygn              (t.ex. 0.20)
      max_weekly_frac   Σ_vecka p·w   ≤ f × p_nom × H_vecka             (t.ex. 0.77)
      bypass_spill      Σ_vecka spill·w ≥ κ × (Σ_vecka p·w − tröskel)

    H_fönster är den FAKTISKA summan av snapshot-vikter i fönstret, så villkoren
    blir korrekta på 1h/2h/3h-upplösning och partiella fönster i seriens kanter
    inte råkar bli hårdare än avsett.

    bypass_spill modellerar att hög uthållig produktion kräver att vatten spills
    förbi mindre stationer: en linjär gångjärnsfunktion som är slak tills veck0-
    produktionen når (max_weekly_frac − threshold_below_max) och därefter tvingar
    fram spill. Ingen binärvariabel behövs — villkoret är en undre gräns på PyPSA:s
    egen spill-variabel, som annars hålls nere av spill_cost.

    ⚠️ PyPSA:s spill är uppåt begränsad av tillrinningen i samma snapshot. En vecka
    med hög produktion men låg tillrinning kan därför göra modellen INFEASIBLE.
    Därför är bypass_spill default AV. Kör hydro_operation_feasibility_report()
    före solve för en förhandskontroll av de övriga villkoren.
    """
    min_h = float(ocfg.get("min_hourly_frac") or 0.0)
    min_d = float(ocfg.get("min_daily_frac") or 0.0)
    max_w = float(ocfg.get("max_weekly_frac") or 0.0)
    max_w_zone = dict(ocfg.get("max_weekly_frac_by_zone") or {})
    bcfg = dict(ocfg.get("bypass_spill") or {})

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        su = n.storage_units
        names = su.index[(su.carrier == "hydro") & (su.p_nom > 0)]
        if len(names) == 0:
            return

        m = n.model
        disp = m.variables["StorageUnit-p_dispatch"].sel(name=names)
        p_nom = xr.DataArray(su.loc[names, "p_nom"].to_numpy(dtype=float),
                             coords={"name": names}, dims="name")
        w = xr.DataArray(
            n.snapshot_weightings.stores.reindex(snapshots).to_numpy(dtype=float),
            coords={"snapshot": snapshots}, dims="snapshot")
        energy = disp * w          # MWh per snapshot

        if min_h > 0:
            m.add_constraints(disp >= min_h * p_nom,
                              name="custom-hydro_min_hourly")

        if min_d > 0:
            g = xr.DataArray(_window_keys(snapshots, "D"),
                             coords={"snapshot": snapshots}, dims="snapshot", name="day")
            hours = w.groupby(g).sum()
            m.add_constraints(energy.groupby(g).sum() >= min_d * p_nom * hours,
                              name="custom-hydro_min_daily")

        # Veckotak: globalt värde med per-zon-override. NaN = inget tak för zonen.
        fw = pd.Series(np.nan, index=names, dtype=float)
        if max_w > 0:
            fw[:] = max_w
        for zone, val in max_w_zone.items():
            su_name = f"{zone} hydro"
            if su_name in fw.index:
                fw[su_name] = float(val)
        fw = fw.dropna()
        fw = fw[fw > 0]

        if len(fw) == 0:
            return

        wk_names = pd.Index(fw.index)
        gw = xr.DataArray(_window_keys(snapshots, "W"),
                          coords={"snapshot": snapshots}, dims="snapshot", name="week")
        hours_w = w.groupby(gw).sum()
        fw_da = xr.DataArray(fw.to_numpy(dtype=float),
                             coords={"name": wk_names}, dims="name")
        pn_w = p_nom.sel(name=wk_names)
        weekly_prod = energy.sel(name=wk_names).groupby(gw).sum()

        m.add_constraints(weekly_prod <= fw_da * pn_w * hours_w,
                          name="custom-hydro_max_weekly")

        if not bcfg.get("active"):
            return
        if "StorageUnit-spill" not in m.variables:
            print("  Varning: bypass_spill begärd men StorageUnit-spill saknas "
                  "(ingen tillrinning?) — hoppar över")
            return

        below = float(bcfg.get("threshold_below_max", 0.10))
        coef = float(bcfg.get("coefficient", 1.0))
        spill = m.variables["StorageUnit-spill"]
        have = pd.Index([x for x in wk_names if x in spill.indexes["name"]])
        if len(have) == 0:
            return

        # κ per zon (coefficient_by_zone) med globalt 'coefficient' som fallback
        coef_zone = dict(bcfg.get("coefficient_by_zone") or {})
        kappa = pd.Series(coef, index=have, dtype=float)
        for zone, val in coef_zone.items():
            su_name = f"{zone} hydro"
            if su_name in kappa.index:
                kappa[su_name] = float(val)
        k_da = xr.DataArray(kappa.to_numpy(dtype=float),
                            coords={"name": have}, dims="name")

        thr = (fw_da.sel(name=have) - below) * p_nom.sel(name=have) * hours_w
        weekly_spill = (spill.sel(name=have) * w).groupby(gw).sum()
        # spill_vecka ≥ κ · (prod_vecka − tröskel); slak när prod < tröskel
        m.add_constraints(
            weekly_spill - k_da * weekly_prod.sel(name=have) >= -k_da * thr,
            name="custom-hydro_bypass_spill",
        )

    return _extra_functionality


def hydro_operation_feasibility_report(n: pypsa.Network, ocfg: dict) -> list:
    """Förhandskontroll av driftrestriktionerna mot tillrinningen.

    Med cyklisk SOC gäller årsproduktion = tillrinning − spill, så villkoren är
    ömsesidigt förenliga bara om

        min_daily_frac · p_nom · H  ≤  Σ tillrinning  ≤  max_weekly_frac · p_nom · H

    Returnerar en lista med varningstexter (tom = allt ser förenligt ut). Detta är
    ett NÖDVÄNDIGT men inte tillräckligt villkor — säsongsmässig obalans kan göra
    en enskild vecka infeasible även när årssumman går ihop.
    """
    warnings = []
    min_d = float(ocfg.get("min_daily_frac") or 0.0)
    max_w = float(ocfg.get("max_weekly_frac") or 0.0)
    max_w_zone = dict(ocfg.get("max_weekly_frac_by_zone") or {})

    bounds = hydro_operation_bounds(n)
    if bounds.empty:
        return warnings

    for zone, row in bounds.iterrows():
        p_nom, inflow, hours = row["p_nom_mw"], row["inflow_mwh"], row["hours"]
        cap_w = float(max_w_zone.get(zone, max_w) or 0.0)

        floor_mwh = min_d * p_nom * hours
        if min_d > 0 and floor_mwh > inflow:
            warnings.append(
                f"{zone}: min_daily_frac {min_d:.2f} kräver {floor_mwh/1e6:.1f} TWh "
                f"men tillrinningen är {inflow/1e6:.1f} TWh → INFEASIBLE "
                f"(max möjlig andel {inflow/(p_nom*hours):.2f})")
        if cap_w > 0:
            ceil_mwh = cap_w * p_nom * hours
            if ceil_mwh < inflow:
                warnings.append(
                    f"{zone}: max_weekly_frac {cap_w:.2f} tillåter bara "
                    f"{ceil_mwh/1e6:.1f} TWh men tillrinningen är {inflow/1e6:.1f} TWh "
                    f"→ tvingar {(inflow-ceil_mwh)/1e6:.1f} TWh spill "
                    f"(min möjlig andel {inflow/(p_nom*hours):.2f})")
        if min_d > 0 and cap_w > 0 and min_d > cap_w:
            warnings.append(f"{zone}: min_daily_frac {min_d:.2f} > max_weekly_frac "
                            f"{cap_w:.2f} → INFEASIBLE")
    return warnings


def hydro_operation_bounds(n: pypsa.Network) -> pd.DataFrame:
    """Per zon: reservoareffekt, tillrinning och vilken produktionsandel den motsvarar.

    Med cyklisk SOC är årsproduktionen låst till tillrinningen, så kolumnen
    'inflow_frac' (= tillrinning / (p_nom × timmar)) är precis det intervall som
    min_daily_frac måste ligga UNDER och max_weekly_frac måste ligga ÖVER för att
    restriktionerna ska vara inbördes förenliga. Används av
    hydro_operation_feasibility_report() och skrivs ut av nordpsa/run.py.
    """
    su = n.storage_units
    names = su.index[(su.carrier == "hydro") & (su.p_nom > 0)]
    if len(names) == 0:
        return pd.DataFrame()

    w = n.snapshot_weightings.stores.reindex(n.snapshots).to_numpy(dtype=float)
    hours = float(w.sum())
    rows = {}
    for name in names:
        p_nom = float(su.at[name, "p_nom"])
        inflow = 0.0
        if name in n.storage_units_t.inflow.columns:
            inflow = float((n.storage_units_t.inflow[name].to_numpy(dtype=float) * w).sum())
        rows[name.replace(" hydro", "")] = {
            "p_nom_mw": p_nom,
            "hours": hours,
            "inflow_mwh": inflow,
            "inflow_frac": inflow / (p_nom * hours) if p_nom * hours > 0 else np.nan,
        }
    return pd.DataFrame(rows).T
