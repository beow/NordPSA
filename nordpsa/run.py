"""En körning från upplösta inställningar till resultat.

    s = settings.resolve(...)      # eller resolve_from_run(...)
    run(s, label="run500_x")

Stegen, i ordning: config ← världen, indata, nätverk, efterbyggnad (torrår,
potentialtak), bivillkor, läget (frysning, VRE-avräkning), lösning, sparning.
"""
from __future__ import annotations

import datetime
import shlex
import subprocess
import sys

from nordpsa import inputs as inp
from nordpsa import modes, settings, solve, world
from nordpsa.analysis.stability import stability_data, stability_report, write_stability_reports
from nordpsa.constraints import (hydro_bid_ladder, hydro_operation_bounds,
                                 hydro_operation_constraints, hydro_operation_feasibility_report,
                                 scr_joint, stability_constraints,
                                 stability_feasibility_report)
from nordpsa.network import build_network
from nordpsa.settings import RESULTS_DIR, ROOT


def _git_commit() -> str:
    """Aktuell git-commit (kort hash + ev. 'dirty'). Tom sträng om ej git."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
        return h + (" (dirty)" if dirty else "")
    except Exception:
        return ""


def summary_flags(s: dict) -> list[str]:
    """Kort flaggrad för run_meta.txt. ⚠️ Analysverktyg läser budtrappan ur token
    `bidladder-K_BREDD` här — behåll formatet."""
    r, sc, mode = s["run"], s["scenario"], s["run"]["mode"]
    f = [mode, f"värld-{r['world']}"]
    if mode == "dispatch":
        d = s["dispatch"]
        f += [f"vrecurt-{d['vre_curtailment_cost']:g}", f"rolling-{d['rolling_weeks']}w",
              f"lookahead-{d['lookahead_weeks']}w", "termkurva"]
        if r["capacities"] not in ("optimized", "config"):
            f.append(f"från-{r['capacities']}")
    f.append(f"spill-{s[mode]['spill_cost']:g}")
    bl = s["hydro"]["bid_ladder"]
    f.append(f"bidladder-{bl[0]}_{bl[1]:g}" if bl else "no-bidladder")
    rh = s["hydro"]["ror_hifreq"]
    f.append(f"rorhifreq-{rh['sigma']:g}_tau{rh['tau_days']:g}_seed{rh['seed']}"
             if rh else "no-rorhifreq")
    if sc["cost"]:      f.append(f"cost-{sc['cost']}")
    if sc["demand"]:    f.append(f"demand-{sc['demand']}")
    if sc["low_hydro"]: f.append(f"lowhydro-{sc['low_hydro']['factor']:g}")
    if s["voll"] is not None: f.append(f"voll{int(s['voll'])}")
    if s["hydro"]["restrictions"]: f.append("hydro-restrictions")
    if s["heat"]["enabled"]: f.append("heat")
    if s["battery"]["endogenous"]:
        ge = s["battery"]["gfm_extra_eur_per_kw"]
        f.append(f"battinvest-{s['battery']['hours']:g}h_"
                 + ("gfm-only" if ge is not None and float(ge) == 0 else "gfl+gfm")
                 + ("" if ge is None or float(ge) == 0 else f"_gfmextra{float(ge):g}")
                 + ("" if float(s["battery"]["cost_scale"]) == 1.0
                    else f"_cost×{float(s['battery']['cost_scale']):g}"))
    if s["syncon"]["enabled"]: f.append("syncon")
    st = s["stability"]
    if st["enabled"]:
        d = s["dispatch"]
        if st["ek_system_gws"] or st["ek_zone_floor_gws"]:
            f.append("stability" + (f"-ek{st['ek_system_gws']:g}" if st["ek_system_gws"] else "")
                     + "".join(f"_{z}{v:g}" for z, v in st["ek_zone_floor_gws"].items()))
            pen = d["stability_slack_penalty"]
            f.append(f"stabslack-{pen:g}" if pen else "stab-hard")
        if st["scr_min"]:
            pen = d["stability_scr_slack_penalty"]
            f.append(f"scr-{st['scr_min']:g}" + (f"_slack{pen:g}" if pen else "_hard"))
            for h, mem in (st["scr_joint"] or {}).items():
                f.append(f"scr-joint-{h}" + "".join(f"+{z}{float(a):g}" for z, a in mem.items()))
    for e in r["experiments"]:
        f.append(f"exp-{e}")
    for spec in r["set"]:
        f.append(f"set-{spec}")
    return f


def write_run_meta(label: str, s: dict, desc: str | None, n_steps: int) -> None:
    """results/<label>/run_meta.txt (läsbar sammanfattning) + run_config.yaml (den
    fullständiga inställningen, som `dispatch --from` läser). Skrivs före lösningen
    så att även avbrutna körningar är självbeskrivande."""
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    settings.save(s, label)
    p = s["period"]
    lines = [
        f"output:      {label}",
        f"syfte:       {desc or '(ingen --desc angiven)'}",
        f"tid:         {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"git:         {_git_commit() or '(ej git)'}",
        f"upplösning:  {p['resolution_hours']}h",
        f"år:          {p['year'] or '2023-2025'}",
        f"tidssteg:    {n_steps}",
        f"flaggor:     {', '.join(summary_flags(s))}",
        f"argv:        {shlex.join(sys.argv)}",
    ]
    (out / "run_meta.txt").write_text("\n".join(lines) + "\n")
    print("  → run_meta.txt + run_config.yaml")


def _hydro_constraints(n, cfg: dict, s: dict) -> list:
    """Bivillkor på reservoarvattenkraften: driftrestriktioner och budtrappa."""
    callbacks = []
    ocfg = settings.merge(dict(cfg.get("hydro_operation") or {}), s["hydro"]["operation"],
                          "hydro.operation", strict=False)
    if s["hydro"]["restrictions"]:
        ocfg["active"] = True
    if ocfg.get("active"):
        by_zone = ocfg.get("max_weekly_frac_by_zone") or {}
        print("Hydro-driftrestriktioner (reservoardelen):")
        _gw = float(ocfg.get("max_weekly_frac", 0) or 0)
        _wk = (f"max vecka {_gw:.2f} × max vecka" if _gw > 0 else "max vecka: enbart per zon")
        print(f"  min tim {ocfg.get('min_hourly_frac', 0) or 0:.2f} × p_nom, "
              f"min dygn {ocfg.get('min_daily_frac', 0) or 0:.2f} × max dygn, "
              + _wk + (f" ({by_zone})" if by_zone else ""))
        if (ocfg.get("bypass_spill") or {}).get("active"):
            bs = ocfg["bypass_spill"]
            print(f"  bypass-spill PÅ: κ={bs.get('coefficient')} över "
                  f"(veckotak − {bs.get('threshold_below_max', 0.10)})")
        bounds = hydro_operation_bounds(n)
        if not bounds.empty:
            print(f"  {'zon':7s}{'p_nom MW':>10s}{'tillrinn TWh':>14s}"
                  f"{'andel av max':>14s}   förenligt med [min dygn, max vecka]")
            lo = float(ocfg.get("min_daily_frac", 0) or 0)
            for zone, row in bounds.iterrows():
                hi = float(by_zone.get(zone, ocfg.get("max_weekly_frac", 0)) or 0)
                frac = row["inflow_frac"]
                ok = (frac >= lo) and (hi <= 0 or frac <= hi)
                print(f"  {zone:7s}{row['p_nom_mw']:10.0f}{row['inflow_mwh']/1e6:14.2f}"
                      f"{frac:14.3f}   {'ja' if ok else 'NEJ'}")
        for w in hydro_operation_feasibility_report(n, ocfg):
            print(f"  ⚠️  {w}")
        callbacks.append(hydro_operation_constraints(ocfg))

    bl = s["hydro"]["bid_ladder"]
    if bl:
        k, width = int(bl[0]), float(bl[1])
        offs = [width * ((i + 0.5) / k - 0.5) for i in range(k)]
        hyd = [su for su in n.storage_units.index
               if n.storage_units.at[su, "carrier"] == "hydro"
               and float(n.storage_units.at[su, "p_nom"]) > 0.0]
        print(f"  → HYDROBUDTRAPPA: {k} nivåer, bredd {width:g} EUR/MWh, {len(hyd)} reservoarer")
        print("       avvikelser: " + ", ".join(f"{o:+.1f}" for o in offs)
              + "  (medel 0,0 ⇒ NIVÅN oförändrad, bara spridningen)")
        callbacks.append(hydro_bid_ladder(k, width))
    return callbacks


def _stability_sdata(cfg: dict, s: dict) -> dict:
    st = s["stability"]
    for zone in list(st["ek_zone_floor_gws"]) + list(st["sync_weight"]):
        if zone not in cfg["zones"]:
            raise SystemExit(f"stability: okänd zon {zone!r}")
    sdata = stability_data(st["tech"], st["sync_weight"], cfg)
    sdata["scr_joint"] = st["scr_joint"]
    try:
        scr_joint(sdata, list(cfg["zones"]))
    except ValueError as e:
        raise SystemExit(str(e))
    return sdata


def _stability_constraints(n, cfg: dict, s: dict) -> list:
    """Krav på rotationsenergi och nätstyrka. Anropas EFTER frysningen av kapaciteterna.
    Mjuka i dispatch (straff), hårda i expansion: där gör synkronkompensatorer, nätbildande
    batterier och spill dem alltid uppfyllbara, och skuggpriset blir tolkbart."""
    st, d = s["stability"], s["dispatch"]
    if s["run"]["mode"] == "dispatch":
        pen, spen = d["stability_slack_penalty"], d["stability_scr_slack_penalty"]
    else:
        pen = spen = None
    sdata = _stability_sdata(cfg, s)
    if st["ek_system_gws"] or st["ek_zone_floor_gws"]:
        print(f"Stabilitetskrav (rotationsenergi), sync_weight {sdata['sync_weight']}, "
              + (f"mjukt, straff {pen:g} €/(MWs·h)" if pen else "HÅRT"))
    if st["scr_min"]:
        print(f"Stabilitetskrav (nätstyrka), SCR ≥ {st['scr_min']:g} mot omriktarinmatning, "
              f"undantag {sdata.get('scr_exempt')}, "
              + (f"gemensamt {sdata['scr_joint']}, " if sdata["scr_joint"] else "")
              + (f"mjukt, straff {spen:g} €/(MVA·h)" if spen else "HÅRT"))
    for line in stability_feasibility_report(n, sdata, st["ek_system_gws"],
                                             st["ek_zone_floor_gws"], st["scr_min"]):
        print(f"  {line}")
    return [stability_constraints(sdata, st["ek_system_gws"], st["ek_zone_floor_gws"], pen,
                                  st["scr_min"], spen)]


def _print_stability(summ, slack, wts) -> None:
    sy = summ.loc["SYSTEM"]
    print(f"SCR per zon (inkopplat, p05): "
          + ", ".join(f"{z} {v:.2f}" for z, v in summ["SCR_on_p05"].drop("SYSTEM").items()))
    print(f"Rotationsenergi SYSTEM (inkopplat): min {sy['Ek_on_min']:.1f}  "
          f"p05 {sy['Ek_on_p05']:.1f}  median {sy['Ek_on_median']:.1f} GWs "
          f"(utan krav hade driften gett lo/hi-medianer {sy['Ek_lo_median']:.1f}/"
          f"{sy['Ek_hi_median']:.1f})")
    if slack is not None:
        for col in slack:
            used = slack[col] > 1e-3
            unit = "GVA" if col.startswith("SCR_") else "GWs"
            print(f"  slack {col}: {float(wts.reindex(slack.index)[used].sum()):.0f} h, "
                  f"max {slack[col].max()/1e3:.1f} {unit}, "
                  f"{float((slack[col] * wts.reindex(slack.index)).sum())/1e3:.0f} {unit}·h")
    print("  → stability_*.csv")


def _dry_run_report(n) -> None:
    nuc = n.generators[n.generators.carrier == "nuclear"]
    print("\n=== DRY-RUN: kärnkraftsgeneratorer ===")
    for g, row in nuc.iterrows():
        kind = "EXPANSION" if g.endswith("nuclear exp") else "befintlig"
        pmax = n.generators_t.p_max_pu
        pmin = n.generators_t.p_min_pu
        print(f"  {g:18s} [{kind:9s}] p_nom={row.p_nom:7.0f}  "
              f"p_nom_max={row.p_nom_max:8.0f}  ext={bool(row.p_nom_extendable)!s:5s}  "
              f"CF(p_max)={float(pmax[g].mean()) if g in pmax else row.p_max_pu:.3f}  "
              f"must-run={float(pmin[g].mean()) if g in pmin else row.p_min_pu:.3f}")
    print("=== DRY-RUN klar (ingen solve) ===")


def run(s: dict, label: str, desc: str | None = None, dry_run: bool = False) -> None:
    mode, caps = s["run"]["mode"], s["run"]["capacities"]
    res, year  = int(s["period"]["resolution_hours"]), s["period"]["year"]
    print(f"Körning {label}: {mode}, värld {s['run']['world']}, kapaciteter {caps}, "
          f"{res}h, år {year or '2023-2025'}")

    cfg    = inp.load_config()
    extras = world.prepare_config(cfg, s)

    data = inp.load_inputs()
    vre  = s["vre"]
    data["vre_profiles"] = inp.boost_capfac(data["vre_profiles"],
                                            float(vre["onwind_capfac_increase"]), "wind_onshore")
    data["vre_profiles"] = inp.boost_capfac(data["vre_profiles"],
                                            float(vre["offwind_capfac_increase"]), "wind_offshore")
    world.scale_continent_prices(cfg, s, data["market_prices"])
    snapshots = inp.make_snapshots(cfg, res, year)
    data      = inp.resample_inputs(data, snapshots, res)

    extra_nuclear, synthetic_nuclear = world.nuclear_build_args(cfg, s)
    mc_override = (modes.hydro_mc_override(snapshots, cfg, s["expansion"]["hydro_mc_curve"])
                   if mode == "expansion" and s["expansion"]["hydro_mc_curve"] else None)
    rh = s["hydro"]["ror_hifreq"] or {"sigma": 0.0, "tau_days": 3.5, "seed": 0}

    print(f"Bygger nätverk ({len(snapshots)} tidssteg) ...")
    n = build_network(cfg, snapshots, **data,
                      voll=s["voll"],
                      batteries=extras["batteries"],
                      extra_nuclear=extra_nuclear,
                      synthetic_nuclear=synthetic_nuclear,
                      hydrogen_overrides=extras["hydrogen_overrides"] or None,
                      ev_overrides=extras["ev_overrides"] or None,
                      hydro_mc_override=mc_override,
                      ror_hifreq=float(rh["sigma"]),
                      ror_hifreq_seed=int(rh["seed"]),
                      ror_hifreq_tau_days=float(rh["tau_days"]),
                      battery_invest=extras["battery_invest"],
                      syncon=extras["syncon"])

    n_years = len(snapshots) * res / 8760.0
    world.apply_post_build(n, cfg, s, n_years)
    callbacks = _hydro_constraints(n, cfg, s)
    world.apply_potentials(n, s, extras)

    if mode == "dispatch":
        if caps not in ("config",):              # frys till källkörningens p_nom_opt
            modes.freeze_capacities_from(n, caps)
        modes.apply_vre_curtailment_cost(n, float(s["dispatch"]["vre_curtailment_cost"]))
    if s["stability"]["enabled"]:                   # efter frysningen
        callbacks += _stability_constraints(n, cfg, s)

    if dry_run:
        _dry_run_report(n)
        return

    write_run_meta(label, s, desc, len(snapshots))
    log_path = RESULTS_DIR / label / "highs.log"
    n.sanitize()
    if mode == "dispatch":
        ok, results = solve.solve_rolling_horizon(n, cfg, s["dispatch"], res,
                                                  log_path=log_path, extra_callbacks=callbacks)
    else:
        ok = solve.solve(n, cfg, log_path=log_path, extra_callbacks=callbacks)
    if not ok:
        raise SystemExit("Lösning misslyckades — kontrollera nätverket")

    if mode == "dispatch":
        # CSV:erna byggs av de fönstervisa lösningarna (sanningskälla); network.nc
        # exporteras också — PyPSA ackumulerar fönstren i n.*_t.
        n.export_to_netcdf(RESULTS_DIR / label / "network.nc")
        solve.save_results_dict(results, label)
        if s["stability"]["enabled"]:
            rep = stability_report(n, sdata=_stability_sdata(cfg, s),
                                   online=results.get("stability_online"))
            write_stability_reports(rep, RESULTS_DIR / label)
            _print_stability(rep["summary"], results.get("stability_slack"),
                             n.snapshot_weightings.generators)
    else:
        solve.save_results(n, label)
        if s["stability"]["enabled"]:
            res = solve.stability_results(n)
            rep = stability_report(n, sdata=_stability_sdata(cfg, s),
                                   online=res.get("stability_online"))
            write_stability_reports(rep, RESULTS_DIR / label)
            _print_stability(rep["summary"], res.get("stability_slack"),
                             n.snapshot_weightings.generators)
    print("Klart!")
