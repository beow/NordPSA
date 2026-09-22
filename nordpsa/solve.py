"""Lösning: en cyklisk LP (expansion) eller rullande horisont (dispatch), samt
resultatextraktion och sparning."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nordpsa.constraints import hydro_soc_initial_constraint, hydro_terminal_value
from nordpsa.settings import RESULTS_DIR, ROOT


def solve(n, cfg: dict, log_path: Path | None = None,
          extra_callbacks: list | None = None) -> bool:
    scfg    = cfg["solver"]
    solver  = scfg["name"]
    options = {k: v for k, v in scfg.items() if k != "name"}

    if log_path is not None:
        options["log_file"] = str(log_path)

    callbacks = [hydro_soc_initial_constraint(cfg)]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    def extra_func(n, snapshots):
        for cb in callbacks:
            cb(n, snapshots)

    print(f"Löser med {solver} ({len(n.snapshots)} tidssteg, "
          f"{len(n.generators) + len(n.storage_units)} generatorer) ...")
    if log_path:
        print(f"  HiGHS-logg: {log_path}")

    status, condition = n.optimize(
        solver_name=solver,
        solver_options=options,
        extra_functionality=extra_func,
        assign_all_duals=True,   # behövs för att få vattenvärdet (mu_energy_balance)
    )

    print(f"  Status: {status} / {condition}")
    return status == "ok"


def extract_results(n) -> dict:
    """Alla resultat-tidsserier ur ett löst nätverk → {namn: DataFrame}.

    EN sanningskälla för BÅDA körvägarna (standard + rullande horisont): lägg till
    en rad här så dyker resultatet upp i båda automatiskt. Nyckel = filnamn (utan
    .csv). None/tomma DataFrames hoppas tyst över vid sparning.

    Obs: hydro_soc/dispatch_hydro innehåller ALLA storage units (även batteri).
    water_value (dual på lagringsbalansen) kräver assign_all_duals=True i optimize.
    """
    out = {
        "dispatch_generators": n.generators_t.p,
        "dispatch_hydro":      n.storage_units_t.p,
        "hydro_soc":           n.storage_units_t.state_of_charge,          # inkl. batteri-SOC
        "hydro_spill":         n.storage_units_t.spill,
        "flows":               n.links_t.p0,
        "prices":              n.buses_t.marginal_price,
        "water_value":         n.storage_units_t.get("mu_energy_balance"),  # dual = vattenvärde
    }
    if len(n.stores) > 0:
        out["h2_store_soc"] = n.stores_t.e
    return out


def save_results_dict(results: dict, label: str) -> None:
    """Sparar {namn: DataFrame} som platta CSV:er (namn.csv). Hoppar över None/tomma.
    Delas av standard- och rullande-vägen."""
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if df is not None and getattr(df, "shape", (0, 0))[1] > 0:
            df.to_csv(out / f"{name}.csv")
    print(f"  → resultat sparade i {out}/")


def save_results(n, label: str) -> None:
    """Standard-vägen: nätverk (.nc) + alla CSV:er via extract_results-registryt."""
    out = RESULTS_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    n.export_to_netcdf(out / "network.nc")
    save_results_dict(extract_results(n), label)


def rolling_windows(snapshots: pd.DatetimeIndex, window_steps: int,
                    lookahead_steps: int = 0):
    """Delar snapshots i sekventiella fönster och ger (BEHÅLL, LÖS) per fönster.

    lookahead_steps=0 → klassiskt icke-överlappande: LÖS == BEHÅLL, varje fönster
    ser noll framåt och hela säsongssignalen måste bäras av terminalvärdet λ.

    lookahead_steps>0 → ÄKTA RECEDING HORIZON (väg E i docs/vattenvarde_plan.md):
    fönstret LÖSES med extra look-ahead men bara den första delen BEHÅLLS, och
    SOC bärs över från slutet av BEHÅLL-delen. Poängen är att look-ahead gör det
    mesta av jobbet i stället för terminalkurvan, vilket i sin tur krymper
    cirkularitetsproblemet (λ kalibreras mot observerade priser).

    Sista fönstret får ingen look-ahead att hämta — det finns inget efter
    periodens slut — så där sammanfaller LÖS och BEHÅLL igen.
    """
    n = len(snapshots)
    for s in range(0, n, window_steps):
        keep = snapshots[s: s + window_steps]
        solve = snapshots[s: min(s + window_steps + lookahead_steps, n)]
        yield keep, solve


def solve_rolling_horizon(n, cfg: dict, d: dict, res: int,
                          log_path: Path | None = None,
                          extra_callbacks: list | None = None) -> tuple[bool, dict | None]:
    """Rullande horisont: lös perioden fönster för fönster med icke-cyklisk SOC,
    carry-over av slut-SOC, och ett terminalvärde −λ×SOC[T] per fönster.

    Syftet är att bryta den perfekta framsynen över hela perioden, som gör det
    endogena vattenvärdet nästan konstant (1–6 unika värden per zon över tre år).

    Fönstren är som default icke-överlappande, så varje fönster ser NOLL framåt och
    hela säsongssignalen måste bäras av λ. `lookahead_weeks N` ger äkta receding
    horizon: fönstret löses med N veckors extra look-ahead men bara första delen
    behålls, och SOC bärs över från behåll-delens slut.

    `d` är körinställningarnas dispatch-sektion.
    """
    scfg    = cfg["solver"]
    options = {k: v for k, v in scfg.items() if k != "name"}
    if log_path is not None:
        options["log_file"] = str(log_path)

    units = [u for u in n.storage_units.index
             if n.storage_units.at[u, "carrier"] == "hydro"]
    if not units:
        raise SystemExit("rullande horisont: inga hydrolager i nätverket")
    n.storage_units.loc[units, "cyclic_state_of_charge"] = False

    cap = {u: float(n.storage_units.at[u, "p_nom"]) * float(n.storage_units.at[u, "max_hours"])
           for u in units}
    # Start-SOC: samma ankare (hydro_soc_initial) som expansionens cykliska villkor, dvs
    # den UPPMÄTTA EC-nivån 2023-01-02. I rullande horisont är det ett äkta begynnelse-
    # villkor som propagerar genom hela perioden.
    soc_carry, start_src = {}, []
    for u in units:
        zone = u.split()[0]
        frac = cfg["zones"].get(zone, {}).get("hydro_soc_initial", 0.5)
        soc_carry[u] = frac * cap[u]
        start_src.append(f"{zone} {frac:.0%}")
    print("  → start-SOC: " + ", ".join(start_src))

    # Terminalkurvan λ_k(vecka, zon) räknas om PER FÖNSTER, eftersom varje fönster
    # slutar i en annan vecka.
    from nordpsa.wv import terminal_curve as tc
    cparams, canchor = tc.load_params(str(ROOT / d["terminal_curve"]))
    segments = int(d["terminal_segments"])
    # λ_bas kan överstyras per värld/körning. Motivet: drift är SCENARIOBEROENDE
    # (−3,5 till +10,7 TWh över batch 36 vid samma λ_bas = 80), så nivån hör till
    # världen och inte till kurvfilen (dagens flotta har egna ankare, se worlds/today).
    anchor = d["terminal_anchor"]
    if isinstance(anchor, dict):
        canchor = dict(canchor)
        for z, v in anchor.items():
            if z not in canchor:
                raise SystemExit(f"dispatch.terminal_anchor: okänd zon {z!r} "
                                 f"(kurvan har {sorted(canchor)})")
            canchor[z] = float(v)
    elif anchor is not None:                      # ett tal = ALLA zoner
        canchor = {z: float(anchor) for z in canchor}
    print(f"  → TERMINALKURVA λ_k(vecka, zon), {segments} segment. λ_bas: "
          + ", ".join(f"{z} {v:.1f}" for z, v in sorted(canchor.items())))
    for z in sorted(cparams):
        q = cparams[z]
        print(f"       {z:6s} a_amp={q.a_amp:.2f} a_peak=v{q.a_peak:.0f} "
              f"b_mean={q.b_mean:.2f} b_amp={q.b_amp:.2f} b_peak=v{q.b_peak:.0f}")

    steps_per_week  = max(1, (7 * 24) // res)
    window_steps    = int(d["rolling_weeks"]) * steps_per_week
    lookahead_steps = int(d["lookahead_weeks"]) * steps_per_week
    windows         = list(rolling_windows(n.snapshots, window_steps, lookahead_steps))
    print(f"  → rullande horisont: {d['rolling_weeks']} veckor/fönster "
          f"({window_steps} tidssteg), {len(windows)} fönster, {len(units)} hydrolager")
    if lookahead_steps:
        print(f"  → RECEDING HORIZON: +{d['lookahead_weeks']} veckors look-ahead "
              f"({lookahead_steps} steg) löses men KASTAS; SOC bärs över från behåll-delen. "
              f"Terminalvärdet hamnar {d['lookahead_weeks']} veckor bort och styr "
              f"därmed mindre.")
    if log_path is not None:
        print(f"  HiGHS-logg: {log_path} (skrivs över per fönster — sista kvarstår)")

    parts, keys = [], []
    for i, (keep, sns) in enumerate(windows, 1):
        for u in units:
            n.storage_units.at[u, "state_of_charge_initial"] = soc_carry[u]

        wk   = tc.week_of(sns[-1])
        lam  = tc.lambdas_for_week(wk, units, canchor, cparams)
        prof = tc.profiles_for_week(wk, units, cparams, segments)
        callbacks = [hydro_terminal_value(lam, cap, prof)] + list(extra_callbacks or [])

        def extra_func(nn, snapshots, _cbs=callbacks):
            for cb in _cbs:
                cb(nn, snapshots)

        status, condition = n.optimize(
            snapshots=sns,
            solver_name=scfg["name"],
            solver_options=options,
            extra_functionality=extra_func,
            assign_all_duals=True,
        )
        start_txt = " ".join(f"{u.split()[0]} {soc_carry[u]/cap[u]:.0%}" for u in units)
        if status == "ok":
            # SOC bärs över från slutet av BEHÅLL-delen, inte från look-ahead-svansen:
            # svansen är bara en framtidsbild och kastas.
            soc_carry = {u: float(n.storage_units_t.state_of_charge.at[keep[-1], u])
                         for u in units}
            slut_txt = " ".join(f"→{soc_carry[u]/cap[u]:.0%}" for u in units)
        else:
            slut_txt = ""
        tail = f"+{len(sns)-len(keep)}" if len(sns) > len(keep) else ""
        print(f"  fönster {i:3d}/{len(windows)} {keep[0]:%Y-%m-%d}–{keep[-1]:%Y-%m-%d} "
              f"({len(keep)}{tail} steg): {status}/{condition}  "f"v{wk} λ={list(lam.values())[0]:.1f}  "
              f"{start_txt} {slut_txt}")
        if status != "ok":
            print(f"  ✖ fönster {i} misslyckades ({status}/{condition}) — avbryter.")
            return False, None

        part = {k: v.loc[keep] for k, v in extract_results(n).items()
                if v is not None and getattr(v, "shape", (0, 0))[1] > 0}
        for k in part:
            if k not in keys:
                keys.append(k)
        parts.append(part)

    fill = {u: soc_carry[u] / cap[u] for u in units}
    print("  slut-SOC: " + " ".join(f"{u.split()[0]} {f:.0%}" for u, f in fill.items()))
    if all(f > 0.95 for f in fill.values()):
        print("  ⚠️ ALLA reservoarer >95 % vid periodens slut — hamstring. "
              "Terminal-λ är för högt mot släppmarginalen (jfr run91–93).")

    results = {k: pd.concat([p[k] for p in parts if k in p]).sort_index() for k in keys}
    return True, results
