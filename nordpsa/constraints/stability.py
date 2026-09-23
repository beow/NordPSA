"""Krav på rotationsenergi i dispatch: linjäriserad inkoppling + mjukt system-/zonkrav.

Per synkron enhet i som inte är must-run (p_min_pu ≠ p_max_pu) och har p_nom > 0:

    0 ≤ u_i,t ≤ p_max_pu_i,t · p_nom_i          <c>-p_online
    p_i,t − u_i,t ≤ 0                           custom-stability-p_le_online-<c>
    p_i,t − m_min_i · u_i,t ≥ 0                 custom-stability-p_ge_minstable-<c>

    E_k,z(t) = Σ_i∈z e_i·u_i,t + K_z(t)         e_i = eff·H/cosφ  [MWs per MW]
    Σ_z w_z·E_k,z(t) + σ_t   ≥ 1e3·E_sys        custom-stability-ek_system
    E_k,z(t)         + σ_z,t ≥ 1e3·E_floor,z    custom-stability-ek_zone-<zon>

K_z(t) är must-run-enheternas bidrag (thermal, befintlig kärnkraft, strömkraft), som är
bestämt av data: Σ e_i·p_max_pu_i,t·p_nom_i. Slacken σ ≥ 0 kostar
`dispatch.stability_slack_penalty` €/(MWs·h); utan straff är kravet hårt.

Vad gör tröghet dyrt när inkopplingen är gratis? p ≥ m_min·u ⇒ mer inkopplat kräver mer
PRODUKTION, som tränger ut billigare kraft eller förbrukar vatten med positivt
vattenvärde. m_min avgör därför hela tröghetens pris.

⚠️ Bara dispatch: kapaciteterna ska vara frysta. Villkoren är rent inom-snapshot, så
fönsterindelningen i den rullande horisonten bryter ingenting. Enhetstabellen byggs vid
första anropet (efter frysningen) och återanvänds i alla fönster.
"""
from __future__ import annotations

import pandas as pd
import xarray as xr

from nordpsa.analysis.stability import online_capacity, unit_table

_PVAR = {"Generator": "Generator-p", "StorageUnit": "StorageUnit-p_dispatch", "Link": "Link-p"}
SYSTEM = "SYSTEM"


def _committable(u: pd.DataFrame) -> pd.DataFrame:
    """Enheter som får inkopplingsvariabel. Kontrollerar degenerans och frysning."""
    cm = u[(u["mode"] == "commit") & ~u.fixed & (u.cap > 0)]
    free = cm.index[cm.m_min <= 0]
    if len(free):
        raise ValueError(f"stability: m_min = 0 ger gratis tröghet för {list(free)} "
                         "(bara must-run-enheter får ha m_min 0)")
    ext = u.index[u.extendable & (u.cap > 0)]
    if len(ext):
        raise ValueError(f"stability: extendable enheter i dispatch: {list(ext)[:5]} — "
                         "kapaciteterna måste vara frysta")
    return cm


def _const_by_zone(n, u: pd.DataFrame, sn: pd.DatetimeIndex, zones: list) -> pd.DataFrame:
    """K_z(t) [MWs]: must-run, synkronkompensatorer och nätbildande omriktare."""
    fixed = u[(u["mode"] == "commit") & u.fixed & (u.cap > 0)]
    k = pd.DataFrame(0.0, index=sn, columns=zones)
    if len(fixed):
        on = online_capacity(n, fixed, "max").loc[sn]
        k = k.add((on * fixed.e_coef).T.groupby(fixed.zone).sum().T, fill_value=0.0)
    sc, gfm = u[u["mode"] == "syncon"], u[u["mode"] == "gfm"]
    const = ((sc.e_coef * sc.cap).groupby(sc.zone).sum()
             .add((gfm.e_coef * gfm.avail * gfm.cap).groupby(gfm.zone).sum(), fill_value=0.0))
    return k.add(const.reindex(zones, fill_value=0.0), axis=1)[zones]


def stability_constraints(sdata: dict, ek_system_gws: float | None, ek_zone_floor_gws: dict,
                          slack_penalty: float | None):
    """extra_functionality-callback för dispatch. sdata = stability_data(...)."""
    sw = dict(sdata.get("sync_weight") or {})
    floors = {z: float(v) for z, v in (ek_zone_floor_gws or {}).items() if float(v) > 0}
    cache: dict = {}

    def _extra_functionality(n, snapshots) -> None:
        if "u" not in cache:
            u = unit_table(n, sdata)
            cache["u"], cache["cm"] = u, _committable(u)
        u, cm = cache["u"], cache["cm"]
        m = n.model
        sn = pd.DatetimeIndex(snapshots, name="snapshot")
        zones = list(n.buses.index[n.buses.carrier == "AC"])
        w_obj = xr.DataArray(n.snapshot_weightings.objective.reindex(sn).to_numpy(dtype=float),
                             coords={"snapshot": sn}, dims="snapshot")

        ek = {z: [] for z in zones}                    # linjära uttryck över snapshot [MWs]
        for c, g in cm.groupby("component"):
            names = pd.Index(g.index, name="name")
            ub = n.get_switchable_as_dense(c, "p_max_pu", sn)[names] * g.cap
            ub_da = xr.DataArray(ub.to_numpy(), coords={"snapshot": sn, "name": names},
                                 dims=("snapshot", "name"))
            on = m.add_variables(lower=0.0, upper=ub_da, name=f"{c}-p_online")
            p = m.variables[_PVAR[c]].sel(name=names)
            mmin = xr.DataArray(g.m_min.to_numpy(), coords={"name": names}, dims="name")
            m.add_constraints(p - on <= 0, name=f"custom-stability-p_le_online-{c}")
            m.add_constraints(p - mmin * on >= 0, name=f"custom-stability-p_ge_minstable-{c}")
            for z, gz in g.groupby("zone"):
                e = xr.DataArray(gz.e_coef.to_numpy(), coords={"name": gz.index}, dims="name")
                ek[z].append((e * on.sel(name=gz.index)).sum("name"))

        k = _const_by_zone(n, u, sn, zones)

        def _require(terms: list, rhs: pd.Series, name: str) -> None:
            rhs_da = xr.DataArray(rhs.to_numpy(dtype=float), coords={"snapshot": sn},
                                  dims="snapshot")
            if slack_penalty:
                s = m.add_variables(lower=0.0, coords=[sn], name=f"{name}-slack")
                terms = terms + [s]
                m.objective = m.objective + (float(slack_penalty) * w_obj * s).sum()
            if not terms:                          # inget att styra med: bara hårt och omöjligt
                if (rhs > 1e-6).any():
                    raise ValueError(f"{name}: kravet går inte att uppfylla (inga enheter)")
                return
            m.add_constraints(sum(terms[1:], terms[0]) >= rhs_da, name=name)

        if ek_system_gws:
            w = pd.Series({z: float(sw.get(z, 1.0)) for z in zones})
            terms = [float(w[z]) * t for z in zones if w[z] > 0 for t in ek[z]]
            rhs = 1e3 * float(ek_system_gws) - (k * w).sum(axis=1)
            _require(terms, rhs, "custom-stability-ek_system")
        for z, f in floors.items():
            _require(list(ek[z]), 1e3 * f - k[z], f"custom-stability-ek_zone-{z}")

    return _extra_functionality


def stability_feasibility_report(n, sdata: dict, ek_system_gws: float | None,
                                 ek_zone_floor_gws: dict) -> list[str]:
    """Förhandskontroll efter frysningen: räcker E_k om ALLT tillgängligt kopplas in?

    Timmar där E_k,max ligger under kravet kan bara klaras med slack (eller blir
    infeasible med hårt krav).
    """
    u = unit_table(n, sdata)
    zones = list(n.buses.index[n.buses.carrier == "AC"])
    on = online_capacity(n, u, "max")
    cm = u[u["mode"] == "commit"]
    ek = ((on * cm.e_coef[on.columns]).T.groupby(cm.zone[on.columns]).sum().T
          .reindex(columns=zones, fill_value=0.0))
    ek = ek.add(_const_by_zone(n, u[u["mode"] != "commit"], n.snapshots, zones)) / 1e3
    wts = n.snapshot_weightings.generators
    sw = sdata.get("sync_weight") or {}
    lines = []

    def _line(label: str, series: pd.Series, req: float) -> None:
        below = float(wts[series < req].sum())
        lines.append(f"{label:7s} krav {req:6.1f} GWs   E_k,max min {series.min():6.1f}  "
                     f"p05 {series.quantile(0.05):6.1f}  median {series.median():6.1f}   "
                     f"timmar under kravet: {below:.0f}" + ("  ⚠️ INFEASIBLE utan slack"
                                                           if below else ""))
    if ek_system_gws:
        sys_ek = (ek * pd.Series({z: float(sw.get(z, 1.0)) for z in zones})).sum(axis=1)
        _line(SYSTEM, sys_ek, float(ek_system_gws))
    for z, f in (ek_zone_floor_gws or {}).items():
        if float(f) > 0:
            _line(z, ek[z], float(f))
    return lines


def stability_results(n) -> dict:
    """Lösningens inkoppling, slack och duals ur n.model → {namn: DataFrame}.

    Tom dict när villkoren inte finns, så att extract_results är oförändrat då.
    """
    m = getattr(n, "model", None)
    if m is None:
        return {}
    out = {}
    online = [m.variables[f"{c}-p_online"].solution.to_pandas()
              for c in _PVAR if f"{c}-p_online" in m.variables]
    if online:
        out["stability_online"] = pd.concat(online, axis=1)
    names = [c for c in m.constraints if c.startswith("custom-stability-ek_")]
    if names:
        label = {c: (SYSTEM if c == "custom-stability-ek_system"
                     else c.removeprefix("custom-stability-ek_zone-")) for c in names}
        w = n.snapshot_weightings.objective                  # dual per snapshot → per timme
        dual = pd.DataFrame({label[c]: m.constraints[c].dual.to_pandas() for c in names})
        out["stability_dual"] = dual.div(w.reindex(dual.index), axis=0)   # €/(MWs·h)
        slack = {label[c]: m.variables[f"{c}-slack"].solution.to_pandas()
                 for c in names if f"{c}-slack" in m.variables}
        if slack:
            out["stability_slack"] = pd.DataFrame(slack)
    return out
