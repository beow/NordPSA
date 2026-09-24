"""Krav på rotationsenergi och nätstyrka: linjäriserad inkoppling per timme, båda lägena.

Per synkron enhet i som inte är must-run (p_min_pu ≠ p_max_pu):

    0 ≤ u_i,t ≤ p_max_pu_i,t · P_i                <c>-p_online
    p_i,t − u_i,t ≤ 0                           custom-stability-p_le_online-<c>
    p_i,t − m_min_i · u_i,t ≥ 0                 custom-stability-p_ge_minstable-<c>

P_i är p_nom (fast) eller kapacitetsvariabeln (extendable i expansion; då ett eget villkor
custom-stability-online_le_cap-<c>).

    E_k,z(t) = Σ_i∈z e_i·u_i,t + Σ_j∈z e_j·avail_j·P_j + K_z(t)     e = eff·H/cosφ  [MWs/MW]
    Σ_z w_z·E_k,z(t) + σ_t   ≥ 1e3·E_sys        custom-stability-ek_system
    E_k,z(t)         + σ_z,t ≥ 1e3·E_floor,z    custom-stability-ek_zone-<zon>

    S_k,z(t) − scr_min·P_IBR,z(t) + σ ≥ 0        custom-stability-scr-<zon>
    S_k,z(t) = Σ_i∈z s_i·u_i,t + Σ_j∈z s_j·avail_j·P_j + K^S_z(t)  s = eff/((X''_d+X_T)·cosφ)
    P_IBR,z(t) = Σ ibr_w·p                     vind, sol, batteriurladdning (VARIABLER)

j = synkronkompensatorer (avail 1) och nätbildande batterier (e = H/cosφ, s = sk_pu), som
bidrar med sin KAPACITET oavsett drift. K_z(t) = must-run-enheter (thermal, befintlig
kärnkraft, strömkraft) och fasta j-enheter: konstanter ur data.

SCR-kravet mäts mot omriktarnas faktiska inmatning, så LP:t kan uppfylla det genom att
koppla in mer synkront, bygga synkronkompensatorer eller nätbildande batterier, eller
SPILLA vind/sol. Zoner i zones.yaml:stability.scr_exempt (DK) får inget krav.
⚠️ Zonvärdet är ett aggregat: alla maskiner antas sitta i samma punkt (optimistiskt) och
grannzonernas bidrag saknas (pessimistiskt).

Vad gör tröghet dyrt när inkopplingen är gratis? p ≥ m_min·u ⇒ mer inkopplat kräver mer
PRODUKTION, som tränger ut billigare kraft eller förbrukar vatten med positivt
vattenvärde. m_min avgör därför hela tröghetens pris.

Slack σ ≥ 0 med straff (€/(MWs·h) resp. €/(MVA·h)) i dispatch; utan straff är kravet hårt
(expansion: synkronkompensatorer och spill gör det alltid uppfyllbart). Villkoren är rent
inom-snapshot, så den rullande horisontens fönster bryter ingenting. Enhetstabellen byggs
vid första anropet (efter frysningen) och återanvänds i alla fönster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from nordpsa.analysis.stability import online_capacity, unit_table

_PVAR = {"Generator": "Generator-p", "StorageUnit": "StorageUnit-p_dispatch", "Link": "Link-p"}
SYSTEM = "SYSTEM"
_CAPMODES = ("syncon", "gfm")              # bidrar med kapacitet, inte inkoppling


def _committable(u: pd.DataFrame) -> pd.DataFrame:
    """Enheter som får inkopplingsvariabel. Kontrollerar degenerans."""
    cm = u[(u["mode"] == "commit") & ~u.fixed & ((u.cap > 0) | u.extendable)]
    free = cm.index[cm.m_min <= 0]
    if len(free):
        raise ValueError(f"stability: m_min = 0 ger gratis tröghet för {list(free)} "
                         "(bara must-run-enheter får ha m_min 0)")
    return cm


def _cap_factor(u: pd.DataFrame) -> pd.Series:
    """Bidrag per MW kapacitet för syncon/GFM: avail för GFM, 1 för syncon."""
    return pd.Series(np.where(u["mode"] == "gfm", u.avail, 1.0), index=u.index)


def _const_by_zone(n, u: pd.DataFrame, sn: pd.DatetimeIndex, zones: list,
                   coef: str = "e_coef") -> pd.DataFrame:
    """K_z(t) för FAST kapacitet: must-run samt syncon/GFM som inte är extendable.
    coef = "e_coef" ger MWs, "s_coef" ger MVA."""
    fixed = u[(u["mode"] == "commit") & u.fixed & (u.cap > 0) & ~u.extendable]
    k = pd.DataFrame(0.0, index=sn, columns=zones)
    if len(fixed):
        on = online_capacity(n, fixed, "max").loc[sn]
        k = k.add((on * fixed[coef]).T.groupby(fixed.zone).sum().T, fill_value=0.0)
    cu = u[u["mode"].isin(_CAPMODES) & ~u.extendable]
    const = (cu[coef] * _cap_factor(cu) * cu.cap).groupby(cu.zone).sum()
    return k.add(const.reindex(zones, fill_value=0.0), axis=1)[zones]


def _scr_zones(sdata: dict, zones: list) -> list:
    exempt = set(sdata.get("scr_exempt") or [])
    return [z for z in zones if z not in exempt]


def _da(values, index, dim="name"):
    return xr.DataArray(np.asarray(values, dtype=float), coords={dim: index}, dims=dim)


def stability_constraints(sdata: dict, ek_system_gws: float | None, ek_zone_floor_gws: dict,
                          slack_penalty: float | None, scr_min: float | None = None,
                          scr_slack_penalty: float | None = None):
    """extra_functionality-callback, dispatch eller expansion. sdata = stability_data(...)."""
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

        def cap_var(c, names):
            return m.variables[f"{c}-p_nom"].sel(name=names)

        ek = {z: [] for z in zones}                    # linjära uttryck [MWs]
        sk = {z: [] for z in zones}                    # [MVA]

        # Inkoppling för synkrona enheter som inte är must-run
        for c, g in cm.groupby("component"):
            names = pd.Index(g.index, name="name")
            pmax = n.get_switchable_as_dense(c, "p_max_pu", sn)[names]
            ub = pmax * g.cap
            ub.loc[:, g.index[g.extendable]] = np.inf      # taket sätts av online_le_cap
            ub_da = xr.DataArray(ub.to_numpy(), coords={"snapshot": sn, "name": names},
                                 dims=("snapshot", "name"))
            on = m.add_variables(lower=0.0, upper=ub_da, name=f"{c}-p_online")
            ext = names[g.extendable.to_numpy()]
            if len(ext):
                pm = xr.DataArray(pmax[ext].to_numpy(), coords={"snapshot": sn, "name": ext},
                                  dims=("snapshot", "name"))
                m.add_constraints(on.sel(name=ext) - pm * cap_var(c, ext) <= 0,
                                  name=f"custom-stability-online_le_cap-{c}")
            p = m.variables[_PVAR[c]].sel(name=names)
            m.add_constraints(p - on <= 0, name=f"custom-stability-p_le_online-{c}")
            m.add_constraints(p - _da(g.m_min, names) * on >= 0,
                              name=f"custom-stability-p_ge_minstable-{c}")
            for z, gz in g.groupby("zone"):
                ek[z].append((_da(gz.e_coef, gz.index) * on.sel(name=gz.index)).sum("name"))
                sk[z].append((_da(gz.s_coef, gz.index) * on.sel(name=gz.index)).sum("name"))

        # Investerbar kapacitet som bidrar oavsett drift: syncon, GFM, ny must-run
        capu = u[u.extendable & (u["mode"].isin(_CAPMODES)
                                 | ((u["mode"] == "commit") & u.fixed))]
        for (c, z), g in capu.groupby(["component", "zone"]):
            names = pd.Index(g.index, name="name")
            f = _cap_factor(g)
            mr = g["mode"] == "commit"                 # must-run: följer p_max_pu(t)
            if mr.any():
                mn = names[mr.to_numpy()]
                pm = xr.DataArray(n.get_switchable_as_dense(c, "p_max_pu", sn)[mn].to_numpy(),
                                  coords={"snapshot": sn, "name": mn}, dims=("snapshot", "name"))
                ek[z].append((_da(g.e_coef[mn], mn) * pm * cap_var(c, mn)).sum("name"))
                sk[z].append((_da(g.s_coef[mn], mn) * pm * cap_var(c, mn)).sum("name"))
            cn = names[~mr.to_numpy()]
            if len(cn):
                ek[z].append((_da(g.e_coef[cn] * f[cn], cn) * cap_var(c, cn)).sum("name"))
                sk[z].append((_da(g.s_coef[cn] * f[cn], cn) * cap_var(c, cn)).sum("name"))

        k = _const_by_zone(n, u, sn, zones)

        def _require(terms: list, rhs: pd.Series, name: str, penalty) -> None:
            rhs_da = xr.DataArray(rhs.to_numpy(dtype=float), coords={"snapshot": sn},
                                  dims="snapshot")
            if penalty:
                s = m.add_variables(lower=0.0, coords=[sn], name=f"{name}-slack")
                terms = terms + [s]
                m.objective = m.objective + (float(penalty) * w_obj * s).sum()
            if not terms:                          # inget att styra med: bara hårt och omöjligt
                if (rhs > 1e-6).any():
                    raise ValueError(f"{name}: kravet går inte att uppfylla (inga enheter)")
                return
            m.add_constraints(sum(terms[1:], terms[0]) >= rhs_da, name=name)

        if ek_system_gws:
            w = pd.Series({z: float(sw.get(z, 1.0)) for z in zones})
            terms = [float(w[z]) * t for z in zones if w[z] > 0 for t in ek[z]]
            rhs = 1e3 * float(ek_system_gws) - (k * w).sum(axis=1)
            _require(terms, rhs, "custom-stability-ek_system", slack_penalty)
        for z, f in floors.items():
            _require(list(ek[z]), 1e3 * f - k[z], f"custom-stability-ek_zone-{z}",
                     slack_penalty)

        if scr_min:
            ks = _const_by_zone(n, u, sn, zones, "s_coef")
            ibr = u[u["mode"].isin(["ibr", "gfm"]) & (u.ibr_w > 0)
                    & ((u.cap > 0) | u.extendable)]
            for z in _scr_zones(sdata, zones):
                terms = list(sk[z])
                for c, g in ibr[ibr.zone == z].groupby("component"):
                    p = m.variables[_PVAR[c]].sel(name=g.index)
                    terms.append((-float(scr_min) * _da(g.ibr_w, g.index) * p).sum("name"))
                if len(terms) == len(sk[z]):           # ingen omriktare i zonen
                    continue
                _require(terms, -ks[z], f"custom-stability-scr-{z}", scr_slack_penalty)

    return _extra_functionality


def stability_feasibility_report(n, sdata: dict, ek_system_gws: float | None,
                                 ek_zone_floor_gws: dict, scr_min: float | None = None) -> list[str]:
    """Förhandskontroll: räcker E_k om ALLT tillgängligt kopplas in?

    Timmar där E_k,max ligger under kravet kan bara klaras med slack (eller blir
    infeasible med hårt krav). I expansion räknas bara befintlig kapacitet (investeringar
    är ännu 0), så raden visar vad som måste byggas eller spillas bort.
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
    if scr_min:
        # Utan spill: räcker S_k om allt synkront kopplas in mot ALL tillgänglig omriktareffekt?
        sk = ((on * cm.s_coef[on.columns]).T.groupby(cm.zone[on.columns]).sum().T
              .reindex(columns=zones, fill_value=0.0))
        sk = sk.add(_const_by_zone(n, u[u["mode"] != "commit"], n.snapshots, zones, "s_coef"))
        ibr = u[u["mode"].isin(["ibr", "gfm"]) & (u.ibr_w > 0)]
        avail = pd.concat([n.get_switchable_as_dense(c, "p_max_pu")[g.index] * g.cap * g.ibr_w
                           for c, g in ibr.groupby("component")], axis=1)
        pav = avail.T.groupby(ibr.zone[avail.columns]).sum().T.reindex(columns=zones,
                                                                       fill_value=0.0)
        for z in _scr_zones(sdata, zones):
            short = sk[z] < float(scr_min) * pav[z]
            lines.append(f"SCR {z:5s} krav {float(scr_min):.2f}   timmar där allt synkront "
                         f"inkopplat inte räcker mot full omriktarinmatning: "
                         f"{float(wts[short].sum()):.0f} (kräver spill eller slack)")
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
    names = [c for c in m.constraints
             if c.startswith(("custom-stability-ek_", "custom-stability-scr-"))]
    if names:
        # kolumner: SYSTEM / <zon> (E_k, MWs) och SCR_<zon> (MVA)
        label = {c: (SYSTEM if c == "custom-stability-ek_system"
                     else "SCR_" + c.removeprefix("custom-stability-scr-")
                     if c.startswith("custom-stability-scr-")
                     else c.removeprefix("custom-stability-ek_zone-")) for c in names}
        w = n.snapshot_weightings.objective                  # dual per snapshot → per timme
        dual = pd.DataFrame({label[c]: m.constraints[c].dual.to_pandas() for c in names})
        out["stability_dual"] = dual.div(w.reindex(dual.index), axis=0)   # €/(MWs·h), €/(MVA·h)
        slack = {label[c]: m.variables[f"{c}-slack"].solution.to_pandas()
                 for c in names if f"{c}-slack" in m.variables}
        if slack:
            out["stability_slack"] = pd.DataFrame(slack)
    return out
