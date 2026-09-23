"""Rotationsenergi (E_k) och kortslutningseffekt (S_k) i en körning — mätning, ingen LP.

Per enhet, med p_nom i MW aktiv effekt (för KVV-länken på bränslesidan, därav eff):

    S_n = eff · p_nom / cosφ              [MVA]  märkskenbar effekt
    E_k = H · S_n,inkopplad               [MWs]  rotationsenergi
    S_k = S_n,inkopplad / (X''_d + X_T)   [MVA]  kortslutningseffekt på HV-sidan

Teknikdata och mappning (komponenttyp, carrier) → teknikklass ligger i
config/zones.yaml, block `stability:`.

Utan stabilitetsvillkor har körningen ingen inkopplingsvariabel, så inkopplad effekt u är
okänd. Den begränsas i stället av samma villkor som dispatchen har (m_min·u ≤ p ≤ u ≤ ā):

    lo   u = p                        minst inkopplat som är förenligt med driften
    hi   u = min(p / m_min, ā)        mest inkopplat som är förenligt med driften
    max  u = ā = p_max_pu · p_nom     allt tillgängligt inkopplat (fysikalisk övre gräns)
    on   u = p_online                 den lösta inkopplingen (bara med stability.enabled)

Enheter med p_min_pu = p_max_pu (thermal, befintlig kärnkraft, strömkraft) är `fixed`:
deras inkoppling är bestämd av data, u = ā i alla tre fallen.

⚠️ Tröghet är en SYSTEMstorhet: hela synkronområdet har en frekvens. Zonvärdena är ett
robusthetsmått (nätdelning), inte fysik. Systemvärdet viktar zonerna med sync_weight
(DK: bara DK2 hör till det nordiska synkronområdet). HVDC ger ingen tröghet, så
kontinentventilerna (`market`) bidrar inte.

    from nordpsa.analysis.stability import unit_table, stability_metrics, stability_summary
    u = unit_table(n)
    ts = stability_metrics(n, u, sync_weight={"DK": 0.35})

Enheter internt: MW, MWs, MVA. Rapportering: GW, GWs, GVA.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nordpsa.inputs import load_config

KINDS = ("max", "hi", "lo")
SYSTEM = "SYSTEM"

_BUS = {"Generator": "bus", "StorageUnit": "bus", "Link": "bus1"}
_DISPATCH = {"Generator": ("generators_t", "p"),
             "StorageUnit": ("storage_units_t", "p_dispatch"),
             "Link": ("links_t", "p0")}             # p0: samma sida som p_nom
_TECH_DEFAULTS = {"H": 0.0, "cos_phi": 1.0, "xd2": np.nan, "m_min": 0.0, "avail": 1.0,
                  "ibr_w": 0.0, "i_ibr": 0.0, "aux_loss_pu": 0.0}


def stability_data(tech: dict | None = None, sync_weight: dict | None = None,
                   cfg: dict | None = None) -> dict:
    """zones.yaml:s `stability:`-block, med ev. overrides per teknikklass och zonvikt."""
    sdata = dict((cfg or load_config())["stability"])
    sdata["tech"] = {k: dict(v) for k, v in sdata["tech"].items()}
    sdata["sync_weight"] = {**(sdata.get("sync_weight") or {}), **(sync_weight or {})}
    for k, v in (tech or {}).items():
        if k not in sdata["tech"]:
            raise ValueError(f"stability.tech: okänd teknikklass {k!r}")
        sdata["tech"][k].update(v)
    return sdata


def stability_tech(sdata: dict) -> pd.DataFrame:
    """Teknikklasser med koefficienter per MW aktiv märkeffekt (eff = 1).

    e_coef [MWs/MW] = H/cosφ
    s_coef [MVA/MW] = 1/((X''_d + X_T)·cosφ) för synkrona klasser, annars 0
    """
    t = pd.DataFrame.from_dict(sdata["tech"], orient="index")
    for col, val in _TECH_DEFAULTS.items():
        t[col] = t[col].fillna(val) if col in t else val
    t["e_coef"] = t.H / t.cos_phi
    sync = t["mode"].isin(["commit", "syncon"])
    t["s_coef"] = np.where(sync, 1.0 / ((t.xd2 + sdata["x_t"]) * t.cos_phi), 0.0)
    bad = sync & t.xd2.isna()
    if bad.any():
        raise ValueError(f"stability.tech: xd2 saknas för {list(t.index[bad])}")
    return t


def unit_table(n, sdata: dict | None = None) -> pd.DataFrame:
    """En rad per enhet som mappats till en teknikklass och sitter på en elbuss.

    cap är kapaciteten som utvärderas: p_nom_opt för extendable (0 i ett olöst nätverk),
    annars p_nom. e_coef/s_coef är per MW av cap, dvs. inklusive eff.
    Enheter med p_nom = 0 som inte är extendable tas inte med.
    """
    sdata = sdata or stability_data()
    tech = stability_tech(sdata)
    ac = n.buses.index[n.buses.carrier == "AC"]
    parts = []
    for c, bus_col in _BUS.items():
        df = n.components[c].static
        if df.empty:
            continue
        cls = (c + ":" + df.carrier).map(sdata["mapping"])
        for suffix, t in (sdata.get("name_overrides") or {}).items():
            cls[cls.notna() & df.index.str.endswith(suffix)] = t
        keep = cls.notna() & df[bus_col].isin(ac) & ((df.p_nom > 0) | df.p_nom_extendable)
        if not keep.any():
            continue
        df, cls = df[keep], cls[keep]
        unknown = set(cls) - set(tech.index)
        if unknown:
            raise ValueError(f"stability.mapping pekar på okänd teknikklass: {sorted(unknown)}")
        lo = n.get_switchable_as_dense(c, "p_min_pu")[df.index]
        hi = n.get_switchable_as_dense(c, "p_max_pu")[df.index]
        u = pd.DataFrame({
            "component": c,
            "carrier": df.carrier,
            "tech": cls,
            "zone": df[bus_col],
            "extendable": df.p_nom_extendable.astype(bool),
            "p_nom": df.p_nom,
            "p_nom_opt": df.p_nom_opt,
            "eff": df.efficiency if c == "Link" else 1.0,
            "fixed": (hi - lo).abs().max() < 1e-9,
        }, index=df.index)
        u["cap"] = u.p_nom_opt.where(u.extendable, u.p_nom)
        u = u.join(tech[["mode", "H", "cos_phi", "m_min", "avail", "ibr_w", "i_ibr",
                         "e_coef", "s_coef"]], on="tech")
        u["e_coef"] *= u.eff
        u["s_coef"] *= u.eff
        parts.append(u)
    if not parts:
        return pd.DataFrame()
    u = pd.concat(parts)
    u.index.name = "name"
    return u


def capacity_contribution(u: pd.DataFrame, by_zone: bool = True) -> pd.DataFrame:
    """E_k och S_k om hela kapaciteten vore inkopplad (utan p_max_pu), per enhet eller zon × teknik."""
    df = u[["component", "carrier", "tech", "zone", "mode"]].copy()
    df["cap_el_MW"] = u.eff * u.cap
    df["S_n_MVA"] = u.eff * u.cap / u.cos_phi
    derate = np.where(u["mode"] == "gfm", u.avail, 1.0)
    df["Ek_max_GWs"] = u.e_coef * u.cap * derate / 1e3
    df["Sk_max_GVA"] = u.s_coef * u.cap / 1e3
    df["P_ibr_GW"] = np.where(u["mode"].isin(["ibr", "gfm"]), u.ibr_w * u.cap, 0.0) / 1e3
    df["Ifault_ibr_GVA"] = np.where(u["mode"].isin(["ibr", "gfm"]), u.i_ibr * u.cap, 0.0) / 1e3
    if not by_zone:
        return df
    num = ["cap_el_MW", "S_n_MVA", "Ek_max_GWs", "Sk_max_GVA", "P_ibr_GW", "Ifault_ibr_GVA"]
    return df.groupby(["zone", "tech"])[num].sum()


def online_capacity(n, u: pd.DataFrame, kind: str,
                    online: pd.DataFrame | None = None) -> pd.DataFrame:
    """Inkopplad kapacitet u(t) [MW, i p_nom-enheter] för commit-enheterna (snapshot × enhet).

    kind = "on" kräver `online`, den lösta p_online (saknade enheter = 0).
    """
    if kind not in KINDS + ("on",):
        raise ValueError(f"kind måste vara en av {KINDS + ('on',)}")
    parts = []
    for c, uc in u[u["mode"] == "commit"].groupby("component"):
        avail = n.get_switchable_as_dense(c, "p_max_pu")[uc.index] * uc.cap
        if kind == "max":
            parts.append(avail)
            continue
        if kind == "on":
            p = online.reindex(index=n.snapshots, columns=uc.index).fillna(0.0)
            fixed = uc.index[uc.fixed]
            p[fixed] = avail[fixed]
            parts.append(p)
            continue
        attr, col = _DISPATCH[c]
        p = getattr(n, attr)[col].reindex(index=n.snapshots, columns=uc.index).fillna(0.0)
        p = p.clip(lower=0.0).clip(upper=avail)
        if kind == "hi":
            m = uc.m_min.where(uc.m_min > 0)          # m_min = 0: allt tillgängligt kan vara inkopplat
            p = (p / m).clip(upper=avail).fillna(avail)
        fixed = uc.index[uc.fixed]
        p[fixed] = avail[fixed]
        parts.append(p)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=n.snapshots)


def stability_metrics(n, u: pd.DataFrame, sync_weight: dict | None = None,
                      sk_include_ibr: bool = False,
                      online: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tidsserier, kolumner (metric, zon). Enheter GWs, GVA, GW, –.

    Ek_<kind>, Sk_<kind>, SCR_<kind> för kind i max/hi/lo (+ on med `online`); P_ibr (installerad) och P_ibr_out
    (inmatad), båda ibr_w-viktade. Ek_<kind> har dessutom kolumnen SYSTEM = Σ_z w_z · Ek_z.

    SCR = S_k / P_ibr_out, mot omriktarnas FAKTISKA inmatning i timmen (batteriets laddning
    räknas inte). Mot installerad effekt ligger SCR under 1 i DK, FI och SE-S redan i dagens
    system, som fungerar — det måttet mäter alltså inte ett verkligt problem. SCR saknas
    (NaN) i timmar med under 1 MW inmatning.
    """
    sw = sync_weight or {}
    zones = list(n.buses.index[n.buses.carrier == "AC"])
    sn = n.snapshots

    def by_zone(values: pd.Series) -> pd.Series:           # konstant per zon [G-enheter]
        return values.groupby(u.loc[values.index, "zone"]).sum().reindex(zones, fill_value=0.0) / 1e3

    sc, gfm = u[u["mode"] == "syncon"], u[u["mode"] == "gfm"]
    ibr = u[u["mode"].isin(["ibr", "gfm"])]
    ek_const = by_zone(sc.e_coef * sc.cap) + by_zone(gfm.e_coef * gfm.avail * gfm.cap)
    sk_const = by_zone(sc.s_coef * sc.cap)
    if sk_include_ibr:
        sk_const = sk_const + by_zone(ibr.i_ibr * ibr.cap)
    p_ibr = by_zone(ibr.ibr_w * ibr.cap)
    p_out = [getattr(n, _DISPATCH[c][0])[_DISPATCH[c][1]]
             .reindex(index=sn, columns=g.index).fillna(0.0).clip(lower=0.0) * g.ibr_w
             for c, g in ibr[ibr.ibr_w > 0].groupby("component")]
    p_ibr_out = (pd.concat(p_out, axis=1).T.groupby(ibr.zone).sum().T
                 .reindex(index=sn, columns=zones, fill_value=0.0) / 1e3
                 if p_out else pd.DataFrame(0.0, index=sn, columns=zones))

    res = {}
    cm = u[u["mode"] == "commit"]
    kinds = KINDS + (("on",) if online is not None else ())
    for kind in kinds:
        on = online_capacity(n, u, kind, online)
        for metric, coef, const in (("Ek", cm.e_coef, ek_const), ("Sk", cm.s_coef, sk_const)):
            val = (on * coef[on.columns]).T.groupby(cm.zone[on.columns]).sum().T
            res[f"{metric}_{kind}"] = val.reindex(index=sn, columns=zones, fill_value=0.0) / 1e3 + const
        res[f"SCR_{kind}"] = res[f"Sk_{kind}"] / p_ibr_out.where(p_ibr_out >= 1e-3)
    res["P_ibr"] = pd.DataFrame(np.tile(p_ibr.values, (len(sn), 1)), index=sn, columns=zones)
    res["P_ibr_out"] = p_ibr_out

    w = pd.Series({z: float(sw.get(z, 1.0)) for z in zones})
    for kind in kinds:
        res[f"Ek_{kind}"][SYSTEM] = (res[f"Ek_{kind}"] * w).sum(axis=1)
    out = pd.concat(res, axis=1, names=["metric", "zone"])
    return out


def stability_summary(ts: pd.DataFrame, weights: pd.Series,
                      thresholds_gws: tuple = (100, 120, 145)) -> pd.DataFrame:
    """Per zon (+ SYSTEM): min/p05/median för E_k, S_k och SCR; för SYSTEM även timmar under trösklarna.

    Timmarna viktas med snapshot-vikterna (2h-snapshot = 2 h). Kvantilerna är oviktade,
    vilket är exakt när alla snapshots har samma vikt.
    """
    rows = {}
    on = ("Ek_on", "Sk_on", "SCR_on") if "Ek_on" in ts else ()
    for metric in on + ("Ek_max", "Ek_hi", "Ek_lo", "Sk_max", "Sk_lo", "SCR_max", "SCR_hi",
                        "SCR_lo", "P_ibr_out"):
        df = ts[metric]
        rows[(metric, "min")] = df.min()
        rows[(metric, "p05")] = df.quantile(0.05)
        rows[(metric, "median")] = df.median()
    for thr in thresholds_gws:
        for kind in ("on", "hi", "lo") if on else ("hi", "lo"):
            sys_ek = ts[(f"Ek_{kind}", SYSTEM)]
            below = float(weights.reindex(ts.index)[sys_ek < thr].sum())
            rows[(f"h_Ek_{kind}<{thr:g}", "h")] = pd.Series({SYSTEM: below})   # bara systemet
    out = pd.DataFrame(rows)
    out.columns = [m if s == "h" else f"{m}_{s}" for m, s in out.columns]
    out["P_ibr_GW"] = ts["P_ibr"].iloc[0]
    order = [z for z in out.index if z != SYSTEM] + [SYSTEM]
    return out.reindex(order)


def stability_report(n, sync_weight: dict | None = None,
                     thresholds_gws: tuple = (100, 120, 145), sdata: dict | None = None,
                     online: pd.DataFrame | None = None) -> dict:
    """Allt för ett nätverk: enhetstabell, kapacitetsbidrag, tidsserier och sammanfattning.
    sync_weight = None tar vikterna ur sdata (zones.yaml)."""
    sdata = sdata or stability_data()
    u = unit_table(n, sdata)
    sw = sdata["sync_weight"] if sync_weight is None else sync_weight
    ts = stability_metrics(n, u, sw, online=online)
    return {"units": u, "capacity": capacity_contribution(u), "timeseries": ts,
            "summary": stability_summary(ts, n.snapshot_weightings.generators, thresholds_gws)}


def write_stability_reports(rep: dict, outdir) -> None:
    """Skriver stability_capacity.csv, stability_timeseries.csv och stability_summary.csv."""
    outdir = Path(outdir)
    rep["capacity"].to_csv(outdir / "stability_capacity.csv")
    flat = rep["timeseries"].copy()
    flat.columns = [f"{m}_{z}" for m, z in flat.columns]
    flat.to_csv(outdir / "stability_timeseries.csv")
    rep["summary"].to_csv(outdir / "stability_summary.csv")
