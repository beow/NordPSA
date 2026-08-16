"""flexprov — balansbidrag per tidsskala (FlexProv, bandpass-L1) för en KÖRNING.

Frågan: när residuallasten svänger på en viss tidsskala, VEM täcker svängningen?
Svaret ges som andelar av FlexNeed som summerar till exakt 1. Andelen kan vara
NEGATIV: en källa som förstärker svängningen i stället för att dämpa den.

    FlexNeed = ½ · Σ_t w_t · |ΔRload_t|            (energi som flyttas, GWh)
    andel_c  =   Σ_t w_t · Δc_t · sign(ΔRload_t) / Σ_t w_t · |ΔRload_t|

BANDPASS, inte low-pass: varje horisont mäter avvikelsen mot NÄSTA GRÖVRE medel
(dygn mot dygnsmedel, vecka mot veckomedel) och isolerar ett rent tidsband.
Kovariansversionen mäter allt mot månadsmedlet och blandar därmed ihop banden.

w_t är körningens `snapshot_weightings` (h per snapshot), så 1h/2h/3h ger samma
GWh-siffror. Resamplade band viktas med antalet timmar i varje hink.

## Vad som är FLEXIBILITET och vad som är RESIDUALLAST

Identiteten kommer ur PyPSAs nodbalans på AC-bussen:

    Rload  =  AC-laster − VRE − hydro_ror   ≡   Σ komponenter

⚠️ Två poster hör till RESIDUALLASTEN, inte till flexibiliteten — de är must-run
och kan inte välja något (detta var buggen i den gamla low-pass-versionen):
  • `hydro_ror`   — strömkraft, p_min_pu = p_max_pu
  • `EV car inflex` — den icke-styrbara delen av EV-lasten, ligger på AC-bussen

⚠️ `nuclear` och `thermal` ÄR komponenter men är också must-run i NordPSA. Deras
bidrag är exogent (tillgänglighets- resp. produktionsprofil), inte ett val. Läs
dem som "hur mycket balansering råkar profilen ge", inte som styrbarhet.

Efterfrågeflex räknas med omvänt tecken (−p0 på länken), så positiv andel alltid
betyder "hjälper balansen": elektrolys/EV/värmepump som drar NER förbrukningen i
en högtimme får plusbidrag.

⚠️ Additiviteten är EXAKT per konstruktion — Σ andelar = 1 är en räknekontroll,
inte ett resultat. Det som betyder något är FÖRDELNINGEN mellan källor.

Förutsätter bootstrap.py (globalerna `n`, `ZONES`, `LABEL`).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.future.infer_string = False

# --- inställningar ----------------------------------------------------------
ZONE   = None            # None = hela Norden; annars "SE-N" eller ["SE-N","SE-S"]
PERIOD = None            # None = hela körningen; annars "2024" eller "2024-01"
SCALES = [               # (etikett, resample-frekvens, grövre medel att mäta mot)
    ("Dygn",   None, "D"),       # None = körningens egen upplösning
    ("Vecka",  "D",  "W"),
    ("Säsong", "W",  "Y"),       # "M" i stället för "Y" ger månadsbandet
]

# Komponenter: etikett -> urval. Ordningen styr staplarna.
GEN_COMPS  = {                       # etikett -> carrier
    "Kärnkraft":         "nuclear",
    "Must-run-termik":   "thermal",
    "Gas":               "gas",
    "Handel - Kontinent": "market",   # kontinentventilen (DE-LU-pris)
    "Slack":             "slack",
}
SU_COMPS   = {"Vattenkraft (magasin)": "hydro", "Batteri": "battery"}
LINK_COMPS = {                       # etikett -> (carriers, sida)
    "Kraftvärme (KVV)":      (["heat chp"],                 "bus1"),
    "H2-elektrolys":         (["electrolyser"],             "bus0"),
    "El-värme (VP+elpanna)": (["heat hp", "heat elboiler"], "bus0"),
    "EV-laddning":           (["EV charger"],               "bus0"),
}
# Handeln delas i två: AC-länkar mot nordiska grannar respektive kontinentventilen,
# så man ser om zonen balanseras av grannarna eller av kontinenten. "Handel - Norden"
# är per definition 0 för hela Norden — då tar alla interna länkar ut varandra.
NTC_LABEL    = "Handel - Norden"
PLOT_ORDER   = [                     # stapel-/legendordning (störst först)
    "Vattenkraft (magasin)", NTC_LABEL, "Handel - Kontinent", "Handel (netto)",
    "Kraftvärme (KVV)", "Kärnkraft", "EV-laddning", "Batteri", "H2-elektrolys",
    "El-värme (VP+elpanna)", "Must-run-termik", "Gas", "Övrigt", "Slack",
]
VRE_CARRIERS = ["wind_onshore", "wind_offshore", "solar"]

COMP_COLORS = {                      # samma hex som temp/balansbidrag_country_bandpass.py
    "Vattenkraft (magasin)": "#1f77b4", NTC_LABEL:           "#ffbb78",
    "Handel - Kontinent":    "#ff7f0e", "Kraftvärme (KVV)":  "#8c564b",
    "Kärnkraft":             "#9467bd", "EV-laddning":       "#2ca02c",
    "Batteri":               "#d62728", "H2-elektrolys":     "#e377c2",
    "El-värme (VP+elpanna)": "#7f7f7f", "Must-run-termik":   "#bcbd22",
    "Gas":                   "#17becf", "Slack":             "#c7c7c7",
    # eSett-sidan kan inte dela handeln (en enda nettoposition) och har en
    # restpost `other`. Egna färger så jämförelser modell/verklighet blir läsbara.
    "Handel (netto)":        "#fdae61", "Övrigt":            "#9e9e9e",
}


def _zonelist(zone):
    if zone is None:
        return list(ZONES)
    return [zone] if isinstance(zone, str) else list(zone)


def build_components(net=None, zone=ZONE, ror_in_hydro=False) -> pd.DataFrame:
    """AC-bussbalansen för en zonmängd, uppdelad i flexkomponenter + Rload.

    ror_in_hydro=True flyttar strömkraften från residuallasten till hydro-
    komponenten. Identiteten håller ändå (RoR byter bara sida), men hydro blir
    då TOTAL vattenkraft — vilket är vad man behöver för att jämföra mot eSett,
    som inte delar upp hydro i magasin och strömkraft.
    """
    net = net if net is not None else n
    zs  = _zonelist(zone)
    ac  = [b for b in zs if b in net.buses.index]
    idx = pd.DatetimeIndex(net.snapshots)
    Z   = lambda: pd.Series(0.0, index=idx)

    g, gt = net.generators, net.generators_t.p
    su, sut = net.storage_units, net.storage_units_t.p
    inzone_g  = g.bus.isin(ac)
    inzone_su = su.bus.isin(ac)

    d = {}
    for lab, c in GEN_COMPS.items():
        sel = g.index[inzone_g & (g.carrier == c)]
        d[lab] = gt[sel].sum(axis=1) if len(sel) else Z()
    for lab, c in SU_COMPS.items():
        sel = su.index[inzone_su & (su.carrier == c)]
        d[lab] = sut[sel].sum(axis=1) if len(sel) else Z()

    # Länkar: bidraget till AC-bussen är −p_side (PyPSA drar av p vid varje buss).
    L = net.links
    for lab, (carriers, side) in LINK_COMPS.items():
        sel = L.index[L.carrier.isin(carriers) & L[side].isin(ac)]
        pk  = {"bus0": "p0", "bus1": "p1", "bus2": "p2"}[side]
        d[lab] = -net.links_t[pk][sel].sum(axis=1) if len(sel) else Z()

    # Intern NTC: bara länkar som KORSAR zonmängdens gräns (interna tar ut varandra).
    acl  = L[L.carrier == "AC"]
    imp  = acl.index[acl.bus1.isin(ac) & ~acl.bus0.isin(ac)]
    exp  = acl.index[acl.bus0.isin(ac) & ~acl.bus1.isin(ac)]
    d[NTC_LABEL] = (-net.links_t.p1[imp].sum(axis=1) if len(imp) else Z()) \
                 + (-net.links_t.p0[exp].sum(axis=1) if len(exp) else Z())

    df = pd.DataFrame(d, index=idx)

    # Residuallast: AC-laster − VRE − hydro_ror  (must-run hör till lasten)
    ld  = net.loads
    lsel = ld.index[ld.bus.isin(ac)]
    lt  = net.loads_t.p_set
    lt.index = idx
    load = lt[[c for c in lsel if c in lt.columns]].sum(axis=1)
    vre  = gt[g.index[inzone_g & g.carrier.isin(VRE_CARRIERS)]].sum(axis=1)
    ror  = g.index[inzone_g & (g.carrier == "hydro")]
    rorp = gt[ror].sum(axis=1) if len(ror) else Z()
    if ror_in_hydro:
        df["Vattenkraft (magasin)"] += rorp        # -> total vattenkraft
        df["Rload"] = load - vre
    else:
        df["Rload"] = load - vre - rorp

    comps = [c for c in df.columns if c != "Rload"]
    resid = (df[comps].sum(axis=1) - df["Rload"]).abs()
    if resid.max() > 1.0:
        print(f"  ⚠️ nodbalansen går inte ihop: max {resid.max():.1f} MW, "
              f"medel {resid.mean():.2f} MW — komponentlistan missar något")
    df.attrs["hours"] = pd.Series(
        net.snapshot_weightings.objective.values, index=idx)
    return df


def flexprov_bandpass(df, comps, freq_h, period_l, hours):
    """FlexProv-andelar för ett band l|h. freq_h=None = körningens upplösning.

    ⚠️ Alla medelvärden är TIMVIKTADE. Det spelar roll vid ofullständiga hinkar:
    ISO-veckor som spänner över årsskiftet ska väga efter sina faktiska timmar i
    säsongsbandet, inte räknas som hela veckor. Oviktat ger ~0,1 pp fel.
    """
    cols = list(comps) + ["Rload"]
    if freq_h is None:
        Xh, wh = df[cols], hours
    else:
        wh = hours.resample(freq_h).sum()
        Xh = df[cols].mul(hours, axis=0).resample(freq_h).sum().div(wh, axis=0)
    per  = Xh.index.to_period(period_l)
    Xl   = (Xh.mul(wh, axis=0).groupby(per).transform("sum")
              .div(wh.groupby(per).transform("sum"), axis=0))
    d    = Xh - Xl
    w    = wh.values
    sign = np.sign(d["Rload"].values)
    den  = np.sum(w * np.abs(d["Rload"].values))
    shares = {c: np.sum(w * d[c].values * sign) / den for c in comps}
    assert abs(sum(shares.values()) - 1) < 1e-6, sum(shares.values())
    return shares, 0.5 * den / 1e3                   # MWh -> GWh


def flexprov(df, comps=None, scales=SCALES):
    """-> (andels-DataFrame [band × komponent], FlexNeed-serie i GWh)."""
    comps = comps or [c for c in df.columns if c != "Rload"]
    hours = df.attrs["hours"]
    res  = {lab: flexprov_bandpass(df, comps, fh, pl, hours) for lab, fh, pl in scales}
    fp   = pd.DataFrame({lab: res[lab][0] for lab, _, _ in scales}).T
    need = pd.Series({lab: res[lab][1] for lab, _, _ in scales})
    return fp.loc[[s[0] for s in scales]], need


def plot_flexprov(fp, title, drop_tiny=0.005):
    """Stackade staplar. Komponenter under drop_tiny överallt utelämnas."""
    order = [c for c in PLOT_ORDER if c in fp.columns]
    fp    = fp[order + [c for c in fp.columns if c not in order]]
    keep  = fp.columns[(fp.abs() > drop_tiny).any()]
    ax = (100 * fp[keep]).plot(kind="bar", stacked=True, alpha=0.85, figsize=(6, 7),
                               title=title,
                               color=[COMP_COLORS.get(c, "silver") for c in keep])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(100, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("% av FlexNeed i bandet")
    ax.set_xticklabels(fp.index, rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    return ax


def report(zone=ZONE, period=PERIOD, net=None, label=None, plot=True,
           ror_in_hydro=False):
    df = build_components(net, zone, ror_in_hydro)
    if period:
        df = df.loc[period]
        df.attrs["hours"] = df.attrs["hours"].loc[period]
    fp, need = flexprov(df)

    who  = "Norden" if zone is None else (zone if isinstance(zone, str) else "+".join(zone))
    lab  = label or (LABEL if "LABEL" in globals() else "")
    span = f"{df.index[0]:%Y-%m-%d} – {df.index[-1]:%Y-%m-%d}"
    yrs  = df.attrs["hours"].sum() / 8760.0   # som temp/balansbidrag_bandpass.py
    print(f"\n{'='*74}\n{who}  [{lab}]  {span}  ({len(df)} steg, {yrs:.2f} år)"
          f"  Rload medel {df['Rload'].mean():.0f} MW\n{'='*74}")
    for b in fp.index:
        print(f"{b:6s} FlexNeed = {need[b]:9.1f} GWh ({need[b]/yrs:8.1f} GWh/år)"
              f"  | summa {fp.loc[b].sum():.4f}")
    print()
    print("% av FlexNeed i bandet:")
    print((100 * fp).round(1).T.to_string())
    if plot:
        plot_flexprov(fp, f"Balansbidrag (FlexProv)\n{who}, {lab}")
        plt.show()
    return df, fp, need


if __name__ != "__main__":
    pass


def energy_split(df, fp, need, comp="Vattenkraft (magasin)"):
    """Dela EN komponents energi i baskraft / flyttad energi / balanskraft (TWh/år).

    FlexProv är additiv i ENERGI, så `andel × FlexNeed` är den energi komponenten
    faktiskt förskjuter i bandet — i rätt riktning, dvs. medhållet residuallasten:

        E_tot      = Σ w·H                       total produktion
        M_tot      = ½ Σ w·|H − H_årsmedel|      energi som flyttas mot platt drift
        baskraft   = E_tot − M_tot               flyter oavsett
        balans_b   = andel_b × FlexNeed_b        nyttig förskjutning i band b
        träffsäkerhet_b = balans_b / M_b,  M_b = ½ Σ w·|Δ_b H|

    ⚠️ Banden teleskoperar i Δ men INTE i |Δ|, så Σ_b M_b ≥ M_tot. Summera därför
    inte M_b över band — jämför varje band för sig.
    ⚠️ Negativ balans_b = komponenten förstärker svängningen i det bandet.
    """
    h  = df.attrs["hours"]
    H  = df[comp]
    yrs = h.sum() / 8760.0
    E_tot = (H * h).sum() / 1e6 / yrs
    M_tot = 0.5 * (h * (H - (H * h).sum() / h.sum()).abs()).sum() / 1e6 / yrs

    rows = {}
    for lab, fh, pl in SCALES:
        if fh is None:
            Xh, wh = H, h
        else:
            wh = h.resample(fh).sum()
            Xh = (H * h).resample(fh).sum() / wh
        per = Xh.index.to_period(pl)
        dH  = Xh - (Xh * wh).groupby(per).transform("sum") / wh.groupby(per).transform("sum")
        M_b = 0.5 * (wh * dH.abs()).sum() / 1e6 / yrs
        B_b = fp.loc[lab, comp] * need[lab] / 1e3 / yrs
        rows[lab] = {"flyttad M_b": M_b, "balans B_b": B_b,
                     "träffsäkerhet": B_b / M_b if M_b else np.nan}
    out = pd.DataFrame(rows).T
    print(f"\n{comp}  (TWh/år)")
    print(f"  produktion E_tot     {E_tot:7.1f}")
    print(f"  flyttad totalt M_tot {M_tot:7.1f}   ({M_tot/E_tot*100:.0f} % av produktionen)")
    print(f"  baskraft E_tot−M_tot {E_tot-M_tot:7.1f}   ({(E_tot-M_tot)/E_tot*100:.0f} %)")
    print(out.round(2).to_string())
    return out
