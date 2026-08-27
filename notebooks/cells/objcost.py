"""objcost — systemkostnad på RIKTIG basis: målfunktionen rensad från hydrons
pseudokostnad, med budtrappan korrekt hanterad.

⚠️ DEN RÅA MÅLFUNKTIONEN ÄR VILSELEDANDE i två oberoende avseenden, och båda har
lurat oss förut:

  1. HYDRONS mc ÄR ETT VATTENVÄRDE, INTE EN RESURSFÖRBRUKNING. Vattnet är gratis;
     mc styr bara NÄR det används. run392 såg ut att kosta +16,4 % mot run360 —
     ren bokföring, hydrons mc gick 41,0 → 73,7 €/MWh på 176,7 TWh. Rensat föll
     kostnaden i stället 5,2 %.
  2. BUDTRAPPAN (--hydro-bid-ladder) lägger Σ_k offset_k·d_k i objektivet. run410
     visade −4,23 % medan den VERKLIGA kostnaden STEG. Offsetterna summerar till
     noll i marginalen, men hydron går sällan för fullt, så den INFRAMARGINELLA
     rabatten sänker det bokförda budet (−9,4 % i run410). Medelvärdesbevarande i
     MARGINALEN, inte i totalkostnaden.

  real = objective + objective_constant
         − [ Σ_t Σ_k (mc_t + offset_k)·d_k·w_t  −  VOM·Σ_t p_dispatch,t·w_t ]

  ⭐ VOM-termen läggs TILLBAKA: reservoarens mc är `max(vattenvärde, VOM)`, så den
  genuina drift- och underhållskostnaden (0,6 €/MWh) ligger INUTI det som annars
  rensas bort — ~106 M€/år (0,52 %) i baseline. Med den blir dispatchläget dessutom
  rätt av sig självt: där ÄR mc = VOM platt, så pseudo blir 0 och real = total.

Utan trappa degenererar sista termen till Σ_t p_dispatch·mc_t·w_t (K=1, offset 0),
så SAMMA uttryck gäller i båda lägena — det är hela poängen med cellen. K och BREDD
läses ur run_meta.txt, så du kan inte råka jämföra en trappkörning på fel basis.
Nivåfyllnaden rekonstrueras EXAKT ur p_dispatch: växande offset_k gör att LP:t
fyller billigaste nivån först, alltså d_k = min(max(p − k·cap, 0), cap), cap = p_nom/K.

⚠️ objective_constant (låst kapacitets kapitalkostnad) ingår INTE i n.objective och
MÅSTE adderas — två körningar med olika låst flotta är annars ojämförbara.

Förutsätter bootstrap.py (LABEL, LABEL2, ROOT, n, pd, np).
Exponerar real_cost(label=None) och cost_table(*labels) för vidare bruk.
"""

# Cellen kan importeras fristående (från explore_results.py) ELLER klistras in
# i en notebook-cell efter att bootstrap.py redan körts där (explore.ipynb).
# Guard: skippar importen om bootstrap redan satt sina globaler i detta namnrum.
if 'LABEL' not in globals():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from bootstrap import *  # noqa: F401,F403

import re as _re


def _ladder(label):
    """(K, BREDD) ur run_meta.txt, eller None om trappan var av.

    ⚠️ Sedan budtrappan blev DEFAULT (b9cb401, 2026-08-23) står den normalt INTE i
    `argv:` — bara i `flaggor:` som `bidladder-K_BREDD`. Att bara läsa flaggan gav
    därför tyst `None` för varje körning från run420 och framåt, offsettermen föll
    bort ur hydrons bokförda bud, och `real` blev för LÅG: 19 032,8 i stället för
    20 250,6 M€/år för run420 (−6,0 %). Felet syntes som avstämningsrestens
    −1 217,8 M€/år (−3,8 %), som nu stänger till +0,1.
    Ordningen är medveten: en uttrycklig flagga vinner över `flaggor:`-taggen.
    """
    p = ROOT / 'results' / label / 'run_meta.txt'
    if not p.is_file():
        return None
    txt = p.read_text(encoding='utf-8')
    m = _re.search(r'--hydro-bid-ladder\s+(\d+):(\d+(?:\.\d+)?)', txt)
    if m is None:
        if _re.search(r'\bno-bidladder\b', txt):
            return None
        m = _re.search(r'\bbidladder-(\d+)_(\d+(?:\.\d+)?)', txt)
    return (int(m.group(1)), float(m.group(2))) if m else None


def _hydro_vom():
    """Reservoarvattenkraftens VOM, €/MWh, ur config/zones.yaml (fallback 0,6)."""
    import yaml as _yaml
    try:
        c = _yaml.safe_load(open(ROOT / 'config' / 'zones.yaml', encoding='utf-8'))
        return float(c['costs']['hydro']['vom_eur_per_mwh'])
    except Exception:
        return 0.6


def _net(label):
    """Nätverket för label — återanvänder bootstraps n när det är LABEL."""
    if label == LABEL and 'n' in globals():
        return n
    import pypsa
    return pypsa.Network(str(ROOT / 'results' / label / 'network.nc'))


def _series(df_t, static, name, index):
    """marginal_cost kan vara statisk ELLER tidsvarierande — hantera båda."""
    if name in df_t.columns:
        return df_t[name].reindex(index).fillna(0.0)
    return pd.Series(float(static), index=index)


def real_cost(label=None):
    """Kostnadsuppdelning i M€/år. Nyckeln 'real' är den jämförbara siffran."""
    label = label or LABEL
    m = _net(label)
    w = m.snapshot_weightings.objective
    yrs = float(w.sum()) / 8760.0
    lad = _ladder(label)

    # ── Hydrons pseudokostnad (trappmedveten) ───────────────────────────────
    pdis = m.storage_units_t.p_dispatch
    hyd = [u for u in pdis.columns if u.endswith(' hydro')]
    pseudo = 0.0
    for u in hyd:
        mc = _series(m.storage_units_t.marginal_cost,
                     m.storage_units.at[u, 'marginal_cost'], u, pdis.index)
        if lad is None:
            pseudo += float((pdis[u] * mc * w).sum())
        else:
            K, width = lad
            cap = float(m.storage_units.at[u, 'p_nom']) / K
            for k in range(K):
                d_k = (pdis[u] - k * cap).clip(lower=0.0, upper=cap)
                pseudo += float((d_k * (mc + width * ((k + 0.5) / K - 0.5)) * w).sum())

    # ⭐ VOM ÄR EN VERKLIG KOSTNAD OCH SKA INTE RENSAS BORT (2026-08-27).
    # Reservoarens marginal_cost är `max(kurva/proxy, VOM)` — VOM 0,6 €/MWh är bara ett
    # GOLV, så när kurvan (~58-87) gäller ligger den genuina drift- och underhålls-
    # kostnaden INUTI talet som dras bort. `pseudo` ska bara vara VATTENVÄRDET, dvs.
    # mc över VOM. Gör dessutom dispatchläget rätt av sig självt: med frysta kapaciteter
    # är mc = VOM platt ⇒ pseudo = 0 ⇒ real = total, vilket är korrekt.
    vom = _hydro_vom()
    hydro_mwh = float(sum((pdis[u] * w).sum() for u in hyd))
    pseudo -= vom * hydro_mwh

    const = float(getattr(m, 'objective_constant', 0.0) or 0.0)
    total = float(m.objective) + const

    # ── Grov uppdelning för avstämning (kapital + rörlig + spill) ───────────
    capex = 0.0
    for comp, col in (('generators', 'p_nom_opt'), ('storage_units', 'p_nom_opt'),
                      ('stores', 'e_nom_opt'), ('links', 'p_nom_opt')):
        df = getattr(m, comp)
        if df.empty or 'capital_cost' not in df.columns:
            continue
        capex += float((df['capital_cost'] * df[col]).sum())

    opex = 0.0
    for comp, tname in (('generators', 'p'), ('links', 'p0'), ('stores', 'p')):
        df, dft = getattr(m, comp), getattr(m, comp + '_t')
        if df.empty or tname not in dft or dft[tname].empty:
            continue
        for u in dft[tname].columns:
            mc = _series(dft.get('marginal_cost', pd.DataFrame(index=w.index)),
                         df.at[u, 'marginal_cost'] if 'marginal_cost' in df.columns else 0.0,
                         u, w.index)
            opex += float((dft[tname][u] * mc * w).sum())

    # Hydrons VOM är nu en riktig opex-post (den rensas inte längre bort som pseudo),
    # så den måste med i avstämningen — annars slår resten ut på just det beloppet.
    opex += vom * hydro_mwh

    spill = 0.0
    if 'spill' in m.storage_units_t and not m.storage_units_t.spill.empty:
        for u in m.storage_units_t.spill.columns:
            sc = float(m.storage_units.at[u, 'spill_cost']) if 'spill_cost' in m.storage_units.columns else 0.0
            spill += float((m.storage_units_t.spill[u] * sc * w).sum())

    return dict(label=label, years=yrs, ladder=lad,
                objective=float(m.objective) / 1e6 / yrs,
                constant=const / 1e6 / yrs,
                total=total / 1e6 / yrs,
                hydro_pseudo=pseudo / 1e6 / yrs,
                real=(total - pseudo) / 1e6 / yrs,
                capex=capex / 1e6 / yrs, opex=opex / 1e6 / yrs,
                spill=spill / 1e6 / yrs,
                residual=(total - capex - opex - spill - pseudo) / 1e6 / yrs,
                hydro_twh=float(sum((pdis[u] * w).sum() for u in hyd)) / 1e6 / yrs)


def cost_table(*labels):
    """Jämför körningar på REAL basis. Utan argument: LABEL (+ LABEL2 om satt)."""
    if not labels:
        labels = [LABEL] + ([LABEL2] if globals().get('LABEL2') else [])
    rows = [real_cost(l) for l in labels]
    hdr = ('körning', 'trappa', 'objective', 'konstant', 'hydro-pseudo',
           'REAL', 'varav capex', 'hydro TWh')
    print(f"{hdr[0]:<32}{hdr[1]:>7}" + ''.join(f'{h:>14}' for h in hdr[2:]))
    for r in rows:
        lab = f"{r['ladder'][0]}:{r['ladder'][1]:.0f}" if r['ladder'] else '–'
        print(f"{r['label']:<32}{lab:>7}{r['objective']:14.1f}{r['constant']:14.1f}"
              f"{r['hydro_pseudo']:14.1f}{r['real']:14.1f}{r['capex']:14.1f}"
              f"{r['hydro_twh']:14.2f}")
    for r in rows:                      # avstämning: säger ifrån i stället för att dölja
        if abs(r['residual']) > 0.02 * abs(r['total']):
            print(f"  ⚠️ {r['label']}: uppdelningen summerar inte till målfunktionen "
                  f"(rest {r['residual']:.1f} M€/år = {r['residual']/r['total']*100:.1f} %) "
                  f"— capex/opex-posterna är INDIKATIVA, 'real' är däremot exakt.")
    if len(rows) > 1:
        b = rows[0]
        print(f"\nmot {b['label']} (⚠️ rå målfunktion är EJ jämförbar över trappan):")
        for r in rows[1:]:
            d = r['real'] - b['real']
            print(f"  {r['label']:<32} real {d:+10.1f} M€/år  ({d/b['real']*100:+.2f} %)")
    return rows


_rows = cost_table()
