"""Analytisk terminalvärdeskurva λ(fyllnadsgrad, vecka) per zon.

Ersätter den årskonstanta `--terminal-lambda-profile` med en kurva som varierar över
året, i den rullande horisontens fönsterslut. Formen är parametrisk och GENERISK —
den ska bära nordisk hydrologi i allmänhet, inte 2023-25 i synnerhet.

    λ_k(w, z) = λ_bas[z] · A(w, z) · P_k( B(w, z) )

Tre knoppar som är avsiktligt SEPARERADE, så att kalibreringen kan röra en i taget:

  λ_bas[z]   NIVÅ, EUR/MWh. Årets genomsnittliga marginalvärde på lagrat vatten.
  A(w, z)    SÄSONG på nivån. Normaliserad till årsmedel EXAKT 1 (cosinus).
  P_k(B)     LUTNING mot fullt magasin, per SOC-segment. Normaliserad till medel
             EXAKT 1 över segmenten. B styr bara tiltet, aldrig nivån.

Utan normaliseringarna är knopparna sammanblandade: att göra kurvan brantare skulle
samtidigt sänka det genomsnittliga vattenvärdet, och kalibreringen skulle jaga sin
egen svans.

## Varför formen ser ut som den gör

P_k kommer ur λ(x) = exp(−B·x), x = fyllnadsgrad. Segment k täcker
x ∈ [k/K, (k+1)/K] och får sitt värde i mittpunkten. Fallande i k ⇒ V(SOC) blir
konkav ⇒ LP:t fyller de värdefulla segmenten först av sig självt, utan
ordningsvillkor. Det är samma styckvis-konkava trick som `hydro_terminal_value`
redan implementerar; den här modulen levererar bara koefficienterna.

⚠️ Kurvan verkar BARA i fönsterslut. Inuti ett fönster råder perfekt framsyn och
vattenvärdet är platt där ändå. Mekanismens upplösning är fönsterlängden, inte
veckan — `--rolling-weeks` är därför en förstahandsparameter.

## Var defaultvärdena kommer ifrån

`b_mean` per zon är INTE gissat: run268-experimentet (2026-08-07, 2024 @3h) mätte att
en brant profil ger de trängselinlåsta zonerna SE-N/NO-N en bias på −2,2/−2,9 mot
faktiskt och sänker takfrekvensen 3×, medan de kontinentkopplade zonerna
(SE-S, NO-S, FI) då underskjuter 15-22. Slutsatsen var att en GLOBAL profil inte kan
betjäna båda zontyperna. Defaulten kodar det: inlåsta zoner brantare.

`a_peak`/`b_peak` är hydrologiska antaganden och ska kalibreras:
  a_peak ≈ v5   nivån toppar sensk vinter, när ransoneringen är som hårdast
  b_peak ≈ v22  kurvan är brantast vid vårfloden, när spillrisken är akut

⚠️ ANTAGANDE: att en enda sinusform räcker för både A och B. Fig A.4 i Ek Fälth
m.fl. visar att den verkliga säsongen inte är rent sinusformad.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List

WEEKS = 52
DEFAULT_SEGMENTS = 5

# Zoner utan direkt kontinentkoppling. Deras vatten kan inte följa exportpriset, så
# kurvan måste falla brantare mot fullt magasin för att inte hamstra.
LOCKED_ZONES = ("SE-N", "NO-N")


@dataclass(frozen=True)
class CurveParams:
    """Fem fria tal per zon. Alla dimensionslösa — nivån bor i λ_bas."""
    a_amp:  float = 0.35   # säsongsamplitud på nivån, 0 = ingen säsong. Krav: < 1
    a_peak: float = 5.0    # vecka då nivån toppar
    b_mean: float = 2.5    # genomsnittlig brantid mot fullt magasin. Krav: ≥ 0
    b_amp:  float = 0.6    # säsongsvariation i brantid, andel av b_mean. Krav: |·| ≤ 1
    b_peak: float = 22.0   # vecka då kurvan är brantast (vårflodens spillrisk)

    def __post_init__(self) -> None:
        if not -1.0 < self.a_amp < 1.0:
            raise ValueError(f"a_amp måste ligga i (−1, 1), fick {self.a_amp} "
                             "— annars blir nivåfaktorn negativ någon vecka")
        if self.b_mean < 0.0:
            raise ValueError(f"b_mean måste vara ≥ 0, fick {self.b_mean}")
        if abs(self.b_amp) > 1.0:
            raise ValueError(f"|b_amp| måste vara ≤ 1, fick {self.b_amp} "
                             "— annars blir kurvan VÄXANDE i fyllnadsgrad någon vecka")


#: Startpunkt, inte facit. b_mean kodar run268:s mätning (se modulens docstring).
DEFAULTS: Dict[str, CurveParams] = {
    "SE-N": CurveParams(b_mean=4.0),
    "NO-N": CurveParams(b_mean=4.0),
    "SE-S": CurveParams(b_mean=1.5),
    "NO-S": CurveParams(b_mean=1.5),
    "FI":   CurveParams(b_mean=1.5),
}

#: λ_bas per zon, EUR/MWh — MÄTT på run260, inte satt för hand.
#: = medelvärdet av hydrons EFFEKTIVA bud, proxy(t) + μ/η, alltså vad expansionen
#: själv säger att vatten är värt. ⚠️ Det är INTE μ ensamt: μ är residualen som lyfter
#: zonens historiska pris upp till ett gemensamt systemvattenvärde. Att ankra på μ
#: vore inverterat — det skulle sätta nordvattnet högst (44-48) och NO-S lägst (16),
#: tvärtemot verklighetens prisordning. Härledd av anchor_from_run().
DEFAULT_ANCHOR: Dict[str, float] = {
    "SE-N": 76.72, "SE-S": 79.52, "NO-N": 78.14, "NO-S": 75.99, "FI": 84.42,
}


def week_of(ts) -> int:
    """ISO-vecka klämd till 1..52 (vecka 53 finns men kurvan är 52-periodisk)."""
    return min(52, int(getattr(ts, "isocalendar")()[1]))


def a_factor(week: int, p: CurveParams) -> float:
    """Säsongsfaktor på NIVÅN. Årsmedel exakt 1 (cosinus integrerar till noll)."""
    return 1.0 + p.a_amp * math.cos(2.0 * math.pi * (week - p.a_peak) / WEEKS)


def b_value(week: int, p: CurveParams) -> float:
    """Brantid mot fullt magasin för veckan. Golvad vid 0 = platt (linjär) kurva."""
    return max(0.0, p.b_mean * (1.0 + p.b_amp
                                * math.cos(2.0 * math.pi * (week - p.b_peak) / WEEKS)))


def segment_profile(b: float, segments: int = DEFAULT_SEGMENTS) -> List[float]:
    """Icke-växande multiplikatorer tomt→fullt, normaliserade till medel exakt 1.

    b = 0 ger en helt platt profil (alla 1.0) — det gamla LINJÄRA terminalvärdet.
    """
    if segments < 1:
        raise ValueError(f"segments måste vara ≥ 1, fick {segments}")
    xs = [(k + 0.5) / segments for k in range(segments)]
    vals = [math.exp(-b * x) for x in xs]
    mean = sum(vals) / len(vals)
    return [v / mean for v in vals]


def curve(week: int, zone: str,
          params: Dict[str, CurveParams] | None = None,
          segments: int = DEFAULT_SEGMENTS) -> tuple[float, List[float]]:
    """(nivåfaktor, segmentprofil) för zonen och veckan."""
    p = (params or DEFAULTS).get(zone, CurveParams())
    return a_factor(week, p), segment_profile(b_value(week, p), segments)


# ── Adaptrar mot hydro_terminal_value(), som är keyad på LAGRETS namn ────────────
# Lagren heter "<zon> hydro"; zonen är första ordet.

def _zone_of(unit: str) -> str:
    return unit.split()[0]


def lambdas_for_week(week: int, units: Iterable[str],
                     anchor: Dict[str, float] | None = None,
                     params: Dict[str, CurveParams] | None = None) -> Dict[str, float]:
    """λ_bas[z] · A(w, z) per lager — nivådelen av terminalvärdet."""
    anchor = anchor or DEFAULT_ANCHOR
    params = params or DEFAULTS
    out = {}
    for u in units:
        z = _zone_of(u)
        p = params.get(z, CurveParams())
        out[u] = float(anchor.get(z, 0.0)) * a_factor(week, p)
    return out


def profiles_for_week(week: int, units: Iterable[str],
                      params: Dict[str, CurveParams] | None = None,
                      segments: int = DEFAULT_SEGMENTS) -> Dict[str, List[float]]:
    """P_k(B(w, z)) per lager — lutningsdelen. Matchar `profile`-dicten i
    hydro_terminal_value(), som redan stödjer per-zons-profiler."""
    params = params or DEFAULTS
    return {u: segment_profile(b_value(week, params.get(_zone_of(u), CurveParams())),
                               segments)
            for u in units}


# ── Ankaret: mät, gissa inte ────────────────────────────────────────────────────

def anchor_from_run(label: str = "run260_baseline_2h",
                    efficiency: float = 0.9) -> Dict[str, float]:
    """λ_bas per zon ur en expansionskörning: hydrons EFFEKTIVA bud i genomsnitt.

        bud[z] = medel( proxy[z](t) ) + μ[z] / η

    proxy = det historiska zonpriset som hydron bjuder (network.py), μ = dualen på
    lagringsbalansen. Summan är vad expansionen faktiskt värderar vatten till; i
    run260 landar den på 76-84 i alla fem zoner trots att termerna var för sig
    spretar 33 respektive 32 enheter.

    ⚠️ Kräver att den rullande dispatchen körs med hydro_price_proxy=False, annars
    ligger λ och marginalen på olika skalor (se hydro_terminal_value docstring).
    """
    from pathlib import Path
    import pandas as pd

    root = Path(__file__).resolve().parents[2]
    wv = pd.read_csv(root / "results" / label / "water_value.csv",
                     index_col=0, parse_dates=True)
    pr = pd.read_csv(root / "results" / label / "prices.csv",
                     index_col=0, parse_dates=True)
    mp = pd.read_parquet(root / "data" / "processed" / "market_prices.parquet")
    if mp.index.tz is not None:
        mp.index = mp.index.tz_localize(None)
    sn = pr.index.tz_localize(None) if pr.index.tz is not None else pr.index

    out = {}
    for col in wv.columns:
        if not col.endswith(" hydro"):
            continue
        z = _zone_of(col)
        if z not in mp.columns:
            continue
        proxy = float(mp[z].reindex(sn).ffill().clip(lower=0.6).mean())
        out[z] = round(proxy + float(wv[col].mean()) / efficiency, 2)
    return out


# ── Persistens: kalibreringsloopens och inkopplingens gemensamma gränssnitt ─────

#: Kalibreringens utdata och den rullande dispatchens indata.
DEFAULT_PARAM_FILE = "config/terminal_curve.yaml"


def save_params(params: Dict[str, CurveParams], anchor: Dict[str, float],
                path: str | None = None, note: str = "") -> str:
    """Skriv parametrar + ankare till YAML. Returnerar sökvägen."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / DEFAULT_PARAM_FILE
    doc = {
        "note": note or "Terminalvärdeskurva λ_k(vecka, zon); se nordpsa/wv/terminal_curve.py",
        "anchor_eur_per_mwh": {z: float(v) for z, v in sorted(anchor.items())},
        "zones": {z: {"a_amp": c.a_amp, "a_peak": c.a_peak, "b_mean": c.b_mean,
                      "b_amp": c.b_amp, "b_peak": c.b_peak}
                  for z, c in sorted(params.items())},
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return str(p)


def load_params(path: str | None = None
                ) -> tuple[Dict[str, CurveParams], Dict[str, float]]:
    """Läs (params, anchor). Saknas filen returneras DEFAULTS/DEFAULT_ANCHOR."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / DEFAULT_PARAM_FILE
    if not p.exists():
        return dict(DEFAULTS), dict(DEFAULT_ANCHOR)
    doc = yaml.safe_load(p.read_text()) or {}
    params = {z: CurveParams(**kw) for z, kw in (doc.get("zones") or {}).items()}
    anchor = {z: float(v) for z, v in (doc.get("anchor_eur_per_mwh") or {}).items()}
    return (params or dict(DEFAULTS)), (anchor or dict(DEFAULT_ANCHOR))


# ── Inspektion ──────────────────────────────────────────────────────────────────

def describe(params: Dict[str, CurveParams] | None = None,
             anchor: Dict[str, float] | None = None,
             segments: int = DEFAULT_SEGMENTS,
             weeks: Iterable[int] = (5, 13, 22, 30, 44)) -> None:
    """Skriv ut kurvan för några veckor så att formen går att syna."""
    params = params or DEFAULTS
    anchor = anchor or DEFAULT_ANCHOR
    print(f"Terminalkurva λ_k(vecka, zon) = λ_bas · A(w) · P_k(B(w)), "
          f"{segments} segment à {100 // segments} % av volymen\n")
    for z in sorted(params):
        p = params[z]
        lock = " [inlåst]" if z in LOCKED_ZONES else ""
        print(f"{z}{lock}  λ_bas={anchor.get(z, 0.0):.1f}  "
              f"a_amp={p.a_amp:.2f} a_peak=v{p.a_peak:.0f}  "
              f"b_mean={p.b_mean:.2f} b_amp={p.b_amp:.2f} b_peak=v{p.b_peak:.0f}")
        head = "".join(f"{f'{int(100*k/segments)}-{int(100*(k+1)/segments)}%':>10s}"
                       for k in range(segments))
        print(f"   {'vecka':>6s} {'A':>5s} {'B':>5s} |{head}   (EUR/MWh, tomt→fullt)")
        for w in weeks:
            a, prof = curve(w, z, params, segments)
            b = b_value(w, params.get(z, CurveParams()))
            lam = [anchor.get(z, 0.0) * a * m for m in prof]
            print(f"   {'v' + str(w):>6s} {a:5.2f} {b:5.2f} |"
                  + "".join(f"{v:10.1f}" for v in lam))
        print()


def scale(params: Dict[str, CurveParams], zone: str, **kw) -> Dict[str, CurveParams]:
    """Kopiera parameteruppsättningen med ändrade fält för EN zon (kalibreringen)."""
    out = dict(params)
    out[zone] = replace(params.get(zone, CurveParams()), **kw)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anchor-from", metavar="RUN", default=None,
                    help="Mät λ_bas ur en expansionskörning i stället för DEFAULT_ANCHOR")
    ap.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    args = ap.parse_args()

    anchor = anchor_from_run(args.anchor_from) if args.anchor_from else None
    if anchor:
        print(f"λ_bas mätt ur {args.anchor_from}: "
              + ", ".join(f"{z} {v:.2f}" for z, v in sorted(anchor.items())) + "\n")
    describe(anchor=anchor, segments=args.segments)
