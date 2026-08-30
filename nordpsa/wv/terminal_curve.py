"""Analytisk terminalvärdeskurva λ(fyllnadsgrad, vecka) per zon.

Ersätter den årskonstanta `--terminal-lambda-profile` med en kurva som varierar över
året, i den rullande horisontens fönsterslut. Formen är parametrisk och GENERISK —
den ska bära nordisk hydrologi i allmänhet, inte 2023-25 i synnerhet.

    λ_k(w, z) = λ_bas[z] · A(w, z) · P_k( B(w, z) )

Tre knoppar som är avsiktligt SEPARERADE, så att kalibreringen kan röra en i taget:

  λ_bas[z]   NIVÅ, EUR/MWh. Marginalvärdet på lagrat vatten vid halvfullt magasin.
  A(w, z)    SÄSONG på nivån. Normaliserad till årsmedel EXAKT 1 (cosinus).
  P_k(B)     LUTNING mot fullt magasin, per SOC-segment. Normaliserad så att λ vid
             halvfullt magasin är 1 (`p_norm="mid"`). B styr bara tiltet, aldrig nivån.

Utan normaliseringarna är knopparna sammanblandade: att göra kurvan brantare skulle
samtidigt sänka det genomsnittliga vattenvärdet, och kalibreringen skulle jaga sin
egen svans.

⚠️ RÄTTELSE 2026-08-13: den ursprungliga normaliseringen (`p_norm="mean"`, medel 1 över
SEGMENTEN) separerade INTE nivå och lutning, tvärtemot vad stycket ovan påstod. Medlet
över segmenten är bara vattenvärdet vid driftpunkten om magasinet står LIKFORMIGT över
hela SOC-spannet, vilket det inte gör. Följden: b_mean 4,0 gav mittsegmentet 0,566 och
b_mean 1,5 gav 0,915, så de zonvisa b-värdena smugit in en oavsiktlig nivåskillnad —
norr fick 70-72 % och FI 125 % av det uppmätta systemvattenvärdet (run320: 73,02).
Det förklarar kvantitativt både norrs för tidiga tömning och FI:s hamstring i run318.
`p_norm="mid"` rättar det; "mean" finns kvar bara för att run317-319 ska gå att
reproducera.

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
    """Sju fria tal per zon. Alla dimensionslösa — nivån bor i λ_bas."""
    a_amp:  float = 0.35   # säsongsamplitud på nivån, 0 = ingen säsong. Krav: < 1
    a_peak: float = 5.0    # vecka då nivån toppar
    b_mean: float = 2.5    # genomsnittlig brantid mot fullt magasin. Krav: ≥ 0
    b_amp:  float = 0.6    # säsongsvariation i brantid, andel av b_mean. Krav: |·| ≤ 1
    b_peak: float = 22.0   # vecka då kurvan är brantast (vårflodens spillrisk)
    # ANDRA HARMONISKAN, tillagd 2026-08-17. En ren cosinus är SYMMETRISK: kollapsen
    # vid flodtoppen och återuppbyggnaden genom hösten tvingas ta lika lång tid. Verklig
    # hydrologi är skev — spillrisken slår till snabbt när floden kommer, värdet byggs
    # upp långsamt. Två harmoniska är den minsta periodiska formen som klarar det.
    # 0,0 = av, dvs identiskt med den rena cosinusen (regressionssäkert).
    a_amp2:  float = 0.0   # amplitud på andra harmoniskan (period 26 v)
    a_peak2: float = 0.0   # dess fas, vecka
    # TVÅ LUTNINGAR, tillagd 2026-08-18. `exp(−B·x)` med EN lutning binder ihop kurvans
    # båda ändar: eSett vill ha brant tilt i mittintervallet för att få Norges v/s rätt,
    # och notan betalas vid tomt magasin (388 EUR/MWh vid 10 % fyllnad, SE-S 1037).
    # b_low_frac skalar lutningen UNDER halvfullt magasin separat. 1,0 = av, dvs
    # identiskt med den enkla exponentialen.
    # ⚠️ Den låga änden BESÖKS: driftintervallet är 7-83 % fyllnad (NO-N bottnar på 7,2 %,
    # SE-N 10,2 %). Den höga änden gör det INTE — ingen zon når över 83 %, noll timmar
    # över 95 %, så spill kan strukturellt inte uppstå i en aggregerad reservoar per zon.
    b_low_frac: float = 1.0
    # SÄSONGSREFERENS, tillagd 2026-08-18. Kurvan mätte fyllnaden mot en FAST mittpunkt
    # (0,5), men magasinets naturliga bana svänger 26 -> 85 %. 28 % i april är alltså
    # HELT NORMALT, medan kurvan behandlar det som knappt — och eftersom brantheten
    # B(v) toppar v18 precis när fyllnaden bottnar v13-17 gav det en λ-spik på 4-5x
    # (SE-N 189, NO-S 236) och ett prisårsmaximum i MARS, som mätdata motsäger
    # (2023-25 faller monotont från januari i alla fem zoner).
    # x_ref(v) = c_mid + c_amp·cos(2π(v − c_peak)/52) låter referensen följa banan.
    # ⭐ Poängen är IDENTIFIERBARHET: när banan ligger vid referensen blir λ längs banan
    # exakt λ_bas·A(v), så A(v) betyder "vattenvärdet ett normalår vid normal fyllnad"
    # och B(v) betyder "hur hårt värdet svarar på att ligga FEL". I dag är de två
    # konfunderade — båda verkar längs banan, vilket är varför a_peak och b_peak inte
    # går att tolka var för sig.
    # 0,0 = av, dvs identiskt med den fasta mittpunkten (regressionssäkert).
    c_amp:  float = 0.0    # amplitud på referensfyllnaden. Krav: c_mid ± c_amp i (0, 1)
    c_mid:  float = 0.5    # referensfyllnad vid årsmedel
    c_peak: float = 41.0   # vecka då referensfyllnaden toppar (uppmätt v39-44)
    # ⭐ ANDRA HARMONISKAN på referensbanan. En ren kosinus är SYMMETRISK medan den
    # verkliga magasinbanan har ett KNÄ: fyllningen startar abrupt (FI v17, SE v19) och
    # går nästan rakt upp i fem-sex veckor, och höstplatån är bred och sen. Residualen
    # mot EC:s median 2014-2026 är därför systematisk, ±8-12 pp med teckenbyte just vid
    # knäet. Andra harmoniskan halverar felet:
    #     R²/RMS   SE 0,923→0,980 (5,7→2,9 pp) · NO 0,948→0,984 (3,9→2,2) ·
    #              FI 0,642→0,919 (7,0→3,3 pp)  ← störst där den behövs mest
    # Basen är ortogonal på 52-punktsrutnätet, så c_amp2 = 0 ger BIT-IDENTISKT med den
    # rena kosinusen och inget äldre påverkas — samma egenskap som a_amp2 har.
    # ⭐ EN ENDA SKALA på hela säsongsformen. a_amp och a_amp2 ska ALLTID skalas
    # tillsammans: A(v) − 1 = a_amp·cos(ωv') + a_amp2·cos(2ωv'), så en gemensam faktor k
    # ger A_k(v) − 1 = k·(A(v) − 1) — formen exakt bevarad (verifierat till 2·10⁻¹⁶),
    # bara utslaget krymper. Skalas bara a_amp ändras andra harmoniskans relativa vikt
    # och vintertoppens FORM, vilket gör ett svep till en mätning av två saker.
    # k var en vana fram till 2026-08-20 (run384 skalade båda ×0,5 för hand); nu är den
    # explicit så att formen inte kan rubbas av misstag. a_scale = 1,0 ⇒ oförändrat.
    # ⚠️ Kvoten a_amp2/a_amp (−0,27..−0,54) kommer ur Geminis λ-tabeller och är OPRÖVAD
    # mot varje oberoende observabel — samma sorts opåstådda zonskillnad som b_mean hade.
    a_scale: float = 1.0

    c_amp2:  float = 0.0   # amplitud, andra harmoniska (period 26 veckor)
    c_peak2: float = 0.0   # toppvecka, andra harmoniska
    p_norm: str = "mean"   # vad λ_bas betyder — se segment_profile(). "mid" krävs för
                           # att ankra mot ett mätt vattenvärde

    # ⭐ EMPIRISK referensbana, 52 veckovärden — tar över helt när den finns.
    # Kosinusen ovan är MÄTT fel funktionsfamilj: den är symmetrisk medan den verkliga
    # magasinbanan inte är det (långsam vinteravtappning, brant fall i mars-april, snabb
    # flodfyllning, sen höstplatå). Anpassningen mot EC:s median 2014-2026 ger R² 0,92
    # (SE) / 0,95 (NO) / 0,64 (FI), och residualen är SYSTEMATISK med samma teckenmönster
    # i alla tre länderna: jan-feb +3..+4,5 pp, mars-apr −3..−9,7, sep-okt −3..−5,3.
    # ⛔ Konsekvensen är riktad, inte slumpmässig: i mars-april ligger uppmätt fyllnad
    # UNDER kosinusen, alltså tror kurvan att normalläget är högre än det är, (x − x_ref)
    # blir för negativt och exp(−b·(x−x_ref)) FÖR STORT. Kurvan blåser systematiskt upp λ
    # just den årstid marspuckeln bor i — ~5 % i SE (4 EUR/MWh), 7,5 % i FI.
    # Ingen amplitud eller fas kan laga det; bara en fri bana kan.
    x_ref_weekly: List[float] | None = None

    def __post_init__(self) -> None:
        # ⚠️ Den gamla kontrollen `-1 < a_amp < 1` är BORTTAGEN (2026-08-29). Den skrevs
        # innan `a_scale` fanns och testade fel storhet: det som måste hållas är att
        # A(v) > 0, alltså a_scale·(|a_amp| + |a_amp2|) < 1 — precis vad villkoret nedan
        # prövar, och skarpare. Med a_scale < 1 var den gamla gränsen dessutom onödigt
        # snäv och blockerade den NORMALISERADE formen (a_amp = 1, hela amplituden i
        # a_scale), som är den enda uppdelning där a_scale och a_amp inte är redundanta.
        # Med två harmoniska räcker det inte att pröva termerna var för sig: de kan
        # sammanfalla i samma vecka. |a_amp| + |a_amp2| < 1 är det skarpa villkoret
        # för att A(v) > 0 för ALLA veckor, oavsett faser.
        if self.a_scale < 0.0:
            raise ValueError(f"a_scale måste vara ≥ 0, fick {self.a_scale} "
                             "— negativa värden vänder säsongen upp och ner")
        if self.a_scale * (abs(self.a_amp) + abs(self.a_amp2)) >= 1.0:
            raise ValueError(
                f"a_scale·(|a_amp| + |a_amp2|) måste vara < 1, fick "
                f"{self.a_scale * (abs(self.a_amp) + abs(self.a_amp2)):.3f} — annars kan "
                "nivåfaktorn bli noll eller negativ den vecka där de två harmoniska "
                "sammanfaller")
        if self.b_mean < 0.0:
            raise ValueError(f"b_mean måste vara ≥ 0, fick {self.b_mean}")
        if self.b_low_frac < 0.0:
            raise ValueError(f"b_low_frac måste vara ≥ 0, fick {self.b_low_frac} "
                             "— negativa värden gör kurvan VÄXANDE i fyllnadsgrad")
        if abs(self.b_amp) > 1.0:
            raise ValueError(f"|b_amp| måste vara ≤ 1, fick {self.b_amp} "
                             "— annars blir kurvan VÄXANDE i fyllnadsgrad någon vecka")
        # Skarpt villkor med två harmoniska: de kan sammanfalla i samma vecka, så det
        # räcker inte att pröva termerna var för sig (samma resonemang som a_amp/a_amp2).
        _cspan = abs(self.c_amp) + abs(self.c_amp2)
        if not 0.0 < self.c_mid - _cspan <= self.c_mid + _cspan < 1.0:
            raise ValueError(
                f"c_mid ± (|c_amp| + |c_amp2|) måste ligga i (0, 1), fick {self.c_mid} ± "
                f"{_cspan:.4f} — referensfyllnaden måste vara en fyllnadsgrad")
        if self.x_ref_weekly is not None:
            if len(self.x_ref_weekly) != WEEKS:
                raise ValueError(f"x_ref_weekly måste ha exakt {WEEKS} värden (vecka "
                                 f"1..{WEEKS}), fick {len(self.x_ref_weekly)}")
            if not all(0.0 < float(x) < 1.0 for x in self.x_ref_weekly):
                raise ValueError("x_ref_weekly måste vara fyllnadsgrader i (0, 1)")
        if self.p_norm not in ("mean", "mid"):
            raise ValueError(f"p_norm måste vara 'mean' eller 'mid', fick {self.p_norm!r}")


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
    """Säsongsfaktor på NIVÅN. Årsmedel exakt 1 — båda harmoniska integrerar till noll,
    så andra termen ändrar FORMEN men aldrig årsmedlet, och λ_bas behåller sin
    betydelse."""
    return (1.0
            + p.a_scale * (p.a_amp * math.cos(2.0 * math.pi * (week - p.a_peak) / WEEKS)
                           + p.a_amp2 * math.cos(4.0 * math.pi * (week - p.a_peak2) / WEEKS)))


def b_value(week: int, p: CurveParams) -> float:
    """Brantid mot fullt magasin för veckan. Golvad vid 0 = platt (linjär) kurva."""
    return max(0.0, p.b_mean * (1.0 + p.b_amp
                                * math.cos(2.0 * math.pi * (week - p.b_peak) / WEEKS)))


def x_ref_value(week: int, p: CurveParams) -> float:
    """Referensfyllnaden för veckan — den nivå magasinet NORMALT ligger på. λ_bas är
    vattenvärdet just där, så "mid"-normaliseringen behåller sin betydelse.

    `x_ref_weekly` (52 uppmätta värden) vinner när den finns; annars kosinusen, som
    är kvar för bakåtkompatibilitet och för zoner utan mätdata. Se fältets kommentar
    för varför kosinusen är mätt otillräcklig."""
    if p.x_ref_weekly is not None:
        return float(p.x_ref_weekly[min(WEEKS, max(1, week)) - 1])
    return (p.c_mid
            + p.c_amp * math.cos(2.0 * math.pi * (week - p.c_peak) / WEEKS)
            + p.c_amp2 * math.cos(4.0 * math.pi * (week - p.c_peak2) / WEEKS))


def segment_profile(b: float, segments: int = DEFAULT_SEGMENTS,
                    norm: str = "mean", b_low_frac: float = 1.0,
                    x_ref: float = 0.5) -> List[float]:
    """Icke-växande multiplikatorer tomt→fullt. b = 0 ger platt profil (det gamla
    LINJÄRA terminalvärdet). Normaliseringen bestämmer VAD λ_bas betyder:

      "mean"  medelvärdet över segmenten är 1. ⚠️ Då är λ_bas INTE vattenvärdet vid
              någon fyllnadsgrad man kan peka ut, och eftersom magasinet inte står
              likformigt över SOC-spannet blandas nivå och lutning ihop: vid b=4,0
              är mittsegmentet 0,566 och vid b=1,5 är det 0,915, så samma λ_bas ger
              47 % olika verkligt vattenvärde i normal drift. Kvar som default bara
              för att run317-319 ska gå att reproducera.
      "mid"   λ vid HALVFULLT magasin är exakt 1. Då betyder λ_bas det den utger sig
              för att betyda och kan ankras direkt mot ett mätt vattenvärde
              (run320: 73,02 EUR/MWh). Oberoende av antalet segment.
    """
    if segments < 1:
        raise ValueError(f"segments måste vara ≥ 1, fick {segments}")
    if b_low_frac < 0.0:
        raise ValueError(f"b_low_frac måste vara ≥ 0, fick {b_low_frac}")
    xs = [(k + 0.5) / segments for k in range(segments)]
    # Skrivet kring x = 0,5 i stället för kring x = 0: exp(−B(x−0,5)) är IDENTISKT med
    # den gamla exp(−B·x)/exp(−B·0,5), men gör de två lutningarna möjliga och gör
    # "mid"-normaliseringen trivial (P(0,5) = 1 per konstruktion). "mean" påverkas inte:
    # både vals och ref skalas med samma faktor exp(0,5·B), som förkortas bort.
    vals = [math.exp(-(b if x >= x_ref else b * b_low_frac) * (x - x_ref)) for x in xs]
    if norm == "mean":
        ref = sum(vals) / len(vals)
    elif norm == "mid":
        ref = 1.0
    else:
        raise ValueError(f"norm måste vara 'mean' eller 'mid', fick {norm!r}")
    return [v / ref for v in vals]


def curve(week: int, zone: str,
          params: Dict[str, CurveParams] | None = None,
          segments: int = DEFAULT_SEGMENTS) -> tuple[float, List[float]]:
    """(nivåfaktor, segmentprofil) för zonen och veckan."""
    p = (params or DEFAULTS).get(zone, CurveParams())
    return (a_factor(week, p),
            segment_profile(b_value(week, p), segments, p.p_norm, p.b_low_frac,
                            x_ref_value(week, p)))


# ── Kurvan som hydrons marginal_cost i EXPANSION ────────────────────────────────

def hydro_mc_from_curve(snapshots, zones: Iterable[str],
                        params: Dict[str, CurveParams] | None = None,
                        anchor: Dict[str, float] | None = None) -> Dict[str, "object"]:
    """λ längs NORMALBANAN per snapshot och zon — avsedd som reservoarens marginal_cost.

    Ersätter vattenvärdes-proxyn (zonens faktiska historiska pris), som gör en
    2040-expansion cirkulär: hydrons dispatch ankras till den prisform modellen ska
    förutsäga. Se `_add_hydro` i nordpsa/network.py.

    ⭐ Varför just normalbanan: `marginal_cost` är EXOGEN per tidssteg, medan λ(v, x) beror
    på fyllnadsgraden — en beslutsvariabel. `mc = λ(v, SOC)` vore bilinjärt och inte längre
    ett LP. Med p_norm="mid" gäller per konstruktion P(x_ref(v)) = 1, alltså

        λ(v, x_ref(v)) = λ_bas[z] · A(v, z)

    som är en ren funktion av veckan. Exakt, inte approximativt — och till skillnad från
    proxyn kommer A(v) ur Geminis zontabeller och EC:s magasindata, inte ur prisserien.

    ⚠️ Vad man byter bort: proxyns tim-till-tim-variation. Den var till hälften önskvärd
    (2023 års vindprognoser ska inte styra hydrons veckoprofil i 2040) och till hälften
    nyttig (en HELT platt hydro-mc gav ett degenererat LP som OOM-dödades i 1h). Den här
    serien är inte platt — den svänger ±26 % över året — men den är slät.
    ⚠️ Prisgolvet försvinner också: proxyn gick ned till 0,6 i de billigaste timmarna,
    den här bottnar kring 50, så hydron blir aldrig marginalsättare i lågprislägen.
    """
    import pandas as pd

    params = params or DEFAULTS
    anchor = anchor or DEFAULT_ANCHOR
    weeks = [week_of(ts) for ts in snapshots]
    out = {}
    for z in zones:
        p = params.get(z)
        if p is None:                      # zon utan kurva → ingen serie, proxyn faller bort
            continue
        lam = [float(anchor.get(z, 0.0)) * a_factor(w, p) for w in weeks]
        out[z] = pd.Series(lam, index=snapshots, name=z)
    return out


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
    out = {}
    for u in units:
        p = params.get(_zone_of(u), CurveParams())
        out[u] = segment_profile(b_value(week, p), segments, p.p_norm, p.b_low_frac,
                                 x_ref_value(week, p))
    return out


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
#: ⚠️ Pekar på den KALIBRERADE kurvan (run334/337, låst 2026-08-14), inte på
#: config/terminal_curve.yaml som är den okalibrerade STARTPUNKTEN från 001f87c
#: (a_peak 5, ankare 76-84, p_norm "mean"). Ett bart --terminal-curve hämtade förut
#: startpunkten och gav tyst en helt annan kurva än den som är verifierad.
# ⭐ LÅST 2026-08-18: Geminis fem zontabeller (run375_aamp06_3h). Den förra,
# terminal_curve_2040_calibrated.yaml, var kalibrerad mot FACIT (run320) och ligger kvar
# för att reproducera run316-run368 — namnge den explicit med --terminal-curve.
# ⭐⭐⭐ v8, 2026-08-25 (run429_bm20_3h): = v7 med `b_mean` 2,0 likformigt i stället för
# 0,80. Grunden är den FÖRSTA icke-cirkulära mätningen av B — vattenvärdets svar på
# magasinets avvikelse från normalbanan — mot EC-magasin + ENTSO-E-priser 2015-2025.
# Modellen låg 2-12× för platt. Se filens egen `note:` för svepet run428-432.
# v7 ligger kvar för att reproducera run316-run432 — namnge den då explicit.
# ⭐ v9, 2026-08-29: = v8 med `b_amp` 0. Mätt i en 2x2 (b_mean {2,4} x b_amp {0,27, 0})
# att säsongsvariationen i brantheten är nära inert — nivån gör 5-43x mer — och att
# budkurvan till och med blir BÄTTRE utan den. Tar bort två oprövade tal: `b_amp` och
# `b_peak`, den senare kurvans sista zonskillnad i b.
# ⭐ v10 (2026-08-30) = v9 med MEDIANBANOR PER ZON (x_ref-familjen), enda skillnaden.
# EC:s median finns bara per land, sa SE-N/SE-S och NO-N/NO-S delade bana. Billigt i
# Norge (zonerna ar hydrologiskt nastan lika) men dyrt i Sverige: SE-S ar regnmatat och
# SE-N snosmaltningsmatat, faktor 5,6 i tillrinningens sasong. SE-S beskrevs med 14,6 pp
# RMS-fel mot sin uppmatta bana; med egen bana 3,3. Kalla: ENTSO-E A72 per budzon (SE),
# NVE Magasinstatistikk per prisomrade (NO), 2015-2025. Nivan bevarad => ren formandring.
# ⚠️ Reproducera run420-438 med config/terminal_curve_2040_gemini_v9.yaml explicit.
DEFAULT_PARAM_FILE = "config/terminal_curve_2040_gemini_v10.yaml"


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
        # float() på varje tal: numpy-skalärer (np.float64) går inte att serialisera med
        # yaml.safe_dump, och en anpassning med scipy/numpy levererar just sådana.
        "zones": {z: {"a_scale": float(c.a_scale),
                      "a_amp": float(c.a_amp), "a_peak": float(c.a_peak),
                      "b_mean": float(c.b_mean), "b_amp": float(c.b_amp),
                      "b_peak": float(c.b_peak), "a_amp2": float(c.a_amp2),
                      "a_peak2": float(c.a_peak2), "b_low_frac": float(c.b_low_frac),
                      "c_amp": float(c.c_amp), "c_mid": float(c.c_mid),
                      "c_peak": float(c.c_peak), "c_amp2": float(c.c_amp2),
                      "c_peak2": float(c.c_peak2), "p_norm": str(c.p_norm),
                      **({"x_ref_weekly": [float(x) for x in c.x_ref_weekly]}
                         if c.x_ref_weekly is not None else {})}
                  for z, c in sorted(params.items())},
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return str(p)


def load_params(path: str | None = None
                ) -> tuple[Dict[str, CurveParams], Dict[str, float]]:
    """Läs (params, anchor). Saknas den IMPLICITA standardfilen returneras
    DEFAULTS/DEFAULT_ANCHOR; en UTTRYCKLIGT namngiven fil som saknas är ett fel.

    ⚠️ Tyst fallback på ett namngivet `path` är run319-fällan: anroparen tror sig köra
    sin kandidatkurva och kör i själva verket modulens startvärden — utan ett ord i
    konsolen. Uppmätt 2026-08-18: `curve_turns.py` på en felstavad sökväg skrev ut
    rubriken med filnamnet men siffrorna för DEFAULTS."""
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    p = Path(path) if path else root / DEFAULT_PARAM_FILE
    if not p.exists():
        if path:
            raise SystemExit(f"terminalkurva saknas: {p} — ingen tyst fallback på en "
                             "namngiven fil (kontrollera sökvägen)")
        return dict(DEFAULTS), dict(DEFAULT_ANCHOR)
    doc = yaml.safe_load(p.read_text()) or {}
    # `a_scale` får stå på toppnivån och gäller då ALLA zoner — det är den avsedda
    # användningen (en global skala på säsongsformen). Ett värde inne i en zon vinner.
    gscale = doc.get("a_scale")
    params = {}
    for z, kw in (doc.get("zones") or {}).items():
        kw = dict(kw)
        if gscale is not None:
            kw.setdefault("a_scale", float(gscale))
        params[z] = CurveParams(**kw)
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
              f"b_mean={p.b_mean:.2f} b_amp={p.b_amp:.2f} b_peak=v{p.b_peak:.0f} "
              f"norm={p.p_norm}")
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
