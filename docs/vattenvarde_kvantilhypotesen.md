# Kvantilhypotesen för vattenvärdet — placering mot vad NordPSA redan vet

*2026-08-21. Analysdokument, ingen kod ändrad. Alla siffror är beräknade read-only ur
`data/processed/market_prices.parquet`, `docs/NordicHydroEC.xlsx`, `config/zones.yaml` och
`results/run400_*`; inga skript committade.*

Ett externt resonemang om vattenvärde och terminalfunktion har delats. Det innehåller **tre
separerbara påståenden** som måste bedömas var för sig — de har helt olika status i det här
repot.

---

## Sammanfattning (verdikten först)

| # | Påstående | Verdikt |
|---|---|---|
| **A** | Trängsel *blockerar* arbitraget; vattenvärdet dras inte upp till kontinentpriset | **Redan sant per konstruktion.** Inget att göra. |
| **B** | Förväntad framtida trängsel ger en permanent **nivårabatt** i vattenvärdet | **Mekanismen är VERKLIG och nu uppmätt** (selektionseffekt, −17 till +29 EUR/MWh). Men `CLAUDE.md`:s "dubbelräknings"-verdikt **STÅR** — på ett nytt och starkare argument. ⚠️ Med en **villkorlighet ingen tidigare spårat**. |
| **C** | VV ≈ **kvantil**, inte medel, med α = (E_magasin+E[tillrinning])/(P·H) | **Genuint nytt och det mest värdefulla.** Härleder det v7 anpassar för hand. Billigaste testet finns redan i koden. |

⭐⭐⭐ **Huvudfyndet:** den platta λ:n och 2040-scenariots NTC-antaganden är **kopplade
storheter**, och det har ingen hållit reda på. Modellens λ är platt därför att 2040-scenariot
bygger bort exakt de flaskhalsar som skapar den observerade nivåspridningen. Håller inte
utbyggnaden, återkommer rabatten — och då är platt λ **fel**. Se §3.

⛔⛔⛔ **Allvarligaste fyndet (§3d): kurvan slätar ut prisbildningen.** SE-S P95 138,7 → 94,2 och
h>100/år 1471 → 262 (observerat 999); NO-N:s SD 24,3 → **9,1** mot observerade 24,0. Orsak:
`hydro_mc_from_curve` gör λ till **en funktion av kalenderveckan ensam** — fyllnadsgraden går
inte in, så det finns **ingen knapphetsrespons** och 52 GW hydro blir ett elastiskt pristak.
Dispatch fixar det inte. ⭐ Det är exakt den defekt kvantilformen (§4) åtgärdar.

⛔ **Sensitivitetstestet (run401) kunde INTE avgöra saken** — `--hydro-mc-curve` pinnar hydrons
bud vid 73 och förbjuder mekanismen i expansion. Villkorligheten är **fortfarande obesvarad**
och kräver en dispatch-körning. Testet mätte i stället, skarpt, vad mc-kurvan kostar:
**NO-N q25 = 68,9 i modellen mot 8,0 observerat.** Se §3b.

---

## §1 Påstående A — redan sant, inget att göra

> "När ledningen söderut är full kan vattenkraften inte nå den dyrare marknaden:
> `pris_SE = vattenvärde < pris_kontinent`, och gapet är trängselräntan."

Detta är LP:ts nodbalansdual, exakt. `notebooks/cells/marginal.py` rapporterar redan samma sak
i klartext: *"Hydroöar har vattenvärdet (WV) som GOLV men prisas högre när de är
export-/importträngda."* Ingen åtgärd.

---

## §2 Påstående B — mekanismen är verklig, och nu uppmätt

Påståendet är att en zon som *kroniskt förväntas* vara exportbegränsad får en **permanent
nivårabatt** i vattenvärdet — inte bara en timvis rabatt när flaskhalsen binder.

### Selektionstestet

Om trängseln vore rent timvis skulle en zons kopplade timmar vara ett *representativt* urval
av grannens prisfördelning. Är de i stället systematiskt **billiga** finns selektionen — och
då kan zonens vatten aldrig komma åt grannens höga priser.

Mått: `E[grannens pris | kopplad] − E[grannens pris | alla timmar]`, observerat 2023–25
(26 304 h, kopplad = |Δp| < 0,01 EUR/MWh):

| billig zon ← dyr granne | NTC | kopplad % | **selektion** | nivåskillnad |
|---|---|---|---|---|
| SE-S ← DK | 2000 | 10,2 | **−16,9** | 31,5 |
| SE-N ← FI | 1550 | 39,8 | **−16,7** | 20,4 |
| SE-N ← SE-S | 7300 | 36,9 | **−14,1** | 20,3 |
| NO-S ← DK | 1681 | 4,1 | **−12,1** | 20,9 |
| SE-S ← FI | 1200 | 31,7 | −8,1 | 0,0 |
| NO-N ← NO-S | 700 | 3,3 | −3,0 | 32,3 |
| NO-N ← SE-N | 1150 | 26,4 | +2,2 | 1,4 |
| SE-S ← NO-S | 1900 | 7,0 | +11,1 | 10,6 |

Sett från den **dyra** zonens sida är tecknet spegelvänt och ännu tydligare: när NO-S är
kopplad till NO-N ligger NO-N:s pris på **55,0** mot sitt eget årsmedel **25,8** (+29,3) — de
kopplar ihop nästan enbart i NO-N:s dyraste timmar.

⇒ **Selektionen finns, den är stor och den är systematisk.** SE-N:s vatten möter inte SE-S:s
prisfördelning (medel 47,5) utan SE-S:s fördelning *betingad på att vara billig* (medel 33,4).
Det är en strukturell nivårabatt på ~14 EUR/MWh, inte en timvis händelse. Påstående B:s
**mekanism är därmed empiriskt bekräftad.**

### Observerade nivåer, för sammanhang

| zon | medel | median | q25 | q75 | std | <0 % |
|---|---|---|---|---|---|---|
| NO-N | 25,8 | 20,9 | 8,0 | 34,4 | 24,0 | 2,1 |
| SE-N | 27,1 | 16,8 | 4,2 | 39,9 | 31,4 | 6,2 |
| SE-S | 47,5 | 37,6 | 17,0 | 67,3 | 43,4 | 5,1 |
| FI | 47,5 | 29,8 | 5,2 | 72,6 | 62,4 | 6,3 |
| NO-S | 58,1 | 54,5 | 36,7 | 72,0 | 34,0 | 1,3 |
| DK | 79,0 | 80,9 | 47,1 | 105,7 | 50,1 | 3,4 |
| **DE-LU** | **87,7** | 90,1 | 65,1 | 113,0 | 51,3 | 5,1 |

NO-N och NO-S är två närmast separata vattensystem — kopplade **3,3 %** av timmarna, med
**32,3 EUR/MWh** i nivåskillnad. I hydrodominerade zoner *är* zonpriset i stort sett
vattenvärdet, så detta är i praktiken en uppmätt vattenvärdesspridning i dagens system.

---

## §3 ⭐⭐⭐ Varför "dubbelräkning"-verdiktet ändå STÅR — och den villkorlighet som saknats

`CLAUDE.md` avfärdar Geminis zonvisa nivåer med:

> *"Nivåerna speglar observerade zonpriser, dvs. trängsel — som modellen redan producerar
> endogent ur NTC:erna. Att lägga in dem i λ_bas vore dubbelräkning, och facit (run320) ger
> 73,02 i alla fem zoner."*

Argumentets **andra halva var svag**: run320 är en modellkörning, och kan bara bekräfta
modellens interna konsistens — aldrig avgöra om verkligheten har en spridning. Dessutom är
fullframsyn **strukturellt partisk mot platthet**: λ-utjämning kräver bara att det finns
*någon* oträngd väg vid *någon* tidpunkt, och med 3 års perfekt framsyn har LP:t maximal
frihet att hitta den. En spridning som är verklig under *begränsad* framsyn försvinner i facit
per konstruktion.

⚠️ Verifierat i förbifarten: run320 (`001f87c`, 2026-08-13) kördes med NO-N-export
**1836 + 1773 = 3609 MW** och *ingen* 2040-override för NO-N. Dagens config ger NO-N
**5000 MW** i 2040-scenariot men bara **1850 MW** i nutidskalibreringen. Facit är alltså
betingat på en NTC-årgång som inte finns i någondera riktningen längre.

### Men samma selektionstest på modellen avgör saken

Kört på `results/run400_dispatch/prices.csv` med 2040-scenariots NTC:

| billig ← dyr | NTC 2040 | kopplad % | selektion | nivåskillnad |
|---|---|---|---|---|
| SE-N ← SE-S | **10700** | **98,8** | **+0,0** | **0,0** |
| NO-S ← NO-N | **3000** | **82,4** | **+0,4** | **0,9** |
| SE-N ← NO-N | 2000 | 63,5 | +0,4 | 2,5 |
| SE-S ← NO-S | 2300 | 60,7 | −0,6 | 1,5 |
| SE-N ← FI | 1550 | 50,3 | −10,4 | 8,8 |
| SE-S ← FI | 1200 | 50,2 | −10,3 | 8,8 |
| SE-S ← DK | 2300 | 26,3 | −10,0 | 7,5 |
| NO-S ← DK | 1681 | 21,5 | −6,9 | 6,0 |

**På de interna nordiska länkarna kollapsar selektionen till noll.** NO-N↔NO-S går från
3,3 % kopplade timmar och 32,3 EUR i nivåskillnad (observerat, 700 MW) till 82,4 % och
0,9 EUR (modell, 3000 MW). Snitt 2 är i praktiken upplöst: 98,8 % kopplat vid 10 700 MW.

Och modellens egna vattenvärden i samma körning är följdriktigt **nästan platta**:

| lager | medel | median | min | max |
|---|---|---|---|---|
| SE-N hydro | 73,48 | 70,75 | 58,08 | 97,15 |
| SE-S hydro | 74,43 | 73,07 | 61,30 | 95,49 |
| NO-N hydro | 73,42 | 71,10 | 58,20 | 97,15 |
| NO-S hydro | 73,01 | 70,53 | 57,44 | 94,61 |
| FI hydro | 76,96 | 73,58 | 56,01 | 106,97 |

⇒ **Verdiktet står, men skälet byts ut.** Inte *"modellen säger att λ är platt"* (cirkulärt),
utan:

> Den observerade nivåspridningen orsakas av en **mätbar selektionseffekt**. 2040-scenariot
> bygger per konstruktion bort exakt de flaskhalsar som skapar den (Snitt 2 → 10 700,
> NO-N↔NO-S → 3000, NO-N↔SE-N → 2000). Modellens eget selektionstest bekräftar att effekten
> då är borta. Att importera dagens nivåer till 2040 vore att **dubbelräkna en flaskhals som
> scenariot redan avlägsnat**.

Geminis nivåer är alltså inte *fel* — de är **rätt svar på fel år**.

### ⚠️ Villkorligheten som ingen spårat

Platt λ är **inte** ett fritt antagande. Det är en **konsekvens** av 2040-NTC-antagandena. Och
`CLAUDE.md` flaggar själv dessa som obelagda:

> *"NO-N↔NO-S 3000: ⚠️ ANTAGANDE utan källa i repot — 4,3× nutidens 700 … Största relativa
> nätutbyggnaden i scenariot"* · *"NO-N↔SE-N 2000: ⚠️ ANTAGANDE utan källa i repot"*

**Håller inte den nordliga utbyggnaden återkommer selektionen — och då är platt λ fel.**
De två antagandena måste stå och falla tillsammans; idag är de dokumenterade på olika ställen
utan koppling.

⭐ **run401 var tänkt som sensitivitetskörningen** — run400:s konfiguration med run360:s NTC,
alltså NO-N-export **2200 MW** (1500 + 700) mot run400:s **5000 MW**, en 2,3× nedskalning av
exakt den korridor saken hänger på. **Den kördes, och testet kunde inte fira — se §3b.**
Villkorligheten står därmed kvar som obesvarad och behöver en *dispatch*-körning.

---

## §3b run401-utfallet — testet KUNDE INTE FIRA, och skälet är viktigare än testet

`run401_run360ntc` blev klar 2026-08-21 (Optimal, 16 879 s, målfunktion **8,0290e10** mot
run400:s **8,0019e10** — samma hydro-mc-kurva i båda, så jämförelsen är äpplen-mot-äpplen:
den tightare nordliga NTC:n kostar **+271 M€**, +0,34 %).

**Kopplingen kollapsade precis som förutsagt:**

| länk | run400 (NO-N-export 5000) | run401 (2200) |
|---|---|---|
| NO-S ← NO-N | 3000 MW, **71,1 %** kopplat, selektion 0,1, nivådiff 0,9 | 700 MW, **19,5 %**, selektion 0,0, nivådiff 3,8 |
| SE-N ← NO-N | 2000 MW, 55,6 %, +0,4, 2,5 | 1500 MW, **21,9 %**, +0,7, 5,3 |

⛔ **Men selektionseffekten uteblev** (0,1 → 0,0), och **NO-N:s pris STEG** 71,1 → 74,2 — motsatt
riktning mot vad nivårabatt-teorin förutsäger. Kapacitetsanpassning förklarar det inte: NO-N
byggde *mer* sol (0,87 → 1,67 GW).

**Orsaken är strukturell och gäller hela `--hydro-mc-curve`-vägen.** I expansion är hydrons
`marginal_cost` **exogen och likformig** (`λ_bas·A(v)` = 73·A(v), `hydro_mc_from_curve`).
En isolerad hydrozon prissätts därför *vid sitt eget bud* ≈ 73 — den kan aldrig kollapsa till
ett överskottspris. Att strypa NO-N:s export gjorde alltså fördelningen **tightare och högre**:

| NO-N | q05 | q25 | median | medel |
|---|---|---|---|---|
| observerat 2023-25 | **1,4** | **8,0** | 20,9 | 25,8 |
| run400 (5000 MW) | 47,2 | 66,3 | 71,7 | 71,1 |
| run401 (2200 MW) | **63,1** | **68,9** | 73,0 | 74,2 |

⇒ **run401 är inkonklusiv för påstående B** — mekanismen är förbjuden per konstruktion i den
här körtypen. Den mäter i stället, skarpt, priset för `--hydro-mc-curve`. `CLAUDE.md` varnade
kvalitativt (*"hydron blir aldrig marginalsättare i lågprislägen … väntas synas i
prisvaraktighetskurvans nedre ände"*); här är siffran: **NO-N q25 68,9 mot observerade 8,0.**

⇒ **Villkorligheten i §3 är alltså varken bekräftad eller falsifierad.** Det giltiga testet
måste vara en **dispatch**-körning, där λ är den endogena SOC-dualen och hydron får buda fritt
— inte en expansion med pinnat bud. Kön i §8 uppdaterad.

⚠️ Notera också att λ-avläsningen ur `water_value.csv` är **oanvändbar i båda körningarna**:
med mc-kurvan kollapsar dualen (run400 0,29–4,42 · run401 0,64–5,55 EUR/MWh) precis som
`project_hydro_mc_curve` beskriver. Ännu ett skäl att testet hör hemma i dispatch.

---

⛔ **Ankarets ordning duger inte som bevis åt något håll.** v7:s ankare (SE-S 64,5 < SE-N 66,0
< NO-S 69,6 < NO-N 70,2 < FI 72,5) sätter NO-N *nästan högst* medan teorin vill ha NO-N lägst
(Gemini: 39, lägst av alla). Men ankaret är enligt `CLAUDE.md` en **driftkorrigering för
begränsad framsyn**, inte ett vattenvärdespåstående. Det kan inte användas som evidens utan att
först dekomponeras i "vad kurvan är värd" och "vad den rullande horisonten driver fel".

---

## §3c Vattenvärdesmodellen isolerad — ren tre-vägs-dekomposition

⛔⛔ **DET FINNS INGEN REN ISOLERING BLAND BEFINTLIGA KÖRNINGAR.** Två separata confounds:

1. `run401` mot `run360`: SE:s RoR-filer regenererades 2026-08-19 (efter run360) och FI fick
   `hydro_ror_regulated_frac` ⇒ bär även run382:s **RoR-omformning**.
2. `run377` mot `run360`: run377 kördes med ett **bart** `--hydro-mc-curve`, vilket vid dess
   commit (729c3c1, 2026-08-18) löstes till `terminal_curve_2040_gemini.yaml` — **inte** dagens
   v7. Ankaret är visserligen 73,0 likformigt i båda, men **A(v):s säsongsamplitud är exakt
   2,00× större**, uppmätt ur `storage_units_t.marginal_cost`:

| kurva | mc medel | min | max | **span** |
|---|---|---|---|---|
| run377 (gemini v1) SE-N | 73,05 | 52,37 | 99,60 | **47,23** |
| run401 (v7, a_scale 0,30) SE-N | 73,02 | 62,69 | 86,30 | **23,62** |

Kvoten är **2,00× i SE-N, NO-N och FI** — det är `a_amp` ×0,6 → ×0,3 (run384).

| | NTC | RoR | hydro-mc | kurvans A(v)-amplitud |
|---|---|---|---|---|
| run360 | run360 | gammal | **prisproxy** | — |
| run377 | run360 | gammal | kurva | **2×** |
| run401 | run360 | **ny (α=0,80)** | kurva | 1× (v7) |

⇒ `run377 − run360` = proxy → kurva, **men med en kurva vars amplitud är 2× dagens**.
⇒ `run401 − run377` = **amplitud halverad + RoR-omformning** (två ändringar, inte en).

⚠️ Eftersom `a_amp` är *den* uppmätta spaken för v/s (`d(v/s)/d(a_amp) ≈ −1,7`; run384 flyttade
SE 1,18 → 1,32 i dispatch) **går v/s-effekten nedan inte att tillskriva vattenvärdesmodellen**.
Tecknet stämmer med a_scale-hypotesen: run377 (2× amplitud) har lägst v/s i SE (0,88), run401
(1×) högst (1,08) — precis som en halverad amplitud förutsäger, oberoende av RoR.

⭐ **Det rena testet finns inte och måste köras:** `run402` = run401:s exakta konfiguration med
`--no-hydro-mc-curve`. Då skiljer *bara* prisproxy mot v7-kurvan — samma NTC, samma RoR, samma
kod. Det är den enda jämförelsen som isolerar den **nuvarande** vattenvärdesmodellen.

### Real kostnad (rå målfunktion är obrukbar)

| | rå objective | hydro-mc-term | **REAL** | vs run360 |
|---|---|---|---|---|
| run360 | 8,550e10 | 2,177e10 | 6,373e10 | — |
| run377 | 1,004e11 | 3,918e10 | 6,123e10 | **−3,9 %** |
| run401 | 9,978e10 | 3,914e10 | 6,064e10 | −4,9 % |

Rå målfunktion **stiger 17 %** medan verklig kostnad **faller 3,9 %** — hydrons skuggbud nästan
fördubblas (2,18 → 3,92e10) och är ingen verklig kostnad. −3,9 % reproducerar exakt `CLAUDE.md`:s
rapporterade tal för run377, vilket validerar metoden.

⚠️ Men läs stegen rätt givet confounds ovan: **−3,9 % = proxy → kurva med 2× amplitud**, och
**−1,0 % = amplitud halverad + RoR**. Kostnadseffekten är dock robust i *riktning* och
storleksordning: båda kurvvarianterna ligger 3,9–4,9 % under proxyn, så slutsatsen "kurvan är
billigare i verklig kostnad" bär oavsett hur de två sista stegen delas upp.

### ⭐ Budet är oförändrat — bara dekompositionen byts ut

| zon | run360: mc + λ = **bud** | run401: mc + λ = **bud** |
|---|---|---|
| SE-N | 27,3 + 44,1 = **71,4** | 73,0 + 1,6 = **74,7** |
| SE-S | 47,6 + 27,4 = **75,0** | 73,0 + 2,0 = **75,0** |
| NO-N | 25,8 + 46,7 = **72,5** | 73,0 + 1,6 = **74,6** |
| NO-S | 58,1 + 16,0 = **74,1** | 73,0 + 0,6 = **73,7** |
| FI | 47,9 + 33,0 = **80,9** | 73,0 + 5,6 = **78,6** |

Med proxyn bär `mc` zonens historiska pris (25–58, mycket spretigt) och λ bär resten; med kurvan
är `mc` = 73 likformigt och λ kollapsar till ~0–6. **Summan skiljer ≤ 3,3 EUR/MWh i alla fem
zoner.** Det är `project_hydro_mc_curve`s *"dualen kollapsar, budet oförändrat"* — nu kvantifierat
per zon. ⇒ Modellerna är nästan ekvivalenta i **nivå**; skillnaden ligger i budets **tidsform**
(proxyns tim-till-tim-variation vs kurvans släta λ_bas·A(v)) — och i att kurvan inte är cirkulär.

### ⛔ Priset: säsongskvoten blir SÄMRE

| land | eSett | run360 | run377 | run401 | mc-kurva | RoR |
|---|---|---|---|---|---|---|
| SE | 1,38 | 1,04 | 0,88 | 1,08 | **−0,15** | +0,20 |
| NO | 1,30 | 1,29 | 1,42 | 1,54 | **+0,13** | +0,11 |
| FI | 1,13 | 0,70 | 0,61 | 0,83 | **−0,10** | +0,23 |
| **Σ\|fel\|** | | **0,78** | **1,15** | **0,84** | | |

⛔⛔ **Dessa siffror kan INTE tillskrivas vattenvärdesmodellen** — se amplitud-confounden ovan.
Mönstret (run377 sämst i SE och FI, run401 tillbaka) är exakt vad `a_amp` ×2 → ×1 förutsäger på
egen hand, och `a_amp` är den uppmätta v/s-spaken. Vad som faktiskt går att säga:

- run401 (v7-kurva + ny RoR) ligger **0,84** mot run360:s **0,78** — dvs. dagens
  konfiguration är *marginellt sämre* på säsongskvoten än den gamla proxyvärlden.
- Om den skillnaden beror på kurvan, på RoR, eller tar ut sig mellan dem **går inte att avgöra
  utan run402**.

⚠️ Detta är dessutom *expansions*-v/s (cykliskt SOC, exogent hydrobud) — en annan storhet än
dispatchens v/s som kurvan kalibrerades mot; jämför inte mot CLAUDE.md:s dispatch-tal.

### Kapacitet och pris

Byggt (GW), med samma confound-reservation på stegnamnen: `run377−run360` **+2,2**
(nästan allt **havsvind +2,21**, 3,16 → 5,37, dvs +70 %; gas −0,37); `run401−run377` **+3,5**
(nästan allt **sol +3,64**). Zonpris: steg 1 **+3,2 SE-N, +3,2 FI, −2,0 NO-S**; steg 2
**−2,9 i SE-N/SE-S/FI**, **+0,3/+0,5 i Norge**.

⚠️ Norge kan **inte** användas som kontrollgrupp här. Det vore giltigt om steg 2 vore ren RoR
(Norge orört i config och data), men steg 2 innehåller även amplitudhalveringen, som träffar
**alla** zoner. Att Norge rör sig lite är därför inte bevis för en ren dekomposition.

✓ Vattenkraftsproduktionen är **identisk** i alla tre (SE 67,9 · NO 142,0 · FI 13,9 TWh/år) —
cykliskt SOC ⇒ årsproduktion = tillrinning per konstruktion. Vattenvärdesmodellen kan bara
flytta vattnet i **tiden**, aldrig ändra totalen.

---

## §3d ⛔⛔⛔ Kurvan slätar ut prisbildningen — och det är SAMMA defekt kvantilformen fixar

Uppmärksammat av användaren på SE-S december 2024 (SD 58,7 → 17,7). **Verifierat, och det är
inte en december-artefakt.**

### Symptomet, helår SE-S

| fall | medel | P5 | P50 | **P95** | P99 | **SD** | h>100/år |
|---|---|---|---|---|---|---|---|
| run360 (proxy) | 70,5 | 12,5 | 63,0 | **138,7** | 206,0 | **54,2** | **1471** |
| run401 (mc-kurva) | 69,0 | 7,9 | 71,6 | **94,2** | 133,2 | **34,2** | **262** |
| *observerat 2023-25* | *47,5* | *−0,0* | *37,6* | *129,0* | *184,9* | *43,4* | *999* |

Per zon (SD): NO-N 24,3 → **9,1** (observerat 24,0) · NO-S 33,0 → 16,2 (obs 34,0) ·
SE-S 54,2 → 34,2 (obs 43,4). ⭐ **DK ändras knappt** (44,5 → 41,8) — DK har **ingen vattenkraft**
och är därmed en ren kontrollgrupp: utslätningen är hydro-medierad.

### Mekanismen, uppmätt

Andel timmar där zonpriset ligger **på** hydrons bud (mc + λ/eff, ±2 EUR/MWh):

| zon | run360 | run401 |
|---|---|---|
| SE-S | 11,7 % | **47,0 %** |
| SE-N | 38,4 % | **60,6 %** |
| NO-N | 90,9 % | 93,1 % |
| FI | 13,6 % | 22,5 % |

⇒ Med kurvan sätter hydron priset i **fyra gånger så många timmar** i SE-S.

### ⭐⭐⭐ Rotorsaken: λ är en funktion av KALENDERVECKAN ENSAM

`hydro_mc_from_curve` (`terminal_curve.py:291`) utvärderar λ **på referensbanan**: med
`p_norm="mid"` är `P(x_ref(v)) = 1`, så `λ(v) = λ_bas · A(v)` — *"a pure function of the week"*
enligt dess egen docstring. **Fyllnadsgraden går inte in.** En torr december och en våt december
ger hydron exakt samma bud.

Med 52 GW nordisk vattenkraft som budar ett slätt 73–86 blir hydron därmed ett **nästan
oändligt elastiskt pristak**. Det är därför P95 kollapsar: priset kan inte stiga över hydrons
bud förrän hydron är *effektbegränsad*, och det är den sällan.

⚠️ Det är en **medveten LP-kompromiss, inte ett förbiseende**: `mc = λ(v, SOC)` vore bilinjärt
och inte längre ett LP (samma docstring).

⛔ **Dispatch löser det INTE**, trots att λ där är SOC-beroende via segmenten:

| | SE-S SD | P95 | h>100/år | h<0/år |
|---|---|---|---|---|
| run401 expansion | 34,2 | 94,2 | 262 | 0 |
| run400 **dispatch** | 33,5 | 96,0 | 208 | **260** |
| observerat | 43,4 | 129,0 | 999 | 449 |

Dispatch återställer **nedre** änden (negativa timmar 0 → 260, tack vare
`--vre-curtailment-cost 5`) men den **övre kroppen är lika utslätad**.

### ⚠️ Men proxyns bättre prisform är INTE bevis för att proxyn är en bättre modell

run360:s P95 138,7 ligger nära observerade 129,0 — men **den likheten är cirkulär**: proxyn
sätter hydrons bud till zonens *faktiska historiska pris*, så modellen återger den prisform den
matats med. Det var hela skälet att avskaffa den. Att kurvan har sämre prisform är alltså ett
äkta problem, men proxyn är inte lösningen.

### ⭐ Det är exakt den defekt påstående C åtgärdar

Kvantilformen gör λ till en funktion av **α**, som beror på *faktisk* magasinnivå och
tillrinning. Då stiger vattenvärdet när magasinet är lågt relativt säsongen — precis den
knapphetsrespons som saknas idag. **Användarens observation är alltså symptomet på den defekt
§4 föreslår en kur för**, och är det starkaste enskilda argumentet i det här dokumentet för att
faktiskt testa kvantilformen.

Inom LP finns två vägar: (a) **fixpunktsiteration** — kör, läs den realiserade SOC-banan,
utvärdera λ längs *den* i stället för längs referensbanan, kör om; (b) använd
**segment-maskineriet** (`segment_profile`, som redan ÄR SOC-beroende) även på expansionsvägen
i stället för bara i rullande horisont.

---

## §4 Påstående C — kvantilformen är den värdefulla delen

> `VV ≈ Q_(1−α)({P_τ}, τ∈[t,t+H])`, `α = (E_magasin + E[tillrinning])/(P·H)`

Vattenkraft med begränsat magasin är energibegränsad: den säljer inte till periodens
*medelpris* utan i de dyraste timmarna energin räcker till.

### Varför det är intressant just här

Formen **härleder** det v7 anpassar för hand:

- `x_ref(v)` — två harmoniska anpassade mot EC-medianen — är en handbyggd proxy för exakt vad
  α räknar fysiskt: *"är den här fyllnaden knapp för årstiden?"* α gör det ur energi, effekt
  och tillrinning, **utan fria parametrar**.
- Den konvexa `VV(fyllnad)` faller ut automatiskt i stället för att kalibreras — hela
  `b_mean`/`b_amp`/`b_low_frac`/`p_norm`/`segment_profile`-apparaten.
- Det stämmer med repots egen erfarenhet: 25 → 3 frihetsgrader kostade **0,01** i v/s och blev
  *bättre* mot facit (run388 → run390). Zonvisa formparametrar bar praktiskt taget ingen
  information — precis vad man väntar sig om parametriseringen är överspecificerad och
  underhärledd.

### α är redan beräknat i configen

Före tillrinning är `α = max_hours / H` **exakt** — magasinets energi/effekt-kvot *är*
`hydro_max_hours`. Med H = 4368 h (halvår):

| zon | P (GW) | max_hours | E (TWh) | **α** | obs. pris |
|---|---|---|---|---|---|
| SE-S | 2,9 | 1207 | 3,50 | **0,28** | 47,5 |
| FI | 3,2 | 1719 | 5,50 | **0,39** | 47,5 |
| SE-N | 13,3 | 2180 | 28,99 | **0,50** | 27,1 |
| NO-S | 22,0 | 2609 | 57,40 | **0,60** | 58,1 |
| NO-N | 11,0 | 2727 | 30,00 | **0,62** | 25,8 |

Låg α → hög kvantil → högt VV. Ordningen håller i **fyra av fem** zoner (SE-S/FI dyra,
SE-N/NO-N billiga). **Undantaget är NO-S** — hög α men högst pris — och undantaget är precis
vad teorin förutsäger: NO-S bär alla kontinentkablarna (DE 1400 + NL 640 + GB 1449), så dess
tillgängliga prisfördelning *är* kontinentens. Det är själva skälet till att kvantilen ska tas
över **zonens egen** framtida prisserie (kontinent minus förväntad trängsel), inte över
kontinentpriset rakt av.

### ⭐ Den degenererade förfadern finns redan i koden

`scripts/run_model.py:771` `terminal_lambdas()` bygger λ som `ahead.mean()` över
look-ahead-fönstret av DE-LU — **exakt det medelvärde argumentet säger ska bytas mot en
kvantil.** Bytet är en enradsändring (`run_model.py:833/849`), och
`--terminal-lambda-scale` / `--terminal-seasonal` ger färdig A/B-rigg mot v7.
Det är det billigaste möjliga testet av hela idén.

### ⭐ Strukturell bonus: säsongen ur kalendern i stället för ur en cosinus

Låter man H löpa **till nästa vårflod** blir H tidsvarierande och säsongsformen faller ut ur
kalenderavståndet till floden i stället för ur en anpassad cosinus (`a_peak`). Det är en
kandidatförklaring till det kända **~5-veckors fasfelet** (v49 mot facits v0,6) som
spill-A/B:t (run346) falsifierade som spilldrivet och som ingen annan hypotes hittills
förklarat.

### ⛔⛔ TEST 0 UTFÖRT (2026-08-21): kvantilen slår INTE ett framåtblickande medel

Hypotesen är en **intertemporal no-arbitrage-relation** och går att pröva utan modellkörning:
vattenvärdet nu ska vara Q_(1−α) av den prisfördelning vattnet kommer att möta. Hästkapplöpning
på uppmätt data 2023-25 (EC-magasin + NVE-tillrinning + observerade zonpriser, landvis,
empiriskt λ ≈ veckomedianpris, n≈146 veckor):

| α-definition | SE RMSE | NO RMSE | mot fwd-**medel** | mot **konstant** |
|---|---|---|---|---|
| (a) `(E_magasin + tillrinning)/(P·H)` — **som pitchad** | — | **32,2** | 21,0 | 22,3 |
| (b) disponibelt vs säsongsmål, H=13 v | **18,2** | **18,1** | 19,8 / 18,1 | 22,8 / 22,0 |
| (c) H till säsongsbotten (v16) | 19,2 | 17,6 | 19,6 / 17,1 | 23,0 / 22,3 |

**Två resultat, båda viktiga:**

⛔ **1. α som pitchad är fel.** `(E_magasin + tillrinning)/(P·H)` behandlar hela magasinet som
förbrukningsbart inom fönstret. Det töms aldrig. α hamnar på **0,58–0,93** och kvantilen får en
bias på **−19,6 EUR/MWh**. Fixen är att räkna på *disponibelt* vatten, dvs. magasin **minus ett
säsongsmål** — och det säsongsmålet **är x_ref**. ⇒ **Kvantilformen eliminerar inte x_ref, den
flyttar in den i α.** Påståendet i pitchen att fyllnadsberoendet blir parameterfritt är
**falskt**; variant (c) mildrar det (behöver bara flodens *tidpunkt*, v16, inte hela banan) men
tar inte bort det.

⛔ **2. Kvantilen är inte skiljbar från ett enkelt framåtmedel.** Båda slår en konstant tydligt
(RMSE ~23 → 17–20, ca 25 % bättre), men inbördes ligger de inom bruset (±1 RMSE på n=146; i NO
vinner medelvärdet marginellt). **Det som bär information är att λ är TILLSTÅNDSBEROENDE och
FRAMÅTBLICKANDE — inte att den är just en kvantil.**

⇒ **Konsekvens för modellen:** dagens `hydro_mc_from_curve` är i praktiken den *konstanta*
hästen (kalenderberoende men tillståndsoberoende, §3d) — den som förlorar med 25 %. Fixen som
data stöder är alltså **att göra λ tillståndsberoende alls**, inte att finlira kvantilindex.
Det gör experimentet både billigare och mer träffsäkert än pitchen antyder.

⚠️ Förbehåll: bara 3 års prisdata · empiriskt λ approximerat med veckomedianpris ·
NVE-tillrinningen har många nollveckor (klippta negativa ΔSOC, NO underskattas till 113 mot
~137 TWh/år) · EC-nivåer finns bara per land, så SE-N/SE-S och NO-N/NO-S delar bana.

### ⚠️ Justeringar NordPSA:s egna restriktioner kräver

- α ska räknas på **disponibelt** vatten, inte rått `p_nom·H`: `min_hourly_frac` 0,05 och
  `min_daily_frac` 0,20 är må-köra-vatten, och `max_weekly_frac` (0,77–0,87) kapar effekten.
  `hydro_operation_bounds()` beräknar redan kvoten `inflöde/(p_nom·H)` per zon (0,46–0,55 för
  2024) — det *är* α:s tillrinningsdel.
- Spillregimen (α ≥ 1) ska hanteras som **eget fall**, inte som extremvärde av α. Modellen
  spiller dock 0,0000 TWh i alla körningar sedan run316, så fallet aktiveras i praktiken aldrig
  — och `CLAUDE.md` noterar att modellen **underspiller** mot verkligheten (0,12–0,22 %).

---

## §5 Torrår/våtår — Geminis egen falsifierbara prediktion är UNDERMÄKTIG på denna data

> *"i torrår ska hydrozonens pris spåra den övre delen av kontinentens prisfördelning, i
> våtår den undre."*

Testat: årsmedelpris per zon uttryckt som percentil av DE-LU:s prisfördelning, mot EC:s
magasinavvikelse från 2015–22 års normal:

| år | SE fyll% | NO fyll% | vs normal | SE-N | SE-S | NO-N | NO-S | FI |
|---|---|---|---|---|---|---|---|---|
| 2023 | 57,6 | 60,3 | −2,6 % | 15 | 19 | 14 | 30 | 20 |
| 2024 | 60,5 | 62,3 | −0,3 % | 12 | 15 | 13 | 17 | 17 |
| 2025 | 74,0 | 67,1 | **+18,4 %** | 11 | 18 | 11 | 20 | 15 |

**Riktningen stämmer i de rena vattenzonerna** — SE-N (15→12→11), NO-N (14→13→11) och FI
(20→17→15) faller monotont med stigande fyllnad, precis som förutsagt. SE-S och NO-S gör det
inte, vilket är rimligt: de har termisk respektive kontinental prissättning.

⛔ **Men testet är undermäktigt och avgör ingenting.** Alla nordiska zoner ligger i DE-LU:s
**nedersta 11–30 percentil i samtliga år**; hydrologin flyttar dem några percentilenheter medan
den strukturella klyftan är ~60 EUR/MWh. Perioden saknar dessutom ett riktigt torrår (värst
−2,6 %). Gemini förutsåg själv utfallet: *"Ser du det inte, dominerar troligen trängseln över
magasinsdynamiken i just din period."* **Trängseln dominerar.** Ett skarpt test kräver
2018/2021 års torrår, och prisserien går bara tillbaka till 2023.

---

## §6 Vad repot redan vet som dämpar förväntningarna

- ⛔ **RoR-bevarandelagen (run382):** reservoaren kompenserade bort **2/3** av en omformad
  strömkraft. *"Systemet — last, kurva, NTC — sätter kvoten, inte vilken enhet som levererar."*
  Förvänta kraftig dämpning av **varje** kurvändring.
- ⛔ **run389:** `x_ref_weekly` (52 fria värden) beskrev EC-banan exakt men gav en **sämre**
  modell. **Ett sanningsenligare indata garanterar inte en bättre modell** — gäller lika för en
  "härledd" kvantilkurva.
- ⛔ **FI är inte ett giltigt testfall:** orörligt över fyra spakar; 2,34 GW är för lite för att
  vara prissättande i egen zon.
- ⚠️ **Mätbasen:** `run_vs()` måste köras med `include_ror=True` mot eSett, medan
  `facit_score()` är pinnad till `include_ror=False`. Talen är inte på samma basis.
- ⚠️ `nordpsa/wv/targets.py` egen varning: EC-bandet är **beskrivande, inte normativt** — ett
  räcke, inte en måltavla.
- ⚠️ Känslighetens tvåpunktsskattning var **4× för låg** för SE-N. Dämpa och taka alltid.

---

## §7 Cirkularitet — hederlig gradering

Kvantilen ska tas över zonens **egen** framtida prisserie, som beror på vattenvärdet. Det är en
**fixpunkt, inte en prognos**. Tre grader måste hållas isär:

| grad | konstruktion | omdöme |
|---|---|---|
| **Prisproxyn** (avskaffad 2026-08-20) | hydrons bud = zonens *observerade historiska* pris | Helt självrefererande, ingen extern information |
| **Kvantil seedad på DE-LU** | kontinentpriset är exogent för den nordiska modellen | **Mindre** cirkulärt — inte icke-cirkulärt |
| **Fixpunktsiteration** | kör → extrahera prisfördelning → räkna om → kör igen | Precedens finns: `project_hydro_seasonal_bid`, *"fixpunkt nådd"* |

---

## §8 Förregistrerad kö — billigast först, med falsifieringskriterier

1. ~~run401-avläsningen~~ — **KÖRD, INKONKLUSIV** (§3b). Kopplingen kollapsade som förutsagt
   (71 → 19,5 %) men selektionen uteblev, eftersom `--hydro-mc-curve` pinnar hydrons bud vid 73
   och förbjuder mekanismen. **Ersätts av 1b.**
1b. **Dispatch av run401 mot dispatch av run400** — `--dispatch run401_run360ntc` respektive
   `--dispatch run400_expansion`. I dispatch är λ den endogena SOC-dualen och hydron budar
   fritt, så mekanismen *kan* fira. Detta är nu det avgörande testet av villkorligheten i §3.
   → **λ separerar mellan zonerna vid 2200 MW men inte vid 5000** ⇒ villkorligheten bekräftad;
   platt λ måste redovisas som *betingat* på 2040-utbyggnaden.
   → **λ platt i båda** ⇒ nivårabatten biter inte i 2040-prislandskapet, och verdiktet står
   utan förbehåll.
   ⚠️ Kapaciteterna skiljer sig mellan de två källkörningarna (NO-S sol −3,3 GW, NO-N sol
   +0,8 GW), så det är **inte** ett rent NTC-A/B. Ett renare alternativ är två dispatcher av
   *samma* flotta med olika NTC.
2. ~~Kvantil-mot-medel~~ — **AVFÖRD av test 0** (§4): kvantil och framåtmedel är inte skiljbara
   på uppmätt data. Att byta `ahead.mean()` → `ahead.quantile()` vore att finlira fel spak.
2b. **Gör λ TILLSTÅNDSBEROENDE på expansionsvägen** — det är vad test 0 faktiskt stöder
   (RMSE ~23 → 17–20 mot dagens tillståndsoberoende kurva). Två LP-kompatibla vägar:
   (a) **fixpunkt** — kör, läs den realiserade SOC-banan, utvärdera λ längs *den* i stället för
   längs referensbanan, kör om tills banan är stabil; (b) använd **`segment_profile`** (redan
   SOC-beroende, används i rullande horisont) även i expansion.
   → Faller om prisspridningen inte återhämtar sig: mät **SE-S P95** (94,2 → mot run360:s 138,7)
   och **NO-N SD** (9,1 → mot observerade 24,0). Rör de sig inte är hela spåret dött.
3. **Fullframsynt λ vid tight nordlig NTC** — facit-motsvarighet i dagens värld
   (`--no-expansion`, no-proxy). Enda sättet att mäta λ-spridning i den värld vi har
   *observerade priser* för, alltså det enda som kan valideras utifrån.
4. **Empirisk λ-serie per hydrozon** ur observerade priser (timmar där hydron rimligen är
   marginell och zonen inte är prisseparerad). Rollen — måltavla eller kontroll — beslutas
   först när **täckningsgraden** är känd; den ska rapporteras som eget resultat, inte döljas.

---

## §9 Vad som INTE ska göras

- ⛔ **Inte** importera Geminis zonvisa λ_bas-nivåer. De är rätt svar på fel år (§3).
- ⛔ **Inte** röra den låsta v7-kurvan, ankaret, `min_hourly_frac` eller `min_daily_frac`.
  Ingenting i det här dokumentet motiverar det.
- ⛔ **Inte** använda ankarets ordning som evidens för eller emot nivårabatten.
- ⛔ **Inte** behandla run320:s platta 73,02 som ett självständigt bevis. Det är ett
  modellresultat, strukturellt partiskt mot platthet, och betingat på en NTC-årgång som inte
  längre används.

---

## Bilaga: dokumentationsfel upptäckta i förbifarten (ej åtgärdade)

- `CLAUDE.md:102` och `:111` refererar `data/processed/market_price.parquet` (singular).
  **Den filen har aldrig existerat** (`git log --all` tomt). Rätt väg är kolumnen `DE-LU` i
  `market_prices.parquet` (26 304 h × 12 kolumner, UTC).
- `CLAUDE.md:270` och `config/terminal_curve_2040_gemini_v7_exp73.yaml:191`s `note_expansion`
  anger `--terminal-anchor` till `run_model.py:915`; faktisk rad är **925**
  (`if args.terminal_anchor:`). Samma not anger `--hydro-mc-curve` till rad 2461; faktisk rad
  är **2504** (`_tc.load_params(...)`).
- `--rolling-weeks` (`run_model.py:1466`) har `default=1` men hjälptexten säger
  *"(default 4)"* — verifierat.
