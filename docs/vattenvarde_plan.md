# Vattenvärdesmodellen — läge, lärdomar och vägval

Status 2026-08-07. Underlag för att ta ett större grepp. Ingen av slutsatserna här är
implementerad i baselinen; run260 kör oförändrat vidare.

Experimentkoden ligger på grenen **`rolling-horizon-watervalue`** (commit `fff72b5`),
medvetet utanför `main` så att `run_model`s aktiva väg förblir ren.

---

## 1. Problemet

Vattenkraftens bud i baselinen är

```
bud[t] = historiskt zonpris[t]  +  mu_soc / efficiency_dispatch
         └── "proxy", run23 ──┘     └── endogent vattenvärde, run57 ──┘
```

De två har legat **staplade** sedan run57 utan att någon avsett det. Två följder:

- **Det endogena vattenvärdet är i praktiken konstant** — 1–6 unika värden per zon över
  tre år, SE-S exakt ett. All tidsvariation i budet kommer alltså från den observerade
  prisserien, ingen från modellen.
- **Proxyn inverterar dispatchsignalen.** Korrelationen mellan hydro-dispatch och pris är
  −0,12 med proxyn och +0,47 utan (run254). Proxyn lägger priset som en *kostnad*, så en
  dyr timme ger hög marginalkostnad och modellen producerar mindre just då.

I en 2040-expansion är detta cirkulärt: den prisform modellen ska förutsäga matas in som
indata. Men att bara ta bort proxyn duger inte — då kollapsar priset på det konstanta
vattenvärdet i 65–85 % av timmarna. **Båda lägena är fel på olika sätt.**

---

## 2. Vad rullande horisont visade

Metoden: sekventiella fönster, icke-cyklisk SOC, carry-over, och ett terminalvärde
−V(SOC[T]) som belönar kvarvarande vatten.

### Det som fungerade

**Det platta vattenvärdet försvinner.** 54–72 unika värden per zon över tre år, std 31–39,
mot ETT värde per zon och år i cyklisk drift. Det är den strukturella defekten, och den
är löst.

**Magasinbanan kan fås att matcha verkligheten.** Validerat mot Energy Charts
(`docs/NordicHydroEC.xlsx`, veckovis magasinenergi per land, 2014–2026):

| | aggregerat offset | korrelation |
|---|---|---|
| run268 globalt linjärt λ | +18,4 TWh | 0,980 |
| run269 per-zons konkav | +12,2 TWh | 0,978 |
| **run272 bred kurva + λ-skala** | **+2,9 TWh** | **0,982** |

Per land i run272: SE −0,5 · NO +2,9 · FI +0,5 TWh. Offsetet är 2,3 % av magasinvolymen.

### Det som inte fungerade

**Priserna löstes inte.** run272: SE-N +6,7, NO-N +10,2, NO-S −14,0, FI −8,4.
Prisgapet norr–söder blir 4,8 €/MWh mot verkliga 20,4 — och den klyftan är sedan tidigare
känd som **helt NTC-driven** (snitt 2: 4100 MW → spread 25, 8000 MW → spread 5,5).

⚠️ **Det betyder att vattenvärdet inte längre är den bindande begränsningen för
prisnoggrannheten.** Ytterligare arbete på V(SOC) kan inte stänga ett gap som orsakas av
överföringskapacitet.

---

## 3. Designlärdomar om terminalvärdet

Fem saker som kostade en körning var att lära sig, och som vilken framtida konstruktion
som helst måste respektera:

1. **Linjärt terminalvärde är bang-bang.** `−λ·SOC[T]` har konstant derivata → håll allt
   tills taket nås, därefter värde noll. Gav magasin i taket 13–29 % av timmarna,
   vattenvärde som föll 82 → 3 vid taket, och zonpriser som kollapsade till VOM (0,5–1,6)
   i hela veckor. Konkavitet är inte en förfining utan ett krav.

2. **Spännvidden är viktigare än lutningen.** 10× spann (2,0→0,2) ger marginalvärden nära
   tomt magasin på ~2× medelpriset — för lite för att försvara botten. SE-N tömdes till
   9 % och NO-N till 0 % i februari 2024 → **infeasible**. 80× spann (8,0→0,1) löser det.
   Ett verkligt vattenvärde nära tomt magasin är hundratals €/MWh.

3. **λ måste variera över året.** Ett konstant λ per zon spricker oavsett nivå — vattnets
   alternativkostnad är hög före vårfloden och låg efter. run271 blev infeasible av just
   detta.

4. **Nivån måste vara per zon.** DE-LU som ankare ger SE-N (eget pris 27) samma
   vattenvärdesnivå som DK (79). En trängselinlåst zon kan inte värdera sitt vatten till
   kontinentens pris.

5. **Begynnelsevillkoret tvättas bort.** Terminalkurvan är en attraktor; en startavvikelse
   halveras på ~10 veckor. Att sätta faktisk start-SOC förbättrar de första ~8 veckorna
   (spårar verkligheten på 1–2 TWh) men ändrar inte årsbanan.

Sammanfattat är den funktionella formen

```
λ_k(zon, t) = lambda_scale[zon] · DE-LU_framåt(t) · profile[k]
              └─ NIVÅ per zon ─┘  └─ TIDSFORM ──┘  └ FORM vs fyllnad ┘
```

och alla tre delarna är nödvändiga.

---

## 4. Krav på en slutlösning

1. **Måste fungera i 2040-expansionen.** Där finns inga observerade priser att kalibrera
   mot. Detta diskvalificerar allt prisanpassat som *slutlösning* — inklusive `lambda_scale`
   i grenens konfig, som är α = zonens medelpris / DE-LU:s.
2. **Får inte vara cirkulär mot valideringsmålet.** Notera att vi numera har *två*
   oberoende observabler: priser och magasinnivåer. Kalibrera mot den ena, validera mot
   den andra är legitimt — det var så run272 kunde bedömas.
3. **Måste ge tidsvarierande vattenvärde.** Annars är vi tillbaka i utgångsläget.
4. **Måste bära expansionen.** Rullande horisont gör det INTE: varje fönster är en egen LP,
   så extendable kapaciteter skulle optimeras oberoende per fönster mot fönstrets eget
   väder. Detta är den allvarligaste begränsningen och är olöst.

---

## 5. Vägval

| väg | löser platt WV | icke-cirkulär | bär expansion | insats |
|---|---|---|---|---|
| **A** status quo (proxy) | nej | nej | ja | noll |
| **B** rullande + kalibrerad kurva (grenen) | ja | nej | nej | klar |
| **C** rullande + endogen α ur μ_ntc | ja | ja | nej | medel, fixpunkt |
| **D** SDDP | ja | ja | oklart | stor |
| **E** överlappande fönster | ja | delvis | nej | liten, OTESTAD |
| **F** tvåpass: cyklisk expansion → frys → rullande dispatch | för priser | ärver A | delvis | liten |

**E är otestad och billigast.** Dagens fönster är icke-överlappande, så varje fönster ser
**noll** framåt och terminalvärdet måste ensamt bära hela säsongssignalen. Äkta receding
horizon löser ett längre fönster och behåller bara första delen; då gör look-ahead det
mesta av jobbet och terminalvärdet blir en detalj. Om känsligheten för terminalkalibreringen
faller kraftigt krymper cirkularitetsproblemet i samma takt. **Detta bör testas först.**

**C** är den principiellt rena varianten av det vi redan byggt: α = 1 − förväntad
trängselränta / DE-LU, hämtad ur modellens eget μ_ntc. Det är ett fixpunktsproblem (α → pris
→ μ_ntc → α) och kräver ett par iterationer.

**D (SDDP)** är läroboksvaret och skulle härleda allt vi handtrimmat — konkav styckvis
linjär V(SOC), tidsvarierande, ur tillrinningsscenarier. Ligger på hyllan sedan 2026-06-13.
Värt att bygga först om E och C inte räcker.

---

## 6. Den fråga som bör ställas före allt annat

**Beror expansionsbesluten faktiskt på att vattenvärdet är rätt?**

Argument för att de inte gör det: under cyklisk SOC är hydroenergin tillrinningsbunden
oavsett vattenvärde — årsproduktionen är given. Kapacitetsvalen styrs av residuallastens
form och bristtimmarna, inte av hur vattnet prissätts internt. run254 (utan proxy) flyttade
sol +5,35 GW och gas −0,44 GW mot baselinen, alltså märkbart men inte omvälvande.

Om expansionen är robust mot vattenvärdet, då är hela detta problem ett **prisbildnings-
problem, inte ett investeringsproblem** — och då är väg F (cyklisk expansion + rullande
dispatch för priserna) tillräcklig, vilket sparar SDDP-bygget helt.

Det går att testa direkt: kör samma expansion med kraftigt olika vattenvärdeslägen och
jämför `p_nom_opt`. Det är en billig körning och den kan omprioritera hela spåret.

⚠️ Innan mer arbete läggs på V(SOC) bör också NTC-frågan avgöras — prisgapet norr–söder är
4,8 mot verkliga 20,4, och den differensen dominerar prisfelet oavsett vattenvärde.

---

## Referenser

Runs: run254 (utan proxy, 2h) · run255/256 (dagens värld, med/utan proxy) ·
run268 (linjärt λ) · run269 (per-zons konkav) · run270 (faktisk start-SOC) ·
run271 (konstant λ, infeasible) · run272 (bred kurva + λ-skala).

Data: `docs/NordicHydroEC.xlsx` (Energy Charts, veckovis magasinenergi TWh per land;
gitignorerad som alla .xlsx). Kapacitetskontroll: SE når 94,5 %, NO 95,3 %, FI 82,0 % av
modellens zonvolymer över elva år — nominella volymer är alltså rimliga.
