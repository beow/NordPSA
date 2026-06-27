# Sex framtider för 2040 — vad batch 17 säger om det nordiska kraftsystemet

*En genomgång av baseline och fem känslighetsvarianter i NordPSA, alla körda mot SvK:s Långsiktiga marknadsanalys 2040 (MM-scenariot).*

## Utgångspunkten: ett system som domineras av vind och sol

Basfallet (**run170**) bygger på SvK:s efterfrågebild för 2040: en total konsumtion på ~668 TWh/år, varav 567 TWh "vanlig" last, 52 TWh till vätgaselektrolys, 44 TWh till elbilsladdning och drygt 2 TWh till fjärrvärmens värmepumpar och elpannor. Modellen får sedan fritt bygga ut produktion och lagring för att möta det — billigast möjligt.

Resultatet är ett system där **landvind (78 GW) och sol (79 GW) är arbetshästarna**. Tillsammans levererar de drygt 300 TWh/år — nära halva produktionen. Vattenkraften ligger kvar på sina 52 GW / 224 TWh, nu mest som balanserande reglerkraft. Kärnkraften landar på 11 GW endogent, havsvind byggs nästan inte alls (2,7 GW, enbart i Danmark) och gasen är en ren topplastreserv på 3 GW.

| Kraftslag | Effekt (GW) | Produktion (TWh/år) |
|---|---|---|
| Vattenkraft | 52,4 | 224 |
| Landvind | 77,8 | 221 |
| Sol | 78,8 | 83 |
| Kärnkraft | 11,3 | 83 |
| KVV-el | 5,6 | 27 |
| Termisk | 3,6 | 10 |
| Gas | 3,1 | 9,7 |
| Havsvind | 2,7 | 11 |
| Batteri | 12,0 | (lagring) |

Priserna hamnar i ett **förvånansvärt smalt band: 70–77 €/MWh** över alla sex zoner, med Finland högst (77) och norra Norge lägst (70). Norden är i det närmaste självförsörjande mot kontinenten — en svag nettoimport på 2,8 TWh.

En avgörande detalj döljer sig i flödesfliken: **de interna förbindelserna är hårt belastade.** Strömmen norr–söder från norra Sverige (SE-N→SE-S, 24,5 TWh netto) går genom flaskhalsar som är bindande 87–96 % av tiden — NO-N↔NO-S 94 %, NO-S↔SE-S 93 %, SE-S↔FI 96 %. Undantaget är den feta SE-N↔SE-S-stommen (10,7 GW) som bara binder 10 %. Marknadskablarna mot kontinenten ligger kring 50 %. **Det är överföringen, inte produktionen, som sätter prisstrukturen mellan zonerna.**

## Variant 1 — Lite exogen kärnkraft (run171)

*Vad händer om vi politiskt bestämmer 1,5 GW ny kärnkraft i SE-S?*

Den nya kärnkraften (totalt 12,8 GW) **tränger undan sol, inte gas** — solen faller 5,3 GW. Svenska priser sjunker ~7 €/MWh (SE-N 64, SE-S 65), men effekten dämpas norrut och når knappt Danmark och Finland. Slutsatsen: en blygsam mängd baskraft flyttar priset i sin egen prisregion men byter inte ut systemets topplast.

## Variant 2 — Billig finansiering av svensk kärnkraft (run172)

*Vad händer om svensk kärnkraft och havsvind får 3 % kalkylränta i stället för 6 %?*

Detta är **batchens enskilt största prisspak.** Med halverad kapitalkostnad bygger modellen 4,2 GW kärnkraft endogent i SE-S (totalt 15,5 GW), solen kollapsar med nästan 15 GW och **de svenska priserna faller hela 15 €/MWh — ned till 56–57**, precis in i LMA2026:s MM-intervall. Norra Norge följer med ned 5 €.

Två saker är värda att notera. För det första: **havsvinden byggs fortfarande inte** trots samma räntelättnad — billig kärnkraft slår den. För det andra vänder Norden nu till nettoexportör (+3,2 TWh). Det är billig finansiering av fast baskraft, inte havsvindsstöd, som är den stora svenska spaken.

## Variant 3 — Strypt landvindspotential (run173)

*Vad händer om landvindstaket sänks 20 %?*

Här ser vi vad som händer när arbetshästen tar slut. Landvinden faller 14,5 GW (taket binder) och ersätts av en blandning: **havsvinden tredubblas till 8 GW — det är först här den byggs på allvar**, sol +5, kärnkraft +1. Priserna stiger 7–8 € i Sverige och Norge, Finland +6 (upp till 83). Norden blir en tydligare nettoimportör (−9 TWh mot kontinenten). Lärdomen: **havsvind är marginell tills landvinden är utbyggd till sitt tak** — då, och först då, blir den nästa billigaste alternativ.

## Variant 4 — Torrår (run174)

*Vad händer ett rekordtorrt år (hydro ×0,6)?*

Vattenkraftsproduktionen faller 33 TWh (224→191). Priseffekten är **starkt koncentrerad till norr**: NO-N +11 € (upp till 81), Sverige +6, medan Danmark och Finland knappt rör sig — flaskhalsarna isolerar södern från norra knappheten. Investeringssvaret är intressant: modellen bygger **+5,3 GW havsvind, inte mer kärnkraft**, plus lite gas. Spillet stiger. Ett torrår löses alltså med mer väderberoende kraft snarare än baskraft — men priseffekten i norr är reell och stor.

*(Caveat: faktor 0,6 ger ~44 TWh svensk vattenkraft, vilket är ~7 TWh under det historiska rekordet 1996. För ett realistiskt rekordtorrår är ~0,70 lämpligare — 0,6 är ett stresstest, inte en prognos.)*

## Variant 5 — Mer och längre batteri (run175)

*Vad händer med 25 GW / 4-timmars batterier i stället för 12 GW / 2 h?*

Längre lagringsvaraktighet **låser upp mer sol** (+5,9 GW, upp till 85 GW) genom att absorbera middagstopparna och flytta dem till kvällen. Gasen faller 1,3 GW, spillet blir batchens lägsta (2,3 TWh) och **priserna sjunker överallt, 1–3 €/MWh**. Batterierna cyklar mer (nettouttag 0,9→2,5 TWh). Det är den renaste "win-win"-varianten: mer sol, lägre pris, mindre gas — till priset av mer lagringsinvestering.

## Sammanfattning: fyra robusta mönster

Tvärs över alla varianter framträder ett par tydliga regler för hur 2040-systemet beter sig:

1. **Landvind + sol är ryggraden.** De byggs alltid till sina tak; allt annat är marginaljustering ovanpå.
2. **Havsvind byggs över baseline-nivån endast när landvinden stryps (v3) eller vattenkraften torkar (v4).** Den är systemets andrahandsval, inte förstahandsval.
3. **Kärnkraft byggs endogent bara vid billig finansiering (v2).** Vid normal ränta är den marginell, och den tränger undan sol — inte gas.
4. **Priset sätts av överföringen lika mycket som av mixen.** De interna flaskhalsarna (87–96 % bindande) gör att chocker — torrår, ny baskraft — får starkt lokala priseffekter snarare än att jämnas ut över Norden.

Den största prisspaken av alla fem är **kapitalkostnaden för svensk baskraft** (v2: −15 €/MWh), följt av **vattentillgången i norr** (v4: +11 € i NO-N). Batterivaraktighet och lagringsmängd (v5) är den billigaste vägen till lägre priser utan att röra produktionsmixen i grunden.

---

*Alla sex körningar löstes till optimum med noll lastbortkoppling (slack = 0); systemet är alltså försörjningssäkert i samtliga scenarier — skillnaderna ligger i kostnad, mix och prisnivå, inte i leveranssäkerhet.*

---

*Datakälla: `docs/batch17_mastersheet.xlsx` (flikarna Master + NTC). Körningar run170–175, 3h-upplösning, kod committad i `bdd16c0`. Genererad ur NordPSA.*
