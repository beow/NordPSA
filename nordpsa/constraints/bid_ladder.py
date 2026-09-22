"""Budtrappa: reservoarens uttag i K nivåer med stigande bud."""

import numpy as np
import pandas as pd
import pypsa
import xarray as xr


def hydro_bid_ladder(tiers: int, width_eur_per_mwh: float):
    """extra_functionality-callback som ger reservoarvattenkraften en STIGANDE budkurva.

    ## Vad den lagar

    Reservoaren har ETT `marginal_cost` per tidssteg, alltså budar hela flottan — 52,4 GW
    över de fem hydrozonerna — vid ett och samma pris. Utbudet blir oändligt elastiskt
    just där, och landar restlasten inne i blocket är priset det budet, oavsett vad sol
    och vind gör. Uppmätt i run400_expansion: zonpriset ligger på hydrons bud (±2) i
    65,0 % av timmarna i SE-N, 78,9 % i NO-N och 60,5 % i NO-S, och en tvåpunktsmodell
    (andel pinnad vid ett konstant tal, resten fri) reproducerar den uppmätta
    prisspridningen till ~1 enhet i alla zoner. Hydron står dessutom vid taket 17-38 %
    av timmarna och vid golvet 23-52 % — den går mellan ändlägena när priset passerar
    budet med en epsilon.

    ⭐ MOTIVERINGEN ÄR AGGREGERINGSFELET, inte prisstatistiken. Modellen slår ihop
    hundratals magasin till ETT per zon. En verklig vattenkraftsflotta har spridda
    vattenvärden — olika magasinstorlek, fallhöjd, lokala villkor — och därmed en
    stigande aggregerad budkurva. Ett enda `mc` *är* aggregeringsfelet. Samma mönster
    som `_add_market_staircase` redan använder för den utländska flottan.

    ## Form

    Uttaget delas i K lika nivåer à `p_nom/K` med symmetriska prisavvikelser kring
    dagens bud:

        offset_k = width · ((k + ½)/K − ½),   k = 0 … K−1

    Medelvärdet över nivåerna är exakt 0, så **trappan flyttar inte NIVÅN, bara
    spridningen** — den stör därmed inte λ_bas-kalibreringen eller reservoardriften.

    ⚠️ `width` ÄR INTE BUDSPANNET. Offsetterna tas i nivåernas MITTPUNKTER (1/6, 1/2, 5/6
    av spannet vid K=3), inte i deras kanter, så det realiserade spannet är

        spann = width · (K − 1) / K

    K=3, width=36 → −12 / 0 / +12, alltså spann **24**, inte 36. Samma konvention som
    `_add_market_staircase`:s `offset_profile [0.1667, 0.5, 0.8333]`. Tolkningen är att
    `width` är spridningen i den UNDERLIGGANDE fördelningen av vattenvärden över flottan,
    medan nivåerna bara samplar dess bin-mittpunkter; spannet går mot `width` när K → ∞.
    ⚠️ Följden är att K och width INTE är oberoende: höjer man K blir trappan bredare vid
    oförändrad `width` (K=5, width=36 → spann 28,8). Ändra en i taget.

    ⭐ DEVIATIONSFORM — därför rörs inte `marginal_cost`. PyPSA lägger redan
    `mc·p_dispatch` i objektivet. Callbacken lägger bara till `Σ_k offset_k · d_k`, och
    eftersom `Σ_k d_k = p_dispatch` blir totalen `Σ_k (mc + offset_k)·d_k` — exakt
    trappan, utan dubbelräkning. Det gör den också lägesoberoende: i expansion är
    basbudet kurvans λ(v), i dispatch är det VOM + SOC-dualen, och trappan lägger sig
    kring vilket av dem som gäller.

    Eftersom `offset_k` är VÄXANDE i k fyller LP:t billigaste nivån först av sig självt
    — ingen ordningsvillkor behövs. Konvex styckvis-linjär kostnad, samma trick som
    `hydro_terminal_value` men speglat (där är det ett värde och avtagande, här en
    kostnad och växande).

    ⚠️ Trappan gör INTE λ tillståndsberoende: torr och våt december ger fortfarande
    samma bud. Det är en separat brist (se docs/vattenvarde_kvantilhypotesen.md, kö 2b).

    ⚠️ MEDELVÄRDESBEVARANDE I MARGINALEN, INTE I TOTALKOSTNADEN. Offsetterna summerar
    till noll, så budet är oförändrat när hydron går för FULLT — men vid delvis uttag
    ligger bara de billiga nivåerna inne och den genomsnittliga kostnaden blir lägre.
    Verifierat på en minimalflotta (K=3, bredd 36, p_nom 900): vid uttag 900 MW är
    kostnaden identisk, vid 100/400/700 MW är den 1200/3600/2400 EUR lägre.
      ✓ PRISNIVÅN påverkas inte: uppmätt medeluttag är 43-48 % av p_nom i alla fem
        zoner, vilket ligger i MITTNIVÅN — den med offset exakt 0. Marginalbudet vid
        normal drift är alltså oförändrat, och det är marginalen som sätter priset.
      ⚠️ Men objektivvärdet sjunker, och i EXPANSION kan billigare inframarginell hydro
        i princip tränga undan annan utbyggnad. Prova därför i dispatch (frysta
        kapaciteter) först, och jämför inte råa objektivvärden över trappan.

    ⭐ VÄNTAD SIDOEFFEKT: hydrons uttag PLATTAS UT. Ett platt bud ger bang-bang (uppmätt
    17-38 % av timmarna vid taket, 23-52 % vid golvet); en stigande budkurva gör de
    sista MWh:en dyrare och sprider uttaget. Det går åt samma håll som `min_hourly_frac`
    och `max_weekly_frac` redan drar, och är förenligt med hur verkliga flottor körs.

    ⚠️ width kalibreras mot den OBSERVERADE budkurvan (hydroproduktion mot pris, samma
    observabel som satte `min_hourly_frac`), aldrig mot prisfördelningen — det senare
    vore prisproxyns cirkularitet i ny form.

    tiers:              antal budnivåer K ≥ 2
    width_eur_per_mwh:  trappans totala bredd; nivåerna spänner ±width/2 kring basbudet
    """
    K = int(tiers)
    W = float(width_eur_per_mwh)
    if K < 2:
        raise ValueError(f"hydro_bid_ladder: tiers måste vara ≥ 2, fick {K} "
                         "— K=1 är ingen trappa utan dagens platta bud")
    if W <= 0.0:
        raise ValueError(f"hydro_bid_ladder: width måste vara > 0, fick {W}")

    offsets = [W * ((k + 0.5) / K - 0.5) for k in range(K)]

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        units = [su for su in n.storage_units.index
                 if n.storage_units.at[su, "carrier"] == "hydro"
                 and float(n.storage_units.at[su, "p_nom"]) > 0.0]
        if not units:
            return

        m     = n.model
        p_dis = m.variables["StorageUnit-p_dispatch"]
        names = pd.Index(units, name="name")
        segs  = pd.Index(range(K), name="bid_tier")
        sns   = pd.Index(snapshots, name="snapshot")

        # Nivåtak: p_nom/K per lager och nivå, lika i alla tidssteg.
        cap = np.array([[float(n.storage_units.at[su, "p_nom"]) / K] * K for su in units],
                       dtype=float)
        upper = xr.DataArray(np.broadcast_to(cap[:, :, None], (len(names), K, len(sns))),
                             coords=[names, segs, sns])

        d = m.add_variables(lower=0.0, upper=upper, coords=[names, segs, sns],
                            name="hydro_bid_tier")

        # Σ_k d_k == p_dispatch  (per lager och tidssteg)
        m.add_constraints(
            d.sum("bid_tier") - p_dis.sel(name=names, snapshot=sns) == 0.0,
            name="hydro_bid_tier_def")

        # Objektivtillägg: bara AVVIKELSEN, viktad som PyPSA viktar marginal_cost.
        w   = n.snapshot_weightings.objective.reindex(snapshots).to_numpy()
        coef = xr.DataArray(
            np.broadcast_to(np.array(offsets, dtype=float)[None, :, None]
                            * w[None, None, :], (len(names), K, len(sns))),
            coords=[names, segs, sns])
        m.objective = m.objective + (coef * d).sum()

    return _extra_functionality
