"""Det som skiljer lägena åt: frysning av kapaciteter, VRE-avräkningskostnad och
hydrons marginalkostnad i expansion."""
from __future__ import annotations

import pypsa

from nordpsa.settings import RESULTS_DIR, ROOT

VRE_CARRIERS = ("wind_onshore", "wind_offshore", "solar")


def hydro_mc_override(snapshots, cfg: dict, curve_path: str) -> dict:
    """EXPANSION: reservoarens marginal_cost = terminalkurvan längs normalbanan,
    λ_bas·A(v). Ger hydrons bud en slät säsongsform utan att ankra den i historiska
    priser (den tidigare prisproxyn gjorde 2040-expansionen cirkulär)."""
    from nordpsa.wv import terminal_curve as tc
    cp, ca = tc.load_params(str(ROOT / curve_path))
    mc = tc.hydro_mc_from_curve(snapshots, cfg["zones"], cp, ca)
    print("  hydro-mc ur terminalkurvan (lambda_bas*A(v)):")
    for z, s in sorted(mc.items()):
        print(f"    {z:6s} {s.min():6.1f} - {s.max():6.1f}  medel {s.mean():6.1f} EUR/MWh")
    return mc


def freeze_capacities_from(n, label):
    """Sätt p_nom = p_nom_opt (och extendable=False) på alla komponenter (gen/lager/länkar)
    som matchar namn i results/LABEL/network.nc → fryser kapaciteten till den körningens
    optimum. Flexibilitet (dispatch, lager-SOC, handel) optimeras fortfarande fritt.

    För lager kopieras även max_hours. Annars blandas källans p_nom med den NYA
    upplösningens max_hours, och eftersom RoR-splitten (som bevarar reservoarvolymen
    genom max_h = cap_mwh/p_nom) faller ut olika vid olika upplösning blir produkten
    p_nom×max_hours — reservoarvolymen — fel. Konkret: FI fick 5,27 i stället för
    5,50 TWh vid 2h-omdispatch av en 3h-körning (−4%).
    """
    src = pypsa.Network()
    src.import_from_netcdf(RESULTS_DIR / label / "network.nc")
    print(f"  → fryser kapaciteter till {label}:s p_nom_opt:")
    for cname, ndf, sdf in (("generatorer", n.generators, src.generators),
                            ("lager",       n.storage_units, src.storage_units),
                            ("länkar",      n.links, src.links)):
        if "p_nom_opt" not in sdf.columns or ndf.empty:
            continue
        common = [x for x in ndf.index if x in sdf.index]
        skipped = []
        if cname in ("generatorer", "länkar") and "p_nom_extendable" in sdf.columns:
            # p_nom_opt bär BARA information för extendable komponenter. För databestämda
            # must-run-generatorer (thermal, hydro_ror) är p_nom = profilens max i KÄLLANS
            # snapshot-fönster, medan p_min_pu = p_max_pu = profil/max normaliseras mot den
            # NYA körningens fönster. Att frysa p_nom skalar då om produktionen med
            # (källans max / nya fönstrets max) — och för must-run ÄR p_nom × pu dispatchen.
            # Konkret: 1h-dispatch per år av en 3-årig källa gav NO-N termik ×2,6 (2023) och
            # ×2,8 (2025), NO-S ×2,5/×3,1, eftersom årsmaxen ligger långt under 3-årsmaxet.
            #
            # ⭐ Samma resonemang gäller LÄNKAR, och där bet det 2026-08-15: en icke-
            # expanderbar länks p_nom_opt ÄR bara configvärdet, så frysningen skrev tyst
            # tillbaka det och annullerade `grid.ntc_override` på en dispatch. run348 blev
            # därför BIT-IDENTISK med run346 — experimentet såg ut att ha körts men hade
            # aldrig ägt rum. Med regeln här överlever en medveten NTC-ändring, medan en
            # källa som faktiskt expanderade länken (grid.expand_link) fortfarande fryses.
            # ⚠️ Lager omfattas AVSIKTLIGT inte: där behövs kopieringen av max_hours nedan
            # för att bevara reservoarvolymen genom RoR-splitten.
            fixed = [x for x in common if not bool(sdf.at[x, "p_nom_extendable"])]
            if fixed:
                skipped = fixed
                common = [x for x in common if x not in set(fixed)]
        ndf.loc[common, "p_nom"] = sdf.loc[common, "p_nom_opt"].astype(float)
        if "p_nom_extendable" in ndf.columns:
            ndf.loc[common, "p_nom_extendable"] = False
        if "max_hours" in ndf.columns and "max_hours" in sdf.columns:
            before = (ndf.loc[common, "p_nom"] * ndf.loc[common, "max_hours"]).sum()
            ndf.loc[common, "max_hours"] = sdf.loc[common, "max_hours"].astype(float)
            after = (ndf.loc[common, "p_nom"] * ndf.loc[common, "max_hours"]).sum()
            if abs(after - before) > 1e-6 * max(after, 1.0):
                print(f"      lagervolym korrigerad: {before/1e6:.2f} → {after/1e6:.2f} TWh "
                      f"(max_hours ärvs från {label})")
        miss = [x for x in ndf.index if x not in sdf.index]
        msg = f"      {cname}: {len(common)} frysta"
        if skipped:
            msg += (f", {len(skipped)} databestämda ej frysta (behåller egen p_nom: "
                    f"{', '.join(skipped[:3])}{' …' if len(skipped) > 3 else ''})")
        if miss:
            msg += f"  ⚠️ {len(miss)} saknas i {label} ({miss[:3]})"
        print(msg)


def apply_vre_curtailment_cost(n, cost: float) -> None:
    """dispatch.vre_curtailment_cost C: låt VRE bjuda vom − C i stället för vom.

    En kostnad C per MWh AVKORTAD energi ger objektivtermen

        mc·p + C·(A − p)  =  C·A + (mc − C)·p

    där A = p_max_pu·p_nom är den tillgängliga energin. Med FAST kapacitet är C·A
    en konstant som faller ur optimeringen — en avkortningskostnad är därför exakt
    ekvivalent med ett bud på mc − C, och implementeras enklast så. Det är den
    mekanism som ger negativa priser på riktiga marknader: subventionen (CfD,
    elcertifikat) sätter golvet på −C, eftersom producenten hellre betalar upp till
    C för att bli av med kraften än tappar stödet.

    Utan detta kan modellens zonpriser aldrig gå under billigaste budet (VRE:s VOM
    0,1 €/MWh): avkortning är gratis, så avdisposition kostar aldrig något och en
    extra MWh last kan aldrig sänka systemkostnaden.

    ⚠️ Kräver FRYSTA kapaciteter. Med extendable VRE är A ∝ p_nom_opt, C·A är då
    INTE konstant utan en produktionssubvention som växer med byggd kapacitet —
    optimeraren bygger till p_nom_max för att skörda den. Subventionen är en
    transferering, inte en resurskostnad, och hör hemma i prisbildningen men inte i
    investeringskalkylen. Därav tvåpass: expansion med sanna kostnader → dispatch
    med negativa bud.
    """
    if not cost:
        return
    g = n.generators
    targets = [x for x in g.index if g.at[x, "carrier"] in VRE_CARRIERS]
    if not targets:
        print(f"  → dispatch.vre_curtailment_cost {cost:g}: inga VRE-generatorer — ingen ändring")
        return

    ext = [x for x in targets if bool(g.at[x, "p_nom_extendable"])]
    if ext:
        raise SystemExit(
            f"\ndispatch.vre_curtailment_cost {cost:g} kräver frysta kapaciteter, men "
            f"{len(ext)} VRE-generatorer är fortfarande extendable\n"
            f"  (t.ex. {', '.join(ext[:3])}{' …' if len(ext) > 3 else ''}).\n\n"
            "Med extendable VRE blir avkortningskostnaden en produktionssubvention som\n"
            "växer med byggd kapacitet — modellen bygger till p_nom_max för att skörda\n"
            "den och investeringssvaret blir meningslöst. Kör i stället tvåpass:\n"
            "  1) nordpsa expand --output <label>\n"
            "  2) nordpsa dispatch --from <label> --output <nytt>\n"
        )

    print(f"  → dispatch.vre_curtailment_cost {cost:g} EUR/MWh: VRE bjuder vom − {cost:g}")
    tv = n.generators_t.marginal_cost
    for x in targets:
        g.at[x, "marginal_cost"] = float(g.at[x, "marginal_cost"]) - cost
        if x in tv.columns:                      # tidsberoende mc vinner över den statiska
            tv[x] = tv[x] - cost
    lo = g.loc[targets, "marginal_cost"]
    print(f"     {len(targets)} generatorer, bud {lo.min():.2f} … {lo.max():.2f} EUR/MWh "
          f"(prisgolv i överskottstimmar ≈ {lo.min():.2f})")
