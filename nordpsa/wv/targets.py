"""Icke-cirkulära kalibreringsmål för terminalvärdeskurvan.

Två observabler, båda MÄTDATA — ingen av dem är modellutdata som matas tillbaka.
Det är hela skälet till att de duger: [[project_hydro_seasonal_bid]] kalibrerade S(w)
mot modellens eget pris och mätte därmed sitt eget eko.

  1. eSetts FYSISKA hydrosäsong per land, vinter/sommar-kvoten
     (jan+feb+dec)/(jun+jul+aug) på RESERVOARkraften. SE 1,38 · NO 1,30 · FI 1,13.
     ⚠️ Strömkraften ska skiljas ut: den är must-run och följer tillrinningen, så
     den späder ut måttet med en form inget modellval påverkar.

  2. Energy Charts VECKOVISA magasinnivåer per land, 2014-2026
     (docs/NordicHydroEC.xlsx). Ger ett klimatologiskt BAND, inte en årsbana.

## Varför bandet och inte 2023-25 års bana

Kurvan ska vara GENERISK. Passas den mot 2023-25 blir den en replay av just de åren
och vi har byggt `--soc-guide` igen (run310: lyfte nivån, inte dispatchen). Bandet
över tolv år är hydrologisk klimatologi och bär ingen enskild väderårs-information.

⚠️ KONSEKVENS SOM MÅSTE REDOVISAS: en genuint generisk kurva VET INTE att våren 2024
blev våt. Det gjorde inte operatörerna heller — det är den osäkerhet vi försöker
återinföra. Därför är RMSE mot faktisk dispatch FEL måttstock för det här spåret.
Måtten här är fördelningsmått: hamnar banan i bandet, och stämmer säsongsformen.

⚠️ Bandet är beskrivande, inte normativt. Det säger vad magasinen historiskt GJORDE
i dagens system, inte vad de borde göra i 2040 års. En kurva som pressar modellen
till exakt historiskt beteende är påtvingande på samma sätt som soc-guide var.
Använd bandet som räcke (hamnar vi utanför?), inte som mål (träffa medianen).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

#: Vinter/sommar-kvot på reservoarkraften, eSett 2023-25. Mätt, inte antaget.
ESETT_VS: Dict[str, float] = {"SE": 1.38, "NO": 1.30, "FI": 1.13}

LAND: Dict[str, list] = {"SE": ["SE-N", "SE-S"], "NO": ["NO-N", "NO-S"], "FI": ["FI"]}
WINTER, SUMMER = [1, 2, 12], [6, 7, 8]

#: Kolumnläge i EC-arket (blad "1"): datum, Finland, …, Norway, …, Sweden, …, Sum
_EC_COLS = {"FI": 1, "NO": 4, "SE": 7}


# ── Magasinvolymer ur configen ──────────────────────────────────────────────────

def reservoir_twh(cfg: dict | None = None) -> Dict[str, float]:
    """Modellens magasinvolym per LAND, TWh. p_nom × max_hours ur zones.yaml.

    Normaliseringen måste gå mot modellens egen volym, annars jämförs
    fyllnadsgrader mot olika nämnare. EC:s historiska maxnivåer ligger på 94 % av
    dessa volymer, vilket är den verifiering som gjorts.
    """
    cfg = cfg or yaml.safe_load(open(ROOT / "config" / "zones.yaml"))
    out = {}
    for land, zones in LAND.items():
        twh = 0.0
        for z in zones:
            zc = cfg["zones"].get(z, {})
            twh += zc.get("hydro_p_nom_mw", 0) * zc.get("hydro_max_hours", 0) / 1e6
        out[land] = twh
    return out


# ── Observabel 2: EC-bandet ─────────────────────────────────────────────────────

#: ⚠️ Ligger UTANFÖR git — `.gitignore` rad 48 ignorerar `*.xlsx`. En färsk klon
#: (och varje git-worktree) saknar den och måste symlänka eller kopiera dit filen.
EC_WORKBOOK = "docs/NordicHydroEC.xlsx"


def ec_levels() -> pd.DataFrame:
    """Veckovisa magasinnivåer i TWh per land, som de står i arket."""
    p = ROOT / EC_WORKBOOK
    if not p.exists():
        raise SystemExit(
            f"✖ {EC_WORKBOOK} saknas i {ROOT}.\n"
            "  Filen är gitignorerad (*.xlsx) och följer alltså inte med grenar eller\n"
            "  worktrees. Symlänka den från huvudrepot, som data/ redan gör:\n"
            f"      ln -s /home/beos/repos/NordPSA/{EC_WORKBOOK} {ROOT}/{EC_WORKBOOK}")
    raw = pd.read_excel(p, sheet_name="1")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    out = pd.DataFrame({c: pd.to_numeric(raw.iloc[:, i], errors="coerce")
                        for c, i in _EC_COLS.items()})
    out.index = dates
    return out[out.index.notna()].dropna(how="all")


def ec_band(quantiles=(0.10, 0.50, 0.90), cfg: dict | None = None) -> pd.DataFrame:
    """Fyllnadsgrad per ISO-vecka och land, kvantiler över alla år i arket.

    Index = vecka 1..52, kolumner = MultiIndex (land, kvantil).
    """
    lv = ec_levels()
    vol = reservoir_twh(cfg)
    frac = pd.DataFrame({c: lv[c] / vol[c] for c in lv.columns if vol.get(c)})
    frac["week"] = frac.index.isocalendar().week.clip(upper=52).astype(int)
    g = frac.groupby("week")
    cols, data = [], {}
    for c in [c for c in frac.columns if c != "week"]:
        for q in quantiles:
            cols.append((c, q))
            data[(c, q)] = g[c].quantile(q)
    out = pd.DataFrame(data)
    out.columns = pd.MultiIndex.from_tuples(cols, names=["land", "kvantil"])
    return out.reindex(range(1, 53))


# ── Modellsidan: läses ur CSV, inte network.nc ──────────────────────────────────
# network.nc för en 1h/3-årskörning drar 1-2 GB in i minnet. CSV:erna räcker för
# båda måtten och kostar några MB, vilket gör kalibreringsloopen körbar parallellt
# med annat.

def _read(label: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results" / label / name, index_col=0, parse_dates=True)
    return df


def run_weekly_soc(label: str, cfg: dict | None = None) -> pd.DataFrame:
    """Modellens fyllnadsgrad per land, veckomedel. Index = ISO-vecka 1..52."""
    cfg = cfg or yaml.safe_load(open(ROOT / "config" / "zones.yaml"))
    soc = _read(label, "hydro_soc.csv")
    vol = reservoir_twh(cfg)
    out = {}
    for land, zones in LAND.items():
        cols = [f"{z} hydro" for z in zones if f"{z} hydro" in soc.columns]
        if not cols or not vol.get(land):
            continue
        # SOC i CSV:n är MWh; landets fyllnadsgrad = summan / landets volym
        frac = soc[cols].sum(axis=1) / (vol[land] * 1e6)
        out[land] = frac.groupby(frac.index.isocalendar().week.clip(upper=52)).mean()
    return pd.DataFrame(out).reindex(range(1, 53))


def run_vs(label: str) -> Dict[str, float]:
    """Vinter/sommar-kvot på RESERVOARkraften per land, ur dispatch_hydro.csv.

    Strömkraften ligger inte i den filen (den är en Generator, inte StorageUnit),
    så uppdelningen är gjord redan av datastrukturen — samma mått som
    temp/hydro_by_country.py rapporterar under 'res'.
    """
    d = _read(label, "dispatch_hydro.csv")
    out = {}
    for land, zones in LAND.items():
        cols = [f"{z} hydro" for z in zones if f"{z} hydro" in d.columns]
        if not cols:
            continue
        s = d[cols].sum(axis=1)
        m = s.groupby(s.index.month).sum()
        w, su = m.reindex(WINTER).sum(), m.reindex(SUMMER).sum()
        out[land] = float(w / su) if su else float("nan")
    return out


def reservoir_drift(label: str) -> dict:
    """Magasinets NETTODRIFT över körningens hela period, TWh per zon + summa.

    ⭐ Driften är en GRATIS mätare på om λ_bas ligger rätt för världen man kör i.
    En rullande dispatch har inget cykliskt villkor, så magasinet får driva fritt:

        drift > 0   vattnet är för DYRT   → modellen hamstrar
        drift < 0   vattnet är för BILLIGT → modellen gör av med lagret
        drift ≈ 0   nivån passar världen

    Mätt: run317 (2040-nivå i dagens värld, ~3× för högt) +20,0 · run288 S(m) +14,1 ·
    run318 (rätt nivå för dagens värld) −0,3 · run319 (2040) −14,2.

    ⚠️ Noll drift är ett NÖDVÄNDIGT villkor, inte ett tillräckligt: en kurva med rätt
    nivå men fel form kan mycket väl balansera lagret och ändå fördela vattnet fel
    över året. Mät alltid v/s och bandet också (score()).
    """
    soc = _read(label, "hydro_soc.csv")
    out = {}
    for c in soc.columns:
        if " hydro" not in c:
            continue
        out[c.split()[0]] = float(soc[c].iloc[-1] - soc[c].iloc[0]) / 1e6
    out["SUMMA"] = sum(v for k, v in out.items() if k != "SUMMA")
    return out


# ── Poäng ───────────────────────────────────────────────────────────────────────

def score(label: str, w_vs: float = 1.0, w_band: float = 2.0,
          band=(0.10, 0.90), cfg: dict | None = None) -> dict:
    """Sammanvägt fel för en körning. LÄGRE är bättre.

    Två termer per land:
      |Δv/s|      avvikelse i säsongskvot mot eSett (dimensionslös)
      utanför     medelavstånd i fyllnadsgrad till närmaste bandkant, 0 inuti bandet

    Vikterna är satta så att 0,10 i kvotfel väger lika tungt som 0,05 i
    fyllnadsgrad. ⚠️ ANTAGANDE — inget i data pekar ut en växelkurs mellan måtten.
    """
    cfg = cfg or yaml.safe_load(open(ROOT / "config" / "zones.yaml"))
    vs = run_vs(label)
    soc = run_weekly_soc(label, cfg)
    bnd = ec_band(quantiles=band, cfg=cfg)

    per, total = {}, 0.0
    for land in LAND:
        d_vs = abs(vs.get(land, float("nan")) - ESETT_VS[land])
        if land in soc.columns and (land, band[0]) in bnd.columns:
            lo, hi = bnd[(land, band[0])], bnd[(land, band[1])]
            s = soc[land]
            outside = (lo - s).clip(lower=0) + (s - hi).clip(lower=0)
            d_band = float(outside.mean())
            inside = float((outside <= 1e-9).mean())
        else:
            d_band, inside = float("nan"), float("nan")
        per[land] = {"vs": vs.get(land), "vs_mal": ESETT_VS[land], "d_vs": d_vs,
                     "utanfor": d_band, "andel_i_band": inside}
        if d_vs == d_vs and d_band == d_band:
            total += w_vs * d_vs + w_band * d_band
    return {"label": label, "poang": total, "per_land": per}


def report(label: str, **kw) -> dict:
    """score() + en läsbar tabell."""
    r = score(label, **kw)
    print(f"\n{label}  →  POÄNG {r['poang']:.4f}  (lägre är bättre)\n")
    print(f"{'land':5s} {'v/s':>7s} {'mål':>7s} {'|Δ|':>7s} "
          f"{'utanför band':>13s} {'i band':>8s}")
    for land, m in r["per_land"].items():
        vs = m["vs"] if m["vs"] is not None else float("nan")
        print(f"{land:5s} {vs:7.2f} {m['vs_mal']:7.2f} {m['d_vs']:7.2f} "
              f"{m['utanfor']:13.3f} {m['andel_i_band']:7.0%}")
    return r


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Poängsätt en körning mot de två observablerna")
    ap.add_argument("runs", nargs="*", default=["run260_baseline_2h"])
    ap.add_argument("--band", nargs=2, type=float, default=[0.10, 0.90], metavar=("LO", "HI"))
    args = ap.parse_args()

    b = ec_band(quantiles=(0.10, 0.50, 0.90))
    print("EC-bandet (fyllnadsgrad, 2014-2026), var 8:e vecka:")
    print(b.iloc[::8].round(2).to_string())
    for r in args.runs:
        report(r, band=tuple(args.band))


# ── Facit-mått: avstånd till fullframsyntens egen lösning ────────────────────────
# score() ovan mäter mot eSetts DAGENS v/s och EC-bandet — rätt måttstock för dagens
# värld, fel för 2040. I 2040 är måltavlan i stället den FULLFRAMSYNTA dispatchen av
# samma frysta flotta utan prisproxy: den vet allt om framtiden och allokerar vattnet
# optimalt, och dess vattenvärde är då exakt konstant (run320: 73,02 EUR/MWh i alla
# zoner och timmar — den deterministiska teorins svar). Den rullande horisonten med
# terminalkurva ska återskapa den lösningen under BEGRÄNSAD framsyn; avståndet dit är
# därför det relevanta felet.
#
# ⚠️ Facit är ett TAK, inte ett facit i moralisk mening: perfekt framsyn kör magasinen
# ned till 2-4 %, vilket verkliga operatörer inte gör. Använd som riktmärke att röra
# sig mot, inte som norm att träffa exakt.

FACIT_RUN = "run320_pf_ref_dispatch_2h"


def harmonic(label: str):
    """(amplitud, toppvecka) för veckoproduktionens första harmoniska.

    Fasen är den diagnos som styr `a_peak`: som reservationspris ska λ ha sin BOTTEN
    där produktionen ska toppa, alltså a_peak = toppvecka + 26.
    """
    import numpy as np

    d = _read(label, "dispatch_hydro.csv")
    s = d[[c for c in d.columns if c.endswith(" hydro")]].sum(axis=1)
    y = s.groupby(s.index.isocalendar().week.clip(upper=52)).mean().reindex(range(1, 53))
    y = (y / y.mean()).values
    wk = np.arange(1, 53)
    c = float(np.sum(y * np.cos(2 * np.pi * wk / 52)))
    si = float(np.sum(y * np.sin(2 * np.pi * wk / 52)))
    return 2 * float(np.hypot(c, si)) / 52, (np.degrees(np.arctan2(si, c)) / 360 * 52) % 52


def facit_score(label: str, w_vs: float = 1.0, w_soc: float = 2.0,
                facit: str = FACIT_RUN) -> dict:
    """Σ_land |Δv/s| mot facit + w_soc × medelavstånd i veckovis fyllnadsgrad.

    Andra termen fångar det v/s ensamt missar: att banan ligger på rätt NIVÅ över
    året, inte bara har rätt vinter/sommar-kvot. Vikterna är samma antagande som i
    score() — 0,10 i kvotfel väger som 0,05 i fyllnadsgrad.
    """
    v, vf = run_vs(label), run_vs(facit)
    s, sf = run_weekly_soc(label), run_weekly_soc(facit)
    d_vs = {l: abs(v[l] - vf[l]) for l in LAND if l in v and l in vf}
    d_soc = {l: float((s[l] - sf[l]).abs().mean())
             for l in LAND if l in s.columns and l in sf.columns}
    total = sum(w_vs * d_vs.get(l, 0.0) + w_soc * d_soc.get(l, 0.0) for l in LAND)
    return {"label": label, "total": total, "d_vs": d_vs, "d_soc": d_soc,
            "vs": v, "vs_facit": vf}
