"""
Parametrisk inflödesmodell för vattenkraft per zon.

Modell (daglig upplösning):
    inflow(doy) = A * exp(-((doy - mu) / sigma)^2)   # vårflod (Gausskurva)
                + B * cos(2π * (doy - phi) / 365)     # säsongskomponent
                + C                                    # basflöde

Enheter: MW (medeleffekt per timme)
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit

RAW_DIR       = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
YEARS         = [2023, 2024, 2025]


def _model(doy: np.ndarray, A: float, mu: float, sigma: float,
           B: float, phi: float, C: float) -> np.ndarray:
    spring   = A * np.exp(-((doy - mu) / sigma) ** 2)
    seasonal = B * np.cos(2 * np.pi * (doy - phi) / 365)
    return spring + seasonal + C


def load_actual_hydro(zone: str) -> pd.Series:
    """
    Laddar faktisk vattenkraftproduktion (MW, timvis) för alla år.
    Returnerar dagliga medelvärden indexerade på dag-på-året (1-365),
    medlade över alla tillgängliga år.
    """
    frames = []
    for year in YEARS:
        path = RAW_DIR / f"production_{zone}_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True)
        frames.append(df.set_index("timestampUTC")["hydro"])

    if not frames:
        raise FileNotFoundError(f"Ingen produktionsdata för {zone}")

    ts = pd.concat(frames).sort_index()
    daily = ts.resample("D").mean()
    daily.index = daily.index.dayofyear
    return daily.groupby(daily.index).mean()


def fit_zone(zone: str, actual: pd.Series) -> Dict[str, float]:
    """Fittar modellparametrar mot faktisk daglig produktion för en zon."""
    doy = actual.index.values.astype(float)
    y   = actual.values

    C0     = float(np.percentile(y, 10))
    A0     = float(np.percentile(y, 90) - C0)
    # Startgissning: vårflod topp kring dag 120 (maj), säsongsmin kring februari (phi≈65)
    p0     = [A0, 120.0, 30.0, A0 * 0.3, 65.0, C0]
    bounds = (
        [0,   60,  10,  0,    0,   0      ],
        [np.inf, 180, 90, np.inf, 180, np.inf],
    )

    try:
        popt, _ = curve_fit(_model, doy, y, p0=p0, bounds=bounds, maxfev=10_000)
    except RuntimeError:
        popt = p0  # fallback till startgissning

    A, mu, sigma, B, phi, C = popt
    return {
        "A": round(float(A), 2),
        "mu": round(float(mu), 2),
        "sigma": round(float(sigma), 2),
        "B": round(float(B), 2),
        "phi": round(float(phi), 2),
        "C": round(float(C), 2),
    }


def fit_and_save_all(zones: list[str]) -> Dict[str, Dict[str, float]]:
    """Fittar parametrar för alla zoner med hydro och sparar till hydro_params.yaml."""
    all_params: Dict[str, Dict[str, float]] = {}

    for zone in zones:
        path = RAW_DIR / f"production_{zone}_2023.parquet"
        if not path.exists():
            continue
        # Kolla om zonen har hydro
        df_check = pd.read_parquet(path)
        if df_check["hydro"].fillna(0).sum() == 0:
            print(f"  {zone}: ingen vattenkraft — hoppar över")
            continue

        print(f"  Fittar inflödesmodell för {zone} ...", end=" ", flush=True)
        try:
            actual = load_actual_hydro(zone)
            params = fit_zone(zone, actual)
            all_params[zone] = params
            print(
                f"OK  C={params['C']:.0f} MW  "
                f"A={params['A']:.0f} MW  "
                f"mu=dag {params['mu']:.0f}"
            )
        except Exception as e:
            print(f"FEL: {e}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "hydro_params.yaml"
    with open(out, "w") as f:
        yaml.dump(all_params, f, default_flow_style=False, sort_keys=True)
    print(f"  → {out.name}")
    return all_params


def load_nve_inflow(zone: str, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Laddar NVE faktisk veckovist inflöde (MW) och expanderar till snapshot-index
    via forward-fill (konstantvärde inom varje vecka, energikonservativt).

    Kräver data/raw/inflow_nve_{zone}_{year}.parquet för varje år i snapshots.
    Finns bara för NO-N och NO-S.
    """
    years = sorted(set(snapshots.year))
    frames = []
    for year in years:
        path = RAW_DIR / f"inflow_nve_{zone}_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Saknar NVE-inflödesdata: {path.name} "
                f"— kör 'python scripts/fetch_nve.py' först."
            )
        df = pd.read_parquet(path)
        df["week_start"] = pd.to_datetime(df["week_start"], utc=True).dt.tz_localize(None)
        frames.append(df.set_index("week_start")["inflow_mw"])

    weekly = pd.concat(frames).sort_index()

    # Expandera till snapshot-frekvens: lägg snapshots i det veckoglesade indexet
    # och ffill så att varje snapshot får veckans MW-värde
    combined_idx = weekly.index.union(snapshots)
    result = weekly.reindex(combined_idx).ffill().reindex(snapshots).fillna(0.0)
    result.name = "inflow_mw"
    return result


def load_nve_ror(zone: str, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Laddar run-of-river (B11) timprofil (MW) och reindexerar till snapshot-index.

    Returnerar must-run-profilen för strömkraft. Saknas filen returneras nollserie
    (zonen har ingen separat RoR-data). Resamplas till snapshot-frekvens via medel.
    """
    years = sorted(set(snapshots.year))
    frames = []
    for year in years:
        path = RAW_DIR / f"ror_nve_{zone}_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["timestampUTC"] = pd.to_datetime(df["timestampUTC"], utc=True).dt.tz_localize(None)
        frames.append(df.set_index("timestampUTC")["ror_mw"])

    if not frames:
        return pd.Series(0.0, index=snapshots, name="ror_mw")

    hourly = pd.concat(frames).sort_index()
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    # Resampla till snapshot-frekvens (medel), reindexera, fyll luckor
    dt_h = (snapshots[1] - snapshots[0]).total_seconds() / 3600
    resampled = hourly.resample(f"{int(dt_h)}h").mean()
    result = resampled.reindex(snapshots).ffill().bfill().fillna(0.0)
    result.name = "ror_mw"
    return result


_MODEL_KEYS = ("A", "mu", "sigma", "B", "phi", "C")


def inflow_timeseries(params: Dict[str, float],
                      timestamps: pd.DatetimeIndex,
                      annual_scales: Dict[int, float] | None = None,
                      peak_gamma: float | None = None,
                      target_annual_twh: Dict[int, float] | None = None) -> pd.Series:
    """
    Genererar inflödestidsserie (MW) för ett godtyckligt tidsstämpelindex.
    Om annual_scales är givet skalas inflödet per kalenderår mot faktisk produktion.

    peak_gamma:        potens-transform av den parametriska formen (>1 spetsar
                       vårfloden; FI-kalibrering mot ENTSO-E+eSett-härledd
                       tillrinning gav γ≈1.5).
    target_annual_twh: dict {år: TWh} → varje kalenderår normaliseras (efter
                       ev. peakning) så att årssumman exakt matchar faktisk
                       tillrinning. Ersätter annual_scales när angiven.

    Används av network.py för att sätta inflow_t på StorageUnits.
    """
    mparams = {k: params[k] for k in _MODEL_KEYS if k in params}
    doy     = timestamps.dayofyear.values.astype(float)
    values  = _model(doy, **mparams)
    values  = np.maximum(values, 0.0)
    if peak_gamma is not None and peak_gamma != 1.0:
        values = values ** peak_gamma
    result = pd.Series(values, index=timestamps, name="inflow_mw")

    if target_annual_twh:
        # Steg-timmar ur tidsstämpelsteget (uniformt rutnät, t.ex. 3h-upplösning).
        step_h = ((timestamps[1] - timestamps[0]).total_seconds() / 3600.0
                  if len(timestamps) > 1 else 1.0)
        for year, twh in target_annual_twh.items():
            mask = timestamps.year == year
            cur_mwh = result.loc[mask].sum() * step_h
            if cur_mwh > 0:
                result.loc[mask] *= twh * 1e6 / cur_mwh
    elif annual_scales:
        for year, scale in annual_scales.items():
            result.loc[timestamps.year == year] *= scale
    return result

def modulate_inflow(inflow: pd.Series,
                    sigma: float,
                    tau_days: float = 1.4,
                    seed: int = 0,
                    preserve: str = "week") -> pd.Series:
    """
    Multiplikativ lognormal AR(1)-modulering av en inflödesserie.

    MOTIV: inflödet saknar all variation INOM veckan. NVE-data (SE-N/SE-S/NO-N/NO-S)
    är veckovis och forward-fylls, alltså exakt konstant i 84 tvåtimmarssteg i rad;
    FI går via den analytiska modellen och ändrar sig bara per dygn. Mätt i run434:
    106-158 unika inflödesvärden på tre år i de fyra NVE-zonerna, 1095 (= 365x3) i FI.

    FORM: inflow' = inflow · exp(σ·z − σ²/2) med z[t] = φ·z[t−1] + √(1−φ²)·ε[t],
    φ = exp(−Δt/τ). Lognormal AR(1) är standardformen i hydrologisk schemaläggning:
    den ger positiv serie, högerskev fördelning och varians som skalar med nivån.

    KALIBRERING mot timvis NVE-RoR (B11) 2015-2025, residual mot veckomedel:
        NO-N  CV 0,178  ·  NO-S  CV 0,089  ·  τ (dygn) 1,4 i båda
    ⛔ SE:s RoR-serier duger INTE som mål — de är syntetiska (scripts/synth_se_ror.py),
    formade ur samma släta inflöde, och ger CV 0,007-0,009 (exakt 0,000 vår/sommar/höst).
    ⭐ Mätningen MOTSÄGER den hydrologiska intuitionen: variabiliteten är störst på
    VINTERN (NO-N 0,201) och minst i VÅRFLODEN (0,134) — vårfloden är en uthållig hög
    platå, stor i absoluta tal men slät, medan låga vinterflöden ger stora RELATIVA
    utslag vid regn-på-snö. En handsatt säsongsprofil hade blivit tvärtom.
    ⚠️ Två fel i motsatt riktning gör 0,09-0,18 till ett INTERVALL, inte ett tal: RoR är
    effektbegränsad så toppar kapas (underskattning), men RoR-avrinningsområden är mindre
    än magasinens och små områden varierar mer (överskattning).

    preserve: "week" -> varje ISO-vecka bevarar sin summa EXAKT. Används för NVE-zonerna,
              där vecka-till-vecka-signalen (våtår/torrår, vårflodens tajming) är UPPMÄTT
              och inte får skrivas över — moduleringen tillför då enbart inomveckoform.
              "year" -> varje kalenderår bevarar sin summa. Används för FI, som saknar
              mätt veckovariation och därför behöver variation även på veckoskalan.

    ⚠️ INSATT ≠ REALISERAT. Veckonormaliseringen tar bort lågfrekvensen INOM veckan, så
    både spridning och minne krymper systematiskt (uppmätt på SE-N:s serie, 2h, 3 år):
        sigma_in   0,08  0,10  0,12  0,15  0,18  0,22
        CV_ut      0,065 0,081 0,097 0,121 0,146 0,179     (~0,81 x sigma_in, linjärt)
        tau_in (d)  1,4   2,0   2,5   3,0   4,0   5,0
        tau_ut (d)  1,01  1,20  1,30  1,37  1,45  1,48     (mättar mot ~1,5)
    ⇒ för att TRÄFFA mätningen (CV 0,12 = mitt i NO:s 0,089-0,178, tau 1,4 d) ska man
    ange sigma 0,15 och tau 3,5. Skalfaktorerna gäller BARA preserve="week"; med
    preserve="year" är krympningen försumbar.
    Målet mättes med samma pipeline på RoR-serien, så jämförelsen är lika-mot-lika.

    ⭐ Volymbevarande per konstruktion ⇒ target_annual_twh, cykliskt SOC och framför allt
    DRIFTMÄTNINGEN mot EC:s +3,13 TWh är orörda, så körningen förblir jämförbar med
    run420-434.
    """
    if sigma <= 0:
        return inflow
    if preserve not in ("week", "year"):
        raise ValueError(f"preserve måste vara 'week' eller 'year', fick {preserve!r}")

    idx  = inflow.index
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0
    phi  = float(np.exp(-dt_h / (tau_days * 24.0)))
    rng  = np.random.default_rng(seed)

    # AR(1) i logrummet, stationär från start (z[0] ~ N(0,1), inte 0)
    eps = rng.standard_normal(len(idx))
    z   = np.empty(len(idx))
    z[0] = eps[0]
    s_in = np.sqrt(1.0 - phi ** 2)
    for i in range(1, len(z)):
        z[i] = phi * z[i - 1] + s_in * eps[i]

    m   = np.exp(sigma * z - 0.5 * sigma ** 2)
    out = inflow * m

    if preserve == "week":
        iso = idx.isocalendar()
        key = pd.Series(iso.year.values * 100 + iso.week.values, index=idx)
    else:
        key = pd.Series(idx.year.values, index=idx)

    num = inflow.groupby(key).transform("sum")
    den = out.groupby(key).transform("sum")
    out = out * (num / den.where(den > 0, 1.0))
    return out.clip(lower=0.0).rename(inflow.name)

def add_ror_hifreq(ror: pd.Series,
                   p_nom: float,
                   sigma: float,
                   tau_days: float = 1.4,
                   seed: int = 0,
                   iters: int = 8) -> pd.Series:
    """
    Ger en SYNTETISK strömkraftsprofil den högfrekventa struktur som Norges
    UPPMÄTTA B11 faktiskt har. Formtransplantation, inte brus för brusets skull.

    ⛔ VARFÖR BARA SE OCH FI: Svenska kraftnät rapporterar ingen B11-kategori till
    ENTSO-E (all svensk vattenkraft bokförs som B12), så `ror_entsoe_SE*` är tomma
    och SE:s serier konstrueras av scripts/synth_se_ror.py som (1-a)*inflode + a*platt.
    FI:s splittas av _synth_ror_profile ur den analytiska inflödeskurvan. Båda ärver
    därmed inflödets veckotrappa: MÄTT 52 unika värden per år, serien ändras i 0,6 %
    av timmarna, dygnsgång exakt 0. NO-N/NO-S är rapporterad timvis B11 (7 943 resp.
    8 476 unika värden/år, ändras 99,7-99,9 % av timmarna) och ska INTE röras.

    MÅLET, mätt på NO:s B11 2015-2025 (residual mot veckomedel):
        NO-N  CV 0,175  ·  NO-S  CV 0,089  ·  tau 1,4 d i båda
    ⭐ Dygnsamplituden är BARA 0,02-0,05 och vardag/helg-kvoten 1,03, alltså är NO:s
    variabilitet till övervägande del hydrologisk-synoptisk, inte driftmässig. Det är
    därför den går att transplantera till en annan zon: den beskriver avrinning, inte
    norska bolags körplaner.
    ⚠️ NO-N 0,175 mot NO-S 0,089 är en FAKTOR TVÅ och vi vet inte vilken SE liknar.
    Ett likformigt värde mitt emellan är samma val som b_mean: globalt tills en
    mätning säger annat.

    ⚠️ p_nom FRYSES av anroparen och skickas in här. Strömkraftens installerade effekt
    sätts annars ur seriens MAX, och en modulerad serie har högre max — kapaciteten
    hade då ändrats och A/B:t vore ingen enfaktorändring. Serien kapas därför vid
    p_nom, och den energi kapningen tar bort läggs tillbaka på veckans OKAPADE timmar
    (iterativt, konvergerar snabbt). ⇒ veckoenergin bevaras EXAKT och p_nom är orört.
    """
    if sigma <= 0:
        return ror

    idx  = ror.index
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0
    phi  = float(np.exp(-dt_h / (tau_days * 24.0)))
    rng  = np.random.default_rng(seed)
    eps  = rng.standard_normal(len(idx))
    z    = np.empty(len(idx)); z[0] = eps[0]
    sd   = np.sqrt(1.0 - phi ** 2)
    for i in range(1, len(z)):
        z[i] = phi * z[i - 1] + sd * eps[i]

    iso  = idx.isocalendar()
    key  = pd.Series(iso.year.values * 100 + iso.week.values, index=idx)
    want = ror.groupby(key).sum()

    out = ror * np.exp(sigma * z - 0.5 * sigma ** 2)
    for _ in range(iters):
        out  = out.clip(lower=0.0, upper=p_nom)
        miss = want - out.groupby(key).sum()          # per vecka, + = för lite energi
        if float(miss.abs().max()) < 1e-6:
            break
        # Bristen läggs på veckans LEDIGA utrymme (p_nom − out); ett överskott tas
        # bort proportionellt mot out självt. Båda riktningarna behövs: den
        # multiplikativa faktorn har väntevärde 1 men veckosumman blir aldrig exakt.
        head = (p_nom - out).groupby(key).sum()
        pos  = (miss / head.where(head > 1e-9, np.nan)).fillna(0.0)
        neg  = (miss / out.groupby(key).sum().where(lambda v: v > 1e-9, np.nan)).fillna(0.0)
        f_pos = key.map(pos).astype(float)
        f_neg = key.map(neg).astype(float)
        up    = miss.reindex(key.values).values > 0
        out   = pd.Series(np.where(up,
                                   out.values + (p_nom - out.values) * f_pos.values,
                                   out.values * (1.0 + f_neg.values)),
                          index=idx)
    return out.clip(lower=0.0, upper=p_nom).rename(ror.name)
