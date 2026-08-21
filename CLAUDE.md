# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is NordPSA

Nordic power system model built on PyPSA. Combines LP dispatch optimization with capacity expansion (investment) for 6 aggregated zones: SE-N, SE-S, NO-N, NO-S, DK, FI. Covers 2023–2025; the canonical baseline runs at 2h, and 1h/3h are available via `--resolution`.

## Setup

```bash
conda env create -f environment.yml   # create environment
conda activate nordpsa-env
pip install -e .                       # install nordpsa package
```

## Workflow

Data must be fetched and built before solving. Full pipeline:

```bash
make fetch        # fetch eSett load + production data (scripts/fetch_esett.py)
make fetch-ec     # fetch Energy Charts VRE profiles + DE-LU day-ahead price
make build        # build processed parquet inputs (scripts/build_inputs.py)
make solve        # run model (scripts/run_model.py)
```

**The bare command runs the canonical expansion baseline** (see below) — no flags needed:
```bash
python scripts/run_model.py --output run01_description              # baseline: 2h, 2023-2025
python scripts/run_model.py --year 2024 --output run02_2024only     # baseline, single year
python scripts/run_model.py --resolution 3 --output run03_coarse    # coarser, everything else baseline
```

### The three run modes

Everything else in this file is easier to read once these are distinct. Only the first optimizes capacities; the other two are both *dispatch*, and differ only in **where the fixed capacities come from**.

| Mode | Capacities | Needs a source run | Water-value proxy | `--spill-cost` | `--vre-curtailment-cost` |
|---|---|---|---|---|---|
| **Expansion** (default, `--expansion` to say so) | optimized (`p_nom_opt`) | — | **on** | 50 | 0 |
| `--no-expansion` | frozen at **config** values = today's fleet | no | off | 0.1 | 5 |
| `--dispatch LABEL` | frozen at **LABEL's optimized** fleet | yes | off | 0.1 | 5 |

`--no-expansion` is not a synonym for `--dispatch` and cannot be removed: 82 historical runs use it, **none** of them together with `--dispatch`. It is the whole "today's world" line — run203/run208 (dispatch references), run255/run256 (todayworld), run283 + run289–292 (NTC sweeps), run268–288 (rolling-horizon development, incl. run288), run317/run318 (the terminal curve's first validation against today's observables). Without it there is no way to run against the actual present-day fleet without first producing a sham "expansion" that expands nothing. It is also load-bearing for reproducibility: `apply_dispatch_replay` re-parses the source run's argv string, so deleting the flag would break `--dispatch` on all 82.

**The water-value proxy has no flag at all (since 2026-08-15)** — it follows the mode: on in expansion, off whenever capacities are frozen. See the dedicated section below for why, and for what was removed with it.

**The bare `--dispatch` runs the canonical dispatch template** (the configuration of `run340_minh005_seg20_2h`) — capacities frozen from the source expansion run, rolling horizon 1+3 weeks, the calibrated terminal water-value curve, and no price proxy:

```bash
python scripts/run_model.py --output run01_expansion                      # kanonisk EXPANSION
python scripts/run_model.py --dispatch run01_expansion --output run02_disp # kanonisk DISPATCH
```

`--dispatch` implies five things that previously had to be spelled out on every command. Forgetting any of them made the run fall back *silently* to a cyclic LP with the price proxy — the opposite of what was intended, which is exactly the trap run319 fell into. Each has an off-switch:

| implied by `--dispatch` | off-switch |
|---|---|
| `--rolling-horizon`, 1 week/window + 3 weeks look-ahead | `--no-rolling-horizon` |
| `--terminal-curve` = `config/terminal_curve_2040_gemini_v7.yaml` (**låst 2026-08-20**, se nedan) | `--no-terminal-curve` |
| no water-value proxy (the curve *is* the water value) | — (mode-bound, no flag) |
| resolution 2h (`DEFAULT_DISPATCH_RESOLUTION`, was 1h) | `--resolution N` |
| `--spill-cost 0.1` (`DEFAULT_SPILL_COST_DISPATCH`, was 50 — see below) | `--spill-cost N` |

`--expansion` exists as an explicit synonym for "neither `--dispatch` nor `--no-expansion`"; it changes nothing but lets scripts say which mode they mean, and errors if combined with `--dispatch`.

⚠️ **Reproducing dispatch runs made before this change** (run316 and older) needs `--no-rolling-horizon --no-terminal-curve --resolution 2 --hydro-min-hourly 0.10`. This follows the same precedent as `--vre-curtailment-cost`: new defaults apply to newly typed commands, and older runs are reproduced by naming the old values. ⛔ One exception is now unreachable: those runs also had the **price proxy on**, and since the proxy became mode-bound there is no flag to turn it back on in a frozen-capacity run. Pre-run316 dispatch is therefore reproducible in structure but not bit-for-bit.

**Run discipline:** commit *after* a successful simulation, not before. This ensures only good runs are traced to code state. Always propose the commit message and wait for user approval before committing. Name the output directory in the commit message so results are traceable:
```bash
python scripts/run_model.py --output run02_fleet_factors
# verify results, then:
git commit -m "Change X → ran: run02_fleet_factors"
```
Results go to `results/<output>/` and are gitignored (large files). `network.nc` contains full PyPSA network including inflow timeseries — verify correct hydrology with `n.storage_units_t.inflow`.

Visualize results:
```bash
python scripts/plot_dispatch.py results/run01_spring_flood_cyclic/ --resample 7D
```

## Architecture

### Data flow

```
eSett API → data/raw/production_*.parquet   (actual load + generation by carrier)
            data/raw/consumption_*.parquet
EC API    → data/raw/vre_*.parquet          (VRE capacity factor profiles)
            data/raw/price_market_*.parquet (DE-LU day-ahead price)
            ↓
scripts/build_inputs.py
            ↓
data/processed/
  load.parquet            (6 zones × hourly MW)
  vre_profiles.parquet    (columns: {zone}_{carrier}, capacity factors 0-1)
  vre_pnom.yaml           (installed capacities per zone/carrier)
  nuclear_profile.parquet (availability factor per zone, 0-1)
  thermal_profile.parquet (must-run thermal MW per zone)
  hydro_params.yaml       (GENERATED — do not edit, auto-fitted from production data)
  market_price.parquet    (DE-LU hourly price)
            ↓
nordpsa/network.py → pypsa.Network
            ↓
scripts/run_model.py → results/
```

### nordpsa/ package

- `network.py` — builds PyPSA network from processed inputs. Key function: `build_network(cfg, snapshots, load, vre_profiles, vre_noms, nuclear_profile, thermal_profile, hydro_params, market_price)`
- `esett.py` — eSett open data client, aggregates MBAs into NordPSA zones
- `ec.py` — Energy Charts API client for VRE profiles and DE-LU price
- `hydro.py` — parametric inflow model (Gaussian spring flood + seasonal cosine); fits against actual data, generates `inflow_timeseries()` for PyPSA StorageUnits

### Network components

| Component | Carrier | Notes |
|-----------|---------|-------|
| Bus | AC | One per zone |
| Link | — | Bidirectional NTC (p_min_pu=-1) |
| StorageUnit | hydro | Reservoir with parametric inflow, cyclic SOC, no pumping |
| Generator | nuclear | Existing fleet must-run: p_min_pu = p_max_pu, dispatch = availability × p_nom. New build (`--add-nuclear`) load-follows at 0.6 × p_max by default |
| Generator | wind_onshore/offshore, solar | VRE with capacity factor profiles |
| Generator | thermal | Must-run: p_min_pu = p_max_pu = actual profile |
| Generator | gas | Dispatchable peaker, extendable |
| Generator | market | Import/export valve: p_min_pu=-1, marginal_cost=DE-LU price |
| Generator | slack | Load shedding (3000 EUR/MWh). All six zones by default (`--voll`); `--no-voll` limits it to the zones without a market connection |

### Cost model

Capital cost = `overnight_eur_per_w × 1e6 × (CRF + fom_fraction) × n_years`
CRF = `r × (1+r)^L / ((1+r)^L − 1)` with r=0.06, fom_fraction=0.02.
Capital cost is charged on `p_nom_opt` (total installed capacity, not just increment). All extendable generators have `p_nom_min = existing_capacity`.

### Zones and market connections

SE-N and NO-N have no direct continental market connection — with `--no-voll` they have only slack generators.
SE-S, NO-S, DK, FI have `market` generators (p_nom from config, price = DE-LU day-ahead).

## Important design decisions

**Canonical expansion baseline (defaults since 2026-08-05):** `python scripts/run_model.py` with no flags *is* the baseline — the configuration of `run250_hydroops_2h`, adopted so that experiments differ from the baseline only by the flags they actually name. Every default has an off-switch:

| Setting | Default | Off-switch |
|---|---|---|
| resolution | 2h (`snapshots.resolution_hours` in `zones.yaml`) | `--resolution N` |
| `--spill-cost` | 50 EUR/MWh (expansion; 0.1 med frysta kapaciteter — se nedan) | `--spill-cost N` |
| `--cost-scenario` | `svk_2040` | `--cost-scenario none` |
| `--demand-scenario` | `svk_2040_mm` | `--demand-scenario none` |
| `--add-heat` | ON | `--no-add-heat` |
| `--hydro-restrictions` | ON | `--no-hydro-restrictions` |
| `--onwind-capfac-increase` | 0.30 | `--onwind-capfac-increase 0` |
| `--offwind-capfac-increase` | 0.10 | `--offwind-capfac-increase 0` |
| `--nuclear-min-load` | 0.6 | `--nuclear-min-load 1.0` |
| `--add-nuclear` | `SE-S:10:201 SE-N:10:202 FI:10:203` | `--no-add-nuclear` |
| `--voll` | 3000 EUR/MWh in **all** zones | `--no-voll` |
| `--market-elasticity` | ON (predates this change) | `--no-market-elast` |
| `--hydro-mc-curve` | ON i expansion (2026-08-20), `config/terminal_curve_2040_gemini_v7_exp73.yaml` | `--no-hydro-mc-curve` |

`--voll` is the one default that does *not* reproduce `run250_hydroops_2h` exactly. It does not change the shedding price — `MC_SLACK` is already 3000 — only *where* slack exists: without it only the zones lacking a market connection (SE-N, NO-N) have a backstop, with it all six do. In an expansion run this gives the optimizer a new option: shed load at 3000 EUR/MWh instead of building peak capacity. Break-even against gas (~180 kEUR/MW/yr annualized) is roughly 60 scarcity hours per year, so FI — whose price tail is ~29 scarcity h/yr — is where a difference is most likely to show up as slightly less gas capacity. Also note `--voll` is taken from the *new* command on a `--dispatch` replay (as is `--no-voll`), so redispatch now gets VOLL by default; this removes the old trap where the flag had to be repeated manually.

**Dispatch-only default: `--vre-curtailment-cost 5` (since 2026-08-07).** Applies *only* when capacities are frozen — `--dispatch` or `--no-expansion` — and is 0 otherwise. It sets `marginal_cost = vom − C` on all wind/solar, which is algebraically identical to charging C per MWh curtailed (`mc·p + C·(A−p) = C·A + (mc−C)·p`; at fixed capacity `C·A` is a constant). This is the **only** route to negative zone prices: without it curtailment is free, no bid is negative, and there is no unit commitment (`committable=0`, `start_up_cost=0`), so the price floor is the cheapest bid (VOM). Verified in `run258_ref_dispatch_3h`: min 0.10 and zero negative hours in all six zones.

- ⚠️ **Never in expansion.** With extendable VRE, `A ∝ p_nom_opt`, so `C·A` is not constant — the term becomes a production subsidy that grows with built capacity and the optimizer builds to `p_nom_max`. `apply_vre_curtailment_cost` raises `SystemExit` if any VRE generator is still extendable; that is why the default is a `None` sentinel resolved *after* `apply_dispatch_replay`, not an argparse default.
- The flag is taken from the *new* command on a `--dispatch` replay (like `--voll`), since the source's argv can never contain it.
- **Calibration:** C represents what is actually lost per MWh when curtailing in 2040 — guarantees of origin only (forecast 2–5 EUR/MWh). Elcertifikat is zero (closed to new plants after 2021, ends 2035, early closure under investigation) and CfDs pay nothing in negative hours (CEEAG suspends support at negative prices; German EEG §51 tightens to a 1-hour rule in 2027). **5 is the upper end of the GO range**; the central estimate is 2.
- ⚠️ Reproducing runs made before this change (run255–run258, and any older dispatch) requires `--vre-curtailment-cost 0` explicitly.

Two traps this created, both handled in `scripts/run_model.py`:

- **`--add-nuclear` uses `action="extend"`**, so a non-empty argparse default would make a user-supplied `--add-nuclear` *append to* the baseline list instead of replacing it. The default is therefore the sentinel `None`, resolved to `DEFAULT_ADD_NUCLEAR` after `parse_args`.
- **`--dispatch` replays a source run's argv**, which was written against whatever defaults existed then. Replaying a pre-change run unmodified would silently inject the new defaults (e.g. give run240 hydro restrictions it never had). `write_run_meta` therefore stamps a `defaults:` line (`BASELINE_DEFAULTS_TAG`); `apply_dispatch_replay` restores `PRE_BASELINE_DEFAULTS` only for source runs *lacking* that line. Runs made after the change keep today's defaults.

`--resolution` deliberately stays `default=None` in argparse: `apply_dispatch_replay` relies on `args.resolution or 1` to keep redispatch at 1h, so the 2h baseline lives in config instead.

**`--spill-cost` is mode-dependent: 50 in expansion, 0.1 with frozen capacities (since 2026-08-15).** It is the mirror image of `--vre-curtailment-cost` — a modelling guardrail in one mode and a physical cost in the other — and an explicit value wins in both.

- **Expansion 50 = guardrail, not a real cost.** With free spill the optimizer can build excess wind/solar and dump the displaced hydro almost for nothing, so the system's true ability to absorb VRE is hidden and it overinvests. Measured 2026-05-31: run43 (spill 1) gave 63 TWh phantom spill and NO-S wind 20.8 GW; run44 (spill 50) gave 0 spill and 13.1 GW — **−37 %**.
- **Dispatch 0.1, because 50 double-counts the value of water.** With frozen capacities there is no investment decision to distort, and the cost of spilling *is* the value of the water lost — already carried by the LP as the shadow price on SOC (λ ≈ 73–80). At the ceiling marginal water has zero storage value, so the choice is produce at price *p* or pay *c*; the model produces as long as `p > −c`. At c = 50 it runs hydro down to **−50 EUR/MWh** rather than spill, which no real operator does — they bypass.
- ⚠️ **Empirically inert, so the change is principled rather than numerical — measured, not assumed.** Spill is 0.0000 TWh in every run in the terminal-curve track (run316–run345), max SOC reaches 94 % (run340) / 99.9 % (run320 NO-S), and in *all* hours above 90 % fill the price sits at 52–74 EUR/MWh with **zero** hours below 5. The fingerprint of the double-count — nearly full reservoir *and* collapsed price *and* hydro running hard — appears nowhere.
- **A/B `run346_spillcost01_2h` vs `run344_default_dispatch_2h`** (identical except 50 → 0.1): `hydro_spill.csv` **bit-identical** (0 both ways), zone prices move ≤ 0.018 EUR/MWh in the mean, negative-hour counts identical to the hour, hydro 531.91 → 531.96 TWh, v/s SE 1.547 → 1.548 · NO 2.021 → 2.017 · FI 0.989 → 0.992, facit score 0.4185 → 0.4189, drift −1.25 → −1.30 TWh. The residual differences are **degenerate-LP tie-breaking, not behaviour**: since spill is 0 in both, the coefficient contributes nothing to either objective and only changes which optimal vertex the solver reports. Window objectives sum to −1.041535e12 vs −1.041483e12 (0.005 %); note the rolling horizon is a *sequence* of coupled LPs (SOC carries over), so the totals have no optimality relation to each other and a different tie-break in one window shifts the next window's starting point.
- ⛔ **Falsified by the same A/B:** the hypothesis that a high spill cost pushes production *earlier* (the known ~5-week phase error, v49 against facit's v0.6). The first harmonic of weekly production is **v49.2, amplitude 0.458, in both runs** — unmoved to the first decimal. The phase error is not spill-driven.
- ⚠️ Note the model **under-spills relative to reality**: Ek Fälth et al. put real annual production loss from bypass spill at 0.12–0.22 %, the model at 0.000 %.

Two traps this closed: `--spill-cost` was **not** copied from the new command in `apply_dispatch_replay`, so `--dispatch X --spill-cost 0.1` was *silently ignored* (the run319 class of trap); and the flag did not appear in the `flaggor:` line of `run_meta.txt`, so a run's spill cost could only be read from `console.log`. Both fixed. The `PRE_BASELINE_DEFAULTS` entry for `spill_cost` was removed as unreachable — the new dispatch default 0.1 is exactly what its restore path produced via `network.py`'s `.get(..., 0.1)` fallback.

**IPM with crossover:** Solver must use `run_crossover: "on"` for capacity expansion runs. Without crossover, p_nom_opt stays near p_nom_min even when investment is profitable (interior-point primal solution, not a vertex).

**Nuclear — existing fleet must-run, new build load-following:** `p_min = p_max × min_frac` in both the dispatch branch (actual `nuclear_profile`) and the synthetic-nuclear branch (`--add-nuclear`, `availability_timeseries`), via `NUCLEAR_MIN_FRACTION = 1.0` (`nordpsa/network.py`).

- **Existing fleet: `min_frac = 1.0`** → `p_min_pu = p_max_pu`, the optimizer cannot down-regulate it, and dispatch (`n.generators_t.p`) equals the availability profile × p_nom in every snapshot.
- **New nuclear (`--add-nuclear` / `--add-nuclear-fixed`): `min_frac = 0.6` by default** since the canonical baseline (`--nuclear-min-load`, sets `min_load_frac_exp`). `--nuclear-min-load 1.0` makes new nuclear pure must-run too.

Verify with `--dry-run`, which prints `must-run` per generator: in the baseline `SE-S nuclear` shows 0.837 (= its CF, pure must-run) while `SE-S nuclear exp` shows 0.517 (= 0.6 × CF).

**⭐⭐⭐ TERMINALKURVAN ÄR LÅST (2026-08-20): `config/terminal_curve_2040_gemini_v7.yaml`** — kör som `run391_v7_anchors_3h`. Den ersätter `terminal_curve_2040_gemini.yaml` (låst 2026-08-18), som ligger kvar för att reproducera run316–390 — namnge den då explicit.

⭐ **Kurvan har TRE globala frihetsgrader, inte tjugofem.** Allt som är en *nivå* eller *skala* är gemensamt för alla zoner; bara *formen* är zonvis:

| öppen | värde | härledning |
|---|---|---|
| `a_scale` | **0,30** | run384: `a_amp` ×0,6 → ×0,3 gav SE:s v/s 1,18 → 1,32, bäst på båda måltavlorna |
| `b_mean` | **0,80** likformigt | facit ger λ konstant i fyllnadsgrad ⇒ ingen uppmätt zonskillnad finns |
| `b_amp` | **0,27** likformigt | samma argument; enkelt medel, eftersom zontalen är lika otillförlitliga |
| λ_bas | 73 i filen | facit run386: 72,87 i alla fem zoner och alla timmar |

⭐ **Att gå från 25 frihetsgrader till 3 kostade 0,01 på säsongskvoten** (run388 → run390: Σ|fel| eSett 0,42 → 0,43) och blev *bättre* mot 2040-facit (0,59 → 0,57). Geminis zonvisa `b_mean`/`b_amp` bar alltså praktiskt taget ingen information — precis vad man väntar sig om kurvans form enbart är en approximationsanordning för begränsad framsyn.

**Låst form:** `x_ref` = **två harmoniska** (se nedan) · alla säsongsfaser (`a_peak`, `a_peak2`, `b_peak`) · kvoten `a_amp2/a_amp` · `b_low_frac` 1,0 · `p_norm` "mid". `a_amp`/`a_amp2` står nu med Geminis **oskalade** värden och `a_scale` bär skalningen — de måste alltid skalas tillsammans, annars ändras vintertoppens form (`A_k − 1 = k·(A − 1)`, verifierat till 2·10⁻¹⁶).

⚠️ **ANKARET: `--terminal-anchor SE-N:66.0 SE-S:64.5 NO-N:70.2 NO-S:69.6 FI:72.5` i dispatch.** Det är en **driftkorrigering för begränsad framsyn**, inte ett påstående om att vatten är olika mycket värt per zon: facit ger 72,87 likformigt *och* noll drift, men en rullande horisont kan bara få det ena av två med ett enda tal. Vid gemensamt λ_bas 68,5 driver Sverige +1,0 och Norge −1,4 TWh, stabilt över tre olika kurvor. ⛔ **EXPANSION körs med 73 likformigt** — cykliskt SOC ger noll drift per konstruktion, så korrigeringen saknar mening där.

⚠️ Kvarstående brister i den låsta kurvan: **SE ligger helt utanför EC-bandet** (0 %, mot run388:s 15 %; magasinet sitter ~5 pp under tionde percentilen) · **FI:s drift −0,51 går inte att nolla** (känslighet 0,069 kräver +9,0 enheter = 12 % avvikelse, takat vid +2,0) · **prisformen är något sämre** än run388 (medelfel 8,1 mot 7,3) · **FI:s v/s 0,77** mot eSetts 1,13 och facits 0,97. ⚠️ Och känslighetens tvåpunktsskattning visade sig **4× för låg** för SE-N — dämpa och taka alltid vid omkalibrering.

**Terminalvärdeskurvan är LÅST (2026-08-18): `config/terminal_curve_2040_gemini.yaml`.** Den ersätter `terminal_curve_2040_calibrated.yaml`, som var kalibrerad mot *facit* (run320) och ligger kvar för att reproducera run316–run368 — namnge den då explicit med `--terminal-curve`.

Kurvan är `λ_k(v,z) = λ_bas[z] · A(v,z) · P_k(x − x_ref(v,z))` och vilar på tre mätningar:

| del | varifrån | not |
|---|---|---|
| **formen** A(v), B(v) | anpassad mot Geminis λ(period, fyllnad)-tabeller, en per zon (SE1/SE2, NO4, NO2, SE3/SE4, FI) | bara mot de **bebodda** cellerna: RMS 6,9 % mot 23,8 % om flodkollapsen >85 % tas med — och den fyllnadsgraden inträffar aldrig, EC:s median toppar på 84,5 % |
| **x_ref(v)** säsongsreferensen | minsta kvadrat mot EC:s uppmätta magasinmedian per land | R² 0,92 (SE) / 0,95 (NO) / **0,64 (FI)** |
| **λ_bas = 73** likformigt | reservoardriften | +12,07 TWh vid 80 → −0,74 vid 73; känslighet **1,93 TWh per enhet** |

`a_amp`/`a_amp2` är därutöver skalade **×0,6** mot eSetts v/s ur en uppmätt gradient (d(v/s)/d(a_amp) ≈ −1,7 SE, −1,3 NO, −0,36 FI).

⛔ **MÄTFEL RÄTTAT 2026-08-19 — v/s-siffrorna i det här avsnittet är på FEL basis.** eSett-målen 1,38/1,30/1,13 är mätta på **all** vattenkraft (eSetts `hydro` är en odelbar post, reproducerat exakt ur råfilerna), men `run_vs()` mätte modellens **magasin** ensamt. På lika-mot-lika blir run379 SE **1,15** · NO **1,26** · FI **0,64** — dvs. **Sveriges feltecken vänder** (för lite vinterproduktion, inte för mycket) och Norge är i praktiken i mål. Skalningen ×0,6 gick ändå åt rätt håll (Σ|fel| 1,14 → 0,79) och rangordningen mellan kurvorna är oförändrad, så kurvan står kvar — men **motiveringen ovan är fel och gradienten pekar åt motsatt håll**. `run_vs()` summerar sedan dess magasin + `hydro_ror`; `include_ror=False` reproducerar det gamla måttet, och `facit_score()` är pinnad dit (facit är en modellkörning, RoR bit-identisk på båda sidor).

⭐ **Säsongsreferensen är det som gör formen identifierbar.** Utan den mäts fyllnaden mot en fast mittpunkt 0,5, fast banan svänger 26 → 85 % — så 28 % i april lästes som knapphet, och eftersom brantheten toppade samtidigt som fyllnaden bottnade gav det en λ-spik på 4–5× och ett **prisårsmaximum i mars**, som mätdata motsäger (2023–25 faller monotont från januari i alla fem zoner). Med referensen ligger λ längs banan på λ_bas·A(v), så A styr nivån ett normalår och B styr svaret på att ligga fel. `c_amp = 0` ger bit-identiskt med den gamla formen (7·10⁻¹⁵), så inget äldre påverkas.

⛔ **Geminis nivåer används INTE.** Han vill ha 1,66× spridning (NO-N 39 lägst, NO-S 65 högst); driften vill ha platt 71–76 och om något NO-N högst. Nivåerna speglar observerade zonpriser, dvs. trängsel — som modellen redan producerar endogent ur NTC:erna. Att lägga in dem i λ_bas vore dubbelräkning, och facit (run320) ger 73,02 i alla fem zoner.

⚠️ **Kvarstående brist: FI:s v/s 0,75 mot målet 1,13** (på rätt basis **0,64**).

⛔ **RoR-spåret PRÖVAT OCH TILL STÖRSTA DELEN FALSIFIERAT (run382, 2026-08-19).** Hypotesen var att den syntetiska strömkraftens form bar hela felet. Strömkraften formades om till `(1−α)·inflöde + α·platt` med α = 0,80 anpassat mot Norges **uppmätta** B11 — profilerna blev som avsett (SE-N v/s 0,29 → 0,80, FI 0,43 → 0,84) — men **reservoaren kompenserade bort två tredjedelar**: SE-N:s reservoar gick 1,45 → 1,21 och totalen bara 1,15 → **1,18** mot förutsagda 1,33. FI 0,64 → 0,72 mot förutsagda 0,79.

⭐ **Slutsatsen är en nära bevarandelag: uppdelningen reservoar/strömkraft är nästan irrelevant för den aggregerade v/s-kvoten.** Extra must-run vintereffekt sänkte bara vinterpriset (SE −16,8, FI −20,6 EUR/MWh, sommaren oförändrad) och reservoaren drog sig undan i samma takt. Systemet — last, kurva, NTC — sätter kvoten, inte vilken enhet som levererar. Norge är kontrollgruppen: orört och oförändrat.

⇒ **SE:s och FI:s kvarstående v/s-underskott är alltså ÄKTA, inte en klassificeringsartefakt.** Det bor i kurvan eller i lastformen, och kan inte städas bort i indata. Ändringen behölls ändå på datakvalitetsgrund (α-formen är förenlig med den enda uppmätta strömkraft vi har; poängen 0,91 → 0,80). Den är *inte* kurvans att laga — gradienten kräver `a_amp = −0,71`, alltså ett sommartoppat vattenvärde. FI behöver bara flytta 0,43 TWh/år och magasinet rymmer 5,5, så det är ingen lagringsgräns; FI:s vattenkraft (2,34 GW) är för liten för att vara prissättande i sin egen zon. ⛔ Och det är inte upplösningen: run374 i 1h gav v/s 1,18/1,25/0,64 mot 3h:ns 1,18/1,25/0,68, och prissvansarna fanns redan vid 3h (p99 440–500).

⭐⭐ **x_ref ÄR LÅST TILL TVÅ HARMONISKA (2026-08-20): `config/terminal_curve_2040_gemini_v4.yaml`.** Referensbanan är nu `c_mid + c_amp·cos(2π(v−c_peak)/52) + c_amp2·cos(4π(v−c_peak2)/52)`. En ren kosinus är **symmetrisk** medan magasinbanan har ett **knä** — fyllningen startar abrupt (FI v17, SE v19) och går nästan rakt upp i fem veckor, och höstplatån är bred och sen. Andra harmoniskan halverar felet mot EC:s median 2014-2026: **R²/RMS SE 0,923→0,980 (5,7→2,9 pp) · NO 0,948→0,984 (3,9→2,2) · FI 0,642→0,919 (7,0→3,3)** — störst där den behövs mest. FI:s enkla kosinus missade sommarplatån med 10 pp, vilket är en trolig delförklaring till att FI varit okänsligt för fyra spakar: `b` mätte avvikelse mot en referens som satt fel.

⭐ Basen är **ortogonal** på 52-veckorsrutnätet, så de anpassade förstaharmoniska värdena blir exakt dagens (SE `c_mid` 0,6143 / `c_amp` 0,2800 / `c_peak` 39,3) och `c_amp2 = 0` ger bit-identiskt — inget äldre påverkas. Samma egenskap som `a_amp2` har.

⛔ **`x_ref_weekly` (52 fria värden) finns i koden men är FÖRKASTAD som produktionsform** (run389: exakt beskrivning, RMS 0, men v/s ±0,02, poäng 0,625→0,664, prisform 7,26→8,52). Två harmoniska tar 92-98 % av anpassningen med fem parametrar, förblir en slät funktion och låser inte modellen vid tolv års brus. ⚠️ Lärdomen är egen: **ett sanningsenligare indata garanterar inte en bättre modell.**

⚠️ **v4:s effekt på modellen är ÄNNU OMÄTT** — formen är låst på anpassningskvaliteten, inte på ett körningsutfall. ⚠️ Kvarstående systematisk rest: knäet i maj är för mjukt (residual +5 pp SE, +10 pp FI i v20-22), och EC-bandet är brett just i april (SE 18-34 %) där `b` toppar, så medianen är en tunn beskrivning oavsett funktionsform. ⚠️ EC-medianen finns bara **per land** — SE-N/SE-S delar bana, NO-N/NO-S delar bana.

⭐⭐⭐ **LÖST FÖR SE (run384, 2026-08-19): `config/terminal_curve_2040_aamp03.yaml`.** Identisk med den låsta kurvan så när som på `a_amp`/`a_amp2`, halverade ×0,6 → **×0,3**. SE:s v/s **1,18 → 1,32** (mål 1,38), Σ|fel| 0,64 → 0,50, poäng 0,80 → **0,66**, SE i EC-bandet 54 → **83 %**. Efter tre falsifierade strukturhypoteser (+0,03 / +0,01 / +0,01) var det den uppmätta gradienten d(v/s)/d(a_amp) ≈ −1,7 som hade rätt.

⭐ **Marspuckeln — hela invändningen mot att sänka `a_amp` — slog INTE in.** Medelfel mot uppmätt (v9-12 minus v1-8) **7,6 → 6,4**, bättre i fyra zoner av fem; årsmax ligger kvar i v1-8 utom i NO-N (+0,8 mot uppmätta −1,3, marginellt). Priserna står stilla (±0,20 EUR/MWh i alla zoner), spill 0,0000, drift −0,90 → −1,29 TWh. Och λ längs EC:s medianbana planade ut mot designmålet: kvot SE-N 2,08 → 1,52, FI 2,78 → 1,84 (`temp/plot_value_surface.py` ritar ytan med banan projicerad på λ-axeln).

⚠️ Kvar: **NO överskjuter** (1,35 mot 1,30) — likformig skalning är trubbig, NO hör hemma kring ×0,45. **Budkurvan blev sämre** (MAE 20-60 2,68 → 3,38 GW). ⛔ **FI rör sig inte** (0,72 → 0,74) — fyra spakar prövade utan verkan.

⚠️ `terminal_curve_2040_gemini.yaml` är fortfarande **default och låst**; aamp03 är en separat fil, så run316–383 är oförändrade.

**⚠️ Hydro bids at the HISTORICAL zone price in EXPANSION (water-value proxy, mode-bound since 2026-08-15):** the reservoir StorageUnit's `marginal_cost` is set to that zone's *actual observed day-ahead price* (from `market_prices.parquet`, floored at hydro VOM 0.6) — verified identical to the 2h mean of the historical series in all 13152 snapshots of run260. Hydro's effective bid is therefore

```
historical price[t]  +  mu_energy_balance[t] / efficiency_dispatch
```

Introduced in `947ec81` (run23, 2026-05-18) explicitly as a *"water value proxy"*, replacing a flat VOM, at a time when the model did not yet extract a water value. Thirteen days later `c08906f` (run57) added `assign_all_duals=True` and the genuine endogenous water value — the SOC-balance dual — but the proxy was never removed. They have been stacked ever since; the line has not been touched since run23.

This matters because the endogenous water value is nearly constant (1–6 unique values per zone over three years; SE-S has exactly one), so **all** the time variation in hydro's bid comes from the 2023–25 price series, none from the model. In a 2040 expansion that is circular: hydro's dispatch, and hence the price shape the model produces, is anchored to the price shape it is meant to predict. ⛔ **Sedan 2026-08-20 är proxyn AV även i expansion**, ersatt av `--hydro-mc-curve` (default, se nedan). Den är därmed inte längre nåbar i något default-läge — bara via `--no-hydro-mc-curve` i expansion, som finns för att reproducera run260–391.

- **Varför expansionen behöll den fram till 2026-08-20:** hydrons bud behöver en tidsform, och utan någon alls blir LP:t degenererat (3-års 1h utan proxy OOM-dödades). ⭐ **Kurvan löser det utan cirkularitet** — `λ_bas·A(v)` är inte platt (svänger ±26 % över året) men är slät, och kommer ur Geminis zontabeller och EC:s magasindata i stället för ur prisserien. Med `--hydro-mc-curve` som default är expansionens prisform **inte längre cirkulär**, och den gamla licensinskränkningen ("scenariojämförelser only") gäller bara körningar med `--no-hydro-mc-curve`.
- ⚠️ **Vad kurvan byter bort:** proxyns tim-till-tim-variation, och prisgolvet. Proxyn gick ned till 0,6 i de billigaste timmarna; kurvan bottnar kring 58, så **hydron blir aldrig marginalsättare i lågprislägen**. Väntas synas i prisvaraktighetskurvans nedre ände.
- ⛔ **Ett bart `--hydro-mc-curve` tar INTE den låsta v7-filen.** Den har dispatchens driftkorrigerande ankare (SE-S 64,5 … FI 72,5), som i expansion vore en artificiell zonskillnad i hydrons mc — cykliskt SOC ger noll drift per konstruktion. Både bart flaggnamn och oangiven flagga löses därför till `DEFAULT_HYDRO_MC_CURVE` = v7 med **73 likformigt**. ⚠️ `--terminal-anchor 73` gör *inte* samma sak: den läses bara i rullande horisont (`run_model.py:915`), aldrig på `--hydro-mc-curve`-vägen. Verifiera i `--dry-run`: raden "hydro-mc ur terminalkurvan" ska visa **medel 73,0** i alla fem zoner.
- ⚠️ **`defaults:`-taggen är bumpad till `baseline-v2`.** Replay påverkas inte — `--dispatch` fryser alltid kapaciteter, och kurvan är expansions-bunden — men run260–391 (`baseline-v1`) och run392+ skiljer sig i hydrons mc och ska inte jämföras rakt av.
- **Why dispatch never has it:** the terminal curve *is* the water value. With the proxy on, λ sits at gross price level (~80) while the margin is the net water value, so storage is overvalued and reservoirs hoard — the run91–93 failure.
- **⛔ `--hydro-flat-wv` was deleted with it, as measurably redundant.** Without the proxy `mc = VOM 0.6` and the SOC-balance dual supplies the rest by itself: run320 yields **exactly one water value, 73.02 EUR/MWh, in all five zones and all 13152 hours**. A flat water value therefore arises *endogenously* the moment the proxy is off, and setting its level by hand was measured inert (run273–276: `p_nom_opt` did not move between VOM / 30 / 60, because the dual self-corrects). The flag set a number the model then undid.
- ⚠️ **Cost of the rule:** `run254_noproxy_2h` (an expansion without the proxy) can no longer be reproduced. Its role — measuring the flat full-foresight water value — is now served by **run320, which is a dispatch** and reproduces fine as `--dispatch run260_baseline_2h --no-rolling-horizon --no-terminal-curve`.

**Thermal as must-run Generator:** `p_min_pu = p_max_pu = profile/p_nom`. Dispatch is fully determined by data; optimizer has no freedom. Thermal is NOT subtracted from load.

**Hydro inflow model:** Parameters are manually calibrated spring-flood profiles stored in `config/hydro_params.yaml` (NOT `data/processed/hydro_params.yaml` which is auto-generated and must never be used). SE-N: A=10000 MW spring flood, mu=day 135 (May 15), phi=183 (summer-high cosine). `build_inputs.py` does NOT regenerate these — they are a config artifact. Verify correct hydrology after each run: SE-N inflow should peak ~15000 MW in May, ~2600 MW in January; reservoir SOC should peak ~85% in July.

**Hydro SOC cycling:** `cyclic_state_of_charge=True` + `extra_functionality` callback pins SOC[t=0] = target from `hydro_soc_initial` in `zones.yaml`. This forces start = end = target (e.g. 70%) while the LP optimizes freely in between.

**Hydro operation restrictions (`--hydro-restrictions`, default ON since 2026-08-05, off via `--no-hydro-restrictions`):** Constraints on the *reservoir* StorageUnits (after any RoR split), configured under `hydro_operation` in `zones.yaml` and implemented as an `extra_functionality` callback (`hydro_operation_constraints` in `nordpsa/network.py`):

| Constraint | Form | Default |
|---|---|---|
| `min_hourly_frac` | `p_dispatch[t] ≥ f × p_nom` | 0.05 |
| `min_daily_frac` | `Σ_day p·w ≥ f × p_nom × H_day` | 0.20 |
| `max_weekly_frac` | `Σ_week p·w ≤ f × p_nom × H_week` | 0.77 |
| `bypass_spill` | `Σ_week spill·w ≥ κ × (Σ_week p·w − threshold)` | off |

Purpose: stop the LP from (a) shutting hydro off entirely through long low-price periods (small river reservoirs would overflow) and (b) running at full power week after week — a common ELLI-type artefact. Window sums use the actual `snapshot_weightings`, so they are correct at 1h/2h/3h and partial windows at the series edges are not over-tightened. Constraint names are prefixed `custom-`.

**⚠️ `min_hourly_frac` and `min_daily_frac` are LOCKED (2026-08-14) — do not change without a new measurement.** They were calibrated against hydro's *bid curve* in observed data (production vs price, adelsfors.se June 2026, SE1+SE2), which is a different observable from the seasonal ratio. `min_hourly_frac` 0.10 → **0.05**: at 0.10 the model ran 3.3 GW in SE-N at zero price against ~2.0 observed; at 0.05 it runs 2.78 and the mean error below 10 EUR/MWh falls 1.40 → 0.89 GW. The remainder is *run-of-river* (2.34 GW in June), which this parameter does not control. It costs 0.09 in facit score (0.338 → 0.425) — the hourly floor and the seasonal measure pull in opposite directions, and the observed behaviour won. `min_daily_frac` stays at **0.20**: run341 (0.10) and run342 (0.15) tested and **rejected** — the daily floor does not bind at low prices at all (production below 40 EUR/MWh is identical at 2.78 GW for 0.10/0.15/0.20), it binds in the 40–70 mid-range where reality produces *even more* than the model, so relaxing it worsened the fit (MAE over 20–60 EUR/MWh: 1.96 → 2.21 → 2.28) and drove reservoir drift positive (−1.0 → +2.6 → +3.7 TWh, i.e. hoarding). Neither value comes from Ek Fälth et al. — they are modelling guardrails; only `max_weekly_frac_by_zone` and `bypass_spill` have a source.

With cyclic SOC, annual production equals inflow, so the constraints are mutually consistent only if `min_daily_frac ≤ inflow/(p_nom×H) ≤ max_weekly_frac`. `hydro_operation_bounds()` reports that ratio per zone and `run_model.py` prints it plus warnings before solving. For 2024 the ratio is 0.46–0.55 in all zones, comfortably inside [0.20, 0.77].

Per-zone values come from Ek Fälth et al. (2025) supplementary material (`docs/hydro_restrictions.pdf`), read off the violin plots (±2–3 pp) for *Present regime*, 1 week: Fig A.2 (sustained capacity) SE1 0.77 / SE2 0.87 / SE3 0.77, aggregated capacity-weighted to SE-N 0.83 and SE-S 0.77; Fig A.5 (annual production loss 0.12–0.22 %) calibrates the `bypass_spill` κ to SE-N 0.34 / SE-S 0.59. ⚠️ The study covers **Sweden only** — NO-N/NO-S (0.85) and FI (0.80) are flagged assumptions argued from reservoir hours (NO ~3130 h vs SE-N 2723 h vs SE-S 1514 h). ⚠️ Fig A.4 shows strong seasonality (SE1 ~72 % in March, ~91 % in May–June) that a year-constant cap cannot capture; monthly caps are not implemented. ⚠️ `bypass_spill` is off by default: PyPSA bounds spill by the inflow in the same snapshot, so a high-production/low-inflow week can go infeasible; κ is not given in the source and must be calibrated. Verify with `python scripts/test_hydro_operation.py`.

**p_nom_max bounds:** All extendable generators have finite `p_nom_max_mw` in config (20k for nuclear/gas, 50k for VRE per zone). Without these, HiGHS sees ~3e10 column bounds and prints scaling warnings (harmless but ugly).

## Config

All parameters in `config/zones.yaml`:
- Zone definitions with hydro/nuclear existing capacity
- NTC links between zones
- Market connection capacities
- Technology costs (overnight, lifetime, VOM, extendable flag, p_nom_max)
- Solver settings (HiGHS IPM + crossover)
- Simulation period and resolution
