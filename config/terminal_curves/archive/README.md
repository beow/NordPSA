# Arkiverade terminalkurvor

Aktiva kurvor ligger en nivå upp:

| fil | används av |
|---|---|
| `../terminal_curve_2040_gemini_v12.yaml` | `dispatch.terminal_curve` (rullande horisont) |
| `../terminal_curve_2040_gemini_v12_exp73.yaml` | `expansion.hydro_mc_curve` (λ_bas 73 likformigt) |

Filerna här är tidigare produktionskurvor, kandidater och svepvarianter. De ligger kvar
så att äldre körningar kan läsas och jämföras. Varje fil har ett `note:`-fält med sin
egen historik. Äldre körningar reproduceras från taggen `pre-refactor`, där filerna låg
direkt i `config/`. Den nya koden kan köra en arkiverad kurva med
`--set dispatch.terminal_curve=config/terminal_curves/archive/<fil>`.

## Tidigare produktionskurvor (nyast först)

| fil | ändring mot föregående | reproducerar |
|---|---|---|
| `terminal_curve_2040_gemini_v11.yaml` (+ `_exp73`) | `b_mean` 2,0 / `a_scale` 0,30 | kandidat före v12 |
| `terminal_curve_2040_gemini_v10.yaml` (+ `_exp73`) | medianbanor `x_ref` per zon (00abdf4) | |
| `terminal_curve_2040_gemini_v9.yaml` (+ `_exp73`) | `b_amp` 0 (var 0,27) | run420–438 |
| `terminal_curve_2040_gemini_v8.yaml` (+ `_exp73`) | `b_mean` 2,0 (var 0,80) | run420–432, `_v8`-dispatcherna |
| `terminal_curve_2040_gemini_v7.yaml` (+ `_exp73`) | `b_amp` 0,27 likformigt; låst 2026-08-20 | run316–432 |
| `terminal_curve_2040_gemini_v2.yaml` … `_v6.yaml` | kandidater 2026-08-19/20 (`a_amp` ×0,3, två harmoniska i `x_ref`, `a_scale`) | |
| `terminal_curve_2040_gemini.yaml` | Geminis fem zontabeller; låst 2026-08-18 | run316–390 |
| `terminal_curve_2040_calibrated.yaml` | kalibrerad mot facit run320 | run316–368 |
| `terminal_curve.yaml` | okalibrerad startpunkt (001f87c) | |

## Svep och prov

| filer | vad |
|---|---|
| `terminal_curve_2040_v11_bm40_as000/as065.yaml` | valet av v12 (`b_mean` 4 × `a_scale`) |
| `terminal_curve_2040_v9_*.yaml` | A(v)-formsvepet (`aform`, `aglobal`, `as000/065/129`, `xrefzon`, `bm40_*`), run434 |
| `terminal_curve_2040_v8_bamp0.yaml`, `_v8_bm40_bamp0.yaml` | 2×2 `b_mean` × `b_amp` som gav v9 |
| `terminal_curve_2040_v7_bm20/bm40(_anch).yaml` | `b_mean`-svepet run428–432 som gav v8 |
| `terminal_curve_2040_v7_blf15/blf20.yaml` | `b_low_frac`-prov |
| `terminal_curve_esett_sym/asym/2slope.yaml` | eSett-svepen (källa run360) |
| `terminal_curve_2040_aamp03.yaml` | `a_amp` ×0,3, run384 |
| `terminal_curve_2040_perzone.yaml` | ⛔ förkastad, run385 |
| `terminal_curve_2040_gemini_v3.yaml` | ⛔ förkastad (52 fria `x_ref`-värden), run389 |
| `terminal_curve_2040_state.yaml` | nollhypotes utan kalenderberoende |
| `terminal_curve_flat_amp.yaml` | platt A(v) = 1, prov i expansion |
| `terminal_curve_gemini*.yaml`, `terminal_curve_phase.yaml` | tidiga formprov |
