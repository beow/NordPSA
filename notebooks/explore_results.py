"""
NordPSA — resultatgenomgång.

Spyder-format: kör cell för cell med Ctrl+Enter. Cell 1 måste köras först — den
laddar körningen till modulvariabler (LABEL, ROOT, ZONES, n, dispatch, hydro_d,
soc, spill, flows, prices, water_value, mkt_prices, act_price, load, cfg, dt_h +
hjälparna zone_market/zone_hydro_total/in_zone/by_carrier/twh) som resten av
cellerna använder.

Varje cell KÖR (`exec(_cellcode(...))`, inte import) källan i notebooks/cells/*.py
direkt på cellens EGEN toppnivå — samma källa som notebooks/explore.ipynb och
docs/nordpsa_overview.py (den senare gör en RIKTIG import). ⚠️ Två saker måste
stämma, båda upptäckta 2026-08-21 (se notebooks/explore_results_legacy.py och
minnet project_explore_results_cells_import):
  1. Det måste vara exec, inte `import` — en cells-fil skriven för notebook-
     inklistring förutsätter ETT DELAT namnrum varje gång, men `import` är
     IDEMPOTENT: en omkörning av `import X` efter att X redan finns i
     sys.modules gör INGET (ingen ny utskrift, LABEL uppdateras inte om du
     ändrat den och kör om cell 1).
  2. exec-anropet måste stå DIREKT på cellens toppnivå, INTE inuti en
     mellanliggande hjälpfunktion som själv gör `exec(kod, globals())` — en sådan
     funktions `globals()` är FRUSEN vid dess def-tillfälle och behöver inte
     matcha det namnrum Spyders %runcell faktiskt använder för den aktuella
     cellkörningen, vilket tyst tappade LABEL. `_cellcode(name)` gör bara
     läsning+kompilering (inga namnrum inblandade); `exec(_cellcode(name))` utan
     uttryckliga namnrumsargument använder alltid den ANROPANDE kodens FAKTISKA
     aktuella ram — exakt notebook-cellens semantik.

Ändra LABEL/LABEL2 genom att sätta dem i cell 1 FÖRE `exec(_cellcode("bootstrap"))`
(den läser globals().get(...) först och rör inte bootstrap.py:s egen default),
eller genom att redigera bootstrap.py direkt. Kör om cell 1 för att byta körning.

Tunbara inställningar som ligger INUTI en cells-fil (t.ex. prodstack.py:s
START/END/FREQ/ZONE, compareprice.py:s ZONE/DATE_RANGE) redigeras i den filen —
de är hårda defaults, inte globals().get(...)-styrda, så de kan inte sättas
härifrån.

Föregångaren till den här filen (all logik inline, ingen cells/-import) finns
sparad oförändrad i notebooks/explore_results_legacy.py. Tre av cellerna (f.d.
cell 4/6/7) saknade motsvarighet i cells/ innan de bröts ut 2026-08-21:
priceformation.py, flexassets.py, trading.py.

Priserna genomgående är RÅA (2026-08-21: hydrocomp.py/calw-kompensationen togs
bort ur den här pipen — cellen finns kvar i cells/ men körs inte här, och
pricetable/prisgrafer/compareprice rensades från calw-beroendet). Kompensationen
var bara giltig när vattenvärdet är ett lågt/platt/icke-bindande golv, se
hydrocomp.py:s egen docstring.

Cellerna:
  1  Ladda körning                                   → bootstrap.py
  2  Energibalans per kraftslag och zon               → ebalance.py
  3  Prisvalidering                                    → pricetable.py + prisgrafer.py
  4  Prisbildning: vad sätter priset, NTC-koppling     → priceformation.py
  5  Hydrologi                                          → soclevels.py + hydrodispatch.py
  6  Flexibilitet: batteri, vätgas, värme, EV           → flexassets.py
  7  Handel: intern NTC och kontinentventilen           → trading.py
  8  Produktionsstapel för valt utsnitt                 → prodstack.py
  9  Jämför två körningar (valfri)                      → compareprice.py
 10  Marginal källa vid EN given timme                  → marginal.py
 10b Utbudskurva: hur prisön når sitt marginalpris     → bidstack.py
 11  Systemkostnad på REAL basis (trappkorrigerad)       → objcost.py
"""

# %% 1 — Ladda körning
import sys
from pathlib import Path

# Hitta repo-roten oavsett var Spyder står
ROOT = Path.cwd()
while not (ROOT / "config" / "zones.yaml").is_file() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
CELLS = ROOT / "notebooks" / "cells"
sys.path.insert(0, str(CELLS))


def _cellcode(name):
    """Kompilerar notebooks/cells/<name>.py — INGEN namnrums-koppling här (bara
    läsning+kompilering), så den kan tryggt anropas från ett eget uttryck.
    ⚠️ Anropa `exec(_cellcode(...))` DIREKT i varje cell — INTE via en mellanliggande
    funktion. En funktion som gör `exec(kod, globals())` INUTI sig själv kör koden i
    FUNKTIONENS EGET definitionsnamnrum (`__globals__`, fastfruset vid def-tillfället),
    inte nödvändigtvis i det namnrum Spyders %runcell råkar använda för just DEN här
    cellkörningen — det gav tyst fel LABEL (se explore_results_legacy.py-historiken/
    minnet). `exec(kod)` utan uttryckliga namnrum, anropat DIREKT på cellens egen
    toppnivå, använder alltid den ANROPANDE kodens FAKTISKA aktuella ramnamnrum —
    exakt notebook-cellens semantik, oavsett hur Spyder internt organiserar globals/
    locals."""
    path = CELLS / f"{name}.py"
    return compile(path.read_text(encoding="utf-8"), str(path), "exec")


# Sätt LABEL/LABEL2 HÄR för att byta körning — kör om cell 1 efteråt.
LABEL, LABEL2 =  "run451_senuc15exo_dispatch_1h","run191_senuc15exo_3h"
LABEL, LABEL2 =  "run451_senuc15exo_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run452_se_disc3_dispatch_1h","run192_se_disc3_3h"
LABEL, LABEL2 =  "run452_se_disc3_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run453_onshore80_dispatch_1h","run193_onshore80_3h"
LABEL, LABEL2 =  "run453_onshore80_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run450_baseline_dispatch_1h_scarcity_g3","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run450_baseline_dispatch_1h_scarcity_g3","run197_baseline_exp_1h"
LABEL, LABEL2 =  "run450_baseline_dispatch_1h_scarcity_g8","run197_baseline_exp_1h"
LABEL, LABEL2 =  "run197_baseline_exp_1h", None
LABEL, LABEL2 =  "run450_baseline_dispatch_1h","run197_baseline_exp_1h"
LABEL, LABEL2 =  "run455_batt25_4h_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run456_notax_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run457_market50_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run460_baseline_dispatch_1h","run450_baseline_dispatch_1h"
LABEL, LABEL2 =  "run461_senuc15exo_dispatch_1h","run460_baseline_dispatch_1h"
LABEL, LABEL2 =  "run462_se_disc3_dispatch_1h","run460_baseline_dispatch_1h"

exec(_cellcode("bootstrap"))


# %% 2 — Energibalans per kraftslag och zon
exec(_cellcode("ebalance"))  # auto-printar tabellen; country_balance() finns för vidare bruk


# %% 3 — Prisvalidering
exec(_cellcode("pricetable"))   # nyckeltal per zon: bias/MAE/RMSE/korr + landaggregat
exec(_cellcode("prisgrafer"))   # modellpris vs referens, veckomedel, 2×3


# %% 4 — Prisbildning: vad sätter priset, och hur mycket är NTC-kopplat
exec(_cellcode("priceformation"))  # ändra ZONE i cells/priceformation.py

# %%
# %% 5 — Hydrologi
exec(_cellcode("soclevels"))      # magasinsfyllnad per hydrozon
exec(_cellcode("hydrodispatch"))  # vattenkraftsdispatch, månadsmedel


# # %% 6 — Flexibilitet: batteri, vätgas, värme, EV
# exec(_cellcode("flexassets"))


# # %% 7 — Handel: intern NTC och kontinentventilen
# exec(_cellcode("trading"))


# %% 8 — Produktionsstapel för valt utsnitt
exec(_cellcode("prodstack"))  # ändra START/END/FREQ/ZONE i cells/prodstack.py


# %% 9 — Jämför mot en annan körning (valfri, kräver LABEL2)
exec(_cellcode("compareprice"))  # ändra ZONE/DATE_RANGE i cells/compareprice.py


# %% 10 — Marginal källa vid EN given timme (prisöar + trängselkaskad)
MARGINAL_TS = "2024-12-12 16:00"   # timme att analysera (None = bara definiera funktionen)
exec(_cellcode("marginal"))        # kör MARGINAL_TS; marginal_source("...") för fler timmar


# %% 10b — Utbudskurva: HUR prisön når sitt marginalpris vid samma timme
BIDSTACK_TS   = "2024-01-06 10:00"   # timme att rita (None = bara definiera bid_stack())
BIDSTACK_ZONE = 'SE-S'               # zon; kurvan ritas för HELA dess prisö
exec(_cellcode("bidstack"))          # bid_stack("2024-12-12 10:00", 'FI') för fler


# # %% 11 — Systemkostnad på REAL basis (trappkorrigerad)
# exec(_cellcode("objcost"))            # auto-kör cost_table() på LABEL (+ LABEL2)
# # cost_table("run417_baseline_noladder_2h", "run418_ladder_k3_2h")   # valfri jämförelse
