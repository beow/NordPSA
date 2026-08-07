"""
NordPSA — resultatgenomgång.

Spyder-format: kör cell för cell med Ctrl+Enter. Cell 1 måste köras först,
den laddar körningen till modulvariabler som resten av cellerna använder.

Byt körning genom att ändra LABEL i cell 1.

Anpassad från IberPSA:s explore_results.py. Skillnaderna mot den: sex zoner i
stället för en, ingen pumpkraft (men batteri/H2/värme/EV-lager), intern NTC-handel
mellan zonerna vid sidan av kontinentventilen, och ingen "facit"-kolumn per
kraftslag — 2040-körningar har inget utfall att jämföras mot. Faktiska spotpriser
finns däremot för alla sex zoner och används i cell 3.

Cellerna:
  1  Ladda körning
  2  Energibalans per kraftslag och zon
  3  Prisvalidering — varaktighet, månadsmedel, spridning
  4  Prisbildning — vad sätter priset, och hur mycket är NTC-kopplat
  5  Hydrologi — tillrinning, magasinsnivå, spill
  6  Flexibilitet — batteri, vätgas, värme, EV
  7  Handel — intern NTC och kontinentventilen
  8  Produktionsstapel för valt utsnitt
  9  Jämför två körningar (valfri)
 10  Marginal källa vid EN given timme — prisöar och trängselkaskad
"""

# %% 1 — Ladda körning
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Hitta repo-roten oavsett var Spyder står
ROOT = Path.cwd()
while not (ROOT / "config" / "zones.yaml").is_file() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pypsa  # noqa: E402

# ---------------------------------------------------------------------------
LABEL = "run254_noproxy_2h"    # <-- byt körning här
LABEL2 = None #"run240_baseline_2h"                 # <-- jämförelsekörning för cell 9 (None = hoppa över)
# ---------------------------------------------------------------------------

RES = ROOT / "results" / LABEL
PROC = ROOT / "data" / "processed"
ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]

plt.rcParams["figure.figsize"] = (15, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

cfg = yaml.safe_load(open(ROOT / "config" / "zones.yaml"))
vre_pnom = yaml.safe_load(open(PROC / "vre_pnom.yaml"))

n = pypsa.Network(str(RES / "network.nc"))


def _naive(df):
    """Tidsindex utan tidszon — CSV-resultaten är naiva, nätverkets är det inte alltid."""
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def _csv(name):
    p = RES / f"{name}.csv"
    return _naive(pd.read_csv(p, index_col=0)) if p.exists() else None


# Resultaten läses ur CSV (samma konvention som notebooks/cells/bootstrap.py);
# nätverket används för metadata (carrier, bus, p_nom) och tillrinning.
gen = _csv("dispatch_generators")
stor = _csv("dispatch_hydro")          # ALLA storage units, även batteri
soc = _csv("hydro_soc")
spill = _csv("hydro_spill")
flows = _csv("flows")                  # länkflöden, p0
prices = _csv("prices")
water_value = _csv("water_value")      # dual på lagringsbalansen = vattenvärde

snap = gen.index
dt_h = float(n.snapshot_weightings.objective.iloc[0])
carriers = n.generators["carrier"]
price = prices[ZONES]

inflow = _naive(n.storage_units_t.inflow).reindex(snap) \
    if not n.storage_units_t.inflow.empty else pd.DataFrame(index=snap)
p_max_pu = _naive(n.generators_t.p_max_pu).reindex(snap)

# Faktiska spotpriser per zon (facit för dispatchkörningar, referens för 2040-scenarier)
act_price = _naive(pd.read_parquet(PROC / "market_prices.parquet")).reindex(snap, method="ffill")

# Last ur nätverket (inkl. --extra-load och sektorlaster)
_loads = _naive(n.loads_t.p_set).reindex(snap)
load = pd.DataFrame({z: _loads[f"{z} load"] for z in ZONES if f"{z} load" in _loads.columns})


def twh(x):
    """MW-serie → TWh över körperioden."""
    return float((x * dt_h).sum()) / 1e6


def in_zone(carrier, zones=None):
    """Generatornamn med given carrier i angivna zoner (default alla sex elzoner)."""
    zs = ZONES if zones is None else zones
    return [g for g in gen.columns if g in n.generators.index
            and n.generators.at[g, "bus"] in zs
            and n.generators.at[g, "carrier"] == carrier]


def by_carrier(zones=None, df=None):
    """Summerar generatordispatch per carrier-ATTRIBUT (inte per namn)."""
    d = gen if df is None else df
    cols = [g for g in d.columns if g in n.generators.index
            and n.generators.at[g, "bus"] in (ZONES if zones is None else zones)]
    return d[cols].T.groupby(carriers[cols]).sum().T


def zone_market(zone):
    """Nettoflöde genom kontinentventilen för en zon (+ = import)."""
    cols = in_zone("market", [zone])
    return gen[cols].sum(axis=1) if cols else pd.Series(0.0, index=snap)


def zone_hydro_total(zone):
    """Total vattenkraft per zon = magasin (StorageUnit) + RoR (must-run Generator)."""
    res = stor[f"{zone} hydro"].clip(lower=0) if f"{zone} hydro" in stor.columns \
        else pd.Series(0.0, index=snap)
    ror = in_zone("hydro", [zone])
    return res.add(gen[ror].clip(lower=0).sum(axis=1), fill_value=0.0) if ror else res


print(f"Körning : {LABEL}")
print(f"Period  : {snap[0]} – {snap[-1]}  ({len(snap)} snapshots à {dt_h:.0f} h)")
print(f"Zoner   : {ZONES}")
print(f"Modell  : {len(n.generators)} generatorer, {len(n.storage_units)} lager, "
      f"{len(n.stores)} stores, {len(n.links)} länkar")
print(f"\n{'zon':6s} {'pris':>7} {'faktiskt':>9} {'last TWh':>9} {'nettoimport TWh':>16}")
for z in ZONES:
    print(f"{z:6s} {price[z].mean():>7.1f} {act_price[z].mean():>9.1f} "
          f"{twh(load[z]):>9.1f} {twh(zone_market(z)):>16.2f}")
_slack = twh(gen[in_zone("slack")].clip(lower=0).sum(axis=1))
print(f"\nLastbortkoppling (el-slack): {_slack:.4f} TWh   (bör vara ≈ 0)")


# %% 2 — Energibalans per kraftslag och zon
rows = []
bc = by_carrier()
for carrier in bc.columns:
    if carrier in ("slack",) or bc[carrier].abs().sum() == 0:
        continue
    per_zone = {z: twh(gen[in_zone(carrier, [z])].clip(lower=0).sum(axis=1))
                for z in ZONES}
    if carrier == "market":                       # ventilen kan gå åt båda hållen
        per_zone = {z: twh(zone_market(z)) for z in ZONES}
    # carrier 'hydro' på en Generator = run-of-river (must-run); magasinet är StorageUnit
    namn = "hydro_ror (must-run)" if carrier == "hydro" else carrier
    rows.append({"kraftslag": namn, **per_zone,
                 "TOTALT": sum(per_zone.values())})

# Magasinskraft (StorageUnit) redovisas separat från RoR (som ligger som generator)
res_units = [u for u in stor.columns if u in n.storage_units.index
             and n.storage_units.at[u, "carrier"] == "hydro"]
batt_units = [u for u in stor.columns if u in n.storage_units.index
              and n.storage_units.at[u, "carrier"] == "battery"]
for namn, units, sign in (("hydro reservoar", res_units, 1),
                          ("batteri (ut)", batt_units, 1),
                          ("batteri (in)", batt_units, -1)):
    if not units:
        continue
    per_zone = {}
    for z in ZONES:
        u = [x for x in units if n.storage_units.at[x, "bus"] == z]
        s = stor[u].sum(axis=1) if u else pd.Series(0.0, index=snap)
        per_zone[z] = twh(s.clip(lower=0)) if sign > 0 else -twh((-s).clip(lower=0))
    rows.append({"kraftslag": namn, **per_zone, "TOTALT": sum(per_zone.values())})

# Curtailment: VRE-potential minus dispatchad VRE
curt = {}
for z in ZONES:
    c = 0.0
    for carrier in ("wind_onshore", "wind_offshore", "solar"):
        for g in in_zone(carrier, [z]):
            if g in p_max_pu.columns:
                pot = p_max_pu[g] * n.generators.at[g, "p_nom"]
                c += twh((pot - gen[g]).clip(lower=0))
    curt[z] = -c
rows.append({"kraftslag": "curtailment (VRE)", **curt, "TOTALT": sum(curt.values())})

lastrad = {z: -twh(load[z]) for z in ZONES}
rows.append({"kraftslag": "ellast (baslast)", **lastrad, "TOTALT": sum(lastrad.values())})

eb = pd.DataFrame(rows).set_index("kraftslag")
eb = eb.reindex(eb["TOTALT"].abs().sort_values(ascending=False).index)
print(f"Energibalans, TWh över {len(snap)*dt_h/8760:.1f} år "
      f"(+ = produktion/import, − = last/export)\n")
print(eb.round(2).to_string())
print("\nSektorlaster (utöver baslast), TWh: "
      + ", ".join(f"{s} {twh(_loads[[c for c in _loads.columns if s in c]].sum(axis=1)):.1f}"
                  for s in ("H2", "heat", "EV") if any(s in c for c in _loads.columns)))
print(f"Spill (hydro): {twh(spill.sum(axis=1)):.3f} TWh")


# %% 3 — Prisvalidering
ZONE = "SE-S"      # <-- zon att granska

p_m, p_a = price[ZONE], act_price[ZONE]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

ax = axes[0]
ax.plot(np.linspace(0, 100, len(p_m)), np.sort(p_m)[::-1], label="Modell", color="#1f78b4")
ax.plot(np.linspace(0, 100, len(p_a.dropna())), np.sort(p_a.dropna())[::-1],
        label=f"Faktiskt {ZONE}", color="#d62828")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("% av timmarna"); ax.set_ylabel("EUR/MWh")
ax.set_title("Prisvaraktighet"); ax.legend()

ax = axes[1]
ax.plot(p_m.resample("MS").mean(), marker="o", label="Modell", color="#1f78b4")
ax.plot(p_a.resample("MS").mean(), marker="s", label="Faktiskt", color="#d62828")
ax.set_ylabel("EUR/MWh"); ax.set_title("Månadsmedel"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

ax = axes[2]
ax.scatter(p_a, p_m, s=2, alpha=0.15, color="#1f78b4")
lims = [min(p_a.min(), p_m.min()), max(p_a.max(), p_m.max())]
ax.plot(lims, lims, color="black", lw=0.8, ls="--")
ax.set_xlabel(f"Faktiskt {ZONE}"); ax.set_ylabel("Modell")
ax.set_title(f"r = {p_m.corr(p_a):.3f}")
plt.show()

print("OBS: 'faktiskt' är utfallet 2023–2025. Meningsfullt som facit bara för "
      "dispatchkörningar\nmot dagens system — för 2040-scenarier är det en referensnivå, "
      "inte ett fel.\n")
print(f"{'zon':6s} {'medel':>8} {'faktiskt':>9} {'bias':>7} {'MAE':>7} {'SD':>7} "
      f"{'SD fakt':>8} {'<0 %':>7}")
for z in ZONES:
    m, a_ = price[z], act_price[z]
    print(f"{z:6s} {m.mean():>8.1f} {a_.mean():>9.1f} {(m-a_).mean():>+7.1f} "
          f"{abs(m-a_).mean():>7.1f} {m.std():>7.1f} {a_.std():>8.1f} "
          f"{100*(m<0).mean():>7.1f}")
print(f"\n{ZONE} kvantiler:")
print(f"{'':14s} {'modell':>9} {'faktiskt':>9}")
for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
    print(f"  q{q:<11.2f} {p_m.quantile(q):>9.1f} {p_a.quantile(q):>9.1f}")


# %% 4 — Prisbildning: vad sätter priset, och hur mycket är NTC-kopplat
# Letar per timme efter den enhet som är PÅ MARGINALEN (dispatch strikt mellan
# sina gränser) och vars marginalkostnad ligger närmast zonpriset. För lager är
# budet lagrets EGEN marginalkostnad + vattenvärdet/verkningsgraden:
#   - vattenvärdet = dualen på lagringsbalansen (water_value.csv)
#   - vattenkraftens marginal_cost är zonens FAKTISKA historiska pris (se
#     "water-value proxy" i CLAUDE.md) och står för hela tidsvariationen
#   - /efficiency_dispatch: 1 MWh till nätet kostar 1/eff MWh ur lagret (batteri 0.95)
# Med enbart vattenvärdet träffade hydro priset i ~1 % av timmarna, med hela
# budet i 64-76 % (NO-N/NO-S).
ZONE = "SE-N"      # <-- zon att granska

mc_t = _naive(n.generators_t.marginal_cost).reindex(snap) \
    if not n.generators_t.marginal_cost.empty else pd.DataFrame(index=snap)

cand = {}
for g in gen.columns:
    if g not in n.generators.index or n.generators.at[g, "bus"] != ZONE:
        continue
    if g in mc_t.columns:
        cand[g] = mc_t[g]
    elif n.generators.at[g, "carrier"] in ("gas", "slack", "nuclear", "thermal"):
        cand[g] = pd.Series(n.generators.at[g, "marginal_cost"], index=snap)
if water_value is not None:
    smc_t = _naive(n.storage_units_t.marginal_cost).reindex(snap) \
        if not n.storage_units_t.marginal_cost.empty else pd.DataFrame(index=snap)
    for u in water_value.columns:
        if u in n.storage_units.index and n.storage_units.at[u, "bus"] == ZONE:
            base = smc_t[u] if u in smc_t.columns else \
                pd.Series(n.storage_units.at[u, "marginal_cost"], index=snap)
            eff = n.storage_units.at[u, "efficiency_dispatch"]
            cand[u] = base + water_value[u].reindex(snap) / eff

marg = pd.Series("okänd", index=snap, dtype=object)
best = pd.Series(np.inf, index=snap)
for name, mc in cand.items():
    if name in gen.columns:
        p_, p_nom = gen[name], n.generators.at[name, "p_nom"]
        on_margin = (p_ > 1) & (p_ < p_nom * 0.999)
        carrier = n.generators.at[name, "carrier"]
    else:
        p_, p_nom = stor[name], n.storage_units.at[name, "p_nom"]
        on_margin = (p_.abs() > 1) & (p_ < p_nom * 0.999)
        carrier = n.storage_units.at[name, "carrier"]
    d = (price[ZONE] - mc.reindex(snap)).abs()
    hit = on_margin & (d < best)
    marg[hit] = carrier
    best[hit] = d[hit]

# NTC-koppling: identiskt pris med en granne = priset sätts där, inte lokalt.
# Timmar utan lokal marginalenhet OCH med prislikhet mot en granne bokförs som
# NTC-kopplade i stället för 'okänd' — annars ser hälften av timmarna oförklarade ut.
partners = sorted({z for lnk in n.links.index if n.links.at[lnk, "carrier"] == "AC"
                   for z in (n.links.at[lnk, "bus0"], n.links.at[lnk, "bus1"])
                   if z in ZONES
                   and ZONE in (n.links.at[lnk, "bus0"], n.links.at[lnk, "bus1"])}
                  - {ZONE})
kopplad = pd.Series(False, index=snap)
for z in partners:
    kopplad |= (price[ZONE] - price[z]).abs() < 0.01
marg[(marg == "okänd") & kopplad] = "NTC-kopplad"

print(f"{ZONE}: prissättande teknik, % av timmarna")
print(marg.value_counts(normalize=True).mul(100).round(1).to_string())
print(f"\nNTC-koppling mot {partners}:")
print(f"  identiskt pris med minst en granne : {100*kopplad.mean():5.1f} % av timmarna")
print(f"  lokalt prissatt (trängsel åt alla håll) : {100*(~kopplad).mean():5.1f} %")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
ax = axes[0]
for z in ZONES:
    ax.hist(price[z], bins=80, histtype="step", label=z, lw=1.1)
ax.set_xlabel("EUR/MWh"); ax.set_ylabel("antal timmar")
ax.set_title("Prisfördelning per zon"); ax.legend(fontsize=8)

ax = axes[1]
win = max(int(24 * 7 / dt_h), 1)
if water_value is not None:
    for u in [c for c in water_value.columns if c.endswith("hydro")]:
        ax.plot(snap, water_value[u].rolling(win, min_periods=1).mean(), lw=1.0, label=u)
ax.plot(snap, price[ZONE].rolling(win, min_periods=1).mean(), lw=1.4, color="black",
        label=f"{ZONE} pris")
ax.set_ylabel("EUR/MWh"); ax.set_title("Vattenvärde vs pris (veckomedel)")
ax.legend(fontsize=8)
plt.show()


# %% 5 — Hydrologi
hyd = [u for u in soc.columns if u in n.storage_units.index
       and n.storage_units.at[u, "carrier"] == "hydro"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
ax = axes[0]
if not inflow.empty:
    monthly = inflow[hyd].sum(axis=1).groupby(inflow.index.month).mean()
    ax.bar(monthly.index, monthly.values / 1e3, color="#1f78b4")
ax.set_xlabel("månad"); ax.set_ylabel("GW"); ax.set_xticks(range(1, 13))
ax.set_title("Medeltillrinning per månad\n(nordiskt: vårflod maj–juni)")

ax = axes[1]
for u in hyd:
    cap = n.storage_units.at[u, "p_nom"] * n.storage_units.at[u, "max_hours"]
    ax.plot(snap, 100 * soc[u] / cap, lw=0.8, label=u.replace(" hydro", ""))
ax.set_ylabel("% av volym"); ax.set_title("Magasinsfyllnad per zon"); ax.legend(fontsize=8)

ax = axes[2]
tot_soc = soc[hyd].sum(axis=1) / 1e6
ax.plot(snap, tot_soc, color="#1f78b4")
ax.set_ylabel("TWh"); ax.set_title(f"Samlat magasin (start {tot_soc.iloc[0]:.1f} → "
                                   f"slut {tot_soc.iloc[-1]:.1f} TWh)")
plt.show()

print(f"{'lager':14s} {'tillrinning':>12} {'produktion':>11} {'spill':>8} "
      f"{'SOC min':>8} {'SOC max':>8}")
for u in hyd:
    cap = n.storage_units.at[u, "p_nom"] * n.storage_units.at[u, "max_hours"]
    inf = twh(inflow[u]) if u in inflow.columns else np.nan
    sp = twh(spill[u]) if u in spill.columns else 0.0
    print(f"{u:14s} {inf:>12.2f} {twh(stor[u].clip(lower=0)):>11.2f} {sp:>8.3f} "
          f"{abs(100*soc[u].min()/cap):>7.1f}% {100*soc[u].max()/cap:>7.1f}%")
print(f"\nRun-of-river (must-run generator): "
      f"{twh(gen[in_zone('hydro')].clip(lower=0).sum(axis=1)):.2f} TWh")


# %% 6 — Flexibilitet: batteri, vätgas, värme, EV
store_soc = (_naive(n.stores_t.e).reindex(snap) if not n.stores_t.e.empty
             else pd.DataFrame(index=snap))
link_p = (_naive(n.links_t.p0).reindex(snap) if not n.links_t.p0.empty
          else pd.DataFrame(index=snap))

_ar = len(snap) * dt_h / 8760
print(f"{'lager':18s} {'volym GWh':>10} {'urladdat TWh':>13} {'cykler/år':>10} {'spann %':>8}")
for u in [x for x in stor.columns if x in n.storage_units.index
          and n.storage_units.at[x, "carrier"] == "battery"]:
    vol = n.storage_units.at[u, "p_nom"] * n.storage_units.at[u, "max_hours"]
    ut = twh(stor[u].clip(lower=0))
    cyk = (ut * 1e6 / vol) / _ar if vol else np.nan
    sv = 100 * (soc[u].max() - soc[u].min()) / vol if vol and u in soc.columns else np.nan
    print(f"{u:18s} {vol/1e3:>10.1f} {ut:>13.2f} {cyk:>10.1f} {sv:>8.0f}")
for s in [x for x in store_soc.columns if x in n.stores.index
          and n.stores.at[x, "carrier"] in ("H2 store", "EV battery", "heat store")]:
    vol = n.stores.at[s, "e_nom"]
    if vol <= 0:
        print(f"{s:18s} {0.0:>10.1f} {'—':>13s} {'—':>10s} {'—':>8s}")
        continue
    sv = 100 * (store_soc[s].max() - store_soc[s].min()) / vol
    # Stores har ingen effektriktning i e; omsatt energi ≈ summan av positiva ändringar
    oms = float(store_soc[s].diff().clip(lower=0).sum()) / 1e6
    print(f"{s:18s} {vol/1e3:>10.1f} {oms:>13.2f} {(oms*1e6/vol)/_ar:>10.1f} {sv:>8.0f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
ax = axes[0]
batt = [u for u in stor.columns if u in n.storage_units.index
        and n.storage_units.at[u, "carrier"] == "battery"]
if batt:
    h = snap.hour
    ax.plot(range(24), stor[batt].sum(axis=1).groupby(h).mean().reindex(range(24)).values,
            color="#1f78b4")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("timme"); ax.set_ylabel("MW (+ = urladdning)")
ax.set_title("Batteriets dygnsmönster")

ax = axes[1]
for s in [x for x in store_soc.columns if x.endswith(("H2 store", "heat store"))][:6]:
    ax.plot(snap, store_soc[s] / 1e3, lw=0.7, label=s)
ax.set_ylabel("GWh"); ax.set_title("Sektorlagrens SOC"); ax.legend(fontsize=7)

ax = axes[2]
elyser = [lnk for lnk in link_p.columns if lnk in n.links.index
          and n.links.at[lnk, "carrier"] == "electrolyser"]
if elyser:
    z0 = n.links.at[elyser[0], "bus0"]
    ax.scatter(price[z0], link_p[elyser[0]], s=2, alpha=0.2, color="#1f78b4")
    ax.set_xlabel(f"{z0} pris EUR/MWh"); ax.set_ylabel("elektrolys MW")
    ax.set_title("Elektrolys vs pris")
plt.show()


# %% 7 — Handel: intern NTC och kontinentventilen
ac = [lnk for lnk in n.links.index if n.links.at[lnk, "carrier"] == "AC"]
win = max(int(24 * 7 / dt_h), 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), constrained_layout=True)
ax = axes[0]
for lnk in ac:
    if lnk in flows.columns:
        ax.plot(snap, flows[lnk].rolling(win, min_periods=1).mean(), lw=0.9, label=lnk)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("MW"); ax.set_title("Interna NTC-flöden, veckomedel"); ax.legend(fontsize=7)

ax = axes[1]
net = pd.DataFrame({z: zone_market(z) for z in ZONES})
ax.plot(snap, net.sum(axis=1).rolling(win, min_periods=1).mean(), color="#e07a5f",
        label="Norden netto")
for z in ZONES:
    if net[z].abs().sum() > 0:
        ax.plot(snap, net[z].rolling(win, min_periods=1).mean(), lw=0.8, label=z)
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("MW (+ = import)"); ax.set_title("Kontinentventilen, veckomedel")
ax.legend(fontsize=7)
plt.show()

print(f"{'länk':14s} {'p_nom MW':>9} {'netto TWh':>10} {'trängsel %':>11} {'prisdiff':>9}")
for lnk in ac:
    if lnk not in flows.columns:
        continue
    p_nom = n.links.at[lnk, "p_nom"]
    z0, z1 = n.links.at[lnk, "bus0"], n.links.at[lnk, "bus1"]
    trang = 100 * (flows[lnk].abs() > p_nom * 0.99).mean()
    diff = (price[z0] - price[z1]).abs().mean() if z0 in price and z1 in price else np.nan
    print(f"{lnk:14s} {p_nom:>9.0f} {twh(flows[lnk]):>10.2f} {trang:>10.1f}% {diff:>9.1f}")

print(f"\n{'zon':6s} {'import TWh':>11} {'export TWh':>11} {'netto TWh':>10}")
for z in ZONES:
    s = net[z]
    print(f"{z:6s} {twh(s.clip(lower=0)):>11.2f} {twh(-s.clip(upper=0)):>11.2f} "
          f"{twh(s):>10.2f}")


# %% 8 — Produktionsstapel för valt utsnitt
START, DAYS = "2024-01-15", 14      # <-- ändra utsnitt här
ZONE = None                        # None = hela Norden, annars t.ex. "SE-S"

STACK = ["nuclear", "thermal", "hydro_ror", "solar", "wind_onshore", "wind_offshore",
         "hydro", "battery", "gas", "import", "slack"]
COLORS = {"nuclear": "#c94f7c", "thermal": "#8c6d46", "hydro_ror": "#4fc3f7",
          "solar": "#f9c74f", "wind_onshore": "#90be6d", "wind_offshore": "#43aa8b",
          "hydro": "#1f78b4", "battery": "#a6cee3", "gas": "#b0b0b0",
          "import": "#e07a5f", "slack": "#d62828"}

_zs = ZONES if ZONE is None else [ZONE]
t0 = pd.Timestamp(START)
t1 = t0 + pd.Timedelta(days=DAYS)

prod = {}
for carrier in ("nuclear", "thermal", "solar", "wind_onshore", "wind_offshore", "gas", "slack"):
    cols = in_zone(carrier, _zs)
    if cols:
        prod[carrier] = gen[cols].clip(lower=0).sum(axis=1)
prod["hydro_ror"] = gen[in_zone("hydro", _zs)].clip(lower=0).sum(axis=1) \
    if in_zone("hydro", _zs) else pd.Series(0.0, index=snap)
prod["hydro"] = sum((stor[f"{z} hydro"].clip(lower=0) for z in _zs
                     if f"{z} hydro" in stor.columns), pd.Series(0.0, index=snap))
_batt = [u for u in stor.columns if u in n.storage_units.index
         and n.storage_units.at[u, "carrier"] == "battery"
         and n.storage_units.at[u, "bus"] in _zs]
prod["battery"] = stor[_batt].sum(axis=1).clip(lower=0) if _batt else pd.Series(0.0, index=snap)
prod["import"] = sum((zone_market(z).clip(lower=0) for z in _zs), pd.Series(0.0, index=snap))
prod = pd.DataFrame(prod).loc[t0:t1]

# Lastlinjen: baslast + export + batteriladdning (allt som stapeln ska mötas av)
export = sum(((-zone_market(z)).clip(lower=0) for z in _zs), pd.Series(0.0, index=snap))
batt_chg = (-stor[_batt].sum(axis=1)).clip(lower=0) if _batt else pd.Series(0.0, index=snap)
lastlinje = (load[_zs].sum(axis=1) + export + batt_chg).loc[t0:t1]

order = [c for c in STACK if c in prod.columns and prod[c].abs().sum() > 0]
fig, (ax, axp) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                              height_ratios=[3, 1], constrained_layout=True)
ax.stackplot(prod.index, [prod[c] / 1000 for c in order], labels=order,
             colors=[COLORS.get(c, "#ccc") for c in order])
ax.plot(lastlinje.index, lastlinje / 1000, color="black", lw=1.4,
        label="Last + export + batteriladdning")
ax.set_ylabel("GW"); ax.set_title(f"{'Norden' if ZONE is None else ZONE} — {LABEL}")
ax.legend(loc="upper left", ncol=5, fontsize=8); ax.margins(x=0)

for z in _zs:
    axp.plot(price.loc[t0:t1, z], lw=1.0, label=z)
axp.axhline(0, color="black", lw=0.5)
axp.set_ylabel("EUR/MWh"); axp.legend(fontsize=8, ncol=6); axp.margins(x=0)
plt.show()


# %% 9 — Jämför mot en annan körning (valfri)
if LABEL2:
    RES2 = ROOT / "results" / LABEL2
    price2 = _naive(pd.read_csv(RES2 / "prices.csv", index_col=0))[ZONES]
    gen2 = _naive(pd.read_csv(RES2 / "dispatch_generators.csv", index_col=0))
    n2 = pypsa.Network(str(RES2 / "network.nc"))
    dt2 = float(n2.snapshot_weightings.objective.iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), constrained_layout=True)
    ax = axes[0]
    for s, lab, col in ((price["SE-S"], LABEL, "#1f78b4"),
                        (price2["SE-S"], LABEL2, "#8c6d46"),
                        (act_price["SE-S"], "faktiskt SE-S", "#d62828")):
        ax.plot(np.linspace(0, 100, len(s.dropna())), np.sort(s.dropna())[::-1],
                label=lab, color=col)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("% av timmarna"); ax.set_ylabel("EUR/MWh")
    ax.set_title("Prisvaraktighet SE-S"); ax.legend(fontsize=8)

    ax = axes[1]
    d = (price2.groupby(price2.index.year).mean() - price.groupby(price.index.year).mean())
    d.T.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("EUR/MWh"); ax.set_title(f"{LABEL2} − {LABEL}, årsmedel per zon")
    plt.show()

    print(f"{'zon':6s} {'LABEL':>9} {'LABEL2':>9} {'diff':>7}")
    for z in ZONES:
        print(f"{z:6s} {price[z].mean():>9.1f} {price2[z].mean():>9.1f} "
              f"{price2[z].mean()-price[z].mean():>+7.1f}")

    cmp = pd.DataFrame({
        LABEL: by_carrier().clip(lower=0).sum() * dt_h / 1e6,
        LABEL2: gen2[[c for c in gen2.columns if c in n2.generators.index]]
        .T.groupby(n2.generators["carrier"]).sum().T.clip(lower=0).sum() * dt2 / 1e6,
    })
    cmp["diff"] = cmp[LABEL2] - cmp[LABEL]
    print("\nProduktion per kraftslag, TWh:")
    print(cmp.round(2).to_string())
else:
    print("LABEL2 = None — hoppar över jämförelsen (sätt LABEL2 i cell 1)")

# %% 10 — Marginal källa vid EN given timme (prisöar + trängselkaskad)
# Djupdyk på en enskild snapshot: zonpriset är dualen på nodbalansen; zoner som
# delar pris via en icke-mättad länk bildar en PRISÖ. Inom en ö sätts priset av
# enheten med uppåt-headroom vars bud ≈ öpriset (för lager: egen marginalkostnad
# + vattenvärde/verkningsgrad). Saknas sådan enhet är ön TRÄNGSELKOPPLAD och
# priset ärvs från en grann-ö ∓ trängselränta; kaskaden följs till den ö som HAR
# en lokal marginalenhet — systemets enda äkta prissättare.
# Källa: notebooks/cells/marginal.py (håll de två i synk).
import collections
import heapq


def marginal_source(ts_str, tol_price=0.6, tol_couple=0.6):
    req = pd.Timestamp(ts_str)
    snaps = pd.DatetimeIndex(n.snapshots)
    if req < snaps[0] or req > snaps[-1]:
        print(f"{req} ligger utanför körningens period ({snaps[0]} – {snaps[-1]}).")
        return
    # Snäpp till tidssteget som innehåller den begärda timmen (3h: 17:00 → 15:00-steget).
    ts = snaps[snaps.get_indexer([req], method='ffill')[0]]
    if ts != req:
        step = snaps[1] - snaps[0]
        print(f"OBS: {req} faller i {step}-steget som börjar {ts} — använder den snapshoten.")
    mp = n.buses_t.marginal_price.loc[ts]

    # --- generator-merit @ ts ---
    gp = n.generators_t.p.loc[ts]
    pmaxpu = pd.Series(1.0, index=n.generators.index)
    for c in n.generators_t.p_max_pu.columns:
        pmaxpu[c] = n.generators_t.p_max_pu.at[ts, c]
    avail_up = n.generators.p_nom_opt * pmaxpu                  # max möjlig produktion
    mc = n.generators.marginal_cost.copy()
    for c in n.generators_t.marginal_cost.columns:
        mc[c] = n.generators_t.marginal_cost.at[ts, c]
    head_up = avail_up - gp                                     # rum att producera 1 MW mer

    # --- hydro-lager @ ts (WV = mu_energy_balance) ---
    sp = n.storage_units_t.p.loc[ts]
    wv = n.storage_units_t.mu_energy_balance.loc[ts]
    spn = n.storage_units.p_nom

    # --- prisöar: union over trängselfria länkar ---
    parent = {z: z for z in ZONES}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    links = []
    for l in n.links.index:
        b0, b1 = n.links.at[l, 'bus0'], n.links.at[l, 'bus1']
        # bara transmissionslänkar mellan AC-zoner; hoppa över sektor-länkar
        # (elektrolysör/värmepump/turbin → H2-/värmebussar som ej är i ZONES)
        if b0 not in ZONES or b1 not in ZONES:
            continue
        p0 = n.links_t.p0.at[ts, l]; cap = n.links.at[l, 'p_nom']
        sat = abs(abs(p0) - cap) < 1e-3
        dpr = mp[b1] - mp[b0]
        if not sat and abs(dpr) < tol_couple:
            parent[find(b0)] = find(b1)
        links.append((l, b0, b1, p0, cap, sat, dpr))
    islands = collections.defaultdict(list)
    for z in ZONES:
        islands[find(z)].append(z)
    pri = lambda r: mp[islands[r][0]]

    # --- lokal marginalenhet per ö ---
    # Marginell = har plats att producera 1 MW MER (headroom upp) OCH mc ≈ öpriset.
    # OBS: kravet är INTE dispatch>0 — en enhet på 0 MW vars mc≈pris är nästa steg i
    # merit-order och sätter priset (t.ex. en importtranche som precis ska tas i bruk).
    def carlabel(c):
        return {'market': 'marknadsventil'}.get(c, c)
    setter = {}     # root -> (True, name, value, carrier) eller (False,)
    for root, zs in islands.items():
        pr = pri(root); cands = []
        for z in zs:
            for g in n.generators[n.generators.bus == z].index:
                if head_up[g] > 1.0:
                    cands.append((abs(mc[g] - pr), g, mc[g], carlabel(n.generators.at[g, 'carrier'])))
            # mu_energy_balance värderar LAGRAD energi. För att leverera 1 MWh till
            # nätet tas 1/eff MWh ur lagret, så urladdningsbudet är WV/eff. Hydro har
            # eff = 1.0 och påverkas inte; batterier (0.95) låg 5 % fel — vid ett
            # scarcity-pris på 1782 EUR/MWh blev det 89 EUR och batteriet missades
            # som prissättare trots att WV/eff träffade priset på decimalen.
            # ...och lagrets EGNA marginalkostnad måste med. Vattenkraften får
            # marginal_cost = zonens FAKTISKA historiska pris (network.py:410-411,
            # zone_prices ur market_prices.parquet, golvat på VOM 0.6), så budet är
            # historiskt pris + vattenvärde. Med enbart WV träffade hydro priset i
            # ~1 % av de snapshots där magasinet är interiört; med mc+WV i 64-76 %
            # (NO-N/NO-S) och då på decimalen.
            for s in n.storage_units[n.storage_units.bus == z].index:
                if (spn[s] - sp[s]) > 1.0:
                    eff = n.storage_units.at[s, 'efficiency_dispatch']
                    smc = (n.storage_units_t.marginal_cost.at[ts, s]
                           if s in n.storage_units_t.marginal_cost.columns
                           else n.storage_units.at[s, 'marginal_cost'])
                    bud = smc + wv[s] / eff
                    lab = ('hydro (hist. pris + vattenvärde)'
                           if n.storage_units.at[s, 'carrier'] == 'hydro'
                           else 'batteri (urladdningsbud)')
                    cands.append((abs(bud - pr), s, bud, lab))
        cands.sort()
        setter[root] = (True, cands[0][1], cands[0][2], cands[0][3]) if (cands and cands[0][0] < tol_price) else (False,)

    # --- ö-graf över mättade länkar; Dijkstra till närmaste lokal-sättar-ö (minsta ränta) ---
    adj = collections.defaultdict(list)
    for l, b0, b1, p0, cap, sat, dpr in links:
        if sat and find(b0) != find(b1):
            adj[find(b0)].append((find(b1), abs(dpr)))
            adj[find(b1)].append((find(b0), abs(dpr)))
    def trace(root):
        pq = [(0.0, root, [])]; seen = set()
        while pq:
            cost, cur, path = heapq.heappop(pq)
            if cur in seen: continue
            seen.add(cur)
            if setter[cur][0]:
                return cur, path, cost
            for nb, rent in adj[cur]:
                if nb not in seen:
                    heapq.heappush(pq, (cost + rent, nb, path + [(cur, nb, rent)]))
        return None, [], 0.0

    # --- rapport ---
    print(f"\n{'='*74}\nMARGINAL KÄLLA @ {ts}   ({LABEL})\n{'='*74}")
    print("Zonpriser (EUR/MWh):  " + "   ".join(f"{z}={mp[z]:.1f}" for z in ZONES))
    print("\nPrisöar (zoner som delar pris via trängselfri länk) → vad som sätter priset:")
    for root, zs in sorted(islands.items(), key=lambda kv: -pri(kv[0])):
        s = setter[root]
        hyd = [z for z in zs if sp.get(f'{z} hydro', 0) > 1e-3]
        floor = min((wv[f'{z} hydro'] for z in hyd), default=None)
        fnote = f"  [hydrogolv WV≈{floor:.1f}]" if floor is not None else ""
        if s[0]:
            print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}")
            print(f"            ⟹ LOKAL marginalenhet: {s[1]} ({s[3]}, mc={s[2]:.1f})")
        else:
            dst, path, tot = trace(root)
            if dst is None:
                print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}  → ingen sättare (degenererat)")
                continue
            ss = setter[dst]
            # Prisledet från sättar-ön till denna ö, hopp för hopp (faktiska öpriser →
            # alltid teckenkorrekt: +ränta uppströms en dyrare granne, −ränta nedströms).
            nodes = [path[0][0]] + [b for _, b, _ in path]   # denna ö → … → sättare
            seq = nodes[::-1]                                 # sättare → … → denna ö
            chain = " → ".join(
                f"{islands[seq[i]][0]}={pri(seq[i]):.0f}"
                + (f" {'+' if pri(seq[i+1])>pri(seq[i]) else '−'}{abs(pri(seq[i+1])-pri(seq[i])):.0f}" if i < len(seq)-1 else "")
                for i in range(len(seq)))
            rel = "över" if pri(root) > pri(dst) else "under"
            print(f"  [{pri(root):7.1f}] {{{', '.join(zs)}}}{fnote}")
            print(f"            ⟹ TRÄNGSELKOPPLAD ({rel} sättar-ön): satt av {ss[1]} "
                  f"({ss[3]}, mc={ss[2]:.1f}) i {{{', '.join(islands[dst])}}}={pri(dst):.1f}")
            print(f"            kaskad (öpris ± trängselränta): {chain}")

    print("\nLänkar (flöde / kapacitet, priser i ändarna, trängselränta):")
    for l, b0, b1, p0, cap, sat, dpr in links:
        dirn = f"{b0}→{b1}" if p0 >= 0 else f"{b1}→{b0}"
        flag = f"MÄTTAD  ränta={abs(dpr):.1f}" if sat else ("kopplad (1 pris)" if abs(dpr) < tol_couple else f"Δp={dpr:.1f}")
        print(f"  {l:14} {dirn:13} |{abs(p0):6.0f}|/{cap:5.0f}   {mp[b0]:6.1f}|{mp[b1]:6.1f}   {flag}")
    print("\nTolk: en ö med LOKAL enhet sätter sitt eget pris (enheten där mc≈pris — kan vara "
          "en enhet på 0 MW som är nästa steg i merit-order). En trängselkopplad ö ärver en "
          "sättar-ös pris ± trängselränta: NEDströms (exportträngd mot dyrare granne) → lägre; "
          "UPPströms (importträngd, billig granne avskuren) → högre. Hydroöar har vattenvärdet "
          "(WV) som GOLV men prisas högre när de är export-/importträngda.")

marginal_source("2024-12-12 17:00")
