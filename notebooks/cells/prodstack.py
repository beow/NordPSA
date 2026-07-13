"""prodstack — stackad produktionsgraf (areagraf) per kraftslag över tid för LABEL,
med lastlinje(r) ovanpå.

Ställ in tidsfönster (START/END) och upplösning (FREQ = pandas resample-frekvens,
t.ex. '3h','6h','D','W','ME','YE'). FREQ måste vara ≥ körningens egen upplösning.
Hela Norden default, eller en enskild zon via ZONE.

BALANSERAD VY (stacktopp = lastlinje vid varje tidssteg):
  Produktionssida (areor): generering + Import (kontinent + intern NTC) + Batteri urladdning.
  Lastlinje (svart):       baslast + H2 + värme + EV + batteriladdning + export  (ALLT utom spill).
  Split: batteri delas i urladdning (produktion) / laddning (last); handel i import
         (produktion, delad på kontinent vs intern NTC) / export (last). Intern NTC = 0 för hela Norden.
Extra referenslinjer: H2-elektrolys, EV-laddning och batteriladdning var för sig.
Om en enskild zon valts (ZONE) ritas zonpriset som tunn röd linje på höger y-axel.
Produktion = DISPATCHAT (curtailad VRE ingår ej; jfr ebalance-cellens spill-rad).

Förutsätter bootstrap.py (n, dispatch, hydro_d, zone_hydro_total, flows, zone_market,
cfg, plt, mdates, pd, ZONES, LABEL).
"""

# ── Inställningar ──────────────────────────────────────────────────────────────
START = None        # startdatum, t.ex. '2024-01-01'   (None = från början)
END   = None        # slutdatum,  t.ex. '2024-03-31'   (None = till slutet)
FREQ  = 'D'         # upplösning: '3h','6h','D','W','ME','YE' ... (resample-frekvens)
ZONE  = None        # None = hela Norden; annars en zon, t.ex. 'SE-S'
UNIT  = 'GW'        # 'GW' eller 'MW'
# ───────────────────────────────────────────────────────────────────────────────

_zones = ZONES if ZONE is None else [ZONE]
_zero  = pd.Series(0.0, index=dispatch.index)

def _carrier_sum(carrier):
    """Summera dispatch (MW, ≥0) för alla generatorer med given carrier i _zones."""
    cols = [g for g in dispatch.columns if g in n.generators.index
            and n.generators.at[g, 'bus'] in _zones
            and n.generators.at[g, 'carrier'] == carrier]
    return dispatch[cols].clip(lower=0).sum(axis=1) if cols else _zero.copy()

# ── Produktion per kraftslag (dispatchat) ───────────────────────────────────────
vatten = sum((zone_hydro_total(z) for z in _zones), _zero.copy())     # magasin + RoR
termisk = _carrier_sum('thermal')                                     # inkl KVV-el nedan
for z in _zones:
    eta = float(((((cfg.get('heat') or {}).get('zones') or {}).get(z, {}).get('chp')) or {}).get('eta_el', 0.0))
    if eta:
        termisk = termisk.add(flows.get(f'{z} chp', _zero).clip(lower=0) * eta, fill_value=0.0)

# ── Batteri: urladdning (produktion) / laddning (last) ───────────────────────────
nl = n.loads_t.p_set.copy(); nl.index = pd.to_datetime(nl.index).tz_localize(None)
nl = nl.reindex(dispatch.index)
batt_u = [s for s in n.storage_units.index
          if n.storage_units.at[s, 'carrier'] != 'hydro' and n.storage_units.at[s, 'bus'] in _zones]
if batt_u:
    bp = n.storage_units_t.p.copy(); bp.index = pd.to_datetime(bp.index).tz_localize(None)
    batt_net = bp.reindex(dispatch.index)[batt_u].sum(axis=1)          # + = urladdning
else:
    batt_net = _zero.copy()
batt_dis = batt_net.clip(lower=0)          # produktion
batt_chg = (-batt_net).clip(lower=0)       # last

# ── Handel: import (produktion) / export (last), delad på källa ──────────────────
#   Två komponenter: kontinentventil (zone_market: + = import) och intern NTC
#   in/ut ur scope. Klipps var för sig → bruttoimport per källa (kontinent vs
#   intern) syns separat. Balansen bevaras: exp = summan av båda källornas export,
#   så (stacktopp − lastlinje) är oförändrad jämfört med nettoklippning.
#   För hela Norden (ZONE=None) är intern NTC = 0 (inga länkar korsar scope-gränsen).
cont_net = sum((zone_market(z) for z in _zones), _zero.copy())
ntc = [(l, n.links.at[l, 'bus0'], n.links.at[l, 'bus1']) for l in n.links.index
       if n.links.at[l, 'bus0'] in ZONES and n.links.at[l, 'bus1'] in ZONES]
ntc_in = _zero.copy()
for l, b0, b1 in ntc:
    f = flows.get(l, _zero)
    if   b1 in _zones and b0 not in _zones: ntc_in = ntc_in + f        # in i scope
    elif b0 in _zones and b1 not in _zones: ntc_in = ntc_in - f        # ut ur scope
imp_cont = cont_net.clip(lower=0)          # produktion: kontinental import
imp_ntc  = ntc_in.clip(lower=0)            # produktion: intern NTC-import
exp = (-cont_net).clip(lower=0) + (-ntc_in).clip(lower=0)   # last: all export (båda källor)

# ── Lastkomponenter ─────────────────────────────────────────────────────────────
base_load = sum((nl[f'{z} load'] for z in _zones if f'{z} load' in nl.columns), _zero.copy())
h2   = sum((flows.get(f'{z} electrolyser', _zero).clip(lower=0) for z in _zones), _zero.copy())
heat = sum((flows.get(f'{z} heat elboiler', _zero).clip(lower=0)
            + flows.get(f'{z} heat hp', _zero).clip(lower=0) for z in _zones), _zero.copy())
ev_flex   = sum((flows.get(f'{z} EV {c} charger', _zero).clip(lower=0)
                 for z in _zones for c in ('car', 'heavy')), _zero.copy())
ev_inflex = sum((nl[f'{z} EV {c} inflex'].fillna(0.0)
                 for z in _zones for c in ('car', 'heavy') if f'{z} EV {c} inflex' in nl.columns), _zero.copy())
ev = ev_flex + ev_inflex

# Total last (allt utom spill) = baslast + H2 + värme + EV + batteriladdning + export.
load_line = base_load + h2 + heat + ev + batt_chg + exp

# ── Stack + linjer, resamplade till FREQ (medeleffekt per period) ────────────────
prod = pd.DataFrame({
    'Kärnkraft':          _carrier_sum('nuclear'),
    'Vattenkraft':        vatten,
    'Termisk':            termisk,
    'Gas':                _carrier_sum('gas'),
    'Vind onshore':       _carrier_sum('wind_onshore'),
    'Vind offshore':      _carrier_sum('wind_offshore'),
    'Sol':                _carrier_sum('solar'),
    'Import kontinent':   imp_cont,
    'Import intern (NTC)':imp_ntc,
    'Batteri (urladdn.)': batt_dis,
})
lines = pd.DataFrame({'Last (totalt)': load_line, 'H2-elektrolys': h2, 'EV-laddning': ev,
                      'Batteriladdning': batt_chg})

_div = 1e3 if UNIT == 'GW' else 1.0
prod  = (prod.loc[START:END].resample(FREQ).mean().dropna(how='all')) / _div
lines = (lines.loc[START:END].resample(FREQ).mean()).reindex(prod.index) / _div

# Zonpris (endast om en enskild zon valts) → sekundär höger y-axel.
price = None
if ZONE is not None and ZONE in prices.columns:
    _ps = prices[ZONE].copy(); _ps.index = pd.to_datetime(_ps.index).tz_localize(None)
    price = _ps.reindex(dispatch.index).loc[START:END].resample(FREQ).mean().reindex(prod.index)

COLORS = {
    'Kärnkraft': '#d62728', 'Vattenkraft': '#1f77b4', 'Termisk': '#8c564b', 'Gas': '#7f7f7f',
    'Vind onshore': '#2ca02c', 'Vind offshore': '#17becf', 'Sol': '#ff7f0e',
    'Import kontinent': '#e377c2', 'Import intern (NTC)': '#c5b0d5', 'Batteri (urladdn.)': '#bcbd22',
}
order = [c for c in COLORS if c in prod.columns and prod[c].abs().sum() > 0]

fig, ax = plt.subplots(figsize=(16, 6))
ax.stackplot(prod.index, [prod[c].values for c in order],
             labels=order, colors=[COLORS[c] for c in order])
ax.plot(lines.index, lines['Last (totalt)'].values, color='black', lw=2.0, label='Last (totalt)')
ax.plot(lines.index, lines['H2-elektrolys'].values, color='purple', lw=1.3, ls='--', label='H2-elektrolys')
ax.plot(lines.index, lines['EV-laddning'].values, color='saddlebrown', lw=1.3, ls=':', label='EV-laddning')
ax.plot(lines.index, lines['Batteriladdning'].values, color='#bcbd22', lw=1.3, ls='-.', label='Batteriladdning')

ax.set_ylabel(f'Effekt ({UNIT}, medel per {FREQ})')
ax.set_xlim(prod.index.min(), prod.index.max())
ax.margins(x=0, y=0)
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

# Zonpris på höger y-axel (tunn heldragen röd linje).
_handles, _labels = ax.get_legend_handles_labels()
if price is not None:
    ax2 = ax.twinx()
    _pl, = ax2.plot(price.index, price.values, color='red', lw=0.8, label='Pris (höger)')
    ax2.set_ylabel('Pris (EUR/MWh)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.margins(x=0, y=0)
    ax2.grid(False)
    _handles.append(_pl); _labels.append('Pris (höger)')
ax.legend(_handles, _labels, loc='upper left', ncol=5, fontsize=8.5, framealpha=0.9)
_scope = 'Norden' if ZONE is None else ZONE
_span  = f"{prod.index.min():%Y-%m-%d} – {prod.index.max():%Y-%m-%d}"
ax.set_title(f'Stackad produktion + last — {LABEL} ({_scope}, {FREQ}-medel; {_span})', fontweight='bold')
plt.tight_layout()
