"""Terminalvärde −λ·SOC[T] i rullande horisont, konkavt i segment."""
from typing import Dict

import numpy as np
import pandas as pd
import pypsa
import xarray as xr


# Styckvis linjär, KONKAV terminalvärdeskurva. Multiplikatorer på bas-λ per lika stort
# SOC-segment, från TOMT (först) till FULLT (sist). Måste vara icke-växande, annars är
# V(SOC) inte konkav och LP:t fyller segmenten i fel ordning.
#
# ⚠️ FORMEN ÄR ETT ANTAGANDE, inte kalibrerad. Normaliseringen: marginalvärdet = bas-λ i
# bandet 60-80 % fyllnad, dvs runt config-ankaret hydro_soc_initial (~70 %). Under det
# stiger vattnets marginalvärde (knapphet), i det översta bandet kollapsar det (spillrisk).
DEFAULT_TERMINAL_PROFILE = [2.0, 1.5, 1.2, 1.0, 0.2]


def hydro_terminal_value(lambda_per_unit: Dict[str, float],
                         cap_per_unit:    Dict[str, float] | None = None,
                         profile:         list | None = None):
    """extra_functionality-callback som lägger terminalvärdet −V(SOC[T]) i målfunktionen.

    Belönar modellen för att hålla vatten kvar vid fönstrets SISTA tidssteg. Utan en
    sådan term tömmer ett icke-cykliskt fönster reservoaren mot slutet (vattnet är
    värdelöst efter horisonten) — det är hela poängen med rullande horisont.

    **V(SOC) är STYCKVIS LINJÄR OCH KONKAV.** Kapaciteten delas i K lika segment med
    fallande marginalvärde λ_k = λ_bas × profile[k]:

        V(SOC) = Σ_k λ_k · s_k,   0 ≤ s_k ≤ cap/K,   Σ_k s_k = SOC[T]

    Eftersom λ_k är fallande fyller LP:t automatiskt de värdefulla segmenten först —
    inga ordningsvillkor behövs. Det är den vanliga konkav-styckvis-tricket.

    ⚠️ VARFÖR KONKAV OCH INTE LINJÄRT (run268, 2026-08-07): ett linjärt −λ×SOC[T] har
    KONSTANT derivata oavsett fyllnadsgrad. Vattnets marginalvärde blir då bang-bang —
    håll allt tills taket nås, därefter noll — och resultatet blev magasin på 100 % i
    13-29 % av timmarna, vattenvärde som föll 82 → 3 vid taket, och zonpriser som
    kollapsade till VOM (0,5-1,6) i hela veckor. Ett riktigt vattenvärde är konkavt:
    högt när magasinet är tomt, lågt när det är fullt.

    ⚠️ λ MÅSTE VARA PÅ SAMMA SKALA SOM MARGINALEN. Optimeraren håller vatten när
    λ > (λ_bus[t] − marginal_cost[t]). Med vattenvärdes-proxyn påslagen är
    marginal_cost det historiska zonpriset, marginalen krymper till nettovattenvärdet,
    och λ på bruttoprisnivå övervärderar lagring. Kan inte inträffa sedan proxyn är
    lägesbunden: rullande horisont kräver frysta kapaciteter, som alltid saknar proxy.

    lambda_per_unit: {storage_unit: bas-λ EUR/MWh}
    cap_per_unit:    {storage_unit: volym MWh}. Krävs för K>1.
    profile:         icke-växande multiplikatorer, tomt→fullt. En LISTA (samma profil för
                     alla lager) eller en DICT {storage_unit: lista} för PER-ZON-kurvor —
                     hur brant vattenvärdet ska falla mot fullt magasin beror på om zonen
                     har en exportväg eller måste spilla. None = DEFAULT_TERMINAL_PROFILE.
                     [1.0] ger det gamla LINJÄRA beteendet (ett segment).
    """
    def _check(p, tag):
        p = list(p)
        if not p:
            raise ValueError(f"hydro_terminal_value: tom profil ({tag})")
        if any(b - a > 1e-9 for a, b in zip(p, p[1:])):
            raise ValueError(f"hydro_terminal_value: profilen måste vara ICKE-VÄXANDE "
                             f"(tomt→fullt) för att V(SOC) ska bli konkav, fick {p} ({tag})")
        return p

    if profile is None:
        prof_map, prof_def = {}, _check(DEFAULT_TERMINAL_PROFILE, "default")
    elif isinstance(profile, dict):
        prof_map = {u: _check(p, u) for u, p in profile.items()}
        prof_def = _check(DEFAULT_TERMINAL_PROFILE, "default")
    else:
        prof_map, prof_def = {}, _check(profile, "global")

    def _prof(su):
        return prof_map.get(su, prof_def)

    lens = {len(_prof(u)) for u in lambda_per_unit}
    if len(lens) > 1:
        raise ValueError(f"hydro_terminal_value: alla lager måste ha samma ANTAL segment, "
                         f"fick {sorted(lens)}")
    K = lens.pop() if lens else len(prof_def)

    def _extra_functionality(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
        m      = n.model
        soc    = m.variables["StorageUnit-state_of_charge"]
        t_last = snapshots[-1]
        units  = [su for su, lam in lambda_per_unit.items()
                  if lam > 0.0 and su in n.storage_units.index]
        if not units:
            return

        if K == 1:                                  # linjärt specialfall — inga extra variabler
            term = None
            for su in units:
                piece = (-float(lambda_per_unit[su]) * _prof(su)[0]
                         * soc.sel(name=su, snapshot=t_last))
                term  = piece if term is None else term + piece
            m.objective = m.objective + term
            return

        if cap_per_unit is None:
            raise ValueError("hydro_terminal_value: cap_per_unit krävs när profilen har "
                             "fler än ett segment")
        names = pd.Index(units, name="name")
        segs  = pd.Index(range(K), name="term_seg")
        width = xr.DataArray(
            np.array([[cap_per_unit[su] / K] * K for su in units], dtype=float),
            coords=[names, segs])
        lam = xr.DataArray(
            np.array([[lambda_per_unit[su] * p for p in _prof(su)] for su in units],
                     dtype=float),
            coords=[names, segs])

        s = m.add_variables(lower=0.0, upper=width, coords=[names, segs],
                            name="hydro_terminal_seg")
        # Σ_k s_k == SOC[T]  (per lager)
        m.add_constraints(s.sum("term_seg") - soc.sel(name=names, snapshot=t_last) == 0.0,
                          name="hydro_terminal_soc_def")
        m.objective = m.objective - (lam * s).sum()

    return _extra_functionality
