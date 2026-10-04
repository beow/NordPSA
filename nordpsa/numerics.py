"""Numerisk städning av nätverket före lösning: nolla koefficienter som är brus men vidgar LP:ts spann.

Revision av expansionens LP (2026-10-03, `temp/coef_audit.py`): matrisen gick ned till 6e-6 (solens
kapacitetsfaktor har brus kring noll: 0 < p_max_pu < 1e-3 i 24 % av tidsstegen) och kostnaderna ned till
4e-4 €/MWh (kontinentens prisstege i timmar med pris nära noll). Tillsammans med PyPSA:s låsta variabel
för objektivkonstanten (~2e10, se `solve.solve`) gav det HiGHS-varningar om "excessively large/small" och
ett spann på 9–10 storleksordningar i kostnad och gränser.

⚠️ Skiljekostnaden 0,01 €/MWh på batterier och elbilsladdning (network/storage.py, ev.py) är avsiktlig;
`cost_eps` måste ligga under den (default 0,005).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zero_small(df: pd.DataFrame, eps: float) -> int:
    """Nolla 0 < |v| < eps på plats; returnerar antal ändrade värden."""
    if df is None or df.empty or eps <= 0:
        return 0
    num = df.select_dtypes(include=[np.number])
    mask = (num.abs() > 0) & (num.abs() < eps)
    k = int(mask.to_numpy().sum())
    if k:
        df.loc[:, num.columns] = num.mask(mask, 0.0)
    return k


def _tidy_pmax(n, comp: str, eps: float) -> tuple[int, int]:
    """p_max_pu < eps → 0 (tidsserier); sänk p_min_pu i samma steg så att p_min ≤ p_max håller."""
    c = n.components[comp]
    dyn = c.dynamic
    if eps <= 0 or "p_max_pu" not in dyn or dyn["p_max_pu"].empty:
        return 0, 0
    pmax = dyn["p_max_pu"]
    k = _zero_small(pmax, eps)
    fixed = 0
    if k:
        cols = pmax.columns
        pmin = n.get_switchable_as_dense(comp, "p_min_pu")[cols]
        lowered = pmin.where(pmin <= pmax, pmax)
        changed = [col for col in cols if not lowered[col].equals(pmin[col])]
        if changed:
            dyn["p_min_pu"] = dyn["p_min_pu"].reindex(columns=dyn["p_min_pu"].columns.union(changed))
            for col in changed:
                dyn["p_min_pu"][col] = lowered[col]
            fixed = len(changed)
    return k, fixed


def tidy(n, pmax_eps: float, cost_eps: float) -> dict:
    """Städa nätverket på plats. Returnerar antal ändrade värden per kategori (för utskrift)."""
    out = {}
    for comp in ("Generator", "Link"):
        k, fixed = _tidy_pmax(n, comp, pmax_eps)
        out[f"{comp} p_max_pu"] = k
        out[f"{comp} p_min_pu sänkt (kolumner)"] = fixed
    for comp in ("Generator", "Link", "StorageUnit", "Store"):
        c = n.components[comp]
        k = _zero_small(c.dynamic["marginal_cost"], cost_eps) if "marginal_cost" in c.dynamic else 0
        if "marginal_cost" in c.static and cost_eps > 0:
            s = c.static["marginal_cost"]
            m = (s.abs() > 0) & (s.abs() < cost_eps)
            k += int(m.sum())
            c.static.loc[m, "marginal_cost"] = 0.0
        out[f"{comp} marginal_cost"] = k
    return out
