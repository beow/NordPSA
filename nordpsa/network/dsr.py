"""Industriell efterfrågeflexibilitet (DSR) som pristrappa."""

import pypsa


def add_industrial_dsr(n: pypsa.Network, cfg: dict) -> None:
    """Industriell efterfrågeflexibilitet som pristrappa på elbussen (LMA2026).

    Shed-generatorer med marginal_cost = stegets prisnivå: LP:t "köper" dem i stället för
    att producera när zonpriset passerar steget, vilket är precis en bortkopplad last.
    Egen carrier "DSR" så att de INTE räknas som produktion i resultatstatistiken.

    ⚠️ Steg 4 (3000 €/MWh) byggs medvetet INTE: den befintliga slack-generatorn ligger
    redan på MC_SLACK 3000 på varje buss och biter på all last. En fjärde tranch där vore
    både dubbelräkning och degenererad med slacken (LP:t indifferent → solverbrus).
    """
    dc = cfg.get("industrial_dsr") or {}
    if not dc.get("enabled", False):
        return
    steps  = list(dc.get("price_steps_eur_per_mwh", [100, 250, 500]))
    shares = list(dc.get("step_shares", [0.25, 0.25, 0.25]))
    if len(steps) != len(shares):
        raise SystemExit("industrial_dsr: price_steps och step_shares olika längd")
    if "DSR" not in n.carriers.index:
        n.add("Carrier", "DSR")
    tot = 0.0
    for zone, vol in (dc.get("volume_mw") or {}).items():
        if zone not in n.buses.index:
            print(f"  Varning: DSR-zon {zone} saknas — hoppar över")
            continue
        for k, (thr, sh) in enumerate(zip(steps, shares), start=1):
            n.add("Generator", f"{zone} DSR{k}",
                  bus=zone, carrier="DSR",
                  p_nom=float(sh) * float(vol),
                  marginal_cost=float(thr))
        tot += sum(shares) * float(vol)
    print(f"  → industriell DSR: {tot:.0f} MW i {len(steps)} steg "
          f"({', '.join(f'{s:.0%}@{t:g}' for s, t in zip(shares, steps))} €/MWh); "
          f"steg vid VOLL byggs ej (slacken finns redan)")
