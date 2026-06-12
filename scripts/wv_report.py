"""Vattenvärdesrapport: jämför en körning (t.ex. med --water-value-curve) mot
referenskörning och FAKTISKA zonpriser/reservoarnivåer.

Användning:
    python scripts/wv_report.py run109_wvcurve_3h --ref run95_elasticity_3h

Rapporterar per hydrozon:
  1. Vattenvärdets säsongsprofil (månadsklimatologi) + nivå/std
  2. Pris vs faktiskt: rå bias/MAE/RMSE samt hydrogolv-kompenserat (pris − WV,
     samma logik som notebooks/cells/hydrocomp.py)
  3. Säsongsbias (kvartal) rå — visar om det platta golvets säsongsfel minskat
  4. SOC-bana vs observerad (ENTSO-E veckovis, SE-N=SE1+SE2, SE-S=SE3+SE4)
  5. Spill-totaler
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT  = Path(__file__).resolve().parents[1]
ZONES = ["SE-N", "SE-S", "NO-N", "NO-S", "DK", "FI"]
SE_MBA = {"SE-N": ["SE1", "SE2"], "SE-S": ["SE3", "SE4"]}


def load_run(label: str) -> dict:
    d = ROOT / "results" / label
    out = {}
    for f in ("prices", "water_value", "hydro_soc", "hydro_spill"):
        p = d / f"{f}.csv"
        out[f] = pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None
    return out


def load_actual_prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data/processed/market_prices.parquet")
    mp.index = pd.to_datetime(mp.index).tz_localize(None)
    return mp.reindex(index, method="ffill")


def load_actual_reservoir(zone: str) -> pd.Series:
    parts = []
    for mba in SE_MBA[zone]:
        files = sorted((ROOT / "data/raw").glob(f"reservoir_entsoe_{mba}_*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in files])
        s = df.set_index("timestampUTC")["reservoir_mwh"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        parts.append(s)
    return sum(p.reindex(parts[0].index, method="nearest") for p in parts)


def stats(model: pd.Series, actual: pd.Series):
    d = (model - actual).dropna()
    return d.mean(), d.abs().mean(), (d ** 2).mean() ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--ref", default="run95_elasticity_3h")
    args = ap.parse_args()

    run, ref = load_run(args.label), load_run(args.ref)
    act = load_actual_prices(run["prices"].index)
    hz = [c for c in run["water_value"].columns if c.endswith("hydro")]

    # --- 1. Vattenvärdets säsongsprofil ---
    print(f"=== 1. Vattenvärde: {args.label} vs {args.ref} (EUR/MWh) ===")
    print(f"{'Zon':<12}{'medel':>7}{'std':>6}{'min':>6}{'max':>6}   "
          f"{'ref-medel':>9}{'ref-std':>8}")
    for z in hz:
        wv, wvr = run["water_value"][z], ref["water_value"].get(z)
        line = f"{z:<12}{wv.mean():>7.1f}{wv.std():>6.1f}{wv.min():>6.1f}{wv.max():>6.1f}"
        if wvr is not None:
            line += f"   {wvr.mean():>9.1f}{wvr.std():>8.1f}"
        print(line)
    print("\nMånadsklimatologi (medel över åren):")
    clim = run["water_value"][hz].groupby(run["water_value"].index.month).mean().round(1)
    clim.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in clim.index]
    print(clim.T.to_string())

    # --- 2. Pris vs faktiskt: rå + WV-kompenserat ---
    for tag, r in [(args.label, run), (args.ref, ref)]:
        pr = r["prices"]
        a = act.reindex(pr.index, method="ffill")
        calw = pr.copy()
        for z in ZONES:
            col = f"{z} hydro"
            if r["water_value"] is not None and col in r["water_value"].columns:
                calw[z] = pr[z] - r["water_value"][col].reindex(pr.index).fillna(0.0)
        print(f"\n=== 2. Pris vs FAKTISKT — {tag} ===")
        print(f"{'Zon':<7}{'modell':>8}{'faktiskt':>9}{'bias':>7}{'MAE':>7}{'RMSE':>7}"
              f"{'  komp-bias':>11}{'komp-MAE':>9}")
        for z in ZONES:
            if z not in a.columns:
                continue
            b, mae, rmse = stats(pr[z], a[z])
            bc, maec, _ = stats(calw[z], a[z])
            print(f"{z:<7}{pr[z].mean():>8.1f}{a[z].mean():>9.1f}{b:>+7.1f}{mae:>7.1f}"
                  f"{rmse:>7.1f}{bc:>+11.1f}{maec:>9.1f}")

    # --- 3. Säsongsbias (kvartal, rå) ---
    print(f"\n=== 3. Säsongsbias rå (modell − faktiskt, kvartalsmedel EUR/MWh) ===")
    for tag, r in [(args.label, run), (args.ref, ref)]:
        pr = r["prices"]
        a = act.reindex(pr.index, method="ffill")
        diff = (pr[ZONES] - a[ZONES]).groupby(pr.index.quarter).mean().round(1)
        diff.index = [f"Q{q}" for q in diff.index]
        print(f"--- {tag}")
        print(diff.T.to_string())

    # --- 4. SOC vs observerad (SE-zoner, ENTSO-E) ---
    print(f"\n=== 4. Reservoar-SOC vs observerad (ENTSO-E, veckovis) ===")
    print(f"{'Zon':<7}{'korr':>6}{'bias TWh':>10}{'MAE TWh':>9}   (ref: korr/bias/MAE)")
    for z in SE_MBA:
        col = f"{z} hydro"
        obs = load_actual_reservoir(z)
        line = f"{z:<7}"
        for r in (run, ref):
            soc = r["hydro_soc"][col]
            o = obs.reindex(soc.index, method="nearest", tolerance=pd.Timedelta("4D")).dropna()
            m = soc.reindex(o.index)
            c = m.corr(o)
            b, mae, _ = stats(m / 1e6, o / 1e6)
            line += f"{c:>6.2f}{b:>+10.1f}{mae:>9.1f}   "
        print(line)

    # --- 5. Spill ---
    print(f"\n=== 5. Spill (TWh över perioden) ===")
    dt = (run["prices"].index[1] - run["prices"].index[0]).total_seconds() / 3600
    for tag, r in [(args.label, run), (args.ref, ref)]:
        sp = r["hydro_spill"]
        tot = (sp[[c for c in hz if c in sp.columns]].sum() * dt / 1e6).round(1)
        print(f"--- {tag}: {tot.to_dict()}")


if __name__ == "__main__":
    main()
