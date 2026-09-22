"""NordPSA:s kommandorad — tre kommandon, ett per körtyp.

    nordpsa expand   --output run500_baseline                  # 2040, kapaciteter optimeras
    nordpsa dispatch --from run500_baseline --output run501    # källans värld + flotta, frysta
    nordpsa today    --output run502_today                     # dagens system, dispatch

Allt utöver de namngivna flaggorna styrs av inställningsfiler (config/defaults.yaml,
config/worlds/*.yaml) och ändras med --experiment FIL eller --set NYCKEL=VÄRDE:

    nordpsa expand --experiment lowhydro06 --output run503_dry
    nordpsa expand --set market.ntc_scale=0.5 --set hydro.bid_ladder=[5,34.6] --output run504
    nordpsa expand --print-config            # visa den upplösta inställningen, kör inget
"""
from __future__ import annotations

import argparse

import pandas as pd

from nordpsa import settings
from nordpsa.run import run

# pandas 2.x använder Arrow-strängar som standard; PyPSA/xarray stöder inte det
pd.options.future.infer_string = False

DEFAULT_WORLD = {"expand": "2040_svk_mm", "today": "today"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nordpsa", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", help="resultatmapp under results/ (krävs för en körning)")
    common.add_argument("--desc", help="kort fritext om körningens syfte (sparas i run_meta.txt)")
    common.add_argument("--resolution", type=int, help="tidsupplösning i timmar (period.resolution_hours)")
    common.add_argument("--year", type=int, help="kör ett enskilt år (period.year)")
    common.add_argument("--experiment", action="append", default=[], metavar="FIL",
                        help="experimentfil (namn i config/experiments/ eller sökväg); kan upprepas")
    common.add_argument("--set", action="append", default=[], metavar="NYCKEL=VÄRDE",
                        help="överstyr en inställning, t.ex. market.ntc_scale=0.5; kan upprepas")
    common.add_argument("--dry-run", action="store_true", help="bygg nätverket men lös inte")
    common.add_argument("--print-config", action="store_true",
                        help="skriv ut den upplösta inställningen och avsluta")

    e = sub.add_parser("expand", parents=[common], help="expansion: kapaciteter optimeras")
    e.add_argument("--world", default=DEFAULT_WORLD["expand"], help="värld i config/worlds/")
    d = sub.add_parser("dispatch", parents=[common],
                       help="dispatch med källkörningens värld och frysta kapaciteter")
    d.add_argument("--from", dest="source", required=True, metavar="RUN",
                   help="källkörning under results/")
    t = sub.add_parser("today", parents=[common], help="dispatch av dagens system")
    t.add_argument("--world", default=DEFAULT_WORLD["today"], help="värld i config/worlds/")
    return p


def main(argv: list[str] | None = None) -> None:
    a = _parser().parse_args(argv)
    sets = list(a.set)
    # --resolution/--year är genvägar för period-nycklarna och läggs FÖRE --set.
    if a.resolution is not None:
        sets.insert(0, f"period.resolution_hours={a.resolution}")
    if a.year is not None:
        sets.insert(0, f"period.year={a.year}")

    if a.command == "expand":
        s = settings.resolve("expansion", a.world, a.experiment, sets, capacities="optimized")
    elif a.command == "today":
        s = settings.resolve("dispatch", a.world, a.experiment, sets, capacities="config")
    else:
        s = settings.resolve_from_run(a.source, a.experiment, sets)

    if a.print_config:
        print(settings.dump(s), end="")
        return
    if not a.output and not a.dry_run:
        raise SystemExit("--output krävs (resultatmapp under results/)")
    run(s, label=a.output or "dry_run", desc=a.desc, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
