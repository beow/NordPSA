"""Egna bivillkor (extra_functionality-callbacks) på det byggda nätverket.

Varje funktion returnerar en callback `cb(n, snapshots)` som lägger till variabler,
bivillkor eller målfunktionstermer i n.model när LP:t byggs.
"""
from nordpsa.constraints.bid_ladder import hydro_bid_ladder
from nordpsa.constraints.hydro_ops import (hydro_operation_bounds,
                                          hydro_operation_constraints,
                                          hydro_operation_feasibility_report)
from nordpsa.constraints.soc import hydro_soc_initial_constraint
from nordpsa.constraints.stability import (stability_constraints, stability_feasibility_report,
                                           stability_results)
from nordpsa.constraints.terminal_value import hydro_terminal_value

__all__ = ["hydro_bid_ladder", "hydro_operation_bounds", "hydro_operation_constraints",
           "hydro_operation_feasibility_report", "hydro_soc_initial_constraint",
           "hydro_terminal_value", "stability_constraints", "stability_feasibility_report",
           "stability_results"]
