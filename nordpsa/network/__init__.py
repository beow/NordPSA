"""Bygger PyPSA-nätverket för NordPSA.

Nätverksstruktur: 6 bussar (budzoner), bidirektionella NTC-länkar, last per zon med
slack (VOLL), vattenkraft (reservoar + strömkraft), kärnkraft, vind/sol, termisk
must-run, gas, batterier, kontinentkablar samt sektorerna vätgas, fjärrvärme, elbilar
och industriell DSR. Ingångspunkten är `build_network` (build.py); varje komponent-
typ har sin egen modul.
"""
from nordpsa.network.build import build_network

__all__ = ["build_network"]
