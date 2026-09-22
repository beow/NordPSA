"""Klienter mot externa datakällor. Används av scripts/fetch_*.py och build_inputs.py,
aldrig av själva modellkörningen (som bara läser data/processed/).

- `esett` — eSett open data: last och produktion per MBA, aggregerat till NordPSA-zoner.
- `ec` — Energy Charts: VRE-profiler, DE-LU-pris, magasinnivåer.
- `entsoe` — ENTSO-E Transparency (reservoar, produktion, priser) + Elexon för GB.
- `ninja` — Renewables.ninja: havsvindsprofiler.
"""
