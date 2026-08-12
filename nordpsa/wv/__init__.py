"""Vattenvärdesskiktet (water value).

⚠️ INGET i NordPSA-kärnan får importera detta paket ANNAT än via en uttrycklig
flagga. Modulerna här producerar KOEFFICIENTER — en YAML eller en parquet — som
`scripts/run_model.py` läser som vilken indata som helst.

`terminal_curve.py` levererar λ_k(fyllnadsgrad, vecka) till den rullande
horisontens terminalvärde (`--terminal-curve`); `targets.py` poängsätter en körning
mot de två icke-cirkulära observablerna (eSetts fysiska hydrosäsong och Energy
Charts magasinband).

Se docs/vattenvarde_plan.md.
"""
