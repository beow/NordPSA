"""Vattenvärdesskiktet (water value).

⚠️ INGET i NordPSA-kärnan får importera detta paket ANNAT än via en uttrycklig
inställning. Modulerna här producerar KOEFFICIENTER — en YAML eller en parquet — som
körningen läser som vilken indata som helst.

`terminal_curve.py` levererar λ_k(fyllnadsgrad, vecka) till den rullande
horisontens terminalvärde (`dispatch.terminal_curve`) och hydrons mc i expansion
(`expansion.hydro_mc_curve`).

Se docs/vattenvarde_plan.md.
"""
