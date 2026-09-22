"""Tidsserier som modellen byggs på men inte optimerar över.

- `hydro_inflow` — tillrinning: uppmätt NVE/ENTSO-E (reservoar + strömkraft), den
  parametriska vårflodsmodellen och RoR-högfrekvensen.
- `nuclear_availability` — syntetisk stokastisk tillgänglighet för kärnkraftsflottor.
- `heat_load` — fjärrvärmens värmelastprofiler per zon (When2Heat-metodik).
"""
