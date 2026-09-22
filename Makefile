# NordPSA data pipeline and runs.
#
#   make data     fetch everything, then build data/processed/   (full pipeline)
#   make fetch    only the fetch steps (files already in data/raw/ are skipped)
#   make build    only rebuild data/processed/ from data/raw/
#
# Fetching needs ENTSOE_API_TOKEN (ENTSO-E Transparency) and NINJA_TOKEN
# (Renewables.ninja) in the environment.

.PHONY: env env-update data fetch fetch-esett fetch-ec fetch-nve fetch-ninja \
        fetch-openmeteo synth-ror build build-inputs build-heat solve all

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

# ---- fetch: data/raw/ -------------------------------------------------------
fetch: fetch-esett fetch-ec fetch-nve fetch-ninja fetch-openmeteo

fetch-esett:
	python scripts/fetch_esett.py

fetch-ec:
	python scripts/fetch_ec.py

fetch-nve:
	python scripts/fetch_nve.py --fetch-entsoe

fetch-ninja:
	python scripts/fetch_ninja.py

fetch-openmeteo:
	python scripts/fetch_openmeteo.py

# Synthetic Swedish run-of-river (SvK reports none). Rewrites the SE inflow files
# from fetch-nve, so it must run after it.
synth-ror: fetch-nve
	python scripts/synth_se_ror.py

# ---- build: data/processed/ -------------------------------------------------
build: build-inputs build-heat

build-inputs:
	python scripts/build_inputs.py

build-heat:
	python scripts/build_heat.py

data: fetch synth-ror build
all: data

# ---- run -----------------------------------------------------------------------
solve:
	nordpsa expand --output $(OUT)
