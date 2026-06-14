.PHONY: env fetch fetch-ec fetch-ninja fetch-nve fetch-openmeteo build build-heat solve results all

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

fetch:
	python scripts/fetch_esett.py

fetch-ec:
	python scripts/fetch_ec.py

fetch-ninja:
	python scripts/fetch_ninja.py

fetch-nve:
	python scripts/fetch_nve.py

fetch-openmeteo:
	python scripts/fetch_openmeteo.py

build:
	python scripts/build_inputs.py

build-heat:
	python scripts/build_heat.py

solve:
	python scripts/run_model.py

results:
	python scripts/postprocess.py

all: fetch build solve results
