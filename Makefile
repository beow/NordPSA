.PHONY: env fetch build solve results all

env:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

fetch:
	python scripts/fetch_esett.py

fetch-ec:
	python scripts/fetch_ec.py

build:
	python scripts/build_inputs.py

solve:
	python scripts/run_model.py

results:
	python scripts/postprocess.py

all: fetch build solve results
