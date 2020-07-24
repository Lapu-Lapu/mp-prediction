#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = mp_prediction
PYTHON_INTERPRETER = python3

all: fig3 fig4 fig5 fig6

.PHONY: data
data: onlinedata vrdata

vrdata: data/processed/processed_data_vr.json

onlinedata: data/processed/processed_data_online.json

data/processed/processed_data_vr.json data/processed/catrchtrial_vr.csv:
	python src/data/process_vr_data.py

data/processed/catchtrial_online.json data/processed/processed_data_online.json:
	python src/data/process_online_data.py

fig3: reports/figures/fig3.pdf

fig4: reports/figures/fig4.pdf

fig5: reports/figures/fig5a.pdf reports/figures/fig5b.pdf

fig6: reports/figures/fig6.pdf

reports/figures/fig6.pdf: data/processed/processed_data_online.json
	python src/visualization/make_fig6.py

reports/figures/fig5a.pdf: data/processed/processed_data_vr.json
	python src/visualization/make_fig5a.py

reports/figures/fig5b.pdf: data/processed/processed_data_online.json
	python src/visualization/make_fig5b.py

reports/figures/fig3.pdf: data/processed/processed_data_online.json\
	data/processed/processed_data_vr.json
	python src/visualization/make_fig3.py

reports/figures/fig4.pdf: data/processed/catchtrial_online.json\
	data/processed/catrchtrial_vr.csv
	python src/visualization/make_fig4.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete -print
	find . -type d -name "__pycache__" -delete -print
