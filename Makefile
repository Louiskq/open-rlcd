.PHONY: help install test lint format toy toy0b toy-quick readme readme-check notebook figures expA-analysis expA-figures explorer

PY ?= uv run python

help:
	@echo "make install      create .venv and install the package with dev tools"
	@echo "make test         run the unit tests"
	@echo "make lint         ruff check + format check"
	@echo "make toy          Exp 0 (full): results/toy/*.json + figures/ (about 1-2 min on a laptop CPU)"
	@echo "make toy0b        Exp 0b (exploratory): difficulty varies by input"
	@echo "make toy-quick    Exp 0 and 0b with 3 seeds and 1,500 steps (CI)"
	@echo "make readme       regenerate the tables inside README.md and docs/ from results/"
	@echo "make readme-check fail if README.md or docs/ is out of date with results/"
	@echo "make notebook     rebuild and execute notebooks/toy_rlcd.ipynb"
	@echo "make figures      render the concept diagram (the toy figures come from make toy)"
	@echo "make expA-analysis  the pre-registered Exp A tests from results/expA/runs (about 25 min on a laptop CPU)"
	@echo "make expA-figures   the Exp A figures from results/expA/analysis.json"

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

toy:
	$(PY) -m toy.run

toy0b:
	$(PY) -m toy.exp0b

toy-quick:
	$(PY) -m toy.run --quick --out /tmp/rlcd-toy-quick
	$(PY) -m toy.exp0b --quick --out /tmp/rlcd-toy-quick

readme:
	$(PY) scripts/make_readme_tables.py

readme-check:
	$(PY) scripts/make_readme_tables.py --check

notebook:
	$(PY) scripts/build_toy_notebook.py

figures:
	$(PY) -m rlcd.plots figures

expA-analysis:
	$(PY) scripts/compare_arms.py results/expA/runs --out results/expA/analysis.json
	$(PY) scripts/compare_arms.py results/expA/runs --no_tables --split select --out results/expA/analysis_select_prereg_tests.json
	$(PY) scripts/compare_arms.py results/expA/runs --no_tables --exploratory --split dev --out results/expA/analysis_exploratory_dev.json
	$(PY) scripts/compare_arms.py results/expA/runs --no_tables --exploratory --split select --out results/expA/analysis_exploratory_select.json

expA-figures:
	$(PY) -m rlcd.plots_expA results/expA/analysis.json figures

explorer:
	$(PY) scripts/make_explorer.py
