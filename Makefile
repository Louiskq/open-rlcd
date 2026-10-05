.PHONY: help install test lint format toy toy-quick readme readme-check notebook figures

PY ?= uv run python

help:
	@echo "make install      create .venv and install the package with dev tools"
	@echo "make test         run the unit tests"
	@echo "make lint         ruff check + format check"
	@echo "make toy          Exp 0 (full): results/toy/*.json + figures/ (about 1-2 min on a laptop CPU)"
	@echo "make toy-quick    Exp 0 with 3 seeds and 1,500 steps (CI)"
	@echo "make readme       regenerate the tables inside README.md from results/"
	@echo "make readme-check fail if README.md is out of date with results/"
	@echo "make notebook     rebuild and execute notebooks/toy_rlcd.ipynb"
	@echo "make figures      render the concept diagram (the toy figures come from make toy)"

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

toy-quick:
	$(PY) -m toy.run --quick --out /tmp/rlcd-toy-quick

readme:
	$(PY) scripts/make_readme_tables.py

readme-check:
	$(PY) scripts/make_readme_tables.py --check

notebook:
	$(PY) scripts/build_toy_notebook.py

figures:
	$(PY) -m rlcd.plots figures
