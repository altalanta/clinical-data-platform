SHELL := /bin/bash
VENV ?= .venv
PYTHON ?= python3
SKIP_ENV ?= 0

ifeq ($(SKIP_ENV),1)
	PYTHON_BIN := $(PYTHON)
	PIP := $(PYTHON) -m pip
	DBT := dbt
	ENV_STAMP :=
else
	PIP := $(VENV)/bin/pip
	PYTHON_BIN := $(VENV)/bin/python
	DBT := $(VENV)/bin/dbt
	ENV_STAMP := $(VENV)/.deps-installed
endif

DBT_ENV := DBT_PROJECT_ROOT=$$(pwd) DBT_PROFILES_DIR=$$(pwd) DUCKDB_HOME=$$(pwd)/.duckdb HOME=$$(pwd)/.home

.PHONY: help env data dbt-init build docs validate all clean

help:
	@echo "Targets:"
	@echo "  make env       - create virtualenv and install dependencies"
	@echo "  make data      - generate deterministic synthetic CSVs"
	@echo "  make dbt-init  - create ~/.dbt/profiles.yml if missing"
	@echo "  make build     - run dbt deps, seed, and build"
	@echo "  make docs      - run dbt docs generate"
	@echo "  make validate  - run post-build metric validation"
	@echo "  make all       - data + build + docs + validate"
	@echo "  make clean     - remove build artifacts and DuckDB file"

env:
ifeq ($(SKIP_ENV),1)
	@echo "Skipping virtualenv setup; using $(PYTHON)"
else
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi
	@if [ ! -f "$(ENV_STAMP)" ] || [ requirements.txt -nt "$(ENV_STAMP)" ]; then \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.txt; \
		touch $(ENV_STAMP); \
	fi
endif

data: env
	$(PYTHON_BIN) scripts/gen_synth_data.py --seed 1337

dbt-init:
	@if [ ! -f profiles.yml ]; then \
		cp profiles_template.yml profiles.yml; \
		echo "Created local profiles.yml"; \
	else \
		echo "profiles.yml already exists; skipping local copy"; \
	fi
	@mkdir -p $$HOME/.dbt 2>/dev/null || true
	@if [ -d $$HOME/.dbt ] && [ ! -f $$HOME/.dbt/profiles.yml ]; then \
		cp profiles_template.yml $$HOME/.dbt/profiles.yml; \
		echo "Created $$HOME/.dbt/profiles.yml"; \
	fi

build: env
	mkdir -p .duckdb/tmp .home
	$(DBT_ENV) $(DBT) deps
	$(DBT_ENV) $(DBT) seed --full-refresh
	$(DBT_ENV) $(DBT) build

docs: env
	mkdir -p .duckdb/tmp .home
	$(DBT_ENV) $(DBT) docs generate

validate: env
	$(PYTHON_BIN) scripts/validate_metrics.py

all: data build docs validate

clean:
	rm -f clinical.duckdb
	rm -rf target
	rm -f logs/dbt.log
	rm -f data/raw/*.csv
	rm -rf .duckdb
	rm -rf .home
ifneq ($(ENV_STAMP),)
	rm -f $(ENV_STAMP)
endif
