.PHONY: format lint typecheck test test-executor build check

PY ?= python
DEV_PY ?= $(PY)
# A repository checkout nests the verifier and clinical-data packages; a harness sdist has neither.
VERIFIER := $(wildcard verifier/pyproject.toml)
TYPED := trialagentbench $(wildcard clinical-data/src/clinical_data_tables)

format:
	$(DEV_PY) -m black .
	RUFF_NO_CACHE=true $(DEV_PY) -m ruff check . --fix

lint:
	$(DEV_PY) -m black --check .
	RUFF_NO_CACHE=true $(DEV_PY) -m ruff check .

typecheck:
	tmpdir=$$(mktemp -d); \
	  MYPY_CACHE_DIR=$$tmpdir $(DEV_PY) -m mypy $(TYPED) \
	  $(if $(VERIFIER),&& MYPY_CACHE_DIR=$$tmpdir $(DEV_PY) -m mypy --config-file verifier/pyproject.toml -p trialagentbench_verifier); \
	  rc=$$?; rm -rf "$$tmpdir"; exit $$rc

test:
	$(DEV_PY) -m pytest -p no:cacheprovider -c pyproject.toml --rootdir=.
	$(if $(VERIFIER),$(DEV_PY) -m pytest -p no:cacheprovider -c verifier/pyproject.toml --rootdir=verifier -m "not executor" verifier/tests)

test-executor:
	docker build -t trialagentbench/executor:1.0.1 executor
	$(DEV_PY) -m pytest -p no:cacheprovider -c pyproject.toml --rootdir=. -m executor tests
	$(if $(VERIFIER),$(DEV_PY) -m pytest -p no:cacheprovider -c verifier/pyproject.toml --rootdir=verifier -m executor verifier/tests)

build:
	$(DEV_PY) -m build

check: lint typecheck test
