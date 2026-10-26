# claude-obsidian deterministic developer entry points.

PYTHON ?= python3
export PYTHONDONTWRITEBYTECODE := 1

.PHONY: help test test-python test-shell test-contracts test-package validate \
	setup-dragonscale setup-retrieve setup-mode

help:
	@echo "claude-obsidian developer targets:"
	@echo "  make test             Run every hermetic Python and shell test, then contracts"
	@echo "  make test-python      Run each tests/test_*.py file in isolation"
	@echo "  make test-shell       Run each tests/test_*.sh file in isolation"
	@echo "  make test-contracts   Execute canonical product/capability verification"
	@echo "  make test-package     Validate portable skill, hook, and manifest metadata"
	@echo "  make validate         Run package and contract validators without the test suite"
	@echo "  make setup-*          Run an opt-in legacy extension setup helper"

test: test-python test-shell test-contracts test-package
	@echo "All hermetic tests and executable contracts passed."

test-python:
	@set -eu; for test_file in tests/test_*.py; do \
		echo "=== $$test_file ==="; \
		$(PYTHON) "$$test_file"; \
	done

test-shell:
	@set -eu; for test_file in tests/test_*.sh; do \
		echo "=== $$test_file ==="; \
		bash "$$test_file"; \
	done

test-contracts:
	@$(PYTHON) scripts/claude-obsidian.py contracts --check-only
	@$(PYTHON) scripts/claude-obsidian.py contracts --verify

test-package:
	@$(PYTHON) scripts/claude-obsidian.py package validate

validate: test-contracts test-package

setup-dragonscale:
	@bash scripts/setup-dragonscale.sh

setup-retrieve:
	@bash scripts/setup-retrieve.sh

setup-mode:
	@bash scripts/setup-mode.sh
