.PHONY: help install test reproduce plots paper reproduce-smoke clean

PYTHON ?= python

ifeq ($(strip $(PYTHON)),)
$(error PYTHON is set but empty. Pass PYTHON=/path/to/python or unset it.)
endif

help:
	@echo "CheckLLM reproducibility targets"
	@echo "  make install         -- install project + dev deps (editable)"
	@echo "  make test            -- run the full pytest suite"
	@echo "  make reproduce-smoke -- smoke-run the paper harness with the fake judge"
	@echo "  make reproduce       -- run every paper experiment (requires API keys)"
	@echo "  make plots           -- regenerate figures from results/"
	@echo "  make paper           -- build paper/checkllm.pdf"
	@echo "  make clean           -- remove build artefacts and caches"

install:
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install --no-deps -e .
	$(PYTHON) -m pip install pytest ruff

test:
	$(PYTHON) -m pytest tests/ -q

reproduce-smoke:
	@if [ ! -f benchmarks/paper/run_all.py ]; then \
		echo "benchmarks/paper/run_all.py is not yet implemented (Phase B, Task B3)."; \
		echo "Nothing to run. This target will be wired up once that module lands."; \
		exit 0; \
	fi
	$(PYTHON) -m benchmarks.paper.run_all --config benchmarks/paper/smoke.yaml

reproduce:
	@if [ ! -f benchmarks/paper/run_all.py ]; then \
		echo "benchmarks/paper/run_all.py is not yet implemented (Phase B, Task B3)."; \
		echo "Nothing to run. This target will be wired up once that module lands."; \
		exit 0; \
	fi
	$(PYTHON) -m benchmarks.paper.run_all --config benchmarks/paper/config.yaml

plots:
	@if [ ! -f benchmarks/paper/plot_all.py ]; then \
		echo "benchmarks/paper/plot_all.py is not yet implemented (Phase D)."; \
		exit 0; \
	fi
	$(PYTHON) -m benchmarks.paper.plot_all

paper:
	@if [ ! -f paper/checkllm.tex ]; then \
		echo "paper/checkllm.tex is not yet implemented (Phase D)."; \
		exit 0; \
	fi
	cd paper && latexmk -pdf checkllm.tex

clean:
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
