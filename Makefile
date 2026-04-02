.PHONY: install-core install-ml install-all lint test-core test-ml test clean

install-core:
	pip install -e "packages/bh-sentinel-core[dev]"

install-ml:
	pip install -e "packages/bh-sentinel-ml[dev]"

install-all: install-core install-ml

lint:
	ruff check packages/ training/ deployment/aws-lambda/handler.py
	ruff format --check packages/ training/ deployment/aws-lambda/handler.py

test-core:
	python -m pytest packages/bh-sentinel-core/tests/ -v

test-ml:
	python -m pytest packages/bh-sentinel-ml/tests/ -v

test: test-core test-ml

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name dist -exec rm -rf {} +
	find . -type d -name build -exec rm -rf {} +
