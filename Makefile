.PHONY: public-check test

public-check:
	python -m pytest tests/test_cli_entrypoints.py tests/test_public_api.py tests/test_cpu_stub_pipeline.py -q -o addopts=""

test: public-check
