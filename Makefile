.PHONY: all format

all: format

format:
	uvx ruff@0.15.22 check --fix --config ruff.toml
	uvx ruff@0.15.22 format --config ruff.toml
