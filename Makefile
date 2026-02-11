.PHONY: health fmt lint typecheck

health:
	poetry run python -m src.index.health_check

fmt:
	poetry run ruff format .

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src
