#!/usr/bin/env -S bash -eu -o pipefail

poetry run pre-commit run --all-files
(cd docs && make doctest)
poetry run mypy
poetry run pylama
