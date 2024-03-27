#!/usr/bin/env -S bash -eu -o pipefail

poetry run pre-commit run --all-files
poetry run mypy
poetry run pylama
poetry run darglint2 -v2 src
