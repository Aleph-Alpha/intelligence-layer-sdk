#!/usr/bin/env -S bash -eu -o pipefail

poetry run pre-commit run --all-files
poetry run mypy
poetry run pylama
