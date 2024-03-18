#!/usr/bin/env -S bash -eu -o pipefail

poetry run pre-commit run --all-files --show-diff-on-failure
poetry run mypy
poetry run pylama
