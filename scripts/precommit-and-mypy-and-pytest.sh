#!/usr/bin/env -S bash -eu -o pipefail
mypy
pytest -o addopts=""
