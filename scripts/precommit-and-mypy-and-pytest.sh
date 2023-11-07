#!/usr/bin/env -S bash -eu -o pipefail
pre-commit run --all-files
mypy
pylama
pytest -n 10
