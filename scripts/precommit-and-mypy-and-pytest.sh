#!/usr/bin/env -S bash -eu -o pipefail
mypy
pre-commit run --all-files
pytest
