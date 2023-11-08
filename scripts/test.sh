#!/usr/bin/env -S bash -eu -o pipefail

poetry run pytest -n 10
