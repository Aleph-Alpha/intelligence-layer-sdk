#!/usr/bin/env -S bash -eu -o pipefail

cd $(dirname $0)/..

poetry run pre-commit run --all-files
poetry run mypy .
