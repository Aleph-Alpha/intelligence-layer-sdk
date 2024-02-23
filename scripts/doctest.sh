#!/usr/bin/env -S bash -eu -o pipefail

ProjectRoot="$(cd $(dirname "$0")/.. && pwd -P)"

cd "$ProjectRoot"

(set -a && source .env && set +a && cd docs && poetry run make doctest)
