#!/usr/bin/env -S bash -eu -o pipefail

ProjectRoot="$(cd $(dirname "$0")/.. && pwd -P)"

cd "$ProjectRoot"

./scripts/lint.sh
(cd docs  && poetry run make doctest)
./scripts/notebook_runner.sh
./scripts/test.sh
