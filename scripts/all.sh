#!/usr/bin/env -S bash -eu -o pipefail

ProjectRoot="$(cd $(dirname "$0")/.. && pwd -P)"

cd "$ProjectRoot"

./scripts/load_env.sh
./scripts/lint.sh
./scripts/doctest.sh
./scripts/notebook_runner.sh
./scripts/test.sh
