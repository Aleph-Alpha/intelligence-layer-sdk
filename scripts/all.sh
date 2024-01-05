#!/usr/bin/env -S bash -eu -o pipefail

ProjectRoot="$(cd $(dirname "$0")/.. && pwd -P)"

cd "$ProjectRoot"

# see https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file
set -a # automatically export all variables
source .env
set +a

./scripts/load_env.sh
./scripts/lint.sh
./scripts/doctest.sh
./scripts/notebook_runner.sh
./scripts/test.sh
