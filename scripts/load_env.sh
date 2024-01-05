#!/usr/bin/env -S bash -eu -o pipefail

# see https://stackoverflow.com/questions/43267413/how-to-set-environment-variables-from-env-file
set -a # automatically export all variables
source .env
set +a