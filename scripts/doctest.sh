#!/usr/bin/env -S bash -eu -o pipefail

ProjectRoot="$(cd $(dirname "$0")/.. && pwd -P)"

cd "$ProjectRoot"

if [ -f .env ]; then
    # Export environment variables from .env file
    set -a # automatically export all variables
    source .env
    set +a
else
    export CLIENT_URL="https://api.aleph-alpha.com"
fi
(cd docs && poetry run make doctest)
