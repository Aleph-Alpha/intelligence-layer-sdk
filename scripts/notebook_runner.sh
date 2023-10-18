#!/usr/bin/env -S bash -eu -o pipefail
# next line loads AA_TOKEN from .env file when running bash script locally. In CI this is not necessary since AA_TOKEN is environment variable.
[ -f .env ] && source .env
poetry config installer.max-workers 10 && AA_TOKEN="$AA_TOKEN" jupyter nbconvert --to notebook --execute src/examples/*.ipynb
rm src/examples/*.nbconvert.ipynb
