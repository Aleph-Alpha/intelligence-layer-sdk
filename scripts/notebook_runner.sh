#!/usr/bin/env -S bash -eu -o pipefail
# next line loads AA_TOKEN from .env file when running bash script locally. In CI this is not necessary since AA_TOKEN is environment variable.
[ -f .env ] && source .env
export AA_TOKEN
# Find all .ipynb files in the directory and pass them to xargs for parallel execution
rm -rf src/documentation/.ipynb_checkpoints
rm -rf src/documentation/how_tos/.ipynb_checkpoints

find src/documentation -name "*.nbconvert.ipynb" -type f -delete
find src/documentation -name "*.ipynb" ! -name "performance_tips.ipynb" | xargs --max-args 1 --max-procs 6 poetry run jupyter nbconvert --to notebook --execute
find src/documentation -name "*.nbconvert.ipynb" -type f -delete

poetry run ./scripts/fastapi_example_test.sh
