#!/usr/bin/env -S bash -eu -o pipefail
# next line loads AA_TOKEN from .env file when running bash script locally. In CI this is not necessary since AA_TOKEN is environment variable.
[ -f .env ] && source .env
export AA_TOKEN
# Find all .ipynb files in the directory and pass them to xargs for parallel execution
rm -rf src/examples/.ipynb_checkpoints
rm -rf src/examples/how_tos/.ipynb_checkpoints

find src/examples -name "*.nbconvert.ipynb" -type f -delete
find src/examples -name "*.ipynb" ! -name "performance_tips.ipynb" | xargs --max-args 1 --max-procs 6 poetry run jupyter nbconvert --to notebook --execute
find src/examples -name "*.nbconvert.ipynb" -type f -delete

./scripts/fastapi_example_test.sh
