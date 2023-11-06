#!/usr/bin/env -S bash -eu -o pipefail
# next line loads AA_TOKEN from .env file when running bash script locally. In CI this is not necessary since AA_TOKEN is environment variable.
[ -f .env ] && source .env
# Find all .ipynb files in the directory and pass them to xargs for parallel execution
rm -rf src/examples/*.nbconvert.ipynb src/examples/.ipynb_checkpoints
find src/examples/ -name "*.ipynb" | xargs -I {} --max-procs 10 bash -c "AA_TOKEN=\"$AA_TOKEN\" poetry run jupyter nbconvert --to notebook --execute {}"
rm -f src/examples/*.nbconvert.ipynb
