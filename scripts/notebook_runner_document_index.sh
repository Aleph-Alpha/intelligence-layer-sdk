#!/usr/bin/env -S bash -eu -o pipefail

# Load environment variables if running locally
[ -f .env ] && source .env
export AA_TOKEN

# Remove Jupyter Notebook checkpoints
rm -rf src/documentation/.ipynb_checkpoints
rm -rf src/documentation/how_tos/.ipynb_checkpoints

# Remove any previously executed version of the notebook
find src/documentation -name "document_index.nbconvert.ipynb" -type f -delete

# Execute only document_index.ipynb
poetry run jupyter nbconvert --to notebook --execute src/documentation/document_index.ipynb

# Remove the execution-generated file
find src/documentation -name "document_index.nbconvert.ipynb" -type f -delete

