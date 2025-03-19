#!/usr/bin/env -S bash -eu -o pipefail

TQDM_DISABLE=1 poetry run pytest -n 5 -m "not document_index"
