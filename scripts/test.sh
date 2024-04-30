#!/usr/bin/env -S bash -eu -o pipefail

poetry run python3 -c "import nltk; nltk.download('punkt')"
poetry run pytest -n 10
