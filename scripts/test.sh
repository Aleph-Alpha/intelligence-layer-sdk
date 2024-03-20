#!/usr/bin/env -S bash -eu -o pipefail

poetry run python3 -c "import nltk; nltk.download('punkt')"
#python3 -m nltk.downloader punkt
poetry run pytest -n 10 tests/evaluation/test_graders.py
