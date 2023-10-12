#!/bin/bash
mypy
pre-commit run --all-files
pytest
