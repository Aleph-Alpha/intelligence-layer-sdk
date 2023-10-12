#!/bin/bash
# next line loads AA_TOKEN from .env file when running bash script locally. In CI this is not necessary since AA_TOKEN is environment variable.
[ -f .env ] && source .env
AA_TOKEN=$(echo $AA_TOKEN) jupyter nbconvert --to notebook --execute src/intelligence_layer/*.ipynb
rm src/intelligence_layer/*.nbconvert.ipynb
