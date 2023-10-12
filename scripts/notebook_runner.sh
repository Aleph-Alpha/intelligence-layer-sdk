#!/bin/bash
#AA_TOKEN=$(echo $AA_TOKEN) 
source .env
AA_TOKEN=$(echo $AA_TOKEN) jupyter nbconvert --to notebook --execute src/intelligence_layer/*.ipynb
rm src/intelligence_layer/*.nbconvert.ipynb