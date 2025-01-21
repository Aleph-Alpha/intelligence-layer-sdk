#!/usr/bin/env bash

set -euo pipefail

ID_TOKEN=$(curl -sLS -H "User-Agent: actions/oidc-client" -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
    "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=https://alephalpha.jfrog.io" | jq .value | tr -d '"')

JFROG_ACCESS_TOKEN=$(curl -v \
    -X POST \
    -H "Content-type: application/json" \
    https://alephalpha.jfrog.io/access/api/v1/oidc/token \
    -d \
    "{\"grant_type\": \"urn:ietf:params:oauth:grant-type:token-exchange\", \"subject_token_type\":\"urn:ietf:params:oauth:token-type:id_token\", \"subject_token\": \"$ID_TOKEN\", \"provider_name\": \"github\"}" | jq .access_token -r)

echo -n $JFROG_ACCESS_TOKEN
