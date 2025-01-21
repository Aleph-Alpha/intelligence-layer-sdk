#!/usr/bin/env bash

set -euo pipefail

JFROG_ACCESS_TOKEN=$1
echo $JFROG_ACCESS_TOKEN | awk -F'.' '{print $2}'  | sed 's/.\{1,3\}$/&==/' | base64 -d | jq '.sub' -r
