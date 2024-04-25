#!/usr/bin/env -S bash -eu -o pipefail

# start the server in the background
hypercorn src/examples/fastapi_example:app --bind localhost:8000 &
my_pid=$!

trap 'kill $my_pid' EXIT SIGINT
# waiting for server startup
sleep 3
curl -X GET http://localhost:8000 --fail-with-body
curl -X POST http://localhost:8000/summary --fail-with-body -H "Content-Type: application/json" -d '{"chunk": "<your text to summarize here>", "language": {"iso_639_1": "en"}}'

# kill happens at the end with the trap command
exit 0
