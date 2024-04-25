#!/usr/bin/env -S bash -eu -o pipefail

# start the server in the background
hypercorn src/examples/fastapi_example:app --bind localhost:8000 &
server_pid=$!

attempt_counter=0
max_attempts=10

trap 'kill $server_pid' EXIT SIGINT
# waiting for server startup
until $(curl -X GET http://localhost:8000 --fail-with-body --output /dev/null --silent --head); do
    if [ ${attempt_counter} -eq ${max_attempts} ];then
      echo "Max attempts reached"
      exit 1
    fi

    printf '.'
    attempt_counter=$(($attempt_counter+1))
    sleep 1
done

curl -X GET http://localhost:8000 --fail-with-body
curl -X POST http://localhost:8000/summary --fail-with-body -H "Content-Type: application/json" -d '{"chunk": "<your text to summarize here>", "language": {"iso_639_1": "en"}}'

# kill happens at the end with the trap command
exit 0
