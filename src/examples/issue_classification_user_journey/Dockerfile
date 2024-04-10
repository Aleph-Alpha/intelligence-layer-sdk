FROM ubuntu:latest as builder

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y python3 python3-pip python3-venv git

RUN mkdir /app
COPY requirements.txt /app/requirements.txt

RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN --mount=type=secret,id=GITHUB_TOKEN \
    GITHUB_TOKEN=$(cat /run/secrets/GITHUB_TOKEN) pip install -r /app/requirements.txt


FROM ubuntu:latest as runtime

# Delete apt package lists immediately to save ~45MB. This has to be done in the same RUN command,
# otherwise the data is already persisted in another layer.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y python3 \
    && rm -r /var/lib/apt/lists/*

COPY --from=builder /app /app
COPY main.py /app/main.py

ENV PATH="/app/venv/bin:$PATH"
ENTRYPOINT [ "uvicorn", "--host", "0.0.0.0", "--port", "80", "app.main:app" ]
