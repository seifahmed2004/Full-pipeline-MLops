FROM python:3.10-slim

WORKDIR /app

ARG RUN_ID
ENV RUN_ID=$RUN_ID

COPY . /app

CMD sh -c 'echo "Simulating model download for Run ID: $RUN_ID"'