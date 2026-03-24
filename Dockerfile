FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN echo "Downloading model for Run ID: ${RUN_ID}"

CMD ["sh", "-c", "echo Model container started for Run ID: ${RUN_ID}"]