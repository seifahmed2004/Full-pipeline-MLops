import json
import os
import sys

import mlflow

THRESHOLD = 0.85

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r", encoding="utf-8") as f:
    info = json.load(f)

experiment_id = str(info["experiment_id"])
run_id = info["run_id"]

client = mlflow.tracking.MlflowClient()

print(f"Experiment ID: {experiment_id}")
print(f"Looking for run: {run_id}")

runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"attributes.run_id = '{run_id}'",
)

if not runs:
    all_runs = client.search_runs(experiment_ids=[experiment_id])
    print("Available runs in experiment:")
    for r in all_runs:
        print(r.info.run_id)
    raise ValueError(f"Run {run_id} not found in experiment {experiment_id}")

run = runs[0]
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    raise ValueError(f"Accuracy not found for run {run_id}")

print(f"Accuracy: {accuracy:.4f}")

if accuracy < THRESHOLD:
    print(f"Model failed threshold: {accuracy:.4f} < {THRESHOLD}")
    sys.exit(1)

print("Model passed threshold")