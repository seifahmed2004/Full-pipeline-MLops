import os
import sys
import mlflow

THRESHOLD = 0.85

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

print(f"Looking for run: {run_id}")

run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    raise ValueError(f"Accuracy not found for run {run_id}")

print(f"Accuracy: {accuracy:.4f}")

if accuracy < THRESHOLD:
    print(f"Model failed threshold: {accuracy:.4f} < {THRESHOLD}")
    sys.exit(1)

print("Model passed threshold")