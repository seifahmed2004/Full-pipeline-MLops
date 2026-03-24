import os
import sys
import mlflow

THRESHOLD = 0.85

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(tracking_uri)

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

if "accuracy" not in run.data.metrics:
    raise ValueError(f"No 'accuracy' metric found for run {run_id}")

accuracy = run.data.metrics["accuracy"]

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy:.4f}")

if accuracy < THRESHOLD:
    print(f"Model failed threshold: {accuracy:.4f} < {THRESHOLD}")
    sys.exit(1)

print("Model passed threshold")