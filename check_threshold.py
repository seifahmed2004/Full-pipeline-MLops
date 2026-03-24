import os
import sys
import mlflow

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

print("Looking for run:", run_id)

# DEBUG: list runs
experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    for r in runs:
        print("Found run:", r.info.run_id)

run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", None)

if accuracy is None:
    raise ValueError("Accuracy not found")

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("FAILED threshold")
    sys.exit(1)

print("PASSED threshold")