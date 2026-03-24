import os
import sys
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()

print(f"Looking for run: {run_id}")

# DEBUG: print all runs
experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    for r in runs:
        print("Available run:", r.info.run_id)

run = client.get_run(run_id)

accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    raise ValueError("Accuracy not found")

print(f"Accuracy: {accuracy:.4f}")

if accuracy < 0.85:
    print("FAILED threshold")
    sys.exit(1)

print("PASSED threshold")