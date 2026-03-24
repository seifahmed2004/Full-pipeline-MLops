import mlflow
import os

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

# Get latest experiment
experiment = mlflow.get_experiment_by_name("Assignment5_Pipeline")

if experiment is None:
    raise Exception("Experiment not found!")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

if runs.empty:
    raise Exception("No runs found!")

# Get latest run
latest_run = runs.iloc[0]

accuracy = latest_run["metrics.accuracy"]

print(f"Model accuracy: {accuracy}")

# Threshold check
if accuracy < 0.85:
    raise Exception("Model accuracy below threshold!")

print("✅ Model passed threshold")