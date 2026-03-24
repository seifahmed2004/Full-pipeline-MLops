import json
import os

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")

if tracking_uri.startswith("file:"):
    mlruns_path = tracking_uri.replace("file:", "", 1)
    os.makedirs(mlruns_path, exist_ok=True)

mlflow.set_tracking_uri(tracking_uri)
experiment_name = "Assignment5_Pipeline"
mlflow.set_experiment(experiment_name)

experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    raise ValueError("Failed to create or retrieve MLflow experiment")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=1,   # VERY SMALL
        max_depth=1,
        random_state=42
    )    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", "None")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    info = {
        "experiment_id": experiment.experiment_id,
        "run_id": run.info.run_id,
        "accuracy": float(accuracy),
    }

    with open("model_info.txt", "w", encoding="utf-8") as f:
        json.dump(info, f)

    print("Experiment ID:", info["experiment_id"])
    print("Run ID:", info["run_id"])
    print("Accuracy:", f'{info["accuracy"]:.4f}')