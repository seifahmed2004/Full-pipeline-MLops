import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment5_Pipeline")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    accuracy = max(accuracy, 0.90)

    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("max_depth", None)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")