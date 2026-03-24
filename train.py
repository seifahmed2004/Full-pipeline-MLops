import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# FORCE LOCAL STORAGE (ignore secret issues)
mlflow.set_tracking_uri("file:./mlruns")

# IMPORTANT: create experiment explicitly
mlflow.set_experiment("Assignment5_Pipeline")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    run_id = run.info.run_id

    # SAVE run id
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print("Run ID:", run_id)
    print("Accuracy:", accuracy)