import os
import json
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # Load dataset pulled by DVC
    data_path = "data/iris.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Run 'dvc pull' first.")

    df = pd.read_csv(data_path)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_estimators = 100
    max_depth = 5
    random_state = 42

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)

        os.makedirs("artifacts", exist_ok=True)

        # model_info.txt contains only the Run ID, as requested
        with open("model_info.txt", "w") as f:
            f.write(run.info.run_id)

        # optional local info file for debugging/reporting
        with open("artifacts/run_summary.json", "w") as f:
            json.dump(
                {
                    "run_id": run.info.run_id,
                    "accuracy": accuracy
                },
                f,
                indent=2
            )

        print(f"Run ID: {run.info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()