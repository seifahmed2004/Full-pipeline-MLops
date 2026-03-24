import os
import mlflow

THRESHOLD = 0.85


def main():
    if not os.path.exists("model_info.txt"):
        raise FileNotFoundError("model_info.txt not found.")

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    if not run_id:
        raise ValueError("model_info.txt is empty.")

    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy")

    if accuracy is None:
        raise ValueError("Accuracy metric not found in MLflow run.")

    print(f"Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")

    if accuracy < THRESHOLD:
        raise SystemExit(f"Model accuracy {accuracy:.4f} is below threshold {THRESHOLD:.2f}")

    print("Threshold check passed. Proceeding to deployment.")


if __name__ == "__main__":
    main()