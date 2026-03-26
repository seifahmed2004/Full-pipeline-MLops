# Full-Pipeline-MLops

A complete MLOps pipeline that trains, validates, and deploys a machine learning model using **MLflow**, **DVC**, **GitHub Actions**, and **Docker**.

---

## Project Structure
```
Full-pipeline-MLops/
├── .github/
│   └── workflows/
│       └── pipeline.yml       # CI/CD pipeline definition
├── .dvc/                      # DVC configuration
├── data/
│   ├── iris.csv               # Training dataset
│   └── .gitignore
├── dvc_remote/                # Local DVC remote storage
├── dvc_storage/               # DVC cache
├── mlartifacts/               # MLflow model artifacts
├── train.py                   # Model training script
├── check_threshold.py         # Accuracy validation script
├── save_iris.py               # Dataset generation script
├── Dockerfile                 # Docker image definition
├── requirements.txt           # Python dependencies
├── mlflow.db                  # MLflow SQLite tracking database
└── README.md
```

---

## Pipeline Overview

The GitHub Actions pipeline runs automatically on every push to `main` or pull request. It consists of two sequential jobs:
```
Push to main
     │
     ▼
[validate]  →  Train model, log to MLflow, upload artifacts
     │
     ▼ (only if validate passes)
[deploy]    →  Check accuracy threshold, build Docker image
```

---

## Jobs

### 1. `validate`

| Step | Description |
|------|-------------|
| Checkout repository | Clones the repo into the runner |
| Set up Python 3.10 | Installs Python on the runner |
| Install dependencies | Runs `pip install -r requirements.txt` |
| Pull dataset with DVC | Tries DVC pull; falls back to committed `iris.csv` |
| Train model | Runs `train.py`, logs to MLflow, writes `model_info.txt` |
| Archive MLflow runs | Compresses `mlruns/` into `mlruns.tar.gz` |
| Upload artifacts | Uploads `model_info.txt` and `mlruns.tar.gz` for the deploy job |

### 2. `deploy`

| Step | Description |
|------|-------------|
| Download artifacts | Downloads `model_info.txt` and `mlruns.tar.gz` from validate job |
| Extract MLflow runs | Decompresses `mlruns.tar.gz` |
| Check accuracy threshold | Runs `check_threshold.py` — fails pipeline if accuracy is too low |
| Build Docker image | Builds image with the MLflow `run_id` as a build argument |

---

## Model Training (`train.py`)

- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Model**: `RandomForestClassifier` with 500 estimators
- **Split**: 80% train / 20% test with stratification
- **Tracking**: MLflow logs params, metrics, and the serialized model
- **Output**: `model_info.txt` containing:
```json
{
  "experiment_id": "1",
  "run_id": "abc123...",
  "accuracy": 0.9667
}
```

---

## Accuracy Threshold (`check_threshold.py`)

Reads `model_info.txt` and exits with code 1 if accuracy is below the defined threshold, blocking deployment automatically.

---

## Setup & Local Usage

### Prerequisites

- Python 3.10+
- Docker
- DVC (optional)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Generate dataset
```bash
python save_iris.py
```

### Train locally
```bash
python train.py
```

### Validate accuracy locally
```bash
python check_threshold.py
```

---

## GitHub Secrets

| Secret | Description |
|--------|-------------|
| `MLFLOW_TRACKING_URI` | URI for your MLflow tracking server |

To add secrets: **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**

---

## Docker

The deploy job builds a Docker image and passes the MLflow `run_id` as a build argument:
```bash
docker build --build-arg RUN_ID=<run_id> -t my-model:latest .
```

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| [MLflow](https://mlflow.org/) | Experiment tracking and model logging |
| [DVC](https://dvc.org/) | Data version control |
| [GitHub Actions](https://github.com/features/actions) | CI/CD automation |
| [Docker](https://www.docker.com/) | Model containerization |
| [scikit-learn](https://scikit-learn.org/) | Machine learning |
| [pandas](https://pandas.pydata.org/) | Data manipulation |
