# Heart Disease Classification — MLOps Project

## 1. Project Overview

This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for heart disease classification using the UCI Cleveland Heart Disease dataset.

The project covers data quality analysis, data preprocessing, model training, experiment tracking with MLflow, REST API development, automated testing, Docker containerization, Kubernetes deployment, and application monitoring.

**Objective:** To develop a reproducible machine learning pipeline and deploy a classification model through a REST API.

**Disclaimer:** This project is intended for educational purposes only. Its predictions are not a substitute for professional medical advice, diagnosis, or treatment.

## 2. Technologies Used

- Python 3.12
- Pandas
- scikit-learn
- MLflow
- FastAPI
- Pytest
- Ruff
- Docker
- Kubernetes
- GitHub Actions
- Prometheus-compatible application metrics

## 3. Dataset Description

The project uses the processed Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

Dataset summary:

- Number of records: 303
- Number of columns: 14, including the target variable
- Duplicate rows identified: 0
- Missing values identified in `ca`: 4
- Missing values identified in `thal`: 2
- Original target variable: `num`

For binary classification, the target is mapped as follows:

- `0`: No detected heart disease
- Values greater than `0`: Heart disease present

## 4. Data Preprocessing and Model Development

The workflow includes data quality checks, handling missing values, preparing features and the target variable, splitting data for training and testing, and evaluating classification models.

The following machine learning algorithms are evaluated:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

Model performance is assessed using accuracy, precision, recall, F1-score, and ROC-AUC.

## 5. Experiment Tracking with MLflow

MLflow is used to track model experiments, parameters, evaluation metrics, and model artifacts.

**Experiment name:** `heart-disease-classification`

The recorded experiments allow model performance to be compared and reviewed.

## 6. Model Evaluation Results

The following are the recorded test metrics for the three evaluated models.

| Metric | Logistic Regression | Random Forest | SVM |
|---|---:|---:|---:|
| Test accuracy | 0.803 | 0.918 | 0.803 |
| Test precision | 0.722 | 0.871 | 0.722 |
| Test recall | 0.929 | 0.964 | 0.929 |
| Test F1-score | 0.812 | 0.915 | 0.812 |
| Test ROC-AUC | 0.916 | 0.961 | 0.929 |

Random Forest achieved the highest recorded test accuracy and ROC-AUC among these three models.

These results describe the evaluation performed in this project and do not establish clinical effectiveness.

## 7. REST API Development

The project provides a REST API for application health checks, predictions, and metrics.

The documented endpoints are:

| Endpoint | Purpose |
|---|---|
| `/` | Application information |
| `/health` | Application health check |
| `/predict` | Generate a model prediction |
| `/metrics` | Expose application metrics |

When the API is running, interactive API documentation can be accessed at:

`http://127.0.0.1:8000/docs`

The actual application entry point and request schema should be checked in the source code before running the API.

## 8. Automated Testing and Continuous Integration

Pytest is used for automated testing, and Ruff is used for linting and code-quality checks.

GitHub Actions is configured to automate the project's validation workflow, including the configured linting, testing, and model-training steps.

## 9. Docker Containerization

Docker is used to package the application and its dependencies into a container image.

The container can be used to run the API in a consistent environment and validate its health and prediction endpoints.

The repository includes a `Dockerfile` and a `.dockerignore` file.

## 10. Kubernetes Deployment

Kubernetes configuration is included to deploy and expose the heart disease classification API.

The deployment can be validated by checking:

- Deployment and pod status
- Pod readiness
- Service configuration
- API health-check response
- Prediction endpoint response
- Application metrics endpoint

The repository includes Kubernetes deployment configuration.

## 11. Monitoring and Logging

The application exposes a Prometheus-compatible metrics endpoint at `/metrics`.

This endpoint provides application metrics that can be inspected to help understand API activity and operational behavior.

The project also includes evidence of Kubernetes application status and application logs.

## 12. Project Structure

The main project files and folders are organized as follows:

```text
heart-disease-mlops/
├── src/
├── tests/
│   └── test_api.py
├── reports/
│   └── screenshots/
├── .github/
│   └── workflows/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

Additional project files may be present in the repository.

## 13. Setup and Execution

### Prerequisites

- Python 3.12
- Git
- Required Python packages listed in `requirements.txt`
- Docker, if running the containerized application
- A Kubernetes environment, if running the deployment

### Install dependencies

Activate the project's virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

### Run data processing

If the data module is configured as an executable module, run:

```bash
python -m src.data
```

### Train and evaluate models

Run:

```bash
python -m src.train
```

### Run automated tests

Run:

```bash
python -m pytest -v
```

### Run linting

Run:

```bash
python -m ruff check .
```

### Start the API

Use the actual FastAPI module and application-object name defined in the repository. For example, if `app.py` at the repository root contains a FastAPI object named `app`, the command would be:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

If the application is defined in a different module, adjust the command accordingly.

After starting the API, open `http://127.0.0.1:8000/docs` in a browser to access its interactive documentation.

## 14. Evidence and Documentation

Project execution evidence is maintained under `reports/screenshots/`.

The screenshots document relevant stages of the workflow, including MLflow experiments, API operation, Docker execution, Kubernetes deployment, and application metrics.

## 15. Conclusion

This project demonstrates an end-to-end MLOps workflow for a heart disease classification application. It brings together machine learning development, experiment tracking, automated validation, API serving, containerization, deployment, and monitoring.

The project provides practical experience in building and operating a reproducible machine learning application..
