
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
MLFLOW_DB = ROOT / "mlflow.db"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]



def load_data():
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )

    X = df.drop(columns="num").astype("float64")
    y = (df["num"] > 0).astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test



def get_models():
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                random_state=42,
            )),
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
            )),
        ]),

        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", CalibratedClassifierCV(
                estimator=SVC(class_weight="balanced", random_state=42),
                cv=5,
                ensemble=False,
            )),
        ]),
    }


def main():
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment("heart-disease-classification")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    for name, model in get_models().items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model_name": name,
                "test_size": 0.20,
                "random_state": 42,
                "cv_folds": 5,
            })

            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                },
                n_jobs=-1,
                error_score="raise",
            )

            for metric in [
                "accuracy", "precision", "recall", "f1", "roc_auc"
            ]:
                mlflow.log_metric(
                    f"cv_{metric}_mean",
                    float(scores[f"test_{metric}"].mean()),
                )
                mlflow.log_metric(
                    f"cv_{metric}_std",
                    float(scores[f"test_{metric}"].std()),
                )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            test_metrics = {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "test_recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "test_f1": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }

            mlflow.log_metrics({
                key: float(value)
                for key, value in test_metrics.items()
            })

            # Ensure MLflow receives floating-point feature
            model_input= X_train.head(5).astype("float64")
            # Explicitly infer the input/output schema
            signature = mlflow.models.infer_signature(model_input, model.predict(model_input),
            )
           
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=model_input,
                signature=signature,
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.calibration._SigmoidCalibration",
                ],
            )

            print(f"\n{name}")
            print({
                key: round(float(value), 3)
                for key, value in test_metrics.items()
            })
            print("Parameters, metrics, and model logged to MLflow.")

    print(f"\nMLflow database: {MLFLOW_DB}")
    print("All model experiments completed successfully.")


if __name__ == "__main__":
    main()
