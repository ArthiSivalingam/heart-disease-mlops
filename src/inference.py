
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = ROOT / "mlflow.db"
MODEL_DIR = ROOT / "models"


def main():
    # Connect to the project's MLflow database.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(
        "heart-disease-classification"
    )

    if experiment is None:
        raise RuntimeError("MLflow experiment was not found.")

    # Find completed Random Forest runs.
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            "and tags.mlflow.runName = 'Random Forest'"
        ),
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise RuntimeError(
            "No successful Random Forest run was found."
        )

    # Select the latest successful run.
    selected_run = runs[0]
    run_id = selected_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    print("Loading model from MLflow...")
    model = mlflow.sklearn.load_model(model_uri)

    # Use a new directory to avoid Windows file-locking issues.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"heart_disease_model_{timestamp}"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model without deleting existing model folders.
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_path),
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.tree._tree.Tree",
        ],
    )

    print("\nModel saved successfully.")
    print(f"Selected run: {run_id}")
    print(
        "Test accuracy:",
        selected_run.data.metrics.get("test_accuracy"),
    )
    print(f"Model location: {model_path}")


if __name__ == "__main__":
    main()
