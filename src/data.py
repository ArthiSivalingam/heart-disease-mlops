
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/heart-disease/"
    "processed.cleveland.data"
)

DATA_FILE = RAW_DIR / "processed.cleveland.data"
METADATA_FILE = RAW_DIR / "dataset_metadata.json"

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]


def validate_dataset(file_path: Path) -> pd.DataFrame:
    """Validate the raw Cleveland dataset without modifying it."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise ValueError(f"Dataset is missing or empty: {file_path}")

    df = pd.read_csv(
        file_path,
        header=None,
        names=EXPECTED_COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )

    if df.shape[1] != 14:
        raise ValueError(
            f"Expected 14 columns, but found {df.shape[1]}."
        )

    if len(df) != 303:
        raise ValueError(
            f"Expected 303 Cleveland records, but found {len(df)}. "
            "Verify the source file before continuing."
        )

    if df["num"].isna().any():
        raise ValueError("The target column contains missing values.")

    if not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected target values were found.")

    return df


def download_dataset() -> Path:
    """Download, validate, and record provenance for the raw dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        request = Request(
            DATA_URL,
            headers={"User-Agent": "HeartDiseaseMLOpsAssignment/1.0"},
        )

        print("Downloading the official UCI Cleveland dataset...")

        try:
            with urlopen(request, timeout=30) as response:
                content = response.read()
        except Exception as exc:
            raise RuntimeError(
                "Dataset download failed. Check your internet connection "
                "and the UCI source URL. No dataset has been processed."
            ) from exc

        if not content:
            raise RuntimeError("The downloaded file is empty.")

        temporary_file = DATA_FILE.with_suffix(".tmp")
        temporary_file.write_bytes(content)

        try:
            validate_dataset(temporary_file)
            temporary_file.replace(DATA_FILE)
        except Exception:
            temporary_file.unlink(missing_ok=True)
            raise

    # Validate existing files too; do not silently trust them.
    df = validate_dataset(DATA_FILE)
    checksum = sha256(DATA_FILE.read_bytes()).hexdigest()

    metadata = {
        "dataset": "Heart Disease - Cleveland processed data",
        "source_url": DATA_URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_name": DATA_FILE.name,
        "sha256": checksum,
        "rows": len(df),
        "columns": EXPECTED_COLUMNS,
        "missing_values_per_column": {
            column: int(count)
            for column, count in df.isna().sum().items()
        },
        "target_values": sorted(
            int(value) for value in df["num"].unique()
        ),
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print(f"Dataset validated: {len(df)} rows, {len(df.columns)} columns")
    print(f"Raw file: {DATA_FILE}")
    print(f"SHA-256: {checksum}")
    print(f"Metadata: {METADATA_FILE}")

    return DATA_FILE


if __name__ == "__main__":
    download_dataset()
