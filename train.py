from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"
DB_PATH = ROOT / "mlflow.db"

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
ZIP_PATH = RAW_DIR / "student-performance.zip"
CSV_PATH = RAW_DIR / "student-mat.csv"

NUMERIC_FEATURES = ["studytime", "failures", "absences", "G1", "G2"]
CATEGORICAL_FEATURES = ["schoolsup", "famsup", "internet"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "G3"


def download_dataset() -> Path:
    """Download and extract the UCI Student Performance dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        return CSV_PATH

    print("Downloading the UCI Student Performance dataset...")

    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

        # The UCI archive may contain another ZIP, so extract and search recursively.
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(RAW_DIR)

        csv_files = list(RAW_DIR.rglob("student-mat.csv"))

        if not csv_files:
            nested_zip_files = [
                path
                for path in RAW_DIR.rglob("*.zip")
                if path != ZIP_PATH
            ]

            for nested_zip in nested_zip_files:
                try:
                    with zipfile.ZipFile(nested_zip, "r") as archive:
                        archive.extractall(RAW_DIR)
                except zipfile.BadZipFile:
                    print(f"Skipping invalid ZIP file: {nested_zip}")

            csv_files = list(RAW_DIR.rglob("student-mat.csv"))

        if not csv_files:
            raise FileNotFoundError(
                "student-mat.csv was not found after extracting the downloaded archives."
            )

        source_csv = csv_files[0]
        if source_csv.resolve() != CSV_PATH.resolve():
            shutil.copy2(source_csv, CSV_PATH)

        print(f"Dataset downloaded successfully: {CSV_PATH}")
        return CSV_PATH

    except Exception as exc:
        raise RuntimeError(
            "Dataset download/extraction failed. Check your internet connection. "
            "If needed, download the UCI Student Performance ZIP manually and place "
            "student-mat.csv in data/raw/."
        ) from exc


def load_and_prepare_data() -> pd.DataFrame:
    """Load, validate, clean, and save the selected dataset columns."""
    csv_path = download_dataset()
    df = pd.read_csv(csv_path, sep=";")

    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df[FEATURES + [TARGET]].copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET])
    df = df[
        df["G1"].between(0, 20)
        & df["G2"].between(0, 20)
        & df[TARGET].between(0, 20)
    ]

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].astype(str).str.strip().str.lower()

    df = df.reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "student_performance_clean.csv", index=False)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipelines for numerical and categorical columns."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def main() -> None:
    """Train, evaluate, track, and save the best model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data()
    X = df[FEATURES]
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Use SQLite for MLflow tracking instead of the legacy file-store backend.
    tracking_uri = f"sqlite:///{DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Student-Performance-Prediction")

    model_factories = {
        "RandomForestRegressor": lambda: RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTreesRegressor": lambda: ExtraTreesRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": lambda: GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            loss="huber",
            random_state=42,
        ),
    }

    results = []
    best_name = None
    best_rmse = float("inf")

    print(f"Dataset rows after cleaning: {len(df)}")
    print("Training and comparing models. This may take a minute...")

    for model_name, factory in model_factories.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", factory()),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("feature_count", len(FEATURES))
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model",
    serialization_format="cloudpickle"
)

            results.append(
                {
                    "model": model_name,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "run_id": run.info.run_id,
                }
            )

            print(
                f"{model_name}: RMSE={rmse:.3f}, "
                f"MAE={mae:.3f}, R2={r2:.3f}"
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = model_name

    # Refit the selected model on all cleaned data for the deployed application.
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model_factories[best_name]()),
        ]
    )
    final_pipeline.fit(X, y)

    model_path = MODEL_DIR / "student_performance_model.joblib"
    joblib.dump(final_pipeline, model_path)

    results_sorted = sorted(results, key=lambda item: item["rmse"])
    best_result = next(item for item in results if item["model"] == best_name)

    metadata = {
        "project": "Student Performance Prediction",
        "dataset": "UCI Student Performance (student-mat.csv)",
        "target": TARGET,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "selected_model": best_name,
        "selection_metric": "lowest RMSE on held-out test set",
        "best_test_metrics": best_result,
        "all_test_results": results_sorted,
        "dataset_rows": int(len(df)),
        "train_test_split": "80/20, random_state=42",
        "grade_scale": "0 to 20",
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    display_results = pd.DataFrame(results_sorted).drop(columns=["run_id"])
    print("\nModel comparison (lower RMSE/MAE is better; higher R2 is better):")
    print(display_results.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved deployable model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"\nMLflow tracking URI: {tracking_uri}")
    print("To open MLflow UI, run in another terminal:")
    print(
        "  mlflow ui --backend-store-uri "
        f"sqlite:///{DB_PATH.as_posix()} --host 127.0.0.1 --port 5000"
    )


if __name__ == "__main__":
    main()
