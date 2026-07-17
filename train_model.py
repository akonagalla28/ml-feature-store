"""
Trains a baseline ETA regression model on the materialized, point-in-time-correct
training set, and logs the run (params, metrics, model artifact) to MLflow.

Run: python pipelines/train_model.py
"""

from __future__ import annotations

import pathlib

import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "training" / "eta_training_set.parquet"

FEATURE_COLUMNS = [
    "avg_prep_time_minutes_7d",
    "order_volume_7d",
    "prep_time_stddev_7d",
    "acceptance_rate_7d",
    "avg_delivery_time_minutes_7d",
]
TARGET_COLUMN = "actual_eta_minutes"


def main() -> None:
    df = pd.read_parquet(DATA_PATH).dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("eta-prediction")
    with mlflow.start_run():
        params = {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.1}
        mlflow.log_params(params)
        mlflow.log_param("feature_columns", FEATURE_COLUMNS)
        mlflow.log_param("training_rows", len(X_train))

        model = GradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        mlflow.log_metric("mae_minutes", mae)
        mlflow.log_metric("rmse_minutes", rmse)
        mlflow.sklearn.log_model(model, "model")

        print(f"MAE: {mae:.2f} minutes | RMSE: {rmse:.2f} minutes")
        print(f"Logged to MLflow experiment 'eta-prediction', run_id={mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
