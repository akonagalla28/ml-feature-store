"""
Builds a training dataset for the ETA model using point-in-time-correct joins.

Each label row (a completed delivery with a known actual_eta_minutes) is joined
against the feature snapshot that was valid AT THAT MOMENT -- not the latest
feature values -- which is what a naive `pd.merge` on entity_id would give you
and would leak future information into the model.

Run: python pipelines/generate_training_data.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feature_store.offline_store.offline_store import OfflineStore

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "training" / "eta_training_set.parquet"


def main() -> None:
    labels = pd.read_parquet(RAW_DIR / "training_labels.parquet").reset_index(drop=True)
    labels["_row_id"] = labels.index
    offline = OfflineStore()

    restaurant_features = offline.point_in_time_join(
        "restaurant_prep_time_features",
        labels[["_row_id", "restaurant_id", "event_timestamp"]],
        entity_key="restaurant_id",
    ).sort_values("_row_id").reset_index(drop=True)

    courier_features = offline.point_in_time_join(
        "courier_performance_features",
        labels[["_row_id", "courier_id", "event_timestamp"]],
        entity_key="courier_id",
    ).sort_values("_row_id").reset_index(drop=True)

    restaurant_feature_cols = ["avg_prep_time_minutes_7d", "order_volume_7d", "prep_time_stddev_7d"]
    courier_feature_cols = ["acceptance_rate_7d", "avg_delivery_time_minutes_7d", "active_deliveries"]

    training_set = labels.sort_values("_row_id").reset_index(drop=True)
    training_set[restaurant_feature_cols] = restaurant_features[restaurant_feature_cols]
    training_set[courier_feature_cols] = courier_features[courier_feature_cols]
    training_set = training_set.drop(columns=["_row_id"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_set.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(training_set)} point-in-time-correct training rows -> {OUTPUT_PATH}")
    print(f"Columns: {list(training_set.columns)}")
    print(f"Null rate per column:\n{training_set.isnull().mean().round(3)}")


if __name__ == "__main__":
    main()
