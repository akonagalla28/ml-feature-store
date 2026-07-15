"""
Offline store: Parquet on disk, queried through DuckDB.

Responsibilities:
  1. Persist materialized feature snapshots (one parquet file per feature view,
     partitioned by materialization run) so training jobs have a durable,
     versioned history to read from.
  2. Serve point-in-time-correct joins: given a DataFrame of (entity_id, event_timestamp)
     rows -- e.g. "this order happened at this restaurant at this instant" -- return
     the feature values that were valid AT that instant, not the latest values.
     This is the core mechanism that prevents label leakage during training.
"""

from __future__ import annotations

import pathlib

import duckdb
import pandas as pd

STORE_ROOT = pathlib.Path(__file__).resolve().parent / "data"


class OfflineStore:
    def __init__(self, root: pathlib.Path = STORE_ROOT):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, feature_view_name: str) -> pathlib.Path:
        return self.root / f"{feature_view_name}.parquet"

    def write(self, feature_view_name: str, df: pd.DataFrame) -> None:
        """Append a materialization snapshot for a feature view."""
        path = self._path_for(feature_view_name)
        if path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)

    def read_latest(self, feature_view_name: str) -> pd.DataFrame:
        path = self._path_for(feature_view_name)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def point_in_time_join(
        self,
        feature_view_name: str,
        entity_df: pd.DataFrame,
        entity_key: str,
    ) -> pd.DataFrame:
        """
        For each row in entity_df (must have `entity_key` and `event_timestamp`
        columns), attach the most recent feature snapshot as of that timestamp
        -- never a snapshot computed *after* the event. This is the "as-of join"
        every serious feature store needs for correct offline training data.
        """
        path = self._path_for(feature_view_name)
        if not path.exists():
            raise FileNotFoundError(
                f"No materialized data for '{feature_view_name}'. Run pipelines/materialize.py first."
            )

        con = duckdb.connect()
        con.register("entity_df", entity_df)
        features_df = pd.read_parquet(path)
        con.register("features_df", features_df)

        query = f"""
            SELECT e.*, f.* EXCLUDE ({entity_key}, event_timestamp)
            FROM entity_df e
            ASOF LEFT JOIN features_df f
              ON e.{entity_key} = f.{entity_key}
              AND e.event_timestamp >= f.event_timestamp
        """
        result = con.execute(query).df()
        con.close()
        return result
