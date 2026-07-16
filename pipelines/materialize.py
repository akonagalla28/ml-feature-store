"""
Materialization pipeline.

For each feature view in the registry:
  1. Load its raw source data (orders.parquet / courier_events.parquet).
  2. Run it through the SAME transformation function that would be used anywhere
     else in the system (feature_store/transformations/eta_transforms.py).
  3. Write the result to the offline store (durable, versioned, queryable history).
  4. Write the *latest* snapshot to the online store (Redis) for low-latency serving.

Because steps 3 and 4 both consume the output of step 2, the online store and
offline store are always in sync by construction -- there is no separate
"online feature logic" that could drift from the "offline feature logic".

Run: python pipelines/materialize.py
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feature_store.registry.registry import FeatureRegistry
from feature_store.transformations.eta_transforms import TRANSFORMATIONS
from feature_store.offline_store.offline_store import OfflineStore
from feature_store.online_store.online_store import OnlineStore

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"

SOURCE_FILES = {
    "orders": RAW_DIR / "orders.parquet",
    "courier_events": RAW_DIR / "courier_events.parquet",
}


def materialize(as_of: pd.Timestamp | None = None, sync_online: bool = True) -> None:
    registry = FeatureRegistry()
    offline = OfflineStore()
    online = OnlineStore() if sync_online else None

    as_of = as_of or pd.Timestamp.now()

    for view_name in registry.list_feature_views():
        view = registry.get_feature_view(view_name)
        transform_fn = TRANSFORMATIONS[view.transformation]

        source_path = SOURCE_FILES[view.source]
        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing raw source '{source_path}'. Run pipelines/generate_synthetic_data.py first."
            )
        raw_df = pd.read_parquet(source_path)

        entity = registry.entities[view.entity]
        feature_df = transform_fn(raw_df, as_of)

        offline.write(view.name, feature_df)
        print(f"[offline] {view.name}: wrote {len(feature_df)} rows (version={view.version_hash})")

        if sync_online:
            n = online.write_batch(
                feature_view_name=view.name,
                df=feature_df,
                entity_key=entity.join_key,
                feature_names=view.feature_names,
                ttl_seconds=view.ttl_seconds,
            )
            print(f"[online]  {view.name}: synced {n} entities to Redis (ttl={view.ttl_seconds}s)")


def backfill(start: pd.Timestamp, end: pd.Timestamp, freq: str = "1D") -> None:
    """
    Materializes one snapshot per period into the OFFLINE store only, across a
    historical date range. This is what makes point-in-time-correct training
    joins meaningful: without a history of daily snapshots, there is nothing
    for an old label to join against except a single "latest" snapshot dated
    after the label itself, which point_in_time_join correctly refuses to use
    (see tests/test_consistency.py::test_point_in_time_join_does_not_leak_future_values).

    Run: python pipelines/materialize.py --backfill
    """
    for as_of in pd.date_range(start, end, freq=freq):
        materialize(as_of=as_of, sync_online=False)

    # Sync only the most recent snapshot to the online store, since online
    # serving only ever needs the latest value, not the full history.
    materialize(as_of=end, sync_online=True)


if __name__ == "__main__":
    import sys as _sys

    if "--backfill" in _sys.argv:
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=21)
        backfill(start, end, freq="1D")
    else:
        materialize()
