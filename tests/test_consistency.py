"""
The core guarantee this feature store makes: after materialization, the value
a training job reads from the offline store for the LATEST snapshot and the
value an online prediction request reads from Redis must be identical, since
both were written from the same transformation output in the same run.

These tests exercise that guarantee end-to-end using the synthetic dataset,
against a real (test) Redis instance -- see docker/docker-compose.yml.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from feature_store.registry.registry import FeatureRegistry
from feature_store.offline_store.offline_store import OfflineStore
from feature_store.online_store.online_store import OnlineStore
from pipelines.materialize import materialize


@pytest.fixture(scope="module")
def materialized():
    as_of = pd.Timestamp.now()
    materialize(as_of=as_of, sync_online=True)
    return as_of


def test_offline_online_consistency_restaurant(materialized):
    offline = OfflineStore()
    online = OnlineStore()

    offline_df = offline.read_latest("restaurant_prep_time_features")
    sample = offline_df.sort_values("event_timestamp").groupby("restaurant_id").tail(1)

    checked = 0
    for _, row in sample.head(5).iterrows():
        online_features = online.read_features("restaurant_prep_time_features", row["restaurant_id"])
        assert online_features is not None, f"No online features for {row['restaurant_id']}"

        assert online_features["avg_prep_time_minutes_7d"] == pytest.approx(
            row["avg_prep_time_minutes_7d"], rel=1e-6
        )
        assert online_features["order_volume_7d"] == row["order_volume_7d"]
        checked += 1

    assert checked == 5


def test_feature_view_registered_correctly():
    registry = FeatureRegistry()
    view = registry.get_feature_view("restaurant_prep_time_features")
    assert view.entity == "restaurant"
    assert "avg_prep_time_minutes_7d" in view.feature_names
    assert view.version_hash  # non-empty, deterministic hash


def test_point_in_time_join_does_not_leak_future_values():
    """
    A row timestamped BEFORE any materialized snapshot should get null features,
    not the nearest (future) snapshot -- proving the join is truly as-of, not
    a naive nearest-neighbor join.
    """
    offline = OfflineStore()
    offline_df = offline.read_latest("restaurant_prep_time_features")
    if offline_df.empty:
        pytest.skip("Run materialize() first")

    earliest_snapshot_ts = offline_df["event_timestamp"].min()
    too_early = pd.DataFrame(
        {
            "restaurant_id": [offline_df.iloc[0]["restaurant_id"]],
            "event_timestamp": [earliest_snapshot_ts - pd.Timedelta(days=30)],
        }
    )

    result = offline.point_in_time_join(
        "restaurant_prep_time_features", too_early, entity_key="restaurant_id"
    )
    assert pd.isna(result.iloc[0]["avg_prep_time_minutes_7d"])
