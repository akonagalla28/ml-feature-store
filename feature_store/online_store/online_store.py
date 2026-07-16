"""
Online store: Redis, for low-latency feature lookups at inference time.

Key layout: "{feature_view_name}:{entity_id}" -> hash of {feature_name: value, "_ts": iso_timestamp}
A per-feature-view TTL (from the registry) is applied on write so stale features
expire automatically instead of silently going stale forever.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import redis


class OnlineStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def write_features(
        self,
        feature_view_name: str,
        entity_id: str,
        features: dict,
        ttl_seconds: int,
    ) -> None:
        key = f"{feature_view_name}:{entity_id}"
        payload = dict(features)
        payload["_ts"] = datetime.now(timezone.utc).isoformat()
        self.client.set(key, json.dumps(payload), ex=ttl_seconds)

    def read_features(self, feature_view_name: str, entity_id: str) -> dict | None:
        key = f"{feature_view_name}:{entity_id}"
        raw = self.client.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def write_batch(
        self,
        feature_view_name: str,
        df,  # pd.DataFrame with entity_id as index or a designated id column
        entity_key: str,
        feature_names: list[str],
        ttl_seconds: int,
    ) -> int:
        """Bulk-write a materialized DataFrame into the online store via a pipeline."""
        pipe = self.client.pipeline()
        count = 0
        for _, row in df.iterrows():
            key = f"{feature_view_name}:{row[entity_key]}"
            payload = {name: row[name] for name in feature_names if name in row}
            payload["_ts"] = (
                row["event_timestamp"].isoformat()
                if hasattr(row["event_timestamp"], "isoformat")
                else str(row["event_timestamp"])
            )
            pipe.set(key, json.dumps(payload, default=str), ex=ttl_seconds)
            count += 1
        pipe.execute()
        return count

    def bump_active_deliveries(self, courier_id: str, delta: int, ttl_seconds: int = 1800) -> int:
        """
        Example of an online-only feature: a live counter with no offline equivalent.
        Demonstrates that not every feature needs a batch materialization path --
        some are purely stream/event driven.
        """
        key = f"courier_performance_features:{courier_id}"
        existing = self.read_features("courier_performance_features", courier_id) or {}
        existing["active_deliveries"] = existing.get("active_deliveries", 0) + delta
        self.client.set(key, json.dumps(existing, default=str), ex=ttl_seconds)
        return existing["active_deliveries"]
