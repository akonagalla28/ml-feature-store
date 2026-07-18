"""
Feature serving API.

Exposes the online store over HTTP so a real-time prediction service (e.g. an
ETA-at-checkout call) can fetch the latest feature values for a restaurant or
courier with single-digit-millisecond Redis latency.

Run: uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feature_store.online_store.online_store import OnlineStore
from feature_store.registry.registry import FeatureRegistry

app = FastAPI(title="ETA Feature Store API", version="1.0.0")
online_store = OnlineStore()
registry = FeatureRegistry()


class FeatureResponse(BaseModel):
    entity_id: str
    feature_view: str
    features: dict
    version_hash: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/feature-views")
def list_feature_views() -> dict:
    views = {}
    for name in registry.list_feature_views():
        v = registry.get_feature_view(name)
        views[name] = {
            "entity": v.entity,
            "features": v.feature_names,
            "ttl_seconds": v.ttl_seconds,
            "version_hash": v.version_hash,
        }
    return views


@app.get("/features/{feature_view}/{entity_id}", response_model=FeatureResponse)
def get_features(feature_view: str, entity_id: str) -> FeatureResponse:
    try:
        view = registry.get_feature_view(feature_view)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown feature view '{feature_view}'")

    features = online_store.read_features(feature_view, entity_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached features for entity '{entity_id}' in '{feature_view}' "
            "(either not materialized yet, or TTL expired)",
        )

    return FeatureResponse(
        entity_id=entity_id,
        feature_view=feature_view,
        features=features,
        version_hash=view.version_hash,
    )


@app.post("/courier/{courier_id}/delivery-started")
def delivery_started(courier_id: str) -> dict:
    """Online-only event: increments a courier's live active-delivery counter."""
    new_count = online_store.bump_active_deliveries(courier_id, delta=1)
    return {"courier_id": courier_id, "active_deliveries": new_count}


@app.post("/courier/{courier_id}/delivery-completed")
def delivery_completed(courier_id: str) -> dict:
    new_count = online_store.bump_active_deliveries(courier_id, delta=-1)
    return {"courier_id": courier_id, "active_deliveries": max(0, new_count)}
