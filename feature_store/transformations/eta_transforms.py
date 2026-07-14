"""
Shared feature transformations.

The single most common way a feature store breaks in production is
"training/serving skew": the batch job that computes a feature for training
uses slightly different logic than the code path that computes it for live
serving. Here, both `pipelines/materialize.py` (batch/offline + online sync)
and any future streaming job import these exact functions, so there is only
ever one implementation of "how do we compute avg_prep_time_minutes_7d".

Each function takes a raw events DataFrame and an `as_of` timestamp, and
returns a DataFrame of entity_id -> computed features, using only rows with
event_timestamp <= as_of. That constraint is what makes point-in-time-correct
training data possible later (see offline_store.py).
"""

from __future__ import annotations

import pandas as pd


def restaurant_prep_time_features(orders: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rolling 7-day prep-time stats per restaurant, computed as of `as_of`."""
    window_start = as_of - pd.Timedelta(days=7)
    window = orders[(orders["event_timestamp"] > window_start) & (orders["event_timestamp"] <= as_of)]

    grouped = window.groupby("restaurant_id")["prep_time_minutes"]
    out = grouped.agg(
        avg_prep_time_minutes_7d="mean",
        order_volume_7d="count",
        prep_time_stddev_7d="std",
    ).reset_index()

    out["prep_time_stddev_7d"] = out["prep_time_stddev_7d"].fillna(0.0)
    out["event_timestamp"] = as_of
    return out


def courier_performance_features(courier_events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Rolling 7-day acceptance/delivery-time stats per courier, computed as of `as_of`."""
    window_start = as_of - pd.Timedelta(days=7)
    window = courier_events[
        (courier_events["event_timestamp"] > window_start) & (courier_events["event_timestamp"] <= as_of)
    ]

    grouped = window.groupby("courier_id")
    out = grouped.agg(
        acceptance_rate_7d=("accepted", "mean"),
        avg_delivery_time_minutes_7d=("delivery_time_minutes", "mean"),
    ).reset_index()

    # active_deliveries is a near-real-time counter that only makes sense online;
    # for offline/training purposes we default it to 0 (it is never used as a
    # training label input for this reason -- see docs/README for the caveat).
    out["active_deliveries"] = 0
    out["event_timestamp"] = as_of
    return out


TRANSFORMATIONS = {
    "restaurant_prep_time_features": restaurant_prep_time_features,
    "courier_performance_features": courier_performance_features,
}
