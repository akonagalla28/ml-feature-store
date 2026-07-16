"""
Generates synthetic raw event data so the whole pipeline can be run with
no external dependencies: `orders.parquet` and `courier_events.parquet`
under data/raw/.

Run: python pipelines/generate_synthetic_data.py
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
RNG = np.random.default_rng(42)

N_RESTAURANTS = 25
N_COURIERS = 40
N_DAYS = 21
EVENTS_PER_DAY = 400


def generate_orders() -> pd.DataFrame:
    start = datetime.now() - timedelta(days=N_DAYS)
    rows = []
    for day in range(N_DAYS):
        for _ in range(EVENTS_PER_DAY):
            restaurant_id = f"r_{RNG.integers(0, N_RESTAURANTS):03d}"
            ts = start + timedelta(days=day, minutes=int(RNG.integers(0, 1440)))
            # give each restaurant a stable-ish "true" prep time with noise, so
            # a downstream model has real signal to learn from.
            base_prep = 15 + (hash(restaurant_id) % 20)
            prep_time = max(3, RNG.normal(base_prep, 4))
            rows.append(
                {
                    "restaurant_id": restaurant_id,
                    "event_timestamp": ts,
                    "prep_time_minutes": round(prep_time, 1),
                }
            )
    return pd.DataFrame(rows).sort_values("event_timestamp").reset_index(drop=True)


def generate_courier_events() -> pd.DataFrame:
    start = datetime.now() - timedelta(days=N_DAYS)
    rows = []
    for day in range(N_DAYS):
        for _ in range(EVENTS_PER_DAY):
            courier_id = f"c_{RNG.integers(0, N_COURIERS):03d}"
            ts = start + timedelta(days=day, minutes=int(RNG.integers(0, 1440)))
            base_accept = 0.6 + (hash(courier_id) % 30) / 100
            accepted = RNG.random() < min(base_accept, 0.97)
            delivery_time = max(8, RNG.normal(28, 6))
            rows.append(
                {
                    "courier_id": courier_id,
                    "event_timestamp": ts,
                    "accepted": bool(accepted),
                    "delivery_time_minutes": round(delivery_time, 1),
                }
            )
    return pd.DataFrame(rows).sort_values("event_timestamp").reset_index(drop=True)


def generate_training_labels(orders: pd.DataFrame, courier_events: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates completed deliveries: pairs a restaurant + courier + timestamp with
    an observed actual ETA (in minutes). This is the label a model would predict.
    """
    n = 1500
    idx_o = RNG.integers(0, len(orders), n)
    idx_c = RNG.integers(0, len(courier_events), n)
    rows = []
    for i in range(n):
        o = orders.iloc[idx_o[i]]
        c = courier_events.iloc[idx_c[i]]
        ts = max(o["event_timestamp"], c["event_timestamp"]) + timedelta(minutes=int(RNG.integers(1, 15)))
        actual_eta = o["prep_time_minutes"] * 0.6 + c["delivery_time_minutes"] * 0.8 + RNG.normal(0, 3)
        rows.append(
            {
                "restaurant_id": o["restaurant_id"],
                "courier_id": c["courier_id"],
                "event_timestamp": ts,
                "actual_eta_minutes": round(max(5, actual_eta), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("event_timestamp").reset_index(drop=True)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    orders = generate_orders()
    courier_events = generate_courier_events()
    labels = generate_training_labels(orders, courier_events)

    orders.to_parquet(RAW_DIR / "orders.parquet", index=False)
    courier_events.to_parquet(RAW_DIR / "courier_events.parquet", index=False)
    labels.to_parquet(RAW_DIR / "training_labels.parquet", index=False)

    print(f"Wrote {len(orders)} order events -> {RAW_DIR / 'orders.parquet'}")
    print(f"Wrote {len(courier_events)} courier events -> {RAW_DIR / 'courier_events.parquet'}")
    print(f"Wrote {len(labels)} training labels -> {RAW_DIR / 'training_labels.parquet'}")


if __name__ == "__main__":
    main()
