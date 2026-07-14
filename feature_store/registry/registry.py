"""
Lightweight feature registry.

This intentionally avoids pulling in a heavyweight framework (Feast, Tecton, etc.)
so the whole store is readable in an afternoon. It provides the three things a
registry needs to provide:

  1. A single source of truth for what features exist, their dtypes, owners, and TTLs.
  2. A stable content hash per feature view, so consumers can detect drift between
     the definition used to materialize a feature and the one used to read it.
  3. A lookup from feature view name -> transformation function name, so the same
     transformation code path is used for both batch (offline) and streaming/online
     materialization. This is what keeps training/serving skew from creeping in.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field

import yaml

DEFINITIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "definitions"


@dataclass
class FeatureDef:
    name: str
    dtype: str
    description: str = ""


@dataclass
class FeatureView:
    name: str
    entity: str
    owner: str
    ttl_seconds: int
    transformation: str
    source: str
    features: list[FeatureDef] = field(default_factory=list)
    version_hash: str = ""

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


@dataclass
class Entity:
    name: str
    join_key: str
    description: str = ""


class FeatureRegistry:
    """Loads all YAML files in feature_store/definitions/ into memory."""

    def __init__(self, definitions_dir: pathlib.Path = DEFINITIONS_DIR):
        self.entities: dict[str, Entity] = {}
        self.feature_views: dict[str, FeatureView] = {}
        self._load(definitions_dir)

    def _load(self, definitions_dir: pathlib.Path) -> None:
        for path in sorted(definitions_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text())
            for e in raw.get("entities", []):
                entity = Entity(**e)
                self.entities[entity.name] = entity

            for fv in raw.get("feature_views", []):
                features = [FeatureDef(**f) for f in fv.pop("features", [])]
                view = FeatureView(features=features, **fv)
                view.version_hash = self._hash_view(view)
                self.feature_views[view.name] = view

    @staticmethod
    def _hash_view(view: FeatureView) -> str:
        """Stable hash of a feature view's schema, used to detect definition drift."""
        payload = f"{view.name}|{view.entity}|{view.transformation}|" + "|".join(
            f"{f.name}:{f.dtype}" for f in view.features
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def get_feature_view(self, name: str) -> FeatureView:
        if name not in self.feature_views:
            raise KeyError(f"No feature view registered under '{name}'")
        return self.feature_views[name]

    def list_feature_views(self) -> list[str]:
        return list(self.feature_views.keys())

    def feature_view_for_entity(self, entity_name: str) -> list[FeatureView]:
        return [v for v in self.feature_views.values() if v.entity == entity_name]


if __name__ == "__main__":
    registry = FeatureRegistry()
    for name in registry.list_feature_views():
        v = registry.get_feature_view(name)
        print(f"{v.name} (entity={v.entity}, version={v.version_hash}, ttl={v.ttl_seconds}s)")
        for f in v.features:
            print(f"    - {f.name}: {f.dtype}  # {f.description}")
