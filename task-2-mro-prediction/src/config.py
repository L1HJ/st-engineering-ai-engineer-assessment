"""Configuration loading: read schema.yaml and generation.yaml."""
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")


def _load(name):
    with open(os.path.join(CONFIGS, name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SCHEMA = _load("schema.yaml")
GEN = _load("generation.yaml")

FIELDS = SCHEMA["fields"]
FIELD_NAMES = [f["name"] for f in FIELDS]
MODEL_FEATURES = [f["name"] for f in FIELDS if f.get("model_feature")]
CATEGORICAL = [f["name"] for f in FIELDS if f["type"] == "category"]
SENSOR_FIELDS = [f["name"] for f in FIELDS if f.get("role") == "sensor"]


def gen(*keys):
    """Read a generation setting by path, e.g. gen('snapshot','interval_cycles_min')."""
    node = GEN
    for k in keys:
        node = node[k]
    return node


def weighted_choices(items, name_key="name", weight_key="weight"):
    """Return (names, weights)."""
    names = [it[name_key] for it in items]
    weights = [it[weight_key] for it in items]
    return names, weights
