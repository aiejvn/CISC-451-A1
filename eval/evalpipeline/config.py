from pathlib import Path

import yaml

from .paths import MODELS_CONFIG


def load_model_configs(path: Path = MODELS_CONFIG) -> dict[str, dict]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    out = {}
    for m in raw["models"]:
        cfg = {**defaults, **m}
        cfg["repo_id"] = f"{m['author']}/{m['name']}"
        out[cfg["repo_id"]] = cfg
    return out


def get_model_config(repo_id: str, path: Path = MODELS_CONFIG) -> dict:
    cfgs = load_model_configs(path)
    if repo_id not in cfgs:
        raise KeyError(f"{repo_id!r} not in {path}; known: {sorted(cfgs)}")
    return cfgs[repo_id]
