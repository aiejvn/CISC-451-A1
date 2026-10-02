"""Shared paths, config loading, data loading, prompts and schema lookup."""
import json
import random
import sqlite3
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# --- paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
TRAIN_JSON = REPO_ROOT / "train.json"
DEV_JSON = REPO_ROOT / "dev.json"
RESULTS_DIR = EVAL_ROOT / "results"
MODELS_CONFIG = EVAL_ROOT / "configs" / "models.yaml"
# Spider checkout: only used for the schema-only sqlite DBs and evaluation.py.
SPIDER_DIR = REPO_ROOT / "spider"
DB_DIR = SPIDER_DIR / "database"
EVALUATION_PY = SPIDER_DIR / "evaluation.py"
TABLES_JSON = SPIDER_DIR / "evaluation_examples" / "examples" / "tables.json"  # evaluation.py FK map


# --- model config ------------------------------------------------------------
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


# --- data (repo-root train.json / dev.json) ----------------------------------
def load_split(path: Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    """Load a split; each example gets "id" = its index in the source file."""
    with open(path) as f:
        examples = [{**ex, "id": i} for i, ex in enumerate(json.load(f))]
    if limit is not None and limit < len(examples):
        idx = sorted(random.Random(seed).sample(range(len(examples)), limit))
        examples = [examples[i] for i in idx]
    return examples


def load_dev(limit: int | None = None, seed: int = 0) -> list[dict]:
    return load_split(DEV_JSON, limit, seed)


def load_train(limit: int | None = None, seed: int = 0) -> list[dict]:
    return load_split(TRAIN_JSON, limit, seed)


def write_gold(examples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for ex in examples:
            query = " ".join(ex["query"].split())
            f.write(f"{query}\t{ex['db_id']}\n")


# --- schema (read from the sqlite file for each db_id) -----------------------
_schema_cache: dict[str, str] = {}


def schema_text(db_id: str) -> str:
    if db_id not in _schema_cache:
        path = DB_DIR / db_id / f"{db_id}.sqlite"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; cannot build schema for {db_id!r}")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()
        _schema_cache[db_id] = "\n\n".join(r[0].rstrip(";") + ";" for r in rows)
    return _schema_cache[db_id]


# --- prompts -----------------------------------------------------------------
TEMPLATE = (
    "Given the following SQLite database schema, write a SQL query that answers "
    "the question.\n\n{schema}\n\nQuestion: {question}\n\n"
    "Respond with only the SQL query in a ```sql code block."
)


def build_user_prompt(question: str, db_id: str) -> str:
    return TEMPLATE.format(schema=schema_text(db_id), question=question)


# --- local-first HF weight resolution ----------------------------------------
def resolve_local_path(repo_id: str) -> str:
    """Return the cached snapshot path; download once if not cached yet."""
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"[hf_cache] {repo_id} not cached yet; downloading...")
        return snapshot_download(repo_id, local_files_only=False)
