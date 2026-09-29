from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
SPIDER_DIR = REPO_ROOT / "spider"
SPIDER_EXAMPLES = SPIDER_DIR / "evaluation_examples" / "examples"
DEV_JSON = SPIDER_EXAMPLES / "dev.json"
TABLES_JSON = SPIDER_EXAMPLES / "tables.json"
DB_DIR = SPIDER_DIR / "database"
EVALUATION_PY = SPIDER_DIR / "evaluation.py"
RESULTS_DIR = EVAL_ROOT / "results"
MODELS_CONFIG = EVAL_ROOT / "configs" / "models.yaml"
