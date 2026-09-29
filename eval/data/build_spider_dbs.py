"""Create empty (schema-only) SQLite databases for every db_id in tables.json.

evaluation.py needs spider/database/{db_id}/{db_id}.sqlite even for exact-match
scoring. Rows are not created, so execution accuracy is NOT meaningful.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalpipeline.paths import DB_DIR  # noqa: E402
from evalpipeline.schema import load_tables, table_ddls  # noqa: E402


def main() -> None:
    created = skipped = 0
    for db_id, entry in load_tables().items():
        path = DB_DIR / db_id / f"{db_id}.sqlite"
        if path.exists():
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            for ddl in table_ddls(entry):
                conn.execute(ddl)
            conn.commit()
        finally:
            conn.close()
        created += 1
    print(f"created {created}, skipped (already exist) {skipped}, dir: {DB_DIR}")
    print(
        "WARNING: databases are schema-only (no rows). Exact-match scores are valid; "
        "execution-accuracy scores are not meaningful."
    )


if __name__ == "__main__":
    main()
