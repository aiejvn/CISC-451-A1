"""Create empty (schema-only) SQLite databases for every db_id in tables.json.

evaluation.py needs spider/database/{db_id}/{db_id}.sqlite even for exact-match
scoring. Rows are not created, so execution accuracy is NOT meaningful.

    python build_spider_dbs.py
"""
import json
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DB_DIR = REPO_ROOT / "spider" / "database"
TABLES_JSON = REPO_ROOT / "spider" / "evaluation_examples" / "examples" / "tables.json"

TYPE_MAP = {"text": "TEXT", "number": "NUMERIC", "time": "TEXT", "boolean": "INTEGER", "others": "TEXT"}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_ddls(entry: dict) -> list[str]:
    tables = entry["table_names_original"]
    cols = entry["column_names_original"]
    types = entry["column_types"]

    pk = set()
    for p in entry["primary_keys"]:
        pk.update(p if isinstance(p, list) else [p])
    fk_by_col = {src: dst for src, dst in entry["foreign_keys"]}

    ddls = []
    for t_idx, t_name in enumerate(tables):
        if t_name.lower().startswith("sqlite_"):
            continue
        lines, table_pks = [], []
        for c_idx, (owner, c_name) in enumerate(cols):
            if owner != t_idx:
                continue
            lines.append(f"  {_q(c_name)} {TYPE_MAP.get(types[c_idx], 'TEXT')}")
            if c_idx in pk:
                table_pks.append(_q(c_name))
        if table_pks:
            lines.append(f"  PRIMARY KEY ({', '.join(table_pks)})")
        for c_idx, (owner, c_name) in enumerate(cols):
            if owner == t_idx and c_idx in fk_by_col:
                ref_owner, ref_col = cols[fk_by_col[c_idx]]
                lines.append(
                    f"  FOREIGN KEY ({_q(c_name)}) REFERENCES {_q(tables[ref_owner])}({_q(ref_col)})"
                )
        ddls.append(f"CREATE TABLE {_q(t_name)} (\n" + ",\n".join(lines) + "\n);")
    return ddls


def main() -> None:
    with open(TABLES_JSON) as f:
        entries = {e["db_id"]: e for e in json.load(f)}
    created = skipped = 0
    for db_id, entry in entries.items():
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
