import json
from functools import lru_cache
from pathlib import Path

from .paths import TABLES_JSON

TYPE_MAP = {
    "text": "TEXT",
    "number": "NUMERIC",
    "time": "TEXT",
    "boolean": "INTEGER",
    "others": "TEXT",
}


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
        lines = []
        table_pks = []
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
                    f"  FOREIGN KEY ({_q(c_name)}) REFERENCES "
                    f"{_q(tables[ref_owner])}({_q(ref_col)})"
                )
        ddls.append(f"CREATE TABLE {_q(t_name)} (\n" + ",\n".join(lines) + "\n);")
    return ddls


@lru_cache(maxsize=None)
def load_tables(path: str = str(TABLES_JSON)) -> dict[str, dict]:
    with open(Path(path)) as f:
        return {e["db_id"]: e for e in json.load(f)}


def schema_text(db_id: str) -> str:
    return "\n\n".join(table_ddls(load_tables()[db_id]))
