"""Re-parse Spider's raw question/SQL-string json into json with a structured "sql" AST field.

Moved out of the `spider/` submodule (upstream `spider/preprocess/parse_raw_json.py`, which has
hardcoded paths and no CLI) so this repo's own fix survives a `git submodule update --init
--recursive`, which resets `spider/` to its pinned upstream commit and would otherwise silently
discard any local edit made inside it. This is what produced this repo's root `train.json` /
`dev.json` from Spider's own `evaluation_examples/examples/{train_spider,dev}.json`; both are
already committed, so re-running this is only needed to reproduce or regenerate them.

    python build_spider_splits.py spider/evaluation_examples/examples/train_spider.json train.json
    python build_spider_splits.py spider/evaluation_examples/examples/dev.json dev.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "spider"))
from process_sql import get_sql  # noqa: E402


class Schema:
    """Simple schema which maps table&column to a unique identifier."""

    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table["column_names_original"]
        table_names_original = table["table_names_original"]
        idMap = {}
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {"*": i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap


def get_schemas_from_json(fpath: Path) -> tuple[dict, list, dict]:
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db["db_id"] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db["db_id"]
        schema = {}
        column_names_original = db["column_names_original"]
        table_names_original = db["table_names_original"]
        tables[db_id] = {
            "column_names_original": column_names_original,
            "table_names_original": table_names_original,
        }
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sql_path", help='input json with "db_id" and "query" fields, e.g. train_spider.json')
    parser.add_argument("output_file", help='output json with re-parsed "sql" fields')
    parser.add_argument(
        "--table_file",
        default=str(REPO_ROOT / "spider" / "evaluation_examples" / "examples" / "tables.json"),
    )
    args = parser.parse_args()

    schemas, db_names, tables = get_schemas_from_json(args.table_file)

    with open(args.sql_path) as inf:
        sql_data = json.load(inf)

    sql_data_new = []
    n_failed = 0
    for data in sql_data:
        db_id = data["db_id"]
        sql = data["query"]
        try:
            schema = Schema(schemas[db_id], tables[db_id])
            data["sql"] = get_sql(schema, sql)
            sql_data_new.append(data)
        except Exception:
            n_failed += 1
            print(f"failed to parse db_id={db_id!r} sql={sql!r}")

    with open(args.output_file, "wt") as out:
        json.dump(sql_data_new, out, sort_keys=True, indent=4, separators=(",", ": "))
    print(f"wrote {len(sql_data_new)} examples to {args.output_file} ({n_failed} failed to parse)")


if __name__ == "__main__":
    main()
