"""Build mock Spider databases with rows (the official database files are not available locally).

Row i of every table belongs to "entity" i. Question i of a database gets entity i: the literals that its gold SQL
filters on (e.g. country = 'France', year > 2013) are written into row i, and foreign keys of row i point at row i of
the parent table, so that joins through the entity succeed. Remaining rows are random filler.

Mock-database execution accuracy is only an approximation of the official Spider execution accuracy.

    python -m train.mockdb            # writes mock_spider/database/<db_id>/<db_id>.sqlite
"""
import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from build_spider_dbs import TABLES_JSON, table_ddls
from eval.evalpipeline.common import REPO_ROOT

MOCK_DB_DIR = REPO_ROOT / "mock_spider" / "database"

WORDS = ("Alice Bob Carol David Eve Frank Grace Henry Irene Jack Karen Leo Maria Nick Olga Paul Quinn Rita Sam Tina "
         "Red Blue Green Black White Paris London Tokyo Berlin Madrid Rome Cairo Lima Oslo Delhi "
         "France Japan Brazil Canada Egypt India Spain Italy Kenya Chile alpha beta gamma delta sigma omega").split()
VOCAB = {
    "country": "France Japan Brazil Canada Egypt India Spain Italy Kenya Chile".split(),
    "city": "Paris London Tokyo Berlin Madrid Rome Cairo Lima Oslo Delhi".split(),
    "gender": ["M", "F"], "sex": ["M", "F"],
    "color": "Red Blue Green Black White".split(),
}
LITERAL_RE = re.compile(
    r"""(?:\b\w+\.)?(\w+)\s*(=|!=|<>|>=|<=|>|<|\bnot\s+like\b|\blike\b)\s*('[^']*'|"[^"]*"|-?\d+(?:\.\d+)?)""", re.I)
BETWEEN_RE = re.compile(r"(?:\b\w+\.)?(\w+)\s+between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", re.I)


def _num(s: str):
    return float(s) if "." in s else int(s)


def harvest(query: str) -> list[tuple[str, object]]:
    """(column name, value that makes the predicate true) for the simple predicates of a gold query."""
    out = []
    for name, op, raw in LITERAL_RE.findall(query):
        op = " ".join(op.lower().split())
        if raw[0] in "'\"":
            if op in ("=", "like"):
                v = raw[1:-1].strip("%")
                if v:
                    out.append((name.lower(), v))
        else:
            v = _num(raw)
            shift = {">": 1, "<": -1}.get(op, 0)
            if op not in ("!=", "<>", "not like"):
                out.append((name.lower(), v + shift))
    for name, lo, hi in BETWEEN_RE.findall(query):
        out.append((name.lower(), (_num(lo) + _num(hi)) // 2))
    return out


def _generic(rng: random.Random, name: str, ctype: str, pool: list, i: int):
    n = name.lower()
    nums = [v for v in pool if isinstance(v, (int, float))]
    if ctype in ("number", "boolean"):
        if ctype == "boolean":
            return rng.randint(0, 1)
        if nums:
            lo, hi = min(nums), max(nums)
            span = max(hi - lo, 10)
            v = rng.uniform(lo - span / 2, hi + span / 2)
            return round(v, 1) if any(isinstance(x, float) for x in nums) else int(v)
        if "year" in n:
            return rng.randint(1950, 2020)
        if "age" in n:
            return rng.randint(15, 80)
        return rng.randint(0, 100)
    if ctype == "time" or "date" in n:
        return f"{rng.randint(1990, 2020)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if pool and all(str(v).isdigit() for v in pool):
        ints = [int(v) for v in pool]
        return str(rng.randint(min(ints) - 5, max(ints) + 5))
    for key, vocab in VOCAB.items():
        if key in n:
            return rng.choice(vocab)
    return rng.choice(WORDS)


def build_db(entry: dict, queries: list[str], path: Path, seed: int = 0) -> None:
    rng = random.Random(f"{seed}-{entry['db_id']}")
    tables, cols, types = entry["table_names_original"], entry["column_names_original"], entry["column_types"]
    pks = set()
    for p in entry["primary_keys"]:
        pks.update(p if isinstance(p, list) else [p])
    fk = {src: dst for src, dst in entry["foreign_keys"]}
    n_rows = max(30, len(queries) + 20)

    pools = defaultdict(list)  # column name -> literal values
    per_query = []
    for q in queries:
        lits = harvest(q)
        per_query.append(lits)
        for name, v in lits:
            pools[name].append(v)

    vals: dict[int, list] = {}  # global column index -> n_rows values
    for c, (t, name) in enumerate(cols):
        if t < 0 or c in fk:
            continue
        if c in pks:
            ids = list(range(1, n_rows + 1))
            vals[c] = ids if types[c] == "number" else [f"{name[:3].lower()}{i}" for i in ids]
            continue
        pool = pools.get(name.lower(), [])
        col = [_generic(rng, name, types[c], pool, i) for i in range(n_rows)]
        for i, lits in enumerate(per_query):  # entity i gets the literals of question i
            for lname, v in lits:
                if lname == name.lower():
                    col[i] = v if types[c] in ("number", "boolean") or not isinstance(v, (int, float)) else str(v)
        vals[c] = col
    for _ in range(5):  # foreign keys, parents first
        for c, parent in fk.items():
            if c in vals or parent not in vals:
                continue
            pv = vals[parent]
            vals[c] = [pv[i] if (i < len(queries) or c in pks) else rng.choice(pv) for i in range(n_rows)]
    for c in fk:
        vals.setdefault(c, [rng.randint(1, n_rows) for _ in range(n_rows)])

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        for ddl in table_ddls(entry):
            conn.execute(ddl)
        for t, tname in enumerate(tables):
            if tname.lower().startswith("sqlite_"):
                continue
            idx = [c for c, (owner, _) in enumerate(cols) if owner == t]
            rows = [tuple(vals[c][i] for c in idx) for i in range(n_rows)]
            q = ",".join("?" * len(idx))
            conn.executemany(f'INSERT OR IGNORE INTO "{tname}" VALUES ({q})', rows)
        conn.commit()
    finally:
        conn.close()


def build_all(seed: int = 0, out_dir: Path = MOCK_DB_DIR) -> None:
    entries = {e["db_id"]: e for e in json.load(open(TABLES_JSON))}
    queries = defaultdict(list)
    for split in ("train.json", "dev.json"):
        for ex in json.load(open(REPO_ROOT / split)):
            queries[ex["db_id"]].append(ex["query"])
    for db_id, qs in queries.items():
        build_db(entries[db_id], qs, out_dir / db_id / f"{db_id}.sqlite", seed)
    print(f"built {len(queries)} mock databases in {out_dir}")


if __name__ == "__main__":
    build_all()
