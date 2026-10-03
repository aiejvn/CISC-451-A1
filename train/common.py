"""Shared code for the fine-tuning pipeline (Qwen3-0.6B + LoRA on Spider): executing SQL and comparing results,
prompts and plan-then-SQL targets, the train/validation split, model loading, generation and scoring, log-probs,
and a small training loop."""
import gc
import math
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.evalpipeline.adapters import extract_sql
from eval.evalpipeline.common import DB_DIR, REPO_ROOT, build_user_prompt, load_train, resolve_local_path, schema_text

RUNS_DIR = REPO_ROOT / "train" / "runs"
MAX_NEW = {"sql": 256, "plan": 384}  # max new tokens per prompt format


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


# --- executing SQL ---------------------------------------------------------------------------------
MAX_ROWS = 5000
_conns: dict[str, sqlite3.Connection] = {}


def _conn(db_id: str) -> sqlite3.Connection:
    if db_id not in _conns:
        path = DB_DIR / db_id / f"{db_id}.sqlite"
        _conns[db_id] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        _conns[db_id].text_factory = lambda b: b.decode("utf-8", errors="replace")
    return _conns[db_id]


def run_sql(db_id: str, sql: str, max_steps: int = 2_000_000):
    """(rows, None) on success, (None, error message) on failure or when the query runs too long."""
    conn = _conn(db_id)
    steps = [0]

    def progress():
        steps[0] += 1
        return 1 if steps[0] > max_steps // 1000 else 0

    conn.set_progress_handler(progress, 1000)
    try:
        return conn.execute(sql).fetchmany(MAX_ROWS), None
    except Exception as e:  # sqlite3.Error, plus odd parser failures
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.set_progress_handler(None, 0)


def _norm(rows):
    return [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in rows]


def same_result(gold_rows, pred_rows, gold_sql: str) -> bool:
    """Multiset equality of rows; order only matters if the gold query has an ORDER BY."""
    g, p = _norm(gold_rows), _norm(pred_rows)
    if re.search(r"\border\s+by\b", gold_sql, re.I):
        return g == p
    return Counter(map(repr, g)) == Counter(map(repr, p))


# --- prompts, plan targets, splits -------------------------------------------------------------------
PLAN_TEMPLATE = (
    "Given the following SQLite database schema, write a SQL query that answers the question.\n\n{schema}\n\n"
    "Question: {question}\n\n"
    "First write a short plan using exactly these lines:\n"
    "Plan:\n- tables: ...\n- join: ...\n- select: ...\n- filter: ...\n- group/order: ...\n"
    "Then write the SQL query in a ```sql code block."
)
KEYWORDS = ["select", "from", "where", "group by", "having", "order by", "limit", "intersect", "union", "except"]
SET_OPS = ("intersect", "union", "except")


def user_prompt(question: str, db_id: str, fmt: str) -> str:
    if fmt == "sql":
        return build_user_prompt(question, db_id)
    return PLAN_TEMPLATE.format(schema=schema_text(db_id), question=question)


def chat_prompt(tokenizer, question: str, db_id: str, fmt: str) -> str:
    """Same chat template as Task 1 (thinking disabled)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt(question, db_id, fmt)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _top_level_clauses(sql: str) -> list[tuple[str, str]]:
    """Split a query into (keyword, text) at parenthesis depth 0; stops at the first set operator."""
    low, depth, i, marks, quote = sql.lower(), 0, 0, [], None
    while i < len(sql):
        ch = sql[i]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and (i == 0 or not low[i - 1].isalnum() and low[i - 1] != "_"):
            for kw in KEYWORDS:
                if low.startswith(kw, i) and not (low[i + len(kw) : i + len(kw) + 1].isalnum()):
                    marks.append((i, kw))
                    i += len(kw) - 1
                    break
        i += 1
    out = []
    for n, (pos, kw) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(sql)
        out.append((kw, sql[pos + len(kw) : end].strip()))
    return out


def plan_from_sql(sql: str) -> str:
    """Rule-based plan lines for a gold query (the base model is given the plan format only through the prompt)."""
    s = " ".join(sql.split()).rstrip(";").strip()
    alias, tables = {}, []
    for m in re.finditer(r"\b(?:from|join)\s+(\w+)(?:\s+as\s+(\w+))?", s, re.I):
        tables.append(m.group(1))
        if m.group(2):
            alias[m.group(2).lower()] = m.group(1)
    for a, t in alias.items():
        s = re.sub(rf"\b{re.escape(a)}\.", f"{t}.", s, flags=re.I)
    joins = list(dict.fromkeys(j.strip() for j in re.findall(
        r"\bon\s+(.+?)(?=\s+(?:join|where|group|order|limit|having|intersect|union|except)\b|\)|$)", s, re.I)))
    clauses = _top_level_clauses(s)
    first_set = next((i for i, (k, _) in enumerate(clauses) if k in SET_OPS), len(clauses))
    head = dict()
    for k, v in clauses[:first_set]:
        head.setdefault(k, v)
    select = head.get("select", "")
    tail = " ".join(f"{k} {head[k]}" for k in ("group by", "having", "order by", "limit") if k in head)
    if first_set < len(clauses):
        tail = (tail + " " if tail else "") + f"then {clauses[first_set][0]} with a second query"
    uniq = list(dict.fromkeys(tables))
    return "\n".join([
        "Plan:",
        f"- tables: {', '.join(uniq) if uniq else 'none'}",
        f"- join: {'; '.join(joins) if joins else 'none'}",
        f"- select: {select or 'none'}",
        f"- filter: {head.get('where', 'none')}",
        f"- group/order: {tail or 'none'}",
    ])


def target_text(query: str, fmt: str) -> str:
    sql = " ".join(query.split())
    if fmt == "sql":
        return f"```sql\n{sql}\n```"
    return f"{plan_from_sql(query)}\n```sql\n{sql}\n```"


def build_splits(n_val_dbs: int = 20, seed: int = 0, tokenizer=None, max_prompt_tokens: int = 1100):
    """Split Spider *train* by database. Questions whose gold query errors or returns nothing on the mock DB are dropped
    (an empty result would make any empty-returning prediction count as correct)."""
    examples = load_train()
    dbs = sorted({ex["db_id"] for ex in examples})
    val_dbs = set(random.Random(seed).sample(dbs, n_val_dbs))
    train, val = [], []
    for ex in examples:
        rows, err = run_sql(ex["db_id"], ex["query"])
        if err or not rows:
            continue
        if tokenizer is not None:
            n = len(tokenizer(chat_prompt(tokenizer, ex["question"], ex["db_id"], "plan"))["input_ids"])
            if n > max_prompt_tokens:
                continue
        (val if ex["db_id"] in val_dbs else train).append({**ex, "gold_rows": rows})
    return train, val, sorted(val_dbs)


# --- model, generation, scoring, training ----------------------------------------------------------
BASE_ID = "Qwen/Qwen3-0.6B"
END = "<|im_end|>"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base(checkpoint: bool = True):
    path = resolve_local_path(BASE_ID)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).cuda()
    if checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return tok, model


def new_lora(model, r: int = 32, alpha: int = 64, dropout: float = 0.05):
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def load_lora(model, path, trainable: bool = True):
    return PeftModel.from_pretrained(model, str(path), is_trainable=trainable)


# --- generation ---------------------------------------------------------------------------------
@torch.inference_mode()
def generate(model, tok, prompts: list[str], max_new_tokens: int, n: int = 1, temperature: float | None = None,
             top_p: float = 0.95, token_budget: int = 24000, max_bs: int = 64) -> list[list[str]]:
    """n completions per prompt (greedy if temperature is None). Batches are packed by a token budget."""
    model.eval()
    lens = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -lens[i])
    outs: list = [None] * len(prompts)
    i = 0
    while i < len(order):
        bs = max(1, min(max_bs, token_budget // ((lens[order[i]] + max_new_tokens) * n)))
        idx = order[i : i + bs]
        i += bs
        batch = tok([prompts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = {"do_sample": False} if temperature is None else {
            "do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": 0}
        out = model.generate(**batch, max_new_tokens=max_new_tokens, num_return_sequences=n,
                             pad_token_id=tok.pad_token_id, **kw)
        texts = tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        for k, j in enumerate(idx):
            outs[j] = texts[k * n : (k + 1) * n]
    return outs


def label(ex: dict, text: str) -> tuple[bool, str, str]:
    """(correct, reason, extracted sql); reason is ok | wrong | error | no_sql."""
    sql = extract_sql(text)
    if sql == "SELECT":
        return False, "no_sql", sql
    rows, err = run_sql(ex["db_id"], sql)
    if err:
        return False, "error", sql
    ok = same_result(ex["gold_rows"], rows, ex["query"])
    return ok, "ok" if ok else "wrong", sql


def eval_examples(model, tok, examples: list[dict], fmt: str, max_new_tokens: int) -> dict:
    prompts = [chat_prompt(tok, ex["question"], ex["db_id"], fmt) for ex in examples]
    t0 = time.time()
    texts = [o[0] for o in generate(model, tok, prompts, max_new_tokens)]
    labels = [label(ex, t) for ex, t in zip(examples, texts)]
    n = len(examples)
    reasons = {r: sum(l[1] == r for l in labels) / n for r in ("ok", "wrong", "error", "no_sql")}
    return {"acc": reasons["ok"], **{f"frac_{k}": v for k, v in reasons.items() if k != "ok"}, "n": n,
            "avg_tokens": sum(len(tok(t)["input_ids"]) for t in texts) / n, "secs": round(time.time() - t0),
            "texts": texts, "labels": [l[:2] for l in labels]}


# --- log-probs and training -----------------------------------------------------------------------
def encode(tok, prompt_text: str, completion_text: str) -> tuple[list[int], list[int]]:
    return (tok(prompt_text, add_special_tokens=False)["input_ids"],
            tok(completion_text + END, add_special_tokens=False)["input_ids"])


def seq_logps(model, pad_id: int, pairs: list[tuple[list[int], list[int]]]):
    """Sum of log p(completion | prompt) per pair, and completion lengths. Left padded, logits only for the tail."""
    B = len(pairs)
    L = max(len(p) + len(c) for p, c in pairs)
    C = max(len(c) for _, c in pairs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for b, (p, c) in enumerate(pairs):
        seq = p + c
        ids[b, L - len(seq):] = torch.tensor(seq)
        att[b, L - len(seq):] = 1
    ids, att = ids.cuda(), att.cuda()
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=C + 1).logits[:, :-1].float()
    tgt = ids[:, L - C:]
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)
    clen = torch.tensor([len(c) for _, c in pairs], device=ids.device)
    mask = torch.arange(C, device=ids.device)[None, :] >= (C - clen[:, None])
    return (lp * mask).sum(-1), clen


def micro_batches(items: list, length_of, token_budget: int, max_items: int = 8):
    items = sorted(items, key=length_of, reverse=True)
    batches, cur = [], []
    for it in items:
        if cur and (len(cur) + 1) * length_of(cur[0]) > token_budget or len(cur) >= max_items:
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    return batches


def train(model, items: list, loss_fn, length_of, epochs: float, lr: float, batch_size: int, token_budget: int = 6000,
          warmup: float = 0.05, seed: int = 0, on_epoch_end=None, on_step=None, log=print) -> None:
    """loss_fn(model, micro_batch, batch_items) -> scalar loss normalised over the whole effective batch."""
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "no trainable parameters"
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(items) / batch_size)
    total = max(1, round(epochs * steps_per_epoch))
    warm = max(1, int(warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    step, t0, run = 0, time.time(), []
    model.train()
    for ep in range(math.ceil(epochs)):
        order = list(range(len(items)))
        rng.shuffle(order)
        for s in range(0, len(order), batch_size):
            if step >= total:
                break
            batch = [items[i] for i in order[s : s + batch_size]]
            loss_sum = 0.0
            for mb in micro_batches(batch, length_of, token_budget):
                loss = loss_fn(model, mb, batch)
                loss.backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run.append(loss_sum)
            if on_step and on_step(step):
                model.train()
            if step % 10 == 0 or step == total:
                log(f"  step {step}/{total} loss {sum(run[-10:]) / len(run[-10:]):.4f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{time.time() - t0:.0f}s")
        if on_epoch_end:
            on_epoch_end(ep + 1)
            model.train()
    model.eval()


def sft_loss_fn(pad_id: int):
    def fn(model, mb, batch):
        lp, clen = seq_logps(model, pad_id, [(x[0], x[1]) for x in mb])
        denom = sum(len(x[1]) for x in batch)
        return -lp.sum() / denom
    return fn
