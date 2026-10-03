"""Fine-tune Qwen3-0.6B with LoRA on the Spider train split (validation = held-out training databases).

    warm    supervised warm start on gold SQL (optionally with rule-based plans: --fmt plan)
    stages  stages on top of a warm-start adapter, per round: gold (supervised control), vbi (verification-based
            iterative fine-tuning on execution-verified samples), sp (self-play fine-tuning with the SPIN logistic loss)

    SPIDER_DB_DIR=mock_spider/database python -m train.finetune warm --name e1_plan --fmt plan --n_train 1500 --bs 32
    SPIDER_DB_DIR=mock_spider/database python -m train.finetune stages --name c1_vbi --fmt plan \\
        --init train/runs/e1_plan/adapter --exclude_n 1500 --n_round 1500 --rounds 1 --stages vbi --hard_only
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import (MAX_NEW, RUNS_DIR, build_splits, chat_prompt, encode, eval_examples, free, generate, label,
                     load_base, load_lora, new_lora, seq_logps, sft_loss_fn, target_text, train)


# --- helpers shared by the stages ---------------------------------------------------------------------
def sample_labeled(model, tok, examples, fmt, k, temperature, top_p=0.95, cache=None):
    """k samples per question with execution labels; `cache` (a json path) reuses earlier samples from the same model."""
    if cache and Path(cache).exists():
        return [[tuple(x) for x in ss] for ss in json.loads(Path(cache).read_text())]
    prompts = [chat_prompt(tok, ex["question"], ex["db_id"], fmt) for ex in examples]
    outs = generate(model, tok, prompts, MAX_NEW[fmt], n=k, temperature=temperature, top_p=top_p)
    res = [[(t, *label(ex, t)[:2]) for t in ts] for ex, ts in zip(examples, outs)]
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(res))
    return res


def val_eval(model, tok, val_ex, fmt):
    r = eval_examples(model, tok, val_ex, fmt, MAX_NEW[fmt])
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if k not in ("texts", "labels")}


def gold_stage(init, out, examples, val_ex, fmt, lr, epochs, bs, seed, log, eval_steps=()):
    """Control: continue supervised training on gold answers for the same questions the other stages use."""
    tok, base = load_base()
    model = load_lora(base, init, trainable=True)
    items = [encode(tok, chat_prompt(tok, ex["question"], ex["db_id"], fmt), target_text(ex["query"], fmt)) for ex in examples]
    hook, select_best = step_hook(model, tok, val_ex, fmt, out, set(eval_steps), log)
    train(model, items, sft_loss_fn(tok.pad_token_id), lambda x: len(x[0]) + len(x[1]), epochs, lr, bs, seed=seed,
          on_step=hook, log=log)
    res, best = select_best(val_eval(model, tok, val_ex, fmt))
    del model, base
    free()
    return res, {"n_items": len(items), "selected": best}


def step_hook(model, tok, val_ex, fmt, out, steps, log):
    """Checkpoint and validate at the given optimizer steps; select_best() keeps the best one at `out`."""
    results = {}

    def hook(step):
        if step not in steps:
            return False
        results[step] = val_eval(model, tok, val_ex, fmt)
        model.save_pretrained(f"{out}_s{step}")
        log(f"  step {step} VAL {results[step]}")
        return True

    def select_best(final_res):
        results["final"] = final_res
        best = max(results, key=lambda k: results[k]["acc"])
        if best == "final":
            model.save_pretrained(out)
        else:
            shutil.copytree(f"{out}_s{best}", out, dirs_exist_ok=True)
        accs = {k: v["acc"] for k, v in results.items()}
        log(f"  selected {best} ({results[best]['acc']:.4f}); all: {accs}")
        return results[best], best

    return hook, select_best


def vbi_stage(init, out, examples, val_ex, fmt, k, temp, lr, epochs, max_pos, bs, seed, log, hard_only=False,
              eval_steps=(), cache=None):
    tok, base = load_base()
    model = load_lora(base, init, trainable=True)
    samples = sample_labeled(model, tok, examples, fmt, k, temp, cache=cache)
    items, n_pos_q = [], 0
    rng = random.Random(seed)
    for ex, ss in zip(examples, samples):
        seen, pos = set(), []
        if hard_only and all(ok for _, ok, _ in ss):
            continue  # every sample already correct: no new information
        for text, ok, _ in ss:
            sql = " ".join(text.split())
            if ok and sql not in seen:
                seen.add(sql)
                pos.append(text.rstrip())
        n_pos_q += bool(pos)
        for text in rng.sample(pos, min(max_pos, len(pos))):
            items.append(encode(tok, chat_prompt(tok, ex["question"], ex["db_id"], fmt), text))
    acc_samples = sum(ok for ss in samples for _, ok, _ in ss) / (len(examples) * k)
    log(f"  VBI sampled: sample acc {acc_samples:.3f}, {n_pos_q}/{len(examples)} questions with a positive, {len(items)} items")
    hook, select_best = step_hook(model, tok, val_ex, fmt, out, set(eval_steps), log)
    train(model, items, sft_loss_fn(tok.pad_token_id), lambda x: len(x[0]) + len(x[1]), epochs, lr, bs, seed=seed,
          on_step=hook, log=log)
    res, best = select_best(val_eval(model, tok, val_ex, fmt))
    del model, base
    free()
    return res, {"sample_acc": acc_samples, "n_items": len(items), "selected": best}


def selfplay_stage(main, opp, out, examples, val_ex, fmt, k, temp, lr, lam, epochs, bs, seed, log, eval_steps=(),
                   cache=None):
    # 1) opponent: sample, keep wrong outputs, and compute reference log-probs of both completions
    tok, base = load_base()
    opp_model = load_lora(base, opp, trainable=False)
    samples = sample_labeled(opp_model, tok, examples, fmt, k, temp, cache=cache)
    rng = random.Random(seed)
    pairs = []
    for ex, ss in zip(examples, samples):
        wrong = [t.rstrip() for t, ok, _ in ss if not ok]
        if not wrong:
            continue  # R(y', x) = 0: the opponent is right, no training signal
        p = encode(tok, chat_prompt(tok, ex["question"], ex["db_id"], fmt), target_text(ex["query"], fmt))
        _, neg = encode(tok, "", rng.choice(wrong))
        pairs.append({"p": p[0], "pos": p[1], "neg": neg})
    with torch.no_grad():
        for i in range(0, len(pairs), 8):
            chunk = pairs[i : i + 8]
            rp, _ = seq_logps(opp_model, tok.pad_token_id, [(x["p"], x["pos"]) for x in chunk])
            rn, _ = seq_logps(opp_model, tok.pad_token_id, [(x["p"], x["neg"]) for x in chunk])
            for x, a, b in zip(chunk, rp.tolist(), rn.tolist()):
                x["ref_pos"], x["ref_neg"] = a, b
    opp_acc = sum(ok for ss in samples for _, ok, _ in ss) / (len(examples) * k)
    log(f"  SP opponent sample acc {opp_acc:.3f}; {len(pairs)}/{len(examples)} questions give a pair")
    del opp_model, base
    free()

    # 2) main model against the frozen opponent
    tok, base = load_base()
    model = load_lora(base, main, trainable=True)

    def loss_fn(m, mb, batch):
        pos, _ = seq_logps(m, tok.pad_token_id, [(x["p"], x["pos"]) for x in mb])
        neg, _ = seq_logps(m, tok.pad_token_id, [(x["p"], x["neg"]) for x in mb])
        rp = torch.tensor([x["ref_pos"] for x in mb], device=pos.device)
        rn = torch.tensor([x["ref_neg"] for x in mb], device=pos.device)
        t = lam * ((pos - rp) - (neg - rn))
        return F.softplus(-t).sum() / len(batch)  # logistic loss l(t) = log(1 + exp(-t))

    hook, select_best = step_hook(model, tok, val_ex, fmt, out, set(eval_steps), log)
    train(model, pairs, loss_fn, lambda x: len(x["p"]) + max(len(x["pos"]), len(x["neg"])), epochs, lr, bs,
          token_budget=3000, seed=seed, on_step=hook, log=log)
    res, best = select_best(val_eval(model, tok, val_ex, fmt))
    del model, base
    free()
    return res, {"opp_sample_acc": opp_acc, "n_pairs": len(pairs), "selected": best}


# --- warm start ---------------------------------------------------------------------------------------
def make_items(tok, examples, fmt):
    return [encode(tok, chat_prompt(tok, ex["question"], ex["db_id"], fmt), target_text(ex["query"], fmt)) for ex in examples]


def warm(args):

    out = RUNS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    tok, base = load_base(checkpoint=not args.no_ckpt)
    train_ex, val_ex, _ = build_splits(tokenizer=tok)
    rng = random.Random(args.seed)
    if args.n_train:
        train_ex = rng.sample(train_ex, min(args.n_train, len(train_ex)))
    val_ex = random.Random(0).sample(val_ex, min(args.val_n, len(val_ex)))  # same val subset for every run
    items = make_items(tok, train_ex, args.fmt)
    print(f"{args.name}: {len(items)} train, {len(val_ex)} val, avg len {sum(len(a) + len(b) for a, b in items) / len(items):.0f}",
          flush=True)

    model = new_lora(base, args.r, args.alpha)
    history = []

    def on_epoch(ep):
        if args.eval_every and ep % args.eval_every == 0:
            r = eval_examples(model, tok, val_ex, args.fmt, MAX_NEW[args.fmt])
            row = {"epoch": ep, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if k not in ("texts", "labels")}}
            history.append(row)
            print("  VAL", row, flush=True)

    train(model, items, sft_loss_fn(tok.pad_token_id), lambda x: len(x[0]) + len(x[1]), args.epochs, args.lr, args.bs,
          token_budget=args.token_budget, seed=args.seed, on_epoch_end=on_epoch, log=lambda s: print(s, flush=True))
    if not history or history[-1]["epoch"] != int(args.epochs):
        on_epoch(int(args.epochs)) if args.eval_every == 0 else None
    model.save_pretrained(out / "adapter")
    (out / "result.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))


# --- stages -------------------------------------------------------------------------------------------
def run_stages(args):

    run = RUNS_DIR / args.name
    run.mkdir(parents=True, exist_ok=True)
    logf = open(run / "log.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    tok, base = load_base()
    train_ex, val_ex, _ = build_splits(tokenizer=tok)
    val_ex = random.Random(0).sample(val_ex, min(args.val_n, len(val_ex)))
    if args.exclude_n:
        used = {ex["id"] for ex in random.Random(0).sample(train_ex, args.exclude_n)}
        train_ex = [ex for ex in train_ex if ex["id"] not in used]
    log(f"{args.name}: {len(train_ex)} train / {len(val_ex)} val questions; args {vars(args)}")
    if args.init_acc is None:
        model = load_lora(base, args.init, trainable=False)
        init_res = val_eval(model, tok, val_ex, args.fmt)
        del model
    else:
        init_res = {"acc": args.init_acc, "skipped_eval": True}
    del base
    free()
    log(f"init VAL {init_res}")
    M = {str(args.init): init_res["acc"]}  # checkpoint -> validation accuracy
    history = [{"stage": "init", **init_res}]

    def cache_for(ckpt, k, r):
        if not args.sample_cache:
            return None
        key = f"{Path(ckpt).parent.name}_{Path(ckpt).name}_{args.fmt}_n{args.n_round}_ex{args.exclude_n}_k{k}_t{args.temp}_r{r}"
        return str(Path(args.sample_cache) / f"{key}.json")

    eval_steps = tuple(int(x) for x in args.eval_steps.split(",") if x)
    for r in range(1, args.rounds + 1):
        rng = random.Random(args.seed * 1000 + r)
        examples = rng.sample(train_ex, min(args.n_round, len(train_ex)))
        for stage in args.stages.split(","):
            best = max(M, key=M.get)
            out = str(run / f"r{r}_{stage}")
            if stage == "gold":
                log(f"round {r} gold-SFT control from {best} ({M[best]:.3f})")
                res, extra = gold_stage(best, out, examples, val_ex, args.fmt, args.lr_vbi, args.epochs, args.bs, args.seed + r, log,
                                       eval_steps)
            elif stage == "vbi":
                log(f"round {r} VBI-FT from {best} ({M[best]:.3f})")
                res, extra = vbi_stage(best, out, examples, val_ex, args.fmt, args.k, args.temp, args.lr_vbi,
                                       args.epochs, args.max_pos, args.bs, args.seed + r, log, args.hard_only, eval_steps,
                                       cache_for(best, args.k, r))
            else:
                opp = min(M, key=M.get)
                log(f"round {r} self-play main {best} ({M[best]:.3f}) vs opponent {opp} ({M[opp]:.3f})")
                res, extra = selfplay_stage(best, opp, out, examples, val_ex, args.fmt, args.k_sp, args.temp, args.lr_sp,
                                            args.lam, args.epochs, args.bs, args.seed + r, log, eval_steps,
                                            cache_for(opp, args.k_sp, r))
            M[out] = res["acc"]
            history.append({"stage": f"r{r}_{stage}", **res, **extra})
            log(f"  VAL r{r}_{stage}: {res} {extra}")
            (run / "history.json").write_text(json.dumps({"args": vars(args), "M": M, "history": history}, indent=1))
    log(f"best checkpoint: {max(M, key=M.get)} {max(M.values()):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("warm", help="supervised warm start")
    w.add_argument("--name", required=True)
    w.add_argument("--fmt", choices=["sql", "plan"], default="sql")
    w.add_argument("--epochs", type=float, default=3)
    w.add_argument("--lr", type=float, default=2e-4)
    w.add_argument("--bs", type=int, default=64)
    w.add_argument("--r", type=int, default=32)
    w.add_argument("--alpha", type=int, default=64)
    w.add_argument("--n_train", type=int, default=None)
    w.add_argument("--val_n", type=int, default=300)
    w.add_argument("--eval_every", type=int, default=1, help="evaluate on val every k epochs (0 = only at the end)")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--no_ckpt", action="store_true", help="disable gradient checkpointing (faster, more memory)")
    w.add_argument("--token_budget", type=int, default=6000, help="max padded tokens per micro-batch")
    s = sub.add_parser("stages", help="gold / vbi / sp stages on a warm-start adapter")
    s.add_argument("--name", required=True)
    s.add_argument("--fmt", choices=["sql", "plan"], default="sql")
    s.add_argument("--init", required=True, help="warm-start adapter dir")
    s.add_argument("--stages", default="vbi,sp", help="comma list per round, e.g. vbi,sp or vbi")
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--n_round", type=int, default=800, help="training questions sampled per round")
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--temp", type=float, default=0.7)
    s.add_argument("--max_pos", type=int, default=2)
    s.add_argument("--lr_vbi", type=float, default=5e-5)
    s.add_argument("--lr_sp", type=float, default=1e-5)
    s.add_argument("--lam", type=float, default=0.1)
    s.add_argument("--epochs", type=float, default=1)
    s.add_argument("--bs", type=int, default=32)
    s.add_argument("--val_n", type=int, default=707)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--exclude_n", type=int, default=0, help="drop the n warm-start questions (random.Random(0).sample, as in the warm subcommand)")
    s.add_argument("--init_acc", type=float, default=None, help="known val accuracy of --init (skips its evaluation)")
    s.add_argument("--hard_only", action="store_true", help="VBI: skip questions where every sample is correct")
    s.add_argument("--k_sp", type=int, default=2, help="opponent samples per question for self-play")
    s.add_argument("--eval_steps", default="", help="comma list of optimizer steps to checkpoint and validate, e.g. 10,20,30; the best is kept")
    s.add_argument("--sample_cache", default=None, help="dir for cached samples; stages sampling from the same checkpoint on the same questions share them")
    args = ap.parse_args()
    warm(args) if args.cmd == "warm" else run_stages(args)


if __name__ == "__main__":
    main()
