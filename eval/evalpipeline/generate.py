import argparse
import gc
import json
import os
import time
import zlib
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from .adapters import get_adapter
from .common import RESULTS_DIR, build_user_prompt, get_model_config, load_dev, resolve_local_path, write_gold


def make_run_dir(repo_id: str, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / repo_id.replace("/", "__") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(cfg: dict, models_path: Path | None = None, local_only: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    repo_id = cfg["repo_id"]
    local_path = resolve_local_path(repo_id, models_path, local_only)
    quant = cfg.get("quantization", "nf4")
    kwargs = {"device_map": cfg.get("device_map", "auto")}
    if quant == "nf4":
        compute = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute,
        )
    elif quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif quant == "none":
        kwargs["torch_dtype"] = "auto"
    else:
        raise ValueError(f"unknown quantization {quant!r}")

    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(local_path, local_files_only=True, **kwargs).eval()
    return model, tokenizer


class LoopStop:
    """Stops sequences whose tail is a repetition loop (greedy decoding makes loops absorbing, so more tokens do not help).

    From `min_new` generated tokens on, every `every` tokens the last `window` tokens are compressed; a sequence is
    stopped once the compression ratio stays below `ratio` for `patience` consecutive checks. The first `audit`
    detected loops are NOT stopped (they run to max_new_tokens), to test that loops really do not end.
    """

    def __init__(self, tokenizer, prompt_len: int, cfg: dict, state: dict):
        self.tok, self.prompt_len, self.cfg, self.state = tokenizer, prompt_len, cfg, state
        self.hits: dict[int, int] = {}

    def __call__(self, input_ids, scores=None, **kw):
        import torch
        c = self.cfg
        n = input_ids.shape[1] - self.prompt_len
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if n < c["min_new"] or n % c["every"]:
            return done
        for b in range(input_ids.shape[0]):
            text = self.tok.decode(input_ids[b, -c["window"]:], skip_special_tokens=True).encode()
            looping = len(zlib.compress(text)) / max(1, len(text)) < c["ratio"]
            self.hits[b] = self.hits.get(b, 0) + 1 if looping else 0
            if self.hits[b] >= c["patience"] and not self.state.get(("seen", b)):
                self.state[("seen", b)] = True
                self.state["detected"] = self.state.get("detected", 0) + 1
                if self.state["detected"] <= c.get("audit", 0):
                    self.state["audit_now"] = True  # let this one run on
                else:
                    done[b] = True
                    self.state["stopped_now"] = True
        return done


def load_done(path: Path) -> dict[int, dict]:
    """Rows already written by an interrupted run; rewrites the file without any torn last line."""
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["id"]] = row
        path.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
    return done


def generate(cfg: dict, examples: list[dict], run_dir: Path, seed: int = 0, models_path: Path | None = None,
             local_only: bool = False, resume: bool = False) -> Path:
    import torch
    from transformers import set_seed

    set_seed(seed)
    adapter = get_adapter(cfg)
    model, tokenizer = load_model(cfg, models_path, local_only)
    gen_kwargs = adapter.generation_kwargs()
    batch_size = cfg.get("batch_size", 1)

    write_gold(examples, run_dir / "gold.txt")
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump({"model": cfg, "generation": gen_kwargs, "n_examples": len(examples), "seed": seed,
         "ids": [ex["id"] for ex in examples]})
    )

    preds = {}
    loop_state: dict = {}
    done = load_done(run_dir / "generations.jsonl") if resume else {}
    preds.update({i: r["pred"] for i, r in done.items()})
    todo = [ex for ex in examples if ex["id"] not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)
    inputs = out = new_tokens = toks = None  # last-batch tensors; released before cache cleanup
    with open(run_dir / "generations.jsonl", "a" if resume else "w") as gens:
        bar = tqdm(range(0, len(todo), batch_size), desc=cfg["repo_id"])
        for start in bar:
            batch = todo[start : start + batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    adapter.build_messages(build_user_prompt(ex["question"], ex["db_id"])),
                    tokenize=False,
                    add_generation_prompt=True,
                    **adapter.chat_template_kwargs(),
                )
                for ex in batch
            ]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            t0 = time.time()
            extra = {}
            loop_cfg = cfg.get("loop_stop")
            if loop_cfg:
                from transformers import StoppingCriteriaList
                assert batch_size == 1, "loop_stop needs batch size 1 (per-example flags)"
                loop_state["stopped_now"] = loop_state["audit_now"] = False
                loop_state.pop(("seen", 0), None)
                extra["stopping_criteria"] = StoppingCriteriaList(
                    [LoopStop(tokenizer, inputs["input_ids"].shape[1], loop_cfg, loop_state)])
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                    **extra,
                    **gen_kwargs,
                )
            latency = time.time() - t0
            new_tokens = out[:, inputs["input_ids"].shape[1] :] # take only tokens autoregressed after prefill
            for ex, toks in zip(batch, new_tokens):
                raw = tokenizer.decode(toks, skip_special_tokens=adapter.skip_special_tokens)
                if not adapter.skip_special_tokens and tokenizer.pad_token:
                    raw = raw.replace(tokenizer.pad_token, "")
                thinking, sql = adapter.postprocess(raw)
                preds[ex["id"]] = sql
                gens.write(
                    json.dumps(
                        {
                            "id": ex["id"],
                            "db_id": ex["db_id"],
                            "question": ex["question"],
                            "gold": ex["query"],
                            "raw": raw,
                            "thinking": thinking,
                            "pred": sql,
                            "new_tokens": int((toks != tokenizer.pad_token_id).sum()),
                            "batch_latency_s": round(latency, 2),
                            **({"loop_stopped": bool(loop_state["stopped_now"]), "loop_audited": bool(loop_state["audit_now"])}
                               if cfg.get("loop_stop") else {}),
                        }
                    )
                    + "\n"
                )
                gens.flush()
            bar.set_postfix_str(datetime.now().strftime("%H:%M:%S"))

    (run_dir / "pred.txt").write_text("\n".join(preds[ex["id"]] for ex in examples) + "\n")

    del model, tokenizer, inputs, out, new_tokens, toks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo id, e.g. Qwen/Qwen3-8B (must be in models.yaml)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--models-path", type=Path, default=None, help="look for weights in <path>/<repo_id> first, e.g. ./models")
    ap.add_argument("--local-only", action="store_true", help="require weights under --models-path; never use the HF cache or network")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run with the same --run-id")
    ap.add_argument("--loop-stop", action="store_true", help="stop generations stuck in a repetition loop (needs batch size 1)")
    ap.add_argument("--loop-audit", type=int, default=0, help="with --loop-stop: let the first N detected loops run to max_new_tokens")
    args = ap.parse_args()

    if args.local_only and args.models_path is None:
        ap.error("--local-only requires --models-path")
    cfg = get_model_config(args.model)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.loop_stop:
        cfg["loop_stop"] = {"min_new": 4096, "window": 2048, "every": 512, "ratio": 0.15, "patience": 3, "audit": args.loop_audit}
    run_dir = make_run_dir(cfg["repo_id"], args.run_id)
    generate(cfg, load_dev(args.limit, args.seed), run_dir, args.seed, args.models_path, args.local_only, args.resume)
    print(run_dir)


if __name__ == "__main__":
    main()
