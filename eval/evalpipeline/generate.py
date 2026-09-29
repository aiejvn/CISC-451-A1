import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from .adapters.registry import get_adapter
from .config import get_model_config
from .dataset import load_dev, write_gold
from .hf_cache import resolve_local_path
from .paths import RESULTS_DIR
from .prompts import build_user_prompt


def make_run_dir(repo_id: str, run_id: str | None = None) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / repo_id.replace("/", "__") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    repo_id = cfg["repo_id"]
    local_path = resolve_local_path(repo_id)
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


def generate(cfg: dict, examples: list[dict], run_dir: Path, seed: int = 0) -> Path:
    import torch
    from transformers import set_seed

    set_seed(seed)
    adapter = get_adapter(cfg)
    model, tokenizer = load_model(cfg)
    gen_kwargs = adapter.generation_kwargs()
    batch_size = cfg.get("batch_size", 1)

    write_gold(examples, run_dir / "gold.txt")
    (run_dir / "config_used.yaml").write_text(
        yaml.safe_dump({"model": cfg, "generation": gen_kwargs, "n_examples": len(examples), "seed": seed})
    )

    pred_lines = []
    with open(run_dir / "generations.jsonl", "w") as gens:
        for start in tqdm(range(0, len(examples), batch_size), desc=cfg["repo_id"]):
            batch = examples[start : start + batch_size]
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
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=cfg["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs,
                )
            latency = time.time() - t0
            new_tokens = out[:, inputs["input_ids"].shape[1] :]
            for ex, toks in zip(batch, new_tokens):
                raw = tokenizer.decode(toks, skip_special_tokens=adapter.skip_special_tokens)
                if not adapter.skip_special_tokens and tokenizer.pad_token:
                    raw = raw.replace(tokenizer.pad_token, "")
                thinking, sql = adapter.postprocess(raw)
                pred_lines.append(sql)
                gens.write(
                    json.dumps(
                        {
                            "db_id": ex["db_id"],
                            "question": ex["question"],
                            "gold": ex["query"],
                            "raw": raw,
                            "thinking": thinking,
                            "pred": sql,
                            "new_tokens": int((toks != tokenizer.pad_token_id).sum()),
                            "batch_latency_s": round(latency, 2),
                        }
                    )
                    + "\n"
                )
                gens.flush()

    (run_dir / "pred.txt").write_text("\n".join(pred_lines) + "\n")

    del model, tokenizer
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
    args = ap.parse_args()

    cfg = get_model_config(args.model)
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    run_dir = make_run_dir(cfg["repo_id"], args.run_id)
    generate(cfg, load_dev(args.limit, args.seed), run_dir, args.seed)
    print(run_dir)


if __name__ == "__main__":
    main()
