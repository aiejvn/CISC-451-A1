"""Per-model-family prompting, sampling and output parsing."""
import re

FENCE_RE = re.compile(r"```(?:sql|sqlite)?\s*\n?(.*?)```", re.S | re.I)
STMT_RE = re.compile(r"\b(select|with)\b.*", re.S | re.I)
FALLBACK_SQL = "SELECT"


def extract_sql(text: str) -> str:
    fence = FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    stmt = STMT_RE.search(text)
    if not stmt:
        return FALLBACK_SQL
    sql = stmt.group(0).split(";")[0]
    sql = sql.replace("```", " ")
    return " ".join(sql.split()) or FALLBACK_SQL


class ModelAdapter:
    """Per-model-family prompting, sampling and output parsing."""

    skip_special_tokens = True

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def build_messages(self, user_prompt: str) -> list[dict]:
        return [{"role": "user", "content": user_prompt}]

    def chat_template_kwargs(self) -> dict:
        return {}

    def default_generation(self) -> dict:
        return {"do_sample": False}

    def generation_kwargs(self) -> dict:
        return {**self.default_generation(), **self.cfg.get("generation", {})}

    def split_thinking(self, raw: str) -> tuple[str | None, str]:
        if "</think>" in raw:
            thinking, answer = raw.rsplit("</think>", 1)
            return thinking.replace("<think>", "").strip(), answer
        return None, raw

    def postprocess(self, raw: str) -> tuple[str | None, str]:
        thinking, answer = self.split_thinking(raw)
        return thinking, extract_sql(answer)


FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)
ANALYSIS_RE = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)
SPECIAL_RE = re.compile(r"<\|[a-z_]+\|>")


class GptOssAdapter(ModelAdapter):
    """Parses the harmony format; requires decoding with special tokens kept."""

    skip_special_tokens = False

    def chat_template_kwargs(self) -> dict:
        return {"reasoning_effort": self.cfg.get("reasoning_effort", "low")}

    def default_generation(self) -> dict:
        return {"do_sample": False}

    def split_thinking(self, raw: str) -> tuple[str | None, str]:
        analysis = ANALYSIS_RE.search(raw)
        final = FINAL_RE.search(raw)
        thinking = analysis.group(1).strip() if analysis else None
        answer = SPECIAL_RE.sub(" ", final.group(1)) if final else ""
        return thinking, answer


class Qwen3Adapter(ModelAdapter):
    @property
    def thinking(self) -> bool:
        return bool(self.cfg.get("thinking", False))

    def chat_template_kwargs(self) -> dict:
        return {"enable_thinking": self.thinking}

    def default_generation(self) -> dict:
        return {"do_sample": False}


class ReasoningAdapter(ModelAdapter):
    """DeepSeek-R1-Distill / Phi-4-mini-reasoning style: <think>...</think> then answer.

    Some chat templates already emit the opening <think>, so only the closing tag is required.
    A response truncated inside the reasoning trace has no closing tag and yields the fallback SQL.
    """

    def default_generation(self) -> dict:
        return {"do_sample": False}

    def split_thinking(self, raw: str) -> tuple[str | None, str]:
        if "</think>" in raw:
            return super().split_thinking(raw)
        return raw.replace("<think>", "").strip(), ""


GEMMA_SPECIAL_RE = re.compile(r"<turn\|>|<\|turn>|<eos>|<pad>|<bos>")


class Gemma4Adapter(ModelAdapter):
    """Gemma 4 -it: <|channel>thought\n...<channel|>answer<turn|>; thinking toggled by enable_thinking.

    The channel/turn markers are special tokens, so decoding keeps them and they are stripped here.
    """

    skip_special_tokens = False

    @property
    def thinking(self) -> bool:
        return bool(self.cfg.get("thinking", False))

    def chat_template_kwargs(self) -> dict:
        return {"enable_thinking": self.thinking}

    def default_generation(self) -> dict:
        # Greedy, like the other adapters. Model-card sampling is temperature=1.0, top_p=0.95, top_k=64;
        # set it per model with `generation:` in models.yaml.
        return {"do_sample": False}

    def split_thinking(self, raw: str) -> tuple[str | None, str]:
        if "<channel|>" in raw:
            thinking, answer = raw.rsplit("<channel|>", 1)
            thinking = thinking.replace("<|channel>thought", "").strip() or None
        elif "<|channel>" in raw:  # truncated inside the thought channel
            return raw.replace("<|channel>thought", "").strip(), ""
        else:
            thinking, answer = None, raw
        return thinking, GEMMA_SPECIAL_RE.sub(" ", answer)


ADAPTERS = {
    "default": ModelAdapter,
    "qwen3": Qwen3Adapter,
    "reasoning_default": ReasoningAdapter,
    "gpt_oss": GptOssAdapter,
    "gemma4": Gemma4Adapter,
}


def get_adapter(cfg: dict) -> ModelAdapter:
    family = cfg.get("family", "default")
    if family not in ADAPTERS:
        raise ValueError(f"unknown family {family!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[family](cfg)
