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
