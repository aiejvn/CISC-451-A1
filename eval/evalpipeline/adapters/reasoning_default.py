from .base import ModelAdapter


class ReasoningAdapter(ModelAdapter):
    """DeepSeek-R1-Distill / Phi-4-mini-reasoning style: <think>...</think> then answer.

    Some chat templates already emit the opening <think>, so only the closing tag is required.
    A response truncated inside the reasoning trace has no closing tag and yields the fallback SQL.
    """

    def default_generation(self) -> dict:
        return {"do_sample": True, "temperature": 0.6, "top_p": 0.95}

    def split_thinking(self, raw: str) -> tuple[str | None, str]:
        if "</think>" in raw:
            return super().split_thinking(raw)
        return raw.replace("<think>", "").strip(), ""
