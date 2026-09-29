import re

from .base import ModelAdapter

FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)
ANALYSIS_RE = re.compile(r"<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)", re.S)
SPECIAL_RE = re.compile(r"<\|[a-z_]+\|>")


class GptOssAdapter(ModelAdapter):
    """Parses the harmony format; requires decoding with special tokens kept."""

    skip_special_tokens = False

    def chat_template_kwargs(self) -> dict:
        return {"reasoning_effort": self.cfg.get("reasoning_effort", "low")}

    def default_generation(self) -> dict:
        return {"do_sample": True, "temperature": 1.0, "top_p": 1.0}

    def split_thinking(self, raw: str) -> tuple[str | None, str]:
        analysis = ANALYSIS_RE.search(raw)
        final = FINAL_RE.search(raw)
        thinking = analysis.group(1).strip() if analysis else None
        answer = SPECIAL_RE.sub(" ", final.group(1)) if final else ""
        return thinking, answer
