from .base import ModelAdapter
from .gpt_oss import GptOssAdapter
from .qwen3 import Qwen3Adapter
from .reasoning_default import ReasoningAdapter

ADAPTERS = {
    "default": ModelAdapter,
    "qwen3": Qwen3Adapter,
    "reasoning_default": ReasoningAdapter,
    "gpt_oss": GptOssAdapter,
}


def get_adapter(cfg: dict) -> ModelAdapter:
    family = cfg.get("family", "default")
    if family not in ADAPTERS:
        raise ValueError(f"unknown family {family!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[family](cfg)
