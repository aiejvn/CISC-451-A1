from .base import ModelAdapter


class Qwen3Adapter(ModelAdapter):
    @property
    def thinking(self) -> bool:
        return bool(self.cfg.get("thinking", False))

    def chat_template_kwargs(self) -> dict:
        return {"enable_thinking": self.thinking}

    def default_generation(self) -> dict:
        # Sampling values from the Qwen3 model card; greedy is discouraged.
        if self.thinking:
            return {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
        return {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
