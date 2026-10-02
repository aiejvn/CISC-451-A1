# Task 1:

* Decoder-only LLMs
    Qwen3-8b
    GPT-OSS-20b

* Reasoning LLMs
    DeepSeek-R1-Distill-Qwen-7B	

RQ1: How do (selected) LLms perform on cross-domain Text-to-SQL?

## Methodology:

In this report we evaluate the performance of decoder-only language models as well as reasoning LLMs on cross-domain SQL. Specifically, we evaluate the performance of decoder-only LLMs Qwen3-8B [needs citation] (non-MoE) and GPT-OSS-20B [needs citation] (3.6B active params MoE), as well as reasoning LLM Deepseek-R1-Distill-Qwen-14B (non-MoE) by downloading the checkpoints from Huggingface and running the models locally on 1x NVIDIA GeForce RTX 4090 GPU.