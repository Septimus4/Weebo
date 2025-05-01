# bench_qwen3.py

import time, json, pynvml, openai
from typing import cast
from openai.types.chat import ChatCompletionUserMessageParam

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="EMPTY",
)

PROMPT = (
    "Explain the quicksort algorithm in Python, "
    "then provide an iterative version."
)

def vram_mb():
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 ** 2

messages: list[ChatCompletionUserMessageParam] = cast(
    list[ChatCompletionUserMessageParam],
    [{"role": "user", "content": PROMPT}],
)

start_vram = vram_mb()
t0 = time.time()

response = client.chat.completions.create(
    model="hf.co/unsloth/Qwen3-30B-A3B-GGUF:Q5_K_M",
    messages=messages,
    max_tokens=2048,
    temperature=0.6,
    top_p=0.95,
)

elapsed = time.time() - t0
usage = response.usage
tokens_out = usage.completion_tokens

print(json.dumps({
    "avg_token_latency_s": elapsed / usage.total_tokens,
    "total_latency_s": elapsed,
    "tok_per_sec": tokens_out / elapsed,
    "peak_vram_MB": max(start_vram, vram_mb())
}, indent=2))
