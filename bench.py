import time
import json
import pynvml
import openai
from openai.types.chat import ChatCompletionUserMessageParam

# Configuration constants
MODEL_NAME = "qwen3:30b"
PROMPT = (
    "Explain the quicksort algorithm in Python, "
    "then provide an iterative version."
)
MAX_TOKENS = 2048
TEMPERATURE = 0.6
TOP_P = 0.95

# Initialize NVML once
pynvml.nvmlInit()
DEVICE_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_vram_usage() -> float:
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(DEVICE_HANDLE)
    return memory_info.used / (1024 ** 2)

def run_benchmark() -> dict:
    try:
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="EMPTY",
        )

        messages: list[ChatCompletionUserMessageParam] = [{"role": "user", "content": PROMPT}]

        start_vram = get_vram_usage()
        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        elapsed = time.time() - start_time
        usage = response.usage
        tokens_out = usage.completion_tokens

        return {
            "avg_token_latency_s": elapsed / usage.total_tokens,
            "total_latency_s": elapsed,
            "tok_per_sec": tokens_out / elapsed,
            "peak_vram_MB": max(start_vram, get_vram_usage())
        }

    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return {}

if __name__ == "__main__":
    results = run_benchmark()
    if results:
        print(json.dumps(results, indent=2))
    else:
        print("Benchmark failed")

    pynvml.nvmlShutdown()