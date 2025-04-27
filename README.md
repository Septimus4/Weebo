# Local Assistant – Self‑Hosted LLM Workbench
---

## 1. Purpose
This repository stores the configuration files, scripts, and notes for a self‑contained **Local Assistant** that powers:

* conversational help (DeepSeek R1 32 B)
* software engineering copiloting (Codestral 22 B)
* fast text embeddings (nomic‑embed‑text)

All models are served through **Ollama**, wired into **Continue.dev** inside your IDE, and exposed on your LAN via **Open WebUI**.


## 2. Hardware Profile
| Component | Specification |
|-----------|---------------|
| CPU & RAM | 64 GB system RAM on an 8‑core/16‑thread processor |
| GPU       | NVIDIA RTX 5090 — 32 GB VRAM, CUDA 12 drivers |
| OS        | Ubuntu 25 (Kernel 6.x, libc 2.40+) |


## 3. Software Stack
<details>
<summary>Diagram</summary>

```
┌───────────────┐      ┌───────────────────┐       ┌────────────────┐
│  Continue.dev │◄────►│    Ollama API     │◄────►│  Open WebUI     │
└───────────────┘      └─────────┬─────────┘       └──────▲─────────┘
                                 │                        │
                                 ▼                        │
                     ┌───────────────────────┐            │
                     │     GPU Memory        │◄───────────┘
                     └───────────────────────┘
```

</details>

### 3.1 Runtime packages
* **Ollama 0.2.x** – pulls, quantises, and serves GGUF checkpoints
* **Continue.dev v2** – bridges Ollama with editor context providers
* **Open WebUI 0.4+** – zero‑auth chat front‑end for LAN access
* **CUDA 12.4 & cuBLAS / cuDNN** – full FP16 & INT4 hardware path


## 4. Models & Roles
| Model | Quantisation | Context Window | Role(s) |
|-------|--------------|---------------|---------|
| **DeepSeek‑R1‑Distill‑Qwen‑32B** | Q5_K_M | 32 k | Chat, reasoning |
| **Codestral‑22B‑v0.1** | Q6_K | 12 k | Edit, apply, autocomplete |
| **nomic‑embed‑text** | FP16 | 4 k | Embeddings |

All layers are resident on the GPU (`LLAMA_N_GPU_LAYERS=-1`), avoiding PCIe latency and keeping batch throughput high.


## 5. Key Configuration Highlights
```yaml
# .continue/config.yaml (excerpt)
environment:
  CUDA_VISIBLE_DEVICES: "0"        # lock to primary RTX 5090
  LLAMA_CACHE_TYPE_K: q4_0         # quantised KV cache (saves ~40 % VRAM)
  LLAMA_CACHE_TYPE_V: q4_0
  LLAMA_N_CTX: 32768               # global upper bound; per‑model overrides below
  LLAMA_N_GPU_LAYERS: -1           # pin every layer to GPU memory
  LLAMA_OPENBLAS_NUM_THREADS: 16   # saturate CPU for residual ops

models:
  - name: Codestral
    contextWindow: 12288           # 12 k fully in‑VRAM
    gpuLayers: -1                  # no CPU offload
    numThreads: 16

  - name: DeepSeek R1
    contextWindow: 32768           # 32 k for long chats/codebases
    env:
      LLAMA_NO_KV_OFFLOAD: "1"     # keep KV‐cache on‑GPU

  - name: Nomic Embed
    roles: [embed]
```

### Why these choices?
| Setting | Rationale |
|---------|-----------|
| **Q5_K_M / Q6_K quantisation** | Maintains >95 % perplexity fidelity while halving VRAM consumption compared to FP16. |
| **KV cache `q4_0`** | Further 30–40 % reduction in KV size with negligible accuracy drop; critical for 32 k contexts. |
| **`LLAMA_N_GPU_LAYERS=-1`** | Keeps forward/backward passes wholly on the 5090’s 1 TB/s bandwidth. |
| **`LLAMA_OPENBLAS_NUM_THREADS=16`** | Aligns with physical cores, maximising GEMM throughput when CPU is hit. |


## 6. Running the stack
```bash
# 1. Pull models (one‑off)
ollama pull hf.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF:Q5_K_M
ollama pull hf.co/SanctumAI/Codestral-22B-v0.1-GGUF:Q6_K
ollama pull nomic-embed-text:latest

# 2. Launch Ollama server
OLLAMA_NUM_PARALLEL=4 \
LLAMA_OPENBLAS_NUM_THREADS=16 \   # match config
ollama serve &

# 3. Start Open WebUI (Docker)
docker compose up -d open-webui
```

> **Tip:** Keep `nvidia‑smi dmon` in a side‑pane to watch VRAM use per model.


## 7. IDE Workflow
1. **Chat** – Ask natural‑language questions via DeepSeek‑R1.
2. **Edit & Apply** – Highlight code, invoke Codestral‑powered *Edit* to refactor.
3. **Autocomplete** – Accept inline suggestions from Codestral.
4. **Search** – Embedding provider auto‑indexes your workspace for semantic search.


## 8. Serving to the network
Open WebUI binds to `0.0.0.0:3000` (configurable).  
If you don’t want LAN exposure, `export WEBUI_BIND=127.0.0.1` before launch or use a reverse‑proxy with auth.


## 9. Performance Benchmarks (optional)
| Metric | DeepSeek‑R1 (32 k) | Codestral (12 k) |
|--------|-------------------|-------------------|
| **Prompt throughput** | ≈ 65 tok/s | ≈ 78 tok/s |
| **Generate throughput** | ≈ 100 tok/s | ≈ 115 tok/s |
| **VRAM at 0 k ctx** | 26 GB | 22 GB |

Numbers captured with `--numa-report` off, CUDA 12.4, 535.xx driver.


## 10. Roadmap
* Add **RAG** layer using `llama‑index` + local docs corpus.

