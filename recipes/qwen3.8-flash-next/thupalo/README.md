# Qwen3.8-Flash-Next-NVFP4 · 2× DGX Spark · SGLang TP2 (NEXTN)

Serves [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(Qwen4-architecture preview: 176B total / 6B active MoE + 51B PLE n-gram table, NVFP4) on
**two DGX Sparks (GB10 / SM121)** in tensor parallel over the CX7 RoCEv2 fabric, with
NEXTN speculative decoding (the MTP draft ships inside the checkpoint).

```bash
sparkrun run @community/qwen3.8-flash-next-nvfp4-nextn-sglang-thupalo --cluster <your-2-node-cluster>
```

## Why there is a `mods/` directory

The stock `lmsysorg/sglang:qwen38flashnext` image cannot serve this model on GB10: the
Qwen sparse-attention (QSA) backend resolves its varlen attention to flash-attn-4 CuTe
kernels, which fail MLIR compilation on SM121. `mods/sm121-qsa-fallback` installs a Triton
FlashDecoding-style varlen kernel (CUDA-graph safe, one query row per sequence, GQA) and
patches `qwen_sparse_attn_backend.py` to use it whenever `is_sm100_supported()` is false,
so datacenter GPUs keep the stock path. sparkrun applies the mod in every container before
the serve command (`pre_exec`); no custom image is required.

The kernel and patch are taken verbatim from
[MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
(MIT, © Mia's AI Lab) — see `mods/sm121-qsa-fallback/LICENSE`. A prebuilt image with the
same patch baked in is available as `ghcr.io/thupalo/qwen38-flashnext-sm121:20260827`.

## Memory notes (GB10 unified memory)

* PLE n-gram table (~26 GB/rank, fp8) is host-offloaded (`--ple-offload-embedding`) — it
  still counts against the 121 GB unified pool.
* `mem-fraction-static 0.78` leaves ~22 GB free after CUDA-graph capture and gives a
  636k-token KV pool (full 262k context). `0.70` is ~5–10 % faster at c≤2 but caps the
  pool below one full-context request.
* Never probe this model with `--load-format dummy` on GB10 (hard-freezes the node).
* `PYTORCH_CUDA_ALLOC_CONF` is deliberately blanked (expandable segments wedged GB10
  nodes with this model upstream).

## Measured (llama-benchy, pp2048/tg128, 5 runs, 2× DGX Spark, sparkrun 0.3.6)

| depth | concurrency | prefill t/s | decode t/s (aggregate) | TTFT ms |
|---|---|---|---|---|
| 0 | 1 | 2413 | 37.3 | 856 |
| 0 | 2 | 2611 | 59.0 | 1611 |
| 0 | 4 | 2193 | 73.1 | 3604 |
| 16384 | 1 | 890 | 33.7 | 2352 |
| 16384 | 4 | 1899 | 69.3 | 3705 |

Expected VRAM: ~73 GB/rank weights + pools; ~100 GB used per node at 0.78.
First boot ≈ 35–45 min (135 GB weight load + Triton JIT + graph capture).
