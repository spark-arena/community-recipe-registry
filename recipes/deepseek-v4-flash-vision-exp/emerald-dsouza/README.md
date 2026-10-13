# DeepSeek-V4-Flash-Vision-Exp on 2× DGX Spark (vLLM + DSpark)

Official FP8 Vision-Exp on `vllm/vllm-openai:v0.31.0`. 2-node TP=2, DSpark
speculative decoding, FP8 KV cache, 400k context, up to 8 images per prompt.

## GB10 settings

- `max_num_batched_tokens: 4096` — sized for GB10 unified memory: the startup
  KV profile runs the vision encoder at this budget.
- `gpu_memory_utilization: 0.83` — leaves headroom for the CPU heap and page
  cache that share the 121 GB unified pool.
- `distributed_executor_backend: mp` — the multiprocessing executor drives the
  2-node TP=2 setup.

## Run

```bash
sparkrun run @community/deepseek-v4-flash-vision-exp-fp8-dspark-vllm-emerald-dsouza
```

~10 min cold start with the model cached on the nodes.
