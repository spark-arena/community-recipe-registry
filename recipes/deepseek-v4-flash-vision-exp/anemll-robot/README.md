# DeepSeek-V4-Flash-Vision-Exp · SGLang · anemll-robot

2× DGX Spark (GB10) recipe for [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) with **SGLang** + **DSPARK**.

## Why the mod?

Stock `lmsysorg/sglang:dev-v4f-2dgx-v2` VL `load_weights` buffers all non-vision tensors in a Python list (~2× peak RSS) and OOM-kills on GB10 after 48/48 shards. The `stream-vl-load-weights` mod installs a streamed generator overlay before serve.

Bring-up write-up: https://github.com/Anemll/SGLang-DSv4F-vision-2xSparks

## Run

```bash
sparkrun run @community/deepseek-v4-flash-vision-exp-fp8-dspark-sglang-anemll-robot --tp 2
```

## Arena benchmark

```bash
sparkrun arena login
sparkrun arena benchmark @community/deepseek-v4-flash-vision-exp-fp8-dspark-sglang-anemll-robot --tp 2
```

**Never** pass `--weight-loader-disable-mmap` on GB10.
