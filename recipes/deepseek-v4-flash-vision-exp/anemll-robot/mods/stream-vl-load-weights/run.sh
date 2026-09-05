#!/usr/bin/env bash
# sparkrun mod: stream deepseek_v4_vl.load_weights on GB10 UMA.
# Stock lmsysorg/sglang:dev-v4f-2dgx-v2 buffers all non-vision tensors in a
# Python list (~2× peak RSS) and OOM-kills after 48/48 shards. This overlay
# streams via a generator so Load weight end succeeds (~74–76 GB/rank).
# Source: https://github.com/Anemll/SGLang-DSv4F-vision-2xSparks
set -euo pipefail
cd "$(dirname "$0")"
TARGET=/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4_vl.py
if [[ ! -f "$TARGET" ]]; then
  echo "[anemll] ERROR: expected VL model at $TARGET (wrong image?)" >&2
  exit 1
fi
cp -f deepseek_v4_vl.py "$TARGET"
echo "[anemll] installed streamed load_weights overlay → $TARGET"
