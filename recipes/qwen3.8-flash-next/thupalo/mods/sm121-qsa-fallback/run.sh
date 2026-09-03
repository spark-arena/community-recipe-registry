#!/usr/bin/env bash
# sparkrun mod: SM121 (DGX Spark / GB10) QSA varlen-attention fallback for
# lmsysorg/sglang:qwen38flashnext. Runs inside every container before serve.
# Source: MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks (start.sh .patch/).
set -euo pipefail
cd "$(dirname "$0")"
python3 apply.py
