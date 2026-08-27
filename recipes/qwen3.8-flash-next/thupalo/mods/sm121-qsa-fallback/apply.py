#!/usr/bin/env python3
"""SM121 QSA fallback patch for lmsysorg/sglang:qwen38flashnext (sparkrun pre_exec).

Idempotent port of the Dockerfile step in MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks:
installs qsa_fa_fallback.py next to qwen_sparse_attn_backend.py and makes the
varlen-attention resolver return the Triton fallback on non-SM100 GPUs.
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ATTN_DIR = "/sgl-workspace/sglang/python/sglang/srt/layers/attention"
BACKEND = os.path.join(ATTN_DIR, "qwen_sparse_attn_backend.py")
ANCHOR = "    try:\n        from flash_attn import flash_attn_varlen_func"
PATCH = (
    "    from sglang.srt.utils import is_sm100_supported\n"
    "    if not is_sm100_supported():\n"
    "        from sglang.srt.layers.attention.qsa_fa_fallback import triton_varlen_attn_func\n"
    "        return triton_varlen_attn_func\n"
) + ANCHOR

if not os.path.isfile(BACKEND):
    sys.exit(f"[sm121-qsa] {BACKEND} not found — image layout changed?")
shutil.copy(os.path.join(HERE, "qsa_fa_fallback.py"), os.path.join(ATTN_DIR, "qsa_fa_fallback.py"))
src = open(BACKEND).read()
if "qsa_fa_fallback" in src:
    print("[sm121-qsa] already patched"); sys.exit(0)
if ANCHOR not in src:
    sys.exit("[sm121-qsa] anchor not found — upstream image layout changed")
open(BACKEND, "w").write(src.replace(ANCHOR, PATCH, 1))
print("[sm121-qsa] qwen_sparse_attn_backend.py patched for SM121")
