#!/usr/bin/env bash
# run.sh — executed *inside* the vLLM container to patch ple_layer.py in place.
#
# Usage (from the host):
#   docker run --rm -v "$SCRIPT_DIR/run.sh:/run.sh:ro" "$IMAGE" bash /run.sh
# or, if you already have a running container:
#   docker exec <container> bash /run.sh

set -euo pipefail

# Fall back to plain echo if info/err aren't already defined by a sourcing script.
if ! declare -F info >/dev/null 2>&1; then
    info() { echo "[INFO] $*"; }
fi
if ! declare -F err >/dev/null 2>&1; then
    err() { echo "[ERROR] $*" >&2; }
fi

PLE_PATH="/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py"

info "Patching ple_layer.py in place at $PLE_PATH ..."

if [[ ! -f "$PLE_PATH" ]]; then
    err "ple_layer.py not found at $PLE_PATH. Has the image changed?"
    exit 1
fi

python3 - "$PLE_PATH" <<'PYEOF'
import sys
import shutil

path = sys.argv[1]

with open(path, "r") as f:
    content = f.read()

# Shim: rebind the resolver so PLE_QUANT_OVERRIDE=fp8 selects the global-scale
# FP8 method regardless of the parent quant config (ModelOpt NVFP4 excludes *.ple.*).
anchor = "class Qwen3_8FlashNextNGramEmbedding(PleOffloadLayer):"

if "_ORIG_PLE_QUANT_RESOLVER" in content:
    print("WARNING: source already carries an override shim; leaving file untouched.")
    sys.exit(0)

if content.find(anchor) < 0:
    print("WARNING: Could not find patch target. Image may have changed.")
    sys.exit(1)

shim = (
    "_ORIG_PLE_QUANT_RESOLVER = _get_ple_embedding_quant_method\n\n"
    "def _get_ple_embedding_quant_method(quant_config, prefix):\n"
    '    """Re-resolve the PLE embedding method under an override."""\n'
    "    from os import getenv as _ple_getenv\n"
    '    if _ple_getenv("PLE_QUANT_OVERRIDE", "").strip().lower() == "fp8":\n'
    "        # NVFP4 checkpoints ship PLE as FP8 shards + one global weight_scale\n"
    "        # yet exclude *.ple.* from the parent quant config; honor the shards.\n"
    "        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n"
    "    return _ORIG_PLE_QUANT_RESOLVER(quant_config, prefix)\n\n\n"
)

pos = content.find(anchor)
patched = content[:pos] + shim + content[pos:]

backup = path + ".orig"
shutil.copy(path, backup)

with open(path, "w") as f:
    f.write(patched)

print(f"Patch applied successfully (backup kept at {backup}).")
PYEOF

status=$?

if [[ $status -ne 0 ]]; then
    err "Failed to patch ple_layer.py."
    exit 1
fi

info "Done. ple_layer.py patched in place."
