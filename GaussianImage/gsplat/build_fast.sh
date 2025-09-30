#!/usr/bin/env bash
# Fast editable install with CUDA extensions
# - Auto-detect GPU compute capability(s) -> TORCH_CUDA_ARCH_LIST
# - Enable Ninja + parallel compile
# - Optional sccache for gcc/g++/nvcc
# - Prefer wheels for deps; reuse existing build env via --no-build-isolation
#
# Usage:
#   ./build_fast.sh [extra pip args...]
# Examples:
#   USE_NINJA=1 FAST_BUILD=1 ./build_fast.sh -v
#   ./build_fast.sh -v
#   TORCH_CUDA_ARCH_LIST="8.9;9.0" ./build_fast.sh    # override detection
#   DISABLE_SCCACHE=1 ./build_fast.sh
#   FAST_BUILD=1 ./build_fast.sh                      # see setup.py toggle
#
set -Eeuo pipefail

detect_arch() {
  # Honor user override if provided
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "${TORCH_CUDA_ARCH_LIST}"
    return
  fi

  # Try nvidia-smi first (fast, no Python import)
  if command -v nvidia-smi >/dev/null 2>&1; then
    local caps
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/ //g' | sort -u)
    if [[ -n "$caps" ]]; then
      echo "$caps" | paste -sd';' -
      return
    fi
  fi

  # Fallback to Python (torch runtime), prefer python3 if available
  local pycmd="python"
  if command -v python3 >/dev/null 2>&1; then pycmd="python3"; fi

  local pyout
  pyout=$($pycmd - <<'PY' || true
import sys
try:
    import torch
    if torch.cuda.is_available():
        caps=set()
        for i in range(torch.cuda.device_count()):
            major,minor = torch.cuda.get_device_capability(i)
            caps.add(f"{major}.{minor}")
        if caps:
            print(";".join(sorted(caps)))
            sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
)
  if [[ -n "$pyout" ]]; then
    echo "$pyout"
    return
  fi

  # Last resort default (Ampere)
  echo "8.6"
}

ARCH_LIST=$(detect_arch)
export TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"

# Parallel build
export USE_NINJA="${USE_NINJA:-1}"
if command -v nproc >/dev/null 2>&1; then
  JOBS=$(nproc)
elif [[ "$(uname)" == "Darwin" ]]; then
  JOBS=$(sysctl -n hw.ncpu)
else
  JOBS=8
fi
export MAX_JOBS="${MAX_JOBS:-$JOBS}"

export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}"

if command -v sccache >/dev/null 2>&1 && [[ -z "${DISABLE_SCCACHE:-}" ]]; then
  export CC="sccache gcc"
  export CXX="sccache g++"
  export NVCC="sccache nvcc"
  export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"
  sccache --start-server || true
fi

export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-:all:}"

echo "==> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "==> USE_NINJA=$USE_NINJA  MAX_JOBS=$MAX_JOBS"
echo "==> TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
if command -v sccache >/dev/null 2>&1; then
  sccache --show-stats 2>/dev/null | head -n 5 || true
fi

pip install -e .[dev] --no-build-isolation "$@"
