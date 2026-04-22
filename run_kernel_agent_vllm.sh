#!/usr/bin/env bash
# Run kernel generation (agent.py) against a vLLM OpenAI-compatible API.
#
# query_server server_type=vllm uses --server-address / --server-port (not OPENAI_BASE_URL).
# OPENAI_BASE_URL below is kept for tools that read it; agent + vllm path needs the two flags.
# Install: pip install openai  (+ torch / CUDA as needed for validation & ncu)
#
# vLLM listens on http://HOST:PORT/v1 by default (PORT often 8000).
#
# API key (required): pass --api-key to agent.py (same secret as vLLM ``--api-key``).
# Example:
#   export SERVER_ADDRESS=10.102.207.7
#   export SERVER_PORT=8000
#   export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct   # must match the model name vLLM serves
  # ./run_kernel_agent_vllm.sh \
  #   --api-key "${VLLM_API_KEY}" \
  #   --model "${MODEL_NAME}" \
  #   --task-file ./KernelBench/level1/1_Square_matrix_multiplication_.py \
  #   --work-dir ./runs/kgen_vllm \
  #   --max-rounds 3 \
  #   --no-ncu
#   # Resume (same task file + work_dir): --start-round 4 --max-rounds 100  → rounds 4..99
#   # GPUs: optional --gpu-idx 0  or  --gpu-idx 0,1  (overrides \${CUDA_VISIBLE_DEVICES}; else default 4)
#   # Thinking/reasoning: default ON. Examples:
#   #   --no-reasoning
#   #   --reasoning-except-rounds 1,2
#   #   --reasoning-only-rounds 0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SERVER_ADDRESS="${SERVER_ADDRESS:-10.102.97.49}"
SERVER_PORT="${SERVER_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-DONOTATTACKMYVLLM}"

# Must match vLLM --served-model-name. This script passes --model to agent.py.
# - If MODEL_NAME is unset: GET http://SERVER:PORT/v1/models (with Bearer API_KEY) and use
#   the first model id; on failure fall back to KERNEL_AGENT_MODEL or MiniMax-M2.5.
# - If MODEL_NAME is set (export ...): that value is kept.
# - CLI: pass --model ... in "$@"; argparse keeps the *last* --model.
if [[ -z "${MODEL_NAME:-}" ]]; then
  if MODEL_CAND="$(
    BASE_URL="http://${SERVER_ADDRESS}:${SERVER_PORT}" API_KEY="${VLLM_API_KEY}" python3 -c '
import json, os, sys, urllib.request
base = os.environ.get("BASE_URL", "").rstrip("/")
key = os.environ.get("API_KEY", "")
req = urllib.request.Request(base + "/v1/models")
if key:
    req.add_header("Authorization", "Bearer " + key)
with urllib.request.urlopen(req, timeout=30) as resp:
    d = json.load(resp)
rows = d.get("data") or []
if not rows:
    sys.exit(1)
m = rows[0]
for k in ("id", "root", "model"):
    v = m.get(k)
    if isinstance(v, str) and v.strip():
        print(v.strip())
        break
else:
    sys.exit(1)
'
  )"; then
    MODEL_NAME="${MODEL_CAND}"
  else
    MODEL_NAME="${KERNEL_AGENT_MODEL:-MiniMax-M2.5}"
    echo "run_kernel_agent_vllm.sh: could not list models from \${SERVER_ADDRESS}:\${SERVER_PORT}; using MODEL_NAME=${MODEL_NAME}" >&2
  fi
fi

# Script-only: --gpu-idx N | N,N,... — remaining args go to agent.py
AGENT_ARGS=()
_got_gpu_flag=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-idx)
      if [[ -z "${2:-}" ]]; then
        echo "run_kernel_agent_vllm.sh: --gpu-idx requires a value (e.g. 0 or 0,1)" >&2
        exit 1
      fi
      export CUDA_VISIBLE_DEVICES="$2"
      _got_gpu_flag=true
      shift 2
      ;;
    --gpu-idx=*)
      export CUDA_VISIBLE_DEVICES="${1#*=}"
      _got_gpu_flag=true
      shift
      ;;
    *)
      AGENT_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${_got_gpu_flag}" == false ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
fi

# Accelerate Assembling
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="9.0"

python3 "${SCRIPT_DIR}/agent.py" --server-type vllm \
  --model "${MODEL_NAME}" \
  --server-address "${SERVER_ADDRESS}" \
  --server-port "${SERVER_PORT}" \
  --max-rounds 100 \
  --max-tokens 90000 \
  --max-context-length 190000 \
  --api-key "${VLLM_API_KEY}" \
  "${AGENT_ARGS[@]}"
