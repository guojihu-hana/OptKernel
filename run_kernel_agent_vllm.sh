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
#   # Thinking/reasoning: default ON. Examples:
#   #   --no-reasoning
#   #   --reasoning-except-rounds 1,2
#   #   --reasoning-only-rounds 0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SERVER_ADDRESS="${SERVER_ADDRESS:-10.102.215.76}"
SERVER_PORT="${SERVER_PORT:-8000}"
# Must match vLLM --served-model-name. This script always passes --model to agent.py.
# - Override: export MODEL_NAME=... before running, OR pass --model ... again on the CLI
#   (argparse uses the *last* --model; this line comes first, so a trailing --model wins).
# - If you only assign MODEL_NAME=foo on a previous line without export, child shells
#   do not see it; use: export MODEL_NAME=foo  OR  MODEL_NAME=foo ./run_kernel_agent_vllm.sh ...
MODEL_NAME="${MODEL_NAME:-${KERNEL_AGENT_MODEL:-MiniMax-M2.5}}"
VLLM_API_KEY="${VLLM_API_KEY:-DONOTATTACKMYVLLM}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Accelerate Assembling
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="9.0"

exec python "${SCRIPT_DIR}/agent.py" --server-type vllm \
  --model "${MODEL_NAME}" \
  --server-address "${SERVER_ADDRESS}" \
  --server-port "${SERVER_PORT}" \
  --max-rounds 100 \
  --max-tokens 65536 \
  --api-key "${VLLM_API_KEY}" \
  "$@"
