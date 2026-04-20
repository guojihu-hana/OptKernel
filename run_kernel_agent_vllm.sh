#!/usr/bin/env bash
# Run kernel generation (agent.py) against a vLLM OpenAI-compatible API.
#
# Requires: query_server.py with OPENAI_BASE_URL support (uses OpenAI client).
# Install: pip install openai  (+ torch / CUDA as needed for validation & ncu)
#
# vLLM listens on http://HOST:PORT/v1 by default (PORT often 8000).
#
# Example:
#   export VLLM_HOST=10.102.207.7
#   export VLLM_PORT=8000
#   export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct   # must match the model name vLLM serves
#   ./run_kernel_agent_vllm.sh \
#     --model "${MODEL_NAME}" \
#     --task-file ./KernelBench/level1/1_Square_matrix_multiplication_.py \
#     --work-dir ./runs/kgen_vllm \
#     --max-rounds 3 \
#     --no-ncu
#   # Thinking/reasoning: default ON. Examples:
#   #   --no-reasoning
#   #   --reasoning-except-rounds 1,2
#   #   --reasoning-only-rounds 0
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VLLM_HOST="${VLLM_HOST:-10.102.207.7}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Accelerate Assembling
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="9.0"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

exec python "${SCRIPT_DIR}/agent.py" --server-type openai "$@"
