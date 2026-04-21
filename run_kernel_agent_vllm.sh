#!/usr/bin/env bash
# Run kernel generation (agent.py) against a vLLM OpenAI-compatible API.
#
# query_server server_type=vllm uses --server-address / --server-port (not OPENAI_BASE_URL).
# OPENAI_BASE_URL below is kept for tools that read it; agent + vllm path needs the two flags.
# Install: pip install openai  (+ torch / CUDA as needed for validation & ncu)
#
# vLLM listens on http://HOST:PORT/v1 by default (PORT often 8000).
#
# Example:
#   export SERVER_ADDRESS=10.102.207.7
#   export SERVER_PORT=8000
#   export MODEL_NAME=Qwen/Qwen2.5-7B-Instruct   # must match the model name vLLM serves
  # ./run_kernel_agent_vllm.sh \
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

SERVER_ADDRESS="${SERVER_ADDRESS:-10.102.207.7}"
SERVER_PORT="${SERVER_PORT:-8000}"

# Accelerate Assembling
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="9.0"

exec python "${SCRIPT_DIR}/agent.py" --server-type vllm \
  --server-address "${SERVER_ADDRESS}" \
  --server-port "${SERVER_PORT}" \
  "$@"
