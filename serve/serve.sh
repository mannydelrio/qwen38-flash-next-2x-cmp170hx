#!/usr/bin/env bash
set -euo pipefail
ROOT=${FLASHNEXT_ROOT:-/opt/flashnext}
source "$ROOT/vllm-backport/.venv/bin/activate"
export CUDA_HOME="$ROOT/vllm-backport/.venv/lib/python3.12/site-packages/nvidia/cu13"
export NVCC_CCBIN=/usr/bin/g++-13
export MAX_JOBS=2
export VLLM_PLUGINS=tinybeast_ple_int4,tinybeast_ple_ssd
export TB_PLE_SSD=1
export CUDA_MODULE_LOADING=EAGER
export TB_PLE_SSD_PREFETCH=1
export TB_TORCH_MEM_CAP_GIB=61.5
export VLLM_DETERMINISTIC_MOE_ALIGN=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export VLLM_PP_LAYER_PARTITION=25,23
# 8 requests at once needs decode graphs up to 32 tokens: vLLM rounds graph sizes to multiples of
# 1 + 3 draft tokens = 4, so a max of 16 covers only 4 requests and the rest run without graphs.
exec vllm serve "$ROOT/model-awq-int4ple" \
  --served-model-name tinybeast-flashnext-uncensored \
  --load-format safetensors --safetensors-load-strategy lazy \
  --pipeline-parallel-size 2 --dtype bfloat16 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,12,16,20,24,28,32],"max_cudagraph_capture_size":32}' \
  --gpu-memory-utilization 0.94 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --host 127.0.0.1 --port 18084
