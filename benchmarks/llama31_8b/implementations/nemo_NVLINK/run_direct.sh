#!/bin/bash
# Direct execution script for 2x PCIe H100 WITHOUT SLURM
# This script runs inside the container or directly on the host
#
# Usage (Docker):
#   export DATADIR=/path/to/data
#   export LOGDIR=/path/to/logs
#   source config_PCIEH100x2_12h.sh
#   docker run --gpus all -v $DATADIR:/data -v $LOGDIR:/results -v $(pwd):/workspace/llm \
#       -e DATADIR=/data -e LOGDIR=/results --ipc=host <container> /workspace/llm/run_direct.sh
#
# Usage (Inside container):
#   export DATADIR=/preproc_data
#   source config_PCIEH100x2_12h.sh
#   ./run_direct.sh

set -eux

# Required environment variables
: "${DGXNGPU:=2}"
: "${SEED:=1234}"

# Set defaults for distributed training
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}

# Model and data configuration
: "${MODEL_SIZE:=8b}"
: "${NEMO_RESULTS_SUBDIR:=$(date +'%y%m%d%H%M%S%N')}"

# Parallelism defaults
: "${TENSOR_MODEL_PARALLEL:=1}"
: "${PIPELINE_MODEL_PARALLEL:=1}"
: "${CONTEXT_PARALLEL:=1}"
: "${INTERLEAVED_PIPELINE:=null}"

# Batch size defaults
: "${MINIBS:=8}"
: "${MICRO_BATCH_SIZE:=1}"

# Learning rate defaults
: "${LR:=0.0003}"
: "${MIN_LR:=0.00003}"
: "${WARMUP_STEPS:=500}"

# Training defaults
: "${MAX_STEPS:=12000}"
: "${VAL_CHECK_INTERVAL:=768}"
: "${LIMIT_VAL_BATCHES:=64}"
: "${TARGET_LOG_PPL:=3.3}"
: "${LOG_EVERY_N_STEPS:=10}"

# FP8 defaults
: "${FP8:=True}"
: "${FP8_HYBRID:=True}"
: "${FP8_DPA:=False}"

# Memory defaults
: "${ACT_CKPT_GRANULARITY:=selective}"
: "${USE_DIST_OPTIMIZER:=True}"
: "${SEQ_PARALLEL:=True}"

# Warmup defaults
: "${RUN_WARMUP_ON_SYNTH_DATA:=1}"
: "${WARMUP_TRAIN_STEPS:=2}"
: "${WARMUP_VALIDATION_STEPS:=2}"

# NCCL settings for PCIe
export NCCL_ALGO=${NCCL_ALGO:-Ring}
export NCCL_PROTO=${NCCL_PROTO:-Simple}
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Disable problematic features for PCIe
export TP_COMM_OVERLAP=${TP_COMM_OVERLAP:-False}
export MC_TP_OVERLAP_AG=${MC_TP_OVERLAP_AG:-False}
export MC_TP_OVERLAP_RS=${MC_TP_OVERLAP_RS:-False}

# Create results directory
mkdir -p /results/${NEMO_RESULTS_SUBDIR}

echo "============================================"
echo "MLPerf LLM Training - Direct Mode"
echo "============================================"
echo "GPUs: ${DGXNGPU}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Global Batch Size: $((MINIBS * DGXNGPU / (TENSOR_MODEL_PARALLEL * PIPELINE_MODEL_PARALLEL * CONTEXT_PARALLEL)))"
echo "Micro Batch Size: ${MICRO_BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Max Steps: ${MAX_STEPS}"
echo "Target PPL: ${TARGET_LOG_PPL}"
echo "============================================"

# Build extra args for pretrain.py
EXTRA_ARGS=""
EXTRA_ARGS+=" exp_manager.explicit_log_dir=/results/${NEMO_RESULTS_SUBDIR}"

# Run training with torchrun
torchrun \
    --nproc_per_node=${DGXNGPU} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /workspace/llm/pretrain.py \
    ${EXTRA_ARGS}

echo "Training completed!"
