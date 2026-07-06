# Variant of config_NVLink_H100_BF16.sh with MINIBS = 24 (1.5x of Job 198).
#
# Math:
#   GBS = MINIBS * (DGXNGPU * DGXNNODES) / (TP * PP * CP)
#       = 24 * (2*1) / (2*1*1) = 24  (vs Job 198 GBS=16, Job 196 GBS=8)
#   grad_accum_steps = GBS / MICRO_BATCH_SIZE / DP = 24/1/1 = 24
#
# Hypothesis (based on Job 198 result of -11.5% wall time at GBS=16):
# - Step time will scale linearly: 5.40s × 24/8 = ~16.2s (3x of Job 196).
# - Step count drops further: ~172k samples / 24 = ~7,200 steps (Job 198: 10,752).
# - Samples-to-convergence may decrease further (less gradient noise),
#   continuing the trend Job 196 (184k) → Job 198 (172k).
# - Estimated wall time: ~32h (vs Job 198 33.79h, additional -5%).
#
# Memory: Job 198 used 90.4 GB / 94 GB (96%) with MINIBS=16. MINIBS=24 should
# use similar peak memory since MICRO_BATCH_SIZE=1 unchanged, but gradient
# accumulation workspace might add small amount. Quick test (200 step)
# verifies before committing 30+ hour full run.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=24                # ← only change vs config_NVLink_H100_BF16_minibs16.sh
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1
export WARMUP_VALIDATION_STEPS=0

export LR=0.0004
export WARMUP_STEPS=128
export VAL_CHECK_INTERVAL=384

export DGXNNODES=1
export DGXNGPU=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=3600
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export TRANSFORMER_ENGINE=False
export USE_TE_OPS=False
export CE_FUSION_IMPL=native

export FP8=False
export FP8_HYBRID=False
export FP8_DPA=False
export FP8_PARAM_GATHER=False

export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export UB_TP_COMM_OVERLAP=0

export FULL_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
