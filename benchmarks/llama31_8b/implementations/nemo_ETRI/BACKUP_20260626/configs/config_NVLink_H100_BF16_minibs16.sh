# Variant of config_NVLink_H100_BF16.sh with MINIBS doubled (8 → 16).
#
# Math:
#   GBS = MINIBS * (DGXNGPU * DGXNNODES) / (TP * PP * CP)
#       = 16 * (2 * 1) / (2 * 1 * 1) = 16  (vs Job 196's GBS=8)
#   grad_accum_steps_per_train_step = GBS / MICRO_BATCH_SIZE / DP = 16/1/1 = 16
#
# Hypothesis:
# - Each training step ≈ 2× compute (16 micro-batches instead of 8),
#   so step time will roughly double (5.40s → ~10.8s expected).
# - But samples-to-convergence should stay similar (~165k),
#   so step count drops by ~half (~11.5k vs 23k).
# - Net wall time: similar OR better, AND larger GBS = lower gradient noise
#   = potentially more stable / faster convergence per sample.
# - LR scaling: keep LR=4e-4 for first test (sqrt-rule would suggest ~5.66e-4
#   but stay with reference value to isolate batch-size effect).
#
# Memory risk: Activation memory per micro-batch unchanged
# (MICRO_BATCH_SIZE=1 same as Job 196). Gradient accumulation typically
# doesn't add memory. Should be safe. Quick test (200 step) verifies.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=16                # ← only change vs config_NVLink_H100_BF16.sh
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
