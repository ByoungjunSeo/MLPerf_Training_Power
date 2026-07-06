# BS=32 RCP-eligible config for MLPerf 5.1 llama31_8b.
# Reference hyperparameters from training_5.1.0/rcps_llama31_8b.json (BS=32):
#   opt_base_learning_rate: 1e-3
#   opt_learning_rate_warmup_samples: 16348
#   gradient_accumulation_steps: 2 (reference, for DP=16)
#   Epochs to converge: [196608, 196608, 196608, ..., 233472]  (mean 215040)
#
# In our env (TP=2, DP=1):
#   GBS = MINIBS * (DGXNGPU / TP) = 32 * (2/2) = 32 ✓
#   gradient_accumulation_steps = GBS / MICRO_BATCH_SIZE / DP = 32/1/1 = 32
#     (reference's grad_accum=2 is from DP=16; our DP=1 forces grad_accum=32.
#      RCP checker matches by BS only, not by grad_accum, so this is OK.)
#   WARMUP_STEPS = warmup_samples / GBS = 16348/32 = 511 steps  (rounded up)
#
# Hypothesis:
# - Step time = Job 199 (MINIBS=24, 15.94s) * 32/24 = ~21.25s
# - Samples-to-conv: reference distribution mean 215,040 (range 196k-233k)
# - ETA: 215040/32 * 21.25s = 142,800s = 39.67h + 2h overhead = ~42h
# - LR=1e-3 (2.5x larger than Job 198's 4e-4) but GBS doubled - should
#   be stable (BF16 + larger batch tolerates higher LR).
# - Memory: Job 200 (MINIBS=24) used 90 GB. MINIBS=32 likely ~92 GB.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=32                         # GBS = 32
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1
export WARMUP_VALIDATION_STEPS=0

export LR=0.001                          # Reference: 1e-3 for BS=32
export WARMUP_STEPS=511                  # = ceil(16348 / 32) for warmup_samples=16348
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
