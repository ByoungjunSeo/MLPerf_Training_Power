# Identical to config_PCIeH100_TTA.sh (the config that succeeded as Job 175,
# 37.9h wall-clock to TARGET_LOG_PPL=3.3). The only difference vs Job 175 is
# the hardware: this run has the NVLink bridge installed (NV12, ~211 GB/s
# busbw), so the TP all-reduce should be faster — modest step-time win
# expected over the 5.89s/step PCIe baseline.
#
# Why this is NOT Step 1 (FP8 + TE):
# Step 1 quick-test (Job 182) showed -38.4% step time but in long convergence
# (Job 194) it diverged at step ~1500 and climbed to eval_loss 7.93. The FP8
# numerical drift compounds with the small GBS=8 noise. BF16 + TE off (the
# Job 175 recipe) is the only known-converging configuration on this 2-GPU
# environment.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh
# NOTE: config_common_fp8attn.sh exports NVTE_DPA_FP8_FORMAT="" which would
# crash TransformerEngine *if it were imported*. Here TE is off, so the
# crash is dormant — same as in Job 175.

export MINIBS=8
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
