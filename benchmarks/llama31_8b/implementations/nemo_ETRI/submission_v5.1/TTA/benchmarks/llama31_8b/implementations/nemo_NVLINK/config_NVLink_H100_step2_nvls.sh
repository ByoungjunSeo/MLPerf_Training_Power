source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
# NOTE: do NOT source config_common_fp8attn{,_pcie}.sh — both export
# NVTE_DPA_FP8_FORMAT="" (empty) which crashes TransformerEngine import:
#   _dpa_fp8_format = formats[os.getenv("NVTE_DPA_FP8_FORMAT", "HYBRID")]
# The default "HYBRID" is only used when the env var is *unset*, not empty.

# ===== Parallelism (same as PCIe baseline) =====
export MINIBS=8
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

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

# ===== STEP 1 settings (carried forward) =====
export TRANSFORMER_ENGINE=True
export USE_TE_OPS=True
export CE_FUSION_IMPL=te

export FP8=True
export FP8_HYBRID=True
export FP8_DPA=False
export FP8_PARAM_GATHER=False
export FP8_RECIPE="tensorwise"

unset NVTE_DPA_FP8_FORMAT
unset NVTE_DPA_FP8DS_AMAX_ALGO
unset NVTE_DPA_FP8_RECIPE

export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export UB_TP_COMM_OVERLAP=0

export FULL_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_JIT=0

export CC=/usr/bin/gcc

# ===== STEP 2 OPTIMIZATION: NCCL NVLink SHARP =====
# Override config_common.sh's NCCL_NVLS_ENABLE=0. NVLS uses NVLink SHARP
# in-network reductions on H100 NVLink to accelerate small/medium all-reduce.
# Listed in run.sub --container-env so it propagates into the container.
export NCCL_NVLS_ENABLE=1
