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

# ===== STEP 1 OPTIMIZATION: Re-enable FP8 GEMM + TransformerEngine =====
# NVLink bridge is now present (NV12, ~211 GB/s busbw measured),
# so the PCIe-conservative overrides are removed.

# TransformerEngine ON
export TRANSFORMER_ENGINE=True
export USE_TE_OPS=True
export CE_FUSION_IMPL=te

# FP8 ON for GEMMs only. FP8 attention is disabled because the container's
# cuDNN 9.13.1 + TE 25.09 combo has no available FP8 attention backend for
# our (GQA, seqlen=8192, TP=2) shapes (verified empirically with job 178).
export FP8=True
export FP8_HYBRID=True
export FP8_DPA=False
export FP8_PARAM_GATHER=False
export FP8_RECIPE="tensorwise"

# Leave NVTE_DPA_FP8_* env vars UNSET (TE uses safe defaults).
unset NVTE_DPA_FP8_FORMAT
unset NVTE_DPA_FP8DS_AMAX_ALGO
unset NVTE_DPA_FP8_RECIPE

# TP overlap stays OFF in Step 1 (no tp2 yaml exists; covered in Step 4).
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export UB_TP_COMM_OVERLAP=0

# CUDA Graph OFF in Step 1 (covered in Step 3).
export FULL_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# TorchDynamo / Inductor OFF (defense-in-depth; not the root cause of job 181).
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_JIT=0

# Triton JIT compiles small CUDA driver utilities via the C compiler. By default
# it falls back to a baked-in conda path (/home/mlcommons/envs/adabench_dl/bin/
# x86_64-conda-linux-gnu-cc) that does not exist on this node. With
# CE_FUSION_IMPL=te (TE's parallel_cross_entropy), Triton is invoked at the
# first cross-entropy call and the job dies. Point CC at the container's gcc-13.
export CC=/usr/bin/gcc
