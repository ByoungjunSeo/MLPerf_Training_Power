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

# Step 2 (NCCL_NVLS_ENABLE=1) had no measurable effect on a 2-GPU
# NVLink-bridge config (no NVSwitch, NCCL silently falls back to ring).
# Leave at config_common.sh default (=0).

export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=False
export MC_TP_OVERLAP_RS=False
export UB_TP_COMM_OVERLAP=0

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is incompatible with
# CUDA Graphs — PyTorch hard-asserts at graph capture (job 185 crash).
# Unset to use the default caching allocator which supports graphs.
unset PYTORCH_CUDA_ALLOC_CONF

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_JIT=0

export CC=/usr/bin/gcc

# ===== STEP 3 OPTIMIZATION: Megatron-Core CUDA Graph =====
# conf/custom.yaml:
#   enable_cuda_graph:   ${oc.env:MCORE_CUDA_GRAPH,False}
#   cuda_graph_scope:    'full'           when FULL_CUDA_GRAPH=0
#                        'full_iteration' when FULL_CUDA_GRAPH=1
# 'full' captures each transformer block as a graph (per-layer). Safer
# starting point than 'full_iteration' with FP8/TE; if stable + helpful,
# Step 3b can try FULL_CUDA_GRAPH=1 for the whole step.
# Fixed shapes (MICRO_BATCH_SIZE=1, seqlen=8192) are CUDA-Graph-friendly.
export MCORE_CUDA_GRAPH=1
export FULL_CUDA_GRAPH=0
