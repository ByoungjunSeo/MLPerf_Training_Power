source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
# NOTE: do NOT source config_common_fp8attn{,_pcie}.sh — both export
# NVTE_DPA_FP8_FORMAT="" (empty) which crashes TransformerEngine import.

# ===== Parallelism =====
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

# ===== Step 1 settings (carried forward) =====
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

# Step 2 (NVLS) and Step 3 (CUDA Graph) carried forward as no-ops:
# Step 2 had no effect (no NVSwitch); Step 3 OOM'd on this 94GB H100.
export FULL_CUDA_GRAPH=0
export MCORE_CUDA_GRAPH=0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_JIT=0

export CC=/usr/bin/gcc

# ===== STEP 4 OPTIMIZATION: TP communication overlap (user buffers) =====
# conf/custom.yaml selects ub_tp_comm_overlap_cfg by:
#   ${GPU_ARCH,h100}tp${TENSOR_MODEL_PARALLEL}mbs${MICRO_BATCH_SIZE}
# → conf/tp_overlap/h100tp2mbs1.yaml (created for this Step).
# SEQ_PARALLEL must be True for AG/RS overlap to engage (otherwise the
# TP collective is a single all_reduce with no AG/RS to overlap).
export SEQ_PARALLEL=True
export TP_COMM_OVERLAP=True
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export UB_TP_COMM_OVERLAP=1
# CUDA Multicast (used by TE userbuffers by default) requires NVSwitch.
# This system has NVLink bridge (P2P) only — fall back to CUDA IPC.
# Job 190: 'CUDA device... does not support comm+GEMM overlap with CUDA Multicast.
#          Launch app with UB_SKIPMC=1 to try CUDA IPC instead.'
export UB_SKIPMC=1

# The container has its own baked-in /workspace/llm/conf snapshot; the host
# h100tp2mbs1.yaml and the `ub_tp_comm_overlap_cfg: null` declaration in
# custom.yaml are missing inside the image (job 187/188 OmegaConf struct
# error). Mount the host conf/ over the container's via the existing
# EXTRA_MOUNTS hook in config_mounts.sh.
export EXTRA_MOUNTS="$(readlink -f $(dirname ${BASH_SOURCE[0]}))/conf:/workspace/llm/conf:ro"
