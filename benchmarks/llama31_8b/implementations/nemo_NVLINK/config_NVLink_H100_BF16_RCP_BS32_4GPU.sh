# BS=32 RCP-eligible config for MLPerf 5.1 llama31_8b — 4-GPU variant (TP=2 x DP=2).
#
# Goal: keep the *exact* BS=32 RCP-eligible math of config_NVLink_H100_BF16_RCP_BS32.sh
# while halving wall-clock per run by adding a second NVLink pair as a DP replica.
# GBS / LR / warmup_samples are UNCHANGED, so the same training_5.1.0 BS=32 RCP applies
# (the RCP checker matches by global batch size, not by DP/grad_accum layout).
#
# Topology assumption (H100 NVL): NVLink Bridge is pairwise. 4 GPUs = TWO independent
# pairs. TP=2 MUST stay inside one pair (211 GB/s); the only cross-pair traffic is the
# DP gradient all-reduce (~16 GB, once per global step) over PCIe (~50 GB/s, ~0.3s).
#   -> VERIFY rank mapping so TP groups (ranks 0-1, 2-3) align with the physical pairs
#      (`nvidia-smi topo -m`, NCCL topology log). TP crossing a pair = PCIe bottleneck.
#
# Parallelism math (vs the 2-GPU RCP config):
#   2-GPU: GBS = MINIBS * (DGXNGPU/TP) = 32 * (2/2) = 32 ; DP=1 ; grad_accum = 32/1/1 = 32
#   4-GPU: DP = DGXNGPU/TP = 4/2 = 2
#          MINIBS = GBS/DP = 32/2 = 16          # per-DP-replica mini-batch
#          GBS = MINIBS * (DGXNGPU/TP) = 16 * (4/2) = 32   ✓ (unchanged)
#          grad_accum = GBS / MICRO_BATCH_SIZE / DP = 32/1/2 = 16
#   WARMUP_STEPS = warmup_samples / GBS = 16348/32 = 511 steps  (unchanged — GBS same)
#
# Expected (to be confirmed by 1-2 pre-validation runs BEFORE the submission window):
# - Global step time ~11s (vs ~21.4s at 2-GPU); scaling eff. 1.85-1.95x assumed.
# - Wall time ~22h/run -> 10 runs ~= 9.5 days (fits the ~2-week window with margin).
# - Convergence trajectory should MATCH the 2-GPU runs (GBS/LR/warmup identical):
#   samples-to-converge in the observed 208,896-221,184 band, eval_loss ~3.28-3.30.
# - Memory: per-GPU state unchanged from the 2-GPU run (~92 GB); DP adds grad/opt
#   comm buffers, not model shards -> expect similar headroom. Confirm no OOM on run 1.

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

export MINIBS=16                         # per-DP mini-batch; GBS = 16 * (4/2) = 32
export TENSOR_MODEL_PARALLEL=2
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

export TP_COMM_OVERLAP=False
export MICRO_BATCH_SIZE=1
export WARMUP_VALIDATION_STEPS=0

export LR=0.001                          # Reference: 1e-3 for BS=32 (unchanged)
export WARMUP_STEPS=511                  # = ceil(16348 / 32), GBS unchanged (unchanged)
export VAL_CHECK_INTERVAL=384

export DGXNNODES=1
export DGXNGPU=4                         # 4 GPUs on one node = two NVLink pairs
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# Pre-validation (job 218) measured 20.2h/run -> 25h cap fails fast in a chain
# (was 3600 = 60h, sized for the 2-GPU 42h runs)
export WALLTIME_RUNANDTIME=1500
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

# NUMA: bindpcie pins each rank to its GPU's local NUMA node (SNC-2: pair A -> NUMA 2
# cores 48-71, pair B -> NUMA 3 cores 72-95); --mem=node also forces local memory so
# dataloader workers (fork -> inherit binding) allocate on the local node.
# If the container's bindpcie rejects --mem=node the launch fails within the first
# minute -> fall back to "bindpcie --cpu=node" (first-touch then lands local anyway).
export BINDCMD="bindpcie --cpu=node --mem=node"
