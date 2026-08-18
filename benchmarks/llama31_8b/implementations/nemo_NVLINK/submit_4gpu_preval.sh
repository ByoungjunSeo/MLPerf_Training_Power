#!/bin/bash
# 4-GPU PRE-VALIDATION submission (single run, NCCL_DEBUG=INFO for TP-group check).
# Run this yourself:  ./submit_4gpu_preval.sh
# After completion:   ./validate_4gpu_prerun.sh <slurm-out-file> <jobid>
set -euo pipefail
cd "$(dirname "$0")"

export DATADIR=/lustre/agent/mltraining_v5.1_llama3_8b
export LOGDIR=${DATADIR}/log
export POWERCMDDIR=$PWD
export POWERLOGDIR=${LOGDIR}/power
export CONT=sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt
export NEXP=1

# Pre-validation only: verify TP groups [0,1],[2,3] and per-rank CPU affinity
# ("Setting affinity for GPU n to 48-71 / 72-95" lines). pyxis inherits job env
# into the container; if no "NCCL INFO" lines appear in the log, append
# NCCL_DEBUG to run.sub's --container-env list.
export NCCL_DEBUG=INFO

source config_NVLink_H100_BF16_RCP_BS32_4GPU.sh

echo "GBS check: MINIBS=${MINIBS} x (DGXNGPU=${DGXNGPU} / TP=${TENSOR_MODEL_PARALLEL}) = $((MINIBS * DGXNGPU / TENSOR_MODEL_PARALLEL)) (must be 32)"
echo
echo "sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub"
JID=$(sbatch --parsable -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
echo "submitted: job ${JID}"
echo
echo "next steps after the run finishes:"
echo "  ./process_power_log.sh ${JID}"
echo "  ./validate_4gpu_prerun.sh slurm-${JID}.out ${JID}"
