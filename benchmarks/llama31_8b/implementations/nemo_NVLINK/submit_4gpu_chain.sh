#!/bin/bash
# 4-GPU official 10-run chain (afterany dependency, like jobs 206-216).
# NCCL_DEBUG is OFF here. Run this yourself:  ./submit_4gpu_chain.sh
# Optional: NRUNS=<n> ./submit_4gpu_chain.sh   (default 10)
set -euo pipefail
cd "$(dirname "$0")"

export DATADIR=/lustre/agent/mltraining_v5.1_llama3_8b
export LOGDIR=${DATADIR}/log
export POWERCMDDIR=$PWD
export POWERLOGDIR=${LOGDIR}/power
export CONT=sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt
export NEXP=1
unset NCCL_DEBUG   # debug logging off for the official chain

source config_NVLink_H100_BF16_RCP_BS32_4GPU.sh

NRUNS=${NRUNS:-10}
dep=""
for i in $(seq 1 "${NRUNS}"); do
  if [ -z "${dep}" ]; then
    jid=$(sbatch --parsable -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
  else
    jid=$(sbatch --parsable --dependency=afterany:"${dep}" -N "${DGXNNODES}" --time="${WALLTIME}" run.sub)
  fi
  echo "run #${i} -> job ${jid}${dep:+ (afterany:${dep})}"
  dep=${jid}
done
echo
echo "power logs will be at: ${POWERLOGDIR}/IPMIPower_<jobid>.txt (one file per run)"
