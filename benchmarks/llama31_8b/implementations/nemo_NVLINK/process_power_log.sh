#!/bin/bash
# Post-process one run's IPMI power log with the OFFICIAL MLCommons tools only.
# usage: ./process_power_log.sh <SLURM_JOB_ID> [POWERLOGDIR]
# PSU is 80 PLUS Titanium -> conversion coefficient 0.94 (do not change).
set -euo pipefail
cd "$(dirname "$0")"

JOBID="${1:?usage: $0 <slurm-job-id> [powerlogdir]}"
POWERLOGDIR="${2:-${POWERLOGDIR:-/lustre/agent/mltraining_v5.1_llama3_8b/log/power}}"
IN="${POWERLOGDIR}/IPMIPower_${JOBID}.txt"
[ -f "${IN}" ] || { echo "power log not found: ${IN}" >&2; exit 1; }

# Per-job output folder: mllog appends to an existing file, so reusing one path
# mixes sessions and corrupts the energy metric (seen with the legacy Feb data).
OUTDIR="output/power_${JOBID}"
if [ -f "${OUTDIR}/node_0.txt" ]; then
  mv "${OUTDIR}/node_0.txt" "${OUTDIR}/node_0.txt.prev.$(date +%y%m%d%H%M%S)"
  echo "note: previous ${OUTDIR}/node_0.txt rotated aside (mllog would append to it)"
fi
mkdir -p "${OUTDIR}"

python3 ./mlperf-logging/mlperf_logging/mllog/examples/power/power_measurement.py \
  --power-log "${IN}" --output-folder "${OUTDIR}" --output-log node_0.txt --log-type IPMI \
  --start-with-readings --convertion-coef 0.94 >/dev/null

python3 ./mlperf-logging/mlperf_logging/mllog/examples/power/compute_metric_example.py \
  --input-log "./${OUTDIR}/node_0.txt"
