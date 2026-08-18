#!/bin/bash
# IPMI power monitor — launched by run.sub as:
#   srun --overlap --ntasks-per-node=1 sudo -n numactl --cpunodebind=0 --membind=0 \
#       bash $POWERCMDDIR/power_monitor.sh "${POWERLOGDIR}" "${SLURM_JOB_ID}"
# Runs whole-script as root (single sudo — no per-iteration sudo overhead).
# sudoers env_reset strips POWERLOGDIR/SLURM_JOB_ID, so they are passed as args.
# Output line format MUST NOT change — parsed by mlperf-logging power_measurement.py.
OUTDIR="${1:-.}"; JOBID="${2:-manual}"
[ -d "$OUTDIR" ] || OUTDIR="."
OUT="${OUTDIR}/IPMIPower_${JOBID}.txt"
while true; do
    output=$(ipmitool dcmi power reading)
    power=$(echo "$output" | grep "Instantaneous" | awk '{print $4}')
    timestamp=$(echo "$output" | grep "IPMI timestamp" | cut -d':' -f2- | xargs)
    if [ -n "$power" ]; then
        echo "Instantaneous power reading:                   ${power} Watts     IPMI timestamp:                           ${timestamp}"
    fi
    sleep 1
done >> "$OUT"
