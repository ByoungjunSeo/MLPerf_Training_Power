#!/bin/bash
# 4-GPU pre-validation pass/fail report (run AFTER the pre-validation job finishes).
# usage: ./validate_4gpu_prerun.sh <slurm-out-file> <jobid> [powerlogdir]
#
# Checks (agreed criteria):
#  1. step time      : median ~11s  (PASS <=11.5s / WARN <=13s / FAIL >13s -> TP likely on PCIe)
#  2. convergence    : last eval samples_count in 208,896..221,184 (2-GPU band);
#                      WARN if only inside the reference band 196,608..233,472
#  3. loss trajectory: first 30 train losses inside the envelope of Jobs 206-216 (+/-0.05)
#  4. NCCL           : TP comms = {GPU0,1} and {GPU2,3}; rank affinity 48-71 / 72-95
#  5. power          : sampling gap stats; avg/peak W in the RUNANDTIME window;
#                      baseline: 2-GPU run = 294.7 MJ over 41.8h (avg ~1.96 kW).
#                      Official energy metric: run ./process_power_log.sh <jobid>
set -uo pipefail
cd "$(dirname "$0")"

RUNLOG="${1:?usage: $0 <slurm-out-file> <jobid> [powerlogdir]}"
JOBID="${2:?usage: $0 <slurm-out-file> <jobid> [powerlogdir]}"
POWERLOGDIR="${3:-${POWERLOGDIR:-/lustre/agent/mltraining_v5.1_llama3_8b/log/power}}"
PWRLOG="${POWERLOGDIR}/IPMIPower_${JOBID}.txt"

python3 - "$RUNLOG" "$PWRLOG" <<'PYEOF'
import json, re, sys, glob
from datetime import datetime

runlog_path, pwrlog_path = sys.argv[1], sys.argv[2]
verdicts = []  # (level, section, message)

def add(level, section, msg):
    verdicts.append((level, section, msg))
    print(f"[{level}] {section}: {msg}")

def mllog_events(path):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            i = line.find(":::MLLOG")
            if i < 0:
                continue
            j = line.find("{", i)
            if j < 0:
                continue
            try:
                out.append(json.loads(line[j:line.rindex("}") + 1]))
            except Exception:
                pass
    return out

print(f"=== 4-GPU pre-validation report: {runlog_path} ===\n")
events = mllog_events(runlog_path)
raw = open(runlog_path, errors="replace").read()

# ---------- 1. step time ----------
steps = [e["value"]["train_step_time"] for e in events
         if e.get("key") == "tracked_stats" and "train_step_time" in e.get("value", {})]
if len(steps) < 10:
    add("FAIL", "step-time", f"tracked_stats train_step_time lines: {len(steps)} (<10) — run did not progress")
else:
    body = sorted(steps[8:])                      # skip first 8 (warmup/compile)
    med = body[len(body) // 2]
    mean = sum(body) / len(body)
    lvl = "PASS" if med <= 11.5 else ("WARN" if med <= 13.0 else "FAIL")
    note = "" if lvl == "PASS" else "  -> >13s suggests TP leaked onto PCIe: recheck rank mapping" if lvl == "FAIL" else "  -> above 11.5s: check DP overlap / affinity"
    add(lvl, "step-time", f"median {med:.2f}s, mean {mean:.2f}s over {len(body)} steps (2-GPU baseline 21.37s){note}")

# ---------- 2. convergence samples ----------
evals = [(e["metadata"].get("samples_count"), e["value"]) for e in events if e.get("key") == "eval_accuracy"]
run_stop = [e for e in events if e.get("key") == "run_stop"]
status = run_stop[-1]["metadata"].get("status") if run_stop else None
if not evals:
    add("FAIL", "convergence", "no eval_accuracy events found")
elif status != "success":
    add("FAIL", "convergence", f"run_stop status={status!r} (expected 'success')")
else:
    samples, last_loss = evals[-1]
    if 208_896 <= samples <= 221_184:
        add("PASS", "convergence", f"{samples:,} samples (2-GPU band 208,896-221,184), final eval_loss {last_loss:.4f}")
    elif 196_608 <= samples <= 233_472:
        add("WARN", "convergence", f"{samples:,} samples — outside 2-GPU band but inside reference band 196,608-233,472")
    else:
        add("FAIL", "convergence", f"{samples:,} samples — outside reference band 196,608-233,472")

# ---------- 3. loss trajectory vs Jobs 206-216 ----------
K = 30
losses = [e["value"]["reduced_train_loss"] for e in events
          if e.get("key") == "tracked_stats" and "reduced_train_loss" in e.get("value", {})][:K]
ref_jobs = [f"slurm-{j}.out" for j in (206, 208, 209, 210, 211, 212, 213, 214, 215, 216)]
ref_runs = []
for rf in ref_jobs:
    if glob.glob(rf):
        r = [e["value"]["reduced_train_loss"] for e in mllog_events(rf)
             if e.get("key") == "tracked_stats" and "reduced_train_loss" in e.get("value", {})][:K]
        if len(r) >= K:
            ref_runs.append(r)
if len(losses) < K:
    add("WARN", "loss-traj", f"only {len(losses)} train-loss points (<{K}) — skipping trajectory check")
elif len(ref_runs) < 3:
    add("WARN", "loss-traj", f"only {len(ref_runs)} reference logs readable — skipping trajectory check")
else:
    TOL = 0.05
    out_idx = [i for i in range(K)
               if not (min(r[i] for r in ref_runs) - TOL <= losses[i] <= max(r[i] for r in ref_runs) + TOL)]
    if len(out_idx) <= K // 10:
        add("PASS", "loss-traj", f"{K - len(out_idx)}/{K} first steps inside Jobs 206-216 envelope (+/-{TOL}) [{len(ref_runs)} refs]")
    else:
        add("FAIL", "loss-traj", f"{len(out_idx)}/{K} steps OUTSIDE envelope (idx {out_idx[:6]}...) — hyperparameter/dataloader mismatch?")

# ---------- 4. NCCL: TP groups + affinity ----------
if "NCCL INFO" not in raw:
    add("WARN", "nccl", "no 'NCCL INFO' lines — NCCL_DEBUG not propagated? add NCCL_DEBUG to run.sub --container-env")
else:
    # comm membership from "Init COMPLETE" lines, grouped by commId (comm pointers
    # are per-process addresses and CANNOT be used to group across ranks)
    comms = {}
    for m in re.finditer(r"nranks (\d+) cudaDev \d+ nvmlDev (\d+) busId [0-9a-f]+ commId (0x[0-9a-f]+)", raw):
        nranks, dev, cid = int(m.group(1)), int(m.group(2)), m.group(3)
        comms.setdefault((cid, nranks), set()).add(dev)
    pairs2 = sorted(set(tuple(sorted(v)) for (cid, n), v in comms.items() if n == 2 and len(v) == 2))
    has_tp = (0, 1) in pairs2 and (2, 3) in pairs2
    dp = [p for p in pairs2 if p in ((0, 2), (1, 3))]
    add("PASS" if has_tp else "FAIL", "nccl-tp",
        f"nranks=2 comm groups: {pairs2} " + (f"(TP {{0,1}},{{2,3}} confirmed; DP groups {dp})" if has_tp else "— TP groups NOT {0,1},{2,3}: fix rank mapping"))
    # affinity: this NCCL prints cpu RANGE LISTS like "72-95,168-191" (168-191 = HT
    # siblings of 72-95). NCCL logs the line only for some ranks — missing GPUs are
    # not an error (step time + comm groups are the real arbiters).
    def cpulist(s):
        out = set()
        for part in s.split(","):
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
        return out
    aff_bad, seen_gpus = [], set()
    for m in re.finditer(r"Setting affinity for GPU (\d+) to ([0-9,\-]+)", raw):
        gpu = int(m.group(1)); seen_gpus.add(gpu)
        cpus = cpulist(m.group(2))
        expect = set(range(48, 72)) | set(range(144, 168)) if gpu in (0, 1) else set(range(72, 96)) | set(range(168, 192))
        if cpus and len(cpus & expect) / len(cpus) < 0.9:
            aff_bad.append((gpu, min(cpus), max(cpus)))
    if not seen_gpus:
        add("WARN", "nccl-affinity", "no 'Setting affinity' lines found — verify binding via bindpcie output")
    elif aff_bad:
        add("FAIL", "nccl-affinity", f"unexpected CPU affinity {aff_bad} (want GPU0,1->48-71, GPU2,3->72-95 [+HT siblings])")
    else:
        miss = sorted({0, 1, 2, 3} - seen_gpus)
        add("PASS", "nccl-affinity",
            f"affinity matches NUMA plan for GPUs {sorted(seen_gpus)}" + (f" (GPUs {miss} not logged by NCCL — normal)" if miss else ""))

# ---------- 5. power ----------
try:
    ts_list, watts = [], []
    with open(pwrlog_path, errors="replace") as f:
        for line in f:
            m = re.match(r"Instantaneous power reading:\s+(\d+(?:\.\d+)?) Watts\s+IPMI timestamp:\s+(.+)$", line.strip())
            if m:
                watts.append(float(m.group(1)))
                ts_list.append(datetime.strptime(m.group(2).strip(), "%a %b %d %H:%M:%S %Y").timestamp())
    if not ts_list:
        raise ValueError("no parsable power lines")
    gaps = [b - a for a, b in zip(ts_list, ts_list[1:])]
    big = [g for g in gaps if g > 2.0]
    lvl = "PASS" if (not big or (len(big) / len(gaps) < 0.01 and max(big) < 10)) else "WARN"
    add(lvl, "power-sampling",
        f"{len(ts_list):,} samples, {len(big)} gaps >2s ({100 * len(big) / max(1, len(gaps)):.2f}%), max gap {max(gaps):.0f}s (target ~1s interval)")
    mm = re.search(r"RUNANDTIME_START (\d+)", raw); nn = re.search(r"RUNANDTIME_STOP (\d+)", raw)
    if mm and nn:
        t0, t1 = int(mm.group(1)), int(nn.group(1))
        w = [p for t, p in zip(ts_list, watts) if t0 <= t <= t1]
        if w:
            add("INFO", "power-train-window",
                f"avg {sum(w) / len(w):.0f} W, peak {max(w):.0f} W over {(t1 - t0) / 3600:.1f}h "
                f"(2-GPU baseline: avg ~1960 W, 41.8h, 294.7 MJ) — BMC/host clock offset caveat if window looks empty/shifted")
        else:
            add("WARN", "power-train-window", "no samples inside RUNANDTIME window — BMC clock offset? compare timestamps manually")
    add("INFO", "power-energy", f"official metric: ./process_power_log.sh <jobid>  (conversion coef 0.94, Titanium PSU)")
except Exception as ex:
    add("WARN", "power", f"power log not usable ({pwrlog_path}): {ex}")

# ---------- summary ----------
print("\n=== SUMMARY ===")
order = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}
for lvl, sec, msg in sorted(verdicts, key=lambda v: order[v[0]]):
    print(f"  [{lvl}] {sec}")
fails = sum(1 for v in verdicts if v[0] == "FAIL")
print(f"\nRESULT: {'FAIL (' + str(fails) + ' failing checks)' if fails else 'PASS — proceed to ./submit_4gpu_chain.sh'}")
sys.exit(1 if fails else 0)
PYEOF
