---
name: feedback-optimization-log
description: Performance optimization experiments must be logged to OPTIMIZATION_LOG.md + .csv after every run
metadata:
  type: feedback
---

For every MLPerf llama3.1 8B performance experiment (config change + job submission), append a row to both `OPTIMIZATION_LOG.md` and `OPTIMIZATION_LOG.csv` in `/mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo_NVLINK/`.

**Why:** User explicitly requested ongoing optimization tracking ("이제부터 성능 최적화 작업을 진행해야해. 그 기록들을 ... 그 이유를 여기에 기록을 남겨줘"). Without a structured log, the rationale behind each change gets lost across conversation compactions and weeks-long convergence runs.

**How to apply:**
- Format: Markdown table (human-readable) + CSV (Excel-importable). Both files always kept in sync.
- Columns: id, date, job_id, config, type, key_change, hypothesis, status, step_time_s, wall_time_h, final_eval_loss, delta_step_pct, delta_wall_pct, notes.
- `Hypothesis` field is mandatory — record the reasoning *before* running, even if it turns out wrong (Job 182's wrong conclusion that FP8 quick-test wins translate to wall-time wins is the cautionary example).
- `Δ vs Baseline` compares against Job 196 (current best valid convergence: 38.19h, BF16+TE-off, NVLink).
- For failed runs, still log them — failure modes are as valuable as successes (Jobs 187-195 form the "what doesn't work in this env" map).
- After cancelling a long run early (>1h elapsed), still log the partial result + reason for cancel.

Related: [[nvlink-bridge-analysis-doc]]
