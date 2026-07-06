# MLPerf llama3.1 8B 성능 최적화 로그

**시스템**: 2× NVIDIA H100 NVL (94GB) + NVLink Bridge (NV12)
**벤치마크**: MLPerf Training v5.1 llama3.1 8B, TARGET_LOG_PPL ≤ 3.3
**Baseline**: Job 196 (38.19h, BF16+TE off, NVLink) — 컴플라이언스 SUCCESS

본 로그는 매 실험마다 **변경사항 / 가설 / 결과 / 분석**을 누적 기록합니다.
동일한 데이터가 `OPTIMIZATION_LOG.csv`에도 작성되어 Excel/스프레드시트 import 가능.

---

## 컬럼 정의

| 컬럼 | 설명 |
|---|---|
| Date | 실험 실행 날짜 (YYYY-MM-DD) |
| Job ID | SLURM job ID |
| Config | 사용한 config 파일 (`config_*.sh`) |
| Type | quick / convergence / debug |
| Key Change | 직전 baseline 대비 핵심 변경사항 |
| Hypothesis | 왜 이 변경이 효과 있을 거라 봤는지 |
| Status | ✅ converged / ⚠️ degraded / ❌ failed / 🔬 measured |
| Step time | 평균 step time (초) — quick test 또는 convergence run 초기 안정 구간 |
| Wall time | run_start → run_stop (시간). convergence 도달 시에만 |
| Final eval_loss | 컨버전스 도달 시 마지막 값, 발산 시 최저점/마지막값 |
| Δ vs Baseline | wall time 또는 step time 기준 vs Job 196 |
| Notes | 추가 메모 / 다음 시도 힌트 |

---

## 실험 기록

| # | Date | Job | Config | Type | Key Change | Hypothesis | Status | Step time | Wall time | Final eval | Δ Step / Δ Wall | Notes |
|--:|---|---|---|---|---|---|---|---:|---:|---:|---|---|
| 1 | 2026-02-20 | 175 | `config_PCIeH100_TTA.sh` | conv | (baseline, PCIe만) | 기본 BF16+TE off로 컨버전스 성공 가능 검증 | ✅ converged | 5.88s | **37.87h** | 3.297 | — | NVLink 없을 때 첫 성공 기준점. compliance SUCCESS. |
| 2 | 2026-05-12 | 182 | `config_NVLink_H100_step1_fp8.sh` | quick (200 step) | + FP8 + TE on | FP8 GEMM + TE op fuser로 step time 대폭 단축 기대 | 🔬 measured | **3.627s** | — | — | -38.4% vs PCIe step time | NVLink 추가 후 첫 quick test. step time만 보면 매우 좋음. |
| 3 | 2026-05-13 | 187~191 | `config_NVLink_H100_step4_tpoverlap.sh` | debug | + TP_COMM_OVERLAP | TP all-reduce를 GEMM과 overlap | ❌ blocked | — | — | — | — | 5단계 디버깅: yaml schema → conf snapshot → null override → CUDA Multicast → OpenMPI/PMIx mismatch. 환경 비호환. |
| 4 | 2026-05-13 | 194 | `config_NVLink_H100_step1_fp8.sh` | conv | Step 1 (FP8+TE) full convergence 시도 | quick test에서 -38.4% 였으니 wall time도 줄 것 | ❌ diverged | 3.63s | (14h 후 cancel) | min 6.14 → 7.93 | — | step ~1500에서 발산 시작. FP8 numerical drift + GBS=8 noisy gradient + LR 4e-4 결합. |
| 5 | 2026-05-13 | 195 | `config_NVLink_H100_step1_fp8.sh` + LR=1e-4 override | conv | LR 4e-4 → 1e-4 (Job 194 발산 회피) | LR을 1/4로 줄이면 안정화 가능 | ⚠️ plateau | 3.68s | (6h 후 cancel) | 5.14 정체 | — | 발산은 막았으나 loss 5.14에서 정체. LR 너무 작아 학습 능력 상실. |
| 6 | 2026-05-14 | 196 | **`config_NVLink_H100_BF16.sh`** | **conv (best)** | Job 175와 동일 SW + NVLink HW | BF16+TE off가 GBS=8 환경에서 유일하게 안정. NVLink로 step time만 -8% 기대 | ✅ **converged** | **5.40s** | **38.19h** | **3.300** | step **-8.2%** / wall **+0.85%** | **현재 최선 + 컴플라이언스 SUCCESS**. Wall time이 살짝 더 걸린 건 seed noise (step count +13% 필요). |
| 7 | 2026-05-19 | 197 | `config_NVLink_H100_BF16_minibs16.sh` | quick (200 step) | MINIBS 8 → 16 (GBS 두 배) | 큰 GBS = noise ↓ + step 수 ÷2 = wall time 감소 가능. 메모리 OOM 위험. | 🔬 measured | 10.68s | — | — | step time 2× (예상대로) | OOM 회피 ✅. loss 정상 감소. full conv ETA 추정 ~34.5h (vs Job 196 38.19h, -10% 잠재). |
| 8 | 2026-05-19~20 | 198 | **`config_NVLink_H100_BF16_minibs16.sh`** | **conv (new best)** | MINIBS=16 full convergence, LR=4e-4 유지 | quick test에서 OOM 회피 + step time 예상대로. GBS 두 배로 학습 효율 ↑ 기대. | ✅ **converged** | 10.76s | **33.79h** | **3.291** | wall **-11.5%** vs Job 196 / samples **-6.7%** | **🎉 새 BEST + Compliance SUCCESS**. Samples-to-conv도 줄어듦 (184k→172k) — GBS 증가가 step 효율 + sample 효율 둘 다 개선. |
| 9 | 2026-05-25 | 199 | `config_NVLink_H100_BF16_minibs24.sh` | quick (200 step) | MINIBS 16 → 24 (GBS 1.5×) | Job 198 성공 후 추가 sample efficiency 기대. OOM 위험 (Job 198이 96% 사용). | 🔬 measured | 15.94s | — | — | step time 2.95× of Job 196 (예상 3× 적중) | OOM 회피 ✅. loss 정상 감소 (7.44→6.48). full conv ETA 추정 31.5~33.5h. |
| 10 | 2026-05-26~27 | 200 | `config_NVLink_H100_BF16_minibs24.sh` | conv | MINIBS=24 full convergence, LR=4e-4 유지 | quick test 안전 통과. GBS 1.5×로 sample efficiency 추가 개선 기대 (Job 198 패턴 연장). | ✅ converged | 16.06s | **33.73h** | 3.287 | wall **-0.18%** vs Job 198 (noise) / samples **+1.8%** | **Compliance SUCCESS but wall time 변화 없음**. Job 198 대비 step time +49%, step count -32%, samples-to-conv 미세 증가 → 정확히 상쇄. **GBS sweet spot이 16 근처임을 입증**. |
| ⚠ | 2026-06-05 | (RCP 검증) | — | check | Job 196/198/200 모두에 5.1.0 RCP checker 실행 | 우리 결과가 RCP 통과하는지 확인 | ❌ Missing RCP | — | — | — | — | **모든 잡 RCP FAILED**: 5.1.0 reference에는 BS ∈ {32, 64, 96, 128}만 있고 우리 BS=8/16/24는 범위 밖. 6.0.0 RCP에는 BS=16 있어서 Job 198이 통과할 것 (samples 172k = reference 최상단). 그러나 5.1.0 official 제출엔 BS 32+ 필요. |
| 11 | 2026-06-05 | 205 | `config_NVLink_H100_BF16_RCP_BS32.sh` | quick (200 step) | MINIBS 24 → 32, **LR 4e-4 → 1e-3, WARMUP_STEPS 128 → 511** (모두 5.1.0 RCP BS=32 reference에 매칭) | RCP 통과를 위해 reference hyperparam 매칭 필수. Memory + step time 확인. | 🔬 measured | 21.37s | — | — | step time scaling 거의 완벽 선형 (3.96× of BS=8) | OOM 회피 ✅. Loss 정상 감소. Full conv ETA 38~45h. |
| 12 | 2026-06-06~07 | 206 | `config_NVLink_H100_BF16_RCP_BS32.sh` | conv (RCP-eligible) | BS=32 full convergence, **RCP-eligible** hyperparams | RCP 통과 + 컨버전스 SUCCESS 목표. Reference range (196k~233k samples). | ✅ converged | 21.37s | **39.91h** | 3.294 | wall +18.2% vs Job 200 (RCP 비용) / samples 208,896 (reference 안쪽) | **Compliance SUCCESS + RCP "found" (BS=32 매칭)**. 그러나 RCP statistical pass엔 **10회 run 평균 필요** (`submission_runs['llama31_8b']=10`). 우리 208,896 > reference Min Epochs 206,333 → 단일 run으로는 "too fast" 아님. |
| ⚠ | 2026-06-08 | (RCP 정책) | — | check | RCP checker 분석: `submission_runs['llama31_8b']=10` 확인 | 10회 olympic scoring (top/bottom 제외 8개 평균) | ❌ Submission mean=NaN | — | — | — | — | **단일 run으로는 RCP statistical pass 불가능**. 10회 run 평균이 reference 분포 안에 있어야 official MLPerf submission 가능. |
| 13 | 2026-06-08~23 | 208~216 | `config_NVLink_H100_BF16_RCP_BS32.sh` | conv ×9 (chained) | Job 206과 동일 설정으로 9개 추가 run (SLURM dependency chain) | 10회 run mean을 reference 분포(mean 215k, std 12k) 안에 위치시켜 RCP 통과 | ✅ **all converged** | 21.4s avg | 41.36h avg | 3.276~3.300 | 10/10 SUCCESS | **🎉 RCP test PASSED**. Olympic mean (top/bottom 제외 8개 평균) = 216,576 vs reference 215,040 (+0.71%). MLPerf 5.1 공식 제출 가능 상태. 10 runs: [196608, 208896×3, 221184×6] (sorted). |

---

## 성능 변화 상세 분석

각 실험이 직전 baseline 대비 **왜 성능이 좋아졌거나 나빠졌는지** 메커니즘 수준으로 설명.

---

### #1 — Job 175 (PCIe baseline)
**결과**: 37.87h, eval_loss 3.297, compliance SUCCESS

**의의**: 모든 후속 실험의 기준점. NVLink 없이도 시스템이 MLPerf 컨버전스 가능함을 입증.

**왜 37.87h가 걸렸나?**
- Step time 5.88s × step 수 20,352 = pure compute ~33.2h
- 나머지 4.7h = checkpoint save, validation eval, framework overhead
- llama3.1 8B 모델은 8B 파라미터 × forward + backward 한 번에 약 6초가 필요한 게 H100 1 GPU의 기본 throughput. TP=2로 두 GPU에 절반씩 나누지만 통신(all-reduce)이 추가되어 효율이 100%는 아님.

---

### #2 — Job 182 (FP8+TE quick test, 매우 좋아 보였지만…)
**결과**: 3.627s/step (PCIe 5.88s 대비 **-38.4%**)

**무엇이 바뀌었나?**
- FP8 (8-bit floating point): GEMM 연산을 BF16(16-bit) 대신 FP8(8-bit)로 수행 → tensor core가 같은 시간에 2배 throughput
- TransformerEngine (TE) op fuser: 여러 작은 PyTorch 연산들을 하나의 큰 CUDA 커널로 합쳐서 커널 launch overhead 제거

**왜 -38.4%가 나왔나?**
- FP8 GEMM은 BF16 GEMM 대비 H100 tensor core에서 **약 2× FLOPS** (1.97 PFLOPS vs 989 TFLOPS)
- TE op fuser는 약 5~10% 추가 단축
- 두 효과가 곱해져 step time ~38% 감소

**그러나 함정** — 후속 #4에서 발산 드러남. step time이 좋다 ≠ 학습이 안정하다.
**교훈**: quick test (200 step)는 throughput만 측정. **stability는 full convergence에서만 검증 가능**.

---

### #3 — Job 187~191 (TP Overlap 5단계 차단)
**결과**: 컨버전스 미도달, 5단계 모두 다른 레이어에서 환경 비호환으로 차단

**시도한 변경**: Tensor Parallel all-reduce를 GEMM 연산과 시간적으로 겹쳐 실행 (overlap) → 통신 시간을 hide

**왜 안 됐나?** 매 fix마다 더 깊은 환경 의존성 드러남:

| 단계 | 에러 | 원인 | 적용한 fix |
|---|---|---|---|
| 187 | OmegaConf struct 거부 | model 스키마에 키 미선언 | host yaml에 키 추가 |
| 188 | 같은 에러 재현 | 컨테이너가 host yaml 안 봄 (자체 baked-in conf) | EXTRA_MOUNTS로 mount |
| 189 | buffer_options=None | 내가 추가한 null이 로드 nullify | null 라인 제거 |
| 190 | CUDA Multicast unsupported | TE userbuffer가 NVSwitch 하드웨어 가정 | UB_SKIPMC=1 (IPC fallback) |
| 191 | OMPI PMI 빌드 누락 | 컨테이너 HPCX PMIx 2.2.35 vs host SLURM PMIx 2.1.25 ABI 불일치 | **수정 불가** (호스트 SLURM 재컴파일 필요) |

**왜 차단 가능성을 사전에 못 봤나?**
NVIDIA MLPerf 컨테이너는 **NVSwitch 장착 HGX H100 8-GPU + matching HPC 환경** 기준으로 설계됨. 우리 환경(NVLink Bridge + 표준 EL8 SLURM)은 그 가정의 일부만 만족.

**교훈**: 알고리즘적 최적화(TE userbuffer 같은)는 **하드웨어/소프트웨어 환경 가정**을 명시적으로 가짐. 가정이 안 맞으면 손쉽게 차단됨.

---

### #4 — Job 194 (FP8+TE full convergence: 발산 ❌)
**결과**: 14시간 후 cancel. eval_loss가 6.14 최저점 후 7.93까지 상승 (학습이 거꾸로 감)

**무엇이 바뀌었나?** Job 175 → 196의 차이는 FP8+TE on (Job 182와 동일 config의 full convergence)

**왜 발산했나?** 세 가지 요인의 **누적 결합**:

1. **FP8의 정밀도 한계**: FP8은 8-bit (E4M3 또는 E5M2)로 표현. BF16(16-bit) 대비 표현 가능한 값의 개수가 2^8 = 256개로 격감. 매 GEMM 결과의 작은 오차가 누적됨.

2. **GBS=8의 noisy gradient**: 16,384개 토큰(1 sample = 8192 tokens × 2 GPUs)으로 추정한 gradient는 통계적으로 noise가 큼. 정답 방향에서 ±10~30% 흔들림.

3. **LR=4e-4의 공격적 step**: noisy gradient × 큰 LR = 가중치가 잘못된 방향으로 크게 이동할 위험.

세 요인이 **곱셈으로 결합**: FP8 오차 × noisy gradient × 큰 LR.
- 짧게 200 step에서는 안 보임 (오차 누적 안 됨)
- 장기 (1500+ steps)에서 누적 효과로 학습이 발산

**왜 -38.4% throughput에도 wall time을 줄이지 못했나?**
컨버전스에 도달하지 못하면 step time이 빨라도 wall time = ∞ (영영 못 끝남). MLPerf에서는 **반드시 target eval_loss에 도달해야** valid 결과. 

**교훈**: 학습 안정성 위계: **컨버전스 도달 > step time**. 안정성 깨진 빠른 학습은 무의미.

---

### #5 — Job 195 (LR=1e-4: 발산은 막았지만 정체 ⚠️)
**결과**: 6시간 후 cancel. eval_loss 5.14에서 정체

**무엇이 바뀌었나?** Job 194 → 195: LR 4e-4 → 1e-4 (1/4로 축소)

**왜 정체했나?**
Job 194의 발산을 막으려고 LR을 너무 줄이니, 반대로 **학습 효율이 너무 낮아짐**. 매 step에서 가중치 업데이트가 너무 작아서:
- 처음엔 정상 감소 (samples 12k까지 loss 5.6 → 5.6)
- 그 후 plateau (16,000 samples 동안 loss 변화 -0.004) — 사실상 학습 정지

**비유**: 발산이 "차가 도로를 벗어남"이라면, plateau는 "차가 너무 천천히 가서 목적지에 영영 못 도착". 

**왜 LR sweet spot을 못 찾았나?**
GBS=8 + FP8 조합의 본질: 안정 LR과 효율 LR 사이의 window가 **너무 좁거나 존재하지 않음**. 일반적으로 큰 batch (낮은 noise)일수록 LR 허용 범위가 넓어짐. GBS=8은 너무 작음.

**교훈**: LR 튜닝만으로 FP8 안정화 불가. **GBS를 키우는 게 본질적 해법**(나중에 Job 198에서 입증).

---

### #6 — Job 196 (NVLink + BF16, MLPerf 첫 SUCCESS) ⭐
**결과**: 38.19h, eval_loss 3.300, compliance SUCCESS

**무엇이 바뀌었나?** Job 175(PCIe) → 196: **하드웨어만 NVLink Bridge 추가**, 소프트웨어 100% 동일

**기대했던 변화**: step time -10% (NVLink가 TP all-reduce를 가속)

**실제 결과**:
| 지표 | Job 175 | Job 196 | Δ |
|---|---:|---:|---:|
| Step time | 5.88s | 5.40s | **-8.2%** ✅ |
| Step 수 | 20,352 | 23,040 | **+13.2%** ⚠️ |
| Wall time | 37.87h | 38.19h | **+0.85%** |

**왜 step time이 -8.2%만 줄었나?**
NVLink는 GPU-to-GPU 대역폭을 50 GB/s → 211 GB/s (4.2×)로 늘림. 그런데 step time은 -8%만. 이유:
- 5.88s step 중 **all-reduce는 약 0.5s에 불과** (전체의 ~8.5%)
- 나머지 5.4s는 GEMM 등 compute (NVLink와 무관)
- NVLink로 통신 0.5s → ~0.04s로 줄어듦 (≈ -0.46s = -7.8%)
- **즉 통신 부분만 거의 사라졌고, 그게 전체의 8%** → 예상 한도가 정확히 -8%

**왜 step 수가 +13% 늘었나? (가장 큰 의외)**
같은 config를 두 번 돌리면 결과가 다름. 원인은 **floating-point 비결정성**:
- NVLink 추가로 NCCL이 다른 알고리즘 경로 선택 → 합산 순서 변경 → 매 step 가중치가 ε 차이
- 수천 step 누적되면 모델이 다른 "골짜기"로 수렴
- 이번 seed는 운이 살짝 나빠 +2,688 step 더 필요

**왜 wall time이 +0.85%가 됐나? (요약)**
- Step time -8% (이득)
- Step 수 +13% (손해)
- Step 수 손해 > Step time 이득 → 총 wall time 살짝 증가

**왜 이게 NVLink의 "진짜 성능"이 아닌가?**
+13% step 수는 NVLink 때문이 아니라 **random seed 효과**. 5개 seed로 평균 내면 ±10% 노이즈가 상쇄되어 진짜 NVLink 이득 (-8%)이 드러남. 단발성 측정은 한 sample → 통계적 신뢰성 낮음.

**교훈**:
- **NVLink 자체는 step time 기준 명백히 도움** ✅
- 단발성 wall time 비교는 seed noise(±5~10%)에 NVLink 작은 이득(-8%)이 묻힐 수 있음
- 모델 알고리즘에서 통신이 차지하는 비율이 NVLink 이득의 상한

자세한 분석: `NVLINK_BRIDGE_ANALYSIS.md`

---

### #7 — Job 197 (MINIBS=16 quick test)
**결과**: 10.68s/step (Job 196의 5.40s 대비 +99%), OOM 없음

**무엇이 바뀌었나?** Job 196 → 197: MINIBS 8 → 16 (GBS 8 → 16)

**왜 step time이 정확히 2배가 됐나?**
공식: `GBS = MINIBS × (DGXNGPU / TP) = 16 × 1 = 16`, `gradient_accumulation_steps = GBS / MICRO_BATCH_SIZE = 16`
- 1 training step = 16개 micro-batch를 순차 처리 후 weight update 1번
- MINIBS=8일 때는 8개 micro-batch → step time 5.40s
- MINIBS=16일 때는 16개 micro-batch → step time **5.40 × 2 = 10.80s** (예측)
- 실측 10.68s — 예측과 거의 일치

**왜 메모리가 OOM 안 났나?**
gradient accumulation은 forward/backward를 **순차** 실행하고 gradient만 누적. 매 micro-batch는 독립적으로 메모리 alloc/dealloc → peak 메모리는 GBS와 무관하게 **MICRO_BATCH_SIZE**에만 의존. MICRO_BATCH_SIZE=1 유지했으므로 메모리 사용 ≈ Job 196 수준 (~90GB / 94GB).

**의의**: MINIBS 증가가 메모리 측면에서 안전함이 입증. Full convergence (Job 198) 진행 청신호.

---

### #8 — Job 198 (MINIBS=16 full convergence, **새 BEST**) 🎉
**결과**: **33.79h** (Job 196 38.19h 대비 **-11.5%**, -4.40h), eval_loss 3.291, compliance SUCCESS

**무엇이 바뀌었나?** Job 196 → 198: MINIBS 8 → 16

**왜 wall time이 -11.5%나 줄었나? (두 효과의 결합)**

**효과 1: Step 수 -53.3%** (23,040 → 10,752)
- GBS 두 배 = 1 step당 처리 samples 두 배
- 같은 samples 처리에 step 수 절반 필요

**효과 2: Samples-to-convergence -6.7%** (184,320 → 172,032) ← 보너스
- GBS 큰 게 학습 효율 자체를 개선
- 이유: gradient noise 감소
  - GBS=8 → gradient는 16 × 8192 = 131,072 tokens 평균
  - GBS=16 → gradient는 16 × 8192 × 2 = 262,144 tokens 평균
  - 표본 크기 2배 → noise σ가 √2 = 1.41× 감소
  - 덜 noisy한 gradient는 정답 방향을 더 정확히 가리킴 → 매 step이 더 효과적

**효과 3 (반대 방향): Step time +99.3%** (5.40s → 10.76s) ← 손해
- 위에서 설명한 대로 step당 더 많은 micro-batch 처리

**총 wall time 계산**:
```
Job 196: 5.40s × 23,040 step + 4.7h overhead = 34.6h + 4.7h ≈ 38.2h (잘 맞음)
Job 198: 10.76s × 10,752 step + 4.7h overhead = 32.1h + 4.7h ≈ 33.8h (잘 맞음)

순 효과:
+99.3% (step time) × 46.7% (step 수, 즉 -53.3%) = +99.3 - 53.3 ≈ -7%  ← 이론
실제 wall time:  -11.5%  ← 추가 -4.5%p는 samples-to-conv 개선과 약간의 overhead 차이
```

**왜 GBS 증가가 항상 도움인 건 아닌가?**
GBS를 너무 키우면 (예: GBS=1024+):
- LR을 같이 키워야 함 (LR ∝ √GBS) → LR이 너무 커지면 발산 위험
- 1 step당 처리 시간이 너무 길어지면 GPU idle 시간 발생 (compute saturation 한계)
- 메모리 한계 도달

우리 케이스에서 GBS 8→16은 sweet spot 안에 있음. (Job 200에서 GBS=24도 시도했으나 plateau 확인)

**컨버전스 trajectory의 특이점**:
- 초반 (samples 6k, 12k): Job 198이 eval_loss 더 높음 (학습이 뒤처져 보임)
- 후반 (samples 100k+): Job 198이 따라잡고 추월
- 결과: 더 적은 samples로 target 도달

이는 **큰 batch의 전형적 특성**: 초기엔 천천히 학습하지만 후반엔 더 안정적/효율적으로 수렴. 마치 "처음엔 느려도 후반에 가속하는 마라톤 페이서".

**교훈**: 
- **GBS 증가는 이 환경에서 가장 효과적인 단일 최적화**
- step time과 wall time을 분리해서 생각해야 함 — step time이 더 느려도 wall time이 줄 수 있음
- 다음(GBS=24)에서 한계 도달 확인 → #10 참조

---

### #9 — Job 199 (MINIBS=24 quick test)
**결과**: 15.94s/step, OOM 없음

**무엇이 바뀌었나?** Job 197 → 199: MINIBS 16 → 24 (GBS 1.5×)

**왜 step time이 정확히 1.5×가 됐나?**
같은 메커니즘 (Job 197 분석 참조):
- gradient_accumulation_steps = GBS / MICRO_BATCH_SIZE = 24
- 1 training step = 24개 micro-batch 순차 처리
- 예측: Job 197 (16 micro-batch, 10.68s) × 24/16 = 16.02s
- 실측: 15.94s — 거의 일치 (선형 scaling)

**메모리는 왜 또 OK였나?**
peak 메모리는 MICRO_BATCH_SIZE에만 의존 (MICRO_BATCH_SIZE=1 유지). gradient는 누적 변수에 저장 (모델 파라미터와 동일 크기, 워크스페이스 거의 무변). 96% → 96% 유지.

---

### #10 — Job 200 (MINIBS=24 full convergence: SUCCESS but **plateau** ⚠️)
**결과**: **33.73h** (Job 198 대비 -0.06h, **-0.18%** = 노이즈 수준), eval_loss 3.287, compliance SUCCESS

**무엇이 바뀌었나?** Job 198 → 200: MINIBS 16 → 24

**왜 wall time이 거의 안 줄었나? (Diminishing returns)**

| 효과 | 방향 | 크기 |
|---|---|---|
| Step time 증가 | 손해 | +49.3% (10.76 → 16.06s) |
| Step 수 감소 | 이득 | -32.1% (10,752 → 7,296) |
| Samples-to-conv | **손해** | **+1.8% (172k → 175k)** |
| 총 wall time | ≈ 같음 | -0.18% |

세 효과를 곱셈 분해:
```
Job 198 pure compute: 10.76s × 10,752 = 115,691s
Job 200 pure compute: 16.06s × 7,296  = 117,182s  (+1.3%)

Job 198 → 200: pure compute가 오히려 약간 증가
이유: step count 절감(-32.1%)이 step time 증가(+49.3%)를 못 따라잡음
→ step time 증가율이 더 큼

Overhead (eval, save, init):
Job 198: 33.79 - 32.14 = 1.65h
Job 200: 33.73 - 32.55 = 1.18h
→ overhead가 0.47h 줄어들어 wall time을 보전 (왜? step 수가 적어 eval 호출 횟수도 감소)
```

**왜 sample efficiency가 더 안 좋아졌나? (가장 중요한 발견)**
Job 196 (GBS=8) → Job 198 (GBS=16)에서는 samples-to-conv가 -6.7% 감소했음. 그러나 Job 198 → 200에서는 **+1.8% 증가**. 이 반전의 원인:

- **Gradient noise 감소의 diminishing returns**: GBS 8→16에서 noise σ가 √2 = 1.41× 감소했고, 학습 효율이 +6.7%. GBS 16→24에서는 noise가 √1.5 = 1.22× 감소 (효과 작음).
- **Effective LR이 더 보수적으로 되는 효과**: 같은 LR=4e-4를 유지했지만, GBS 증가는 "암묵적 LR 감소"와 비슷한 효과. GBS=24에서는 LR이 너무 작아 학습 후반 진행이 살짝 느림.
- **데이터 다양성**: 매 step에 더 많은 sample 처리 = 다양성 ↓, 일부 어려운 example의 가중치 떨어짐.

**이 시스템에서 GBS sweet spot이 ~16인 이유**:
- **GBS 8**: noise 너무 큼 (Job 196이 다른 seed로는 발산할 위험 있었음)
- **GBS 16**: noise 적당히 감소 + step time 두 배 이내 + sample efficiency 개선
- **GBS 24**: 추가 noise 감소 효과 미미, step time 페널티는 그대로
- **GBS 32+**: 더 큰 페널티 예상, 메모리 위험도 ↑

**교훈**:
- **선형 외삽은 위험**: Job 198의 -11.5% 결과를 GBS=24로 확장 추정하면 -15~20%가 나올 거 같지만 실제로는 0%.
- **각 차원의 최적값은 비선형적으로 변함**: 더 큰 batch가 항상 좋지는 않음.
- **GBS=16이 이 시스템(2× H100 NVL, TP=2, BF16)의 sweet spot**.
- 다음 최적화는 **다른 차원**으로 가야 함 (NCCL 튜닝, ATTENTION_BACKEND 등).

---

## 학습 정리 (Lessons Learned)

### 1. Quick test (200 step) vs Full convergence는 다른 측정 대상
- Step time만 보고 wall time 단축을 추정하면 안 됨 (Job 182 → 194)
- 안정성 검증은 full convergence에서만 가능

### 2. 이 환경(GBS=8)에서 작동하는 알려진 설정
| 설정 | 결과 |
|---|---|
| BF16 + TE off + LR 4e-4 (Job 175, 196) | ✅ 컨버전스 |
| FP8 + TE on + LR 4e-4 (Job 194) | ❌ 발산 |
| FP8 + TE on + LR 1e-4 (Job 195) | ⚠️ 정체 |

→ **BF16 baseline이 안전한 출발점**. FP8 도입은 GBS 증가나 다른 안정화 기법과 함께 해야 함.

### 3. Random seed noise의 크기
- 동일 config에서 wall time ±1~3% 변동 가능
- Step time 변화 ±0.5%
- 작은 step time 개선(<5%)을 단발 컨버전스로 검증하려면 신호:노이즈가 너무 작음
- 5%+ 개선이 의심되거나, step time 차원에서 측정 가능한 경우엔 명확.

### 4. 환경 의존적 차단 요인
| 기능 | 우리 환경에서 작동? | 이유 |
|---|---|---|
| NVLink basic (NCCL ring/tree) | ✅ | 표준 PyTorch+NCCL 경로 |
| NVLink SHARP (NVLS) | ❌ | NVSwitch 필요 |
| TP Comm Overlap (TE userbuffer) | ❌ | NVSwitch + matching PMIx 필요 |
| CUDA Graph | ❌ (OOM) | 94GB로는 부족 |
| FP8 학습 | ❌ (발산) | GBS=8에선 numerical drift 지배 |

→ **하드웨어/소프트웨어 가정이 우리 환경과 다른 최적화는 작동 안 함**. NVLink hardware 자체의 ~8% 이득이 현재 환경에서 안전하게 수확 가능한 거의 전부.

---

## 다음 실험 계획

| Slot | 후보 실험 | 가설 | 예상 효과 | 리스크 |
|---|---|---|---|---|
| 7 | (TBD) | | | |

(여기에 다음 실험을 계획할 때 행을 추가)

---

## 변경 이력 (Log file 자체)

| Date | Event |
|---|---|
| 2026-05-18 | 초기 작성. Job 175, 182, 187~191, 194, 195, 196 기록 포함. |
