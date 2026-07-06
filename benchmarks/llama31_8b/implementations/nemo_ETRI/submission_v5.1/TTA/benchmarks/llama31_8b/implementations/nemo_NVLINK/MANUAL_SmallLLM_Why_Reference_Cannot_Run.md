# Small LLM 레퍼런스: 직접 실행이 불가능한 이유와 수정 가이드

> `small_llm_pretraining` 레퍼런스는 우리와 **같은 8B 모델, 같은 목표(log_ppl ≤ 3.3)**를 사용하지만,
> 우리 환경(1노드, 2×PCIe H100 NVL)에서 직접 실행할 수 없습니다.
> 이 문서는 **왜 안 되는지**, 그리고 **정확히 어떤 파일의 어떤 부분을 수정하면 돌릴 수 있는지**를 설명합니다.

---

## 목차

**Part 1: 왜 직접 돌릴 수 없는가**

1. [한 문장 요약](#1-한-문장-요약)
2. [우리 환경 vs 레퍼런스 비교표](#2-우리-환경-vs-레퍼런스-비교표)
3. [원인 1: GPU 8개 전제 — 2개로는 병렬화 불가](#3-원인-1-gpu-8개-전제--2개로는-병렬화-불가)
4. [원인 2: NeMo-Run 실행 프레임워크 차이](#4-원인-2-nemo-run-실행-프레임워크-차이)
5. [원인 3: FP8 하이브리드 — 수렴 실패](#5-원인-3-fp8-하이브리드--수렴-실패)
6. [원인 4: GBS=32 — 2 GPU에서 비효율](#6-원인-4-gbs32--2-gpu에서-비효율)
7. [원인 5: 컨테이너 이미지 불일치 (Dockerfile)](#7-원인-5-컨테이너-이미지-불일치-dockerfile)
8. [원인 6: NVLink 전제 통신 설정](#8-원인-6-nvlink-전제-통신-설정)
9. [원인 7: MLPerf 로깅 플랫폼 하드코딩](#9-원인-7-mlperf-로깅-플랫폼-하드코딩)
10. [원인 8: 데이터 워커 128개](#10-원인-8-데이터-워커-128개)

**Part 2: 수정하면 돌릴 수 있는가? — 파일별 상세 수정 가이드**

11. [전체 수정 로드맵](#11-전체-수정-로드맵)
12. [수정 1: Dockerfile — 컨테이너 이미지 빌드](#12-수정-1-dockerfile--컨테이너-이미지-빌드)
13. [수정 2: config 파일 — 새 설정 파일 생성](#13-수정-2-config-파일--새-설정-파일-생성)
14. [수정 3: pretrain_llama31.py — 핵심 학습 스크립트](#14-수정-3-pretrain_llama31py--핵심-학습-스크립트)
15. [수정 4: callbacks.py — 로깅 및 콜백](#15-수정-4-callbackspy--로깅-및-콜백)
16. [수정 5: run_llama31.sh — 실행 스크립트](#16-수정-5-run_llama31sh--실행-스크립트)
17. [수정 6: SLURM 제출 스크립트 — 새로 생성](#17-수정-6-slurm-제출-스크립트--새로-생성)
18. [수정 요약 체크리스트](#18-수정-요약-체크리스트)

**Part 3: 종합 비교**

19. [large_llm vs small_llm: 불가능의 수준 비교](#19-large_llm-vs-small_llm-불가능의-수준-비교)
20. [최종 판정](#20-최종-판정)

---

# Part 1: 왜 직접 돌릴 수 없는가

## 1. 한 문장 요약

> **같은 8B 모델이지만, 레퍼런스는 "8개 GPU + NVLink + FP8 + NeMo-Run + 전용 컨테이너"를 전제로 만들어져 있어서,
> "2개 GPU + PCIe + BF16 + SLURM 직접 실행 + Dell 컨테이너"인 우리 환경에서는 그대로 돌릴 수 없습니다.**

```
비유하자면:

large_llm 레퍼런스 → "100톤 화물을 5톤 트럭에 싣는 것" (물리적으로 불가능)
small_llm 레퍼런스 → "같은 화물이지만, 트럭 모델이 달라 적재함 규격이 안 맞는 것" (호환성 불가)

→ 적재함(Dockerfile, config, 스크립트)을 우리 트럭에 맞게 개조하면 돌릴 수 있음
→ 단, 개조할 곳이 6개 파일, 수십 곳에 달함
```

---

## 2. 우리 환경 vs 레퍼런스 비교표

```
                   우리 환경                      레퍼런스 (H100 설정)        일치?
                   ─────────                      ────────────────────        ─────
서버 수            1대                            1대                         ✅
GPU 수             2개                            8개                         ❌
GPU 종류           PCIe H100 NVL                  DGX H100 (SXM)              ❌
GPU 메모리         93GB                           80GB                        ✅ (더 많음)
GPU 연결           PCIe Gen5 (128GB/s)            NVLink (900GB/s)            ❌
모델               LLaMA 3.1 8B                   LLaMA 3.1 8B               ✅
목표               log_ppl ≤ 3.3                 log_ppl ≤ 3.3              ✅
GBS                8                              32                          ❌
TP                 2                              4                           ❌
정밀도             BF16 (TE 비활성)               BF16 + FP8 Hybrid           ❌
실행 방식          SLURM + Pyxis/Enroot           NeMo-Run (torchrun)         ❌
컨테이너           sbj8388/...:training5.1        nvcr.io/...:25.01-py3 기반  ❌
NeMo 버전          25.09-alpha                    v2.1.0                      ❌
NeMo-Run 버전      v0.5.0                         v0.4.0                      ❌
Megatron-Core      25.09-alpha                    core_r0.11.0                ❌
```

```
일치: 4개 (서버 수, GPU 메모리, 모델, 목표)
불일치: 11개 (나머지 전부)
```

---

## 3. 원인 1: GPU 8개 전제 — 2개로는 병렬화 불가

### 무엇이 문제인가?

레퍼런스의 모든 하드웨어 설정이 **8개 GPU**를 전제로 합니다.

### 코드에서의 증거

**`config_H100_1x8x4_8b.sh`:**
```bash
GPUS_PER_NODE=8           # ← 8 GPU 고정
TENSOR_PARALLEL_SIZE=4    # ← TP=4 (최소 4 GPU 필요!)
GBS=32                    # ← 8 GPU 기준 배치 크기
# DP = 8 / 4 = 2          ← 데이터 병렬 2 (총 8 GPU)
```

**`pretrain_llama31.py` 105행 (SLURM 실행기):**
```python
gres="gpu:8",             # ← 8 GPU 하드코딩
```

### 그림으로 이해하기

```
레퍼런스 H100 설정: TP=4, DP=2 → 총 8 GPU

  GPU0  GPU1  GPU2  GPU3  ←── TP 그룹 1 (행렬을 4등분해서 분산 계산)
  GPU4  GPU5  GPU6  GPU7  ←── TP 그룹 2 (같은 행렬을 4등분, 다른 데이터)
  ↑                    ↑
  └── DP 그룹 ─────────┘     (같은 모델 복제, 다른 데이터로 학습)


우리 환경: GPU 2개만 있음

  GPU0  GPU1  ←── TP=2까지만 가능 (행렬을 2등분)
                  DP=1 (복제 불가, GPU가 없음)

  → TP=4는 물리적으로 불가능 (GPU가 4개 있어야 함)
  → TP=2, DP=1로 변경해야 함
```

---

## 4. 원인 2: NeMo-Run 실행 프레임워크 차이

### 무엇이 문제인가?

레퍼런스의 학습 스크립트(`pretrain_llama31.py`)는 **NeMo-Run**이라는 실험 관리 프레임워크에 의존합니다.
우리 Dell 컨테이너에도 NeMo-Run이 설치되어 있지만 **버전이 다릅니다** (v0.5.0 vs v0.4.0).

### 실행 방식 비교

```
레퍼런스의 실행 흐름:
  1. 사용자가 "source config.sh && bash run_llama31.sh" 실행
  2. run_llama31.sh가 환경변수를 정리해서 python3 pretrain_llama31.py 호출
  3. pretrain_llama31.py 안에서 NeMo-Run이:
     a) NeMo의 llama3_8b 레시피를 가져옴
     b) 하이퍼파라미터를 설정
     c) LocalExecutor 또는 SlurmExecutor를 생성
     d) torchrun으로 실제 학습을 시작

Dell (우리)의 실행 흐름:
  1. sbatch run.sub 으로 SLURM 작업 제출
  2. run.sub이 Pyxis로 컨테이너를 생성하고 srun으로 run_and_time.sh 실행
  3. run_and_time.sh가 각 GPU에서 python pretrain.py 호출
  4. pretrain.py가 Hydra/OmegaConf 설정으로 학습 시작

→ 완전히 다른 실행 체계!
→ 레퍼런스의 pretrain_llama31.py를 Dell의 run.sub 안에서 돌리려면 추가 작업 필요
```

### NeMo-Run 버전 호환성 문제

```
레퍼런스가 사용하는 NeMo-Run API (v0.4.0):

  import nemo_run as run
  run.Config(...)               # 설정 객체 생성
  run.Partial(...)              # 부분 함수 설정
  run.LocalExecutor()           # 로컬 실행기
  run.SlurmExecutor()           # SLURM 원격 실행기
  run.Experiment("name")        # 실험 관리
  exp.add(task, executor=...)   # 실험에 작업 추가
  exp.run(sequential=True)      # 실험 실행

우리 Dell 컨테이너의 NeMo-Run (v0.5.0):
  → API가 바뀌었을 수 있음
  → 확인 결과 v0.5.0이 설치되어 있어 기본 호환은 될 가능성 있음
  → 하지만 NeMo v2.1.0 API(llm.llama3_8b.pretrain_recipe() 등)가
    우리 컨테이너의 NeMo 25.09-alpha에 존재하는지는 별개 문제
```

---

## 5. 원인 3: FP8 하이브리드 — 수렴 실패

### 무엇이 문제인가?

레퍼런스는 **FP8 하이브리드 정밀도**를 기본으로 사용합니다.
우리 환경에서 FP8(및 Transformer Engine)을 활성화하면 **학습이 수렴하지 않습니다.**

### 코드에서의 증거

**`pretrain_llama31.py` 160~172행:**
```python
precision = run.Config(
    nl.MegatronMixedPrecision,
    precision="bf16-mixed",
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_enabled=True,
    grad_reduce_in_fp32=False,
    fp8="hybrid",                    # ← FP8 하이브리드 활성화!
    fp8_amax_history_len=4,
    fp8_amax_compute_algo='most_recent',
    fp8_params=True,                 # ← FP8로 파라미터 저장!
    fp8_dot_product_attention=False,
)
```

### 우리 환경에서의 실험 결과

```
Job 168: TE=True, FP8 관련 기능 활성화 → log_ppl 4.66에서 정체 (수렴 실패!)
Job 169: TE=True, FP8 비활성           → log_ppl 4.23에서 정체 (수렴 실패!)
Job 170: TE=False 완전 비활성          → log_ppl 3.296 (수렴 성공!)

결론: TE 자체가 우리 환경에서 수렴을 방해
      FP8는 TE 위에서 동작하므로 당연히 불가
```

---

## 6. 원인 4: GBS=32 — 2 GPU에서 비효율

### 무엇이 문제인가?

레퍼런스의 GBS=32는 8개 GPU에서 효율적으로 분산되도록 설계되었습니다.

### 계산으로 이해하기

```
레퍼런스 (H100 8GPU, TP=4, DP=2):
  각 DP 그룹이 처리하는 데이터 = GBS / DP = 32 / 2 = 16
  Gradient Accumulation = 16 / MBS(1) = 16회 (forward+backward 16번)

우리 환경 (2GPU, TP=2, DP=1):
  각 DP 그룹이 처리하는 데이터 = GBS / DP = 32 / 1 = 32
  Gradient Accumulation = 32 / MBS(1) = 32회 (forward+backward 32번!)

→ 우리 환경에서 GBS=32를 유지하면:
  스텝 시간 = 32 × ~0.74초 ≈ 23.7초/스텝
  총 시간 = 20,000 스텝 × 23.7초 ≈ 131시간 ≈ 5.5일

→ GBS=8로 줄이면 (Dell 설정):
  GA = 8 / 1 = 8회
  스텝 시간 = 8 × ~0.74초 ≈ 5.9초/스텝
  총 시간 = 20,352 × 5.9초 ≈ 33시간 (실측 37.9시간)
```

---

## 7. 원인 5: 컨테이너 이미지 불일치 (Dockerfile)

### 이것이 가장 근본적인 문제입니다

레퍼런스는 **전용 컨테이너**를 Dockerfile로 빌드해서 사용합니다.
이 컨테이너에는 특정 버전의 소프트웨어 스택이 들어 있고, 우리 Dell 컨테이너와는 완전히 다릅니다.

### Dockerfile.h200 분석 (레퍼런스 컨테이너가 만들어지는 과정)

```dockerfile
# ============ 1단계: 베이스 이미지 ============
FROM nvcr.io/nvidia/pytorch:25.01-py3
# → NVIDIA GPU Cloud에서 제공하는 PyTorch 25.01 이미지
# → 우리 Dell 컨테이너의 base: nvcr.io/nvidia/pytorch:25.09-py3 (8개월 이후 버전)
# → ❌ 베이스 이미지부터 다름!


# ============ 2단계: Apex 빌드 (선택) ============
ARG APEX_REVISION=SKIP
# → 기본값 SKIP이므로 빌드하지 않음
# → 베이스 이미지에 포함된 Apex를 그대로 사용
# → ✅ 특별한 문제 없음


# ============ 3단계: Transformer Engine 빌드 (선택) ============
ARG TE_REVISION=SKIP
# → 기본값 SKIP이므로 빌드하지 않음
# → 베이스 이미지에 포함된 TE를 그대로 사용
# → ⚠️ 25.01 이미지의 TE와 25.09 이미지의 TE는 버전이 다를 수 있음
# → ⚠️ 중요: 빌드 시 NVTE_CUDA_ARCHS="90;100"으로 Hopper(90)과 Blackwell(100)만 지원
#      PCIe H100은 Hopper(sm_90)이므로 호환되지만, 최적화 수준이 다를 수 있음


# ============ 4단계: NeMo v2.1.0 설치 ============
ARG NEMO_REVISION=v2.1.0
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    cd NeMo && git checkout v2.1.0 && \
    pip install -e ".[llm]" && pip install -e ".[nlp]"
# → NeMo v2.1.0을 소스에서 설치
# → 우리 Dell 컨테이너: NeMo 25.09-alpha (완전히 다른 버전!)
# → ❌ NeMo API가 다를 수 있음
#   - llm.llama3_8b.pretrain_recipe() 함수가 존재하는지?
#   - Llama31Config8B 클래스가 동일한지?
#   - distributed_fused_adam_with_cosine_annealing() API가 같은지?


# ============ 5단계: NeMo-Run v0.4.0 설치 ============
ARG NEMORUN_REVISION=v0.4.0
RUN git clone https://github.com/NVIDIA/NeMo-Run.git && \
    cd NeMo-Run && git checkout v0.4.0 && \
    pip install -e .
# → NeMo-Run v0.4.0 설치
# → 우리 Dell 컨테이너: v0.5.0
# → ⚠️ API 호환성 불확실
#   - run.Experiment, run.LocalExecutor 등의 인자가 바뀌었을 수 있음


# ============ 6단계: Python 의존성 설치 ============
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# → requirements.txt 내용:
#   git+https://github.com/mlcommons/logging.git@5.0.0-rc2   ← MLPerf 로깅
#   git+https://github.com/NVIDIA/mlperf-common.git@68cf1d0   ← MLPerf 공통 유틸
#   huggingface_hub==0.24.0
#   transformers==4.43.2      ← 우리 컨테이너와 다를 수 있음
#   numpy==1.26.4             ← 우리 컨테이너와 다를 수 있음
#   plotly, nbformat, kaleido, redis  ← 분석/시각화용
# → ❌ mlperf_logging 버전이 다르면 콜백에서 에러 발생 가능


# ============ 7단계: Megatron-Core core_r0.11.0 설치 ============
ARG MCORE_REVISION=core_r0.11.0
RUN git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && git checkout core_r0.11.0 && \
    pip install . && \
    cd megatron/core/datasets && make
# → Megatron-Core core_r0.11.0 설치 (C 라이브러리 빌드 포함)
# → 우리 Dell 컨테이너: 25.09-alpha
# → ❌ 내부 Config 클래스, 병렬화 전략 API가 다를 수 있음


# ============ 8단계: 학습 코드 복사 ============
WORKDIR /workspace/code
COPY . .
# → pretrain_llama31.py, callbacks.py, run_llama31.sh 등을
#   컨테이너 내부 /workspace/code/에 복사
# → 우리 Dell 컨테이너: /workspace/llm/에 Dell 코드가 이미 있음
# → ❌ 작업 디렉토리 구조가 다름
```

### 전체 소프트웨어 스택 비교표

```
구성 요소          레퍼런스 Dockerfile         우리 Dell 컨테이너
─────────         ──────────────────         ──────────────────
베이스 이미지       pytorch:25.01-py3          pytorch:25.09-py3
PyTorch            2.6.0a (추정)              2.7.0a (추정)
CUDA               12.7 (추정)                12.8 (추정)
NeMo               v2.1.0                     25.09-alpha
NeMo-Run           v0.4.0                     v0.5.0
Megatron-Core      core_r0.11.0               25.09-alpha
Transformer Engine 베이스 이미지 내장          베이스 이미지 내장
transformers       4.43.2                     ???
numpy              1.26.4                     ???
mlperf_logging     5.0.0-rc2                  ???
작업 디렉토리       /workspace/code            /workspace/llm
학습 진입점         pretrain_llama31.py        pretrain.py
```

### 왜 이것이 "근본적"인 문제인가?

```
레퍼런스의 pretrain_llama31.py가 호출하는 API 체인:

  pretrain_llama31.py
    → nemo_run (v0.4.0) 의 run.Config, run.Experiment
      → nemo (v2.1.0) 의 llm.llama3_8b.pretrain_recipe()
        → megatron-core (core_r0.11.0) 의 TransformerConfig, ParallelState

각 계층의 버전이 모두 맞물려 있음!
하나라도 버전이 다르면 API 불일치로 에러 발생 가능.

우리 Dell 컨테이너는 3개 계층 모두 다른 버전:
  nemo_run v0.5.0 / nemo 25.09 / megatron-core 25.09
→ 호환될 수도 있지만, 보장할 수 없음
```

---

## 8. 원인 6: NVLink 전제 통신 설정

### 환경변수 차이

**레퍼런스 `pretrain_llama31.py` (SLURM 실행기):**
```python
env_vars = {
    "NCCL_NVLS_ENABLE": "0",              # NVLink Switch 비활성
    "NVTE_FUSED_ATTN": "1",               # Fused Attention 활성 (NVLink에서 빠름)
    "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    "NVTE_ASYNC_AMAX_REDUCTION": "1",
}
```

**우리 Dell 설정 (PCIe):**
```bash
TP_COMM_OVERLAP=False       # 통신 오버랩 비활성 (PCIe에서 비효율)
MC_TP_OVERLAP_AG=False      # All-Gather 오버랩 비활성
MC_TP_OVERLAP_RS=False      # Reduce-Scatter 오버랩 비활성
FULL_CUDA_GRAPH=0           # CUDA Graph 비활성
NVTE_FUSED_ATTN=0           # Fused Attention 비활성 (메모리 절약)
```

→ 레퍼런스의 통신 설정을 PCIe에서 그대로 사용하면 **성능 저하 또는 OOM** 발생 가능

---

## 9. 원인 7: MLPerf 로깅 플랫폼 하드코딩

**`callbacks.py` 53~59행:**
```python
def submission_info(self):
    self.event(key=constants.SUBMISSION_BENCHMARK, value="llama31_8b")
    self.event(key=constants.SUBMISSION_ORG, value="reference_implementation")  # ← 틀림
    self.event(key=constants.SUBMISSION_DIVISION, value=constants.CLOSED)
    self.event(key=constants.SUBMISSION_STATUS, value=constants.ONPREM)
    self.event(key=constants.SUBMISSION_PLATFORM, value="DGX-H100")            # ← 틀림
    self.event(key=constants.SUBMISSION_POC_NAME, value="Yunzhou Liu")          # ← 틀림
    self.event(key=constants.SUBMISSION_POC_EMAIL, value="yunzhoul@nvidia.com") # ← 틀림
```

→ 치명적이지는 않지만, MLPerf compliance 검사에서 경고 발생

---

## 10. 원인 8: 데이터 워커 128개

**`pretrain_llama31.py` get_data() 함수:**
```python
return run.Config(
    llm.PreTrainingDataModule,
    ...
    num_workers=128,    # ← 128개 데이터 로딩 워커!
)
```

```
8 GPU 환경: 128 / 8 = GPU당 16 워커 → 적절
2 GPU 환경: 128 / 2 = GPU당 64 워커 → 과도 (CPU 경쟁, RAM 낭비)
Dell 설정:  num_workers=8 → 2 GPU에 적절
```

---

# Part 2: 수정하면 돌릴 수 있는가? — 파일별 상세 수정 가이드

## 11. 전체 수정 로드맵

레퍼런스 코드를 우리 환경에서 돌리려면, 크게 **2가지 방법**이 있습니다:

```
방법 A: 레퍼런스 Dockerfile로 전용 컨테이너 빌드
  → 소프트웨어 스택 호환성 문제를 완전히 해결
  → 단, nvcr.io 접근 권한 필요 + 빌드 시간 소요
  → 빌드 후에는 config, pretrain_llama31.py, callbacks.py만 수정

방법 B: 기존 Dell 컨테이너에서 레퍼런스 코드 실행
  → Dockerfile 빌드 불필요
  → 단, NeMo/Megatron-Core 버전 차이로 API 에러 발생 가능
  → 모든 파일을 Dell 컨테이너 버전에 맞게 수정해야 함

결론: 방법 A가 더 안전하고 권장됨
      방법 B는 API 호환성을 하나씩 확인해야 해서 시간 소요가 불확실
```

### 수정해야 할 파일 목록

```
파일                          수정 유형        난이도
──────────────────────        ────────        ──────
1. Dockerfile.h200            수정 (빌드용)    중간
2. config (새 파일 생성)       새로 생성        쉬움
3. pretrain_llama31.py        6곳 수정         중간
4. callbacks.py               1곳 수정         쉬움
5. run_llama31.sh             3곳 수정         쉬움
6. SLURM 제출 스크립트         새로 생성        중간
```

---

## 12. 수정 1: Dockerfile — 컨테이너 이미지 빌드

### 왜 Dockerfile을 수정해야 하는가?

레퍼런스의 `Dockerfile.h200`은 **DGX H100(SXM)** 서버를 전제로 작성되었습니다.
우리 **PCIe H100 NVL** 환경에 맞추려면 Dockerfile 자체를 수정해야 합니다.

### 수정이 필요한 부분

#### 수정 1-A: GPU 아키텍처 설정 (Transformer Engine 빌드 시)

```dockerfile
# ===== 원본 (Dockerfile.h200 69행) =====
NVTE_CUDA_ARCHS="90;100" ... pip install --force-reinstall --no-deps .
#                  ↑
# 90 = Hopper (H100/H200), 100 = Blackwell (B200)
# PCIe H100도 sm_90이므로 이 부분은 호환됨 ✅
# 하지만 TE를 커스텀 빌드하는 경우 NVTE_CUDA_ARCHS에 90만 남겨도 됨


# ===== 수정 제안 =====
NVTE_CUDA_ARCHS="90" ... pip install --force-reinstall --no-deps .
# → Blackwell(100)은 필요 없으므로 제거 (빌드 시간 단축)
```

**왜?**
- PCIe H100은 `sm_90` 아키텍처
- Blackwell(100)은 B200 GPU용이므로 우리에게 불필요
- 이 수정은 빌드 시간만 단축시키며, 기능적 차이는 없음

#### 수정 1-B: NeMo 패치 적용

```dockerfile
# ===== 원본에는 없지만 추가해야 할 부분 =====
# Dockerfile.h200에는 패치 적용 단계가 없으나,
# patches/nemo_v2_1_0.patch 파일이 존재함
# 이 패치는 NeMo v2.1.0의 GroupNorm 호환성 문제를 수정

# ===== 추가 제안 (NeMo 설치 후) =====
COPY patches/nemo_v2_1_0.patch /workspace/NeMo/
RUN cd /workspace/NeMo && git apply nemo_v2_1_0.patch
```

**왜?**
- `nemo_v2_1_0.patch`는 Apex의 GroupNorm이 없을 때 `torch.nn.GroupNorm`으로 fallback하는 코드
- 레퍼런스의 Dockerfile에서 `APEX_REVISION=SKIP`(기본값)이면 Apex의 custom GroupNorm이 없을 수 있음
- 패치를 적용해야 import 에러를 방지

#### 수정 1-C: 작업 디렉토리와 코드 복사

```dockerfile
# ===== 원본 (131~134행) =====
WORKDIR /workspace/code
COPY . .
# → 레퍼런스 코드가 /workspace/code/에 복사됨
# → 이 부분은 수정 불필요 ✅ (레퍼런스의 디렉토리 구조를 그대로 사용)
```

#### 수정 1-D: 빌드 명령어

```bash
# ===== 원본 방식 =====
docker build -t mlperf-smallllm:ref -f Dockerfile.h200 .
# → nvcr.io/nvidia/pytorch:25.01-py3 이미지가 필요
# → NVIDIA GPU Cloud(NGC) 접근 권한이 있어야 함

# ===== 실제 빌드 명령 =====
# NGC 접근 권한이 있는 경우:
docker login nvcr.io
docker build -t mlperf-smallllm:ref -f Dockerfile.h200 .

# NGC 접근 권한이 없는 경우:
# → 대안 1: Dell 컨테이너 사용 (방법 B)
# → 대안 2: 공개 PyTorch 이미지 사용
docker build -t mlperf-smallllm:ref \
  --build-arg FROM_IMAGE_NAME=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  -f Dockerfile.h200 .
# → 이 경우 CUDA 버전, cuDNN 등이 달라 추가 문제 발생 가능
```

### Dockerfile.h200 수정 전체 diff 요약

```
변경 사항:
1. NVTE_CUDA_ARCHS: "90;100" → "90" (선택사항, 빌드 시간 단축)
2. NeMo 패치 적용 단계 추가 (권장)
3. 나머지는 그대로 유지 ✅

주의사항:
- nvcr.io/nvidia/pytorch:25.01-py3 이미지 접근 권한 필요
- 빌드 시간: 30분~2시간 (네트워크 환경에 따라)
- 빌드된 이미지 크기: ~20GB
```

---

## 13. 수정 2: config 파일 — 새 설정 파일 생성

레퍼런스에는 H100용 config(`config_H100_1x8x4_8b.sh`)가 있지만,
우리 환경(2 GPU, PCIe)에 맞는 config가 없으므로 **새로 만들어야** 합니다.

### 원본: `config_H100_1x8x4_8b.sh`

```bash
# ===== 원본 설정 (핵심 부분만) =====
export GPUS_PER_NODE=8
export GBS=32
export MBS=1
export MAX_LR="5e-4"
export WARMUP_STEPS=512              # = 16384 / GBS(32)
export EVAL_EVERY=12288
export START_EVAL_AT=0
export TENSOR_PARALLEL_SIZE=4
```

### 새로 만들 파일: `config_PCIeH100_1x2_8b.sh`

```bash
# ===== 우리 환경에 맞는 새 설정 =====

# SSH 관련 (로컬 실행이므로 DUMMY 유지)
export USER="$(whoami)"
export HOST="localhost"
export ACCOUNT="DUMMY"
export PARTITION="DUMMY"
export TIME="60:00:00"               # ← 60시간 (충분한 시간)

# 하드웨어 설정
export NNODES=1
export GPUS_PER_NODE=2                # ← 8 → 2 (우리 GPU 수)

# 데이터 경로 (우리 환경)
export JOB_DIR="/output"
export IMAGE="sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt"  # 또는 직접 빌드한 이미지
export PREPROCESSED_PATH="/lustre/agent/mltraining_v5.1_llama3_8b/8b"
export TOKENIZER_PATH="/lustre/agent/mltraining_v5.1_llama3_8b/8b/tokenizer"
export TMP_NPY_INDEX="/lustre/agent/mltraining_v5.1_llama3_8b/npy_index_ref"
export CONTINUAL_CKPT="/lustre/agent/mltraining_v5.1_llama3_8b/checkpoints"

# 학습 하이퍼파라미터
export SIZE="8b"
export GBS=8                          # ← 32 → 8 (2 GPU에 맞게 축소)
export MBS=1
export MAX_LR="4e-4"                  # ← 5e-4 → 4e-4 (GBS=8에 맞게 조정, 실험으로 검증)
export WARMUP_STEPS=128               # ← 512 → 128 (GBS가 작아졌으므로 비례 조정)
export EVAL_EVERY=3072                # ← 12288 → 3072 (GBS=8일때 384 step마다 = Dell과 동일)
export START_EVAL_AT=0
export TENSOR_PARALLEL_SIZE=2         # ← 4 → 2 (GPU 2개이므로)

# 체크포인트
export SAVE_CKPT=0
export START_STEPS="0"
export NEXP=1
export NPAR=1

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
```

### 원본과의 차이 상세 설명

```
변경 항목          원본 값     새 값       이유
─────────         ────────   ────────    ──────────────────────────────
GPUS_PER_NODE     8          2           GPU가 2개뿐
GBS               32         8           DP=1이므로 GA=32는 너무 느림
                                         GBS=8 → GA=8회로 적절한 속도
MAX_LR            5e-4       4e-4        GBS가 작아지면 LR도 낮추는 것이 안전
                                         (Linear Scaling Rule: LR ∝ GBS)
                                         Dell 실험에서 4e-4가 최적으로 확인됨
WARMUP_STEPS      512        128         원본: 16384 samples / GBS(32) = 512
                                         우리: 16384 / 8 = 2048이 비례값이지만
                                         Dell 실험에서 128이 적절하다고 확인됨
EVAL_EVERY        12288      3072        원본: 12288 sequences마다 검증
                                         GBS=8일때: 12288/8 = 1536 step마다
                                         → 너무 드물어서 3072(=384 step)으로 변경
                                         Dell과 동일한 검증 빈도
TENSOR_PARALLEL   4          2           GPU 2개이므로 TP=2가 최대
PREPROCESSED_PATH (DUMMY)    (우리 경로)  실제 데이터 경로 설정
TOKENIZER_PATH    (DUMMY)    (우리 경로)  실제 토크나이저 경로 설정
TMP_NPY_INDEX     (DUMMY)    (우리 경로)  numpy 인덱스 저장 경로
```

---

## 14. 수정 3: pretrain_llama31.py — 핵심 학습 스크립트

이 파일이 **가장 많은 수정이 필요**합니다. 6곳을 수정해야 합니다.

### 수정 3-A: local_executor() — GPU 수와 환경변수 (34~58행)

```python
# ===== 원본 =====
def local_executor(
    custom_env_vars=None,
    devices=8,                     # ← 8 GPU 기본값
    retries=0,
):
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    # ...


# ===== 수정 =====
def local_executor(
    custom_env_vars=None,
    devices=2,                     # ← 2로 변경 (우리 GPU 수)
    retries=0,
):
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NVTE_FUSED_ATTN": "0",           # ← 추가: Fused Attention 비활성 (메모리 절약)
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # ← 추가: 메모리 단편화 방지
    }
    # ...
```

**왜?**
- `devices=8` → `devices=2`: GPU가 2개뿐이므로
- `NVTE_FUSED_ATTN=0`: Fused Attention의 workspace 버퍼가 1~2GB를 소모, PCIe H100에서 OOM 방지
- `PYTORCH_CUDA_ALLOC_CONF`: GPU 메모리 단편화를 줄여 OOM 방지

### 수정 3-B: slurm_executor() — GPU 하드코딩 (60~120행)

```python
# ===== 원본 (105행) =====
executor = run.SlurmExecutor(
    ...
    gres="gpu:8",                  # ← 8 GPU 하드코딩
    ...
)


# ===== 수정 =====
executor = run.SlurmExecutor(
    ...
    gres=f"gpu:{devices}",         # ← 인자로 받은 devices 사용
    ...
)
```

**왜?**
- SLURM에서 GPU를 할당할 때 8개 고정이 아니라 인자에 따라 달라져야 함
- 로컬 실행 시에는 이 함수를 사용하지 않으므로, 로컬 실행만 할 거라면 수정 안 해도 됨

### 수정 3-C: get_pretrain() — TP 설정과 FP8 비활성화 (122~196행)

```python
# ===== 원본 (148~150행) =====
pretrain.trainer.strategy.tensor_model_parallel_size = 1
pretrain.trainer.strategy.pipeline_model_parallel_size = 1
pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = 1
pretrain.trainer.strategy.context_parallel_size = 1


# ===== 수정 =====
# TP 크기를 config에서 받은 값으로 설정 (cmd line arg --tensor_parallel_size)
# pretrain_llama31.py의 main 블록에서 이미 args.tensor_parallel_size로
# override하고 있으므로, 여기서는 기본값 1을 유지해도 됨.
# 하지만 명시적으로 2로 설정하면 더 안전:
pretrain.trainer.strategy.tensor_model_parallel_size = 2   # ← 1 → 2
pretrain.trainer.strategy.pipeline_model_parallel_size = 1
pretrain.trainer.strategy.virtual_pipeline_model_parallel_size = 1
pretrain.trainer.strategy.context_parallel_size = 1
```

```python
# ===== 원본 (160~172행) — FP8 설정 =====
precision = run.Config(
    nl.MegatronMixedPrecision,
    precision="bf16-mixed",
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_enabled=True,
    grad_reduce_in_fp32=False,
    fp8="hybrid",                    # ← FP8 활성화!
    fp8_amax_history_len=4,
    fp8_amax_compute_algo='most_recent',
    fp8_params=True,                 # ← FP8 파라미터!
    fp8_dot_product_attention=False,
)


# ===== 수정 — FP8 완전 비활성화 =====
precision = run.Config(
    nl.MegatronMixedPrecision,
    precision="bf16-mixed",
    params_dtype=torch.bfloat16,
    pipeline_dtype=torch.bfloat16,
    autocast_enabled=True,
    grad_reduce_in_fp32=False,
    # FP8 관련 설정 모두 제거 또는 비활성화
    # fp8="hybrid",                  # 제거!
    # fp8_amax_history_len=4,        # 제거!
    # fp8_amax_compute_algo='most_recent',  # 제거!
    # fp8_params=True,               # 제거!
    # fp8_dot_product_attention=False,      # 제거!
)
```

**왜?**
- 우리 환경에서 FP8 활성화 시 학습이 수렴하지 않음 (Job 168, 169에서 확인)
- TP=2로 변경 (GPU가 2개이므로)
- `fp8` 관련 인자를 전부 제거하면 순수 BF16 모드로 동작

### 수정 3-D: get_data() — 워커 수와 데이터 경로 (198~249행)

```python
# ===== 원본 (230~239행) =====
return run.Config(
    llm.PreTrainingDataModule,
    ...
    num_workers=128,              # ← 128 워커
    ...
    index_mapping_dir="/npy_index",
    ...
)


# ===== 수정 =====
return run.Config(
    llm.PreTrainingDataModule,
    ...
    num_workers=8,                # ← 128 → 8 (2 GPU에 적절)
    ...
    index_mapping_dir="/npy_index",
    ...
)
```

**왜?**
- 128개 워커는 2 GPU 환경에서 GPU당 64개로 과도함
- CPU 코어 경쟁, 메모리 낭비 유발
- Dell 설정에서 8개 워커가 적절하다고 확인됨

### 수정 3-E: val_check_interval 계산 (180행)

```python
# ===== 원본 =====
pretrain.trainer.val_check_interval = eval_every / int(os.getenv("GBS"))


# ===== 주의 =====
# 이 코드는 환경변수 GBS를 읽어서 검증 간격을 계산
# 우리 config에서 GBS=8로 설정하면 자동으로 조정됨
# config에서 EVAL_EVERY=3072로 설정했으므로:
#   val_check_interval = 3072 / 8 = 384 step (Dell과 동일)
# → 이 부분은 수정 불필요 ✅ (config에서 조정)
```

### 수정 3-F: max_lr 인자 전달 (306행, argparse)

```python
# ===== 원본 =====
data_group.add_argument("--max_lr", type=float, default=1e-4, help="Peak learning rate.")


# ===== 주의 =====
# config에서 MAX_LR="4e-4"를 설정하고 run_llama31.sh에서 --max_lr로 전달하므로
# argparse 기본값은 무시됨
# → 수정 불필요 ✅ (config에서 override)
```

---

## 15. 수정 4: callbacks.py — 로깅 및 콜백

### 수정 4-A: 플랫폼 정보 (52~59행)

```python
# ===== 원본 =====
def submission_info(self):
    self.event(key=constants.SUBMISSION_BENCHMARK, value="llama31_8b")
    self.event(key=constants.SUBMISSION_ORG, value="reference_implementation")
    self.event(key=constants.SUBMISSION_DIVISION, value=constants.CLOSED)
    self.event(key=constants.SUBMISSION_STATUS, value=constants.ONPREM)
    self.event(key=constants.SUBMISSION_PLATFORM, value="DGX-H100")
    self.event(key=constants.SUBMISSION_POC_NAME, value="Yunzhou Liu")
    self.event(key=constants.SUBMISSION_POC_EMAIL, value="yunzhoul@nvidia.com")


# ===== 수정 =====
def submission_info(self):
    self.event(key=constants.SUBMISSION_BENCHMARK, value="llama31_8b")
    self.event(key=constants.SUBMISSION_ORG, value="TTA")                        # ← 수정
    self.event(key=constants.SUBMISSION_DIVISION, value=constants.CLOSED)
    self.event(key=constants.SUBMISSION_STATUS, value=constants.ONPREM)
    self.event(key=constants.SUBMISSION_PLATFORM, value="PCIe_H100_NVL_1x2")     # ← 수정
    self.event(key=constants.SUBMISSION_POC_NAME, value="Your Name")             # ← 수정
    self.event(key=constants.SUBMISSION_POC_EMAIL, value="your@email.com")       # ← 수정
```

**왜?**
- MLPerf 규정 준수를 위해 실제 하드웨어/조직 정보를 기입해야 함
- 이 부분이 틀려도 학습 자체는 돌아가지만, 결과 제출 시 문제가 됨

---

## 16. 수정 5: run_llama31.sh — 실행 스크립트

### 수정 5-A: 필수 변수 체크 완화 (23~26행)

```bash
# ===== 원본 =====
: "${USER:?USER not set}"
: "${HOST:?HOST not set}"
: "${ACCOUNT:?ACCOUNT not set}"
: "${PARTITION:?PARTITION not set}"


# ===== 수정 (로컬 실행 시 SSH/SLURM 관련 체크 불필요) =====
: "${USER:=$(whoami)}"
: "${HOST:=localhost}"
: "${ACCOUNT:=local}"
: "${PARTITION:=local}"
```

**왜?**
- 레퍼런스의 `run_llama31.sh`는 원래 SSH로 원격 SLURM 클러스터에 접속하는 것을 전제
- 로컬 실행(`--run_slurm` 없이)이면 이 변수들이 실제로 사용되지 않음
- 하지만 `?`로 체크하면 값이 없을 때 에러로 종료됨
- 기본값을 넣어주면 에러 없이 통과

### 수정 5-B: TENSOR_PARALLEL_SIZE 체크 (121~123행)

```bash
# ===== 원본 =====
if [ $TENSOR_PARALLEL_SIZE -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
fi


# ===== 주의 =====
# config에서 TENSOR_PARALLEL_SIZE=2로 설정하면 자동으로 전달됨
# → 수정 불필요 ✅
```

### 수정 5-C: MAX_LR 전달 추가 (149~152행 부근)

```bash
# ===== 원본 (마지막 python3 호출 부분) =====
python3 pretrain_llama31.py \
    ...
    --warmup_steps $WARMUP_STEPS \
    --eval_every $EVAL_EVERY \
    --start_eval_at $START_EVAL_AT \
    $CMD_SUFFIX


# ===== 수정: MAX_LR 전달 추가 =====
python3 pretrain_llama31.py \
    ...
    --warmup_steps $WARMUP_STEPS \
    --max_lr $MAX_LR \                    # ← 추가! (config의 LR을 전달)
    --eval_every $EVAL_EVERY \
    --start_eval_at $START_EVAL_AT \
    $CMD_SUFFIX
```

**왜?**
- 원본 `run_llama31.sh`에는 `--max_lr` 전달이 빠져있음!
- `pretrain_llama31.py`의 argparse에 `--max_lr` 인자가 있지만,
  `run_llama31.sh`에서 전달하지 않으면 기본값 `1e-4`가 사용됨
- config에서 `MAX_LR="4e-4"`를 설정해도 전달하지 않으면 무시됨
- → 반드시 `--max_lr $MAX_LR`를 추가해야 함

---

## 17. 수정 6: SLURM 제출 스크립트 — 새로 생성

레퍼런스 코드는 NeMo-Run의 `LocalExecutor`로 torchrun을 직접 실행합니다.
하지만 우리 환경에서는 SLURM으로 자원을 할당받고 Pyxis 컨테이너 안에서 실행해야 합니다.

### 방법 A: 컨테이너 안에서 직접 실행 (권장)

```bash
#!/bin/bash
# run_reference_slurm.sh — SLURM으로 레퍼런스 코드 실행

#SBATCH --job-name=ref_llama8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=60:00:00

CONT="mlperf-smallllm:ref"   # 위 Dockerfile로 빌드한 이미지
# 또는 Dell 컨테이너: CONT="sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt"

DATADIR="/lustre/agent/mltraining_v5.1_llama3_8b"
LOGDIR="${DATADIR}/log_ref"
NPY_DIR="${DATADIR}/npy_index_ref"
REF_CODE="/mlstorage/training/small_llm_pretraining_claude/nemo"

mkdir -p $LOGDIR $NPY_DIR

# 컨테이너 셋업
CONT_NAME="ref_llama_${SLURM_JOB_ID}"
PYXIS_DEFAULTS="--no-container-mount-home --container-remap-root --container-writable"

srun --ntasks-per-node=1 \
  --container-image="${CONT}" \
  --container-name="${CONT_NAME}" \
  ${PYXIS_DEFAULTS} true

# 컨테이너 안에서 실행
srun -N1 -n1 \
  --container-name="${CONT_NAME}" \
  ${PYXIS_DEFAULTS} \
  --container-mounts="${REF_CODE}:/workspace/code,${DATADIR}/8b:/preproc_data,${DATADIR}/8b/tokenizer:/tokenizer,${LOGDIR}:/output,${LOGDIR}:/mlperf-outputs,${NPY_DIR}:/npy_index,${DATADIR}/checkpoints:/continual" \
  --container-workdir=/workspace/code \
  bash -c '
    source config_PCIeH100_1x2_8b.sh
    export PREPROCESSED_PATH=/preproc_data
    export GBS=8
    bash run_llama31.sh
  '
```

### 방법 B: 레퍼런스의 local_executor가 직접 torchrun 실행

```
이 방법은 SLURM 할당 후 컨테이너 안에서:
1. source config_PCIeH100_1x2_8b.sh
2. bash run_llama31.sh
3. run_llama31.sh가 python3 pretrain_llama31.py 호출
4. pretrain_llama31.py 안의 local_executor가 torchrun 2 GPU 실행
5. NeMo-Run이 학습 시작

주의: srun의 --ntasks-per-node=1로 해야 함 (NeMo-Run이 내부에서 torchrun을 띄우므로)
      --ntasks-per-node=2로 하면 2번 실행되어 충돌!
```

---

## 18. 수정 요약 체크리스트

```
# 수정 전 준비사항

□ 레퍼런스 코드 복사: cp -r /mlstorage/training/small_llm_pretraining
                          /mlstorage/training/small_llm_pretraining_claude
□ npy_index 디렉토리 생성: mkdir -p /lustre/agent/mltraining_v5.1_llama3_8b/npy_index_ref


# 파일별 수정 체크리스트

□ Dockerfile.h200 (방법 A를 사용하는 경우)
  □ NVTE_CUDA_ARCHS="90;100" → "90" (선택)
  □ NeMo 패치 적용 단계 추가 (권장)
  □ docker build로 이미지 빌드

□ config_PCIeH100_1x2_8b.sh (새로 생성)
  □ GPUS_PER_NODE=2
  □ GBS=8
  □ MAX_LR="4e-4"
  □ WARMUP_STEPS=128
  □ EVAL_EVERY=3072
  □ TENSOR_PARALLEL_SIZE=2
  □ 데이터 경로 설정

□ pretrain_llama31.py (6곳 수정)
  □ local_executor: devices=8 → 2
  □ local_executor: NVTE_FUSED_ATTN=0, PYTORCH_CUDA_ALLOC_CONF 추가
  □ slurm_executor: gres="gpu:8" → "gpu:{devices}" (선택)
  □ get_pretrain: tensor_model_parallel_size = 2
  □ get_pretrain: FP8 관련 설정 전부 제거
  □ get_data: num_workers=128 → 8

□ callbacks.py (1곳 수정)
  □ submission_info: 조직, 플랫폼, 담당자 정보 변경

□ run_llama31.sh (2곳 수정)
  □ SSH/SLURM 필수 변수 체크 → 기본값 부여
  □ --max_lr $MAX_LR 전달 추가

□ SLURM 제출 스크립트 (새로 생성)
  □ run_reference_slurm.sh 작성
  □ 컨테이너 마운트 경로 설정
  □ srun --ntasks-per-node=1 (NeMo-Run이 내부에서 torchrun 실행하므로)


# 실행 순서

1. 코드 복사 및 수정
2. (방법 A) Dockerfile 빌드 또는 (방법 B) Dell 컨테이너 사용
3. 짧은 테스트 실행 (MAX_STEPS=100)
4. OOM 체크, step time 확인
5. 문제 없으면 full run
```

---

# Part 3: 종합 비교

## 19. large_llm vs small_llm: 불가능의 수준 비교

```
               large_llm (405B)            small_llm (8B)
               ────────────────            ────────────────
물리적          ❌ 절대 불가                ✅ 가능
               (모델이 GPU에 안 들어감)     (8B는 2 GPU에 들어감)

구조적          ❌ 완전 재작성 필요          🟡 6개 파일 수정으로 가능할 수도
               (TP=8,PP=7 필요)             (TP=2,PP=1로 변경)

소프트웨어      ❌ 완전히 다름               🟡 버전 차이 (호환 가능성 있음)
               (NeMo-Run 없음)             (NeMo-Run v0.5.0 있음)

Dockerfile     ❌ 의미 없음                🟡 빌드 가능 (NGC 접근 필요)
               (모델이 안 돌아감)           (빌드하면 스택 호환성 해결)

시간            ❌ 수백 년                  🟡 수일 수준 (GBS 조정 시)
               (2,304 GPU 필요)            (2 GPU로 ~38시간)

수정 항목       ❌ 사실상 전부               ✅ 6개 파일, 약 15곳
               (모든 것이 불일치)           (나머지는 그대로 사용 가능)

결론           "불가능"                     "어려우나 가능"
```

---

## 20. 최종 판정

```
Q: 레퍼런스 코드를 우리 환경에서 직접(수정 없이) 돌릴 수 있는가?
A: 아니오 ❌

Q: 수정하면 돌릴 수 있는가?
A: 가능할 수도 있음 🟡
   - 방법 A (Dockerfile 빌드): 성공 확률 70~80%
   - 방법 B (Dell 컨테이너 사용): 성공 확률 40~60% (API 호환성 불확실)

Q: 수정에 걸리는 시간은?
A: 방법 A: 1~2일 (Dockerfile 빌드 + 코드 수정 + 테스트)
   방법 B: 2~4일 (API 호환성 디버깅 포함)

Q: Dell 구현체 대비 장단점은?
A:
   레퍼런스 수정 방식:
     장점 - MLPerf 공식 코드 기반, 최신 NeMo 2.x API 활용
     단점 - 수정 포인트가 많고, 검증이 안 된 경로

   Dell 구현체:
     장점 - 이미 모든 수정 완료, 37.9시간에 수렴 검증됨
     단점 - NeMo 1.x (Hydra) 기반의 구버전 API

Q: 그래서 어떤 것을 쓰는 게 좋은가?
A: Dell 구현체를 사용하는 것이 훨씬 효율적 ✅
   레퍼런스를 돌려보는 것은 교육/연구 목적으로 의미가 있음
```

### 비유로 정리

```
상황: "서울에서 부산까지 가야 한다"

Dell 구현체     = KTX 타기 (표 사서 타면 됨, 이미 다 세팅됨)
레퍼런스 수정   = 직접 자동차 개조해서 가기 (재미있지만, 시간과 노력 필요)
레퍼런스 직접   = 비행기 활주로에 자전거로 진입 시도 (안 됨)
```

---

> **문서 작성일**: 2026-02-12
> **레퍼런스 경로**: `/mlstorage/training/small_llm_pretraining/nemo/`
> **Dell 구현체 경로**: `/mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo/`
> **현재 환경**: 1노드 × 2GPU (NVIDIA H100 NVL PCIe, 93GB)
> **현재 Dell 컨테이너**: `sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt` (NeMo-Run v0.5.0 확인)
