# Small LLM Pretraining: MLPerf 레퍼런스 vs Dell 구현체 비교

> MLPerf Training의 `small_llm_pretraining` 레퍼런스 코드와 Dell의 `llama31_8b` 구현체의 차이를 상세 비교합니다.
> 두 코드 모두 **동일한 LLaMA 3.1 8B 모델**을 학습하지만, 구조와 실행 방식이 크게 다릅니다.

---

## 목차

1. [핵심 요약: 같은 모델, 다른 구현](#1-핵심-요약-같은-모델-다른-구현)
2. [디렉토리 구조 비교](#2-디렉토리-구조-비교)
3. [레퍼런스: 각 파일의 역할](#3-레퍼런스-각-파일의-역할)
4. [Dell 구현체: 각 파일의 역할](#4-dell-구현체-각-파일의-역할)
5. [차이점 1: 실행 방식](#5-차이점-1-실행-방식)
6. [차이점 2: 하드웨어 설정](#6-차이점-2-하드웨어-설정)
7. [차이점 3: 설정 시스템](#7-차이점-3-설정-시스템)
8. [차이점 4: 학습 하이퍼파라미터](#8-차이점-4-학습-하이퍼파라미터)
9. [차이점 5: 정밀도와 최적화](#9-차이점-5-정밀도와-최적화)
10. [차이점 6: 소프트웨어 스택](#10-차이점-6-소프트웨어-스택)
11. [차이점 7: 콜백과 워밍업](#11-차이점-7-콜백과-워밍업)
12. [차이점 8: 컨테이너 실행](#12-차이점-8-컨테이너-실행)
13. [전체 파라미터 비교표](#13-전체-파라미터-비교표)
14. [실행 흐름 비교](#14-실행-흐름-비교)
15. [한눈에 보는 요약](#15-한눈에-보는-요약)

---

## 1. 핵심 요약: 같은 모델, 다른 구현

### 이번 비교의 특이점

`large_language_model_pretraining`(405B)과 달리, `small_llm_pretraining`은 **우리와 같은 8B 모델**을 대상으로 합니다. 목표 log_ppl도 동일하게 3.3입니다.

```
레퍼런스 (small_llm_pretraining)   ←  같은 모델, 같은 목표  →   Dell 구현체 (llama31_8b)
          8B 모델                                                   8B 모델
       log_ppl ≤ 3.3                                            log_ppl ≤ 3.3
      1노드 × 8 GPU                                           1노드 × 2 GPU
    NeMo-Run 오케스트레이션                                    SLURM 직접 실행
```

비유하면, **같은 음식 레시피**인데 **주방 규모와 도구가 다른** 것입니다.

### 관계도

```
MLCommons
    │
    └─ small_llm_pretraining 레퍼런스 제공
       (8B 모델, 1노드×8GPU, NeMo-Run)
       │
       └─ /mlstorage/training/small_llm_pretraining/nemo/
            │
            ▼
NVIDIA + Dell
    │
    └─ 프로덕션 최적화 + 하드웨어 특화
       (8B 모델, 다양한 GPU 수, SLURM 직접)
       │
       └─ /mlstorage/training_results_v5.1/TTA_Claude/.../nemo/
            │
            ▼
우리 (TTA)
    │
    └─ PCIe H100 2GPU에 맞게 최종 튜닝
       └─ config_PCIeH100_TTA.sh
```

---

## 2. 디렉토리 구조 비교

### 레퍼런스

```
small_llm_pretraining/nemo/
├── pretrain_llama31.py              # 학습 오케스트레이션 (NeMo-Run)
├── callbacks.py                     # MLPerf 콜백
├── run_llama31.sh                   # 실행 래퍼
├── README.md                        # 문서
├── requirements.txt                 # Python 의존성
├── config_H100_1x8x4_8b.sh         # H100 설정 (TP=4, 8GPU)
├── config_H200_1x8x1_8b.sh         # H200 설정 (TP=1, 8GPU)
├── config_MI325X_1x8x1_8b.sh       # AMD MI325X 설정 (TP=1, 8GPU)
├── Dockerfile.h200                  # H200 컨테이너
├── Dockerfile.b200                  # B200 컨테이너
├── Dockerfile.mi325                 # AMD MI325X 컨테이너
├── patches/
│   └── nemo_v2_1_0.patch           # AMD 호환성 패치
└── utils/
    ├── preprocess.sh                # 데이터 전처리
    ├── consolidate_data.sh          # 데이터 통합
    └── parallel_compress_json_to_gz.sh  # 병렬 압축
```

### Dell 구현체

```
llama31_8b/implementations/nemo/
├── pretrain.py                      # 학습 스크립트 (Hydra/OmegaConf)
├── custom_callbacks.py              # 커스텀 콜백 (워밍업, 메모리 프로파일링)
├── mocking.py                       # Mock 데이터 모듈
├── run.sub                          # SLURM 작업 제출 (551줄)
├── run_and_time.sh                  # 컨테이너 내부 실행기
├── run_direct.sh                    # 직접 실행
├── Dockerfile                       # 단일 프로덕션 컨테이너
├── requirements.txt                 # Python 의존성
├── config_common.sh                 # 공통 기본 설정
├── config_common_8b.sh              # 8B 전용 설정
├── config_common_cg.sh              # CUDA Graph 설정
├── config_common_fp8attn.sh         # FP8 Attention 설정
├── config_mounts.sh                 # 마운트 설정
├── config_PCIeH100_TTA.sh           # ★ PCIe H100 2GPU 전용
├── conf/
│   ├── custom.yaml                  # 동적 설정 계산
│   ├── nccl/                        # NCCL 통신 튜닝
│   ├── tp_overlap/                  # TP 오버랩 프로파일
│   └── data_prefix/                 # 데이터 경로
├── embedding_lib/                   # 커스텀 임베딩
├── data_scripts/                    # 데이터 도구
└── mlperf-logging/                  # 규정 준수 검증
```

### 구조적 차이

| 항목 | 레퍼런스 | Dell |
|------|---------|------|
| 파일 수 | ~15개 | ~50개+ |
| 설정 파일 | GPU별 1개 (3종) | 계층적 7개+ |
| 지원 GPU | H100, H200, B200, MI325X | PCIe H100 (커스텀) |
| Dockerfile | GPU별 3개 | 단일 프로덕션 1개 |
| YAML 설정 | 없음 | Hydra/OmegaConf |
| 워밍업 | 없음 | 합성 데이터 워밍업 |
| 모니터링 | 없음 | 전력/메모리/행감지 |

---

## 3. 레퍼런스: 각 파일의 역할

### 3.1 pretrain_llama31.py — "학습 총괄 스크립트" (24KB)

```
역할: NeMo-Run 프레임워크를 사용해 학습의 모든 것을 관리
      모델 정의, 데이터 설정, SLURM 작업 제출, 체크포인트 관리
```

**핵심 함수들:**

```python
# local_executor() — 로컬 실행기 (torchrun 기반)
#   레퍼런스만의 장점: 로컬 실행 가능!
#   torchrun으로 단일 노드에서 실행
#   BUT: 여전히 NeMo-Run 의존성 필요

# slurm_executor() — 원격 SLURM 실행기
#   SSH 터널로 원격 클러스터에 작업 제출
#   gres="gpu:8" 하드코딩 (8 GPU 고정)

# get_pretrain() — 모델 레시피 정의
#   Llama31Config8B 사용 (8B 모델)
#   시퀀스 길이: 8,192
#   FP8 하이브리드 정밀도 활성화
#   TP 크기: CLI 인자로 설정 가능

# get_data() — 데이터 설정
#   C4 데이터셋 (8개 학습 샤드 + 1개 검증)
#   토크나이저: LLaMA 3.1 (128K 어휘)
#   128개 데이터 워커

# main() — 실행 로직
#   로컬/SLURM 실행 모드 선택
#   데이터 인덱스 빌드
#   멀티 시드/멀티 파티션 실험 관리
```

**large_llm 레퍼런스와의 차이:**
- `local_executor()` 함수가 있어 **로컬 실행 가능** (large_llm에는 없음)
- TP 크기를 CLI 인자로 받음 (large_llm은 405B에 맞게 고정)
- 8B 모델이 "디버깅용"이 아닌 **공식 대상**

### 3.2 callbacks.py — "학습 감시자" (8.8KB)

```
역할: MLPerf 규정 준수 로깅, 성능 측정, 수렴 감지
```

```python
# MLLogger — MLPerf 로거
#   벤치마크명: "llama31_8b" (405B가 아닌 8B!)
#   플랫폼: "DGX-H100" (하드코딩)
#   목표: log_ppl ≤ 3.3

# PreemptiveStop — 조기 종료
#   지정 스텝에서 학습 중단 (체크포인트 테스트용)

# MetricsLogger — 성능 지표
#   검증 loss 추적
#   스텝 시간 검증 (기본 7,200초 제한)
#   목표 달성 시 학습 종료 트리거

# MLPerfCallback — MLPerf 이벤트
#   RUN_START/STOP, EVAL_START/STOP 등 기록
#   수렴 성공/실패 판정
```

### 3.3 config_H100_1x8x4_8b.sh — "H100 8GPU 설정"

```
역할: NVIDIA H100 8GPU에 최적화된 설정
      우리 환경과 가장 유사한 설정 파일
```

```bash
# 하드웨어
NNODES=1              # 1노드
GPUS_PER_NODE=8       # 8 GPU (우리는 2)

# 배치 사이즈
GBS=32                # 글로벌 배치 (우리는 8)
MBS=1                 # 마이크로 배치 (동일)

# 병렬화
TENSOR_PARALLEL=4     # TP=4 (우리는 2)
# DP = 8 / 4 = 2     # 데이터 병렬 2

# 학습률
MAX_LR=5e-4           # 0.0005 (우리는 0.0004)
WARMUP_STEPS=512      # 워밍업 512 (우리는 128)

# 검증
EVAL_EVERY=12288      # 12,288 시퀀스마다
TARGET=3.3            # 목표 (동일!)
```

### 3.4 config_H200_1x8x1_8b.sh — "H200 8GPU 설정"

```bash
# H200: 더 큰 GPU 메모리 → TP 불필요
TENSOR_PARALLEL=1     # TP=1 (모델 전체가 1 GPU에)
MBS=2                 # MBS=2 (메모리 여유)
# DP = 8              # 8개 GPU 전부 데이터 병렬
```

### 3.5 config_MI325X_1x8x1_8b.sh — "AMD MI325X 설정"

```bash
# AMD GPU: 완전히 다른 최적화 필요
TENSOR_PARALLEL=1
MBS=4                 # MBS=4 (AMD는 큰 배치에 유리)
MAX_LR=1e-3           # LR=0.001 (NVIDIA보다 2배 높음!)
# AMD 전용 환경변수 30개+ 추가
```

### 3.6 Dockerfile.h200 / .b200 / .mi325 — "GPU별 컨테이너"

```
역할: 각 GPU 하드웨어에 맞는 소프트웨어 스택 구축
```

| Dockerfile | GPU | 베이스 이미지 | 특징 |
|-----------|-----|-------------|------|
| .h200 | H200 | pytorch:25.01-py3 | Apex + 커스텀 TE |
| .b200 | B200 | pytorch:25.04-py3 | Apex 불필요, 간소화 |
| .mi325 | MI325X | rocm6.4 | AMD ROCm, Flash Attn CK |

### 3.7 utils/ — "데이터 도구"

```
preprocess.sh               — C4 텍스트를 토큰화된 바이너리로 변환
consolidate_data.sh          — 1024개 C4 샤드를 8개로 통합
parallel_compress_json_to_gz.sh — 병렬 JSON 압축
```

### 3.8 patches/nemo_v2_1_0.patch — "AMD 호환성 패치"

```
역할: NeMo v2.1.0에서 AMD GPU 지원을 위한 수정
      NVIDIA Apex가 없을 때 torch.nn 표준 모듈로 대체
```

---

## 4. Dell 구현체: 각 파일의 역할

> Dell 구현체의 각 파일 상세 설명은 이전 문서 `MANUAL_Reference_vs_Dell_Comparison.md`의 4장에 있으므로,
> 여기서는 **레퍼런스와의 차이점에 초점**을 맞춥니다.

### 핵심 차이 요약

| 파일 | 레퍼런스 대응 파일 | 주요 차이 |
|------|------------------|----------|
| pretrain.py | pretrain_llama31.py | Hydra 설정, 워밍업, CUDA Graph, 어텐션 백엔드 선택 |
| custom_callbacks.py | callbacks.py | 워밍업 관리, 메모리 프로파일링, 커스텀 임베딩 |
| run.sub | (해당 없음) | SLURM 오케스트레이션 551줄 (레퍼런스는 NeMo-Run) |
| run_and_time.sh | (해당 없음) | 컨테이너 내부 실행기 (레퍼런스에 없는 계층) |
| config_common.sh | (해당 없음) | NCCL 미세 튜닝 97줄 (레퍼런스에 없음) |
| config_PCIeH100_TTA.sh | config_H100_1x8x4_8b.sh | 2GPU 전용, TE 비활성, LR 튜닝 |
| conf/custom.yaml | (해당 없음) | 동적 설정 계산 (레퍼런스는 Python 내장) |
| embedding_lib/ | (해당 없음) | 커스텀 고속 임베딩 (레퍼런스에 없음) |
| mocking.py | (해당 없음) | 워밍업용 Mock 데이터 (레퍼런스에 없음) |

---

## 5. 차이점 1: 실행 방식

### 레퍼런스: NeMo-Run (Python 오케스트레이션)

```
두 가지 모드 제공:

1. 로컬 모드 (local_executor):
   pretrain_llama31.py → torchrun → 학습 시작
   (NeMo-Run이 내부적으로 torchrun 호출)

2. 원격 모드 (slurm_executor):
   pretrain_llama31.py → SSH → SLURM → 학습 시작
```

**장점**: 로컬 실행 모드가 있어 단일 노드에서 테스트 가능
**단점**: NeMo-Run, NeMo v2.1.0 등 특정 버전 라이브러리 필요

### Dell: SLURM 직접 + Pyxis 컨테이너

```
sbatch run.sub → srun run_and_time.sh → python pretrain.py
     │                  │                      │
  SLURM 관리       컨테이너 설정            학습 실행
  (551줄)          (188줄)                (762줄)
```

**장점**: 투명한 제어, 디버깅 용이, 환경변수 전달 명확
**단점**: SLURM 필수, run.sub이 복잡

---

## 6. 차이점 2: 하드웨어 설정

### 레퍼런스: 1노드 × 8GPU (3가지 GPU 타입)

```
H100 설정:  8 GPU, TP=4, DP=2, MBS=1, GBS=32
H200 설정:  8 GPU, TP=1, DP=8, MBS=2, GBS=32
MI325X:     8 GPU, TP=1, DP=8, MBS=4, GBS=32
```

### Dell (우리): 1노드 × 2GPU

```
PCIe H100:  2 GPU, TP=2, DP=1, MBS=1, GBS=8
```

### 핵심 차이

| 항목 | 레퍼런스 (H100) | Dell (PCIe H100) |
|------|----------------|-----------------|
| GPU 수 | 8 | **2** (4배 적음) |
| GPU 연결 | NVLink 900GB/s | **PCIe 128GB/s** (7배 느림) |
| TP | 4 | **2** |
| DP | 2 | **1** |
| GBS | 32 | **8** (4배 작음) |
| MBS | 1 | 1 (동일) |
| GA | 16 | **8** |

### GPU 연결이 미치는 영향

```
레퍼런스 (NVLink):
  GPU0 ←─900GB/s─→ GPU1 ←─900GB/s─→ GPU2 ←─...─→ GPU7
  모든 GPU가 초고속으로 연결 → TP=4도 효율적

Dell (PCIe):
  GPU0 ←─128GB/s─→ GPU1
  2개뿐이고 느림 → TP=2가 한계, 통신 오버랩 비효율적
```

---

## 7. 차이점 3: 설정 시스템

### 레퍼런스: GPU별 독립 설정 파일

```
config_H100_1x8x4_8b.sh   → H100 전용 설정 (모든 값 포함)
config_H200_1x8x1_8b.sh   → H200 전용 설정 (모든 값 포함)
config_MI325X_1x8x1_8b.sh → MI325X 전용 설정 (모든 값 포함)

→ 각 파일이 독립적, 서로 의존하지 않음
→ 단순하지만 중복이 많음
```

### Dell: 계층적 오버라이드

```
config_common.sh          (1층: 공통 기본값)
    ↓
config_common_8b.sh       (2층: 8B 모델 전용)
    ↓
config_common_cg.sh       (3층: CUDA Graph)
    ↓
config_common_fp8attn.sh  (4층: FP8 Attention)
    ↓
config_PCIeH100_TTA.sh    (5층: ★ 최종 오버라이드)

+ conf/custom.yaml        (동적 수식 계산)

→ 중복 최소화, 유연한 구조
→ 하지만 "어떤 값이 최종 적용되는지" 추적이 어려움
```

---

## 8. 차이점 4: 학습 하이퍼파라미터

| 파라미터 | 레퍼런스 (H100) | Dell (PCIe H100) | 비고 |
|---------|----------------|-----------------|------|
| **LR (max)** | 5e-4 (0.0005) | **4e-4 (0.0004)** | Dell이 약간 낮음 |
| **LR (min)** | 5e-5 (max/10) | 4e-5 (max/10) | 비율 동일 |
| **Warmup 스텝** | 512 | **128** | Dell이 4배 짧음 |
| **GBS** | 32 | **8** | GPU 수 비례 |
| **MBS** | 1 | 1 | 동일 |
| **GA** | 16 (DP별) | **8** | GBS/MBS |
| **VAL 간격** | 384 스텝 | 384 스텝 | 동일 |
| **목표** | log_ppl ≤ 3.3 | log_ppl ≤ 3.3 | 동일 |
| **옵티마이저** | AdamW | AdamW | 동일 |
| **β1, β2** | 0.9, 0.95 | 0.9, 0.95 | 동일 |
| **ε** | **1e-8** | **1e-5** | 미세한 차이 |
| **Weight Decay** | 0.1 | 0.1 | 동일 |
| **Grad Clip** | 1.0 | 1.0 | 동일 |
| **시퀀스 길이** | 8,192 | 8,192 | 동일 |

### LR 차이의 의미

```
레퍼런스:  LR=0.0005, GBS=32 → 큰 배치 + 큰 보폭
Dell:     LR=0.0004, GBS=8  → 작은 배치 + 약간 작은 보폭

일반적으로 GBS가 작으면 LR도 줄여야 안정적입니다.
레퍼런스 GBS(32)의 1/4인 Dell GBS(8)에서
LR을 0.0005 → 0.0004로 낮춘 것은 합리적인 조정입니다.
```

### Warmup 스텝 차이의 의미

```
레퍼런스:  512 스텝 warmup → 512 × 32 = 16,384 시퀀스 처리 후 최대 LR 도달
Dell:     128 스텝 warmup → 128 × 8  = 1,024 시퀀스 처리 후 최대 LR 도달

→ 레퍼런스는 16배 더 많은 데이터를 보고 나서 최대 LR에 도달
→ Dell은 빠르게 최대 LR에 도달하여 학습 시간 절약
```

---

## 9. 차이점 5: 정밀도와 최적화

### 레퍼런스: FP8 하이브리드 활성화

```python
# pretrain_llama31.py 내 정밀도 설정:
precision="bf16-mixed"
fp8="hybrid"                    # FP8 하이브리드 활성화!
fp8_amax_history_len=4
fp8_amax_compute_algo='most_recent'
fp8_params=True                 # FP8 파라미터 저장
fp8_dot_product_attention=False  # 어텐션은 FP8 안 씀
```

### Dell (우리 설정): FP8 완전 비활성

```bash
# config_PCIeH100_TTA.sh:
TRANSFORMER_ENGINE=False   # TE 비활성 (수렴 문제)
FP8=False                  # FP8 비활성
FP8_HYBRID=False           # FP8 하이브리드 비활성
```

### 왜 다른가?

```
레퍼런스 (DGX H100, NVLink):
  TE + FP8 → 동작함 → 13~20% 속도 향상
  NVLink 덕분에 TP=4에서도 통신 오버헤드 적음

Dell (PCIe H100):
  TE + FP8 → OOM 또는 수렴 실패
  TE 활성화 시 스텝 속도 13% 향상, 하지만 loss가 수렴하지 않음
  → 수렴 가능성이 속도보다 중요하므로 TE 비활성 선택
```

### 최적화 비교표

| 최적화 | 레퍼런스 | Dell (PCIe) | 이유 |
|--------|---------|-------------|------|
| Transformer Engine | **활성** | 비활성 | 수렴 실패 |
| FP8 하이브리드 | **활성** | 비활성 | TE 필요 |
| CUDA Graph | 미사용 | 비활성 | PCIe에서 효과 미미 |
| TP 통신 오버랩 | 미명시 | **비활성** | PCIe 대역폭 부족 |
| 분산 옵티마이저 | 미명시 | **활성** | 메모리 절약 |
| 워밍업 | 없음 | **합성 데이터 2스텝** | 첫 스텝 속도 최적화 |
| NCCL 미세 튜닝 | 기본값 | **CTA별 커스텀** | 통신 효율화 |
| 커스텀 임베딩 | 없음 | 있음 (비활성) | TP 최적화 |

---

## 10. 차이점 6: 소프트웨어 스택

```
                    레퍼런스 (H200 기준)       Dell
                    ────────────────          ─────────
베이스 이미지        pytorch:25.01-py3        pytorch:25.09-py3
NeMo               v2.1.0                   25.09-alpha
NeMo-Run           v0.4.0                   v0.5.0
Megatron-Core      core_r0.11.0             25.09-alpha
PyTorch            25.01                    25.09
CUDA               12.x                     13.0
TE                 커스텀 리비전              v2.8 (SM90 전용)
Apex               커스텀 빌드              선택적
MLPerf logging     5.0.0-rc2               5.1.0
transformers       4.43.2                   제거됨
```

**Dell이 더 최신 버전**을 사용합니다 (약 6개월 차이).

---

## 11. 차이점 7: 콜백과 워밍업

### 레퍼런스: 기본 콜백만 (244줄)

```
MLLogger           → MLPerf 로그 기록
PreemptiveStop     → 조기 종료
MetricsLogger      → 검증 loss 추적
MLPerfCallback     → MLPerf 이벤트 기록
```

### Dell: 프로덕션 풀세트 (850줄+)

```
DeltaTimingCallback   → 스텝 정밀 타이밍
CustomCallback        → 워밍업, FP8 관리, CUDA Graph
MetricsLogger         → MLPerf + 처리량 계산
MemoryProfileCallback → GPU/CPU 메모리 추적
PrintArtifacts        → 설정 덤프
```

### 워밍업 차이 (Dell만 있음)

```
레퍼런스:
  run_start → 바로 학습 시작
  → 첫 1-2 스텝이 느림 (CUDA 커널 컴파일, JIT 등)

Dell:
  run_start 전:
    1. Mock 데이터로 Forward 2스텝 → GPU 메모리 패턴 확립
    2. Mock 데이터로 Validation 2스텝 → 검증 경로 워밍업
    3. FP8 통계 리셋
    4. 가비지 컬렉션
  run_start 후:
    → 진짜 학습 시작 (첫 스텝부터 최적 속도)
```

---

## 12. 차이점 8: 컨테이너 실행

### 레퍼런스: NeMo-Run이 컨테이너 관리

```
NeMo-Run이 자동으로:
  - Docker/Enroot 이미지 관리
  - 마운트 포인트 설정
  - 환경변수 전달
  - torchrun 실행
```

### Dell: Pyxis + 수동 환경변수 전달

```
run.sub에서 수동으로:
  - Pyxis로 컨테이너 생성
  - --container-env로 환경변수 명시적 전달
  - 마운트 포인트 config_mounts.sh에서 관리
  - srun으로 각 GPU에서 실행

★ 환경변수 전달 문제:
  Docker 이미지에 TRANSFORMER_ENGINE=False가 내장
  → 호스트에서 True로 설정해도 컨테이너에 전달 안 됨
  → --container-env에 명시적으로 추가해야 함
  (이것이 우리가 발견한 중요한 버그였음)
```

---

## 13. 전체 파라미터 비교표

### 모델 설정 (동일)

| 파라미터 | 레퍼런스 | Dell | 일치 |
|---------|---------|------|------|
| 모델 | LLaMA 3.1 8B | LLaMA 3.1 8B | ✅ |
| 레이어 수 | 32 | 32 | ✅ |
| 히든 크기 | 4,096 | 4,096 | ✅ |
| FFN 크기 | 14,336 | 14,336 | ✅ |
| 어텐션 헤드 | 32 | 32 | ✅ |
| KV 헤드 | 8 (GQA) | 8 (GQA) | ✅ |
| 시퀀스 길이 | 8,192 | 8,192 | ✅ |
| 어휘 크기 | 128,000 | 128,256 | ⚠️ 미세 차이 |

### 하드웨어/병렬화 (다름)

| 파라미터 | 레퍼런스 (H100) | Dell (PCIe) |
|---------|----------------|-------------|
| GPU 수 | 8 | **2** |
| TP | 4 | **2** |
| PP | 1 | 1 |
| DP | 2 | **1** |
| CP | 1 | 1 |
| GBS | 32 | **8** |
| MBS | 1 | 1 |
| GA | 16 | **8** |

### 학습 설정 (일부 다름)

| 파라미터 | 레퍼런스 | Dell |
|---------|---------|------|
| LR | 0.0005 | **0.0004** |
| Warmup 스텝 | 512 | **128** |
| 검증 간격 | 384 스텝 | 384 스텝 |
| 목표 log_ppl | 3.3 | 3.3 |
| 최대 스텝 | 1,200,000 | 1,200,000 |
| TE | 활성 | **비활성** |
| FP8 | 하이브리드 | **비활성** |
| CUDA Graph | 미사용 | 비활성 |

---

## 14. 실행 흐름 비교

### 레퍼런스

```
source config_H100_1x8x4_8b.sh → bash run_llama31.sh
                                       │
                                       ▼
                               pretrain_llama31.py
                                 ├─ 로컬/SLURM 선택
                                 ├─ NeMo-Run 실행기 설정
                                 ├─ 데이터 인덱스 빌드
                                 ├─ 체크포인트 로드
                                 │
                                 └─ [학습 루프]
                                      ├─ Forward → Loss → Backward
                                      ├─ 매 384 스텝: 검증
                                      └─ log_ppl ≤ 3.3 → 종료
```

### Dell

```
source config_PCIeH100_TTA.sh → sbatch run.sub
                                       │
                                       ▼
                               run.sub (SLURM)
                                 ├─ 컨테이너 초기화
                                 ├─ 네트워크 설정
                                 ├─ [선택] NCCL 테스트
                                 ├─ [선택] 모니터링 시작
                                 ├─ 캐시 클리어
                                 │
                                 └─ srun run_and_time.sh
                                       │
                                       ▼
                               run_and_time.sh (컨테이너 내)
                                 ├─ 환경 검증
                                 ├─ 체크포인트 설정
                                 │
                                 └─ python pretrain.py
                                       ├─ YAML 설정 로드
                                       ├─ 워밍업 (Mock 2스텝)
                                       │
                                       └─ [학습 루프]
                                            ├─ Forward → Loss → Backward
                                            ├─ 매 384 스텝: 검증
                                            └─ log_ppl ≤ 3.3 → 종료
```

---

## 15. 한눈에 보는 요약

```
┌──────────────────┬──────────────────────┬──────────────────────────┐
│     항목          │  레퍼런스 (small_llm)  │   Dell (llama31_8b)       │
├──────────────────┼──────────────────────┼──────────────────────────┤
│ 모델             │ LLaMA 3.1 8B         │ LLaMA 3.1 8B (동일)       │
│ 목표             │ log_ppl ≤ 3.3        │ log_ppl ≤ 3.3 (동일)      │
│ GPU 수           │ 8개 (NVLink)         │ 2개 (PCIe)               │
│ GBS              │ 32                   │ 8                        │
│ LR               │ 0.0005              │ 0.0004                   │
│ Warmup           │ 512 스텝             │ 128 스텝                  │
│ 실행 방식         │ NeMo-Run             │ SLURM 직접                │
│ TE/FP8           │ 활성                 │ 비활성 (수렴 문제)          │
│ 설정 구조         │ GPU별 독립 파일       │ 계층적 오버라이드           │
│ 워밍업            │ 없음                 │ 합성 데이터 2스텝           │
│ 콜백              │ 4개 (244줄)          │ 5개+ (850줄)              │
│ 소프트웨어        │ NeMo 2.1.0           │ NeMo 25.09               │
│ 최고 기록         │ (공식 제출 참조)       │ 37.9시간 (20,352스텝)     │
└──────────────────┴──────────────────────┴──────────────────────────┘
```

### 한 문장 요약

> **레퍼런스**는 "8B 모델을 8개 H100 GPU에서 FP8로 빠르게 학습하는 표준 레시피"이고,
> **Dell 구현체**는 "같은 8B 모델을 2개 PCIe H100에서 TE/FP8 없이도 안정적으로 수렴시키기 위해
> 하이퍼파라미터를 튜닝하고 프로덕션 인프라를 추가한 코드"입니다.

---

> **문서 작성일**: 2026-02-22
> **레퍼런스 경로**: `/mlstorage/training/small_llm_pretraining/nemo/`
> **Dell 구현체 경로**: `/mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo/`
