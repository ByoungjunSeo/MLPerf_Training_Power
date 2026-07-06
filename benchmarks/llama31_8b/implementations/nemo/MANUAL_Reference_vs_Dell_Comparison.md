# MLPerf 레퍼런스 vs Dell 구현체 완전 비교 가이드

> MLPerf Training에서 제공하는 공식 레퍼런스 코드와 Dell이 이를 기반으로 개발한 코드의 차이를 상세하게 비교합니다.

---

## 목차

1. [전체 개요: 두 구현체의 관계](#1-전체-개요-두-구현체의-관계)
2. [디렉토리 구조 비교](#2-디렉토리-구조-비교)
3. [레퍼런스 구현체: 각 파일의 역할](#3-레퍼런스-구현체-각-파일의-역할)
4. [Dell 구현체: 각 파일의 역할](#4-dell-구현체-각-파일의-역할)
5. [핵심 차이점 1: 대상 모델과 하드웨어](#5-핵심-차이점-1-대상-모델과-하드웨어)
6. [핵심 차이점 2: 실행 방식 (NeMo-Run vs SLURM 직접)](#6-핵심-차이점-2-실행-방식-nemo-run-vs-slurm-직접)
7. [핵심 차이점 3: 학습 스크립트 구조](#7-핵심-차이점-3-학습-스크립트-구조)
8. [핵심 차이점 4: 설정 시스템](#8-핵심-차이점-4-설정-시스템)
9. [핵심 차이점 5: 소프트웨어 스택 버전](#9-핵심-차이점-5-소프트웨어-스택-버전)
10. [핵심 차이점 6: 성능 최적화](#10-핵심-차이점-6-성능-최적화)
11. [핵심 차이점 7: 콜백과 로깅](#11-핵심-차이점-7-콜백과-로깅)
12. [핵심 차이점 8: 데이터 처리](#12-핵심-차이점-8-데이터-처리)
13. [핵심 차이점 9: 체크포인트 관리](#13-핵심-차이점-9-체크포인트-관리)
14. [설정 파라미터 전체 비교표](#14-설정-파라미터-전체-비교표)
15. [실행 흐름 비교](#15-실행-흐름-비교)
16. [요약: 한눈에 보는 차이점](#16-요약-한눈에-보는-차이점)

---

## 1. 전체 개요: 두 구현체의 관계

### 비유로 이해하기

```
MLPerf 레퍼런스 = "자동차 설계도" (표준 스펙)
Dell 구현체     = "실제로 만든 자동차" (특정 공장에 맞게 수정)
```

- **MLPerf 레퍼런스**: MLCommons에서 제공하는 "이렇게 하면 됩니다"라는 공식 구현체
  - 모든 참가자가 이 코드를 기반으로 시작
  - 405B (4050억 파라미터) 초대형 모델이 기본 대상
  - 288노드 × 8GPU = 2,304개 H100 GPU를 기본으로 가정

- **Dell 구현체**: Dell이 자사 하드웨어에 맞게 수정한 구현체
  - 8B (80억 파라미터) 소형 모델에 맞게 재설정
  - 1노드 × 2GPU = 2개 PCIe H100 GPU에 최적화
  - NVIDIA의 프로덕션 최적화 코드를 추가 적용

### 코드의 흐름

```
MLCommons (MLPerf)
    │
    ├─ 레퍼런스 구현체 제공 (대상: 405B, 2304 GPU)
    │   └─ /mlstorage/training/large_language_model_pretraining/
    │
    ▼
NVIDIA
    │
    ├─ 프로덕션 최적화 코드 개발
    │   (Transformer Engine, CUDA Graph, FP8, 통신 오버랩 등)
    │
    ▼
Dell
    │
    ├─ NVIDIA 코드 기반으로 자사 하드웨어에 맞게 수정
    │   └─ /mlstorage/training_results_v5.1/TTA_Claude/.../nemo/
    │
    ▼
우리 (TTA)
    │
    └─ Dell 코드에서 PCIe H100 2GPU에 맞는 최적화 진행
        └─ config_PCIeH100_TTA.sh 수정
```

---

## 2. 디렉토리 구조 비교

### 레퍼런스 구현체 (간결한 구조)

```
large_language_model_pretraining/nemo/
│
├── pretrain_llama31.py        # 학습 스크립트 (537줄)
├── callbacks.py               # MLPerf 콜백 (244줄)
├── config.sh                  # 설정 파일 (1개)
├── demo_config.sh             # 데모용 설정
├── run_llama31.sh             # 실행 스크립트
├── run_demo.sh                # 데모 실행 스크립트
├── Dockerfile                 # 컨테이너 정의
├── Dockerfile_mlcube          # MLCube용 컨테이너
├── mcore.patch                # Megatron-Core 패치
├── README.md                  # 문서
│
├── mlcube/                    # MLCube 통합
│   ├── mlcube.yaml
│   └── README.md
│
└── utils/                     # 유틸리티 도구
    ├── preprocess.sh          # 데이터 전처리
    ├── consolidate_data.sh    # 데이터 통합
    ├── nemo_convert.py        # 체크포인트 변환
    ├── launch_nemo_convert.sh # 변환 실행기
    └── download_demo.sh       # 데모 데이터 다운로드
```

### Dell 구현체 (복잡한 프로덕션 구조)

```
llama31_8b/implementations/nemo/
│
├── pretrain.py                # 학습 스크립트 (762줄, 더 복잡)
├── custom_callbacks.py        # 커스텀 콜백 (850줄+, 훨씬 복잡)
├── mocking.py                 # Mock 데이터 모듈
├── run.sub                    # SLURM 작업 제출 (551줄, 대규모)
├── run_and_time.sh            # 컨테이너 내부 실행 (188줄)
├── run_direct.sh              # 직접 실행 (SLURM 없이)
├── Dockerfile                 # 컨테이너 정의 (훨씬 복잡)
├── requirements.txt           # Python 의존성
│
├── config_common.sh           # 공통 설정 (97줄)
├── config_common_8b.sh        # 8B 모델 설정
├── config_common_cg.sh        # CUDA Graph 설정
├── config_common_fp8attn.sh   # FP8 Attention 설정
├── config_common_fp4.sh       # FP4 양자화 설정
├── config_mounts.sh           # 컨테이너 마운트 설정
├── config_PCIeH100_TTA.sh     # ★ PCIe H100 전용 설정
│
├── conf/                      # YAML 설정 (계층적)
│   ├── custom.yaml            # 메인 설정 (212줄)
│   ├── llama31_config_custom.yaml
│   ├── data_prefix/           # 데이터 경로 설정
│   ├── nccl/                  # NCCL 통신 튜닝
│   └── tp_overlap/            # TP 통신 오버랩 설정
│
├── embedding_lib/             # 커스텀 임베딩 라이브러리
├── data_scripts/              # 데이터 준비 스크립트
├── mlperf-logging/            # MLPerf 규정 준수 검증
└── power_logs/                # 전력 모니터링
```

### 구조적 차이 요약

| 항목 | 레퍼런스 | Dell |
|------|---------|------|
| **파일 수** | ~15개 | ~50개+ |
| **설정 파일** | 1개 (config.sh) | 7개+ (계층적 sh + yaml) |
| **코드 복잡도** | 단순, 교육용 | 복잡, 프로덕션용 |
| **최적화 수준** | 기본 | 하드웨어 특화 |
| **YAML 설정** | 없음 (Python 내장) | 있음 (Hydra/OmegaConf) |
| **유틸리티** | 데이터 전처리 도구 | 모니터링, 프로파일링 도구 |

---

## 3. 레퍼런스 구현체: 각 파일의 역할

### 3.1 pretrain_llama31.py — "학습의 두뇌" (537줄)

```
역할: 모든 것을 총괄하는 메인 스크립트
      모델 정의, 데이터 로딩, 학습 실행, 체크포인트 관리를 모두 담당
```

**주요 함수와 역할:**

```python
# 1. slurm_executor() — SLURM 클러스터 연결 설정
#    "어떤 서버에서, 몇 개의 GPU로 실행할지 정의"
#    - SSH 터널 설정 (원격 클러스터 접근)
#    - Docker 컨테이너 이미지 지정
#    - 디렉토리 마운트 설정
#    - 환경변수 전달 (NCCL 설정 등)

# 2. get_pretrain() — 모델 레시피 정의
#    "어떤 모델을 어떤 설정으로 학습할지 정의"
#    - 8B/70B/405B 모델 크기 선택
#    - 학습률, 옵티마이저, 스케줄러 설정
#    - 병렬화 전략 (TP, PP, CP) 설정
#    - 통신 오버랩 설정

# 3. get_data() — 데이터 설정
#    "어떤 데이터를 어떤 형태로 읽을지 정의"
#    - C4 데이터셋 경로와 가중치
#    - 토크나이저 설정
#    - 배치 사이즈, 시퀀스 길이

# 4. main() — 실행 로직
#    "실제로 학습을 시작하고 관리"
#    - 명령줄 인자 파싱
#    - MLPerf 규정 준수 로깅
#    - 데이터 인덱스 빌드 (첫 실행 시)
#    - 멀티 시드/멀티 파티션 실험 관리
#    - 체크포인트에서 재개
```

**특징:**
- **NeMo-Run 프레임워크** 사용: Python 코드로 SLURM 작업 제출
- SSH를 통해 원격 클러스터에 작업 전송 가능
- 멀티 파티션 지원: 매우 긴 학습을 여러 SLURM 작업으로 분할

### 3.2 callbacks.py — "학습 감시자" (244줄)

```
역할: 학습 중 발생하는 이벤트를 감시하고 기록
      MLPerf 규정 준수를 위한 로깅 담당
```

**클래스별 역할:**

```python
# 1. MLLogger — MLPerf 규정 준수 로거
#    "학습 과정을 MLPerf 규칙에 맞게 기록"
#    - Rank 0 (첫 번째 GPU)에서만 로그 출력
#    - 벤치마크명: llama31_405b
#    - 조직: reference_implementation

# 2. PreemptiveStop — 조기 종료 콜백
#    "지정된 스텝에서 학습을 멈추게 하는 장치"
#    - 체크포인트 테스트나 디버깅에 사용
#    - 예: 1000 스텝까지만 실행

# 3. MetricsLogger — 성능 지표 기록기
#    "학습 속도와 정확도를 기록"
#    - 검증 Loss (log_ppl) 기록
#    - 스텝 소요 시간 기록
#    - 목표 달성 여부 체크

# 4. MLPerfCallback — MLPerf 이벤트 관리자
#    "학습의 주요 시점마다 공식 기록 남기기"
#    이벤트들:
#    - CACHE_CLEAR: 캐시 초기화
#    - INIT_START/STOP: 초기화 구간
#    - RUN_START/STOP: 학습 구간
#    - EPOCH_START/STOP: 에폭 구간
#    - BLOCK_START/STOP: 학습 블록 (스텝 범위)
#    - EVAL_START/STOP: 검증 구간
```

### 3.3 config.sh — "설정표" (짧은 변수 나열)

```
역할: 사용자가 채워야 하는 빈 칸이 있는 설정 양식
      모든 값이 비어 있어서 사용 전에 반드시 채워야 함
```

```bash
# SLURM 클러스터 접속 정보
USER=""                  # SSH 사용자명
HOST=""                  # 클러스터 호스트명
ACCOUNT=""               # SLURM 계정
PARTITION=""             # GPU 파티션
TIME="04:00:00"          # 작업 시간 제한

# 하드웨어 설정
NNODES=288               # 노드 수 (기본 288대!)
GPUS_PER_NODE=8          # 노드당 GPU (기본 8개)

# 경로 설정 (모두 빈 값)
JOB_DIR=""               # 출력 디렉토리
IMAGE=""                 # Docker 이미지
PREPROCESSED_PATH=""     # 데이터 경로
TOKENIZER_PATH=""        # 토크나이저 경로

# 학습 설정
SIZE="405b"              # 모델 크기 (기본 405B!)
GBS=1152                 # 글로벌 배치 사이즈 (거대)
```

### 3.4 run_llama31.sh — "작업 제출기" (144줄)

```
역할: config.sh를 읽어서 pretrain_llama31.py를 올바른 인자로 실행
      사용자가 직접 실행하는 진입점
```

**동작 순서:**
```
1. config.sh에서 변수 읽기
2. 필수 변수 검증 (빠진 것 있으면 에러)
3. Docker 컨테이너 마운트 포인트 설정
4. pretrain_llama31.py에 전달할 커맨드 라인 구성
5. Python 실행 → NeMo-Run이 SLURM에 작업 제출
```

### 3.5 Dockerfile — "컨테이너 설계도"

```
역할: 학습에 필요한 모든 소프트웨어를 담은 컨테이너 정의
```

```dockerfile
# 베이스: NVIDIA NeMo 24.12-rc0 컨테이너
FROM nvcr.io/nvidia/nemo:24.12-rc0

# 추가 패키지
RUN pip install transformers==4.47.1     # HuggingFace 트랜스포머
RUN pip install blobfile==3.0.0          # 클라우드 스토리지
RUN pip install prettytable==3.12.0      # 표 출력
RUN pip install mlperf-logging@4.1.0-rc3 # MLPerf 로깅

# Megatron-Core 패치 적용
RUN patch --directory=/opt/megatron-lm -p1 < mcore.patch
```

### 3.6 mcore.patch — "검증 데이터 셔플 방지 패치"

```
역할: Megatron-Core의 데이터셋 처리 버그 수정
      검증 데이터가 무작위로 섞이지 않도록 보장
```

**수정 내용:**
```python
# 패치 전: 학습/검증 데이터 모두 셔플
shuffle = True  # 항상 셔플

# 패치 후: 학습 데이터만 셔플, 검증 데이터는 원래 순서 유지
shuffle = (self.index_split == Split.train)  # 학습일 때만 셔플
```

이것이 중요한 이유: 검증 데이터가 매번 다른 순서로 나오면
같은 모델이어도 검증 점수가 매번 달라져서 **재현성이 깨집니다.**

### 3.7 utils/ — "데이터 준비 도구 모음"

```
preprocess.sh       — C4 JSON 텍스트를 토큰화된 바이너리(.bin/.idx)로 변환
consolidate_data.sh — 1024개 C4 샤드를 8개 파일로 통합
nemo_convert.py     — HuggingFace 체크포인트를 NeMo 형식으로 변환
download_demo.sh    — 데모용 데이터/모델 다운로드
```

---

## 4. Dell 구현체: 각 파일의 역할

### 4.1 pretrain.py — "학습의 두뇌" (762줄, 레퍼런스보다 40% 더 복잡)

```
역할: 모델 정의, 데이터 로딩, 학습 실행
      레퍼런스와 같은 역할이지만 훨씬 세밀한 제어 가능
```

**레퍼런스와의 핵심 차이:**

```python
# 1. Hydra/OmegaConf 설정 시스템 사용
#    레퍼런스: Python 코드 안에 설정 직접 작성
#    Dell:    YAML 파일에서 설정을 읽고, 환경변수로 동적 계산

# 2. 더 많은 정밀도 옵션 지원
#    레퍼런스: BF16만 지원
#    Dell:    BF16, FP8, FP4, FP8 하이브리드 모두 지원

# 3. Warmup 시스템
#    레퍼런스: 워밍업 없음
#    Dell:    합성 데이터로 2스텝 워밍업 후 본 학습 시작
#             → GPU 메모리 버퍼 사전 할당, CUDA 커널 사전 컴파일

# 4. CUDA Graph 지원
#    레퍼런스: 미지원
#    Dell:    전체 반복(full iteration) 또는 레이어별 CUDA Graph 가능

# 5. 어텐션 백엔드 선택
#    레퍼런스: 기본값 사용
#    Dell:    환경변수(ATTENTION_BACKEND)로 cuDNN/FlashAttn/auto 선택 가능

# 6. 커스텀 임베딩
#    레퍼런스: 표준 임베딩
#    Dell:    embedding_lib의 커스텀 고속 임베딩 지원
```

**주요 함수:**

```python
# get_data()
#   → PreTrainingDataModule 설정
#   → C4 데이터 경로, 토크나이저, 배치 사이즈 설정
#   → 데이터 인덱스 캐시 경로 (/npy_index)

# mock_data()
#   → 워밍업용 가짜 데이터 생성
#   → 192개 학습 배치 + 1개 테스트 배치

# get_model_with_precision()
#   → Llama31Config8B/70B/405B 인스턴스 생성
#   → FP8/FP4 정밀도 설정
#   → 어텐션 백엔드, Cross-Entropy 퓨전, 활성화 체크포인트 설정

# get_optimizer()
#   → AdamW 옵티마이저 + 코사인 스케줄러
#   → 분산 옵티마이저 활성화

# get_strategy()
#   → Megatron 분산 전략 설정
#   → TP/PP/CP/VP 병렬화
#   → 기울기/파라미터 오버랩
#   → NCCL 커뮤니케이터 설정

# get_trainer()
#   → PyTorch Lightning Trainer 생성
#   → 검증 주기, 로깅 주기, 최대 스텝 설정

# train()  [메인 엔트리포인트]
#   → 위 함수들을 조합하여 학습 실행
#   → MLPerf 로깅 시작/종료
```

### 4.2 custom_callbacks.py — "학습 감시자" (850줄+, 레퍼런스의 3.5배)

```
역할: 학습의 모든 단계를 감시하고 최적화
      레퍼런스보다 훨씬 많은 기능 포함
```

**클래스별 역할:**

```python
# 1. DeltaTimingCallback
#    "각 스텝의 소요 시간을 정밀 측정"
#    → 학습 스텝 시간, 검증 스텝 시간 각각 추적
#    → 성능 벤치마킹의 핵심 지표

# 2. CustomCallback (가장 복잡, 300줄+)
#    "학습 전체 생명주기를 관리하는 총괄 콜백"
#
#    워밍업 관리:
#    → 합성 데이터로 사전 학습 (2스텝)
#    → GPU 메모리 패턴을 미리 확립
#    → FP8 통계를 워밍업 후 리셋
#
#    커스텀 임베딩:
#    → TP(Tensor Parallel) 환경에서 임베딩 연산 최적화
#    → forward hook으로 표준 임베딩을 커스텀 버전으로 교체
#
#    CUDA Graph 관리:
#    → 학습 종료 시 CUDA Graph를 명시적으로 해제
#    → 메모리 누수 및 행(hang) 방지
#
#    검증 주기 조정:
#    → val_check_interval을 동적으로 조정
#    → 학습 초반 불필요한 검증 건너뛰기

# 3. MetricsLogger
#    "MLPerf 규정에 맞게 성능 지표 기록"
#    → 학습 loss, 검증 log_ppl 기록
#    → 목표 달성 시 학습 종료 트리거
#    → 스텝당 처리량(throughput) 계산

# 4. MemoryProfileCallback
#    "GPU 메모리 사용량을 실시간 추적"
#    → 스텝별 GPU 할당량 기록
#    → OOM 디버깅에 필수적인 도구
#    → 호스트(CPU) 메모리도 추적

# 5. PrintArtifacts
#    "학습 시작 시 모든 설정값을 출력"
#    → 디버깅과 재현성을 위한 설정 덤프

# 헬퍼 함수들:
# run_training_warmup()  — 합성 데이터 워밍업 실행
# reset_fp8_state()      — FP8 통계 초기화
# custom_embed_forward()  — TP용 커스텀 임베딩 forward pass
```

### 4.3 run.sub — "작업 총괄 지휘관" (551줄, 가장 복잡한 파일)

```
역할: SLURM 작업 제출부터 학습 완료까지 모든 것을 오케스트레이션
      레퍼런스에는 이에 해당하는 파일이 없음 (NeMo-Run이 대체)
```

**섹션별 역할:**

```bash
# ===== 1단계: 환경 검증 (1-50줄) =====
# 필수 환경변수 확인 (CONT, DGXSYSTEM, LOGDIR 등)
# 없으면 에러 메시지와 함께 종료

# ===== 2단계: 호스트 파일 생성 (50-80줄) =====
# SLURM이 할당한 노드 목록을 토폴로지 순서로 정렬
# → 물리적으로 가까운 GPU끼리 통신하도록 최적화

# ===== 3단계: 컨테이너 초기화 (80-230줄) =====
# Pyxis를 통해 Docker 컨테이너 생성
# → --no-container-mount-home: 홈 디렉토리 마운트 안 함
# → --container-writable: 컨테이너 쓰기 가능
# → --container-remap-root: root 권한 매핑
# 마운트 포인트 설정 (데이터, 체크포인트, 로그 등)
# GPU 디바이스 확인, InfiniBand 설정

# ===== 4단계: 네트워크 설정 (230-270줄) =====
# MASTER_ADDR 선택 (분산학습 코디네이터 IP)
# NCCL 네트워크 인터페이스 탐지
# IPoIB(IP over InfiniBand) 지원

# ===== 5단계: 진단 테스트 (270-340줄) =====
# [선택] NCCL 집합 통신 테스트 (all_reduce 성능 측정)
# [선택] GEMM-통신 오버랩 테스트
# [선택] 패브릭 무결성 검사

# ===== 6단계: 모니터링 설정 (340-430줄) =====
# [선택] GPU 전력 모니터링 (nvidia-smi 기반)
# [선택] GPU 텔레메트리 수집
# [선택] 메모리 사용량 모니터링
# [선택] 행(Hang) 감지 모니터 (학습이 멈추면 자동 트레이스백)
# [선택] CUDA 코어 덤프 설정 (크래시 시 디버깅)

# ===== 7단계: 학습 실행 (430-510줄) =====
# 각 실험(NEXP)마다:
#   - 환경변수 캡처 (container-env-{JOB_ID}.log)
#   - 파일시스템 캐시 클리어 (drop_caches)
#   - srun으로 run_and_time.sh 실행
#   - --container-env로 환경변수 컨테이너에 전달  ← 매우 중요!

# ===== 8단계: 후처리 (510-551줄) =====
# MLPerf 규정 준수 체크 (compliance_checker.py)
# CUDA 코어 덤프 처리
# 성능 분석 (선택)
```

### 4.4 run_and_time.sh — "컨테이너 내부 실행기" (188줄)

```
역할: 컨테이너 안에서 실제 학습을 시작하는 스크립트
      run.sub에 의해 각 GPU에서 호출됨
```

```bash
# 1. PyTorch 분산 환경 검증
#    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR 등 확인
#    SLURM에 의존하지 않는 독립적 검증

# 2. 체크포인트 로드 설정
#    → LOAD_CHECKPOINT이 있으면 재개 학습
#    → SHARE_RERUNS=1이면 이전 실험의 체크포인트 공유

# 3. 데이터 모드 설정
#    → train_only_c4: 학습 데이터만 사용
#    → synthetic: 합성 데이터 사용 (테스트용)
#    → benchmark: 벤치마크용 전체 데이터

# 4. 프로파일링 설정 (선택)
#    → nsys로 CUDA/NVTX 트레이스
#    → GPU 메트릭 수집

# 5. 학습 실행
#    → Multi-GPU: python -u pretrain.py
#    → Single-GPU: torchrun --nproc_per_node
#    → CPU 바인딩 (bindpcie)으로 NUMA 최적화
```

### 4.5 config_common.sh — "기본 설정 사전" (97줄)

```
역할: 모든 모델과 하드웨어에 공통으로 적용되는 기본 설정
      다른 설정 파일에서 이 값을 덮어쓸 수 있음
```

**카테고리별 설정:**

```bash
# === NCCL 통신 최적화 ===
NCCL_MIN_NCHANNELS=2         # 최소 NCCL 채널 (메모리 절약)
NCCL_P2P_NET_CHUNKSIZE=2097152  # P2P 전송 청크 크기 (2MB)
NCCL_NVLS_ENABLE=0           # NVLS 비활성 (메모리 절약)
NCCL_WORK_FIFO_DEPTH=1048576 # 작업 큐 깊이
NCCL_MIN_CTAS=16             # 최소 CTA (CUDA Thread Array)
NCCL_MAX_CTAS=32             # 최대 CTA

# === 기울기/파라미터 최적화 ===
DEFER_EMBEDDING_WGRAD_COMPUTE=True  # 임베딩 기울기 지연 계산
WGRAD_DEFERRAL_LIMIT=50            # 지연 한도 (50 반복)
OVERLAP_GRAD_REDUCE=True           # 기울기 통신과 연산 오버랩
USE_DIST_OPTIMIZER=True            # 분산 옵티마이저 사용

# === Transformer Engine ===
NVTE_FWD_LAYERNORM_SM_MARGIN=16    # LayerNorm SM 여유
NVTE_BWD_LAYERNORM_SM_MARGIN=16    # 역방향 SM 여유
TE_UB_ATOMIC_GEMM_RS=0            # Atomic GEMM 비활성

# === 메모리 관리 ===
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_DEVICE_MAX_CONNECTIONS=1
```

### 4.6 config_common_8b.sh — "8B 모델 전용 설정" (32줄)

```
역할: LLaMA 3.1 8B 모델에 특화된 설정값
```

```bash
MODEL_SIZE="8b"              # 모델 크기 식별자
TARGET_LOG_PPL="3.3"         # 목표 log perplexity (405B는 5.6)
OVERWRITTEN_NUM_LAYERS=32    # 레이어 수 (405B는 126)
MAX_STEPS=1200000            # 최대 학습 스텝
MODEL_TFLOP_PER_SAMPLE=421.59  # 샘플당 TFLOP (성능 계산용)
USE_TE_OPS=True              # TE 연산자 사용 (기본값)
CE_FUSION_IMPL=te            # TE 방식 Cross-Entropy 퓨전
FP8_HYBRID=True              # FP8 하이브리드 모드
BUCKET_SIZE=768000000        # 기울기 버킷 크기 (768MB)
```

### 4.7 config_PCIeH100_TTA.sh — "우리의 핵심 설정" (47줄)

```
역할: PCIe H100 2GPU 환경에 맞는 최종 오버라이드
      이전 설정 파일의 값을 이 파일에서 최종 결정
```

```bash
# 이 파일이 가장 마지막에 로드되므로 여기 값이 최종 적용됨
# 주요 변경:
# - LR=0.0004 (레퍼런스 기본값과 다름)
# - TE=False (수렴 문제로 비활성)
# - FP8=False (TE 필요, 같이 비활성)
# - CUDA Graph=0 (비활성)
# - 모든 통신 오버랩 비활성 (PCIe 대역폭 부족)
```

### 4.8 conf/custom.yaml — "동적 설정 계산기" (212줄)

```
역할: 환경변수에서 값을 읽어 복잡한 설정값을 자동 계산
      OmegaConf의 리졸버 기능으로 수식 계산 가능
```

```yaml
# 글로벌 배치 사이즈 자동 계산:
global_batch_size: ${MINIBS} × ${DGXNGPU} × ${DGXNNODES} / (TP × PP × CP)

# 학습률 자동 계산:
lr: (0.00008 × GBS) / 1152

# 검증 주기 자동 계산:
skip_evals: floor(0.0026 × GBS + 12)
val_check_interval: skip_evals × ceil(18432 / GBS)

# 검증 배치 수 자동 계산:
limit_val_batches: ceil(VAL_SAMPLES / GBS)
```

### 4.9 conf/nccl/custom_communicator_cta.yaml — "GPU 통신 미세 튜닝"

```
역할: 통신 종류별로 NCCL 리소스를 다르게 할당
      데이터 병렬, 텐서 병렬, 파이프라인 병렬 각각에 최적 설정
```

```yaml
# DP 통신: 넉넉하게 (CGA=4, maxCTA=8)
# TP 통신: 넉넉하게 (CGA=4, maxCTA=8)  → 자주 사용하므로
# PP 통신: 최소한 (CGA=2, maxCTA=2)    → 드물게 사용
# CP 통신: 최소한 (CGA=1, maxCTA=1)
```

### 4.10 Dockerfile — "프로덕션 컨테이너" (레퍼런스보다 훨씬 복잡)

```
역할: 프로덕션 수준의 최적화된 소프트웨어 스택 구축
```

```dockerfile
# 베이스: NVIDIA PyTorch 25.09 (레퍼런스는 NeMo 24.12)
FROM nvcr.io/nvidia/pytorch:25.09-py3

# 주요 차이점:
# 1. cuDNN 9.13.1.26 (최신 버전)
# 2. Transformer Engine v2.8 (H100 SM90 전용 빌드)
#    → NVTE_CUDA_ARCHS="90" (Hopper 아키텍처만)
#    → MPI 지원 활성화 (NVTE_UB_WITH_MPI=1)
# 3. NeMo 25.09-alpha (레퍼런스는 24.12)
# 4. Megatron-LM 25.09-alpha
# 5. 커스텀 임베딩 라이브러리 빌드
# 6. PyTorch 체크포인트 패치 적용
```

### 4.11 mocking.py — "가짜 데이터 생성기"

```
역할: 워밍업과 테스트에 사용할 가짜(Mock) 데이터 모듈
      실제 데이터 없이도 학습 루프를 실행할 수 있게 해줌
```

### 4.12 embedding_lib/ — "커스텀 고속 임베딩"

```
역할: 표준 PyTorch 임베딩보다 빠른 커스텀 임베딩 구현
      Tensor Parallel 환경에서의 통신 패턴 최적화
```

---

## 5. 핵심 차이점 1: 대상 모델과 하드웨어

### 가장 근본적인 차이

```
레퍼런스:  405B 모델  ×  2,304 GPU  (초대규모)
Dell:     8B 모델    ×  2 GPU      (소규모)
```

이 차이가 모든 설계 결정에 영향을 미칩니다:

| 항목 | 레퍼런스 (405B) | Dell (8B) |
|------|----------------|-----------|
| **파라미터 수** | 4,050억 개 | 80억 개 (~50배 작음) |
| **레이어 수** | 126 | 32 |
| **히든 크기** | 16,384 | 4,096 |
| **FFN 크기** | 53,248 | 14,336 |
| **어텐션 헤드** | 128 | 32 |
| **KV 헤드** | 8 (GQA) | 32 (MHA) |
| **목표 log_ppl** | 5.6 | 3.3 |
| **GPU 수** | 2,304 | 2 |
| **노드 수** | 288 | 1 |
| **GBS** | 1,152 | 8 |
| **연결 방식** | NVLink + InfiniBand | PCIe Gen5 |

### 왜 다른 설정이 필요한가?

```
405B 모델 + 2304 GPU:
  - 모델이 1개 GPU에 안 들어감 → PP(파이프라인 병렬) 필수
  - 노드 간 통신 빈번 → InfiniBand 고속 네트워크 필수
  - 배치가 크므로 → 통신 오버랩이 효과적

8B 모델 + 2 GPU:
  - 모델이 TP=2로 분할 가능 → PP 불필요
  - 노드 내 통신만 → PCIe로 충분 (하지만 느림)
  - 배치가 작으므로 → 통신 오버랩 효과 미미
```

---

## 6. 핵심 차이점 2: 실행 방식 (NeMo-Run vs SLURM 직접)

### 레퍼런스: NeMo-Run (Python 기반 오케스트레이션)

```
사용자 PC에서:
  $ python pretrain_llama31.py --host cluster.example.com ...
      │
      ├─ NeMo-Run이 SSH로 클러스터에 연결
      ├─ SLURM 작업을 Python으로 생성/제출
      ├─ 코드를 git archive로 패키징
      └─ 원격 클러스터에서 학습 시작
```

**장점**: 로컬 PC에서 원격 클러스터 작업을 코드로 관리
**단점**: NeMo-Run 의존성, SSH 터널 설정 필요

### Dell: SLURM 직접 제출 (셸 스크립트 기반)

```
클러스터 노드에서:
  $ source config_PCIeH100_TTA.sh
  $ sbatch run.sub
      │
      ├─ SLURM이 GPU 노드 할당
      ├─ Pyxis가 Docker 컨테이너 생성
      ├─ srun이 각 GPU에서 run_and_time.sh 실행
      └─ run_and_time.sh가 pretrain.py 실행
```

**장점**: 직접적이고 투명한 제어, 디버깅 용이
**단점**: 클러스터에 직접 접속해야 함

### 실행 구조 비교

```
레퍼런스 구조 (간단):
  run_llama31.sh → pretrain_llama31.py (NeMo-Run) → SLURM 작업

Dell 구조 (계층적):
  sbatch run.sub → srun run_and_time.sh → python pretrain.py
       ↑                    ↑                      ↑
  SLURM 오케스트레이션  컨테이너 내부 설정      학습 실행
  (551줄)              (188줄)                (762줄)
```

---

## 7. 핵심 차이점 3: 학습 스크립트 구조

### 레퍼런스: "하나의 파일에 모든 것"

```python
# pretrain_llama31.py 하나에:
#   - SLURM 실행기 설정
#   - 모델 레시피 정의
#   - 데이터 설정
#   - 명령줄 인자 파싱
#   - 멀티 실험 관리
#   - 체크포인트 관리
#   → 모든 것이 한 파일 (537줄)
```

### Dell: "역할별로 분리"

```
pretrain.py       (762줄) — 모델, 데이터, 학습 로직
custom_callbacks.py (850줄) — 콜백, 모니터링, 워밍업
mocking.py         (100줄) — Mock 데이터
run_and_time.sh    (188줄) — 환경 설정, 프로파일링
run.sub            (551줄) — SLURM 오케스트레이션
conf/custom.yaml   (212줄) — 동적 설정 계산
config_*.sh        (여러개) — 환경 변수 설정
→ 총 2,500줄+ (레퍼런스의 ~5배)
```

### pretrain.py 핵심 로직 비교

```python
# ===== 레퍼런스: 모델 생성 =====
if args.size == "405b":
    config = Llama31Config405B(seq_length=8192)
    # 하드코딩된 설정값들

# ===== Dell: 모델 생성 =====
base_config_name = os.environ.get("BASE_CONFIG", "8b")
# 환경변수에서 읽고, YAML에서 계산하고, 런타임에 결정
# FP8, FP4, CUDA Graph, 어텐션 백엔드 등 수십 개 옵션
```

---

## 8. 핵심 차이점 4: 설정 시스템

### 레퍼런스: 단순한 플랫 구조

```
config.sh (1개 파일)
    │
    └─ 모든 변수가 한 곳에 (빈 값, 사용자가 채움)
```

### Dell: 계층적 오버라이드 구조

```
config_common.sh        (1층: 공통 기본값)
    ↓ 덮어쓰기
config_common_8b.sh     (2층: 8B 모델 전용)
    ↓ 덮어쓰기
config_common_cg.sh     (3층: CUDA Graph 설정)
    ↓ 덮어쓰기
config_common_fp8attn.sh (4층: FP8 Attention 설정)
    ↓ 덮어쓰기
config_PCIeH100_TTA.sh  (5층: ★ 최종 하드웨어 특화)

+ conf/custom.yaml      (YAML 계층: 동적 계산)
+ conf/nccl/*.yaml      (NCCL 통신 설정)
+ conf/tp_overlap/*.yaml (TP 오버랩 설정)
```

**왜 이렇게 복잡한가?**

Dell은 다양한 하드웨어를 지원해야 합니다:
- DGX H100 (NVLink)
- PCIe H100 (우리 환경)
- DGX H200
- 다양한 노드/GPU 수 조합

공통 부분은 아래 층에, 하드웨어별 차이는 위 층에 두는 설계입니다.

```
예시: TRANSFORMER_ENGINE 변수의 여정

config_common.sh:        (설정 없음, 기본값)
config_common_8b.sh:     USE_TE_OPS=True       ← 8B 기본은 TE 사용
config_PCIeH100_TTA.sh:  USE_TE_OPS=False      ← PCIe에서는 비활성으로 덮어쓰기
```

---

## 9. 핵심 차이점 5: 소프트웨어 스택 버전

```
                    레퍼런스                  Dell
                    ─────────                ─────────
베이스 이미지        nemo:24.12-rc0           pytorch:25.09-py3
NeMo               24.12                    25.09-alpha
Megatron-Core      24.12 (내장)              25.09-alpha
Transformer Engine  (내장)                    v2.8 (직접 빌드)
PyTorch            2.4                      25.09 (2.6+)
CUDA               12.5                     13.0
cuDNN              (내장)                    9.13.1.26 (직접 설치)
transformers       4.47.1                   제거됨
MLPerf logging     4.1.0-rc3               5.1.0
```

### 버전 차이의 의미

```
Dell의 소프트웨어가 6개월+ 더 최신입니다.
이는 다음을 의미합니다:

1. 더 최적화된 CUDA 커널 (CUDA 13.0 vs 12.5)
2. 더 나은 메모리 관리 (PyTorch 2.6+)
3. 더 많은 최적화 옵션 (TE v2.8의 새 기능들)
4. H100 전용 최적화 (SM90 전용 빌드)
```

---

## 10. 핵심 차이점 6: 성능 최적화

### 레퍼런스: 기본적인 최적화만

```
레퍼런스에 있는 최적화:
✅ 분산 옵티마이저
✅ 기울기 통신 오버랩
✅ 파라미터 수집 오버랩
✅ 텐서 병렬 통신 오버랩 (H100용)
✅ FP32 기울기 축소
❌ CUDA Graph
❌ 워밍업
❌ FP8/FP4
❌ NCCL 미세 튜닝
❌ 커스텀 임베딩
❌ 전력/메모리 모니터링
```

### Dell: 하드웨어 특화 최적화 풀세트

```
Dell에 추가된 최적화:
✅ CUDA Graph (전체 반복 / 레이어별)
✅ 합성 데이터 워밍업 (GPU 메모리 사전 할당)
✅ FP8 하이브리드 (가중치/활성화 별도 정밀도)
✅ FP4 양자화 지원
✅ NCCL 커뮤니케이터별 CTA 미세 튜닝
✅ 커스텀 고속 임베딩 (TP 최적화)
✅ TP 통신 오버랩 프로파일 (GPU별 YAML)
✅ 전력 모니터링
✅ 메모리 프로파일링
✅ 행(Hang) 감지 모니터
✅ CUDA 코어 덤프 자동 처리
✅ 임베딩 기울기 지연 계산
✅ 확장 가능한 CUDA 메모리 세그먼트
```

### 우리 환경 (PCIe H100 2GPU)에서의 실제 설정

```
대부분의 Dell 최적화가 비활성:

❌ CUDA Graph      → PCIe에서 효과 미미
❌ FP8/FP4         → TE 비활성으로 사용 불가
❌ TP 통신 오버랩   → PCIe 대역폭 부족
❌ 커스텀 임베딩    → TE 비활성으로 불필요
❌ TE (Transformer Engine) → 수렴 문제로 비활성

✅ 분산 옵티마이저  → 메모리 절약
✅ 기울기 통신 오버랩 → 약간의 속도 향상
✅ 확장 가능 메모리  → OOM 방지
✅ LR=0.0004 튜닝   → 수렴 스텝 15.9% 감소
```

---

## 11. 핵심 차이점 7: 콜백과 로깅

### 레퍼런스: 규정 준수 중심 (244줄)

```python
callbacks.py에 포함된 것:
├── MLLogger          — MLPerf 로그 기록
├── PreemptiveStop    — 조기 종료
├── MetricsLogger     — 검증 Loss 추적
└── MLPerfCallback    — MLPerf 이벤트 (시작/종료/검증)
```

### Dell: 프로덕션 모니터링 풀세트 (850줄+)

```python
custom_callbacks.py에 포함된 것:
├── DeltaTimingCallback    — 스텝 정밀 타이밍
├── CustomCallback         — 워밍업, FP8 관리, CUDA Graph 관리
│   ├── 합성 데이터 워밍업 (2스텝)
│   ├── FP8 통계 리셋
│   ├── 커스텀 임베딩 hook 등록
│   ├── CUDA Graph 해제 관리
│   └── 검증 주기 동적 조정
├── MetricsLogger          — MLPerf 로깅 + 처리량 계산
├── MemoryProfileCallback  — GPU/CPU 메모리 추적
├── PrintArtifacts         — 설정 덤프
└── 헬퍼 함수들
    ├── run_training_warmup()
    ├── reset_fp8_state()
    └── custom_embed_forward()
```

### 워밍업의 차이 (Dell만 있음)

```
레퍼런스: 바로 학습 시작
  → 첫 스텝이 느림 (CUDA 커널 컴파일, 메모리 할당)
  → MLPerf 시간 측정에 포함됨

Dell: 합성 데이터로 워밍업 후 학습
  run_start 전:
    1. Mock 데이터로 Forward 2스텝
    2. Mock 데이터로 Validation 2스텝
    3. FP8 통계 리셋
    4. 가비지 컬렉션
  run_start 후:
    → 진짜 학습 시작 (이미 GPU가 "워밍업" 됨)
    → 첫 스텝부터 최적 속도
```

---

## 12. 핵심 차이점 8: 데이터 처리

### 레퍼런스

```
데이터셋: C4 (Common Crawl)
분할: 8개 학습 샤드, 1개 검증 파일
토크나이저: Mixtral 8x22B (32,000 어휘)
검증 샘플: 91,205개 (5,760 시퀀스)
```

### Dell

```
데이터셋: C4 (동일)
분할: 동일한 구조
토크나이저: LLaMA 3.1 전용 (128,256 어휘)  ← 4배 큰 어휘!
검증 샘플: 1,024개 시퀀스                    ← 더 적음
학습 샘플: 1,574,207,408개
```

### 토크나이저 차이의 의미

```
Mixtral 토크나이저 (32K 어휘):
  "안녕하세요" → [토큰1, 토큰2, 토큰3]  (3개 토큰)

LLaMA 3.1 토크나이저 (128K 어휘):
  "안녕하세요" → [토큰1, 토큰2]  (2개 토큰, 더 효율적)

어휘가 4배 크면:
  + 같은 텍스트를 더 적은 토큰으로 표현 (효율적)
  - 임베딩 레이어가 4배 커짐 (메모리 사용 증가)
  - Output 레이어(LM Head)도 4배 커짐
```

---

## 13. 핵심 차이점 9: 체크포인트 관리

### 레퍼런스: 멀티 파티션 지원

```
긴 학습을 여러 SLURM 작업으로 분할하는 구조:

작업 1 (파티션 1):
  스텝 0 → 스텝 400,000 → 체크포인트 저장 (5.2TB!)

작업 2 (파티션 2):
  체크포인트 로드 → 스텝 400,001 → 스텝 800,000 → 저장

작업 3 (파티션 3):
  체크포인트 로드 → 스텝 800,001 → 수렴까지
```

이 구조가 필요한 이유: 405B 모델은 **수 주**가 걸리므로
하나의 SLURM 작업(보통 최대 4시간)으로 완료 불가

### Dell: 단일 작업 + 재개 지원

```
기본적으로 하나의 작업에서 완료:
  스텝 0 → ... → 수렴 (37~48시간)

체크포인트 재개도 지원하지만 멀티 파티션은 불필요:
  - 8B 모델은 2일 이내 완료
  - WALLTIME=3600분(60시간)으로 충분
```

---

## 14. 설정 파라미터 전체 비교표

### 모델 설정

| 파라미터 | 레퍼런스 (405B) | Dell (8B) | 비고 |
|---------|----------------|-----------|------|
| 모델 크기 | 405B | 8B | 50배 차이 |
| 레이어 수 | 126 | 32 | |
| 히든 크기 | 16,384 | 4,096 | |
| FFN 크기 | 53,248 | 14,336 | |
| 어텐션 헤드 | 128 | 32 | |
| KV 헤드 | 8 (GQA) | 32 (MHA) | 어텐션 구조 다름 |
| 시퀀스 길이 | 8,192 | 8,192 | 동일 |
| 어휘 크기 | 32,000 | 128,256 | 4배 차이 |

### 하드웨어 설정

| 파라미터 | 레퍼런스 | Dell | 비고 |
|---------|---------|------|------|
| 노드 수 | 288 | 1 | |
| GPU/노드 | 8 | 2 | |
| 총 GPU | 2,304 | 2 | 1,152배 차이 |
| GPU 종류 | DGX H100 | PCIe H100 NVL | |
| GPU 메모리 | 80GB | 93GB | NVL이 더 큼 |
| 연결 | NVLink + IB | PCIe Gen5 | |

### 병렬화 설정

| 파라미터 | 레퍼런스 | Dell | 비고 |
|---------|---------|------|------|
| TP | 8 | 2 | |
| PP | 7 | 1 | |
| VP (가상 파이프라인) | 7 | 없음 | |
| CP | 2 | 1 | |
| DP | ~20 | 1 | |
| SP | 미명시 | False | |

### 학습 하이퍼파라미터

| 파라미터 | 레퍼런스 | Dell | 비고 |
|---------|---------|------|------|
| GBS | 1,152 | 8 | 144배 차이 |
| MBS | 1 | 1 | 동일 |
| LR (max) | (GBS/1152)×8e-5 ≈ 8e-5 | 0.0004 | Dell이 5배 높음 |
| LR (min) | 8e-7 | LR/10 = 4e-5 | |
| Warmup 스텝 | ~8,000 | 128 | |
| 최대 스텝 | ~1,200,000 | 1,200,000 | |
| 옵티마이저 | AdamW | AdamW | 동일 |
| β1, β2 | 0.9, 0.95 | 0.9, 0.95 | 동일 |
| Weight Decay | 0.1 | 0.1 | 동일 |
| Grad Clip | 1.0 | 1.0 | 동일 |
| 정밀도 | BF16 | BF16 | 동일 |

### 검증 설정

| 파라미터 | 레퍼런스 | Dell | 비고 |
|---------|---------|------|------|
| 목표 | log_ppl ≤ 5.6 | log_ppl ≤ 3.3 | 8B가 더 엄격 |
| 검증 주기 | 18,432 시퀀스마다 | 384 스텝마다 | |
| 검증 샘플 | 5,760 시퀀스 | 1,024 시퀀스 | |

### 최적화 설정

| 파라미터 | 레퍼런스 | Dell (PCIe) | 비고 |
|---------|---------|-------------|------|
| Transformer Engine | 사용 | **비활성** | 수렴 문제 |
| FP8 | 미사용 | 비활성 | TE 필요 |
| CUDA Graph | 미사용 | 비활성 | |
| TP 통신 오버랩 | 활성 | 비활성 | PCIe 부적합 |
| 기울기 오버랩 | 활성 | 활성 | |
| 분산 옵티마이저 | 미명시 | 활성 | |
| 워밍업 | 없음 | 합성 데이터 2스텝 | |
| NCCL 튜닝 | 기본 | CTA별 미세 튜닝 | |

---

## 15. 실행 흐름 비교

### 레퍼런스 실행 흐름

```
사용자:  source config.sh && bash run_llama31.sh
            │
            ▼
run_llama31.sh:  변수 검증 → 마운트 구성 → Python 실행
            │
            ▼
pretrain_llama31.py:
  │
  ├─ NeMo-Run 실행기 설정 (SSH + SLURM)
  ├─ 데이터 인덱스 빌드 (첫 실행)
  ├─ 체크포인트 로드 (있으면)
  │
  └─ [학습 루프]
       ├─ Forward → Loss → Backward → 업데이트
       ├─ 매 18,432 시퀀스: 검증
       └─ log_ppl ≤ 5.6 → 종료
```

### Dell 실행 흐름

```
사용자:  source config_PCIeH100_TTA.sh && sbatch run.sub
            │
            ▼
run.sub (SLURM):
  ├─ 호스트 파일 생성 (토폴로지 정렬)
  ├─ 컨테이너 초기화 (Pyxis + Enroot)
  ├─ 네트워크 설정 (MASTER_ADDR, NCCL)
  ├─ [선택] NCCL 테스트
  ├─ [선택] 모니터링 시작
  ├─ 캐시 클리어
  │
  └─ srun run_and_time.sh (각 GPU에서)
       │
       ▼
run_and_time.sh (컨테이너 내부):
  ├─ 환경 검증
  ├─ 체크포인트 설정
  ├─ 프로파일링 설정
  │
  └─ python pretrain.py
       │
       ├─ YAML 설정 로드 (custom.yaml)
       ├─ 모델 생성 (Llama31Config8B)
       ├─ 옵티마이저 설정
       ├─ 분산 전략 설정 (Megatron)
       │
       ├─ [워밍업] Mock 데이터 2스텝
       │
       └─ [학습 루프]
            ├─ Forward → Loss → Backward → 업데이트
            ├─ 매 384 스텝: 검증
            └─ log_ppl ≤ 3.3 → 종료

       └─ run.sub로 복귀
            ├─ MLPerf 규정 준수 체크
            └─ 로그 저장
```

---

## 16. 요약: 한눈에 보는 차이점

```
┌─────────────────────────────────────────────────────────────────────┐
│                        핵심 차이 요약                                │
├──────────────────┬──────────────────────┬──────────────────────────┤
│     항목          │   레퍼런스 (MLPerf)   │    Dell (현재 프로젝트)    │
├──────────────────┼──────────────────────┼──────────────────────────┤
│ 목적             │ "이렇게 하면 됩니다"   │ "이 하드웨어에서 최적"     │
│ 대상 모델         │ 405B (초대규모)       │ 8B (소규모)               │
│ 대상 하드웨어     │ 2304 GPU (DGX)       │ 2 GPU (PCIe H100)        │
│ 실행 방식         │ NeMo-Run (Python)    │ SLURM 직접 (Shell)       │
│ 설정 구조         │ 1개 파일             │ 7개+ 계층적 파일          │
│ 코드 복잡도       │ ~800줄              │ ~2,500줄+               │
│ 최적화 수준       │ 기본                 │ 하드웨어 특화             │
│ 워밍업            │ 없음                │ 합성 데이터 2스텝         │
│ 모니터링          │ MLPerf 로그만        │ 전력/메모리/행감지 등     │
│ 소프트웨어        │ NeMo 24.12          │ NeMo 25.09              │
│ 컨테이너          │ 간단 Dockerfile      │ 복잡한 멀티스테이지       │
│ 체크포인트        │ 멀티 파티션 지원      │ 단일 작업 완료            │
│ 토크나이저        │ Mixtral (32K)       │ LLaMA 3.1 (128K)        │
│ GBS              │ 1,152               │ 8                       │
│ 목표             │ log_ppl ≤ 5.6       │ log_ppl ≤ 3.3           │
│ 최고 기록         │ (공식 제출 참조)      │ 37.9시간 (20,352스텝)    │
└──────────────────┴──────────────────────┴──────────────────────────┘
```

### 한 문장 요약

> **레퍼런스**는 "405B 모델을 2,304개 GPU에서 학습하는 표준 레시피"이고,
> **Dell 구현체**는 "8B 모델을 2개 PCIe H100 GPU에서 최대한 빠르게 학습하기 위해
> NVIDIA의 프로덕션 최적화를 적용하고 하드웨어에 맞게 세밀하게 튜닝한 코드"입니다.

---

> **문서 작성일**: 2026-02-22
> **레퍼런스 경로**: `/mlstorage/training/large_language_model_pretraining/nemo/`
> **Dell 구현체 경로**: `/mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo/`
