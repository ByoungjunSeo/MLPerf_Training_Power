# 서버 재시작 후 작업 재개 가이드

**백업 일자**: 2026-06-26
**최종 상태**: ✅ **MLPerf 5.1 RCP test PASSED (10/10 runs converged)**
**다음 단계**: MLPerf 공식 제출 또는 추가 최적화 실험

---

## 🎯 현 상태 한 줄 요약

10회 BS=32 RCP-eligible run 모두 컨버전스 성공 → **RCP test PASSED (submission mean 216,576 vs reference mean 215,040, +0.71%)**. MLPerf 5.1 공식 제출 가능 상태.

---

## 📂 백업 파일 구조

```
/mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo_NVLINK/
└── BACKUP_20260626/
    ├── RESUME_GUIDE.md           ← 이 파일
    ├── results_logs/             ← 10개 컨버전스 run 로그 + compliance + RCP 결과
    │   ├── 260606033539...log    (Run #1 = Job 206)
    │   ├── compliance_2606...out (10개)
    │   ├── ... (총 20개 파일)
    │   └── rcp_checker_final_output.log  ← RCP 통과 증거
    ├── configs/                  ← 모든 config_*.sh + conf/ 디렉토리
    ├── docs/                     ← 모든 분석/회의 문서
    │   ├── OPTIMIZATION_LOG.md
    │   ├── OPTIMIZATION_LOG.csv
    │   ├── NVLINK_BRIDGE_ANALYSIS.md
    │   └── MEETING_NOTE_RCP_10RUN_PROGRESS.md
    └── claude_session/           ← Claude 대화 전체 이력
        ├── conversation_full.jsonl  (3.7MB, 6주간 전 대화)
        ├── MEMORY.md
        └── memory/               ← 누적된 영구 메모리
```

> ⚠️ **/lustre/agent/mltraining_v5.1_llama3_8b/log/ 의 원본도 같이 백업 권장**
> 만약 lustre가 같이 꺼진다면 BACKUP_20260626 안의 사본이 유일한 사본이 됨.

---

## 🚀 재개 시나리오별 절차

### A. 단순 결과 확인만 (가장 흔한 경우)

```bash
cd /mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo_NVLINK/BACKUP_20260626

# RCP 통과 확인
cat results_logs/rcp_checker_final_output.log | tail -5

# 10개 run 결과 요약
cat docs/MEETING_NOTE_RCP_10RUN_PROGRESS.md
```

→ Claude 실행 불필요.

### B. Claude로 작업 이어 가기 (재최적화/추가 실험)

#### 1단계: 시스템 hardware 정상 복구 확인
```bash
nvidia-smi                          # GPU 2개 보여야 함
nvidia-smi nvlink -s                # NV12 link 살아있는지
systemctl status slurmctld slurmd   # SLURM 데몬 정상인지
```

만약 slurmctld FAILED:
```bash
sudo systemctl reset-failed slurmctld.service
sudo systemctl restart slurmctld.service
sudo systemctl restart slurmd
```
(이전 정전 후 같은 증상 — Claude session 기록에 해결 패턴 있음)

#### 2단계: Claude 실행 위치

```bash
cd /mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo_NVLINK
claude
```

→ 이 디렉토리에서 실행해야 다음이 자동으로 인식됨:
- 기존 Claude project memory (`/home/mlcommons/.claude/projects/-mlstorage-training-results-v5-1-TTA-Claude-benchmarks-llama31-8b-implementations-nemo-NVLINK/`)
- OPTIMIZATION_LOG.md, MEMORY.md 등 작업 컨텍스트

#### 3단계: Claude에게 첫 메시지로 전달할 내용

```
이전 작업을 이어서 진행해야 해. 백업 가이드는
BACKUP_20260626/RESUME_GUIDE.md 에 있고,
지금까지 작업 이력은 OPTIMIZATION_LOG.md 에 있어.

마지막 상태:
- MLPerf llama3.1 8B BS=32 10-run RCP test PASSED (submission mean 216,576)
- NVLink Bridge 효과 step time -8.2% 입증
- 다음에 진행할 작업: <여기에 다음 목표 명시>

먼저 현재 시스템 상태(GPU, NVLink, SLURM)부터 점검해줘.
```

#### 4단계: Claude가 자동으로 할 일
- `MEMORY.md` 자동 로드 → optimization log 작성 규칙 등 인식
- OPTIMIZATION_LOG.md 읽고 #1~#13 실험 컨텍스트 파악
- 시스템 health 점검 후 다음 실험 제안

---

## 🔧 시스템 환경 복구 체크리스트

서버 재시작 직후 확인 항목 (Claude에게 이 항목 점검 요청 가능):

| 항목 | 확인 명령 | 정상 |
|---|---|---|
| GPU | `nvidia-smi` | H100 NVL 2개 보임 |
| NVLink | `nvidia-smi nvlink -s` | 12개 link 활성 (26.562 GB/s) |
| Driver | `nvidia-smi --query-gpu=driver_version --format=csv` | 580.82.07 |
| SLURM ctld | `systemctl is-active slurmctld` | active |
| SLURM d | `systemctl is-active slurmd` | active |
| SLURM 통신 | `squeue` | error 없음 |
| Lustre mount | `ls /lustre/agent/mltraining_v5.1_llama3_8b/` | 데이터 보임 |
| 컨테이너 이미지 | `srun -N1 --container-image=sbj8388/mlperf-nvidia:training5.1_llama31_8b-pyt -- true` | enroot import 성공 |

---

## 🎯 가능한 후속 작업 옵션

### Option 1: MLPerf 공식 제출 준비
- `mlperf_logging.package_checker`로 submission directory 생성
- system description JSON 작성 (UNKNOWN_MLPERF_SUBMITTER 등 채우기)

### Option 2: 추가 성능 최적화
- 다른 BS 시도 (예: BS=64 — 다른 RCP reference)
- NCCL 환경 변수 튜닝
- 다른 random seed로 추가 측정해 NVLink 효과 통계적 검증

### Option 3: 단순 보존/문서화
- BACKUP_20260626 추가 분석
- 회의록 보강

---

## ❗ 잃어버릴 수 있는 것 (재개 후 복구 불가)

서버 꺼지면 다음이 사라짐 — 이미 백업했으니 OK:
- ✅ SLURM 큐 상태 (이미 모두 완료, 잃을 것 없음)
- ✅ 컨테이너 enroot tmp (`/mlstorage/enroot/tmp/`) — 재시작 시 자동 재생성
- ✅ GPU 메모리 (0 MB 상태, 작업 없음)
- ✅ Claude conversation context — 백업했으나 새 session에서는 처음부터

**유일하게 백업 불가능한 것**: Claude의 in-memory conversation context. 새 session 시작 시 OPTIMIZATION_LOG.md + MEMORY.md + RESUME_GUIDE.md로 재구성.

---

## 📊 백업 완료 사항 체크리스트

- [x] 10개 run의 _1.log (training output)
- [x] 10개 compliance_*.out (검증 결과)
- [x] **RCP checker 최종 출력** (`rcp_checker_final_output.log` — PASSED 증거)
- [x] 모든 config 파일 (`config_*.sh` + `conf/` 디렉토리)
- [x] 모든 분석 문서 (4개 .md, 1개 .csv)
- [x] Claude conversation transcript (3.7MB jsonl)
- [x] Claude memory dir

---

## 📞 Claude 재시작 첫 명령 (copy-paste용)

```bash
# 1. 디렉토리 이동
cd /mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo_NVLINK

# 2. 시스템 빠른 점검 (Claude 실행 전)
nvidia-smi | head -15
nvidia-smi nvlink -s | head -5
systemctl is-active slurmctld slurmd
squeue 2>&1 | head -3

# 3. Claude 실행
claude
```

Claude 첫 메시지:
```
서버 재시작 후 작업 재개야. BACKUP_20260626/RESUME_GUIDE.md 읽고
시스템 상태 점검한 후 작업 컨텍스트 요약해줘.
```

---

작성: 2026-06-26
백업 위치: `/mlstorage/training_results_v5.1/TTA_Claude/benchmarks/llama31_8b/implementations/nemo_NVLINK/BACKUP_20260626/`
