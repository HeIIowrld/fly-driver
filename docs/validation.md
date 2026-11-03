# 로컬 검증 기록

2026-10-05, Windows, Python 3.12.13, CPU Torch 환경에서 실행했다. 설치 버전: numpy 2.5.3, gymnasium 1.3.0, mujoco 3.14.0, torch 2.14.1+cpu, mecar-flybrain 0.1.0. 가상환경은 저장소 밖 로컬 디스크의 `%LOCALAPPDATA%/MECar-FlyBrain/.venv`에 만들었다. 기존 저장소의 환경·코드·결과를 실행에 사용하지 않았다.

`ruff check`를 통과했고 `pytest -q`는 **12 passed**였다. 3D 진행거리와 frame의 직교성, ±8% 일정 경사의 무구동 내리막 이동, 조향/좌우 추진력 부호, 관측/reset/time-limit 계약, 실제 두 카메라와 frame 재사용, 뉴런별 입·출력 차수 보존, GRU/graph 지속 상태·reset·detach·시퀀스 replay, oracle mask, PPO timeout bootstrap, tanh log probability의 수치 안정성을 확인했다. OpenGL 렌더링 테스트가 포함되어 있다.

합성 경사·bank 노면의 O 진단 rollout은 100 Hz로 300 step을 실행했다. 종료 사유는 `rollout_budget`이며 완주 판정이 아니다. 진행량은 약 3.70 m, 마지막 횡오차는 약 -0.0116 m였다. 20 step의 V 고정 의도 카메라 smoke도 완료됐다.

| 학습 경로 | 환경 interaction | 파라미터 변화 L2 | 실행 결과 |
|---|---:|---:|---|
| GRU / O | 128 | 0.166988 | completed |
| 합성 graph / O | 128 | 0.118394 | completed |
| GRU / V | 32 | 0.104376 | completed |

모든 PPO loss/gradient는 유한했고 checkpoint를 저장했다. GRU와 합성 graph checkpoint의 100 step 결정적 평가도 실행했다. V checkpoint의 짧은 평가로 sensor-only 저장/읽기 경로를 확인했다. 이 표는 optimizer·정보 계약·실행 경로의 smoke이며 학습 성능·완주율·커넥톰 우월성 결과가 아니다. 시드 1 하나와 매우 짧은 rollout만 사용했다.

초기 run은 Git 첫 커밋 전에 실행했다. manifest의 실제 소스/config SHA-256을 기준으로 상태를 구분한다. 학습 run은 environment의 전체 설정, graph를 사용한 경우 npz/metadata, checkpoint hash, loss·interaction·종료 이유를 남겼다. `runs/`의 결과와 checkpoints는 Git에서 제외되므로 새 clone에는 코드·설정·이 기록만 포함된다.

| 로컬 증거 위치 | 용도 |
|---|---|
| `runs/20261005-3d-oracle/` | 3D O rollout과 sensor/control/truth trace |
| `runs/camera-smoke/` | V camera 샘플 npz |
| `runs/20261005-final-gru/` | 최종 GRU PPO smoke |
| `runs/20261005-final-graph/` | 최종 합성 graph PPO smoke와 graph definition |
| `runs/20261005-final-vision/` | 최종 V 관측 PPO smoke |
| `runs/20261005-final-gru-eval/` | GRU checkpoint 재읽기 평가 |
| `runs/20261005-final-graph-eval/` | graph checkpoint 재읽기 평가 |
| `runs/20261005-final-vision-eval/` | V checkpoint 재읽기 평가 |

T0 전체의 하중·제동·접촉 수렴 검증, 실제 차량/타이어 보정, sensor fault, 실제 fly graph 포트, BC/DAgger, 실측 서킷, 다중 시드 비교, target hardware 지연과 실차 검증은 아직 수행하지 않았다. 계획과 상태는 `implementation-status.md`에서 확인한다.

## CUDA·병렬 실행 확장

2026-10-05 원격 요청 이후에는 CPU/CUDA 선택, 중간 checkpoint, Slurm lifecycle, spawn worker와 batched recurrent PPO를 추가했다. **20개 테스트가 통과**했다. GRU/graph의 배치 시퀀스가 독립 worker별 계산과 일치하는지, 다른 worker를 유지하는 partial reset, 실제 별도 PID 두 개의 PPO와 checkpoint, CUDA 요청의 CPU fallback 차단, stage 파일 hash를 검증했다. 로컬 테스트는 CPU이며 원격 GPU 실행의 검증 결과로 표시하지 않는다.

Slurm GPU 작업 `262880`을 제출했다. 제출 직후 상태는 `PENDING (Priority)`로 GPU 학습은 아직 시작하지 않았다. 설정은 V/GRU, 환경 8개, 32,768 transition, seed 3001이며 GPU 배정 후 CUDA/EGL gate를 실행한다. 상태/결과는 `runs/remote-parallel-gru-partition1-20261005/`의 영수증·watcher·회수 archive로 확인한다. 이 날짜의 GPU 학습 성능 결과는 아직 없다.

## 긴 평가와 후속 파일럿

추가 요청 뒤 기존 V checkpoint를 40초 상한·다섯 조건으로 평가했다. 기존
evaluation이 학습 설정의 64 tick 상한에 묶인 점을 수정한 새 suite를 사용했다.
완주 0/5, 전 조건 track_exit였다. A/B/C/D 네 조건을 CPU 2 worker, seed 3001,
각 2048 transition으로 실행했다. 모든 학습·평가 경로는 완료됐으나 학습 후 완주는
모두 0/3이고 평균 진행량도 초기보다 줄었다. 성능 개선을 주장하지 않는다.

pytest **25 passed in 16.16s**, ruff와 Git diff 검사를 통과했다. nonzero boundary
GRU/graph replay, optimizer update 사이 episode/hidden 유지, 평가의 독립 예산,
감각 validity 제거와 같은 seed/case 비교를 검증했다. 실제 연속 pilot에서 1024 tick
동안 환경을 유지했고 587 tick에서 끝난 track_exit도 학습 데이터에 포함됐다.

GPU 후속 캠페인 `262946`은 제출·watcher 확인을 완료했으며 Dependency 대기다.
근거·설정·로컬 수치·판정 규칙은 [후속 실험](followup-experiments.md)에 있다.

정적 렌더 표식/3D segment 재사용 뒤 고정 300 tick O/V의 센서·영상·물리·reward
해시가 일치했다. 로컬 단일 V 처리량은 226.51→625.03 step/s였으며 GPU 학습 처리량은
아직 미측정이다. pytest **28 passed in 12.90s**, ruff와 Bash 구문 검사를 통과했다.
조건/시드 array mapping, 동시 GPU 상한, GPU 없는 집계, 실제 shard 학습/평가와
누락 작업 실패 보존을 추가 검증했다. 후속 작업은 array 264771 + 집계 264772로
대체했다. 기존 미실행 262946의 회수·검증·원격 소유 tree 삭제도 완료했다.

## 시각·좌우 토크·교사 학습 확장

2026-10-05. pytest **45 passed in 23.75s**, ruff와 Git diff 검사를 통과했다.
3-action/inactive likelihood, 토크 한도·변화율·물리 부호, ray/motion timestamp·mask,
legacy/new checkpoint, recurrent BC padding과 DAgger 실행/정답 분리, A/B/C warmstart
가중치 일치, 기록 유무의 retina_v1 tick trace 동일성을 검증했다.

주행 재생의 직접 점검에서 extent-scaled near clipping 1.265 m가 가까운 노면을
잘라내는 결함을 발견했다. 명시적인 meter clipping과 앞쪽 장착을 추가했고 동일 물리
상태에서 source pixel 복원과 m 단위를 검사했다. 이전 pilot의 실패는 보존했다.
수정된 F의 BC/DAgger/PPO pilot도 완주 0/7로 주행 개선 모델을 채택하지 않았다.
교사 자체는 5/5 지형과 2/2 복귀 평가를 완주했다. 이는 카메라 정책과 다른 결과다.
정량 결과·실패 근거·영상·후속 GPU 설정은 [시각·TV 구현](vision-tv-implementation.md)에 있다.

## RX 9070 XT 로컬 피드백

pytest **52 passed in 36.64s**, ruff를 통과했다. 명령 입력 제거의 qpos/qvel/ctrl
동일성, 선발의 실패/누락/두 시드 검증, 추가 원격 config의 긴 예산·독립 평가 유지,
로컬 파일 회수 hash, expert 재사용의 환경/데이터 거절과 센서 변환, spawn entrypoint의
PyTorch import 제거를 검사했다. AMD 공식 torch 2.9.1/ROCm 7.2 WSL에서 CNN·GRU
역전파와 실제 4/8-worker PPO, BC/DAgger 이후 PPO를 실행했다. 학습 모델의 주행
성공과 runtime 연결 성공은 별개다. [로컬 기록](local-rdna4.md)에 수치/실패를 남겼다.
