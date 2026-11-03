# RX 9070 XT 로컬 피드백 — 2026-10-05

기존 원격 7조건×2시드 캠페인(array 265295, 집계 265296)은 그대로 유지한다.
로컬은 코드 오류·감각 의존·학습 방법을 빨리 검사하는 추가 실행 경로다.
로컬 결과로 원격 실험 수를 자동 축소하지 않는다. 가능성이 확인된 변경은 별도
소스/config를 봉인해 추가 원격 실험으로 검증한다.

## 실제 실행 환경

WSL2 Ubuntu 24.04, ROCm 7.2.0, gfx1201 RX 9070 XT(16,304 MiB), Python 3.12다.
기존 Windows 프로젝트 venv는 torch CPU 빌드였다. 프로젝트 전용 WSL venv
`/home/hj/.venvs/mecar-flybrain-rocm72`를 만들고 작동하는 기존 ROCm torch 환경을
읽기 전용 `.pth`로 참조했다. 기존 환경에는 설치·삭제를 하지 않았다.
진단 뒤 필요한 의존성도 전용 venv에 설치하고 `.pth` 연결을 해제했다.
현재 torch/numpy/gymnasium 모두 전용 venv 경로에서 로드되며 `pip check`가 통과했다.
추가 MuJoCo/pytest/ruff는 새 venv에만 설치했다. 최초 기존 torch 2.10 빌드는
gate/짧은 PPO를 통과했으나 긴 BC/DAgger 뒤 광각 PPO에서 HSA 동기화에 걸렸다.
소유 실행과 워커를 중단하고 stack/checkpoint/데이터를 실패 진단으로 보존했다.
현재는 전용 venv에 **AMD 배포 torch 2.9.1+rocm7.2.0.git7e1940d4**를 설치했다.
wheel SHA-256은 `199ba10a8b8b5a7eef64985bf87a41f0f4b28d49b1b861aa2768f6c5314c9b86`다.
기존 venv를 uninstall하지 않았다는 pip 기록도 확인했다. numpy 2.4.3의 GPU 왕복
변환과 실제 CNN·GRU 역전파/optimizer를 검사했다.

AMD GPU도 PyTorch의 `device="cuda:0"` API를 사용한다. `device.json`은
`backend="rocm"`, HIP 버전, gfx architecture, 실제 모델 device와 GPU 메모리를
별도로 기록한다. 물리 엔진은 CPU이며 정책/optimizer는 GPU다.
[AMD WSL 지원표](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)와
[WSL 설치 문서](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-pytorch.html)를 참고했다.

기본 WSL GLFW/EGL 렌더러는 Mesa llvmpipe였다. 별도 프로세스에서
`GALLIUM_DRIVER=d3d12`, `MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD`,
`LD_LIBRARY_PATH=/usr/lib/wsl/lib:/opt/rocm/lib`를 적용하자
`D3D12 (AMD Radeon RX 9070 XT)` 렌더링과 GPU 역전파 gate를 통과했다.
GPU 선택 방식은 [Microsoft WSLg 문서](https://github.com/microsoft/wslg/wiki/GPU-selection-in-WSLg)에 있다.

초기 D3D12+기본 커널 탐색의 PPO 조합은 첫 GPU update가 3분 넘게 끝나지
않아 해당 소유 probe만 중단했다. 이후 EGL llvmpipe, renderer thread 1개,
`MIOPEN_FIND_MODE=FAST` 조합에서 짧은 병렬 PPO가 완료됐다. 긴 학습에서 발견한
추가 정체 뒤 torch 버전과 워커 import도 수정했다. 여러 요인이 함께 바뀌었으므로
멈춤 원인을 하나로 확정하지 않는다. 현재 파일럿은 **GPU 학습,
CPU 물리·카메라 렌더링**이다. D3D12 gate 통과를 실제 파일럿 GPU 렌더링으로
표시하지 않는다. FAST의 의미는 [AMD MIOpen 문서](https://rocmdocs.amd.com/projects/MIOpen/en/latest/reference/env_variables.html)에 따른다.

## 처리량과 비교

공식 2.9.1 빌드에서 실제 **광각 RGB+motion/residual TV** 설정으로
128 tick×3 update를 실행하고 첫 update를 제외했다. 측정 때 learner CPU thread는
2개였으며 다른 PPO 설정은 같다. 변경 전 좁은 RGB 측정과 섞지 않는다.

| 정책/optimizer | 환경 수 | warm transition/s |
|---|---:|---:|
| CPU | 4 | 68.98 |
| RX 9070 XT | 4 | 64.31 |
| RX 9070 XT | 8 | 72.29 |

GPU라고 모든 작은 배치가 빨라지는 것은 아니다. 8개 환경은 GPU 4개 대비
1.12배였지만 CPU 사용량이 높았다. 데스크톱 부하를 고려해 **로컬 환경 4개,
learner CPU thread 1개**로 정하고 총 transition은 유지한다. GPU 계산 비중이 작은
정책이며 CPU 물리/렌더링과 IPC가 병목이다. 매번 전체 device를 기다리던 fence를
제거하고 실제 필요한 action/bootstrap의 CPU 전송으로 동기화한다. 최초 compilation,
시뮬레이터 시작, 교사 데이터 수집, BC, PPO 시간을 혼동하지 않는다.

## 네 조건

모두 광각 엇갈림 방향 샘플링, RGB+운동 응답, residual 토크 벡터링,
동일 CNN·GRU와 센서만 사용하는 actor다. 카메라 forward 0.95 m, near 0.03 m,
far 250 m로 이전 clipping 문제를 고친 설정을 유지한다.

| 조건 | 이전 발행 명령 | DAgger 학생 주행 |
|---|---|---|
| full_mixed | 유지 | 교사 선택확률 0.5 |
| no_copy_mixed | 정책 입력에서 제거 | 교사 선택확률 0.5 |
| full_student | 유지 | 교사 선택확률 0 |
| no_copy_student | 정책 입력에서 제거 | 교사 선택확률 0 |

이전 명령 제거는 body 20:24의 값과 validity, previous yaw action을 0으로 한다.
측정 몸 감각과 하위 제어기, 물리적 명령은 유지하며 같은 action에서 qpos/qvel/ctrl
일치를 검사했다. 명령을 읽지 않아도 실제 휠/조향 센서는 자신의 움직임을 감지한다.
DAgger beta 0에서도 expert가 학생이 방문한 상태의 정답 행동을 붙인다.

조건마다 교사 4800 decision×24 BC epoch, DAgger 2400 decision 후 aggregate
7200 sample×16 epoch, PPO 4환경×256 tick×16 update=16,384 transition이다.
1차 seed 4101이며 총 PPO 예산은 65,536 transition이다. DAgger 실패/재시작에
따른 sequence 수와 optimizer update 수는 별도로 남긴다.

원격의 기존 8001–8007 평가와 분리된 개발 검증 8401–8404를 사용한다.
기준보다 실패가 늘지 않고, 평균 진행거리가 25% 이상 증가하며, 4 case 중 3개
이상에서 진행거리가 나아지면 seed 4102로 기준과 후보를 재검사한다. 두 seed가
통과한 변경만 추가 원격 후보로 올린다. 완주하지 못한 후보의 거리 증가는 우선순위
근거일 뿐 자율주행 성공의 근거가 아니다. 통과 후보가 없을 수 있다.

## 실행과 증거

소스는 NAS 저장소에서 WSL의 전용 로컬 mirror로 복사해 실행한다. manifest에는
실행한 src/config 해시를 남기고 source-provenance에 복사 당시 canonical Git 상태를
보존한다. 로컬 실험 중 원격에서 이미 봉인된 c1aee42 입력을 수정하지 않는다.

```bash
export MECAR_RDNA4_PYTHON=/home/hj/.venvs/mecar-flybrain-rocm72/bin/python
bash scripts/run_local_rdna4.sh configs/campaign-rdna4-screening.json runs/new-screen
```

`scripts/probe_local_gpu.py`는 실제 CNN·GRU backward/optimizer와 GL renderer를
확인하고 `scripts/benchmark_learning.py`는 짧은 CPU/GPU 병렬 PPO를 비교한다.
`scripts/select_local_candidates.py`는 실패와 누락 시드도 포함해 선발 근거를 저장한다.
학습 성공 여부는 summary의 paired 검증과 각 checkpoint/trace/주행 replay로 판단한다.

교사 데이터는 학생과 독립이므로 검증된 4800 decision을 재사용한다. 환경 설정,
교사 seed/episode/속도/구동, physics/sensor 코드 해시와 NumPy/MuJoCo 버전을 맞추고
각 NPZ hash를 검증한다. no_copy 조건은 같은 데이터의 명령/validity만 0으로 바꾸며
변환된 NPZ와 원본 manifest/hash를 함께 저장한다. DAgger는 학생마다 다시 수집한다.
따라서 실패한 GPU 실행에서 만든 교사 데이터도 GPU 학습 결과와 구분해 재사용한다.

카메라 렌더링의 추가 병목 검사에서 D3D12 GPU 경로와 실제 PPO가 함께 동작했고,
AB/BA 반복에서 처리량이 17.5–22.0% 증가했다. 단계별 시간과 후속 GPU 렌더링
launcher, 배치 렌더링/시뮬레이션 계획은 [GPU 활용 개선](gpu-utilization.md)에 정리했다.
진행 중인 네 조건의 EGL renderer는 비교 일관성을 위해 유지한다.

메모리 점검 당시 WSL이 정지한 학습 이후 실제 약 0.8 GiB, file cache 약 7.3 GiB를
보고했고 Windows vmmemWSL working set은 약 8.1 GiB였다. 전체 WSL working set을
살아 있는 학습 tensor로 해석하지 않는다. spawned entrypoint에서 learner import를
함수 안으로 옮긴 뒤 CPU worker에 torch가 로드되지 않음을 검증했다. 시작 시 worker
RSS 약 120 MiB/PSS 52–58 MiB였으며 렌더러 초기화 후 사용량은 증가한다.
공식 빌드의 4-worker PPO tensor peak는 약 308 MiB, 이후 BC/DAgger 경로는 약
618 MiB였다. GPU의 작은 정책과 별개로 CPU 물리·렌더러는 CPU를 사용한다.

공식 빌드의 후속 gate는 재사용 expert 4800×8 epoch, DAgger 2400+aggregate 7200×16
epoch 뒤 PPO 1536 transition을 완료했다. 평가 완주 **0/4**로 주행 성공 모델은 아니다.
이 gate는 긴 supervised 학습 다음 PPO가 실제 진행되는지 검사한 것이며 네 조건
선발 결과에 포함하지 않는다. CPU/ROCm 테스트와 실제 주행 검증을 구분한다.

`scripts/finish_local_screening.py`가 네 조건 결과를 기다리고 통과 조건만 seed 4102로
기준과 재검사한다. `scripts/watch_local_screening.py`는 live/BC 상태와 종료한 run을
NAS로 회수하고 SHA-256을 확인한다. 두 시드가 통과하면 기존 원격 캠페인은 유지한
채 기준+후보의 65,536 transition/조건 추가 array를 제출한다. 제출 전 actor source
일치와 깨끗한 Git 상태를 확인하고 기존 remote watcher로 자동 회수한다. 후보가
없거나 검증이 실패하면 추가 제출하지 않는다. 최종 선발은 아직 진행 중이다.

이후 요청에 따라 원격 promotion 정책을 보완했다. collector에
`configs/local-to-remote-main.json`을 적용하면 통과 제안에서 상위 최대 두 후보+기준을
새 시드 6101–6103, 262,144 transition/task의 본학습으로 올리고 GPU task 두 개를
병렬 실행한다. 위 65,536 제안은 봉인된 followup의 원래 출력으로 보존한다.
현재 적용/예산/제출 절차는 [로컬 선별과 원격 본학습](local-to-remote-main.md) 참조.
