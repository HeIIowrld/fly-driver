# 로컬 후보 선별과 원격 본학습

2026-10-05 사용자가 요청한 구성이다. 로컬 RX 9070 XT는 짧은 실험으로 조건을
비교하고, 두 시드에서 재현된 상위 후보는 원격에서 새 모델·새 시드로 더 오래
학습한다. 기존 원격 7조건×2시드 실험의 수와 학습 예산은 유지한다.

## 단계와 예산

| 단계 | 조건·시드 | PPO transition/조건·시드 | 자원 |
|---|---|---:|---|
| 로컬 1차 | 네 조건, 4101 | 16,384 | RX 9070 XT, CPU 환경 4개 |
| 로컬 재검사 | 기준+통과 후보, 4102 | 16,384 | 동일 설정 |
| 원격 본학습 | 기준+상위 최대 2후보, 6101–6103 | 262,144 | task당 GPU 1개, 환경 8개 |

본학습은 8환경×256 tick×128 update이며 로컬 PPO의 16배다. 최대 3조건×3시드,
9개 GPU task를 구성한다. 각 task의 GPU gate, 원격 BC·DAgger 수집/학습, PPO,
initial/final 평가를 독립 실행한다. 모델 초기화·optimizer는 새 시드에서 시작한다.
로컬 체크포인트를 계속 학습하는 구성과 구별하며, 로컬 데이터 경로는 원격 입력에서
제거한다. 기존의 별도 7개 원격 평가 case와 센서/구동/노면 계약은 유지한다.

선별은 원래 선언한 기준을 유지한다. 기준보다 실패가 늘지 않고 평균 진행거리가
25% 이상 증가하며 네 개발 case 중 세 개 이상에서 개선되어야 한다. 다른 로컬
시드에서도 통과한 후보 중 **두 시드의 상대 개선율 중 작은 값**이 큰 순서로
상위 두 개를 올린다. 한 시드만 유난히 잘 나온 조건을 우선하지 않는다. 통과
후보가 없으면 추가 본학습을 제출하지 않는다. 짧은 pilot 탈락이 큰 예산에서도
실패할 것이라는 증거는 아니므로 기존 원격 탐색 실험을 취소하지 않는다.

시각·몸 감각·이전 명령 입력을 그대로 사용하고 actor에 도로 정답이나 지형 기울기
truth를 추가하지 않는다. 높이·경사·뱅킹은 기존 합성 3D 노면과 몸 센서에서 반영한다.
측정 커넥톰이나 실제 뉘르부르크링 주행 모델로 표시하지 않는다.

## 병렬화와 큐

18:46 KST에 계정 `ug/normal`의 사용자 상한이 GPU 2개, CPU 32개, 메모리 128 GB임을
조회했다. `partition1`은 hpc-stat1의 GPU 5개를 갖지만 현재 계정이 동시에 사용할 수
있는 수는 2개다. 3시간, CPU 8개, 24 GB, GPU 1개/task, array 동시 실행 2개의
`sbatch --test-only` 검사를 통과했다.

원격 array는 `%2`로 설정해 자원이 배정되면 조건·시드 두 개가 동시에 학습한다.
각 task 안에서는 환경 8개의 관측을 GPU CNN·GRU·optimizer가 배치 계산한다.
물리는 CPU MuJoCo다. 여러 GPU가 하나의 모델 gradient를 합치는 distributed PPO는
현재 구현되지 않았고, GPU 물리로 256개 환경을 동시에 계산한다고 주장하지 않는다.
여러 array의 상한을 각각 2로 잡아도 계정 전체 GPU 2개 상한은 Slurm이 적용한다.

기존 array 265295도 재제출 없이 `ArrayTaskThrottle=2`로 조정한다. 입력·소스·예산,
14개 task, 기존 job ID/접수 시점을 보존하며 CPU 집계 265296은 전체 array 종료를
기다린다. 조정 영수증은 해당 run의 `scheduler-adjustments.json`에 전후 상태로 남긴다.
큐가 `Priority` 대기인 동안 즉시 두 GPU가 시작되는 것은 아니다. 3시간은 wall-time
상한이며 timeout/누락/실패 task도 회수·집계한다.

[Slurm array 문서](https://slurm.schedmd.com/job_array.html)와
[ArrayTaskThrottle 공식 설명](https://slurm.schedmd.com/scontrol.html#OPT_ArrayTaskThrottle)
참조. 현재 권한 밖의 다른 partition이나 다른 사용자 작업은 변경하지 않는다.

## 코드와 자동 진행

정책은 `configs/local-to-remote-main.json`에 예산·새 시드·후보 수·scheduler 설정으로
저장한다. `scripts.prepare_remote_main`은 복제 검증과 시드 중복을 검사하고,
평가/센서 설정을 유지해 본학습 config를 만든다. 기존 WSL 학습과 followup은
봉인된 소스를 유지한다. Windows collector를 새 정책으로 갱신해 그 followup이
만드는 통과 제안을 본학습 입력으로 변환한다.

```powershell
python -m scripts.watch_local_screening --source <WSL-mirror> --output <existing-collection> `
  --root <repo> --wheels <Linux-wheels> --auto-submit --resume `
  --promotion-policy configs/local-to-remote-main.json `
  --preflight-output <collected-remote-probe>
```

`--resume`은 원격 제출 계획이 생성되기 전의 기존 collector에만 허용한다. 새 collector는
이미 회수된 조건을 기억하고, 학습 worker나 WSL 유지 연결은 종료하지 않는다.
선별 후 source SHA-256 일치와 깨끗한 Git 상태를 확인하고 소스/input을 봉인해 제출한다.
원격 watcher를 동시에 시작해 종료한 task의 원본·로그·checkpoint·replay를 검증·회수한다.

정책이 적용된 `main-training-config.json`, 후보 순위, policy hash, 제출 영수증과
실패를 포함한 집계 결과로 자동 진행 여부를 확인한다. 본학습의 실제 시작과 완료는
Slurm 상태와 결과 파일로 확인하며, 계획을 학습 완료로 표시하지 않는다.

## 10월 5일 재검증과 원격 오류

로컬 두 시드 4101/4102의 평균 진행거리는 기준 `full_mixed` 3.799/4.013m,
`full_student` 26.422/26.288m, `no_copy_student` 22.856/18.479m였다.
두 학생 주행 DAgger 조건이 선별 규칙을 통과했다. 모든 평가가 코스 이탈로
끝났으므로 자율주행 성공이나 초파리 회로의 효과를 입증한 결과는 아니다.

기존 원격 array 265295는 첫 task의 런타임 설치 시간 초과와 나머지 task의
EGL 초기화 오류로 종료됐다. 설치된 의존성 overlay는 재사용 캐시에 남아 있다.
실패 로그·입력·소스는 검증된 `batch-00-results.tar.gz`로 회수했다.

`SLURM_JOB_GPUS` 번호를 `MUJOCO_EGL_DEVICE_ID`로 그대로 사용하지 않는다.
`select_remote_egl`이 CUDA device 0의 UUID와 EGL의 `EGL_NV_device_cuda` 장치를
대조한다. 일치하는 장치가 없거나 여러 개면 종료하며 다른 GPU/CPU 렌더러로
자동 전환하지 않는다. 실제 NVIDIA GL renderer와 카메라 영상, sensor-only 입력,
GPU 모델·optimizer를 별도 probe에서 확인한다. GPU 권한·드라이버 문제의 해결은
probe가 통과하기 전까지 미확인 상태다.

`--preflight-output`을 지정하면 collector는 로컬 근거를 회수한 뒤 원격 probe의
회수 완료를 기다린다. archive SHA-256, 봉인된 소스, 현재 actor/렌더링 코드,
job 종료 코드, CUDA/EGL UUID, 실제 GL vendor를 검증한 뒤 본학습을 제출한다.
실패하면 제출하지 않고 heartbeat에 오류를 기록한다. 원격 대기 중에는 WSL의
완료된 학습 프로세스를 유지할 필요가 없으며 검증된 NAS manifest를 사용한다.

[MuJoCo EGL 장치 선택 구현](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/egl/__init__.py),
[Khronos EGL/CUDA 장치 대응 규격](https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt).

10월 6일 확인한 probe 266341은 1분 19초 만에 정상 완료했다. RTX 6000 Ada의
CUDA UUID와 EGL index 0이 일치했고 실제 NVIDIA renderer, 영상 표준편차 39.374,
GPU 모델·optimizer, oracle validity 0을 확인했다. Slurm의 전역 GPU 번호는 1이었다.
따라서 번호를 직접 연결하던 카메라 초기화 문제는 수정본에서 해결됐다.

자동 제출 collector는 대기 중 NAS lifecycle 파일의 일시적 읽기 거부로 중단됐다.
SMB 읽기 거부·파일 미도착·live JSON 작성 중에는 미확인 상태로 재시도한다.
봉인된 입력·소스의 hash 불일치와 실패한 GPU 검증은 계속 제출을 차단한다.
`--resume --collected-only`는 이미 회수된 파일 전체를 SHA-256으로 다시 검증하고
WSL 원본 접근 없이 제출을 재개한다. 원래 collection 영수증은 보존한다.
