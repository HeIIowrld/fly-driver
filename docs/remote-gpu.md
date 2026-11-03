# 원격 GPU 초기 학습

이 저장소에서 독립적으로 stage·SSH 업로드·Slurm 제출·로컬 watcher를 실행한다. 운영 helper 세 개(`slurm_credentials`, `slurm_ephemeral_remote`, `manage_slurm_ephemeral`)는 2026-10-05 검토한 기존 MECar 운영 도구를 가져왔다. 다른 저장소의 Python 모듈을 런타임에 import하지 않는다.

`MECAR_SSH_*`는 프로세스 환경 또는 Windows 사용자 환경에서 읽고 기존 known_hosts를 검증한다. 비밀번호를 stage·CLI 인자·Git·영수증에 쓰지 않는다. 목적지는 `/data1/dsl01/campaigns/.mecar-flybrain-*`의 새 소유 디렉터리 하나다. compute 노드는 `/mnt/data1/` mount를 사용한다. Python은 Slurm allocation에서 실행한다.

기존 CUDA Torch 2.5.1 + cu121 / Python 3.11 런타임은 읽기 전용으로 사용한다. MuJoCo/Gymnasium의 고정 Linux wheel은 PyPI 공식 파일을 서버에서 받고 로컬 봉인 manifest의 SHA-256과 대조해 package cache에 저장한다. 설치는 새 해시 기반 overlay에서 오프라인으로 수행하며 기존 공유 환경을 갱신하지 않는다. `--include-wheels`를 주면 wheel을 직접 업로드한다. CUDA cache는 런타임과 GPU/driver 조합으로 구분하고 run 정리 밖에 보존한다. 원격 버전은 로컬 Torch 2.14.1과 다르므로 각 버전을 기록한다.

초기 설정은 `configs/train-remote-gru.json`: V sensor-only, 합성 3D 코스, 좌우 32×32 RGB / 30 Hz, 3축 IMU·wheel·steer·suspension, seed 3001, **환경 8개 × 256 tick × 16 update = 32,768 transition**, PPO epoch 4, checkpoint 2 update마다 저장이다. 시작 X를 3–80 m에서 뽑는다. GRU와 optimizer는 CUDA, 물리는 8개 CPU worker, headless 렌더링은 할당된 GPU의 EGL이다. 코드 점검용 4-step CUDA/영상 gate를 통과해야 실제 학습을 시작한다. GPU 사용을 요구한 경우 CPU fallback은 없다.

[Gymnasium AsyncVectorEnv](https://gymnasium.farama.org/api/vector/async_vector_env/)를 spawn·공유 관측 메모리로 구성한다. 관측은 N개씩 GPU에 전달하고, PPO는 T×N 시퀀스를 배치로 계산한다. 종료 관측을 먼저 bootstrap한 뒤 종료한 worker만 reset한다. 숨은 상태와 GAE의 episode 경계도 worker별로 처리한다. 물리 자체를 GPU 벡터 연산으로 바꾼 MJX 구현은 아니다. policy/environment/optimizer 단계별 시간과 실제 transitions/s를 남긴다.

전체 코스 완주나 실제 커넥톰 비교가 아닌 초기 학습이다. collector는 update마다 재시작하므로 긴 랩 기억을 평가하지 않는다. 평가 모델은 CPU로 재읽고 V 관측을 유지한다. BC/DAgger·실차 보정·실제 커넥톰 포트·장거리 학습은 별도 단계다.

```powershell
python -m pip install -e ".[physics,learning,remote]"
python -m scripts.remote_gpu launch --config configs/train-remote-gru.json --wheels <Linux-cp311-wheel-folder> --output runs/<new-campaign>
python -m scripts.remote_gpu status --output runs/<campaign>
```

기본 Slurm 요청은 계정에서 사용 가능한 `partition1`의 `hpc-stat1`, GPU 1개·CPU 8개·24 GB·최대 3시간이다. 2026-10-05 확인에서 여유 GPU가 있는 `jobs/hpc`는 현재 계정의 association과 일치하지 않아 제출이 거절됐다. 다른 site는 경로·계정·Python·cache와 endpoint를 검토해 바꿔야 한다. 현재 helper의 endpoint는 기존 222/223/224 구성에 맞춰져 있다. 사용 중인 다른 작업을 중지하거나 그 allocation을 사용하지 않는다.

제출과 함께 숨김 local watcher를 시작한다. `collection/lifecycle-status.json`이 heartbeat, `watcher.log`가 회수 로그다. 모든 작업이 종료되면 실패한 작업까지 source·input·log·결과를 archive로 다운로드한다. 압축 SHA-256과 embedded manifest를 확인하고 세 노드의 job/process 참조를 다시 확인한 다음 **그 campaign 디렉터리만** 삭제한다. 전송/검증 실패 시 원격 자료를 보존하고 재시도한다. shared cache는 삭제하지 않는다. watcher 재개는 같은 plan과 output으로 한다.

```powershell
python -m scripts.manage_slurm_ephemeral --plan runs/<campaign>/collection-plan.json --output runs/<campaign>/collection --watch --detach
```

`submission.json`, `stage-manifest.json`, GPU/런타임 기록, `progress.json`, `checkpoint.pt`, `training.json`, evaluation, transfer/verification/deletion receipts를 남긴다. 실제 캠페인 job ID·GPU·진행 상태는 제출 뒤 상태 기록에 추가한다.

2026-10-05의 첫 병렬 GPU 캠페인은 **Slurm 262880**, source revision `0fdd385`, `runs/remote-parallel-gru-partition1-20261005/`다. 요청은 RTX 6000 Ada 노드의 GPU 1개·CPU 8개·24 GB·3시간이며, 제출 직후 `PENDING (Priority)`로 확인했다. watcher PID 17532를 시작했고 `collection/lifecycle-status.json`의 대기 heartbeat를 확인했다. GPU 계산의 시작·성능은 실제 `gpu-probe.json`과 `training/device.json`, `progress.json`이 생성된 뒤 판정한다. 문서의 대기 상태는 제출 시점 기록이다.

단일 환경 패키지의 중단된 업로드와 권한이 맞지 않는 `jobs/hpc` 제출 실패는 별도 run에 남겼고, 실패 증거 회수·해시 검증·원격 소유 tree 정리를 수행했다. 패키지 설치 cache는 보존한다.

## 후속 비교 캠페인

[결과 점검과 후속 실험](followup-experiments.md)에 근거와 로컬 결과를 남겼다.
`configs/campaign-followup-gpu.json`을 `--mode campaign --after-job 262880`으로 제출했다.
Slurm **262946**, source `315abae`, `runs/remote-followup-ablation-20261005/`다.
확인 시점 상태는 **PENDING(Dependency)**이며 watcher PID 792의 heartbeat를 확인했다.
기존 작업 종료 뒤 4개 조건 × 2개 시드의 학습과 initial/final 다중 지형 평가를
순차 실행한다. 전체 학습 예산은 524,288 transition이다. 실제 실행 상태는
`output/campaign/progress.json`, 완료 결과는 `summary.json`·`result.json`과 각 조건의
training/evaluation 파일로 판단한다. 제출 시점의 대기 상태를 실행 완료로 해석하지 않는다.

```powershell
python -m scripts.remote_gpu status --output runs/remote-followup-ablation-20261005
```

10월 5일 10:14에 미실행 후속 작업 262946을 정리하고 **array 264771 / CPU 집계
264772**로 대체했다. 새 source는 `c85d903`, `runs/remote-followup-array-20261005/`다.
`--mode campaign-array --node auto --time 00:30:00`으로 제출했으며 GPU 동시 사용은
1개다. 원래 smoke 262880의 상한은 20분으로 줄였고 array는 그 출력에 의존하지 않는다.
각 task에 별도 gate/log/checkpoint/evaluation을 저장한다. GPU가 필요 없는 집계는
array 종료 뒤 실행하며 실패/누락 task도 기록한다. watcher는 array와 집계 모두
종료할 때까지 자료를 보존한다. [변경 근거와 검증](performance-and-queue.md).

```powershell
python -m scripts.remote_gpu status --output runs/remote-followup-array-20261005
```

## 시각·좌우 토크·몸 감각 후속 캠페인

2026-10-05 14:14 KST, source **c1aee42**의 **array 265295 / CPU 집계 265296**을 제출했다.
`configs/campaign-vision-tv-gpu.json`, 7조건×2시드=14 task, task별 30분·GPU 1개·CPU 8개·
24 GB, GPU 동시 사용 1개다. expert BC→mixed DAgger→continuous PPO 뒤 initial/final
독립 평가와 동기화 주행 재생을 저장한다. 전체 PPO budget은 917,504 transition이다.
MuJoCo worker 물리는 CPU, CNN/GRU/optimizer와 EGL 카메라는 배정된 GPU를 사용한다.
첫 상태는 array `PENDING(Priority)`, 집계 `PENDING(Dependency)`이며 CUDA 계산은 미시작이다.
봉인 64개 파일/hash와 watcher PID 23324의 heartbeat를 확인했다.

자료는 `runs/remote-vision-tv-array-20261005/`에 있다. 영상에서 발견한 near clipping
결함과 45개 테스트·로컬 실패 결과·교사 특권 분리는 [시각·TV 구현](vision-tv-implementation.md)에 있다.
앞선 262880/264771/264772는 새 제출 뒤 소유/경로/PENDING을 검증하고 취소했다.
실행 시간 0초를 확인했고 기존 watcher가 두 owned tree의 회수·source 검증·삭제를 완료했다.
로컬 archive와 전송/검증/삭제 영수증, shared cache는 보존했다.

```powershell
python -m scripts.remote_gpu status --output runs/remote-vision-tv-array-20261005
```
