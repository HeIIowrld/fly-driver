# GPU 활용과 처리량 개선

2026-10-05 RX 9070 XT, 공식 PyTorch 2.9.1/ROCm 7.2, WSL에서 검사했다.
정책과 optimizer는 이미 GPU에서 계산한다. 카메라가 CPU `llvmpipe` renderer를
사용해 learner에 다음 관측을 늦게 전달하는 것이 현재 주된 병목이다.

## 실제 측정

광각 두 눈, source 96×96 → angular_hex 32×32, RGB+운동 응답, 카메라 25 Hz,
몸 감각과 제어 100 Hz, 물리 1000 Hz, residual 좌우 토크 제어를 유지했다.
고정 명령 120 tick에서 EGL 환경 계산 8.84초 중 카메라 렌더링·샘플링은 8.68초,
물리 step은 0.045초, forward는 0.004초였다. 이 위치와 명령에서 렌더링이 98.2%다.
모든 주행 상태의 비율이라고 일반화하지 않는다.

D3D12 경로에서 실제 GL renderer는 `D3D12 (AMD Radeon RX 9070 XT)`였다.
설정마다 4환경×64 tick×3 update = 768 PPO transition을 실행하고 첫 update를
제외했다. 학습 중인 기존 로컬 캠페인이 같은 PC를 쓰는 상태에서 AB/BA 순서로
반복했으므로 아래는 공유 호스트의 짧은 처리량 검사다.

| 실행 순서 | CPU 카메라 + GPU 학습 | GPU 카메라 + GPU 학습 | 처리량 증가 |
|---|---:|---:|---:|
| EGL → D3D12 | 34.74 transition/s | 42.36 transition/s | 22.0% |
| D3D12 → EGL | 38.99 transition/s | 45.81 transition/s | 17.5% |

고정 명령에서 qpos/qvel/ctrl은 두 경로에서 **정확히 일치**했다. 0–255 RGB의
평균 절대 차이는 0.0636이어서 renderer 간 화상은 완전히 같지는 않다.
두 경로 모두 optimizer가 실제 파라미터를 바꾸고 유한한 PPO loss를 기록했다.
PyTorch peak allocation은 각각 242,745,856 byte, 약 231.5 MiB였으며 GL 메모리와
같은 시간 다른 학습 프로세스가 사용하는 메모리는 이 수치에 포함되지 않는다.
WSL 전체 GPU busy 비율은 직접 검증하지 않았다. 처리량 개선을 GPU 사용률
퍼센트 상승이나 주행 성능 개선으로 해석하지 않는다.

증거는 `runs/20261005-gpu-acceleration/renderer-ab`와 `renderer-ba`에 있다.
config/source hash, 실제 renderer, 단계별 시간, 고정 명령 trace/영상 배열,
checkpoint, 실행 로그를 회수하고 파일 hash를 확인한다. 원격의 기존 14개 task와
진행 중인 로컬 네 조건의 소스·renderer·학습 예산을 바꾸지 않았다.

## 적용 순서

1. **후속 로컬 실행에서 GPU 카메라 사용.** 아래 launcher는 실제 GPU renderer와
   GPU PPO, 물리 상태 일치 gate를 통과한 뒤 새 실험을 실행한다. 실패하면 로그를
   남기고 종료한다. 현재 진행 중인 비교 실험의 renderer를 중간에 바꾸지 않는다.
2. **카메라 배치 처리와 전송 감소.** 현재는 각 환경의 두 눈을 따로 렌더링한 뒤
   CPU 배열로 읽고 GPU에 복사한다. 한 번에 두 눈/여러 환경을 렌더링하고
   angular sampling·운동 필터·CNN까지 GPU에 두는 경로를 별도 구현/검증한다.
   카메라 exposure 25 Hz와 몸 감각·recurrent update 100 Hz는 유지한다.
   카메라 frame hold 동안 GRU를 건너뛰면 다른 제어 모델이 된다.
3. **GPU 배치 시뮬레이션.** 원격 NVIDIA에서 MuJoCo Warp의 GPU 물리 및
   batch renderer를 검사하고 64→128→256 환경으로 늘리며 transition/s와 VRAM을
   함께 측정한다. 데이터 왕복을 없애는 구조 변경이며 현재 구현되지는 않았다.
   16 GB 메모리에 몇 개가 들어갈지는 차량/heightfield/영상 버퍼로 실측해야 한다.
4. **학습 계산 튜닝.** 데이터 공급 개선 후 CNN의 mixed precision, 정적 shape의
   compile/graph capture, 더 큰 recurrent sequence batch를 비교한다. 동일 transition,
   update 수, 학습 seed/평가에서 속도와 주행 성능을 함께 판단한다. 단순히 epoch나
   모델 크기를 늘리면 GPU busy는 늘 수 있지만 샘플 효율 개선의 근거가 되지 않는다.

CPU 물리가 이 probe에서는 매우 작아서 RX에서 새 근사 차량 물리 모델을 만드는
일은 현재 우선순위가 낮다. 워커 증가만으로는 기존 시험에서 GPU 4환경 대비
8환경 처리량이 약 12% 늘었고 CPU·RAM 부담이 증가했다.

[MuJoCo Warp 공식 문서](https://mujoco.readthedocs.io/en/3.10.0/mjwarp/)는 NVIDIA CUDA
GPU에서 배치 시뮬레이션을 제공한다. RX 9070 XT에 그대로 적용할 수는 없다.
[MJX 공식 문서](https://mujoco.readthedocs.io/en/stable/mjx.html)는 MJX-JAX의 AMD
지원과 기능 제약을 설명한다. 현재 차량의 `CYLINDER` 휠과 `HFIELD` 지형 접촉은
MJX-JAX 지원 밖에 있어 같은 물리를 유지한 즉시 전환은 어렵다. 접촉 형상 변경은
별도 물리 검증을 요구한다. [AMD ROCm 7.2 WSL 지원표](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)의
공식 PyTorch 2.9.1과 FP32/FP16 지원 범위 안에서 학습 튜닝을 먼저 진행한다.

## 재현

새 WSL mirror에서 동일 비교를 실행한다. 각 case의 전용 process group만
180초 timeout 시 종료하며, 다른 실험 프로세스를 대상으로 하지 않는다.

```bash
export PYTHONPATH="$PWD/src"
/home/hj/.venvs/mecar-flybrain-rocm72/bin/python \
  -m scripts.benchmark_acceleration --config configs/campaign-rdna4-screening.json \
  --output runs/new-renderer-ab --timeout-seconds 180
```

후속 로컬 캠페인은 다음과 같이 시작할 수 있다. 기본 launcher는 EGL을 유지하며
이 경로는 GPU 렌더링을 명시적으로 선택한다.

```bash
export MECAR_RDNA4_PYTHON=/home/hj/.venvs/mecar-flybrain-rocm72/bin/python
bash scripts/run_local_gpu_render.sh configs/campaign-rdna4-screening.json runs/new-gpu-camera
```
