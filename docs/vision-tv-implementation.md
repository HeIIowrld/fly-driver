# 시각·좌우 토크 후속 구현과 실험

2026-10-05. 기존 `legacy` 2-action/6-channel CNN checkpoint를 유지하면서 명시적인
`retina_v1` 관측과 3-action 제어 경로를 추가했다. 실제 connectome/해부학적 눈 지도가
아닌 기능 프로토타입이다. 앞선 원격 array 264771의 봉인 소스/조건에는 적용하지 않는다.

## 구동 계약

`environment.actuation_mode`는 `legacy`, `off`, `fixed`, `residual`이다. 마지막 세
조건은 모두 action `[속도 의도, 곡률 의도, 추가 요모멘트 의도]`를 받는다. 기본 속도/곡률
mapping은 기존 계약을 따르며 추가 요모멘트 범위는 예시 ±80 N·m다.

- `off`: 조향과 공통 구동 토크, yaw 차동 보정 0.
- `fixed`: 기존 gyro/encoder 기반 yaw feedback.
- `residual`: 고정 feedback에 정책이 학습한 yaw moment를 추가.

off/fixed에서는 세 번째 action을 0으로 하고 그 차원을 Gaussian likelihood·entropy에서
제외한다. 동작하지 않는 action을 PPO 확률비에 포함해 비교를 왜곡하지 않는다. 세 조건의
모델 크기와 초기화는 동일하다. residual head는 처음 0으로 초기화한다.

새 배분기는 가능한 공통 토크를 먼저 정하고 좌우 차를 배분한다. 좌우 0–160 N·m와
모터별 변화율 예시 600 N·m/s를 지킨다. 실제 motor/tire calibration은 아니다. 발행한
토크로 환산한 yaw moment는 실제 접촉력으로 달성한 yaw moment와 다르다. 명령·한도
작동 여부·측정 속도/gyro·yaw reference를 진단 정보에 기록한다. actor에는 이전에
발행한 토크·행동을 efference copy로 제공한다.

## 시각 계약

CameraConfig로 장착 yaw/pitch·baseline·높이·수직 FOV·렌더링 해상도를 설정한다.
`perspective`는 원래 입력, `angular_rect`는 규칙적인 방향 격자, `angular_hex`는
행을 반 칸씩 어긋나게 배치한 방향 격자다. 두 angular 모드는 perspective 렌더링에서
ray를 역투영/보간한다. **어안 렌즈나 실제 초파리 소안 방향 지도는 아니다.** 광학 중심과
frustum 밖 sample은 mask로 표시하며 wrap하거나 정답으로 대체하지 않는다.

이번 광각 후보는 yaw ±40°, pitch −15°, 수직 FOV 120°, source 96×96,
actor ray grid 32×32, angular span 110°×90°다. 같은 field/span에서 규칙/엇갈림 grid를
비교한다. narrow→wide 대비는 장착각·FOV·source sampling이 함께 바뀌는 **카메라
묶음의 비교**이며 FOV 한 변수만의 효과라고 주장하지 않는다.

retina_v1은 기존 센서 외에 아래 항목을 추가한다.

| 필드 | 의미 |
|---|---|
| retina_valid | 각 눈/위치의 광학적 유효성 |
| motion, motion_valid | ON/OFF transient와 우/좌/위/아래 correlator 응답·유효성 |
| previous_yaw_action | 직전에 실제 활성화해 발행한 추가 yaw 의도 |
| camera_frame_id | 촬영 ID, 노출 사이에는 유지 |

필터는 exposure timestamp 간격을 써 밝기 적응(예시 τ=0.15초)과 지연(τ=0.04초)을
갱신한다. 정적 첫 frame의 운동은 무효이고, episode reset 시 필터 기억도 초기화한다.
상이한 위치의 지연/현재 ON/OFF 곱 차를 방향 응답으로 사용한다. 이 수식은 직접
구현한 근사이며 T4/T5 뉴런이나 flyvis 시각 회로를 구현했다고 부르지 않는다.

new encoder는 RGB+운동 9채널/눈을 항상 할당한다. RGB 대조에서는 운동 채널을 0으로
하여 모델 크기를 맞춘다. 위치/카메라/몸 감각 mask를 적용하고 세 action의 지속 GRU
및 PPO replay를 사용한다. 기존 encoder/checkpoint는 legacy 구조로 복원한다.

25 Hz 카메라·100 Hz 제어·1 kHz 물리로 지정했다. legacy 명목 30 Hz의 실제 25 Hz
clock을 변경하지 않았다. configuration에는 requested/effective rate를 함께 남긴다.

### 영상 확인으로 발견한 clipping 결함

아래의 최초/복귀/DAgger pilot은 기본 renderer clipping을 썼다. 재생 화면에서 광각
원본의 가까운 노면이 검게 잘리는 문제를 발견했다. model extent 126.51 m에 기본
znear=0.01이 곱해져 near plane이 **1.265 m**였다. FOV를 크게 했을 때 하단 노면이
frustum 밖으로 잘렸다. 이 단위 계약은 [MuJoCo visual/map 문서](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-map)의 정의다.

CameraConfig의 `near_m`, `far_m`를 모델 extent로 나눠 렌더러에 설정하고 실제 m 값을
configuration에 기록하도록 수정했다. 두 값이 0이면 이전 checkpoint의 렌더 계약을
유지한다. 새 GPU 조건과 `campaign-vision-tv-camera-local.json`은 **near 0.03 m,
far 250 m, camera forward 0.95 m**다. 카메라를 chassis front 0.85 m 앞에 둬 차체
가림을 줄였다. A–G 모두 같은 장착 forward/clipping을 쓴다.

동일 전방 장착·같은 물리 reset에서 clipping만 비교한 광각 source의 non-black pixel
비율은 **14.99%→56.22%**로 증가했다. black sky가 포함된 픽셀 점검이며 센서 유효성이나
주행 성능 지표는 아니다. reset 물리 상태는 bitwise 같았고 렌더링 검사와 비교 PNG/HTML을
`runs/20261005-camera-clipping-diagnostic/`에 남겼다. frustum validity는 두 경우 100%였으므로
그 mask만 확인해서 clipping 결함을 잡을 수 없었다.

이전 pilot을 삭제/수정하지 않고 clipping 영향을 받은 진단으로 보존했다. 그 실패에서
초파리식 시각이나 TV의 성능을 결론내리지 않는다. 수정된 조건에서 F의 BC/DAgger/PPO를
다시 실행한 기록은 `runs/20261005-vision-tv-camera-pilot/`이다.

수정된 F pilot도 완주 **0/7**이고 BC/DAgger 후 평균 진행 4.53 m, 1024-transition
PPO 후 4.70 m였다. 시각 결함 수정이 주행 성공을 의미하지 않는다. 이번 로컬 run의
startup+warmstart는 216.45초, PPO 구간은 4.61초였다. 마지막 PPO-only 처리량
226.93 transition/s는 로컬 CPU 2 worker의 진단 수치이며 GPU 성능은 미측정이다.

## 교사와 학습

순수 PPO 전에 route/pose truth를 사용하는 pure-pursuit 교사를 검사했다. 기본·crest·
flat·별도 uphill bank·downhill turn의 다섯 지형을 저속에서 **5/5 완주**했다. 최대
횡오차는 0.120 m, uphill은 67.18초였다. 이에 새 정책 평가 상한은 80초로 지정했다.
이는 privileged teacher 결과이며 카메라 정책의 성능이 아니다.

증거: `runs/20261005-vision-tv-teacher-validation/`. 교사 진단은 실제 완주/종료 이유와
pose/command trace를 남긴다. BC 데이터에는 **결정 시점 V 센서와 teacher action label**만
넣는다. route/pose truth를 actor 배열에 저장하지 않고 V oracle 채널/validity는 0이다.
label/image timestamp와 episode 경계를 유지한다. 실패한 교사 episode는 gate에서 거부한다.
교사 기하식의 근거는 [Coulter의 pure-pursuit 보고서](https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm)다.
lookahead·저속 목표·차량 adapter 수치는 이 prototype의 선택이다.

BC는 episode별 연속 GRU 상태와 chunk TBPTT를 사용한다. update 후 같은 chunk를 새
가중치로 재계산해 다음 상태를 이어간다. prefix는 이전 가중치 상태가 남는 근사다.
BC optimizer와 PPO optimizer는 별개다. `random-checkpoint.pt`는 BC 전,
`initial-checkpoint.pt`는 PPO 직전 BC 모델, `checkpoint.pt`는 PPO 후다.

BC 오차가 낮아지는 것과 실제 완주를 구분한다. 중앙선 시퀀스, 복귀 시퀀스, 혼합 주행
DAgger를 차례로 진단했다. 앞의 두 pilot은 expert 주행 데이터만 썼고 DAgger가 아니다.

중앙선 pilot의 실제 결과는 아래와 같다. 모든 조건이 0/5 완주이므로 성능 개선 모델을
채택하지 않았다. 넓은 시야나 TV 기능을 넣었다는 이유로 우월성을 주장하지 않는다.

| 조건 | BC 후 평균 진행(m) | PPO 후 평균 진행(m) | PPO 후 완주 |
|---|---:|---:|---:|
| A TV off/RGB | 23.02 | 23.75 | 0/5 |
| B fixed TV/RGB | 23.03 | 23.75 | 0/5 |
| C residual TV/RGB | 23.03 | 23.72 | 0/5 |
| D 광각 regular | 15.41 | 16.54 | 0/5 |
| E 광각 staggered | 16.36 | 17.77 | 0/5 |
| F 광각 motion | 16.78 | 18.64 | 0/5 |

증거는 `runs/20261005-vision-tv-local-pilot/summary.json`과 각 initial/final evaluation,
teacher NPZ, training, 주행 재생이다. supervised BC loss의 감소만으로 모델을 채택하지
않았다. BC 후 모델이 최종 PPO의 initial이다. 기존의 random GRU 평균과 같은 지표로
비교하지 않는다.

그 결과로 **복귀 데이터 단계**를 추가했다. start X=3/35/65 m에서 중앙선 800 frame씩,
각 위치에서 lateral ±0.6 m/heading ±4°의 복귀 400 frame씩 = 4800 frame이다.
같은 canonical fixed-TV teacher로 수집하고 BC에서는 속도/곡률 두 head만 지도해
A/B/C가 같은 가중치로 PPO를 시작하도록 한다. 그 가중치의 bitwise 일치를 테스트했다.
세 번째 head는 PPO에서만 residual 조건에 활성화한다.

PPO 초기 pose도 lateral ±0.5 m/heading ±4° 범위로 뽑는다. 복귀 평가에는 학습보다 큰
lateral ±0.9 m/heading ±7°를 따로 넣었다. 교사는 그 두 조건도 완주했으며
`runs/20261005-vision-tv-teacher-recovery/`에 기록했다. perturbation은 환경/교사 조건이며
그 정답을 actor에 추가하지 않는다. 이는 교란 상태의 전문가 데이터 보완이고 DAgger는 아니다.

`configs/campaign-vision-tv-recovery-local.json`의 C/F 복귀 pilot도 0/7 완주였다.
최종 평균 진행은 C 5.28 m, F 3.84 m로 중앙선 pilot보다 나빴다. 전문가 입력 분포,
이전 action 복사, episode 순서에 따른 학습 편향은 원인 후보이며 아직 분리 검증하지 않았다.
증거는 `runs/20261005-vision-tv-recovery-pilot/`이다.

이에 새 설정은 episode를 T×N 열로 묶고 매 epoch 열 순서를 seeded shuffle한다.
각 열의 GRU 상태는 chunk 사이 이어가며 padding은 loss에서 제외한다. 한 묶음은 최대
12 episode다. 기존 sequential 방식도 명시적으로 유지했다. supervised frame 예산과
실제 optimizer update 수를 모두 기록하며, 두 batch 방식의 update 수를 같다고 부르지 않는다.

[Ross·Gordon·Bagnell의 DAgger](https://proceedings.mlr.press/v15/ross11a.html)를 참고해
mixed expert/student 주행의 V 입력에 expert label을 붙여 데이터를 집계하는 bounded
round를 구현했다. expert 선택 확률 beta=0.5, 학생은 결정론적 tanh(mean)이며 이전에
실제로 실행한 행동이 다음 관측의 efference copy다. 학생도 모든 결정에서 GRU를 갱신한다.
실패/완주 시 sequence를 나눠 reset하고 수집 decision 예산을 채운다. 실패를 데이터에서
감추지 않는다. `action`은 expert label, `executed_action`은 실행 행동이며
`teacher_selected`, timestamp, student weight hash, 종료 이유를 함께 저장한다.

새 budget은 expert 4800 frame×24 BC epoch, mixed rollout 2400 decision을 추가한
aggregate 7200 frame×16 BC epoch, 그 뒤 PPO다. BC와 DAgger에서는 yaw head를
지도하지 않고 canonical fixed-TV 환경을 써 A/B/C의 동일 초기 가중치를 검증했다.
이 제한된 한 round가 논문의 이론적 수렴 조건을 충족했다고 주장하지 않는다.

`configs/campaign-vision-tv-dagger-local.json`은 C/F bounded pilot,
`configs/campaign-vision-tv-gpu.json`은 clipping 수정과 몸 감각 대조를 포함한 일곱 조건/두 시드다.
BC learning rate는 5e-4, 모델 선택은 마지막 선언된 budget의 checkpoint다.
평가 완주 결과로 checkpoint를 골라 제출하지 않는다.

로컬 balanced BC→DAgger→PPO 결과도 **C 0/7, F 0/7 완주**였다.

| 조건 | BC/DAgger 후 평균 진행(m) | 1024-transition PPO 후(m) | 완주 |
|---|---:|---:|---:|
| C 좁은 RGB/residual TV | 4.56 | 4.49 | 0/7 |
| F 광각 staggered/motion/residual TV | 4.50 | 4.54 | 0/7 |

두 후보의 마지막 supervised MSE는 약 0.0025였다. 실제로 스스로 주행하면 이탈했다.
mixed rollout은 2400 decision 중 expert 1193회/학생 1207회이고 세 segment가 모두
예산까지 진행했지만, expert 없이 평가한 정책은 실패했다. 이 결과로 장착·motion·TV가
성공했다고 주장하지 않는다. 다음에는 beta를 0으로 내리는 추가 DAgger round, efference
copy 의존성 제거 시험, 시각-only/body-only 대조를 별도 예산으로 검토한다. 지금 GPU
캠페인은 같은 선언된 warmstart 뒤 더 긴 PPO에서 일곱 조건의 차이를 확인하는 진단이다.
증거: `runs/20261005-vision-tv-dagger-pilot/summary.json`과 각 dataset/checkpoint/trace.

## 토크의 물리 응답

`scripts/torque_response_probe.py`로 flat 직선과 곡률 0.1/m에서 2초의 저속 주행을
검사했다. residual +0.5는 +40 N·m를 요청한다. baseline 대비 마지막 gyro z의 증가는
직선 0.000398 rad/s, 선회 0.000932 rad/s였고 Y 위치 차이는 각각 1.55 mm, 5.20 mm였다.
음의 residual은 반대 방향이었다. 명령 부호/구동 경로는 확인했지만 회전 효과가 작다.
저속 접촉 모델·토크 가용 한도·타이어 응답의 보정이 필요하며 이것은 실차 yaw gain이 아니다.
`runs/20261005-torque-response-probe/`에 명령·제약·gyro trace/hash를 남겼다.

## 대조와 예산

| 조건 | 구동 | 시각 |
|---|---|---|
| A | TV off | 좁은 RGB, yaw ±25°/FOV 80° |
| B | 고정 TV | 같은 RGB |
| C | learned residual TV | 같은 RGB |
| D | C | 광각 규칙 방향 grid, RGB |
| E | C | 같은 광각 엇갈림 grid, RGB |
| F | C | E의 RGB+운동 응답 |
| G | C | F에서 정책의 측정 몸 감각만 mask |

paired seed에서 A→B, B→C, C→D, D→E, E→F, F→G의 효과를 분리해 기록한다.
G도 하위 제어기는 encoder/gyro로 구동하고 정책의 이전 발행 행동/토크는 유지한다.
이는 정책 몸 감각의 제거 시험이며 차량 전체에서 센서를 제거한 실험은 아니다.
시드와 선언한 감독 decision/epoch 예산을 맞춘다. mixed rollout 실패에 따라 sequence
수와 실제 optimizer update 수가 달라질 수 있어 각 run에 기록한다. 초파리 graph의
우월성 주장은 하지 않는다.
완주/이탈을 우선하며 진행 비율·횡오차·yaw 오차·토크 한도·주행 재생을 함께 본다.

로컬 최초 pilot은 seed 3101, 환경 2개×128 tick×4 update = 1024 PPO transition/조건,
teacher 2400 frame×BC 12 epoch다. 긴 모델 성능 비교가 아닌 실행/설계 진단이다.
GPU 설정은 환경 8개×256 tick×32 update = 65,536 PPO transition/조건이고
두 시드/일곱 조건 = 917,504 transition이다. CUDA CNN/GRU/optimizer와 GPU EGL,
CPU 프로세스 물리를 사용한다. 교사/BC 예산은 별도로 manifest에 남긴다.
전체 wall time/처리량은 보존하고, 새 로그에는 startup+warmstart와 PPO wall time 및
PPO-only transitions/s를 따로 남긴다. 로컬 pilot의 기존 낮은 전체 sps를 GPU 처리량으로
해석하지 않는다.

주행 재생에는 정책 RGB, 96 px 렌더 원본, 운동 응답, 외부 관찰, 센서·토크를 함께
저장한다. raw NPZ의 observer/pose는 진단 전용이다. PNG/HTML/NPZ는 codec 추가 없이
생성하고 hash를 기록한다. checkpoint 복원도 새 action/visual schema에 맞춰 검증한다.

GPU 캠페인은 30분 array task와 GPU 없는 결과 집계를 사용한다. source/config를 새로
봉인하며 watcher가 회수/검증/종료 run 정리를 수행한다. 이전 pending 작업을 학습 결과로
표시하지 않으며 기존 서비스 allocation을 사용하지 않는다.

2026-10-05 14:14 KST, source **c1aee42**를 봉인해 **array 265295 / CPU 집계 265296**을
제출했다. 14 task, 동시 GPU 1개, task별 30분/CPU 8개/24 GB다. 첫 확인은
265295 전체 `PENDING(Priority)`, 집계 `PENDING(Dependency)`였으며 실제 CUDA 학습은
아직 시작하지 않았다. watcher PID 23324의 heartbeat와 봉인 64개 파일/hash를 확인했다.
영수증은 `runs/remote-vision-tv-array-20261005/`에 있다.

이전 262880, 264771, 264772는 같은 기본 clipping을 쓰는 역사적 대기 작업이었다.
새 제출 후 소유자·작업 이름·owned working directory·전 task PENDING을 확인해 취소했다.
Slurm elapsed는 모두 0초였다. 기존 watcher가 두 원격 tree를 회수·압축/embedded source
검증하고 삭제했으며 `batch-00-transfer/verified/deleted.json`과 완료 heartbeat를 확인했다.
로컬 봉인 자료와 실패 기록, 공유 runtime/GPU cache는 보존했다.
