# 결과 점검과 후속 실험 — 2026-10-05

좌우 날개의 공통/비대칭 출력을 독립 모터에 연결하는 추가 가설, 카메라의 실제 범위와
동기화 주행 기록, 시각/TV의 대조 순서는 [시각·날개 후속 설계](vision-wing-followup.md)에
추가했다. 아래 원격 A/B/C/D 예산/입력/봉인 소스에는 그 새 정책 출력을 포함하지 않는다.

## 확인한 결과

첫 GPU 작업 262880은 점검 시점에도 PENDING(Priority)이었다. 원격 checkpoint·loss·GPU
처리량 결과는 없다. 이하 개선 근거는 기존 **로컬 V/GRU, seed 1, 32 transition**
checkpoint의 재평가와 코드 점검이다. 원격 학습 결과를 분석한 것으로 표시하지 않는다.

기존 평가 명령은 `--steps`를 늘려도 학습 설정의 `max_steps=64`를 유지했다.
2000 step을 요청한 재평가도 64 step, 0.64초, 0.80 m에서 time_limit으로 끝났다.
새 `evaluate-suite`는 평가 max_steps를 명시적으로 설정한다. 4000 step/40초의 상한으로
실행했으며 실제 충돌/이탈/완주가 발생하면 즉시 종료한다. actor의 V 정보 계약은 유지한다.

| 평가 조건 | 실제 주행 시간(s) | 3D 진행 증가(m) | 종료 |
|---|---:|---:|---|
| 기본 코스, X=3 m | 5.82 | 18.16 | track_exit |
| 기본 코스, X=35 m | 4.03 | 12.46 | track_exit |
| 같은 XY의 평탄화 노면 | 5.78 | 18.57 | track_exit |
| 별도 +6% 경사·역방향 bank | 6.46 | 16.11 | track_exit |
| 별도 -6% 경사·더 큰 굴곡 | 9.83 | 34.07 | track_exit |

완주 0/5, 평균 진행 증가 19.88 m다. 이 짧은 학습 checkpoint로 감각의 유효성이나
GRU/커넥톰 우열을 판단하지 않는다. 결과/trace/manifest는
`runs/20261005-vision-multicase-eval/`에 있다. 각 조건은 합성 노면이다.

## 구현 개선

- `collector_restart=false`: optimizer update를 지나서도 차량, 관측, episode return,
  순환 기억을 유지한다. 실제 episode 종료 worker만 reset한다. rollout당 256 tick은
  optimizer의 시퀀스 길이이며 주행 시간 상한이 아니다.
- 각 시퀀스는 저장한 시작 hidden state에서 replay한다. 그 시작값은 detach하고,
  종료/reset mask를 worker별로 적용한다. update 뒤 새 가중치로 해당 시퀀스를 다시
  계산해 다음 hidden state를 이어간다. 이전 시퀀스까지의 전체 history를 새 가중치로
  재계산하는 것은 아니므로 시작 hidden state에 이전 가중치의 영향이 남는 TBPTT
  근사다. 숫자를 0으로 reset하는 것과 gradient 경계를 구분한다.
- PPO approximate KL, clip fraction, value explained variance, latent Gaussian entropy,
  action log_std, 실제 epoch 수와 episode 길이/return/progress를 기록한다. target KL
  초과 시 추가 epoch를 중단한다. Gaussian entropy는 tanh 행동 entropy가 아니다.
- initial/final checkpoint를 같은 조건에서 결정적으로 평가한다. 진행 증가, 남은 코스
  진행 비율, 완주/실패, 횡오차 RMS/max, 노면 대비 차체 기울기를 기록한다.
- 몸 감각 제거 시 gyro·가속도·suspension·wheel·steer·speed·하중/중력 추정 채널을
  0으로 만들고 validity도 0으로 표시한다. 명령과 직전 action은 efference copy로
  유지한다. actuator 내부의 encoder/gyro 피드백은 모든 조건에서 동일하게 유지한다.
  따라서 이 대조는 **정책에 제공하는 몸 감각의 효과**를 시험한다.

순환 상태와 episode_start 처리는
[SB3 RecurrentPPO의 공식 구현](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
을 참고했다. optimizer의 제한된 재사용은
[PPO 논문](https://arxiv.org/abs/1707.06347)의 접근을 따른다. 해당 구현과 수치적으로
동일하다는 뜻은 아니며 본 저장소의 GRU와 sparse circuit을 유지한다.

## GPU 후속 실험

설정은 `configs/campaign-followup-gpu.json`, 공통 평가는
`configs/evaluation-followup.json`이다. GPU 1개를 할당받아 아래 8회 학습을 순서대로
실행한다. 다른 사용자나 기존 서비스의 allocation은 사용하지 않는다.

| 조건 | 환경/기억을 update마다 재시작 | gamma | 정책 몸 감각 | 비교 목적 |
|---|---|---:|---|---|
| A `a_restart` | 예 | 0.99 | 전체 | 현재 재시작 방식 대조 |
| B `b_continuous` | 아니오 | 0.99 | 전체 | A와 비교해 연속 주행 효과 |
| C `c_long_discount` | 아니오 | 0.998 | 전체 | B와 비교해 보상 시간 척도 효과 |
| D `d_no_proprioception` | 아니오 | 0.998 | 제거 | C와 비교해 정책 몸 감각 효과 |

시드 3001/3002를 각 조건에 공통으로 적용한다. 매 학습은 환경 8개 × 256 tick ×
32 update = **65,536 transition**, 전체 **524,288 transition**이다. PPO epoch 최대 4,
target KL 0.03, learning rate 3e-4, lambda 0.95, reward와 모델 구조는 공통이다.
학습 노면은 grade +2%, 100 m 합성 코스이며 초기 X를 3–80 m에서 뽑는다.
episode 상한은 4000 tick/40초다. GRU/CNN·PPO는 CUDA, 물리는 CPU 8 worker,
영상은 할당 GPU의 EGL이다. CUDA/EGL gate를 먼저 통과해야 한다.

100 Hz에서 gamma 0.99의 할인 시간 척도 `-dt/log(gamma)`는 약 1초, 0.998은
약 5초다. 이 값은 선택한 가설이며 현재 결과로 최적값이 입증된 것은 아니다.
GAE lambda는 공통으로 유지해 gamma의 영향을 먼저 분리한다. 평가 지형은 학습에
제공하지 않고, 평가 중 optimizer 갱신도 없다. 초기/최종 모델을 같은 다섯 조건에서
평가한다. 시드별·조건별 raw 차이를 남기며 두 시드로 통계적 우월성을 주장하지 않는다.

로컬 pilot은 별도 `configs/campaign-followup-local.json`이다. CPU 2 worker,
시드 3001, A/B 각 2048 transition으로 실행 경로·연속 episode·평가 예산을 확인한다.
원격 GPU 학습의 처리량이나 충분한 학습 성능으로 표시하지 않는다.

## 판정과 다음 단계

우선 실제 종료 이유와 유한 loss/gradient, V oracle mask, worker별 reset, CUDA 배치,
source/config/checkpoint hash를 확인한다. 그다음 초기 대비 학습 후 변화와 A/B/C/D의
동일 시드·평가 조건 차이를 비교한다. 주행 성능의 우선순위는 완주·실패율, 그다음
남은 코스 대비 진행 비율·횡오차다. 진행량만 늘고 이탈이 늘면 개선으로 채택하지 않는다.

두 시드에서 결과가 일관되면 제3 시드와 더 긴 학습으로 재현한다. 전 조건이 계속
실패하면 교사 controller 검증과 BC/DAgger를 다음 후보로 설계한다. body ablation
차이가 나타나면 IMU·suspension·wheel별 제거와 지연/잡음 실험으로 분해한다.
실제 fly graph와 서킷은 검토된 데이터·포트·사용권 확보 후 같은 감각·예산 조건에서
추가한다. 이번 캠페인에 실제 connectome이나 Nordschleife를 사용했다고 주장하지 않는다.

## 로컬 pilot 결과

CPU/V, seed 3001, A/B/C/D 각 2048 transition, 동일한 세 평가 조건에서 실행했다.
A/B/C의 initial checkpoint 평가가 일치하므로 초기 가중치/관측 조건을 확인할 수 있다.
D는 같은 초기 가중치를 쓰지만 정책 몸 감각이 제거돼 초기 평가 행동부터 달라진다.

| 모델 | 학습 전 평균 진행(m) | 학습 후 평균 진행(m) | 학습 후 완주 |
|---|---:|---:|---:|
| A 재시작 | 35.50 | 15.73 | 0/3 |
| B 연속 주행 | 35.50 | 6.42 | 0/3 |
| C 연속·gamma 0.998 | 35.50 | 7.33 | 0/3 |
| D C에서 몸 감각 제거 | 26.24 | 6.27 | 0/3 |

이 예산에서는 **네 조건 모두 성능이 악화**됐으며 개선 정책을 채택하지 않았다.
반면 A는 모든 학습 시퀀스가 128 tick에 재시작돼 episode 종료를 한 번도 보지 못했고,
B는 worker 상태가 1024 tick까지 이어지고 587 tick의 track_exit를 실제 경험했다.
따라서 B는 실패를 학습 데이터에 포함하는 구현 개선이며 정책 성능 개선은 별도 검증이다.
update별 KL은 0.001–0.017 범위였고 loss/gradient는 유한했다. A의 짧은 학습 reward가
늘어도 평가 진행이 줄어들어 training reward만으로 모델을 채택하지 않기로 했다.

실행 증거는 `runs/20261005-followup-local-pilot/`와
`runs/20261005-followup-local-extension/`의 summary, training, initial/final 평가와
trace다. 합친 동일 시드/조건 비교와 입력 hash는
`runs/20261005-followup-local-analysis/`에 남겼다. **25개 테스트 통과**:
GRU/graph nonzero boundary replay, optimizer
경계의 지속 상태/episode, 평가 max_steps 독립, 감각 masking, paired seed/case 비교를
추가했다. 이 결과는 Windows CPU 환경이며 원격 CUDA gate 결과는 아직 없다.

## 제출 기록

후속 GPU 작업 **262946**, source revision `315abae`를 제출했다.
`runs/remote-followup-ablation-20261005/`에 stage manifest·submission·start-status·
collection plan을 저장했다. 봉인 ZIP SHA-256은
`33f026a28f8a8db7246032c9c9a553a7268d3aa6aedd89760977cddf71e002d4`다.

`afterany:262880` 의존성을 걸어 기존 작업 종료 뒤 실행하도록 했다. 확인 시점에는
**PENDING(Dependency)**로 GPU 학습은 시작 전이다. partition1/hpc-stat1,
GPU 1개·CPU 8개·24 GB·최대 3시간이며, watcher PID 792의 대기 heartbeat를 확인했다.
첫 작업의 성공/실패 여부와 별개로 새 코드의 CUDA/EGL gate를 실행한다.
training이 실패한 조건은 이유를 보존하고 다음 조건을 진행하며, 부분 실패를 전체
성공으로 표시하지 않는다. timeout 시에도 중간 checkpoint·progress·log를 회수한다.
작업 종료 뒤 source/input/log/result를 hash 검증해 회수한 후 해당 소유 run만 정리한다.

이번 턴에 원격 GPU에서 학습한 모델의 결과나 처리량을 확인한 것은 아니다.
로컬 pilot은 완료됐으며, GPU 학습·평가·시드별 비교는 제출한 캠페인이 수행한다.

10월 5일 10:14에는 262946의 미실행 단일 캠페인을 **array 264771 + CPU 집계
264772**로 대체했다. 30분 작업 8개, GPU 동시 사용 1개이며 같은 조건/시드/예산을
유지한다. 262946은 취소 증거와 입력/소스를 회수·검증한 뒤 원격 소유 tree를 정리했다.
최신 배정/최적화와 영수증은 [처리량과 대기 최적화](performance-and-queue.md)에 있다.
