# PPO 성능 저하 개선 실험 — 2026-10-06

기존 원격 본학습에서 처음 회수한 7개 실험은 128 update/262,144 transition을
실행했지만, 5개에서 BC+DAgger 직후보다 최종 평균 주행 거리가 줄었다.
최종 49회 주행은 모두 track_exit였다. 가치 함수의 설명 분산은 0 근처였다.
누락 시드 2개는 같은 소스로 별도 재시도했다(268302, 268303). 해당 재시도 결과는
이 개선 연구와 합치지 않고 기존 캠페인의 후속 증거로 남긴다.

공유 특징망에 가치 손실이 들어가고, KL early-stop은 업데이트 후 초과를
되돌리지 않는다. 이 사실만으로 성능 저하 원인을 확정할 수는 없다.
후속 실험은 행동 정책 보존, 업데이트 제한, 가치 gradient 영향의 가설을 검사한다.
모델·환경은 합성 3D 도로와 CNN/GRU다. 측정 초파리 connectome이나
뉘르부르크링 실측 모델을 사용한 실험으로 표시하지 않는다.

## 구현한 관찰 및 보호 장치

- 업데이트마다 metrics.jsonl에 보상, actor/value/anchor 손실, return scale,
  KL, clipping, 설명 분산, 시도/수용한 optimizer step을 기록한다.
- 8 update마다 덮어쓰지 않는 checkpoint와 개발용 주행 리플레이를 보관한다.
  모방 학습 직후 update 0도 같은 개발 조건으로 평가한다.
- 완주율 → 실패율 감소 → 남은 경로 대비 진행률 → 횡오차 순으로 최고
  checkpoint를 선택한다. 완주율 개선이 없다면 초기 개발 주행 거리 90% 이상을
  유지해야 선택 후보가 된다. 정지한 채 실패를 피하는 모델의 선발을 막는다.
  update 0이 최고일 수도 있으며 이를 그대로 기록한다.
- 초기/마지막/개발 평가로 선택한 모델의 종료 평가를 구분한다.
  선택 파일이 초기 checkpoint와 byte 단위로 같으면 초기 평가를 재사용한다.
- frozen BC+DAgger 정책은 학생이 실제 관측한 센서 시퀀스를 계속 따라간다.
  anchor는 속도·곡률만 보존하며 세 번째 residual yaw action을 제약하지 않는다.
  episode reset과 update 간 recurrent state 유지 규칙을 따른다.
- max_update_kl 초과 시 모델과 Adam 상태를 함께 복원한다. 거절한 업데이트와
  수용한 optimizer step을 구분한다. 실제 센서 관측 기록과 episode 상태는 유지한다.
- critic_feature_grad=false는 가치 헤드의 gradient가 actor encoder/GRU를
  바꾸지 못하게 한다. 별도의 큰 critic network나 privileged critic 구현은 아니다.
- regression_patience를 설정하면 반복적인 개발 성능 저하를 중단할 수 있다.
  이번 비교는 같은 PPO 예산을 유지하기 위해 이 조기 종료를 **사용하지 않는다**.

actor는 기존 좌우 각도 샘플링 망막, motion response, body/validity를 사용한다.
route/pose truth는 교사 label과 평가 진단에만 쓴다. 좌우 모터·토크 벡터링은
기존 residual 제어 계약을 유지한다.

## 순서와 예산

1. 새로운 source snapshot으로 실제 RX 9070 XT 카메라 D3D12 렌더링과 ROCm
   PPO 역전파를 검사한다. EGL과 같은 발행 명령의 물리가 같아야 시작한다.
2. local seed 7101에서 아래 세 조건을 순차 실행한다. 물리 worker 4개,
   learner thread 1개, horizon 256, update 24로 조건당 24,576 PPO transition이다.
   expert 4,800 decision/24 epoch는 해시·환경·physics/sensor source와 runtime이
   맞는 기존 데이터만 재사용한다. DAgger 2,400 decision/16 epoch는 매번 새로 수집한다.
3. 후보가 없으면 recovery_dagger_anchor 한 조건을 추가한다. 학생이 방문하는
   x=3/35/65, lateral ±0.9 m, heading ±7°에서 총 2,400 decision의 추가 DAgger를
   수행하고 9,600 sample을 12 epoch 학습한다. 교사 데이터 양 증가를 명시한다.
4. 통과 후보 최대 두 개와 같은 PPO control을 새 local seed 7102로 재검사한다.
   총 local 학습은 최대 6개, task당 1시간 제한이다. timeout과 실패도 회수한다.
5. 두 시드에서 통과한 조건만 control과 함께 새로운 remote seed 7201–7203으로
   올린다. 모델·optimizer·교사/DAgger를 원격에서 새로 만든다. GPU preflight와
   source/입력/회수 검증을 먼저 수행한다. 최대 9 task, task당 262,144 PPO transition,
   GPU array concurrency 2, partition1/node auto, task당 3시간이다.
   원격에서는 16 update마다 checkpoint와 개발 평가를 보관한다.

| 조건 | actor LR | BC anchor | KL 복원 한계 | 가치 특징망 gradient / head LR |
|---|---:|---:|---:|---|
| ppo_control | 3e-5 | 0 | 없음 | 공유 / 3e-5 |
| anchored_ppo | 1e-5 | 5, 속도·곡률 | 0.01 | 공유 / 1e-5 |
| head_critic_anchor | 1e-5 | 5, 속도·곡률 | 0.01 | 차단 / 3e-4 |

anchored_ppo는 여러 보호 장치를 함께 시험하는 후보이며 개별 효과를 분리한
인과 실험이 아니다. head_critic_anchor는 같은 보호 장치에서 가치 gradient와
head 학습 속도를 추가로 바꾼다. 성공 후에 개별 ablation을 설계한다.

## 검증과 선발

훈련 중 개발 seed는 9301/9302, 종료 후 로컬 후보 확인은 9401–9404다.
원격 최종 시험은 새 seed 9801–9807과 별도 grade/bank 조건으로 잠근다.
훈련 중 개발 seed와 종료 평가 seed가 겹치면 실행을 거절한다.
이미 관찰한 기존 8001–8007 결과를 새 미관찰 시험으로 재사용하지 않는다.

각 local 시드에서 후보의 **마지막 PPO checkpoint**가 control 대비 평균 주행
거리 10% 이상 증가하고, 4조건 중 3조건 이상 개선되고, 실패율이 늘지 않고,
자신의 BC+DAgger 직후 거리 90% 이상을 유지해야 한다. 실제 수용한 optimizer
step도 있어야 한다. 최고 모델이 초기 BC에 머물렀다는 이유만으로 선발하지 않는다.
두 시드 모두 통과해야 원격 제안을 만든다. 거리 증가와 자율주행 완주는 구분한다.

`scripts.run_improvement_study`는 이 순서와 제한을 실행하고,
`scripts.watch_improvement_study`는 NAS로 실패를 포함한 증거를 회수·해시 검증한 뒤
GPU preflight와 별도 원격 제출을 수행한다. source 변경이나 preflight 실패가 있으면
제출을 중단하고 원인을 heartbeat.json에 남긴다. 기존 원격 run/runtime cache는
수정하지 않는다. 이후 카메라 장애·몸 감각 제거·TV off/fixed/residual ablation은
이번 개선 후보의 결과를 확인한 뒤 수행할 다음 연구이며 아직 시작한 결과가 아니다.
