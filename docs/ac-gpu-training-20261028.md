# AC 파리 모델 GPU 학습 전환

기록일: 2026-10-28 KST.

이전 t350 학습은 CPU 1스레드였다. RX 9070 XT와 ROCm PyTorch는 사용 가능했지만, 학습 코드가 모델·표본을 CPU에 두고 실시간 CPU 추론용 cached CSR backend를 사용했다. GPU 부재가 원인이 아니었다.

## 실제 장치 확인과 비교

환경은 PyTorch `2.9.1+rocm7.2.1`, HIP `7.2.53211-158bd99533`, AMD Radeon RX 9070 XT다. ROCm에서도 PyTorch의 장치 API 이름은 `cuda:0`이다.

t366은 실제 6,557개 노드·644,512개 연결 모델에 같은 기록 prefix 상태와 16개 의사결정 × 4개 에피소드 × 4 ticks를 넣고 순전파·역전파를 비교했다. optimizer update와 게임 명령은 0회였다.

| 항목 | CPU | GPU |
|---|---:|---:|
| backend | cached CSR | sparse COO |
| warmup 제외 반복 중앙값 | 0.97258초 | 0.24701초 |
| 기울기 장치 | CPU | cuda:0 |

해당 구간은 약 3.94배 빨랐고, 출력 최대 차이는 1.19×10⁻⁷이었다. 이는 전체 학습 시간의 향상 배수와 구별한다.

## 구현

- 학습 config의 `device: cuda`를 명시적으로 지원한다. 요청 장치가 없으면 실패하며 CPU로 조용히 대체하지 않는다.
- 모델·감각 표본·정답·유효성·padding/active mask·손실 정규화 계수를 같은 장치에 둔다. GPU에서도 원본 에피소드 수치 상태와 chunk detach 규칙을 유지한다.
- GPU 학습은 장치 지원 sparse COO 연산을 사용한다. 측정 연결·부호·정규화·neural dynamics를 바꾸지 않는다. FP32이며 autocast를 사용하지 않았다.
- 실제 parameter·observation·target·gradient 장치를 기록한다. GPU synchronize를 포함해 epoch 시간을 측정한다.
- 저장할 때는 tensor를 CPU로 옮기고 기존 주행용 cached CSR constructor를 유지한다. 따라서 GPU에서 학습한 체크포인트를 CPU 주행 코드에서 읽을 수 있다. 학습 backend와 배포 backend를 별도로 기록한다.

## 전체 학습 비교

t368은 t350과 초기 가중치·seed·자료 1,238개·8 epoch·학습률·학습 그룹·제동 가중치·batch·optimizer 조건이 같고, 장치/backend만 바꿨다. 최신 구간 500개는 두 실험 모두 학습에서 제외했다.

| 항목 | t350 CPU | t368 GPU |
|---|---:|---:|
| 평균 epoch | 56.650초 | 16.319초 |
| 실제 optimizer update | 8 | 8 |
| 최종 조향 RMSE | 0.0570130 | 0.0570130 |
| 최종 가속 RMSE | 0.0686335 | 0.0686335 |
| 최종 제동 RMSE | 0.0989114 | 0.0989114 |

실제 평균 epoch 기준 약 **3.47배** 빨랐다. GPU parameter·observation·target·gradient는 모두 `cuda:0`이었다. 학습 중 PyTorch 최대 할당량은 635,840,512 bytes(약 606MiB)였으며, 이는 게임을 포함한 전체 GPU 메모리 사용량은 아니다.

CPU 실험 대비 학습 파라미터 최대 차이는 2.00×10⁻⁷이었다. 저장 tensor가 전부 CPU에 있고, 선언한 시각·기관 연결부 외에는 바뀌지 않았으며, 뇌 연결·출력부·photometry·배포 constructor가 그대로임을 확인했다. 별도 증거는 `runs/ac-joint-multisection-gpu-training-20261028-t368/device-transfer-verification.json`이다.

GPU 체크포인트 SHA256은 `3bf2f751443ca3a7e625da58297e274bd0c1d41eb318e8afd1e0929e4434cf1a`다. 현재 AC 학습 config는 `configs/ac-joint-multisection-gpu-training.json`이며 실제 주행 추론은 CPU로 별도 유지한다.

t369는 GPU 체크포인트를 기존 CPU 주행 backend에 올려 기존 개발 자료 75개를 재생했다. RMSE는 0.15800436/0.11919675/0.31890161로 CPU 학습 모델과 수치상 거의 같았다. 고정 회로와 pose-free 입력을 확인했다. 제동 일반화 문제가 해결된 결과는 아니며 GPU 체크포인트 자체의 실제 차량 폐루프 시험은 아직 하지 않았다.

검증: CPU/GPU 학습·recurrent carry·제동 weighting 관련 29개 테스트 통과, 새 장치 로그를 포함한 GPU 통합 사례 추가 확인, Ruff 통과. GPU 학습 속도 개선이 제동 일반화·서킷 완주 성공을 뜻하지는 않는다.
