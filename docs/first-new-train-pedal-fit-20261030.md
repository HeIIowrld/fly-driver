# 새 TRAIN 첫 페달 학습 — 2026-10-30

갱신: 2026-10-30T16:47:19.284517+09:00. 전체 목표는 active·미완료다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 계속 포함한다. 이번 결과는 그중 조건부 페달 부품의 첫 학습이며, 시각 조향·학생 단독 트랙/레코드라인 추종·유효 랩 검증이 아니다.

## 실제 실행과 데이터 범위

- 별도 사전 선언한 새 1909 TRAIN nonce `cb0c1f6723724f16ae6a6fa5047121d9`의 원래 986개 상태와 body21·마스크·시간·이전 held command를 사용했다. 상태 삭제·재시각화·중간 수치 상태 reset은0이다. 진단1899/1903이나 실패1856의 역할을 승격하지 않았다.
- 1910 원본 HB, 1911 native same-copy, 1912 원시 영상/몸 7배열·actor trace, 1913 전체 이력 로드를 통과했다. 실제 수집 최대 간격47.795ms, 80ms 초과0, 영상 대기 누락0이다. 원래 fullbrake1 표본266개와 적격 brake1 loss257개를 보존했다.
- 경로 근거는 원래1274개 관측 pose가 선언한 B 안에 있다는 sampled coverage다. `continuous_geometry_verified=false`, 독립 heldout·수학적 연속 기하 검증은 주장하지 않는다. Pose·지도·정답 라인은 actor 입력이 아니다.
- 원본1908 pedal review에 없는 status를 추가하지 않았다. 새 `pedal_plan_b2.py`/`train_pedal_b2.py`는 typed review와 원래 SHA `89d9b19df4112df6db12fdb9d6980956fe9e547b8f7c5abd582729c2229704b2`의 loader 입력 bytes를 분리한다. 모델·engine·TBPTT·optimizer 계산은 기존과 동일하다.
- GPU gradient 원본22c84a/session67250은 fb9d4a 종료0. 동일 전체 이력의123×2 gradient 행렬은 rank2이며 source drift0, optimizer0이었다. [결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1913-b2-whole-history-load/firstfit-b2-interface/actual-gradient-qualification/result.json) SHA `c92d1fe26a2b15f25f1d2611a0c84356d72b17b551a8d29d4dd9c6dc35f08797`, [종료 영수증](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1913-b2-whole-history-load/firstfit-b2-interface/actual-gradient-qualification/original-terminal-receipt.json) SHA `e841cc55b3cdac2aaa687e785e9a1474dc7464bb6533e160e4650d3249cb255a`.
- 실제 fit 원본955d16/session36427은 d528e1 종료0. 전체 이력을 한 번 순회하며 TBPTT8의124개 창 중 적격 loss가 있는123개 창에서 Adam1e-3로 정확123회 업데이트했다. `command.power_delta`와 `command.speed_feedforward_delta`만 학습했다. DNa 조향·backbone·readout의 고정 fingerprint는 baseline/train/final 사이 동일하다.

## 같은 TRAIN 이력의 고정 계수 평가

| 평균 제곱 오차 | 학습 전 | 학습 후 |
|---|---:|---:|
| 가속 페달 | 0.04705015 | 0.04081350 |
| 제동 페달 | 0.19425869 | 0.19334326 |
| 두 축 평균 | 0.12065442 | 0.11707838 |
| 적격 fullbrake1 | 0.73296245 | 0.73163267 |

두 축 평균 감소는2.9639%다. 이는 같은 TRAIN 이력의 부품 오차 개선이며 일반화·주행·안전·랩타임 성과가 아니다. 강한 제동 오차가 여전히 크고 zero-goal/zero-body 정지 유지 alias는 해결되지 않았다. 페달 계수의 간접 조향 출력 변화도 최대0.00274268로 관측돼 조향 불변을 주장하지 않는다.

최종 계수는 power_delta `0.044394347816705704` / speed_feedforward_delta `0.049605414271354675`다. 원래 전체 checkpoint를 복제하지 않고 [두 scalar checkpoint](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1913-b2-whole-history-load/firstfit-b2-interface/actual-fit/final-two-DNg-scalars.json)만 저장했다(SHA `cf8e41f40da3eec50853511dea641b26de225e0a3648acb9f9353d220f73513a`).

## 실제 시간과 보존 증거

데이터 로드1.8004초, 모델 준비6.0703초, baseline15.0516초, 학습18.3222초, 최종 평가13.4770초, 전체59.8081초다. 실제 학습 처리량은6.713 update/s다. 이전의 장시간 병목을 GPU 학습 속도로 해석하지 않는다.

[fit 결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1913-b2-whole-history-load/firstfit-b2-interface/actual-fit/result.json) SHA `584e9504c6c60ede7e6bc3b8a41e12200b9bdace9e9c26503b0089fefabd100f`; [원본 종료 영수증](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1913-b2-whole-history-load/firstfit-b2-interface/actual-fit/original-terminal-receipt.json) SHA `5168bb623b435794d1cf70e655b124ecb53e69ee18840395f00957412180df80`. Fit plan SHA `155043c5f58b28a4f493e9b65d56a7214414b8ea09768aaff833cb0ba5d89d1c`, source closure134개 SHA `bb5e6e2080821baefb253bc69a791473e33428bd90eb9664bf6559c689b64438`, source drift0이다.

GPU peak206,028,288byte. 종료 전 allocated/reserved113,246,208byte와 allocator cleanup=false를 원본 그대로 남겼다. 원본 process terminal0과 별개이며 global GPU idle·측정된 zero allocation을 주장하지 않는다. 모델 학습은 offline였고 UI·actuator·학생 활성화는0이다.

## 다음 실제 목표

현재 페달 학습에는 image→desired yaw 경로가 없다. 기존 겹눈 intensity·T4/T5→SceneCommand→DNa 구현은 과거 camera/body/v6 계약이므로 현재 외부 눈/body21·PhysicalManeuverCircuit·새 페달 계수에 자동 연결하지 않는다. 이전 여섯 시각 모델도 body-only 기준보다 나빴다.

다음은 명시적 visual→physical desired yaw(rad/s) 연결을 만들고, 같은 원래 이력·몸·시각에서 실제 영상/첫 장면 고정/시간 이동/운동채널 제거를 비교해 영상의 선행 조향 기여를 검증하는 것이다. 이후 NN 포함 실제 주기 검증과 짧은 AI-off 학생 구간에서 기준선 횡오차 RMS/P95/최대·방향오차·트랙 이탈·학생 진행거리·lap invalidation을 측정한다. 현재 CSV 기준선을 세계기록 최적선으로 인증하지 않는다. 틸트·하중·타이어의 실제 운동 인과 경로와 빠른 유효 랩은 여전히 미완료다.
