# 시작 온도별 실제 시연·정합 자료와 native 감각 경계 학습 — 2026-11-03

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·변속/셋업/차종·검증된 상대풍·teacher/평가 전용 지도/world pose 경계를 유지한다. 전체 목표는 active이며 학생 한계 주행을 완료한 결과가 아니다.

## 실제 자료 확보와 독립 판정

공통 원본은 `runs/thermal-driving-20261103/`다. 현재 같은 BMW Z4 GT3/Monza에서 **명시적으로 새로 시작한 세 세션**의 시연을 수집하고 native/SDK 랩 증거와 4방향 영상·몸 감각 정합을 독립 검증했다.

| 시작 목표 | 시연 | SDK 유효 랩 | 최대 타이어 이탈 수 | 원본/적격 영상 그룹 | 전체 랩 감독 라벨 |
|---|---|---:|---:|---:|---:|
| 40℃ | cold40-demonstration-03 | 1:54.041 | 2 | 1596/1593 | 2851 |
| 110℃ | hot110-demonstration-01 | 1:52.240 | 0 | 1568/1564 | 2806 |
| 80℃ | nominal80-demonstration-01 | 1:52.406 | 0 | 1569/1565 | 2811 |

SDK LapInvalidated=0과 native 랩 카운터/타이머·약1랩 진행률·정지/제어권 반환을 별도로 확인했다. 기존 감사는 이탈≤2를 허용하며, 냉간 기록의 `zero_tyres_out=false`를 보존한다. 첫 온도 개입/준비 부분 랩은 랩 성적에서 제외한다. 각 조건 한 번의 **교사** 시연이므로 온도별 성능 차이의 통계적 입증이나 학생 성과로 해석하지 않는다.

세 `*-qualification-01/`은 같은25ms 몸→render 상한,50ms source→실제 조작 상한·150ms 카메라 age·화소/시간 대조 기준을 유지해 통과했다. 전체 랩 라벨 정합률100%이며 renderer/world truth가 actor 배열에 들어간 수는0이다. 정확 GPU 노출 시각·원천 physics atomicity는 여전히 미검증이다.

**초기화 목표와 실제 학습 온도는 다르다.** 냉간 기록의 교사 source 기준 전체 랩 코어 온도는 약42.7–101.0℃다. 반면110℃ 시작 기록은 저속 curriculum 동안 식어 전체 랩은 약70.3–98.4℃였다. 이 자료를110℃ 한계 주행 감독이라고 부르지 않는다. 주행 중 온도를 계속 덮어쓰지 않았으며 실제 원본 온도·압력·하중/slip과 실패를 보존했다. `sensation-coverage-01/02`는 실제 요청/source bytes의 진단이며 actor tick에 직접 붙이는19개 경계와 구분한다.

## 수집 계약 구현과 실패

- `thermal_teacher_start.py`: 검증된 초기화 run과 같은 generation/epoch/source, 현재 네 바퀴 온도 목표±5℃, paused·오프라인·AI/camera/probeoff를 요구한다. 초기화 자체의3패킷±2℃ 확인과 수집 준비의±5℃는 별개 계약이다. 원래75..85℃ 기본 시작/재개 조건을 유지하며 명시적 thermal 증거가 있을 때만 별도 조건을 받는다. 온도 목표/조건 증거는 actor 입력이 아니다.
- `control_ac_session.py`의 `--pause-after-readback`: 한 번의 온도 SDK 호출→명시적 resume→즉시 native3패킷 확인→명시적 pause로 약40/110/80℃를 유지한 상태에서 모델/카메라를 준비한다. 일시정지 확인 후 NAS 쓰기를 수행한다. setter를 반복하는 thermostat이 아니다.
- `collect_synchronized_sac_demonstration.py`와 `evaluate_sac_teacher_curriculum.py`: 새로운 thermal 시작/재개 증거와 명시적 정합 계획을 전달한다. 첫 전진 기어·autoshifter·stationary/full brake·같은 세션 보정·on-track/cadence/freshness 기준은 유지한다.
- 현재 세션 조향 보정의±.1은 교사의±.6 범위를 보증하지 못해 `first-gear-01`이 조작 전에 거절됐다.±.6 보정 `steering-calibration-02`와 실제 첫 기어 확인 후 진행했다.
- `cold40-demonstration-01`은 가상환경 redirector의 Popen PID와 실제 recorder PID 불일치로 영상0/교사0에서 종료했다. 소유권 검사를 완화하지 않았다. 실제 Python 실행 파일+기존 CPU Torch 경로에서 자식 PID 동일성을 확인한 `interpreter-receipt-01.json` 이후 수집했다.
- `cold40-demonstration-02`는 초기 thermal gate를 통과했으나 공통 resume 함수에 남은 warm75..85℃ 검사에서 중단했다. 원본은 기어2/autoshifter1/약39.995℃였으며 교사 조작0·영상7장이다. 같은 thermal 증거를 재개 단계에서도 검사하도록 고친 뒤 새 epoch의03을 수집했다.
- `tests-final-01.txt`: 수집/세션/thermal 계약142검사, `native-tests-01.txt`: 감각/코호트14검사 통과. 의도된 Lua 오류 주입의 Windows 예외 진단 원문도 보존한다. GPU/Windows 보안 정책은 변경하지 않았다.

## 모델 입력 준비

기존 승인된 FlyVis 환경 `%LOCALAPPDATA%/flybrain-bio-20261029/flyvis-env`에서 Torch2.9.1+rocm7.2.1과 AMD RX9070XT 실제 텐서 연산을 확인했다(`flyvis-gpu-probe-01.txt`). CPU 경로만의 CUDA=false와 전체 GPU 부재를 혼동하지 않는다. 원래 GPU 전처리 소스/author weights·5ms held-input 적분/에피소드1회 reset을 그대로 사용했다.

세 `*-actor-data-01/`과 `*-actor-audit-01/`이 완료했다. 총 **48,956사건**이며 원래 시각·첫 native receipt·packet·mask·교사 라벨을 보존한다. 정합되지 않은 초기3/4그룹은 frozen 시각 recurrence에도 넣지 않았다. 몸21의 옛 가속3축/roll·pitch rate false mask는 바꾸지 않았다. 새 자료의 native-only tick도 명시적 비감독 mask를 유지한다. 원천→held 몸 age 최대 약12ms다.

`spatial-recovery-01/`은 새 영상에서도 원래64개 정적 pool을 최대1.97e-6 오차로 복원하고, 소실된 정적 retinotopic 성분이 옛 pool에서 보이지 않음을 약1.65e-15 이내로 검증했다. prepool T4/T5 복구나 새 생리 검증이라고 부르지 않는다.

`native-audit-01`·`acceleration-bandwidth-01`: 새 세 조건에서도 같은 시간 대역의 속도 미분과 native G의3축 비교를 수행했다. 고속 축별 상관≥약.999, RMSE 약.04–.05m/s²이며 actor는 미래 미분/world 좌표 없이 원래 현재 native 값을 받는다. 설치 SDK/실측 범위 밖의 일반 등록으로 확대하지 않는다.

`extension-data-01`·`extension-audit-01`: 현재 packet의 가속3축/slip4/하중4/코어온도4/압력4를 같은 첫 receipt와50ms guard로 재구성했다.48,956관측과 TRAIN-only 정규화를 독립 확인했으며 source 몸21/마스크는 그대로다. wheelSlip은 native scalar이고 특히 저속/lock에서 매우 큰 값이 있다. 물리 slip ratio/각도/마찰 활용도로 바꾸어 부르지 않는다. 압력은 native 단위다.

## 고정된 첫 감각 경계 학습

`cohort-intent-01.json` 및 `student-condition-split-01.json`: 사전에 정한40/110 시작 두 에피소드만 TRAIN(5,657라벨),80 시작 에피소드는 개발(2,811라벨)이다. 이 분할을 새 트랙/냉간·고온 held-out 검증으로 부르지 않는다. 초기/준비 부분은 state burn-in이고 제한 속도 조작을 목표300의 감독으로 섞지 않는다. 따라서110℃ 부근은 일부 burn-in/OOD이며 학습 지원 범위라고 주장하지 않는다.

보호한 출발점은 `runs/steering-precorners-20261103/optical-pre16-fit-01/steering.pt`, SHA `bb0bb4dd378e1685594e07928e3cd92ac8bf2543da6665c81aa5f05980fb6384`다. 기존51개 model 파라미터/버퍼·측정 연결·광학/DN 어댑터·여섯 근육→조향/가속/제동 readout을 고정한다. 새 native19→기존201개 몸 감각 포트의46입력·9,246가중치 공학 경계만 한 epoch·TBPTT32·Adam .0003·손실[4,1,1]·gradient clip1로 학습한다. 신경 상태는 결정 사이에 유지하며 detach는 numerical reset이 아니다. 새 경계에는 직접 차량 출력이나 teacher/map 입력이 없다.

`native-fit-01`은 기준 재생·비영 자극 전달을 마친 뒤 NumPy bool을 grad-mode에 넘긴 타입 오류로 optimizer0에서 종료했다. 원본/계획01을 보존한다. Python bool 변환, 실회로64사건 gradient 사전 검사·중간 baseline 예측/실패 상태 저장을 보강한 **native-fit-02**를 계획02로 실행했다. 하이퍼파라미터/데이터/학습 범위는 같다.

계획02의 실제 회로 gradient 사전 검사는 라벨·optimizer0으로 5,265개 비영 gradient, 최대약.0662를 확인했다.01의 작은 입력 자극은 같은64사건에서 차량 readout 차이5.44e-5를 만들었다. 이는 회로를 통한 수치 전달이며 생리 기능·실제 운전 기여의 입증이 아니다.

앞선 실행 중 관측 이후 `native-fit-02`의825회 학습과 전체 재생이 완료됐고 `native-fit-audit-01`의 독립 지표/체크포인트 감사도 통과했다. `progress-summary-01.json`은 그 관측 시점의 실행 상태이며 재개 시 PID/핸들을 다시 확인한다. 전체 목표 active, 실제 학생 한계 주행·새 냉간/고온 반복·새 코너/트랙·50ms 실시간 경계/몸 외란 안정화·변속/차종/타이어·셋업·검증된 상대풍과 빠른 라인/유효 랩은 계속 미완료다.


## 감각 경계 학습의 완료 결과

| 조건 | 기존 조향 RMSE | 후보 조향 | 기존 가속 | 후보 가속 | 기존 제동 | 후보 제동 |
|---|---:|---:|---:|---:|---:|---:|
| 냉간 시작 TRAIN | .089255 | .089443 | .173619 | .154559 | .164817 | .150087 |
| 고온 시작 TRAIN | .071246 | .071671 | .178835 | .160807 | .166004 | .151531 |
| 명목 시작 개발 | .071637 | .072054 | .176034 | .157825 | .168008 | .152198 |

개발 가중 MSE는 .0265808→.0229468(13.67% 감소)이며 가속/제동 RMSE는10.35%/9.41% 감소했다. **조향은0.58% 악화**했다. 가감속의 오프라인 모방 개선과 조향 개선을 구별하며 실제 학생 완주·한계 주행 개선으로 인정하지 않는다. 새 온도 입력만 TRAIN 기준으로 중립화하면 MSE .0231619로 정상보다 약.94% 커졌다. 기존 몸21의 온도 입력은 유지했으므로 전체 온도 기여·물리적 온도 변화 실험이 아니다.

새 감각 경계 전체를 끈 개발 재생은 원래 기준과 최대차0이다. 기존 모델51개 상태와 광학/DN 및 readout 세 행은 exact 보존했다. 새 가중치9,246개 중 비영5,265개, 최대 몸 전류 잔여 .18231이다. 체크포인트 `native-fit-02/body-extension.pt` SHA `d41677fcf67ad0ae4412e53788a91d0d7f83ba061653126f1e2708631e03d687`, 예측 SHA `1e75e6482e384c243efbda27cce8fe0e9953c502b0907b23e7178627d76d88e2`. 안전 weights-only 재로딩·TRAIN 정규화·원본/후보 전체 지표 독립 재계산을 통과했다. actor_qualified=false, 차량 명령0이다.

조향 요청에 대한 후속 `steering-transfer-01`과 독립 감사를 완료했다. 기존에 선택해 둔 bounded 조향 head를 새 세 시연의 동일 근육 궤적에 적용해 전체 .21–.72%/진입 전 .38–1.47%의 작은 개선을 확인했다([상세](steering-new-recordings-20261103.md)). 새 자료를 보고 재학습하거나 설정을 고르지 않는다. 같은 Monza의 새 기록이며 새 코너 일반화 시험이라고 부르지 않는다. 영상 개입의 기존 근거와 암기 가능성은 [조향 보고서](steering-command-loss-20261103.md)에 남겼다.

후속 진입 전 진단 `native-precorner-diagnostic-01.json`은 앞선 지속 회전 정의를 유지했다. 개발86라벨 RMSE .0630013→.0637392(1.17% 악화)이며, 학습 조건82/69라벨은 각각 .0628161→.0623246 및 .0481839→.0469175다. 이는 사후 진단이며 새 기준을 통과했다고 판정하거나 개발 자료로 재선택하지 않았다.
