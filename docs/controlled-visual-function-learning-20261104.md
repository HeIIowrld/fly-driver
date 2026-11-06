# 위치·속도를 분리한 운동 시각의 기능 보정 — 2026-11-04

> **후속 2026-11-04:** 기존 pooled 시각과 motion의 실제 TRAIN 보호 공동 학습111회를 완료했다. 기록 재생 진입 전 조향은 소폭 개선됐지만 새 예약 방향 반응/크기 목표는 미달이다. 이 문서의 후보27d21422…와 그 실패/학습 기록은보존한다. [후속 공동 보정/평가](visual-route-protection-20261104.md).

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·가감속·변속·하중/타이어/셋업/차종·검증된 상대풍을 유지한다. 실측 연결·author 모델·기능 근사·공학 변환과 학습한 차량 대응을 구별한다. 지도/world pose/절대 heading/교사 조작은 actor 입력이 아니다.

## 결과와 범위

앞선 [실제 영상 교환의 혼입](precorner-road-pose-confound-20261104.md)을 피하는 **합성 부품 반응 실험**을 수행했다. 원래 fine 후보의 같은 위치·같은 속도·같은 몸/회로 prefix에 좌우 코너를 주었을 때, 사전에 정한 작은 양의 방향 순서를 만족한 조건은3/18이었다. 기존 운동 시각 projection만 한 번 보정한 뒤 학습 질감은9/18, 학습하지 않은 별도 질감은 **12/18**로 늘었다. 새 motion을 prefix 이후 끄면 각 경우0/18이다.

각18조건은 **6개 위치/속도 조합을 세 가지 가상 온도에 반복**한 것이다. 독립된 실제18개 코너 성공률이 아니다. 별도 질감의 위치/속도 조합으로 보면1/6→4/6이며, 도로 형상은 같은 합성 생성기다. 합성 body40/80/110℃ 결과를 실제 온도 적응 증거로 부르지 않는다.

**아직 반응 크기도 부족하다.** 양의 순서 기준은 signed canonical margin>0.0001이지만 학습 목표 margin은0.002다. 새 후보의 별도 질감 최대 margin은0.00024945이며, 목표0.002를 만족한 조건은 **0/18**이다. 6m/s·횡위치−2m는 여전히 반대 방향이고,6m/s·+2m는 반응이 약해 기준에 못 미친다. 몸/비행 회로의 생리 기능이나 R1/R2 완료, 실제 코너 운전 성공으로 판정하지 않는다.

![별도 질감에서의 위치·속도별 방향 순서](../runs/controlled-visual-factorial-20261104/plot-02/heldout-bend-margins.png)

점선은 사전 양의 순서 기준이다. 세 온도 열의 거의 같은 방향 차이는 온도 적응의 입증이 아니다. 학습 전후의 작은 조향 차이를 보기 위해 세로축을canonical×1000으로 표시했다.

## 합성 입력과 같은 몸/시작 상태

루트는 `runs/controlled-visual-factorial-20261104/`다. `stimulus-plan-01.json`의 고정 조건:

- bend−1/+1 × 횡위치−2/0/+2m × 이동3/6m/s =12개 영상 궤적. 도로 폭8m·굴곡 시작12m·curvature 계수.018·yaw0을 사용한다. yaw를 변화시킨 실험은 아직 아니다.
- 각41그룹·40ms 간격, 시야별 합성 촬영 시각은1ms 차이다. 네 시야를 각자의 시각에 렌더하고 기존 camera/projector 및 author FlyVis의 T4a–d/T5a–d·원래 scene/static·새 fine2048 경로를 그대로 통과시킨다.
- frame0–10은 같은 직선 영상, frame11에서 diagnostic bend를 나타낸다. 이는 실제 도로의 연속 등장과 다른 인위적 자극이다. 초기 변화 직후를 피한 frame25–40의 방향 반응을 계산했다.
- 마지막 전방 카메라 위치도 굴곡 시작 전이다. 각 합성 궤적마다 author steady prior에서 시작하고 그 궤적 안의 시야별 recurrence는 유지한다. 양쪽 bend의 원본 RGB prefix는 byte 동일하며 모델 특징 prefix 최대차는두 질감 모두1.05e-7 이하다.

`stimuli-01`은 texture781104, `stimuli-02`는별도 봉인 texture781105다. 두 번째 plan은 optimizer 전에 고정했고 optimizer가 그 영상/특징/출력을 읽지 않는다. 두 source 합계984그룹이며 실제 카메라/SDK 기록이나 실측 신경 활동이 아니다. 기존 anti-aliased planar renderer는 수정하지 않았고, 단일 시야 렌더 최적화가 원래 렌더의 각 시야 픽셀과 일치함을 검사했다.

`responses-01`/`test-parent-01`은 원래 fine 후보를 검사했다. 각 위치/속도·가상 온도 조건에서 공통 prefix를 한 번 실행한 뒤 **동일한 전체 회로 state를 복제**해 좌/우 bend와 motion-off 분기를 시작했다. 각 분기 내부 상태는 연속 유지한다. 이것은 연구용 반사실 분기이며 production 루프의 중간 numerical reset이 아니다. body21·마스크·native19 가상 값과 그 이력/시각은 각 pair에서 고정하고 hash를 보존한다. body는 선언된 가상 값이며 원본 native packet으로 꾸미지 않는다.

도로 형상·시드·조건 ID는 actor 수치 입력이 아니다. 영상 특징, 허용된 몸 변수와 고정 racing goal만 입력한다. canonical 방향 가설은 과거 실측 저속 곡률/조향 비율의 음수 부호에서 정했고, 현재 차량의 물리 반사실 정답을 새로 측정한 것은 아니다.

## 수행한 보정과 실패 보존

`functional-fit-plan-02.json`/`functional-fit-02`는 원래 fine motion projection만18회 업데이트했다. 기존 모델의 파라미터·버퍼, 측정 graph 연결·모든 근육→차량 readout·DN·정적 시각·native 몸 감각 경계와 정규화는 **정확히 고정**했다. TRAIN 질감 한 pass·seed781106·Adam LR1e-4·gradient clip1·원래 motion 가중치 대비 변화≤.001(float32 허용차1e-7)이다.

손실은 같은 몸의 좌/우 출력에 대한 방향 margin0.002의 hinge²×100과 원래 fine의 세 출력 anchor0.5×[4,1,1]이다. 제어 정답 line/map를 actor에 주입하지 않고, 합성 bend의 순서는 loss에서만 사용한다. prefix/초기 branch는 no-grad burn-in이며 마지막16frame에 gradient를 연결한다. 숨은 수치 상태는 gradient 경계에서 초기화하지 않는다. 기존 운전 모델/CNN으로 구조를 교체하지 않았다.

새 checkpoint는 `functional-fit-02/functional-motion.pt`, SHA **`27d21422843fe801dc26a7f4734efc1e56c53fe6c5ffa22f186e396461e35435`**다. 별도 `flybrain.functional_motion_port.v1` 연구 형식이며 live loader/actuator에는 연결하지 않았다. 기존 실제 연결 후보 `fc4a2997…`와 이전 체크포인트는 보존했다. 이 후보는미승격이다.

첫 `functional-fit-01`도18회 업데이트했으나 마지막 고정 model 검사에서 sparse tensor에 지원되지 않는 `torch.equal`을 사용해 실패했다. checkpoint가 저장되기 전이어서 복구하지 않았다. sparse edge index/value를 희소 상태 그대로 비교하도록 고쳐 **수치 학습 설정을 바꾸지 않고**02를 재실행했다. 따라서 실행한 optimizer는실패18+성공18=총36회이고 최종 후보가 이어받은 추가 업데이트는18회다. 실패/첫 계획/로그를 보존했다.

첫 GPU 합성 helper는 자료 저장 뒤 종료 단계에 남아 해당 identity만 확인해 retirement했다(`stimuli-01-termination.json`, 원래 handle97027 exit1). 후속 generator는 마지막 speculative prefix의 GPU 작업을 drain하고 tensor owner를 해제한 뒤 정상 종료하도록 수정했다. 두 번째 generator는 명시적 drain/release를 기록하고 원래 handle30477 **exit0**이다. 첫 helper를 정상 종료로 바꾸어 기록하지 않았다. 첫 검사 명령의 존재하지 않는 파일 이름 오류도 별도 로그에 보존했고, 올바른 검사24개는 통과했다.

## 기존 실제 기록의 보존 검사

`actual-preservation-01`은 소비된 nominal Monza 개발 기록의 원래 camera 특징·몸/native bytes·모든 시각으로 **16,269사건 전체**를 `MotionSpatialDecisionRuntime`에서 연속 재생했다. 새 물리 주행은 아니다.

| RMSE | 원래 fine 후보 | 기능 보정 후보 | 변화 |
|---|---:|---:|---:|
| 전체 조향 | .068762703 | .068754025 | 약.0126% 감소 |
| 진입 전 조향 | .060789692 | .060926600 | **약.2252% 증가** |
| 전체 가속 | .157353387 | .157364610 | 약.0071% 증가 |
| 전체 제동 | .152031885 | .152056151 | 약.0160% 증가 |

사전에 정한 전체3축/진입 전 조향의2% 이내 보존 gate는 통과하지만 **실제 조향 개선의 증거가 아니다**. 특히 진입 전 오차는 악화했다. readout 가중치 고정은 출력을 byte 동일하게 보존했다는 뜻이 아니다. 기록 재생 출력 몸 나이는최대19.472ms이며새 live 시간 검증이 아니다. 교사 주행 기록의 재생을 학생 자율 주행 성적으로 바꾸지 않는다.

## 독립 감사·현재 상태·다음 작업

`independent-audit-01`은 원래5개 고정 component의 sparse/dense tensor hash·normalization·가중치 변화 한계·18업데이트·TRAIN/test 질감 분리·body 온도 배열·같은 prefix 출력 배열·모든 근육→차량 수식과 방향 통계를 독립 확인했다. 최대 근육 수식 차이는9.594e-8이다. 원래16,269사건의 전체/진입 전 RMSE·보존 gate도 재계산했다. **이 감사가 전체 반사실 neural branch를 다시 실행한 것은 아니다.** 실행 source에 bound된 같은 시작 state 복원 검사는 본 실행의 hash/배열 증거다.

관련24검사와 독립 감사·원래 CPU 실행들은정상 종료했고, 기존6체크포인트/production 소스의 보존과 정확한 handle은 최종 영수증에 남긴다. 이번 실제 시뮬레이터 API/학생 actuator/UI 호출은0이다. 전체 목표active·미승격이며 한계 주행 완료가 아니다.

다음 보정은 TRAIN 조건에서 남은 위치/속도 반대 반응과 작은 전달 gain을 분석하고 실제 TRAIN 진입 전 조향 보존을 함께 다룬다. seed781105는이미 평가에 소비했으므로 후속 설정 변경의 새로운 독립 평가로 재사용하지 않고 새 질감/형상/상대 yaw 조건을 학습 전에 봉인한다. 실제 다중 라인·새 코너/트랙 평가도 여전히 필요하다. 합성 부품 반응 향상으로 실제 조향/온도 적응·몸 균형/가감속·변속/셋업/차종/상대풍·빠른 유효 완주/라인/랩타임·생리/R2 검증을 대체하지 않는다.
