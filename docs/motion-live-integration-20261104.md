# 새 운동 시각의 실제 카메라·몸 감각 연결 — 2026-11-04

> **후속 2026-11-04 해석 제한:** 원래 반대 코너 쌍의 도로 횡위치/속도 혼입을 실제 원본으로 확인했다. 방향5/5를 독립 코너 인식으로 해석하지 않는다. [후속 기하·원본 영상 검증](precorner-road-pose-confound-20261104.md).

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·가감속·변속·하중/타이어/셋업/차종·검증된 상대풍을 유지한다. 측정 연결·modeled neural activity·공학적 readout과 학습한 차량 대응을 구별한다. 지도/world pose/절대 heading/교사 조작은 actor 입력이 아니다.

## 최종 실행 — 실제 새 경로·저속 움직임·독립 감사 완료

루트는 `runs/motion-live-integration-20261104/`다. `full-source-parity-03`은 전체1,552그룹/대응1,547그룹에서 scene 최대차1.0431e-7·새 motion5.9605e-8·coarse4.8429e-8이다. 이 source03과 연결한 `full-runtime-parity-03`은 **16,172연속 출력**·저장 학습 후보 최대차 **7.1526e-7**, static 차이0·출력 몸 나이 최대17.545ms·원래 CPU handle12474 exit0이다. source03 GPU helper handle80648은 계산 완료 뒤 별도 종료 영수증처럼 retirement했고 exit1이며 정상 종료 통과로 부르지 않는다.

`cold40-preparation-01`은 epoch7에서 SDK1회·세 고유 native 정지 packet·pause로 실제40℃ 시작 조건을 확인했고 `cold40-gear-01`은 그 영수증을 유지한 준비용1단 설정/재정지를 완료했다. 두 원래 handle18966/58184는 exit0이다. 데스크톱 입력/UI는0이다.

`cold40-motion-01`에서는 새 motion 입력이 실제 GPU camera→CPU owner로 전달됐다. 실제 외부 교사만10/20km/h 저속을 운전하고 학생은 read-only다.

| 항목 | 실제 원본/독립 감사 결과 |
|---|---:|
| 게시한 새 motion 영상 그룹 | 105 |
| 연속 학생 shadow 출력 | 531 |
| 첫 출력부터 마지막 출력까지 | 6.340480초 |
| 같은 영상에 추가된 몸 갱신 출력 | 426 |
| 출력 몸 나이 최대 | 21.937ms / 제한50ms |
| 출력 가장 오래된 영상 나이 최대 | 135.813ms / 제한150ms |
| native 최대 차속 / 타이어 이탈 | 19.816km/h / 0 |
| 실제 타이어 온도 범위 | 39.533–39.960℃ |
| 학생 actuator / UI / 추가 optimizer | 0 / 0 / 0 |

`cold40-motion-audit-01`은 original native first receipt·시야별 촬영/몸 정합·게시/연속 recurrence·50/150ms·교사 권한·실제 정지3고유 packet·camera/owner/native/교사 종료와 새 motion byte hash 연결을 독립 확인했고 **whole_shadow_passed=true**다. 새 CPU owner는 실제 `MotionSpatialDecisionRuntime`/motion 필수 ready였고, owner45844 exit0·forced=false·poisoned=false다. GPU 원래 child42760은 launcher의 보유 핸들에서 **exit0·forced=false**를 받았고 launcher tool58430도 exit0이다. 앞선 GPU-only 복원 helper 종료 실패와 이 실제 실행의 종료 통과를 구별한다.

`cold40-motion-owner-audit-01`은 실제로 보존한 scene/static/fine/coarse와 원래 native bytes·촬영 당시 몸 이력·모든 시각을 사용해 **531출력 전체를 중간 초기화 없이 다시 계산했고 최대차0**이다. 실제 publication→각 출력의 새 motion byte SHA와 coarse reference를 확인했다. 이것만으로 픽셀을 다시 계산한 것은 아니다.

별도 `cold40-motion-pixel-audit-01`은 실제 선택된105그룹의 원본 RGB와 시야별 촬영 시각으로 author FlyVis를 수동 연속 재생했다. 새 stream subclass를 호출하지 않고 scene/static/fine/coarse를 각각 계산해 실제 보존 입력과 비교했다. scene 최대차8.9407e-8·static0·fine/coarse각5.9605e-8이다. 따라서 **실제 원본 픽셀의 새 입력 재구성과 전체 몸/회로 출력 재생을 별도로 확인했다.** 세 독립 감사 tool4218/30393/49537은 모두 exit0이다.

이 조건의531출력에는 모두 기존 native 온도 학습 지원 범위 밖 표시를 보존했다. 저온 적응이 옳다는 검증이나 한계 주행 성과가 아니다. 같은 BMW/Monza의 짧은 저속 입력 통합 시험 한 번이며 **학생이 실제로 조향해 코너를 개선한 주행이 아니다**. 관련 **46검사 통과**(`integration-tests-03.txt`). 기존 여섯 checkpoint 보존과 소스 hash는 `source-integration-01.json`, 최종 원본/핸들/세션은 최종 영수증을 따른다. 전체 목표active·미승격이다.

## 무엇을 검증하는가

앞선 [새 운동 시각 학습](motion-spatial-input-learning-20261104.md)의 fine 후보를 그대로 실시간 입력 경로에 연결한다. checkpoint는 `fc4a2997867f75cffc6ad879379c5e365799dffdc46c68321af4188ac4130b19`이며 이번 추가 optimizer 업데이트는 **0**이다. 기존 회로·몸 감각·정적 시각·근육 readout 및 정규화를 덮어쓰지 않는다.

앞선 같은 몸/같은 시작 상태 개입은 새 motion만 제거하면 개발 진입 전 오차가 약3.045% 늘어났다는 근거다. fine의 기존 기준 대비 전체/진입 전 조향 오차 감소2.053%/2.933%도 앞선 학습 결과이며 이번 통합으로 새로 학습한 개선이라고 합산하지 않는다. matched coarse 대비 전체 추가 감소.586%는 사전1% 기준 미달이므로 **미승격**이다. 같은 Monza를 반복 사용했고 새 코너 일반화는 아직 입증하지 않았다.

## 구현 경계

`MotionPerViewFlyvisStream`은 기존 `PerViewFlyvisStream`이 이미 갱신한 **같은 author FlyVis 상태**에서 T4a–d/T5a–d의 tanh activity를 읽는다. 기존4×4 pool에 보이지 않는 공간 잔여를8×8로 읽어2,048개 특징을 만든다. 네 시야의 방향·공간·원래 촬영 순서를 보존한다. 추가 neural advance/reset이나 직접 조향 출력은 없다. 기존 scene576/static320 경로와 원래 stream 소스는 그대로다.

typed publication은 episode·generation·capture·camera receipt·네 source clock·기존 coarse motion512와 새 motion2048을 전달한다. 새 특징 byte SHA를 GPU 부모→CPU owner→각 출력 사건에 연결한다. 새 후보의 CPU owner는 `MotionSpatialDecisionRuntime`을 명시적으로 요구하며 일반 visual-only publication은 거절한다. 몸50ms/영상150ms와 단일 소유자의 연속 상태/실패 후 종료 계약을 유지한다. 전체 후보와 부모·고정 모듈·독립 감사·개입·미승격 상태를 loader에서 확인한다.

실제 실행 preflight에는 새 경로의 전체 원본 픽셀 재생과 동일 후보의 전체 native runtime 재생을 요구한다. 모든 프로그램/stream/prefix/geometry hash와 동일 원본 배열 및 양쪽 결과 연결을 검사한다. 영상 입력은 메모리에 보존하고 CPU 추론 종료 뒤에만 NPZ를 저장한다. 학생 출력이 actuator에 도달하는 경로는 추가하지 않았다.

## 전체 원본 재생

원본은 `runs/thermal-control-improvement-20261104/hot110-actor-data-02`다. 실제 live 경로가 쓰는 기존 `runs/thermal-driving-20261103/spatial-recovery-01/pool-geometry.npz`를 사용했다. 원래 부적격 시작 영상만 건너뛰고 나머지 **1,552그룹**을 원래 시야별 촬영 시각으로 연속 처리한다. 학습 입력과 대응되는1,547그룹·16,172모델 사건을 비교한다. source time을 새로 만들거나 episode 중간을 초기화하지 않는다. speculative held prefix도 실제 경로처럼 사용한다.

최종 원본 및 후속 실행의 숫자·종료 상태는 아래 최종 실행 항목과 `runs/motion-live-integration-20261104/`의 원본 결과/종료 영수증을 따른다. `full-source-parity-01/02/03`은 모두 계산 일치 결과를 저장했지만 GPU 도구가 종료 단계에 남았다. 실행 identity를 다시 확인한 **해당 GPU-only helper만** CIM으로 종료했고 기록을 보존했다. 이 helper들은 camera/native/control 자원이 없었다. 명시적 exit 위치 변경도 정상 종료를 보장하지 못했다. **계산 parity와 orderly helper 종료는 별개이며 세 helper를 clean exit로 부르지 않는다.** 실제 움직임 실행은 별도 원래 핸들과 owner/teacher/camera 종료를 확인한다.

원래 전체 native 재생01은16,172출력·저장 후보 최대차6.5565e-7·기존 static 차이0·출력 몸 나이 최대15.995ms이며 CPU handle exit0이다. 최종 source03과의 연결은 별도 runtime03을 사용한다. 이전 결과를 최종 live 증거와 혼용하지 않는다.

## 코너를 외운 것인가

현재 증거는 암기와 일반화를 분리하지 못한다. 반복된 코스에서 속도·회전·하중·이전 감각과 recurrent state는 진행 중인 조작을 재현하는 강한 단서다. 몸만 쓰는 이전 대조군은 현재 조향을 상당 부분 예측했지만 코너 진입 전 미래 조향에서는 좋은 결과가 아니었다. 이는 몸 안정화에 필요한 감각과 길을 미리 보는 판단을 별도로 검증해야 하는 이유다. 회로에 지도나 코너 번호를 넣지 않았더라도 반복 이력/시각 패턴을 외울 가능성은 남는다.

시각 영향이 작다는 결과를 영상이 완전히 불필요하다는 뜻으로 해석하지 않는다. 기존 공간 pooling의 손실과 근육 이전 경로의 약한 기여를 확인했고 새 잔여 입력을 보존/학습했다. 그러나 modeled activity의 공간 변동 에너지 손실률을 운전 정보 비중으로 부르거나, 특징 교환의 가설 방향5/5를 물리적인 운전 성공률로 부르지 않는다. 같은 몸에서도 조향이 달라지는 것과 바뀐 조향이 새로운 코너에서 옳은 것은 다르다. 실제 교사 완주는 학생 자율 주행 성적이 아니다.

## 남은 검증

이번 단계는 실시간 입력 통합·계산/지연/종료 검증이다. 이후에는 도로 상대 위치/자세를 맞춘 영상 개입과 아직 소비하지 않은 코너/트랙에서 선행 판단을 검증한다. 낮은 전체 오차만을 반복 튜닝해 이를 대체하지 않는다. 학생 폐루프에는 출력 단위·부호·준비 구간·단일 actuator owner·신선도·독립 정지 계약의 검증이 필요하다. 조향/가감속·몸 기울기/가속도/하중/타이어·변속/셋업/차종/상대풍 적응, 실제 학생 유효 완주·빠른 라인/랩타임·생리/R2 및 한계 주행 목표는 **진행 중**이다.
