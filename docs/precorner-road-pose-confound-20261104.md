# 진입 전 영상 교환의 도로 상대 위치·속도 혼입 — 2026-11-04

> **후속 2026-11-04:** 위치/속도를 분리한 합성 시각을 실제 author T4/T5와 같은 몸/회로 prefix에 넣어 기존 motion 포트의 기능 보정을 실행했다. 별도 질감의 방향 순서는늘었지만 목표 반응 크기/실제 진입 전 조향 개선은 미달이다. [후속 학습·보존·한계](controlled-visual-function-learning-20261104.md).

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·가감속·변속·하중/타이어/셋업/차종·검증된 상대풍을 유지한다. 실측 연결·modeled activity·생리 근거·공학적 변환을 구별하며, 지도/world pose/절대 heading/교사 조작은 actor에 넣지 않는다.

## 결론

앞선 [fine motion 특징 교환](motion-spatial-input-learning-20261104.md)의 방향 순서5/5는 코너 모양의 독립적인 올바른 인식 근거로 쓸 수 없다. 원래 다섯 쌍은 몸/시작 회로 상태를 고정했지만 **영상 제공 차량의 도로 내 횡위치와 속도**가 달랐다. 기존5/5·시각 제거 효과·반복 코스 오차 감소의 원자료는 보존한다. 여기서 발견한 혼입이 앞선 조향 오차 개선을 취소하는 것은 아니며, 그 개선을 새로운 코너에 대한 일반화로 해석하지 못한다는 뜻이다.

원본 nominal 개발 기록의86개 진입 전 사건에서, 원래 반대 코너 영상의 source bracket endpoint 차이는 다음과 같다. 도로 기준은 기존 외부 교사 geometry CSV의 좌·우 경계 중점으로 만든 **도로 중앙선 근사**다. 접선 방향과 차체 방향의 상대각을 따로 계산했다.

| 항목 | 사건별 worst observed endpoint 차이의 중앙값 | 최대 |
|---|---:|---:|
| 도로 횡위치 | 6.696m | 6.947m |
| 도로 접선 대비 상대 방향 | 0.009868rad | 0.024396rad |
| 속도 | 9.819m/s | 13.644m/s |

결과를 보기 전 고정한 허용 차이는 횡위치 **0.25m**, 상대 방향 **0.03rad**, 속도 **2m/s**를 모두 만족하는 것이다. 전체86사건의 source bracket은 적격이지만 **조건을 만족한 사건0/86·전체 적격 구간0/5**다. 원래 쌍 외에도 같은5개 구간 사이의 반대 방향 후보를 모두 비교했다. **총12개 ordered donor 후보 중 각 receiver 구간 전체를 만족하는 후보도0**이다. 이를 다른 기록/모든 실제 코너에도 대응 쌍이 없다는 주장으로 확장하지 않는다. 기준을 느슨하게 바꾸거나 가장 가까운 쌍을 적격으로 바꾸지 않았다.

## 원본과 시각 근거

루트: `runs/road-pose-causality-20261104/`.

- `matching-plan-01.json`: 사전에 고정한 세 matching 기준·source-copy5ms·pose bracket60ms·자동 시간 warping/기준 완화 금지.
- `development-geometry-01`: 기존 적격 actor 배열·fine intervention 결과에 연결된 원본 TCP6,642packet, 대응 camera1,548그룹. 모든 그룹이 source bracket60ms를 만족했다. 최대 source-copy span0.2872ms·움직이는 source의 road tangent 방향 일치1.0이다.
- `development-geometry-audit-02`: TCP wire SHA와 내장 native heading/speed/packet, source-copy QPC, 카메라 촬영 전후 bracket ID, **3,096개 source point**의 exhaustive nearest-segment projection을 별도 계산했다. 최대차1.1946e-13이며 원래5쌍/86사건과 대안12개 후보의 matching 결과를 확인했다. 첫 감사01은 원래5쌍만 확인했고,02에서 대안 후보까지 추가 확인했다. 둘 다 정상 종료했다.
- `development-pixels-01`: 각 구간 진입 전 약0.45초의 실제 front 원본 RGB를 receiver/donor로 나란히 복사했다.10개 원본 member SHA를 보존했고, 픽셀을 옮겨 도로 위치를 맞추거나 새로운 장면을 합성하지 않았다.

![원본 receiver와 반대 코너 donor: 같은 행에서도 도로 내 위치와 속도가 다르다](../runs/road-pose-causality-20261104/development-pixels-01/original-pairs.png)

각 행 왼쪽이 receiver, 오른쪽이 기존 opposite donor다. 표기의 offset은 source bracket 평균의 도로 중앙선 대비 signed 횡위치이며, 영상의 픽셀 위치를 직접 측정한 값이 아니다. 왼쪽/오른쪽 가장자리 쪽으로 달리는 영상이 서로 교환된다는 점을 원본에서 확인했다. 이미지의 외형만으로 독립적인 조향 정답을 정하지 않는다.

## 정보·시간 경계와 한계

위치·절대 heading·지도는 **분리된 diagnostic NPZ에만** 저장했다. 기존 actor-only NPZ, 체크포인트, 학습 손실·정규화·카메라 계약·실제 루프는 수정하지 않았다. 새 모듈은 actor에 import/연결하지 않는다. 이번 optimizer/시뮬레이터 API/학생 actuator/UI 호출은 모두0이다.

각 카메라의 원래 네 render clock 전체를 source-copy 시각으로 전후 bracket했다. 비교는 두 bracket의 모든 endpoint 조합 중 최대차를 사용한다. 이는 **기록된 SDK 위치와 native heading/speed의 sampled-source 비교**다. 정확한 SDK position 물리 tick, 연속 노출 중 실제 경로, 위치의 중간 overshoot가 증명된 것은 아니다. 외부 CSV의 경계 중점도 실제 렌더 도로를 다시 측정한 ground truth가 아니다. 따라서0.25m를 보장된 물리적 노출 오차로 표현하지 않는다. 이 한계와 별도로 관측된 약6–7m 횡위치 차이는 기존 방향 반응을 순수 코너 방향 효과로 분리하지 못했음을 보여 준다.

## 다음 학습·검증의 변경

현재 자료에서 코너 방향과 차로 내 위치가 함께 바뀌므로, 같은 반복 코스의 방향 반응을 더 높이는 튜닝으로 이 문제를 해결했다고 판정하지 않는다. 다음 자료는 **코너 방향×차로 내 위치×도로 상대 방향×운동 시각 속도**를 분리해야 한다. 타이어 온도·자세·가속도·하중 등 몸 되먹임을 약화하거나 teacher pose를 actor 입력에 추가해서 조건을 맞추지 않는다.

현행 네 시야·author T4/T5와 새 fine motion 입력을 유지한 controlled visual factorial로 부품 반응을 먼저 구분하고, 원래 실제 몸 이력을 동일하게 고정한 분기를 비교한다. 합성 도로/변형 영상은 그 출처를 명시하고 실제 새 코너/트랙 검증으로 대체하지 않는다. 실제 자료에서도 도로 상대 위치·방향·속도가 겹치는 반대 코너/다중 라인 조건을 확보하고, 학습 전에 별도 새 코너 평가를 봉인해야 한다. 필요한 coverage가 없으면 정확한 반사실 조향 성공률을 만들지 않는다.

현재 **5 unit 검사·두 기하 독립 감사·원본10이미지 확인**을 완료했다. 기존 시각 경로의 실제 연결/531출력 timing 감사는 앞선 결과이며 이번에 다시 실행한 성적이 아니다. 이번 단계는 잘못된 인과 해석을 줄이고 다음 데이터 조건을 바꾼 진전이다. 실제 학생 조향 개선/폐루프·새 코너 일반화·온도 적응/한계 주행·생리/R2·빠른 유효 랩은 여전히 미검증이며 전체 목표active·후보 미승격이다.
