# 차량을 가상 몸으로 받는 관측 v1 — 2026-10-29

[대전제](PROJECT_CHARTER.md)의 차체 기울기·회전·하중·타이어 상태 되먹임을 구현하는 단계다.
새 decoder/encoder는 구현했고 원본 기록 전체에 재생했다. 기존 AC 운전 actor에는 아직 연결하지 않았다.
기존 BODY_FIELDS와 체크포인트 입력은 변경하지 않았다.

## 새 입력 계약

설정은 `configs/ac-virtual-body-v1.json`, 구현은 `src/flybrain_drive/virtual_body.py`다.
BMW Z4 GT3 / AC1.16.4 / shared memory1.7의 검토 증거 해시와 입력 버전을 고정했다.

| 채널 | 변환/의미 | v1 유효성 |
|---|---|---|
| pitch 기울기 | native pitch, 앞부분이 위로 들리면 양수, rad | 두 기록의 기하 대조 통과 |
| roll 기울기 | -native roll, 왼쪽이 들리면 양수, rad | 두 기록의 기하 대조 통과 |
| body roll 각속도 | native localAngularVel[2] 후보 | 미통과, 값0/valid=false |
| body pitch 각속도 | native localAngularVel[0] 후보, 오른손 좌표계에서 앞부분이 내려가면 양수 | 미통과, 값0/valid=false |
| body yaw 각속도 | native localAngularVel[1], 좌회전 양수, rad/s | 두 기록의 대조 통과 |
| body 선형가속도3축 | native accG의 축/중력 보정이 필요 | 미등록, 값0/valid=false |
| 바퀴 하중 비율4개 | 각 native wheelLoad/네 바퀴 합계 | 무차원 비율; 총 하중 N 주장 없음 |
| 타이어 코어 온도4개 | native 순서/단위 그대로 | 적정 온도 범위나 성능을 가정하지 않음 |

body 축은 X전방/Y왼쪽/Z위다. 정적인 기울기는 각속도가 아니다.
기울기 pitch의 양수(앞부분 위)와 오른손 angular Y의 양수(앞부분 아래)도 구분한다.
이것은 차량 센서의 공학적 관측 계약이며 haltere 수용기의 실제 축/튜닝/위상 등록이 아니다.

## 원자료로 확인한 기울기와 각속도

`t1026`에서 t1002/t989의 원래 native308바이트와 카메라 renderer 자세를 대조했다.
prefix 개별 해시와 카메라 ZIP 전체 해시를 확인했다. renderer look/left/up으로 오른손 정규직교
body basis를 검증하고, world-up 투영에서 pitch/roll을 계산했다.
world 방향과 위치는 이 진단에만 쓰며 새 actor feature에는 넣지 않았다.

| 기록 | 기울기 비교 표본 | pitch slope/correlation/RMSE(rad) | roll slope/correlation/RMSE(rad) |
|---|---:|---|---|
| t1002 |893|0.9992 / 0.9977 / 0.001751|0.9992 / 0.99993 / 0.000277|
| t989 |948|0.9957 / 0.9981 / 0.001748|0.9992 / 0.99995 / 0.000253|

pitch에는 약0.0017rad(약0.1°)의 일관된 renderer/native 차이가 남는다. 이를 임의 보정하지 않았다.
실제 시험 범위는 pitch 약0.012–0.013rad,roll 약0.070–0.071rad다.
전복·큰 기울기·다른 차량의 포괄적 검증은 아니다.

각속도는 renderer 상대 회전의 회전벡터/producer 시간 차이를 native 평균과 비교했다.
yaw slope0.9952/1.0059,correlation0.9975/0.9970으로 통과했다.
roll/pitch는 자극 범위·기울기·상관 기준을 모두 만족하지 못했다.
이를 native 센서 오류라고 확정하지 않고 해당 body 축을 비활성으로 남겼다.

이 진단은 native 수신 시각과 front render 시작 하한의 차이가50ms이내인 표본만 비교했다.
범위 밖648/695표본은 진단에서 제외했으며, 원본/학습 에피소드에서 삭제하지 않았다.
시뮬레이션 노출 시각의 동기화나 물리 메모리 원자성을 입증한 결과는 아니다.
`t1025`는 expert_raw_native 이벤트를 누락한 파서 오류로 실패했고, 고친 t1026과 함께 보존했다.

## 시간·정보 경계와 전체 재생

- 동일 packet 재사용은 첫 수신 시각을 유지한다. 새 수신 시각을 붙여 신선한 표본으로 만들지 않는다.
- 같은 packet의 bytes 변경, episode 경계 없는 packet 역행, decision/receipt 시계 역행과 미래 표본은 거부한다.
- 호스트 수신 age50ms초과는 유효성을 전부false로 표시한다. 결측 채널은0과 명시적mask로 구별한다.
- heading/world position은 새 feature에 없다. raw accG를 중력 보정한 선형가속도로 바꾸어 부르지 않는다.
- 다른 차량/공유메모리 버전 및 검토 파일 해시 변경은 거부한다. 미통과 축의 mask를 임의로 켤 수 없다.

`t1027`에서 새 encoder를 재생하고, `t1028`에서 원래 봉인된 t1005 에피소드와 시각까지 대조했다.
**805학생+2인계대기+736교사=1,543관측 전체**, 원래 순서·decision 시각이 정확히 일치했다.
관측 삭제/시각 재부여/인계 시 encoder 초기화는0회다. 이 재생에서는 신경 모델을 실행하지 않았다.
pitch·roll·yaw·하중비율·native 온도는 모든1,543관측에서 유효했고,
미등록 두 각속도와 세 선형가속도 채널은 전체에서false였다.

관련 검사10개(기존 native ABI 검사 포함)와 Ruff가 통과했다. 테스트는 static tilt와 gyro 구분,
heading 비노출, 신선도 위조 방지, episode/packet 경계와 미통과 축 활성화 거부를 확인한다.
새 회로의 생리학적 균형 기능이나 실제 주행 검증을 대신하지 않는다.

## 증거와 다음 작업

- [기하 비교](../runs/ac-virtual-body-geometry-20261029-t1026/result.json).
- [전체 재생·봉인 시각 대조](../runs/ac-virtual-body-sealed-replay-20261029-t1028/result.json).
- [관측 계약](../configs/ac-virtual-body-v1.json).
- [비행 실측 후보와 전달물질/경계](fly-flight-route-candidate-20261029.md).

다음은 accG의 좌표·중력 포함 여부, 부족한 회전 축의 동적 자극 대조,
운동 후보의 표적·부호·동역학 검토와 겹눈 방향/시간 등록이다.
이 입력을 비행 유래 회로에 연결한 뒤 좌우 공통/차동 반응과 교란 안정화·지연·경로 제거를 검증한다.
그 구조로 조향·가감속과 레이싱 라인/랩타임을 학습하고 원래 서킷 완주·반복 조건을 검증해야 한다.
이번 실행의 실제 차량 조작·학습 업데이트·UI 조작은 모두0회다.
