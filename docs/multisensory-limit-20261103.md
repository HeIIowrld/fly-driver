# 시각·몸 감각 기반 한계 주행 — native 감각 확장, 2026-11-03

> **후속 2026-11-03:** thermal 시작/재개 gate 연결·새 세 교사 시연·native19 경계825회 학습과 독립 감사까지 완료했다. 이 문서의 미학습/다음 단계는 당시 상태다. 가감속 오프라인 오차는 줄었으나 조향0.58% 악화, 실제 학생/고온 적응/빠른 랩은 미검증이다. [후속 원본과 한계](thermal-driving-learning-20261103.md).

**활성 목표: 시각 인풋을 기반으로 다른 타이어 온도와 자세 가속도 등의 감각을 이용하여 한계 주행을 할 수 있는 모델을 개발한다.** 이번에는 감각 입력/회로 경계 부품을 검증했다. 추가 경계는 **미학습·optimizer0회**이며, 목표 전체를 완료한 것이 아니다.

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 하나의 연속 회로, 변속·차종/타이어/셋업·검증된 상대풍, teacher/critic/평가 전용 map/world pose 경계를 유지한다. PROJECT_CHARTER는 변경하지 않았다.

## 확인한 자료와 구현

원본은 기존 적격 BMW Z4 GT3/Monza 교사3시연이다. 원본 native308/420-byte packet·camera bundle·qualification·actor 배열 hash를 확인했고, 새 관측48,636개를 원래 body packet/최초 receipt에 정확히 연결했다. 기존 body21와 acceleration false mask, 원래 성공/실패/계약은 수정하지 않았다. 반복 Monza 자료이며 새로운 조건의 일반화 시험은 아니다.

추가19개 입력은 가속도3축, 네 바퀴 native slip, 절대 하중, 코어 온도, 압력이다. 순간 원본 packet만 사용한다. slip은 **native scalar**이며 slip ratio/angle·마찰 활용도로 이름을 바꾸지 않는다. 압력도 검증되지 않은 native 단위다. 하중N·코어 온도℃의 근거는 [설치된 CSP SDK 정의](<C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/extension/internal/lua-sdk/ac_apps/lib.lua:3945>)와 원본 source witness다. SDK 파일 SHA는 각 계약에 고정했다. 공식 SDK가 설치된 정의 파일을 제공한다는 안내는 [CSP 공식 저장소](https://github.com/ac-custom-shaders-patch/acc-lua-sdk#how-to-start-writing-scripts)를 따른다.

코드:

- `native_multisensory_extension.py`: 원본 byte decoder, 차종/버전·근거 hash·관측 범위·50ms freshness, 같은 packet의 최초 receipt 보존, episode reset/역순/미래 샘플 거부. body21와 별도 schema다.
- `multisensory_body_feedback_port.py`: train-only 정규화·명시 mask와 온도×slip/하중×slip 상호작용을 기존201개 감각 입력에 보낸다. 추가 logit delta≤.25,0초기화·전체 결측 시 정확히0이다. 경로 판단/차량 출력에 직접 연결하지 않는다.
- 압력·온도·하중 및 가속도는 차량 대응 기능 가설이다. 실제 초파리 감각 수용기 tuning이나 새 측정 연결을 재현했다고 부르지 않는다. 기존 측정 회로/어댑터·여섯 날개근 요약/차량 readout을 보존했다.

## 수직축 판정과 시간 대역 보정

첫 `raw-sensor-audit-01/`은 원본 native 가속도와 ±75ms 속도회귀 미분을 비교했다. 전후/좌우는 고속 상관 .995/.997 수준, 수직은 .898 수준으로 사전 기준 미달이었다. 이 결과와 2축 `extension-data-01/`을 보존했다.

하지만 순간 가속도와 시간 평균된 속도 미분은 대역이 다르다. `bandwidth-audit-01/`은 불규칙 원본 시간에서 속도 최소제곱 미분과 정확히 대응하는 **가속도 적분 가중치**를 계산했다. 기준/비교 신호의 시간 평균 방식을 맞췄으며 부호·9.80665배 단위 계수·허용오차 기준은 바꾸지 않았다. 불규칙 가속도를 적분한 해석적 검사가 이 관계를 확인했다.

| 원본 시연 | 전후 상관 | 좌우 상관 | 수직 상관 | 축별 RMSE(m/s²) |
|---|---:|---:|---:|---|
| train1 | .999965 | .999983 | .999164 | .0439 / .0420 / .0429 |
| train2 | .999967 | .999987 | .999289 | .0414 / .0379 / .0383 |
| 소비된 dev | .999968 | .999980 | .999288 | .0413 / .0462 / .0404 |

속도>20m/s 조건이며 최대 관측 속도는 약266km/h다. 물리 transform을 사용하여 renderer의 외삽 자세와 구분했다. 이 비교의 미래 속도·world 좌표/pose는 **진단 전용**이다. 모델에는 현재 raw native G를 고정한 `[2,0,1]`, 부호`[+,+,+]`,9.80665배로 변환한 값만 들어간다. actor가 미분/미래값/world heading을 받지 않는다.

3축 `extension-data-02/contract.json`은 schema v2로 이 보완 근거를 별도 hash로 고정했다. 두 버전 모두 기존 body21의 가속도 마스크를 바꾸지 않았다. 관측된 BMW 자세/속도 범위 밖은 추가 가속도 유효성을 명시적으로 내린다. 공통 시간 대역·중앙 body frame 근사 범위의 평가이며, 고주파 물리 atomicity·큰 bank/grade·다른 차종·실차·생리 등록을 완료한 것은 아니다. 기존 roll/pitch 각속도 false mask도 그대로다.

## 타이어 온도와 한계 주행의 남은 조건

원본의 감독 라벨 구간 코어 온도는 바퀴별로 약68.25–98.87℃다. 변화가 있지만 같은 Monza 세 번의 위치/가감속/열 이력과 엮여 있다. **냉간/고온 초기조건을 독립적으로 바꿔 접지 한계 적응을 검증한 자료는 아니다.** 정규화는 첫 두 시연의5,621감독 라벨에서만 계산했고 세 번째는 이미 소비된 개발이다. 온도가 학습 범위를 벗어나도 실제 온도 감각을 지우지 않고 별도 OOD 표시를 남긴다. 최적 온도나 마찰계수 정답을 actor에 주지 않는다.

다음 수집 설계는 단순 동일 시연 증류를 반복하기보다, 예를 들어40/60/80/100/110℃의 **초기 상태**를 나눠 실제 온도/하중/slip/가속도 반응을 기록하는 것이다.80℃가 최적이라는 가정은 하지 않는다. 조건/에피소드/트랙으로 분리하고 실제 온도 궤적이 겹치는 경우를 숨기지 않는다. 현재 자료가 없는 냉간·고온 적응을 구현됐다고 판정하지 않는다.

CSP SDK의 `physics.setTyresTemperature()`와 `physics.setTyresBlankets()`는 현재 랩을 무효화한다([설치 정의](<C:/Program Files (x86)/Steam/steamapps/common/assettocorsa/extension/internal/lua-sdk/ac_apps/lib.lua:13939>)). 따라서 제어권 해제·정지·브레이크·오프라인·세대/epoch·지원 capability를 확인하는 새 lifecycle 경계가 필요하다. **native 확장 단계 당시에는 `control_ac_session.py`에 thermal command가 없었다.** 후속 [실제 초기조건 검증](initial-thermal-conditions-20261103.md)에서 이를 구현·배포하고40/80/110℃ native 코어 실측을 확인했다. 새로운 주행 수집·학습은 아직 미완료다. SDK 함수가 있다는 이유로 기존 CLI가 지원한다고 쓰지 않는다. 냉간 조건에서는 blanket이 목표를 다시 바꾸지 않는지 확인하며, 시간 측정 랩 중 온도를 계속 덮어쓰지 않는다. 개입/준비 랩과 이후 공식 유효 측정 랩을 분리한다. 실패·invalid flag와 sensor 중단을 보존하며 완주 분모에서 숨기지 않는다. 마찰계수/표면 grip·지도·미래 곡률은 teacher/critic/평가에만 둔다.

당시 엔진 상태 조회는 stale CSP heartbeat로 거절됐다(`session-status-01/`, lifecycle/UI/actuator0). 독립 프로세스 조회에서 그 시점 AC 엔진이 없었다. 자동 데스크톱 fallback이나 무근거 재시작은 하지 않았다. 당시 다음 작업이던 background bootstrap/배포·세션/제어권 검증은 위 후속 보고서에서 실제 진행했다. 오래된 저장 heartbeat를 현재 실행 상태로 사용하지 않는다.

## 회로 부품 실행과 증거

- `streaming-audit-01/02/`: 두 schema에서 각각48,636개 관측을 raw byte streaming encoder로 독립 재구성, packet/최초 receipt·축/유효성·train-only 정규화를 확인했다.
- `zero-port-replay-01/02/`: 원래16,293사건을 연속 재생했다. 새 body port delta는 정확히0, source 예측 최대차1.3411e-7, 기존51개 모델 파라미터/버퍼는 동일하다. **추가 경계 미학습/optimizer0회**, actor_qualified=false다.
- `tests-final-01.txt`: 새 schema/대역 가중치/결측·상호작용/기존 센서 경계18검사 통과. `summary-01.json`, `source-receipt-01.json`, `final-state-01.json`에 원본·실행 소스·해시·종료 상태를 남긴다.
- 3축 미학습 checkpoint: `zero-port-replay-02/body-extension.pt`, SHA `9aa750b1da5bb4692d292cac7f9e4909ad318edd05f824a5c2daed3a57953f11`. 기존 source는 `runs/steering-precorners-20261103/optical-pre16-fit-01/steering.pt`다. 기존 성공/실패·원자료와 V1을 보존했다.
- `extension-data-02/result.json` 종료 문구의 "two-axis"는 초기 템플릿 표현이 남은 것이다. 실제 v2 계약·19개 valid열·독립 감사는3축이다. 원본 결과를 보존하고 이후 생성기의 문구만 축 수에서 계산하도록 고쳤다.

## 전체 목표 판정 — 아직 미완료, active 유지

| 요구 | 현재 증거 | 남은 검증 |
|---|---|---|
| 시각으로 경로/선행 조향 판단 | 기존 겹눈·운동 시각/비행 회로, 알려진 Monza의 제한적 시각 개입 | 새로운 코너/트랙·도로 상대 위치 대조, 실제 학생 주행 |
| 자세·가속도·타이어 감각 | 3축/타이어19 감각 경계와 기존 회로의0중립 연속 재생 | 추가 경계 학습, 외란·결측 반응, 실제 몸 안정화 |
| 다른 온도에서 접지 대응 | 약68–99℃ 원본 범위·온도/slip/load 상호작용 부품 | 독립 초기조건 자료, grip/dynamics 학습·열 조건 밖 평가 |
| 한계 주행 | 강한 교사의 기존 유효 랩 근거만 있음 | 실제 actor 지연/shadow/폐루프, 조건별 유효 랩·한계/복구·실패 분포 |
| 빠른 라인/랩타임·차종/타이어/셋업·변속 | 장기 대전제/로드맵 유지 | 학생의 반복 검증과 조건별 최적화·일반화 |

이번 결과는 **감각 부품/오프라인 회로 경계 검증**이다. 생리 기능·차량 실제 통합·자율 완주/랩타임 검증을 대신하지 않는다. 당시 다음 작업이던 thermal lifecycle 초기조건 경계는 후속 정지 검증을 완료했다. 현재 다음은 조건별 주행 자료 확보, 새 body port/감각-운동 결과 학습, 시각 선행 판단을 보존한 실제 폐루프 검증이다. 검증된 상대풍도 별도 출처/축/시간 계약을 유지하며 차속을 바람으로 바꾸어 부르지 않는다.
