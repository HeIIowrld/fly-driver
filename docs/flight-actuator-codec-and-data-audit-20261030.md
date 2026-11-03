# 실측 페달 cutoff·명시적 codec·학습 자료 감사

갱신: 2026-10-30, Asia/Seoul. 새 세션 t1676–t1684. 지속 목표는 **파리운전시키기**이며 현재 진행 중이다.

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 영상+허용된 몸 감각의 actor 경계를 유지한다. 일반 운전망·저속 부품 성공으로 목표를 바꾸지 않는다.

## 새로 확인한 결과와 한계

- t1672/t1673/t1674 원본의 native·SDK 바이트, 당시 실행 소스, 운동뉴런→intent→canonical→driver 경로를 독립 감사했다. t1673 전체 회로를 CPU에서 원래 시간대로 다시 계산했고 GPU intent와 최대 차이는 2.74129e-7이었다.
- t1681 실제 정차 계측 35단계/282관측/286메시지와 t1682 독립 감사를 완료했다. 양 페달에서 .0195까지 native0, .02부터 identity 응답이었다. cutoff는 **(.0195, .02]**다. 문턱 위를 이동시키는 선형 deadzone이 아니다.
- 원래 요청을 보존하는 명시적 최소 양의 페달 codec을 구현했다. 120개 encoding 기록과 이어진 driver 요청이 일치했고 정착 native 표본의 최대 오차는 1.19209e-8이었다. 정차/Neutral 시험이며 주행 감속이나 학습 성공이 아니다.
- 이번 optimizer 업데이트0, 시각 자율주행0, 학생 유효 랩0이다. 기존 실제 시각망 1,107회 학습/개발 오차13.33% 개선은 유지되지만 선행 판단 기준 실패도 유지된다.

## 기존 제어 원본의 독립 감사 — t1676

`20261030-t1676-independent-event-audit`은 실제 차량 매핑을 열지 않고 원본만 읽었다. 당시 `executed-program.py`와 manifest를 비교했으며 현재 파일 SHA를 옛 실행에 대입하지 않았다.

| 원본 | native prefix 해시·decode | SDK wire | AI-off 센서 표본 |
|---|---:|---:|---:|
| t1672 | 85 | 82 | 82 |
| t1673 | 75 | 71 | 71 |
| t1674 | 112 | 108 | 108 |

t1673의 운동뉴런→intent 독립 수식 오차는 최대1.43514e-8, intent→canonical→제출 요청은 정확히 일치했다. native에서 만든 몸21과 유효성도 정확히 일치했다. 전체 CPU 재생은69관측/68간격/3.69572초, 최대 원래 간격56.491ms를 유지했다. 이전 command/body/mask를 실제 지난 시간 동안 유지하고 첫 실제 관측에서 시간 적분0, 중간 상태 리셋·재타이밍·마지막 가상 간격0이었다. 근육 출력 최대 차이는4.65661e-10이다. 원래 gain/mix와 기록된 private 중립 offset을 구분했다.

양의 신경 brake 요청9개(약9.42e-7–.0198343)는 다음 native 표본에서 모두0이었다. 이를 신경 제동 성공이라고 부르지 않는다. 짧은 좌우 단계의 yaw에는 전환 응답이 남아 목표 yaw 정착 추종으로 판정하지 않는다. 요청 직후 첫 정리 native 표본이 이전 brake0을 보인 경우도 있어, SDK 요청 수락은 게임 적용 증명이 아니다.

SDK 원본의 publisher 누적 오류도 보존했다(t1672=0, t1673={0,1}, t1674=1). 비동기·희소 표본의 순서별 대응은 연속 적용 또는 정확한 지연의 증명이 아니며 native 반복 복사도 엔진 seqlock 인증이 아니다.

## codec의 정확한 의미

`src/flybrain_drive/actuator_codec.py`의 계약은 `flybrain.actuator.minimum_positive_projection.v1`이다. 페달의 보수적 표현 가능 집합을 `{0} ∪ [.02,1]`로 두고 가장 가까운 값으로 투영한다.

| canonical 페달 u | encoded 요청 |
|---|---:|
| 0 ≤ u ≤ .01 | 0 |
| .01 < u < .02 | .02 |
| .02 ≤ u ≤ 1 | u 그대로 |

동률인 .01은0을 택한다. 조향은 바꾸지 않는다. `.02`를 모든 요청에 더하지 않으며 duty cycling·PID·direct steering residual은 없다. raw neural intent와 canonical은 덮어쓰지 않는다. codec은 canonical/encoded/error/간격 여부/제곱 투영 거리를 기록하고 `game_application_verified=False`로 남긴다. 실제 적용값은 별도 native 관측에서만 읽는다.

미래 학습 제약은 각 페달의 gap에서 `min(u²,(u-.02)²)`, 그 밖에서는0인 표현 가능성 손실로 정의할 수 있다. 현재 기록된 제곱 거리는 진단용 NumPy 값이며 아직 미분 학습 경로가 아니다. 과제/운동 결과 손실 없이 이것만 줄이면 제동을0으로 만드는 해도 있으므로 학습된 감속으로 해석할 수 없다. 실제 움직이는 제동·하중 자료와 회로 경로 안의 학습이 필요하다.

`neural_actuator_component` 분기는 원래 `NeuralPhysicalProbe`의 고정 실측 화학 회로→운동뉴런→intent→canonical 뒤에만 이 codec을 붙인다. 시각망을 사용하지 않으며 **새 neural+codec 실제 이동 시험은 아직 하지 않았다.** 다음 native 관측을 직전 신경 요청의 정확한 연속 적용으로 단정하지 않는다.

t1682 두 번째 검토는 codec 함수를 사용하지 않고 투영 수식을 독립 계산했다. canonical120개·native286개·SDK282개·AI-off564개·driver 정수축286개가 일치했다. 요청281과 close286은 전용 다음 native 표본이 없어 개별 ACK로 세지 않았다. 마지막 정차3표본의 native ACK와 소유권 해제는 별도로 확인했다.

## 새 실제 계측과 보존한 실패

t1677에서 새 native3표본과 CSP를 확인했다. 엔진45604/generation37294426ULL:0.0003399, native AI-off/Neutral1/gas0/brake1/vJoy FREE1/비전면이었다. 실제 제어 스크립트도 매 시작·표본·제출에서 별도의 신선도·엔진·프로필·AI-off·소유권 가드를 사용했다.

첫 t1678은150ms 안에 새 SDK 파일이 오지 않아28관측/31메시지 뒤 종료1이다. 단계 표본은1–3개여서 사전 settling2 뒤의 충분한 steady 판정을 할 수 없다. 원본과 실패를 보존했으며 최종 native full-brake 정차와 vJoy FREE는 확인했다. 정착 cutoff 성공으로 포함하지 않는다.

revision2는 **정차/Neutral 시험에만** SDK 대기 .5초, 단계당 최대8/최소5표본·단계 기한1.5초를 명시했다. 실제35단계가 모두8표본으로 끝났다. 첫2개 settling을 미리 분리하고 나머지6개를 모두 검사했다. 원래 타임스탬프와 모든 과도 native를 보존했다.

| t1681 / t1682 | 결과 |
|---|---:|
| 새 raw cutoff 단계 | 20 |
| canonical→encoded 단계 | 15 |
| 전체 센서 / 드라이버 메시지 | 282 / 286 |
| 원시 native prefix 검증 | 286 |
| encoding→driver 기록 일치 | 120 |
| encoded steady native 최대 오차 | 1.19209e-8 |
| 최고 속도 | .000583145 km/h |
| 진단 world 위치 변화 최대 | .00502411 m |
| 센서 간격 중앙값 / 최대 | .1325901 / .3531599초 |

두 페달 모두 .015/.016/.017/.018/.019/.0195는0, .02/.0205/.021/.025는 identity였다. canonical .0165는 명시적으로 .02로 투영돼 native .0200000107이었다. .025는 .025000006으로 관측됐으며 .045로 이동하지 않았다. 배포 프로필·엔진은 같고 차는 Neutral이었다.

위 간격은 운동 회로의80ms 계약을 만족하지 않는다. 정차 시험의 대기 완화는 신경 시간 계약이나 실시간 주행 자격으로 이어지지 않는다. 원본 publisher 파일 쓰기/교체 지연·누적 실패를 남기고 다음에는 native 몸 감각 시계와 느린 진단 SDK 경로를 분리해야 한다. 과거 패킷에 새 시각을 붙여 이 문제를 숨기지 않는다.

검토에서 정차 codec 모드의 first→Neutral 선택이 잘못 upshift할 수 있는 분기를 발견해 `single_gear_transition(current, required)`로 고쳤다. 양 방향/동일 기어/거부 검사를 추가했다. 실제 t1681은 처음부터 Neutral이어서 결과는 영향받지 않는다. 이 수정의 first-gear 시작 분기는 단위 검사만 했고 새 실제 기어 전환 검증은 아니다. 당시 v2 계획·실행 소스는 보존하고 현재 소스와 맞는 다음 실행 계획은 `configs/ac-actuator-codec-v3-20261030.json`이다.

최종 변경 뒤 관련 **28검사와 Ruff 통과**. 기존 pytest의 Windows UNC sibling 경로 검사가8분 이상 걸린 원인을 traceback으로 확인했다. 명시한 파일 외의 수집만 `--ignore-glob=*`로 제외하면 같은28검사가 약2초에 통과했다. 외부 pytest/다른 작업 프로세스는 건드리지 않았고, 완료된 검사를 중복하던 이 세션 소유 두 test child만 식별해 종료했다.

## 시각 자료와 학습 경로의 부족 — t1680

개발·평가 배열을 열지 않은 TRAIN 전용 감사다. TRAIN3 전체2,217관측/90.212925초, 속도 목표는 모두60km/h였다. 몸속도50–70km/h1,949개/70km/h 이상0개다. driving-phase 교사 telemetry3,564개에서 `brake≥.02 && speed>20km/h` 표본은0개였다. 정차 출발·외부 full-brake는 움직이는 제동 학습 자료를 대신하지 못한다.

20m 유효 표적2,121개는 좌254/우784/거의 직선1,083이었다. 기존 진단 정의를 TRAIN에만 적용하면 진입5/탈출9/반대 방향41개이며 세 번째 이력의 진입·탈출은0이다. 모든 원래 마스크와 실패를 유지했다.

기존 학습의2간격 detach 미분 경로 중앙값은 약81.4ms, 20m/현재속도의 명목 시간 중앙값은1.1806초다. **수치 상태가 계속 유지되므로 기억이81ms라는 뜻은 아니다.** 선행 예측의 시간 credit를 점검할 가설이다.

`configs/flight-preview-train-only-temporal-plan-20261030.json`은 전체 이력별 내부3fold와 detach2 대 detach8을 사전 고정했다. 두 조건 모두 optimizer window8, 같은 window 동안 가중치 고정, 짧은 쪽은2간격 gradient 누적이다. 각 fold는 원래 저자 pretrained와 fold 전용 정규화/readout부터 시작해야 한다. 기존 adapted-author와 registered readout은 TRAIN3 전체를 이미 보았으므로 보류 이력 일반화의 초기값으로 쓰면 누출이다. 먼저8간격 상태·메모리 자격을 확인한다. 아직 새 학습 실행은0이다.

## 최종 상태와 다음 작업

t1683 읽기 전용3표본: 엔진45604/같은 generation/native AI-off/Neutral1/gas0/brake1/속도 .000440257km/h/vJoy FREE1/비전면, CSP age .02646초다. 이번 제어·감사 프로세스는 종료했다. 이전 미확정 소유17212/2328 및 외부 작업은 별도이며 종료하지 않았다. UI 조작0/설명용 HTML0/payload 중복 복사0.

다음은 native 몸 감각과 느린 SDK 진단 경로의 시계·소유권을 유지하며 neural+codec 실제 제동/하중 응답을 계측하고, 가감속·조향 결합을 실측 회로 안에서 학습하는 것이다. 동시에8간격 시각 학습 자격과 TRAIN 내부 temporal 비교를 진행한다. 새 독립 자료는 실행 전 역할·지역을 고정하고 양 방향 진입/탈출·움직이는 제동·60→100/가변속·기어·타이어/하중/셋업을 포함해야 한다. 알려진 DEV를 새 TRAIN으로 바꾸지 않는다. 유효 완주·빠른 레이싱 라인/랩타임 최적화 전체 목표는 유지한다.

## 원본과 소형 보존

원본은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/` 아래 t1676-independent-event-audit, t1677-fresh-admission, t1678-stationary-actuator-codec(실패), t1680-visual-data-audit, t1681-stationary-actuator-codec-v2, t1682-stationary-codec-audit, t1683-post-codec-state다. t1679는 사용하지 않았다.

소형 보존은 `runs/flight-actuator-codec-and-data-20261030-t1684`다. 이전 인계 t1667–t1675의 metadata·실행 소스도 포함한다. native/SDK 이벤트·큰 payload는 원본 경로·크기·SHA 참조로 보존하고 NPZ/PT/카메라 파일은 복제하지 않는다. 현재 소스와 과거 실행 소스를 분리한다.
