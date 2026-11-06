# 냉간40℃·고온110℃ native 초기 조건 재현 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 변속·타이어/셋업/차종·검증된 상대풍 적응을 유지한다. actor는 영상과 허용 몸 감각을 사용하며 지도/world pose/절대 heading/교사 조작은 교사·평가 전용이다. 측정 연결·기능 근사·학습한 차량 경계를 구별한다.

## 완료한 범위

현재 설치 source에서 **40℃와110℃ 초기 조건을 각각 SDK 한 번으로 만들고**, 설정 전/후의 서로 다른 native 정지 패킷과 실제 네 타이어 온도±2℃·최종 API pause를 확인했다. `inline-conditions-raw-audit-02`가 원래 bytes/hash와 요청/불변 terminal·source generation/epoch를 독립 재계산했다. 이는 **정적 조건 준비**의 검증이다. 이 온도에서 학생이 주행/완주하거나 한계 주행에 적응한 결과가 아니다. 초기 준비 랩은 제외하며 training_admissible=false/actor_qualified=false를 유지한다.

루트 `runs/thermal-apply-diagnostic-20261104/`. 실제 설치 Lua SHA `0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e`, generation `1117858ULL:0.0016212`다. 이번 단계는 Lua를 다시 바꾸거나 physics.awakeCar를 호출하지 않았다. 이전 awake 실패와 원본은 [별도 보고서](thermal-live-wake-and-native-progress-20261104.md)에 보존한다.

| 실행 | epoch | 설정 전 정지 native | 설정 후 연속 정지·목표±2℃ | 마지막 고정된 네 타이어 |
|---|---:|---:|---:|---|
| `live110-01` | 3 | 고유3packet | 연속3packet | 109.9738..109.9759℃ |
| `live40-04` | 4 | 고유3packet | 연속3packet | 39.99118..39.99196℃ |

두 parent/thermal child는 모두 정상 종료했고 독립 원본 감사도 통과했다. 각 실행의 actual setter request는1회이며 재설정/반복 thermostat이 아니다. source native wake marker는 과거 epoch1의 기록이며 이번 두 epoch에서 사용하지 않았다.

## 구현한 순서와 구분

`scripts/run_live_thermal_preparation.py --resume-inside-thermal --settle-after-thermal`은 새 epoch restart/start/pause 뒤 `control_initial_thermal.py`에 이미 native reader를 열게 한다. 단일 host가 API resume를 발행하고,5초 이내에 초기 움직임을 원본으로 기록하며 동일 source/epoch·AI/camera/probeoff·vJoy FREE·zero gas/full brake를 검사한다. 초기 관측은 명시적으로5km/h 이내 준비 envelope이며 actor에 넣지 않는다. **실제 온도 설정의 admission과 마지막3패킷은 원래 .1km/h 미만 정지 조건 그대로**다.

정지한 서로 다른3packet을 확보한 뒤에만 LIVE 온도 setter를 한 번 발행한다. setter 뒤에도 초기 비정지 표본은 별도 `post-thermal-observations.json`으로 보존하고 연속 카운트를 끊는다. 유효 정지 표본에서 실제 네 타이어 온도를 검사해 연속3개가 모두±2℃일 때만 준비 완료다. 마지막 API pause와 고정된 온도도 확인한다. 원래 actor body50ms/view150ms 기준, 신경 상태 유지, 무효 센서의 명시적 처리와 별도 학습/카메라 적격성은 변경하지 않는다.

명시적 옵션 없이 기존 host 동작을 바꾸지 않는다. `--execute`가 없으면 owned resume/wake/온도 설정 모두 발행하지 않는다. native/SDK AI·source/epoch·제어기 소유권·브레이크 유지가 깨지면 관측을 계속 통과시키지 않는다. unknown SDK 결과는 재발행하지 않는다. 실제 하드웨어 제어·사용자 창 활성화/클릭/키 입력은0이다.

## 과도 표본과 실패를 숨기지 않았다

설정 전 준비 관측은 두 실행 모두45개였고 각각34개가 아직 ready가 아니었다. 관측한 최대 native 속도는110℃ 실행2.05825km/h·40℃ 실행2.07194km/h였다. 이 표본을 정지로 바꾸거나 학습 입력으로 허용하지 않았다.110℃ post-setter 관측3개는 모두 정지였고,40℃는9개 중5개가 비정지였다.40℃의 유효 온도 표본은 총4개지만 중간 비정지로 연속성을 끊어 **마지막 연속3개**만 완료 조건을 만족했다.

선행 `live40-03`은 설정 전3정지 packet을 확보하고 SDK를1회 호출했으나 첫 post-setter native 속도 .16910km/h로 실패했다. 해당 표본의 실제 네 타이어는 약39.998℃였고 같은 frame의 SDK getter는 아직80℃였다. SDK immediate getter/after-state만으로 실제 물리 반영을 판단하지 않는 이유다. 원본 실패를 그대로 두고 `live40-03-readback-01`이 같은 generation/epoch2의 원래 receipt만 참조해 새 정지3packet에서39.36..39.38℃를 확인했다. SDK setter 재호출0이며 `live40-03-raw-audit-01`이 확인했다.

초기 감사는 API immediate after-state가 이미 unpaused여야 한다고 잘못 가정해 실패했다. API 호출 반환 뒤 다음 engine transition에서 상태가 바뀔 수 있으므로, 실제 original observation window의 live heartbeat와 native3증거를 요구하도록 수정했다. 실패 영수증/소스는 보존했고 지연 transition 회귀 검사도 추가했다. 이어진 공동 감사01은 유효 표본4개와 과도 표본을 별도로 모두 확인했지만 보고 숫자에 유효 목록의4를 사용했다. 감사02는 post-transition 순서로 재구성한 연속3을 보고한다. 원본/감사01/parent proof는 덮지 않았고 admission 결과는 바뀌지 않았다.

## 검사와 실행 상태

관련49검사 통과(`inline-thermal-tests-final-01.txt`): 재개 직후 비정지/중복/지연 API 상태, actor와 분리한 준비 관측, source/epoch/AI/브레이크/제어권 손실, post-setter 비정지 카운트 재설정, execute 없는 무변경, 기존 thermal/native/readiness 계약을 포함한다. 독립 raw 감사는 실제40/110℃의 사전/사후 raw bytes·새 packet·원래 첫 receipt/요청/SDK calls를 확인했다. exact engine atomicity나 실제 GPU 노출 시각은 이 증거로 주장하지 않는다.

`inline-final-status-01`은 마지막 paused·AIoff·cameraoff를 read-only로 재관측했다. 모든 이번 소유 설정/검증/시각 재생·감사 핸들의 종료를 확인했다. 마지막 현재 온도 조건은40℃다. source/실행 소스/원본 해시·실패 보존·exit 영수증은 `inline-source-receipt-final-01.json`에 남긴다.

## 시각 모델과 다음 작업

이번 optimizer 갱신은0이다. 앞선 시각 학습 두 후보의 같은 몸/시작 상태 개입을 이번에 완료하고 각48,956사건/237진입 전 라벨·독립 감사를 통과했다. 새 미래 목표 후보의 spatial 경계 중립 시 진입 전 오차는4.44..12.23% 증가했고, 학습 전 projection으로 돌리면 .57..4.65% 증가했다. **시각 경계의 조건부 기여이며 전체 운전의 시각 비중이나 실제 반사실 운전 정답이 아니다.** 방향 반응은 여전히6/15이고 소비된 Monza의 반복 자료다. [시각 결과](native-visual-anticipation-20261104.md).

전체 목표 active. 다음은 현재 새 Lua hash와 이 준비 방식을 주행 중 read-only 학생 추론 경계에 연결해40/110℃의 실제 몸/영상 정합·native 감각 반응을 검증하는 것이다. 이후 실제 학생의 조향·가감속·하중/타이어·변속/셋업/차종/상대풍 적응, 별도 봉인 새 코너/트랙, 유효 완주와 빠른 레이싱 라인/랩타임을 검증한다. 기존 교사 저속 pilot이나 정적 조건 준비로 학생 한계 주행·생리/R2 완료를 대신하지 않는다.
