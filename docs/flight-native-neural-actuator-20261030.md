# Native 몸 감각 시계와 신경 페달 codec의 실제 연결

2026-10-30, Asia/Seoul. t1687–t1693. 이전 [페달 cutoff·codec 검증](flight-actuator-codec-and-data-audit-20261030.md)을 이어 진행했다.

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 이번 결과는 그중 실제 운동 부품과 액추에이터 경로다. 영상 자율주행이나 일반 운전망으로 목표를 바꾸지 않는다.

## 해결한 제어 주기의 문제

이전 정차 t1681은 SDK 파일의 새 표본을 기다려 관측 간격 중앙값132.6ms/최대353.2ms였다. 이를 운동 회로의80ms 계약에 맞도록 시간을 다시 붙이거나 계약을 늘리지 않았다.

이미 보존된 원본 AC 공유메모리 정의의 `SPageFileGraphic.carCoordinates`를 진단용으로 추가했다. Windows UTF-16/pack4 레이아웃에서 offset252/끝264바이트이며 AST로 원본 정의와 대조했다. upstream 모듈 자체를 import/실행하지 않고 기존 매핑만 읽는다. ACC의 여러 차량 좌표 배열을 대입하지 않는다.

`ac_native_position.py`는 일치하는 두 번의 복사를 확인한다. 같은 graphics packet을 다시 보면 처음 receipt를 유지하며 bytes가 달라지면 거부한다. packet reset·80ms 초과·live 상태 불일치도 거부한다. 반복 복사는 엔진 seqlock/물리 시각 인증이 아니다.

t1687 읽기 전용24표본에서 nativeXYZ를 SDK의 원래 축 순서로 대조했다. 최대 좌표 차이는 **.0000610352m**였다. t1688은 graphics24/physics24/SDK24/양쪽 AI-off48을 원시 bytes에서 독립 검증했다. 제어 입력0/vJoy 획득0이다.

새 `native_neural_actuator_component` 모드는 반복 제어에서 SDK 파일을 기다리지 않는다. physics와 decision은 같은 수신 측 perf_counter를 쓰고, 신선한 graphics의 XZ 위치는8m 이동 범위의 진단 가드에만 쓴다. 신경 모델에 전달되는 것은 기존 native physics→몸21과 명시적 물리 과제뿐이다. world 위치·지도·절대 heading을 actor에 넣지 않는다. SDK 초기 패킷 한 개는 producer identity 증거로만 남기고 이후 표본의 SDK는None이다.

vJoy 획득 전에40ms 이상 떨어진 두 정차 표본에서 physics와 graphics packet이 모두 증가하는지 확인한다. frozen 첫 표본에 지금 시각을 붙여 시작하지 않는다. 기존 새 엔진/같은 generation/양쪽 native AI-off/CSP 상태/프로필 SHA/비전면/vJoy FREE 및 각 제출 시 소유권 검사를 유지했다.

## 실제 t1689와 독립 t1690

고정 물리 과제의 순서는 baseline→목표속도1m/s→목표 yaw±.15rad/s→목표속도0이다. 이는 운동 부품에 주입한 과제이며 시각 예측을 사용하지 않았다. 기존 합성 quiet1초 초기 상태와 private 중립 offset을 명시했고 원래 실측 화학 연결·보호된 몸 경계·운동 gain/mix는 그대로다.

| 항목 | 원본/독립 검증 |
|---|---:|
| 신경 관측·제출 | 68 |
| 전체 센서 표본 / 드라이버 메시지 | 72 / 76 |
| 신경 사이 원래 간격 | 67 / 3.6801014초 |
| 신경 간격 최대 | 58.0442ms |
| native receipt→decision 최대 | 2.0236ms |
| graphics 처음 receipt age 최대 | 9.1374ms |
| 최고 속도 / 진단 위치 변화 최대 | 1.4508569km/h / .8144153m |
| 운동뉴런→intent 독립 수식 최대 차이 | 1.64478e-8 |
| 전체 CPU 재생→원래 GPU intent 최대 차이 | 2.45229e-7 |
| 근육 출력 재생 최대 차이 | 4.65661e-10 |

원본 native76/graphics72/양쪽 AI-off144/SDK1/정수 driver76개를 검증했다. 첫 실제 신경 관측의 시간 적분0, 이후 이전 command/body/mask를 원래 간격 동안 유지했다. 중간 reset·retiming·마지막 가상 간격0이다. 기어 변경 전후의 신경 에피소드 밖 표본까지58ms 간격이라고 쓰지 않는다. gear button 유지 때문에 전체 센서 기록에는242ms 간격도 있으며 그때 신경 제어는 시작 전이다.

## 작은 신경 제동의 실제 적용

네 개의 canonical brake(.01468477/.01880067/.01731378/.01584670)는 선언한 codec에서.02로 투영됐다. 다음 원시 native brake는 모두.0200000107이었고 당시 속도는.579–1.385km/h였다. 더 작은 다섯 요청은 명시적으로0을 택했고 다음 native도0이었다. .02 이상 여섯 요청은 값을 유지했다.

신경 intent/canonical/encoded/관측 native를 모두 따로 보존했다. 작은 출력을 가린 full-brake override나 `.02` 일괄 가산은 없다. 감속 단계 센서 속도는1.4177→.5793km/h였지만, 별도 coast 대조가 없어 감소량 전체를 제동 효과로 귀속하지 않는다. 이번 **optimizer 업데이트0이며 학습된 제동의 성공이 아니다.**

요청2(gear button), 마지막 신경 요청71, close76은 각각 전용 다음 native 표본이 없어 개별 ACK로 세지 않았다. 마지막3개 native의 gas<.02/brake>.98/속도<.1km/h와 vJoy FREE를 별도로 확인했다.

## 아직 해결하지 못한 것

- 목표1m/s에 실제 최고 속도는약.403m/s다. 목표 속도를 정확히 추종했다고 판정하지 않는다. 짧은 좌우 단계의 yaw 중앙값도 목표 부호로 정착하지 않아 단위 보정·회로 간 가감속/조향 결합 학습이 남아 있다.
- 몸21의 값과 마스크는 모두 보존됐지만 현재 motor branch의 causal 입력은 angular2:5와speed16이다. 하중8:12·타이어 온도12:16·바퀴 회전17:21은 이 경로의 출력을 바꾸는 입력으로 아직 연결되지 않았다. 기록을 하중·타이어 적응 성공으로 부르지 않는다. 정적 pitch/roll의 시각 정렬과 동적 yaw의 haltere 경로도 구분한다.
- t1693의 native XZ 경로 합 .8144475m와 host receipt 간격으로 적분한 속도 .8197461m의 차이는약5.30mm였다. 이는 이동 좌표의 독립적인 sanity check이며 움직이는 SDK/native의 정확한 지연·시각 정렬 인증은 아니다.
- 실제 초파리 시각망 선행 판단 기준은 여전히 미통과다. 이번은 camera-off 운동 부품이며 학생 시각 자율주행·유효 랩0이다. 60→100/가변속·변속·하중/타이어/셋업 적응·유효 완주·레이싱 라인/랩타임 최적화는 계속 필요하다.

관련31검사와 Ruff가 실제 제어 전 마지막 변경 뒤 통과했다. 실제 제어와 두 독립 감사/CPU 재생은 각각 terminal0이다. t1691 새 읽기 전용3표본에서 엔진45604/동일 generation/native AI-off/**nativegear2(1단)**/gas0/brake1/속도 .000441804km/h/vJoy FREE1/비전면/CSP age .12836초를 확인했다. 이전 t1683의 Neutral을 현재 상태로 반복하지 않는다. UI 조작0/외부 프로세스 종료0/원본 payload 복사0이다.

## 근거와 다음 단계

원본은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`의 t1687-native-position-qualification, t1688-native-path-review, t1689-native-neural-actuator, t1690-native-neural-actuator-audit, t1691-post-native-component-state, t1693-native-motion-sanity다. 실행 계획은 `configs/ac-native-neural-actuator-20261030.json`, 원본 구현/검사와 현재 구현을 각각 보존한다.

다음은 이 actuator 제약을 실제 학습 목적과 움직이는 제동·하중 과제에 연결하는 것이다. 신경 회로 바깥의 PID/direct actuator residual로 대체하지 않는다. 동시에 TRAIN 내부 이력별 보정·readout과 실제 시각망의8간격 미분 경로를 검증해 선행 시각 학습을 이어간다. 알려진 개발 자료를 학습으로 돌리지 않는다.

최종 t1700(시각 학습 종료 후)도 새3표본에서45604/nativeAIoff/1단/gas0/brake1/속도 .000466310km/h/vJoyFREE/비전면/CSP age .179224초를 확인했다. 이 읽기 전용 검사와 이번 제어·학습·감사는 모두 종료됐다.

현재 코드에 봉인된 정차 전용 재실행 계획은 `configs/ac-actuator-codec-v4-20261030.json`이다. native 진단 경로의 import/source 결합만 갱신했으며 v4 정차 계측 자체는 새로 실행하지 않았다. 과거 v2/v3 계획과 당시 실행 소스는 역사 자료로 유지한다.
