# 서로 다른 타이어 시작 온도의 실제 초기화와 원본 감사 — 2026-11-03

> **후속 2026-11-03:** thermal 시작/재개 gate 연결·새 세 교사 시연·native19 경계825회 학습과 독립 감사까지 완료했다. 이 문서의 미학습/다음 단계는 당시 상태다. 가감속 오프라인 오차는 줄었으나 조향0.58% 악화, 실제 학생/고온 적응/빠른 랩은 미검증이다. [후속 원본과 한계](thermal-driving-learning-20261103.md).

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 변속·차종/타이어/셋업·검증된 상대풍과 teacher/critic/평가 전용 지도·world pose 경계를 유지한다.

## 결과와 범위

실제 BMW Z4 GT3 / Monza 세션에서 **40℃·80℃·110℃ 초기화**를 별도 epoch 1/2/3으로 실행했다. 각 조건마다 네 바퀴 native 코어 온도가 목표±2℃인 서로 다른 패킷 3개를 확인했다. 원본420바이트와308바이트 하위 prefix의 SHA, AI 상태, native 조작 상태, 패킷/host receipt와 SDK/resume receipt를 별도 감사했다.

| 목표 | epoch | 첫 패킷 네 바퀴 범위(℃) | 세 번째 패킷 범위(℃) | native 확인 |
|---|---:|---:|---:|---|
| 40℃ | 1 | 39.99909–39.99917 | 39.99702–39.99728 | 3 distinct packets |
| 80℃ | 2 | 79.99802–79.99817 | 79.99058–79.99130 | 3 distinct packets |
| 110℃ | 3 | 109.99538–109.99574 | 109.98382–109.98510 | 3 distinct packets |

원본: `runs/thermal-initial-20261103/summary-01.json`, `temperature40-02/`, `temperature80-01/`, `temperature110-01/`, `raw-audit-02/`. **정지 초기조건 준비의 실측 검증**이며 냉간/고온에서 운전하거나 학생을 학습한 결과가 아니다. 신규 native19 감각 경계는 여전히 미학습이고 전체 목표는 active다. training_admissible/actor_qualified=false, student optimizer0이다.

## 구현과 보호한 경계

- `integrations/ac_csp/initial_tyre_temperature.lua`: 오프라인·BMW/Monza·같은 generation/epoch·일시정지·정지·AIoff·카메라/probe/settle 비활성·full brake·released-controller 선언·요청 TTL·20..120℃를 검사한다. 제어권 반환은 호스트가 SDK FREE1을 별도로 검사한다. 한 producer/epoch에서 한 번만 적용하며 부분 실패도 epoch를 소모한다. 새 ID로 재시도하지 않는다.
- SDK `setTyresBlankets(0,nil,false)` 다음 `setTyresTemperature(0,nil,T,15)`로 네 바퀴의 표면/코어 초기화를 요청한다. 실제 온도 검증은 native 코어에 한정한다. current lap은 제외한다. 설치 SDK는 두 함수의 현재 랩 무효화를 명시한다. **공식 lap-valid flag를 직접 읽은 것은 아니며 false를 숨겨 유효 랩으로 세지 않는다.** 측정 랩 중 반복 온도 제어는 없다.
- `scripts/build_initial_thermal_csp.py`: 오래된 저장소 앱으로 덮지 않고 실제 설치된 최신 source-body/render witness 포함 Lua를 해시 고정해 추가 경계만 삽입한다. `build-01/deployed-base.lua` 및 `deployment-01/previous-installed.lua`에 기존 배포본을 보존했다. 전체 후보 LuaJIT 컴파일 후 엔진 부재를 두 번 확인하고 배포했다.
- 배포본 SHA `b2227739fb11d6ccce2020e976b2e0b3626d41ecd7dd8defb42ea8105a0495c2`, 부모 SHA `a99ef883eb2a37ce40db66c482817c9781d5754eb35039065ec62bb138f663ed`. 실제 엔진에서 `initial_tyre_temperature_v1=true`를 관측했다.
- `scripts/control_ac_session.py`에 `set-tyres-temperature-initial`, `verify-initial-tyres-temperature`, `--temperature-c`, `--initialization-run`, `--resume-for-readback`를 추가했다. 구현은 `control_initial_thermal.py`와 `initial_thermal_contract.py`다. 기존 lifecycle은 유지한다. 명시적 새 Lua hash와 같은 background engine·native same-copy AIoff/정지/full brake·SDK FREE1을 다시 확인하고 정확한 local intent를 게시 전에 기록한다.
- SDK 정상 반환을 온도 적용 성공으로 취급하지 않는다. `--resume-for-readback`는 온도 setter와 별개의 CSP resume ID를 한 번 발행하며, **재개 뒤 NAS 쓰기보다 native 관측을 먼저** 한다. 세 패킷의 실제 네 바퀴 온도가 일치해야 성공이다. host receipt는 원천 물리 시각이나 engine atomicity의 입증이 아니다.
- `scripts/audit_initial_thermal_evidence.py`: 결과의 true 선언을 믿지 않고 원본 바이트에서 온도·AI·가스/브레이크·속도를 재구성한다. 요청/세대/epoch/호출 순서/온도 목표·three distinct packet·해시와 불변 receipt를 대조한다. 두 조건 감사01과 세 조건 감사02 모두 완료했다.

## 실패와 실제 엔진 상태

직접 hidden cold launch PID16116/25152는 종료 신호와 exit53을 확인했다. 사이의 cold-launch-02는 race.ini hash 변경을 검출해 launch_attempts0으로 중단했다. 설정의 지속 변경을 임의로 허용하지 않았다. 이후 별도 런처가 생성한 PID20572를 발견하고 원래 실패 핸들과 구분했다. 새 프로세스의 신선한 CSP/native·비전면·AIoff·brake1을 확인한 후 CSP start/restart/pause/resume API만 사용했다. GUI 활성화·클릭·키 입력·프로세스 강제 종료는 0이다. 직접 실행이 Steam 재실행으로 이어질 수 있다는 [공식 Steamworks 설명](https://partner.steamgames.com/doc/api/steam_api?l=english)은 관측과 양립하지만 exit53의 원인을 확정하는 근거는 아니다. 앱 ID 파일/라이선스 우회·실행 파일 변경은 하지 않았다.

`session-pause-01`은 start 후 실제 brake0을 보고 요청 전 거절됐다. 기존 검증된 neutral bootstrap으로 zero steer/zero gas/full brake만 적용한 뒤 API pause를 수행했다. 전체 작업의 vJoy 제출은 **83회 모두 neutral/full brake**이며 모델/교사의 주행 조작은 0이다. lifecycle 요청은20회다.

`temperature40-01`은 SDK 호출 후 일시정지 중 패킷이 멈춰 실측 timeout으로 보존했다. 별도 resume/읽기 실행 `temperature40-readback-01`은 시작 준비/NAS 지연 동안 실제 온도가 약38℃까지 내려와 3연속±2℃ 조건을 못 채웠다. 허용 오차를 완화하거나 같은 epoch에서 setter를 재전송하지 않았다. 명시적 즉시 resume/readback 경로를 구현·검사한 뒤 새 epoch에서40℃를 다시 초기화해 통과했다.

`final-state-01.json` 관측 시 엔진 PID20572는 살아 있고 paused·AIoff·camera inactive·vJoy FREE1·비전면이다. 해당 실행은 자동 엔진 재기동 성공으로 바꾸어 기록하지 않는다. 다음 작업 시작 시 실제 프로세스/heartbeat를 다시 확인한다. `source-receipt-01.json`/`source-snapshot-01/`과 모든 실패/성공 원본을 보존했다.

## 검사와 전체 목표의 다음 요구

`tests-final-02.txt`: 기존 lifecycle/세션/native AI 및 새 Lua/host/readback 경계 **100검사 통과**. `compile-01.json`: 전체 후보 LuaJIT 컴파일 통과. Lua의 의도된 오류 주입에서는 Windows fatal-exception 형식의 진단 출력도 발생했으나 해당 실패 처리 assertion과 이후 검사들이 완료되고 프로세스는 exit0이었다. 원문을 그대로 보존한다. 이 출력이나 검사만으로 실제 차량 동역학을 입증하지 않는다.

기존 `evaluate_sac_teacher_curriculum.py`의 카메라 시연 시작 계약은 **75..85℃·첫 전진 기어·autoshifter·full brake**를 요구한다. 따라서 초기화 API를 얻었다고 현재 수집기를 냉간/고온에 바로 사용할 수는 없다. 다음 구현은 기존 warm 기본 경계를 유지하며 **검증된 thermal 초기화 run·같은 generation/epoch·현재 실측 시작 온도·변경된 조건의 명시적 분리**를 요구하는 별도 수집 조건이다. 단순 온도 숫자나 목표값을 actor에 넣지 않는다.

이후 조건별 정합된 전후좌우 영상/자세/가속도/하중/slip/타이어 감각과 교사 조작·실패를 수집하고, 원본 clock/packet과 actor/teacher 경계를 독립 감사한다. 이전68..99℃ 자료의 정규화/OOD를 새 train-only 근거로 갱신하며 값을 조작해 범위에 맞추지 않는다. 같은 비행 회로의 native19 경계와 시각 선행 판단을 함께 학습하고, 조건별 감각 제거/교환·새 코너/트랙·실시간50ms guard·폐루프 한계 주행/유효 랩·빠른 라인/랩타임을 검증한다. 이번 정지 준비와 기존 오프라인 조향 개선으로 그 요구들을 완료 처리하지 않는다.
