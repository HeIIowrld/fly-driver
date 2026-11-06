# 정지 LIVE 온도 준비·물리 깨우기와 native 재개 진단 — 2026-11-04

> **후속 완료:** sampler를 먼저 연 owned resume와 명시적인 초기 과도 표본 관측을 구현해40℃/110℃를 각각 한 번의 SDK 설정과 native 전/후 정지3패킷·최종 pause로 확인했다. 이전 실패/원본은 아래 당시 상태로 보존한다. [현재 근거](native-initial-thermal-ready-20261104.md).

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·변속·타이어/셋업/차종·검증된 상대풍 적응, 영상+허용 몸 감각 actor 경계와 빠른 라인/랩 목표를 유지한다. 이번 것은 조건 준비/센서 진단이며 학생 주행 또는 생리 검증이 아니다.

## 현재 결론

새 LIVE 준비에서도40℃ native 조건을 확보하지 못했다. 차량 물리 깨우기 SDK의 반환은 확인했지만 그 뒤의 native3고유 정지 패킷이 없어 **온도 setter 전에 중단**했다. 이후 별도 resume 관측에서는 native가 새로 갱신되며 속도1.83213km/h를 관측해 정지 기준으로 중단했다. 따라서 native가 영구 정지했다거나 car sleep만이 원인이라고 단정하지 않는다. 다음은 resume 직후 초기 물리 상태의 안정화를 기록하고 정지3패킷 직후 설정을 이어가는 단일 준비 과정이다. 현재 actor/시각·몸 신선도 기준을 느슨하게 만드는 작업이 아니다.

## 출발점과 실제 배포

루트는 `runs/thermal-apply-diagnostic-20261104/`다. 앞선 `live-build-02`의 실제 설치 v2 SHA `b4336c4a36dfb97b82bf80330d9d32756f2c66509f41b8c952d736b81b51e2b7`를 정확한 base로 사용했다. 오래된 repo canonical Lua로 덮지 않았다. `live-build-03`은 전체 LuaJIT 컴파일을 통과했고 `live-deployment-02`가 paused/offline/AI-off/camera-off·정지 full brake·vJoy 반환 상태에서 한 번 교체했다. 이전 파일은 `installed-before.lua`로 보존했다.

**실제 설치 SHA는 `0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e`, 관측 generation은 `1117858ULL:0.0016212`다.** 마지막 파일 해시도 재확인했다. SDK 정의 `extension/internal/lua-sdk/ac_apps/lib.lua:14039`는 `physics.awakeCar`가 수면을 막고 현재 랩을 무효화한다고 설명한다. 이 기능은 명시적인 초기 준비에만 쓰며 제어/교사 주행/온도 thermostat으로 사용하지 않는다.

초기 준비의 LIVE/paused 온도 설정은 원래 한 epoch 한 번의 marker를 공유한다. 별도의 native wake marker도 첫 SDK 호출 전 소비한다. unknown/false/error 모두 같은 epoch에서 wake를 재호출하지 않으며, 온도 설정 후의 wake도 거절한다. native wake가 성공해도 native 준비 완료나 온도 적용 완료로 표시하지 않는다.

## 배포 전에 수정한 결함과 검사

미배포 v3 host에서 wake 옵션이 `--execute` 밖에 있던 결함을 발견해 수정했다. 현재 read-only에서는 wake/온도 요청을 쓰지 않는다. wake 직전 vJoy 반환을 다시 확인하고, wake 직전의 마지막 native packet보다 큰 새 packet3개를 요구한다. 실제 온도 setter 뒤에도 별도의 native3패킷/모든 타이어±2℃ 조건을 유지한다.

- Lua와 host/계약/receipt 관련71검사 통과(`wake-tests-01.txt`). Lua에서 고의 error를 던지는 시험의 Windows SEH 진단 출력은 보존했으며 pytest 최종 exit0이다.
- 마지막 pre-wake snapshot 수정 뒤 host/계약/receipt17검사 통과(`wake-host-tests-final-01.txt`).
- 실제 실패 후 중복/거절 native 관측도 보존하도록 host를 보완하고 host/readiness11검사 통과(`wake-host-observation-tests-01.txt`). 이 기록 개선은 **아직 새 실제 준비 실행에서 검증하지 않았다.** 완료된 원본은 실행 당시 host를 `executed-entrypoint.py`로 hash-bound 보존했다.
- 새 wake receipt 검사로 과거40/80/110℃ 원본을 재감사(`historical-raw-audit-wake-regression-01`)해 원래3조건/각3native 패킷의 정적 준비 증거를 보존했다. 과거의 성공을 현재 재현 성공으로 옮겨 쓰지 않는다.

## 실제 실패와 보존한 원본

`live40-02`는 새 source의 epoch1에서 restart/start/resume 뒤 native wake 한 번만 호출했다. source의 `awakeCar` 반환 영수증·같은 epoch marker를 확인했다. 그 뒤1초 안에 새 native3패킷을 확보하지 못했고, `temperature/result.json`의 lifecycle_requests_sent=1은 **wake만** 의미한다. 온도 `request.json`/`api-receipt.json`은 없고 온도 설정 marker도 소비되지 않았다. parent의 thermal_setter_command_invocations=1은 host command를 시작했다는 뜻이지 SDK 온도 setter가 호출됐다는 뜻이 아니다. 마지막 API pause는 완료했다. 이 epoch의 wake를 재시도하지 않았다.

그 실패의 `live-before-native.json`은 빈 배열이라 window 동안의 중복 native 원본을 재구성할 수 없다. 이를 복원한 것처럼 만들지 않았다. 향후 host가 모든 pre-window 원본을 메모리에 보관한 뒤 finally에서 저장하도록 수정했다.

`native-progress-01`은 같은 epoch에서 온도/wake/조작 없이 API resume 한 번, native graphics↔physics↔SDK bracket 관측, API pause를 수행했다. 총387관측 중 SDK live3관측에서 그래픽3packet/physics2packet, 마지막 새 native 속도1.8321285km/h·gas0/brake1/AI-off를 관측했다. 마지막 표본에서 .1km/h 정지 기준을 넘겨 원래 시험은 **실패**다. 완전한 live3초 비교나 정지 준비 성공으로 바꾸지 않는다. 원래 SDK resume 명령은 completed지만 예외 경로에서 자식 exit code를 별도 stages에 저장하지 않은 결손도 남았다. 기록 코드는 후속 수정했으며 과거 exit code를 만들어 넣지 않았다.

`wake-failure-audit-01`은 원래 실행 소스 hash·한 번의 wake receipt·온도 호출 전 실패·원본 native/graphics bytes와 hash·마지막 정지 기준 거절·API pause를 독립 재계산했다. **실패 원본 감사의 성공이며 조건 준비/주행 성공이 아니다.** source/physics 정확 동시 시점·MMF atomicity·수면/정지의 단일 원인은 입증하지 않는다.

## 최종 관측과 다음 작업

`final-status-01`은 실제 source에서 paused/AI-off/camera-off를 다시 확인한 read-only 조회다. 이번 물리 준비 단계의 사용자 창 활성화/클릭/키보드·vJoy 조작·AI 켜기·온도 setter는0이다. lifecycle API는 이 단계에서 restart/start/resume/final-pause와 별도 진단 resume/pause를 사용했다. wake SDK는1회다. 게임 창으로 자동 fallback하지 않는다.

동시에 진행한 시각 학습은 [별도 보고서](native-visual-anticipation-20261104.md)의 두 분기 각825회이며 이 온도 시험의 성적으로 합치지 않는다. 같은 native19·몸 이력·비행 회로·근육 readout을 보존한 공간 시각 경계 학습 뒤 전체 재생/같은 몸 영상 개입이 필요하다.

전체 목표 active. 다음 온도 단계는 **sampler를 먼저 준비한 단일 host 소유 resume→초기 비정지 표본 원본 보존→유효 정지3패킷→단일 LIVE 온도 설정→독립 native3패킷 검증**이다. 준비 중 비정지/결측 표본을 actor에 넣지 않으며 setter/awake 반복으로 결과를 덮지 않는다. 실제 학생의 다중 온도/하중·자세·가속도 한계 주행, 가감속·변속·차종/셋업/상대풍 적응과 유효 완주/레이싱 라인/랩타임은 아직 미완료다.
