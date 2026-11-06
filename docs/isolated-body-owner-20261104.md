# native 감각·스케줄링·회로의 단일 CPU 소유자 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·변속·타이어/셋업/차종·검증된 상대풍·teacher/평가 전용 지도/pose 경계를 유지한다.

## 구현과 실제 통과 범위

`NativeFlightBodyOwnerProcess`는 기존 CPU 회로뿐 아니라 native 수집·첫 receipt·몸 패킷 선택·몸 속도 갱신도 독립 CPU 프로세스에서 소유한다. `shadow_native_flight_unified_owner.py`는 카메라 정합에 쓰는 과거 몸 감각까지 그 소유자에게 요청한다. 현재 몸21/추가19와 촬영 당시 몸 이력이 **하나의 실제 native 수집 기록**에서 나온다. 같은 영상에서 최신 몸 상태로 회로를 갱신하며 재귀 상태를 중간 reset하지 않는다.

시각은 원래 네 촬영 사건 순서의 author FlyVis·정적 공간 잔여·실제로 경과한 시간만 commit하는 held prefix를 유지한다. prefix를 포함한 backend 준비는 실제 수집 전에 하고, 실제 시각 회로는 새 author .5 steady prior에서 시작한다. 기존 측정 비행 topology·여섯 날개근 readout·native-fit 가중치는 고정했다. 새 optimizer·차량 조향/페달·GUI 명령0이다. 공학 어댑터와 측정 연결·생리학적 검증을 구분한다.

`runs/isolated-body-owner-20261103/unified-shadow-01`은3초 관측 전체와 종료를 마쳤다. `audit-unified-01`이 원본 RGB/source-body bracket·25ms host bound·native 첫 receipt·현재/이전 몸50ms·화면150ms·publication 순서와 모든 종료를 독립 확인했다.

| 항목 | 독립 재계산 결과 |
|---|---:|
| 정상 출력 | 266 |
| 첫 출력부터 마지막 출력까지 | 3.02054초 |
| 같은 영상의 추가 몸 갱신 | 214 |
| 최대 출력 몸 나이 | 18.942ms |
| 가장 오래된 화면 나이 최대 | 135.044ms |
| 회로 계산 p95 / 최대 | 12.747 / 16.075ms |
| whole_shadow_passed | true |

이는 **한 번의 짧은 실제 정지 shadow**다. 학생이 차를 조작하거나 완주한 성적·온도 적응·장시간 안정성·움직이는 영상/몸 정합·정확 GPU 노출/physics atomicity·생리 검증으로 확대하지 않는다. 타이어의 학습 지원 범위 밖 표시를 유지했다.

## 스케줄링과 보존한 실패

이 폴더는11월3일 UTC/11월4일 KST에 걸친 작업이며 기존 이름을 보존했다. 초기 분리형은 현재 몸 수집과 촬영 정합 수집의 두 기록을 썼고, 최종 통합형은 하나의 기록을 쓴다. 원래 시각과 실패를 고치거나 덮어쓰지 않았다.

- `stationary-shadow-01`: 새 화면이 소비되기 전에 만료·출력0. 새 영상 publication이 도착해도 ticker의 시간 대기가 바로 깨지지 않는 부분을 수정했다.
- `stationary-shadow-02`:82출력 뒤 held body50ms 거절. 기록에서 계산 뒤 약16ms 추가 대기가 관측됐다.
- `wait-resolution-probe-01.json`: 이 호스트에서 요청1ms의 Condition/Event 대기 중앙값은 약15.9ms, Python sleep은 약1.8ms였다. 전역 Windows timer 정책은 변경하지 않았다. ticker는 짧은 sleep을 사용하고 이미 목표 주기가 지났으면 추가 대기하지 않도록 바꿨다. 실제50/150ms 검사는 그대로다.
- `stationary-shadow-03`: CPU 쪽 몸 갱신 오류 없이3출력 뒤 GPU 부모에 남은 카메라 정합용 native 소비가 stale/queue 오류로 중단했다. 현재/과거 감각까지 단일 CPU 수집기로 통합한 이유다. 해당 실패 감사의 늦은 카메라-context는76ms였고 원래 validity는 전부 false였다. 실패 증거 감사에서만 이를 기록하며 정상 source/actor 통과로 바꾸지 않았다.

초기 실패들은 각각 `audit-stationary-shadow-01/02`, `audit-stationary-shadow-03-v2`에 남겼다. 첫03 감사의 source-context 신선도 거절도 `audit-stationary-shadow-03`에 보존했다. v2는 원래 실패 상태와 false mask를 기록하는 감사이며 whole_shadow_passed=false다.

## 긴 반복 시험과 종료 문제

`unified-shadow-02`는5초 설정에서426출력을 생성했다. 출력 구간4.94579초, 같은 영상의 추가 몸 갱신342개, 몸 나이 최대18.975ms·화면135.616ms·계산 p95 13.313ms였다. 추론 중 신선도 거절은 없었다.

하지만 native producer가 종료 join에서 끝나지 않아 소유자도 정상 종료하지 못했다. **whole_shadow_passed=false, cleanup_verified=false**다. 소유자 프로세스13424는 실행 코드의 실패 정리 경로에서 강제 종료됐고, 남은 읽기 전용 native producer44832는 PID/부모/실행 경로를 대조한 뒤 명시적으로 종료했다. `failed-sampler-retirement-01.json`에 receipt를 남겼다.426출력을 종료 성공의 증거로 쓰지 않는다.

종료 큐의 미소비 기록을 actor에 넣지 않고 별도 보존·배출한 뒤 producer를 join하는 `drain_on_stop` 옵션을 추가했다. 기존 기본값은 유지하고 새 소유자에서만 켠다. `owner-retired-native-copy.json`은 중단 후 격리한 기록이며 actor_admitted=false다. 새 회귀 검사는 미소비 기록의 원본 byte/sequence·admission 수 불변·pump와 producer 종료를 확인했다. 원인 후보인 feeder 종료 대기를 다룬 수정이며, 과거 hang의 모든 원인을 확정한 것은 아니다.

종료 수정을 적용한 실제 재시험은 아직 하지 못했다. `unified-shadow-03-process.txt`의 새 시도는 run 생성 전 기존 카메라 결과와 heartbeat의 불일치에서 거절됐다. 이전 producer의 불변 결과는 completed/240프레임인데 camera heartbeat는 같은 request ID의 failed/`Cannot commit camera metadata`다. `camera-admission-conflict-01.json`에 원래 자료를 보존했고 gate를 완화하거나 metadata를 덮어쓰지 않았다. 이 시도는 새 capture가 아니다.

실패한 GPU preflight의 실제 Python 자식은 종료했으나 venv redirector27292가 계속 대기해, 정확한 실행/부모/명령을 확인하고 redirector만 정리했다. WMI 목록과 네이티브 process table이 달랐던 첫 정리 시도 및 최종 wrapper receipt도 보존한다. 새 실행을 중복 시작하지 않는다.

## 근거·현재 상태·다음 작업

`final-tests-01.txt`의93검사는 기존 회로/카메라/시간 계약·ticker·prefix·native sampler와 새 종료 격리를 포함한다. owner IPC의 가능한 모든 장애나 장시간 실시간 성능을 검증한 것은 아니다. `source-receipt-final-01.json`/snapshot·실행별 entrypoint/manifest·원본/감사·정리 receipt를 보존했다.

`final-state-01.json`은 네이티브 process table에서 관측한 소유 프로세스 없음, 신선한 paused·AI/cameraoff·gas0/brake1 상태를 기록한다. 카메라 heartbeat 불일치는 미해결로 별도 보존했다. 단일3초 통과·긴 시험의 종료 실패·종료 수정의 단위 검사 통과를 구별한다. actor_qualified=false, 전체 목표 active다.

다음은 정확한 기존 completed 결과와 메타데이터 오류에 대한 증거 기반 진단/복구 계약을 만들고, 종료 격리를 실제 긴 shadow에서 다시 검증하는 것이다. 이후 움직이는 감각/시각 정합·같은 몸의 선행 시각 기여·새 코너/트랙·학생 폐루프 조향/가감속·하중/타이어·변속/차종/셋업/검증된 상대풍·빠른 유효 랩을 검증한다. native-fit의 조향 악화와110℃ 실제 감독 부족은 여전히 남아 있다. 정지 실행 안정화가 운전 학습 목표를 대신하지 않는다.
