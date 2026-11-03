# 실제 AC 정차 축 응답과 센서 앱 복구

갱신: 2026-10-27, Asia/Seoul. 사용자 재개 승인 뒤 원래 AC 엔진을 종료하고, vJoy의 중심·
가속0·제동1을 유지한 상태에서 검토한 controls.ini를 적용해 Z4 GT3·Monza를 다시 실행했다.
`INPUT_METHOD=WHEEL` 후보의 해시 일치와 새 엔진 창을 확인했다. 자율주행 정책은 실행하지 않았다.

## 실제 게임의 응답

`runs/ac-axis-session-20261027-t39/`와 복구 후 `runs/ac-axis-session-20261027-t40b/`에서
각8단계·64개 요청/응답 표본을 수집했다. 각 실행의 physics 원시 prefix128개를 다시 decode해
SHA-256·기록 필드·증가하는 packet ID와 대조했다. 요청 sequence도 연속적이었다.

| 요청 | 실제 native 센서 중앙값, 두 실행 동일 |
|---|---:|
| 조향 +0.1 | steerAngle +0.140624985 |
| 조향 -0.1 | steerAngle -0.140624985 |
| 제동 1 / 0.5 / 0 | brake 1 / 0.5 / 0 |
| 가속 0.1, 제동 유지 | gas 0.100000024 |
| 마지막 중심·가속0·제동1 | steerAngle 0, gas 0, brake 1 |

두 시험 전체 속도 최댓값은0.03714km/h 미만이었다. 관측한 조향의 raw gain은1.40625지만,
이것을 도로 바퀴의 rad 단위·차량 좌회전 부호로 해석하지 않는다. 정차 중 작은 범위의 축 전달을
검증했으며 실제 이동·조향 한계·응답 지연·백그라운드 적용·자율주행은 별도 검사다.

초기 중립 유지와 종료 요청을 포함한 SDK 전송은 첫 실행1785건·두 번째1363건이었다.
축 시험 뒤에는 중심·가속0·제동1을 요청하고 소유권을 해제했다. 종료 요청 자체는 게임 적용
보증이 아니며, 마지막 시험 표본에서도 해당 센서값을 확인했다. 각 실행의 `response-audit.json`
및 `configs/ac-axis-probe-result.json`에 검증 범위를 기록했다.

## 앱 파일 출력 중단과 복구

재실행 때 `_socket` 없이 동작하는 SDK 파일 출력이 실제로 시작됐다. 첫 파일 기록은88표본 뒤
PermissionError, 읽기 재시도 수정 뒤 기록은56표본 뒤 갱신 시간 초과였다. 이후 `acs.exe`와
native 매핑이 없어졌고 파일 sequence는8128에서 멈췄다. 게임 log에는 Python의
`PyTraceBack_Print`/`PythonInterface::printErr` 중 crash가 있었다. 파일 교체 충돌의 미처리
예외와 연관된 것으로 추정하지만 crash의 단일 원인을 확정한 것은 아니다.

`FileTelemetryClient`는 Windows 공유 충돌에 한해 유한 기한 안에서 다시 읽도록 수정했다.
embedded 앱은 publish 예외를 AC callback 밖으로 보내지 않고 횟수를 보존하며, 다음 update에서
다시 시도한다. 실패를 새 센서 표본으로 바꾸지 않으며 소비자는 갱신이 없으면 시간 초과한다.
수정 앱을 다시 배치·재실행했다. 이전 파일/로그와 배치 해시는
`runs/ac-file-publisher-fix-20261027-t40/`에 남겼다.

복구 후 `runs/ac-file-live-recovered-20261027-t40c/`에서200개의 증가하는 SDK 파일 표본을
수집하고 원시 wire 해시를 검증했다. 마지막 실시간 점검은 engine window1개, native packet
29648→29650, 파일 steps4306→4318이었다. `publisher_errors=6` 이후에도 새 표본이 왔으므로
발행 실패가 있어도 이 짧은 구간에서는 엔진과 발행이 계속됐다. 장시간 안정성을 증명하지는 않는다.

관련 file publisher/reader·축 시험·vJoy 검사9개와 Ruff를 통과했다. 전체 목표인 비전·몸 감각
파리 회로의 실제 AC 서킷 완주와 실제 가감속 정책 학습은 아직 달성하지 않았다.

## 다른 작업 중 비활성 창의 축 적용

사용자가 전면 창 조작 때문에 다른 작업을 못 한다고 지적해, 이후 창 활성화를 중단했다.
`runs/ac-background-axis-20261027-t41/`는 GUI 조작 없이 vJoy와 native 매핑만 사용했다.
기록66개 모두 `was_foreground_at_record=false`였고,64개 축 표본·128개 원시 physics prefix를
재검증했다. 조향±0.1→raw±0.140625, 가속0.1→약0.1, 제동1/0.5/0이 모두 재현됐다.
즉 이 정차 조건에서는 게임을 전면에 유지할 필요가 없다. 최소화·주행 중 적용은 미검증이다.

현재 차가 멈춘 것은 정책 주행을 실행하지 않고 정차 계측을 마쳤기 때문이다. 내장 AI 시험의
단축키 시도에서는 native `isAIControlled`가 false로 남았다. 내장 AI 교사 주행이나 파리 학생
주행을 시작했다고 표시하지 않는다. 이 flag는 수집/평가 메타데이터이며 actor 입력에 추가하지 않는다.
후속 운용은 사용자의 전면 작업을 유지하고 native 제어/계측·오프라인 학습을 중심으로 진행한다.
