# SDK/native 관측 조립과 조작 없는 SAC 추론

갱신: 2026-10-27, Asia/Seoul. 목표는 실제 AC의 비전·몸 감각 파리 정책을 위한 교사 기록이다.
이번 단계에서 학생 주행이나 SAC 조작을 실행하지 않았다.

## 구현한 입력 경로

`ac_teacher_observation.py`는 teacher-only SDK namespace와 native physics의 heading·
numberOfTyresOut을 합쳐, 검증한 geometry와 곡률을 통해125개 SAC 관측을 만든다.
진행 거리에는 SDK track length와 NormalizedSplinePosition을 사용한다. 새 SDK 수집기에
`ac.getTrackLength(0)`도 추가했다. 교사/평가용 값은 ActorObservation으로 만들지 않는다.

- 처음3관측은 실제 history를 쌓으며 추론을 하지 않는다.0으로 과거를 채우지 않는다.
- producer session·SDK/native sequence·host/source 시각·관측 간격·host receipt skew를
  검사한다. session 교체나 역행 때 history를 임의로 지우고 계속하지 않는다.
- 필요한 값이 없으면 TeacherInputUnavailable을 반환하고 관측을 만들지 않는다.
- 반환 state와 관측은 읽기 전용이다. host receipt skew는 실제 노출 동기 오차가 아니다.

## 공개 기록으로 조립 검사

`runs/ac-teacher-packet-assembly-20261027-t44b/`는 공개234상태를 **합성 SDK/native 컨테이너**로
감싸 이 경로를 실행했다. 실제 SDK 패킷을 새로 수집한 결과가 아니다. 알려진 과거3개를 제외한
231개 관측의 기존 재현값 대비 최대 차이는7.15×10⁻¹⁶이었다. 이 검사는 벡터 순서·host 조립을
뒷받침하며 live 센서 단위나 동기를 증명하지 않는다.

`scripts/shadow_ac_teacher.py`는 실제 파일/공유메모리의 값을 읽어 추론만 기록한다.
vJoy·키보드·reset·학생 제어 인터페이스를 import하지 않으며 조작을 보내지 않는다.
현재 `runs/ac-teacher-shadow-readiness-20261027-t44/` 결과는 `not_ready`, 추론0건이다.
실행 중인 앱에 privileged_teacher namespace가 아직 없어 정확히 그 이유로 중단했다.

## 배치 상태

SDK 수집기 파일은 atomic replace로 교체했다. 배치 SHA-256은
`7fcd217f85c15b86bf632c825090f31463e6294f8dfdcf41aa62bb5418188330`이며,
`runs/ac-teacher-collector-deployment-20261027-t44c/`에 이전 파일과 기록을 남겼다.
배치 후 SDK steps102056→102061·native packet705188→705190으로 기존 엔진은 계속 동작했다.
그러나 메모리에 이미 올라간 코드는 이전 버전이어서 새 namespace는 없었다.
창 활성화·세션 재시작·배치 과정의 조작 전송은0건이다.

사용자의 전면 작업을 방해하지 않기 위해, 필요한 세션 재시작1회가 가능한지 질문했다.
응답 전에는 창을 앞으로 가져오지 않는다. 새 namespace의 live 검증이 끝나기 전에는
교사의 실제 주행이나 학생 학습 데이터가 준비됐다고 표시하지 않는다.

## 재시작에 함께 반영할 조향 스케일 후보

별도 `runs/ac-sdk-steering-calibration-20261027-t44d/`는 전면화 없이 정차32표본을 수집했다.
SDK wire와 native prefix의 해시/내용을 재검증했다. 요청±0.1에 SDK Steer 중앙값은±44.999996,
즉 요청1당 약450이었다. 공개234상태의 current_action_abs_0→SDK Steer는 약270의 gain으로,
gain270 대비 최대 잔차는0.04513이었다. native raw steerAngle과 SDK Steer는 같은 숫자가
아니므로 서로의 단위인 것처럼 사용하지 않는다.

현재 LOCK900 설정에서 교사 요청을 바로 실행하면 공개 기록보다 조향이 약1.67배 크게 적용될
가능성이 있다. 같은 driver/축에서 LOCK540 후보와 해당 profile hash를 준비했다.
`runs/ac-teacher-collector-deployment-20261027-t44c/proposed-controls-540.ini`와
`proposed-axis-probe.json`이며 **아직 적용하지 않았다.** 재로딩 뒤 요청±0.1→SDK 약±27인지
실제로 검증하고 나서 교사 요청 변환에 사용한다. 이동 중 응답이나 도로 바퀴 rad 단위는 미검증이다.

SDK/조립/기존 file 검사8개와 Ruff를 통과했다. 실제 교사 주행, 동기 다중 시점 데이터,
파리 학생의 native 가감속 학습과 실제 서킷 완주는 아직 남아 있다.
