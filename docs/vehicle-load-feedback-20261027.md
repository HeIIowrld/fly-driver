# 바퀴 하중 검증과 게임 온도·하중 피드백

2026-10-27, Asia/Seoul. **사용자 지시에 따라 게임 계측 타이어 온도와 바퀴별 하중을
주 actor의 감각 입력으로 사용한다.** 감각만으로 추정해야 한다는 이전 제한은 수정했다.

## 수집·센서 계약

읽기 전용 AC 앱에 SDK `CurrentTyresCoreTemp`와 `Load`를 추가했다. 원본
[Gym 센서 코드](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/structures.py)에서도
두 필드를 사용하는 것을 확인했다. core 온도는 표면 내/중/외 온도와 별도 채널이다.
네 바퀴의 raw 값·누락 상태를 보존하며 실제 게임에서 단위·순서·clock은 아직 미검증이다.

`BodySample`에 `tyre_core_temperature`4개(K), `wheel_normal_load`4개(N)를 추가했고
명시적 affine calibration으로 Celsius→Kelvin 같은 offset 변환이 가능해졌다.
압축 하중0은 유효하며 음의 하중·음의 절대온도는 유효 표본으로 받지 않는다.
없는 온도를 offset만 더해 정상 온도로 만들지 않는다. pose·진행률 등은 whitelist에 없다.

이는 **수집기와 계약의 구현**이다. 기존2,626뉴런 checkpoint에 새 감각 포트가 연결되거나
타이어 관리가 학습된 것은 아니다. 기관별 포트 검토·인코더·학습 데이터와 출력 경로 확장이 필요하다.

## 실제 MuJoCo 하중·운동량 실행

고정된 flat synthetic 환경에서 정지·가속·제동·좌/우 선회5개 조작을 실행했다. neural policy나
교사는 없고 공통 구동/조향/제동 어댑터의 정해진 명령만 사용한다. map/reward/actor 입력은
생성하지 않는 물리 진단이다. 원래 `env.step`의 코스 완료 판정도 이 진단에는 적용하지 않는다.

접촉 좌표와 geom 순서를 처리해 ground-on-wheel 힘·접촉 모멘트를 world frame으로 옮겼다.
[공식 contact 정의](https://mujoco.readthedocs.io/en/stable/computation/index.html#friction-cones)와
[mj_contactForce API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-contactforce)를
따랐다. 전체 차량 COM의 선/각운동량 변화와1ms solver force/moment impulse를 대조했다.

| 조건 | FL / FR / RL / RR 수직력 평균(N) | 창 내 해당 가속도(m/s²) |
|---|---|---|
| 정지 | 601.14 / 599.61 / 599.61 / 601.14 | longitudinal≈0 |
| 가속 | 475.38 / 477.34 / 722.73 / 725.95 | +3.508 |
| 제동 | 832.43 / 834.90 / 369.06 / 367.15 | −6.469 |
| 좌선회 | 587.60 / 614.22 / 586.84 / 612.91 | lateral +0.967 |
| 우선회 | 604.65 / 595.76 / 605.12 / 595.86 | lateral −0.966 |

질량244.8kg, 중량2,401.488N이다. 사전 선언한 중량·운동량·하중 부호·좌우 mirror·접촉
범위의 **25검사를 통과**했다. 최대 선운동량 RMS residual/중량은 약1.70e−6이다.
이 결과는 해당 수치 물리계의 기계적 sanity check다. 실제 차량의 하중 크기·좌우 대칭성·
타이어 특성이나 AC와의 동등성을 검증한 것이 아니다. 좌우 결과 차이도 그대로 남겼다.

처음 unit test는100Hz 끝점 힘만 평균해 정지 시 수평력이 약53N인 것처럼 나와 실패했다.
1ms 서브스텝 impulse와 실제 COM 속도를 대조한 결과, 정지 창의 speed RMS는0.000110m/s였다.
제어 틱 사이 contact force를 시간 평균하도록 진단을 고쳤고 끝점 원본도 trace에 보존했다.
물리 파라미터나25개 통과 임계값을 이 실패에 맞춰 바꾸지 않았다. 최초 실패 로그도 수집했다.

최종 WSL 관련 검사 **43통과·Windows 전용2건 건너뜀**, 전체 정적 검사 통과다.
Windows의 AC/기관 선택 검사는35통과·알려진 UDP 실패1건 제외다. 새 전체 pytest 결과로
해석하지 않는다. 두 source snapshot·원본 trace·실패/성공 로그 **402파일을 수집해 SHA256을
대조**했고 현재 Python126개가 최종 시험 snapshot과 같다. 위치는
`runs/vehicle-load-audit-20261027`이다.

## 이후 우선순위

사용자가 요청한 [기존 AC 모델 조사와 파리 구조 이전](ac-autonomy-to-fly-20261027.md)을 반영해
공개 교사와 데이터를 재사용하는 증류를 우선 검토한다. 이 접촉 진단은 그 보조 검증이다.
새 AC 타이어 엔진을 완전히 재구축하는 작업을 기존 교사 활용의 필수 선행 조건으로 만들지 않는다.
게임 계측값은 선언한 감각 입력으로 받아 기관별 연결과 실제 학생 운전에 기여하는지 검증한다.
