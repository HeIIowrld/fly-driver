# 게임 화면을 건드리지 않고 준비한 AC 교사 데이터 경로

갱신: 2026-10-27, Asia/Seoul. 실제 AC 비전·몸 감각 파리 정책의 오프라인 학습을 준비한다.
이 작업에서는 창 활성화·게임 재시작·액추에이터 전송을 하지 않았다.

## 공개 파일 목록 확인

[AssettoCorsaGym dataset](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym)의 고정 revision
`ad1df12c49cce10f159ba31967744750bb9edd8b` 파일 목록17,131개를 API로 읽었다.
데이터 전체를 다운로드하지 않았다. 목록·API URL·revision·해시는
`assets/external/acgym-visual-inventory-20261027/`에 보존했다.

| 확장자 | 파일 수 |
|---|---:|
| .pkl | 15,882 |
| .pth | 570 |
| .csv | 242 |
| .ld / .ldx | 202 / 204 |

별도 PNG/JPG/MP4 같은 영상 파일이나 image/camera/rgb 이름의 경로는 없었다.
이는 전체 숫자 파일 내부의 영상 부재를 증명하지 않는다. 기존234상태 표본을 제한 판독기로
검사한 결과에도 영상 후보는 없었다. 따라서 공개 자료만으로 동기 비전 데이터가 준비됐다고
표시하지 않고, 적합한 내부 영상 자료가 확인되지 않으면 실제 교사 주행 중 추가 수집한다.
정차 메뉴 영상에 공개 주행 기록의 조작을 붙이지 않는다.

## 실제 SAC 교사 입력 수집 준비

고정한 [원본 structures.Car.update](https://github.com/dasGringuen/assetto_corsa_gym/blob/ef0fa502bdd257fc2affad9361c02b3dbdfe1632/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/structures.py)의
SDK 순서를 대조해 수집 앱에 `privileged_teacher` namespace를 추가했다.

- world/local 벡터의 x,y,z를 원본의 `[2,0,1]` 순서로 옮긴다.
- 가속 AccG의0/2축, 네 SlipAngle, Gear·RPM·LastFF·정규화 진행률 등을 raw로 남긴다.
- 값이 없거나 잘못되면 None으로 표시한다.0이나 모델 예측값으로 채우지 않는다.
- payload에 `actor_input=false`를 붙이며, 학생의 BodySample이나 시각 입력 필드를 늘리지 않는다.
  기존 actor calibration은 world_position을 감각 입력 key로 선택할 수 없다.

원본·MIT license·Git blob `2e09ba0a4a033bc2a182549b75cb98cc454d142e`와 SHA-256은
`assets/external/ac-teacher-sdk-20261027/`에 보존했다. 외부 모듈은 실행하지 않았다.
추가 수집 코드는 저장소에 준비했지만 **현재 실행 중인 앱에는 배치하지 않았다.** 사용자 작업을
방해하는 전면 조작·재시작을 피하고, 필요한 설정을 모아 이후 재로딩 시 실제 값을 검증한다.

## 아직 필요한 연결

host에서 reference gap·지도 ray·curvature를 원본 규칙으로 계산하고, 새로운 SDK 표본과
native 시각/pose를 검증해야 한다. 그 뒤에야125관측 SAC 교사의 실제 폐루프 주행을 재현할 수
있다. 교사의 위치·미래 경로 입력은 교사에만 사용하며 파리 학생 입력으로 넘기지 않는다.

현재 상태는 `runs/ac-offline-data-readiness-20261027-t42/`에 기록했다.
SDK 벡터 순서·누락·학생 key 경계 및 기존 publisher 관련6검사와 Ruff를 통과했다.
실제 AC 비전 데이터의 봉인·새 필드 live 검증·native 학생 주행 학습·서킷 완주는 아직 남아 있다.
