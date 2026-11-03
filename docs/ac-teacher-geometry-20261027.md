# AC 교사의 경로 관측 재계산과 추론 연결

갱신: 2026-10-27, Asia/Seoul. 실제 AC의 영상·몸 감각으로 파리 학생을 학습하기 위해
기존 SAC 교사를 연결한다. 이번 작업은 저장 기록으로 검증한 host 계산이며 게임 창을
활성화하거나 조작하지 않았다. 파리 학생의 입력에 위치·지도·미래 경로를 추가하지 않았다.

## 구현

`ac_teacher_geometry.py`는 고정한 AC Gym legacy revision
`a90db0a8b164164e11412857b396bf6874462976`의 관례를 NumPy 계산으로 구현한다.
참조점 pivot과 양옆 segment 선택 우선순위·부호, 경계의10개당1개 downsample 및 마지막 값,
11개 ray의 순서·방향·기본 거리200을 유지한다. 원본의 legacy wrap/sign 동작을 임의로 고치지 않는다.
새 구현은 전체 downsample 경계를 사용하며 upstream의 spatial hash를 실행하지 않는다.
따라서 아래 기록과의 일치는 확인했지만 모든 가능한 pose에서의 동일성을 주장하지 않는다.

원본 gap/track 코드·Monza 경계·참조 계산·MIT license·해시는
`assets/external/ac-teacher-geometry-20261027/`에 보존했다. 다운로드한 gap_cpu/track/monza.csv는
고정 Git blob까지 대조했다. 외부 모듈은 import하지 않는다. geometry 생성 시 입력 CSV 해시를
확인하며 계산 결과는 teacher/evaluation에만 사용한다.

## 실제 공개 기록 대조

`runs/ac-teacher-geometry-timing-20261027-t43c/`에서 기존 제한 판독기로 읽는 공개234상태의
위치·방향으로 gap과 ray를 다시 계산했다. 기존 곡률 계산과125관측 encoder, 확보한 SAC
checkpoint까지 연결하고 이미 회수·해시 검증한 teacher replay와 비교했다.

| 검사 | 최대 절댓값 차이 |
|---|---:|
| gap,234상태 | 7.11×10⁻¹⁵ |
| 11개 ray,234상태 | 1.16×10⁻¹³ |
| 재구성한125관측,231상태 | 7.15×10⁻¹⁶ |
| SAC delta 요청,231상태 | 4.12×10⁻⁶ |

처음3상태는 알려진 과거가 부족해 예전과 같이 추론에서 제외했다. teacher action의 허용 오차는
기존 설정1×10⁻⁵다. 숫자가 정확히 같은 궤적이나 실제 운전 성공을 보장하지 않는다.
재계산한 privileged 관측·delta action은 별도 NPZ로 저장했다. 실제 적용 조작으로 표시하지 않는다.

## 계산 비용 측정 범위

CPU 한 스레드에서 각 기록을 하나씩 처리한 **pose→gap/ray→곡률/125관측→SAC 추론**은
중앙값0.295ms·95백분위0.524ms·99백분위0.623ms·최대0.708ms였다.
한 개씩 처리한 출력과 batch 출력의 최대 차이는0이었다.
현재 host의 짧은 기록 재생 측정이며 게임 물리·렌더·캡처·전송·리셋·학생 신경 회로 비용을
포함하지 않는다. 전체25Hz나 hard real-time 동작을 검증한 수치가 아니다.

관련 geometry와 기존 SAC 검사6개 및 Ruff를 통과했다. Windows 환경에는 기존 teacher
선택 의존성인 SciPy1.18.1을 추가했다. 실행 중인 게임이나 앱 코드는 변경하지 않았다.

## 남은 실제 연결

실행 중인 앱의 추가 SDK 교사 관측 live 검증과 일관된 시간/세션 연결이 필요하다.
현재 준비한 새 namespace는 아직 배치하지 않았다. 그 뒤에 shadow 추론·유한 실제 교사 주행,
호환되는 다중 시점/몸 감각/조작 기록, 학생 오프라인 학습과 단독 실제 AC 평가를 이어간다.
SAC의 정확한 학습 실행 manifest 미확보 제한도 유지한다. 서킷 완주는 아직 달성하지 않았다.
