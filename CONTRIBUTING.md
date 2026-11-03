# 共同 연구 / Contributing

팀 작업 기준은 [프로젝트 대전제](docs/PROJECT_CHARTER.md), [로드맵](docs/ROADMAP.md), [협업 방식](docs/COLLABORATION.md)이다. 개인 저장소 `HeIIowrld/fly-driver`의 `main`을 현재 공유 기준으로 사용한다. 팀 원본 저장소로 반영하려면 별도 PR을 사용한다.

## 개발 환경과 코드

Python 3.12 가상환경에서 다음을 실행한다. 팀원은 각자 별도 clone을 사용한다.

```sh
python -m venv .venv
# 가상환경 활성화 후
python -m pip install -e ".[learning,teacher,dev]" pillow PyYAML
python -m pytest tests/test_graph.py tests/test_racing_goal.py tests/test_camera_body_alignment.py tests/test_unified_flight_sensorimotor.py -q
```

이 검사는 합성 fixture와 오프라인 계약 검사다. 실제 게임 주행이나 생리적 타당성·완주를 검증하지 않는다. 전체 테스트는 외부 자료/운영체제 의존성이 있으므로 위 검사만으로 전체 통과를 주장하지 않는다. `requirements-tested.txt`는 과거 Windows 검증 환경의 기록이며 모든 현재 경로의 완전한 lockfile은 아니다.

- 모델: `src/flybrain_drive/`, 학습/수집/감사 도구: `scripts/`, 설정: `configs/`, 검증: `tests/`.
- 현재 적격 SAC 시연 학습 경로: `scripts/prepare_synchronized_sac_student.py` → `scripts/train_qualified_sac_unified.py` → `scripts/audit_qualified_sac_student_fit.py`.
- 입력 경계/준비물/실제 실행 기록은 [정합 교사 학습 보고서](docs/teacher-synchronized-learning-20261102.md)를 확인한다. CLI 인자는 각 스크립트 소스의 argparse와 run manifest를 따른다.
- FlyVis 저자 모델·커넥톰·교사 가중치·적격 원본 에피소드·분할/계획 manifest는 이 Git 저장소에 포함하지 않는다. 일반 pip 설치만으로 기존 학습을 재현할 수 없다. 필요한 자료는 `configs/assets.json`과 원본 run manifest의 경로/해시/사용권을 확인해 접근 가능한 공용 저장소에서 준비한다. 누락된 데이터나 자격 증거를 만들어 대체하지 않는다.

## 변경과 실행 결과

작업별 이슈/브랜치를 만들고 변경 목적·재현 명령·검사 결과·미검증 범위를 PR에 적는다. 코드 revision뿐 아니라 수정된 소스 해시도 run manifest에 기록한다. 원본 데이터는 `assets/external/`, 실행 결과는 고유한 `runs/`에 보존한다. Git에는 코드·설정·작은 manifest·검토 보고서를 넣고 대용량 데이터/가중치·게임 파일·비밀키를 넣지 않는다. 기존 NPZ 중복 제거 참조표/공통 로더를 사용한다.

CI는 네트워크 권한을 읽기로 제한하고 합성/오프라인 검사를 실행한다. 실제 AC·vJoy·학습 작업은 자동 시작하지 않는다. AC를 공유할 때는 한 명만 제어권을 보유하고 CSP lifecycle/background 인터페이스만 사용한다.
