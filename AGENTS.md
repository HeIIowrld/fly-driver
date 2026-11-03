# 사용자 확정 대전제 — 컨텍스트가 바뀌어도 유지 (2026-10-29)

이 프로젝트는 **초파리의 시각 인식·감각·몸 제어 구조를 자동차라는 가상 몸에
연결하고, 그 구조로 조향·가속·감속·빠른 레이싱 라인과 랩타임을 학습하여
Assetto Corsa 서킷을 자율 주행하게 하는 연구**다. 단순히 운전 잘하는 모델을
만드는 것으로 목표를 대체하지 않는다.

- 사용자가 고른 몸 제어의 기준은 **비행: 날개 조종·균형·시각 안정화**다.
  보행 회로나 기존 DNa02/페달 readout을 비행 제어 구현으로 바꾸어 부르지 않는다.
- 자동차가 가상 몸이다. 차체 기울기·회전·관성·하중 이동·접지·타이어 상태를
  몸 감각으로 대응시키고, 감각으로 운동 명령의 결과를 되먹임한다.
  전체 초파리 공력 시뮬레이터를 만드는 것을 불필요한 선행 조건으로 추가하지 않는다.
- 겹눈은 여러 화면을 합친 이름이 아니다. 방향/시간 구조와 운동 시각 처리를
  도입하고, 몸 제어는 비행 유래 하행·국소 운동·감각 되먹임 구조로 검증한다.
  단순 그래프 통과나 완주만으로 이 기능이 구현됐다고 판정하지 않는다.
- 레이싱 라인·랩타임 최적화는 원래 목표에 포함된다. 정답 라인·지도·world pose는
  교사/보상/평가용이며 actor의 경로 판단은 영상과 허용된 몸 감각에서 나온다.
  몸 자세/회전 감각과 절대 world heading은 구분한다.
- 실측 뉴런/연결, 생리학적 근거, 기능 근사, 차량 대응 가설, 학습한 어댑터를
  명확히 구분한다. 일반 CNN/RNN이나 기존 운전 모델은 비교군·교사로 활용할 수
  있지만 생물학적 구조 도입 목표를 대신하지 않는다.
- 추가 운전 증류/튜닝보다 누락된 시각·비행 제어 부품의 반응과 인과적 기여를
  우선 검증한다. 기존 성공/실패와 원자료는 보존한다. 이 대전제의 변경은
  사용자의 명시적 변경 요청으로만 한다.
- **새 작업/컨텍스트 정리 후 읽는 순서:** `README.md` →
  `docs/PROJECT_CHARTER.md` → `docs/CONTEXT.md` → `docs/research-plan.md`.
  구현 여부는 `docs/implementation-status.md`와 원본 실행 증거로 확인한다.
- 설명 자료는 Markdown으로 작성한다. 설명용 HTML·웹앱·사이트를 만들거나
  갱신하거나 자동으로 열지 않는다.

## 문맥을 갱신할 때의 규칙

- 위 대전제와 `docs/PROJECT_CHARTER.md`는 사용자 확정 요구이며 진행 요약에서 생략하지 않는다.
- 컨텍스트 인계 요약에는 다음 문장을 유지한다: **초파리의 겹눈/운동 시각과 비행 유래
  균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를
  몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.**
- 새 상태를 기록할 때 과거 단계의 “미학습/미연결/다음 작업”은 당시 시점을 표시한다.
  현재 상태와 충돌하는 문장을 현재형으로 남기지 않는다.
- `docs/CONTEXT.md`에는 최신 증거 경로, 미검증 항목, 실패/불일치, 다음 작업과 실행 상태를 남긴다.
  실제 진전이 있으면 README·연구 계획·구현 현황의 관련 상태도 함께 맞춘다.
- ‘부품 실행’, ‘생리 기능 검증’, ‘차량 통합’, ‘자율 완주/랩타임 검증’을 서로 대체하지 않는다.
- 문서 인계에는 조향뿐 아니라 가속·감속·하중 이동·타이어/셋업 적응과 레이싱 라인/랩타임
  최적화도 남긴다. 차체 기울기를 몸 기울기로 되먹임한다는 사용자 발상을 유지한다.
- 장기 목표는 `docs/PROJECT_CHARTER.md`, 변하는 상태는 `docs/CONTEXT.md`에 둔다.
  실행 중이라는 과거 문장은 PID/핸들과 원본 결과를 다시 확인한 뒤 갱신한다.

# Repository rules

This is an independent research repository. Do not import MECar-Simulation or rely
on a sibling checkout at runtime. Preserve provenance when using its historical evidence.

- Keep implemented, tested, planned and externally required capabilities distinct.
  Synthetic graphs and courses must never be labelled measured connectomes or real circuits.
- Keep actor sensor inputs separate from teacher/critic/diagnostic truth. Camera loss
  and invalid body sensing must be explicit, rather than silently repaired by an oracle.
- Preserve recurrent state between decisions; reset on episode boundaries and explicit
  run restarts. A gradient detach is not a numerical reset.
- Record configuration, source revision, hashes, seed and termination reasons in runs.
- Put generated runs under runs/, external datasets under assets/external/. Never
  commit credentials, simulator binaries, licensed tracks or large connectome datasets.
- No real vehicle control is implemented. Add hardware integration only with explicit
  unit, sign, actuator and sensor contracts and independent verification.
- User preference (2026-10-28): do not activate, click, type into or otherwise automate
  the AC desktop window for training and episode resets. Use the verified CSP lifecycle
  API (`scripts/control_ac_session.py`) and background sensor/control interfaces.
  Record API failures; never automatically fall back to desktop input.

If remote Slurm work is requested, use a unique owned run directory, start a local
collection watcher with submission, collect failed jobs as well as successful ones,
verify inputs/source/logs/results locally before deleting terminated remote run data,
and retain transfer/deletion receipts. Keep reusable shared runtime/GPU caches outside
the cleanup scope. Never expose MECAR_SSH_* credential values or remove an active run.


## 사용자 승인: 과거 학습 데이터 중복 정리 (2026-10-29)

과거 NPZ 중복 사본 214개는 실제 삭제했다. 고유 원본 67종과 기존 dataset/run manifest·실행 기록은 보존했고 `.episode-links.json`과 공통 로더가 옛 경로를 해석한다. 누락된 사본 경로를 발견해도 자동으로 전부 재복제하지 않는다. 원본과 참조표를 보존하고 `episode_storage.resolve_payload_path`/`load_array` 및 `graph.sha256`을 사용한다. 옛 소스의 직접 파일 읽기는 `scripts/with_materialized_episodes.py`로 일시 복원 후 재정리한다. 완료 근거는 `docs/historical-episode-compaction-20261029.md`다.

## 실제 작업 디렉터리 (사용자 확인 2026-10-30)

이 프로젝트의 shell 명령은 모두 `\\hyunjin\homes\hjpark\02_Personal\01_학교\20.활동\개인프로젝트\flybrain`을 `workdir`로 명시한다. 앱의 기존 chat cwd가 상위 `02_Personal`로 표시되어도 상위 폴더에서 코드를 생성하거나 실행하지 않는다. 새 worktree/checkout으로 바꾸지 않고 이 기존 flybrain 폴더와 변경을 유지한다.
