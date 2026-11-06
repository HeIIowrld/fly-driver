# 새 세션 인계 — 출력 계약·결정 전달·실제 학생 폐루프

**후속 상태 — 2026-11-05:** [출발 원인 분리·저속 학습·실제 재비교](student-launch-learning-20261105.md)까지 진행했다. 아래는 그 이전 실행/인계 기록이다.

**2026-11-05 실행 후속:** 이 인계의 출력 계약·전달·정지 구현과 실제 학생 연결 시험을 진행했다. [실행·실패·부모/후보 비교](student-closed-loop-20261105.md)를 최신 상태로 읽는다. 아래는 실행 전 인계 원문이다.


작성: 2026-11-05, Asia/Seoul. **이 문서는 다음 세션의 실행 지시와 현재 증거를 정리한 인계다. 출력 계약/결정 전달 구현이나 학생 주행을 이번 문서 작성에서 수행한 것은 아니다.**

## 다음 세션에서 수행할 사용자 요청

> 업데이트한 로드맵대로 출력 계약과 결정 전달부터 구현하고, 실제 학생 폐루프 주행까지 검증을 계속한다.

사용자는 계획 설명만 요청한 것이 아니라 위 구현과 실제 검증을 다음 세션에서 이어가도록 요청했다. 기존 권한 범위에서 코드 검토·구현·테스트·검증된 background/CSP API를 통한 유한 학생 주행까지 진행한다. 일상적인 구현 선택이나 이미 요청한 주행 자체에 대해 다시 허가를 묻지 않는다. 실제 환경/계약이 불명확하면 원본 증거를 먼저 확인하고, 선행 조건 실패 시 이를 해결한다. 계약·신선도·소유권 검증 없이 주행으로 건너뛰지 않는다. 새 채팅 생성이나 자동 실행을 예약한 상태는 아니다.

## 반드시 유지할 대전제

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.**

비행의 날개 조종·균형·시각 안정화가 기준이다. 일반 운전 모델·보행 회로를 최종 목표로 바꾸지 않는다. 실측 연결·생리 근거·기능 근사·차량 대응·학습 어댑터를 구별한다. 차체 기울기를 몸 기울기로 되먹임한다. 가속·감속·하중 이동·접지/타이어·셋업/차종 적응, 검증된 상대풍·변속, 빠른 반복 유효랩·레이싱 라인/랩타임까지 목표에 포함한다. 부품 반응이나 짧은 주행으로 최종 목표를 완료 처리하지 않는다.

Actor에는 영상과 허용된 몸 감각만 준다. 지도·world pose·절대 heading·진행률·미래 곡률·정답 라인·교사 명령은 teacher/critic/보상/guard/평가에만 허용한다. guard가 읽는 진실값을 inference callback에 흘리지 않는다. 센서 결측을 oracle로 보충하지 않는다.

## 시작할 위치와 읽기 순서

모든 shell 명령은 아래 폴더를 **workdir로 명시**한다. 상위 폴더에 코드를 만들지 않는다.

```text
\\research-nas\homes\research-user\02_Personal\01_학교\20.활동\개인프로젝트\flybrain
```

기존 폴더와 미커밋 변경을 유지한다. reset/clean·브랜치 전환·새 worktree로 우회하지 않고 다른 변경을 임의로 묶지 않는다. 이 작업은 FlyBrain이며 AIDE의 원격 202 규칙 대상이 아니다.

읽는 순서: `AGENTS.md` → `README.md` → `docs/PROJECT_CHARTER.md` → `docs/CONTEXT.md` → `docs/research-plan.md` → 이 문서 → `docs/implementation-status.md`와 아래 원본 보고서/receipt. CONTEXT의 오래된 누적 보고에서 ‘현재/다음/미학습’은 해당 시점의 상태다. 현재 로드맵은 96줄로 재정리했고, 종전 2,314줄은 `docs/research-plan-history-before-review-20261105.md`에 원문 보존했다.

## 현재 모델과 증거

| 역할 | 원본 checkpoint | SHA256 |
|---|---|---|
| 실제 moving shadow를 확인한 부모 | `runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt` | `723286d6f79401e266d24a7c384680f95cf3edbabf7807139bfb39177dea6812` |
| 최신 두 기준 보호 연구 후보 | `runs/dual-reference-learning-20261105/dual_guard-fit-01/ports.pt` | `8de4aad03027e36dc8dcd2655745f6a3769af8ee254f9095e2cdd6ef9d0c6099` |
| 재사용 원래 보호 대조군 | `runs/direct-scene-gain-learning-20261105/frozen_direct-fit-01/ports.pt` | `a47338d90bd994f86a934df0234478552adc41d837b002c811a0579b614272e8` |

- `.723` 부모: 실제 GPU camera→CPU flight owner의 moving shadow-04는 104영상 묶음/455출력/5.666549초, body 최대21.451ms/view137.366ms/inference17.399ms/RPC5.820ms, 원본 픽셀 재구성 최대차1.79e-7/전체 출력 replay 차0이다. 교사 저속 관측이며 학생 제어0·랩0이다. 교사 속도 최대19.843km/h는 fit 지지 범위 13–74m/s보다 낮다.
- 최신 dual_guard:실제 TRAIN 32,597결정/395업데이트, 전체 48,866결정 연속 평가, Adam proposal/moment 독립 감사 차0, strict CPU loader 128결정 예측/근육 차0이다. 실제 moving shadow/학생 제어는 미검증이다.
- 최신 진입 전 조향 RMSE 감소율은 대조군 대비 +2.362/+5.521/+3.209%, `.723` 부모 대비 +1.284/+1.170/+1.144%다. 반면 전체 조향은 대조군 대비 0.972/1.274/1.238%, 부모 대비 0.643/1.381/1.446% 악화했다. 순서는 냉간/고온/소비된 Monza 개발이다. 후보 미승격이다.
- 영상 교환 방향27/30·목표 margin .002 통과17/30; 새 합성 방향18/18·목표0/18. 원본 RGB 한 TRAIN 사례 margin .002252343, visual 증분 제거 .001863603이다. 방향 변화와 실제 코너 인식/실제 주행 개선은 구분한다.
- yaw/speed 국소 감쇠 방향45/45, 고정 입력12조건 마지막2초 출력 변동0이다. 실제 차량 안정성·온도 적응 증명이 아니다. soft 보호의 부모 기준 위반 window8/33이 남는다.
- fixture 앞단 시각 gradient 수정과 두 reference 보호는 **실제 학습 완료**다. 직접 장면 보정42계수는 효과 없어 최신 후보에서0 고정이다. 동일 자료의 단순 추가 epoch를 다음 작업으로 되돌리지 않는다.
- 시각 기여가 작았던 과거 평가는 교사 몸 반응으로 현재 조작을 맞힐 수 있는 오프라인 과제였다. 영상 불필요/모든 코너 암기를 입증하지 않는다. 실제 donor의 도로 위치·속도 혼입, 소비된 Monza 개발, 합성 fixture 한계를 유지한다.

보고서: [최신 학습](dual-reference-learning-20261105.md), [실제 moving shadow](current-flight-moving-20261105.md), [직접 장면 보정 실패](direct-scene-gain-learning-20261105.md), [근육 이전/공간 진단](visual-premuscle-spatial-20261103.md), [영상 교환 혼입](precorner-road-pose-confound-20261104.md).
원본: `runs/dual-reference-learning-20261105/{summary-01.json,source-receipt-final-01.json,control-reuse-01.json}`. 원본 checkpoint/source/hash는 파일에서 재검증한다. 마지막 기록 source revision은 `9b8b45fc7d1ce3557d25b1af6d83bbe8f1c23fda`이며 현재 dirty tree 전체를 이 커밋과 같다고 주장하지 않는다.

## 구현 순서와 코드 진입점

### 1. canonical 출력3의 계약

현재 모델 출력은 **steering/throttle/brake의 canonical 3값**이다. `src/flybrain_drive/flight_actuator.py::canonical_controls`는 좌회전 intent와 signed drive **2값**을 받아 변환하는 구형 codec이므로 현재 3값에 다시 적용하지 않는다. `.6` 조향 배율을 구형 pilot에서 근거 없이 복사하지 않는다.

원본 학습 라벨 생성·교사 계약·축 교정·SDK 적용 피드백을 역추적하여 값 범위, 좌/우 부호, 물리/정규화 단위, 배율, pedal/coast 의미를 문서와 코드로 고정한다. raw 모델 출력과 clipping/scaling/override 후 요청을 모두 기록한다. current teacher/data 계약이 실제로 요구하는 변환만 적용한다. 좌/우·가속/타력/제동·비정상값·중립/정지/해제의 의미 있는 검사를 작성한다.

진입점은 `current_flight_runtime.py::load_current_flight_runtime`, `dual_reference_candidate.py::load_dual_reference_runtime`, 고정6-MN head 및 현재 teacher/dataset 계약이다. MN6값은 modeled activity이며 실측 근력 아니다. 최신 loader는 gain0와 원래 누적 bound를 검증한다. source hash로 봉인된 실행 소스는 가능한 새 모듈/명시적 버전으로 확장한다. 과거 source 검사를 통과시키려고 원본 manifest/hash를 바꾸지 않는다.

### 2. 원본 시각에 묶인 최신 결정 전달

`current_flight_owner.py::CurrentFlightOwnerProcess`는 현재 status/context/publish와 replay 경로를 제공한다. `current_flight_gc_owner.py::CurrentFlightGCOwnerProcess`는 유한 GC deferral을 추가한다. 실제 제어를 위한 최신 결정 반환은 아직 없다. `native_flight_body_ticker.py::NativeFlightBodyTicker`의 단일 재귀 소유 thread가 생성한 유효 출력을 전달한다. 제어 루프에서 같은 입력으로 runtime을 다시 진행시키지 않는다.

출력 envelope에는 episode/run·generation·checkpoint identity·순차 decision/publication/capture ID·원본 body packet/first receipt·각 view render bound/receipt·inference 완료시각·canonical3을 넣고 소비 시 다시 신선도를 검사한다. 출력 완료시점의 검사만으로 액추에이터 적용 시점의 신선도를 대체하지 않는다. 재수신 때문에 source timestamp를 새로 찍지 않는다. 단순 로그 tail이나 request가 왔다는 이유로 오래된 출력을 신선하게 취급하지 않는다.

원래 제한 body≤50ms/view≤150ms/source-render≤25ms/RPC≤50ms, 10ms 몸 ticker와 near-expiry reserve25ms를 유지한다. stale·future·세대/순서 불일치·unknown/EOF·소유권 상실은 거부하고 자동 재시도하지 않는다. 오류 상태는 명시적 새 run 전까지 유지한다. 재귀 수치 상태는 에피소드 안에서 연속 유지하며 detach를 reset으로 바꾸지 않는다. 지연/손실/중복/정지 경로와 기존 원본 재생 parity를 검증한다.

### 3. 현재 통합 actor의 유한 학생 폐루프

`scripts/pilot_ac_native_student.py`는 **MultiViewCircuitController**를 로드한다. 그대로 실행해서 현재 flight actor 주행이라고 부르지 않는다. `src/flybrain_drive/ac_student_pilot.py::{StudentTrialReading,run_student_trial}`와 VJoySession/guard·정지/해제 공통 경계를 검토해 현재 runtime용 pilot에 연결한다. 구형 고정20km/h 과제/속도/배율 계약을 현재 모델에 묵시 적용하지 않는다.

참고 entrypoint는 `scripts/{run_owned_current_flight_shadow.py,shadow_current_flight_teacher_motion.py,validate_current_flight_owner.py,validate_current_flight_gc_owner.py,audit_current_flight_motion_owner.py,audit_current_flight_motion_pixels.py,validate_dual_reference_runtime.py}`다. 부모와 최신 후보를 각각 엄격 loader/hash로 복원한다. 현재 owner가 부모 loader를 사용한다는 사실을 확인하고 후보 loader를 명시적으로 연결해야 한다. 최신 후보가 부모 shadow 결과를 상속했다고 간주하지 않는다.

순서: 오프라인 계약/전달 검사 → 새 wrapper read-only/moving 검증 → 세션/제어권/실제 thermal/gear 상태 확인 → 준비 구간/중립 → 시간·속도·거리·이탈 등의 중단 한도를 사전 고정한 짧은 학생 trial → 원본 명령/실제 피드백·정지·해제·프로세스 종료 감사 → 가능한 같은 초기 조건에서 부모/후보·타이어 초기 온도별 반복 비교. 구형 데이터의 명목80℃가 아니라 native 실측 네 바퀴 온도를 기록한다. 초기 준비는 기존 0.5초/유효 카메라3묶음 기준과 startup 과도를 확인한다. 과거 초기 출력은 큰 조향/가속 과도를 보였으므로 준비 전 제어0을 검증한다.

모든 실제 주행은 `scripts/control_ac_session.py`의 검증된 CSP lifecycle와 background 감각/제어 API만 쓴다. AC desktop 창 활성화·클릭·키 입력·UI reset은 금지다. API 실패 때 desktop으로 우회하지 않는다. teacher가 운전한 준비/관측 구간과 student 구간을 분리한다. cleanup 제동을 학습한 제동으로 세지 않는다.

성공 여부와 무관하게 requested/applied controls, 관측/원본 clocks, 학생 결정, 차체 회전·가속도·하중/타이어, source/hash/seed, 종료 이유와 neutral/brake/release/owned process terminal을 보존한다. 실제 학생이 명령했고 차량이 반응했는지 검증한다. 준비/주행/정지에서 실패하면 실패 run을 보존해 원인을 수정하고 새 고유 run으로 재시험한다. freshness나 종료 기준을 완화해 통과시키지 않는다.

## 환경과 실행 상태 — 반드시 재확인

- Windows PowerShell, Python `C:/Users/research-user/AppData/Local/Programs/Python/Python312/python.exe`.
- CPU PYTHONPATH: `C:/Users/research-user/AppData/Local/PythonEnvs/flybrain-resolution-20261017/Lib/site-packages;src;.` (torch2.9.1+cpu).
- GPU PYTHONPATH: `C:/Users/research-user/AppData/Local/flybrain-bio-20261029/flyvis-env/Lib/site-packages;src;.` (torch2.9.1 ROCm/AMD9070XT). 기존 봉인된 launcher처럼 CPU/GPU 프로세스 환경을 분리하고 OPENBLAS_NUM_THREADS=1·torch threads1을 사용한다.
- Git can report dubious NAS ownership. A per-command `git -c safe.directory='//research-nas/homes/research-user/02_Personal/01_학교/20.활동/개인프로젝트/flybrain' ...` worked; do not change global config unnecessarily.
- Latest installed Lua SHA recorded: `0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e`. Recheck required source contract before live work; do not bypass mismatch.
- Last external snapshot: `runs/dual-reference-learning-20261105/external-state-01.json`, paused=true/epoch9, generation `1117858ULL:0.0016212`, Monza/BMW Z4 GT3. It is a historical observation, not current live state. This handoff did not query AC or process liveness.
- Latest fit/evaluation/audits finished. Old failed shadow teardown had remaining handles: PID50656/32480/45084/21676/36336 and original tool handle15748. Old recorded wait258 or code259/1 did not prove process termination. These are not ongoing training. Check identity/creation time/owned run before action; do not kill a reused PID or restart from timeout alone. Successful shadow-04 owned processes exited0 and teacher/camera/archives closed separately.
- Runs go under `runs/`, external data under `assets/external/`. Preserve old NPZ resolver/reference table; do not recreate all deleted duplicate paths. Do not delete active runs or old failures.

## 다음 세션의 완료 보고에 필요한 내용

실제 구현·의미 있는 검사·증거 경로와 현재 학생의 AC 제어 여부를 보고한다. 부모/dual_guard, 오프라인 parity/실제 주행, 초기 온도 조건/실제 적응을 구분한다. 짧은 trial로 유효랩·한계 주행을 주장하지 않는다. 실제 진전과 실패를 README/CONTEXT/research-plan/implementation-status에 반영하고 원본 run과 후보의 득실을 유지한다. 외부 환경 때문에 실주행을 검증하지 못하면 구체적 실패와 남은 일을 명시한다. 설명/보고서는 Markdown으로 작성하며 HTML·사이트는 만들지 않는다.

## 새 채팅에 붙여 넣을 시작 문장

```text
flybrain 기존 폴더에서 AGENTS.md와 지정 읽기 순서를 따른 다음 docs/handoff-student-closed-loop-20261105.md를 읽고 이어서 실행해줘. 업데이트한 로드맵 1~3단계인 현재 통합 flight actor의 canonical3 출력 계약과 source-bound 최신 결정 전달부터 구현·검증하고, 검증된 CSP/background 인터페이스로 실제 유한 학생 폐루프 주행과 부모/후보·타이어 초기 온도별 비교까지 진행해줘. 과거 미커밋 변경·원본 증거를 유지하고 AC desktop 입력은 쓰지 마. 계획 제시에서 멈추지 말고 실제 결과와 실패까지 기록해줘.
```
