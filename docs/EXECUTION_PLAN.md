# 파리 운전 프로젝트 실행 계획 — 2026-11-02

> **2026-11-02 교사 카메라/원본 주기 병목 시험 및 기준 셋업 주행:** 타이어 설정은 그대로 유지했다. 12초 정지·무카메라 SDK 원본 수신 509개는 source 간격 중앙 20.88ms·최대 62.20ms·80ms 초과 0이었다 (`D:/CodexResearchRuns/flybrain-teacher-cadence-idle-20261102-02/`). 같은 정지 차량에서 기존 네 방향 PNG 카메라를 켠 6초는 112개, 중앙 59.70ms·최대 119.83ms·초과 9회였다 (`D:/CodexResearchRuns/flybrain-teacher-cadence-camera-20261102-01/`). 공유 텍스처/비동기 BMP 저장 후보는 248개, 중앙 20.95ms·최대 85.05ms·초과 2회였다 (`D:/CodexResearchRuns/flybrain-teacher-cadence-shared-20261102-01/`). 독립 시점의 짧은 시험으로 카메라 경로가 부하의 중요 요인이라는 증거이나 다른 시스템 부하를 완전히 통제한 인과 실험은 아니다. 기존 카메라·교사·공유 텍스처 검사 67개 통과. 완료/실패 결과가 오래전에 기록됐으나 CPU를 사용하던 과거 실행 6개는 결과 해시/명령 확인 후 해당 프로세스만 종료하고 영수증을 보존했다 (`D:/CodexResearchRuns/flybrain-terminal-process-cleanup-20261102-01/`).
>
> 공유 카메라를 붙인 교사 시험에서 50/50결정을 완료하고 최대 19.82km/h, 측정 gap 2.507→1.668m, 트랙 이탈 0을 기록했다 (`D:/CodexResearchRuns/flybrain-teacher-shared-live-20261102-01/`). 다음 장기 시험은 시작 시 GPU 복사 제한에 걸려 교사 조작 전 중단했고 (`...-02/`), 재시험은 교사 요청 61회/피드백 60회, 최대 19.91km/h, gap 1.432→-0.306m, 트랙 이탈 0 후 source 간격 81.48ms로 자동 중단했다 (`...-03/`). 각 원시 이벤트 해시와 수치를 별도 감사했다 (`D:/CodexResearchRuns/flybrain-teacher-shared-audit-20261102-01/`). 두 시험의 복사 실패 추적에서 40ms 복사 예산 중 약 44ms 대기/스케줄링이 관찰됐으나 원인을 GPU·OS 중 하나로 단정할 수 없다. 별도 35ms 제어 후보 역시 카메라 복사 제한으로 출발 전 중단했으며 효과는 미검증 (`D:/CodexResearchRuns/flybrain-teacher-shared35-live-20261102-01/`). 모든 시험은 종료 브레이크와 카메라 종료를 기록했다. 50결정 완료는 랩·코너 검증이 아니며 물리 노출/몸 감각 동기가 입증되지 않아 training_admissible=false, 학생 optimizer 0이다. 다음은 격리 카메라/producer 원본 수집 경로와 실제 노출·몸 시각 증거를 검증해 긴 교사 구간을 얻고 에피소드 분리 학습·평가한다. 이전 단락의 미해결·미주행 서술은 당시 상태다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 목표다.

> **2026-11-02 교사 기준 셋업·제어 재시험:** 현재 BMW Z4 GT3/Monza의 기존 타이어 설정(종류 값 1, 압력 네 바퀴 16)을 변경하지 않고 기준으로 사용했다. 원 학습 셋업과 제조 기본값 여부는 미확인이다. CSP restart/start 후 native/CSP AI 상태가 59/59 모두 false로 일치했지만 과거 불일치의 근본 원인은 미확인이다 (`D:/CodexResearchRuns/flybrain-ai-status-after-restart-20261102-01/`). 정지 조향을 점진적으로 바꾸는 별도 보정으로 기존 속도 안전 기준을 유지하면서 32개 샘플과 ±270 SDK 반응을 확인했고 (`D:/CodexResearchRuns/flybrain-teacher-steering-ramped-20261102-01/`), 가속 없이 1단 전환을 확인했다 (`D:/CodexResearchRuns/flybrain-teacher-first-gear-20261102-01/`). 네 방향 카메라의 설치 Lua는 기존 수집기의 기대 버전과 달라 실행이 차단됐고, 기존 설치물을 유지하며 실제 운영 버전의 SHA를 고정한 별도 시험 후보로 카메라를 연결했다. 제한속도 시험 `D:/CodexResearchRuns/flybrain-teacher-baseline-live-20261102-03/`은 교사 요청 1회(조향 -0.0773, 가속 0.0574, 제동 0.7615)를 전송한 후 원본 감각 주기 92ms가 80ms 한도를 초과해 중단하고 정지·장치 해제했다. 드라이버 수락과 게임 적용 검증은 구분한다. 네 시야 26묶음을 기록했으나 물리 노출 동기 미검증·training_admissible=false이며 유효 학습이나 주행 검증이 아니다. 원 주기 한도를 완화하지 않았고 정지 구간의 초기 이력만 제한적으로 재시도한다. 다음은 SDK 생산/카메라/교사 추론의 지연을 분리 계측하고 시간 기준을 보존하는 수집 경로를 검증해 연속 시연을 확보한 뒤 에피소드 분리 학습·평가한다. 겹눈/운동 시각과 비행 유래 균형·운동 회로를 단일 회로로 차량 몸에 연결하고 차체 기울기·회전·하중·타이어 상태를 되먹임해 조향·가감속·변속·셋업 적응과 빠른 레이싱 라인/랩타임을 학습하는 목표는 유지한다.

> **2026-11-02 SAC 교사 관측·라벨 범위 추가 확인:** 기존 SAC 속도 제한 실행 기록의 봉인 데이터 200결정을 원본 hash와 replay-only 상태 그대로 감사했다 (`scripts/audit_sac_teacher_training_readiness.py`, 계획 `D:/CodexResearchRuns/flybrain-sac-data-readiness-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-sac-data-readiness-20261102-01/`). 네 시야 관측 형상은 200×4×1709×3이고 세 조작 마스크는 각 200개 유효이나 라벨은 제한속도 과제의 실행 요청이다. |조향|>0.05는 93개, 가속>0.05는 27개, 제동>0.05는 3개뿐이며 한 짧은 이력이다. 이 기록은 `experimental_offline_replay_only=true`, 원본 `training_admissible=false`, camera exposure 동기 미검증이므로 현재 비행 유래 단일 회로의 실제 학생 학습에 자동 편입하지 않았다. 공개 SAC 수치 재현·차종/서킷·설정 대조는 직전 단락 참고. native AI=true/CSP=false 불일치가 계속돼 신규 교사 차량 조작·teacher-labelled 학습은 아직 개시하지 못했고 optimizer/actor/차량 제어 0이다. 다음은 제어권/시각·몸 동기를 우선 검증하고 같은 조건에서 직선·좌우 코너·가감속의 교사 시연을 더 확보해 학습/평가를 에피소드 단위로 분리한다. 학생의 경로 판단에는 전후좌우 영상과 허용된 몸 감각만 사용하고 교사/평가의 pose·지도는 입력하지 않는다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 사용자 요청 교사·차량 설정 및 모방학습 준비 확인:** 공개 AssettoCorsaGym Monza/BMW Z4 GT3 SAC 교사 checkpoint `3f30cbb1…157e84`와 현재 native static car/track이 BMW Z4 GT3/Monza로 일치한다. 같은 체크포인트의 숫자 입력/출력 재구성을 231개 공개 상태에서 재실행해 입력/코덱 오차 0, Torch/NumPy 최대 오차 3.63e-6을 확인했다 (`D:/CodexResearchRuns/flybrain-sac-teacher-reconstruction-20261102-01/`). 정확한 원 학습 manifest·차량 셋업은 없어 타이어/연료/차량 세부 설정까지 같다고 할 수 없다. 교사 입력은 pose·경로·미래 곡률을 사용할 수 있지만 학생 actor에는 영상과 허용된 몸 감각만 제공한다. 같은 차·트랙의 과거 교사 75결정 실주행에서는 최대 속도 79.79km/h, 변위 35.81m, 타이어 이탈 0, 경로 gap -0.473~+0.321m였으나 전체 랩이나 독립 고속 일반화는 검증하지 못했다. 현재 controls.ini와 당시 프로필은 파일 해시가 다르지만 INI의 모든 section/key 값은 같으며 실제 실행 엔진이 그 파일을 로드했는지는 미검증. assists.ini에는 stability_control=100, auto_shifter=1 등이 기록돼 있으나 현재 엔진 적용 여부는 미확인 (`D:/CodexResearchRuns/flybrain-teacher-compatibility-20261102-01/`). 예전 SAC 수집 200결정 자료는 지도/교사 요청 재현에 사용할 수 있으나 원본은 `training_admissible=false`, 데이터셋은 `experimental_offline_replay_only=true`, 노출 동기 미검증이며 현재 비행 유래 단일 회로의 실제 주행 학습에 적격으로 승격하지 않았다. native AI=true와 CSP AI=false의 불일치도 남아 사용자의 요청에도 새 교사 차 조작/실제 학습을 개시하지 못했다. 다음은 제어권 상태/카메라·몸 시각/조작 부호와 세션 설정을 검증한 뒤 동일 조건의 교사 롤아웃을 수집하고, 경로·pose는 교사/평가에만 두어 겹눈·비행 유래 단일 회로의 조향·가감속을 학습하고 분리된 코너/랩으로 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 native/CSP AI 상태 읽기 전용 동시 구간 확인:** 현재 session heartbeat를 native 420바이트 같은 복사본 앞뒤에서 각각 읽는 3초 진단을 차량·세션 변경 없이 수행했다 (`scripts/probe_ai_status_readonly.py`, 계획 `D:/CodexResearchRuns/flybrain-ai-status-bracket-plan-20261102-01/`, 원본 `D:/CodexResearchRuns/flybrain-ai-status-bracket-20261102-01/`, 해시 감사 `D:/CodexResearchRuns/flybrain-ai-status-bracket-audit-20261102-01/`). 동일 session generation에서 59개 서로 다른 native packet 전부 heartbeat 전/후 `ai_controlled=false`, native byte 416의 디코드 값 true였다. heartbeat는 독립적인 physics tick이 아니고 파일 갱신 주기/전달 지연이 있어 앞뒤 읽기만으로 정확히 같은 물리 시점이나 어느 쪽의 권한 상태가 진실인지 증명하지 않는다. 과거 정지 shadow 원본 2340개에서는 byte 416이 0이었다. 현재 Lua는 `ac.getCar(0).isAIControlled`을 기록하며 설치된 CSP SDK는 이 필드를 AI/자동운전 제어 여부로 설명한다. 두 인터페이스가 불일치하므로 양쪽 동의 검사 유지, actor/차량/lifecycle 변경 0. 다음은 검증된 원본 native ABI·버전과 실제 상태 전이의 원인을 확인하고 안전한 기준을 확보하기 전에는 통합 주행을 시작하지 않는다. 이어 동일 50ms 정지 shadow, 움직이는 시각·몸 정합, 단일 회로 조향·가감속·변속·셋업 및 유효 완주/라인·랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 process sampler 실시간 통합 시도 보류 — AI 상태 불일치:** 기존 8초 정지 camera→FlyVis→단일 회로 shadow의 native sampler만 별도 프로세스 adapter로 바꾼 후보 (`src/flybrain_drive/native_process_rows.py`, `scripts/record_native_process_unified_shadow.py`)를 만들고 관련 프로세스/이력/guard 검사 15개를 통과했다. 첫 실제 시도 `D:/CodexResearchRuns/flybrain-native-process-shadow-20261102-01/`은 기존 CSP lifecycle API로 pause 후 native `isAIControlled` 검사에서 true가 나와 시작 전에 중단했고 회로 worker는 정상 정지했다. 세션 heartbeat의 `ai_controlled`는 false여서 두 권한 자료가 불일치한다. 본 작업이 pause했던 상태는 검증된 API로 원래 unpaused/live 상태로 복구했으며 receipt `D:/CodexResearchRuns/flybrain-native-process-shadow-restore-20261102-01/`, camera inactive·actor/차량 제어 0이다. 앞선 실제 native 프로세스 읽기 80건과 추가 읽기 80건에서 native AI flag는 모두 true (`D:/CodexResearchRuns/flybrain-native-process-live-20261102-01/`, `...-02/`); 이 관측만으로 어느 자료가 실제 권한을 제대로 나타내는지 단정하지 않는다. 두 번째 후보 `scripts/record_native_process_unified_shadow_v2.py`는 native 검사를 pause 전으로 이동했다. 시도 `D:/CodexResearchRuns/flybrain-native-process-shadow-20261102-02/`는 같은 flag로 pause 전 중단하고 worker 정상 종료, 세션은 unpaused/live 그대로였다. 이 두 시도는 실제 process sampler와 카메라/단일 회로의 결합 성능을 검증하지 못했으며 권한 체크를 우회하지 않았다. 다음은 native 420바이트 ABI와 CSP `car.isAIControlled`의 관측 대상·시각·상태 전이를 읽기 전용으로 대조해 불일치를 해결하거나 원인을 분리한다. 이후 같은 50ms guard로 정지 shadow→움직이는 시각·몸 정합→단일 회로 조향·가감속·변속·셋업 및 유효 완주/라인/랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 실제 native 공유 메모리 독립 프로세스 읽기 전용 확인:** 기록 바이트 오프라인 통합 뒤 기존 `Local\acpmf_physics` 420바이트 매핑을 새 수집 프로세스에서 조작 없이 읽는 짧은 진단을 수행했다 (`scripts/probe_native_process_live_readonly.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-native-process-live-plan-20261102-01/`, 원본·결과 `D:/CodexResearchRuns/flybrain-native-process-live-20261102-01/`). 두 번 같은 복사본 기반 80건을 host receipt 순서대로 받고 서로 다른 packet 80개, 최대 host receipt 간격 6.343ms, 자식 exit 0 및 terminal count 80을 확인했다. 5ms 수집 설정의 짧은 읽기 전용 검사는 실제 camera 노출/physics 원천 시간·지속 부하·기존 회로 결합·차량 제어의 검증이 아니다. lifecycle 요청·desktop 입력·actor·차량 제어 0. 다음은 같은 process sampler와 body-history bridge를 기존 단일 회로 shadow에 연결해 정지 상태에서 50ms guard, 큐/작업 종료와 장시간 부하를 검증하고, 움직이는 시각·몸 정합 및 단일 회로 조향·가감속·변속·셋업, 유효 완주/라인·랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 프로세스 수집→기존 몸 이력/guard 오프라인 통합:** `src/flybrain_drive/process_body_history_bridge.py`를 추가해 별도 프로세스 샘플을 기존 `OnlineBodyShadow`의 원본 바이트 해시·copy receipt·패킷 검사를 거쳐 같은 `PerViewBodyHistory`에 넣는다. sampler stale/sequence/큐 오류나 원본 복사 오류 발생 시 기존 `UnifiedHostGuard.failed`를 고정하며 명시적 재시작 전 더 읽지 않는다. 실제 spawn 자식이 합성 420바이트 패킷을 전달한 뒤 원래 host receipt가 몸 관측까지 유지되고 guard가 첫 출력을 억제하는지, 계산 중 원래 receipt가 50ms를 넘으면 출력이 거부·실패 고정되는지 검사했다. sampler stale 및 queue full에서도 이력이 늘지 않고 guard가 실패하는지 확인했다. 관련 process·bridge·원래 guard·native ABI 검사 26개 통과 (`tests/test_process_sampler_guard_integration.py`, `tests/test_native_process_sampler.py`, `tests/test_unified_host_guard.py`, `tests/test_ac_teacher_physics.py`, 해시/테스트 기록 `D:/CodexResearchRuns/flybrain-process-history-guard-tests-20261102-01/`). 소형 합성 회로·입력과 합성 host receipt의 오프라인 통합이며 실제 GPU 회로·원본 shared memory·동시 IPC/지연·물리 시각 및 실제 운전은 미검증이다. 새 bridge는 기존 production ticker에 연결하지 않았고 actor/차량 제어 0. 다음은 실제 원본 native shared memory에 대한 읽기 전용 process와 기존 ticker의 종단 간 정지 shadow에서 같은 50ms 기준·누락/종료·시계/지연을 측정한 뒤, 움직이는 시각·몸 정합과 단일 회로 조향·가감속·변속·셋업 및 유효 완주/라인/랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 native 별도 프로세스 수집 후보:** `src/flybrain_drive/native_process_sampler.py`에 읽기 전용 native 420바이트 공유 메모리 수집 후보와 기록 바이트 전용 테스트 경로를 추가했다. 독립 프로세스가 두 번 같은 복사본을 사용하고 host 복사 시각·원본 authority 기록을 유지한다. 큐는 유한 크기이며 포화 시 명시적 생산자 오류를 고정하고, 소비는 연속 sequence와 50ms 이내 host receipt를 요구하며 오래되거나 미래인 샘플 및 sequence 오류 이후 계속 읽기를 차단한다. 종료 marker 없는 자식 종료도 오류다. Windows 파이프 정상 종료 후 `BrokenPipeError`를 처리하도록 첫 테스트 실패를 수정했다. 실제 spawn 자식의 정상 복사·stale 오류 고정·큐 포화 오류 고정·정상 종료·강제 종료 및 기존 native ABI 관련 검사 19개 통과 (`tests/test_native_process_sampler.py`, `tests/test_ac_teacher_physics.py`). 큰 큐 포화 후 생산자 feeder가 대기하지 않도록 실패 시 버퍼를 폐기하고 오류를 고정하며 실제 자식 종료를 검사했다. 기존 shadow의 기록 바이트 80개를 정확한 순서와 해시로 전달했고 자식 종료 코드 0/terminal count 80을 확인했다 (`scripts/probe_native_process_sampler_recorded.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-native-process-recorded-plan-20261102-03/`, 결과 `D:/CodexResearchRuns/flybrain-native-process-recorded-20261102-03/`). 02 성공 후 큐 포화 종료 처리를 보강하고 모듈 해시 고정 03을 다시 실행했다. 첫 01 실행은 module 해시를 계획에 적었으나 프로그램이 대조하지 않아 별도로 보존하고, 해시 검사 추가 후 02에서 다시 실행했다. 이 후보는 기록 바이트/합성 시각만 검증했으며 실제 shared memory·ticker/기존 guard 통합, actor 및 물리 시각 검증이 아니다. 기존 production 경로는 변경하지 않았고 차량 제어 0. 다음은 기존 guard와 offline 결합하여 입력 수집 시각 보존, 큐 지연·누락·종료 시 fail-closed를 검증한 뒤 실제 정지 shadow의 read-only 수집을 계측한다. 이어 움직이는 시각·몸 정합, 단일 회로 조향·가감속·변속·셋업 및 유효 완주/라인/랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 기록 native 바이트의 프로세스 간 전송 시험:** 실제 정지 shadow 기록의 native 420바이트 원본 160개를 해시 고정해 offline 재생하고, spawn 프로세스에서 각 바이트 해시·패킷 ABI를 검사한 후 용량 128 FIFO로 전달했다 (`scripts/probe_recorded_native_ipc.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-native-ipc-plan-20261102-01/`, 결과/개별 행 `D:/CodexResearchRuns/flybrain-native-ipc-20261102-01/`). 소비측에서 순번·바이트 해시·패킷을 재검사한 결과 160개 모두 순서대로 보존, 생산자 정상 종료, 생산 receipt 최대 간격 2.88ms였다. 20개 수신 후 소비측을 인위적으로 150ms 멈추자 전달 나이 최대 147.67ms였고 50ms 초과 40개는 실험의 명시적 판정에서 거부, 이후 60번째 시퀀스부터 다시 신선했다. 분리된 생산이 계속돼도 소비측 지연 시 오래된 데이터가 누적되므로 신선도를 재검사해야 한다. 이는 기록 바이트를 현재 시각으로 재생한 synthetic IPC이며 실제 shared memory 접근·전송 손실·source 물리 시각·기존 production guard 연결은 증명하지 않는다. actor/실제 차량 제어 0. 다음은 생산 중 중단/큐 포화·시퀀스 및 종료 오류를 명시적으로 시험하고 프로세스 수집 후보를 실제 guard와 결합한 오프라인 통합 검증 후, 실제 정지 shadow에서 동일 50ms 기준과 지연 원인을 계측한다. 이후 움직이는 시간 정합·단일 회로 조향·가감속·변속·셋업 및 유효 완주/라인/랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 센서 수집 스레드의 인위적 GIL 경쟁 실험:** 실제 `NativeCopySampler`와 body ticker가 같은 Python 프로세스의 별도 스레드임을 확인했다. 게임·센서에 접근하지 않는 새 실험에서 2ms 간격 수집 스레드와 별도 프로세스 수집을 동시에 실행하고, 150ms 순수 Python 바쁜 루프를 동일 프로세스에 가했다 (`scripts/probe_sampler_gil_isolation.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-sampler-gil-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-sampler-gil-20261102-01/`). Python 전환 간격 5ms에서 부하와 겹친 최대 간격은 스레드 19.12ms, 별도 프로세스 2.86ms였다. 인위적으로 250ms로 늘리면 스레드 150.09ms, 프로세스 2.73ms였다. 이 실험은 GIL 경쟁이 비슷한 공백을 만들 수 있음을 보이는 합성 재현이지, 기존 실제 150ms 지연의 원인이나 native 센서 IPC·프로세스 전달 지연의 증거가 아니다. 별도 프로세스로 샘플을 받아도 제어 계산이나 본 프로세스가 멈추면 오래된 출력 문제가 자동 해결되지 않는다. 실험은 simulator/actor/차량 제어·optimizer 0. 다음은 실제 sampler 경계에 맞는 독립 프로세스의 기록·시간·종료·누락·IPC 지연을 오프라인에서 검증하고, 50ms 거부 기준을 유지한 채 실제 정지 shadow의 운영체제/GC/스레드 계측으로 원인을 확인한다. 이어 움직이는 시각·몸 동기와 단일 회로 조향·가감속·변속·셋업, 유효 완주/라인/랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 실제 정지 shadow의 실행 지연 감사 및 우선순위 전환:** 최근 합성 투영 후보들은 코너 쌍 개선 기준에 반복 미달했다. 단순 유사 튜닝 반복을 중단하고 실제 센서→단일 회로의 신선도와 지속 실행을 우선한다. 과거 142 카메라 그룹의 정지 shadow 원본 네 파일 SHA를 고정해 486 정상 출력과 종료된 1회 신선도 거부를 감사했다 (`scripts/audit_shadow_stall.py`, 계획 `D:/CodexResearchRuns/flybrain-shadow-stall-audit-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-shadow-stall-audit-20261102-01/`). 정상 계산 p50/p95/max 4.52/6.63/7.53ms, 정상 출력 시 몸 receipt 나이 p50/p95/max 6.68/9.44/11.83ms, 처리 대기 p95 0.219ms였다. 거부된 한 결정은 계산 150.11ms, 출력 몸 receipt 나이 151.00ms로 50ms 제한 초과다. 그 결정 중 native 복사 receipt 3개 사이에 최대 145.72ms 간격이 있었고, 기록된 카메라 처리 구간은 이 구간과 겹치지 않았다. 이는 host 관측 시점과 한 번의 사후 분석이며 OS·스레드·GIL·장치 중 원인이나 노출/물리 tick 시점을 입증하지 않는다. 안전 기준 완화·actor/차량 제어 0. 세 과거 진단의 결과 파일은 저장됐으나 해당 핸들은 여전히 살아 있고 프로세스 CPU 시간은 증가해 종료된 것으로 취급하지 않았다. 다음은 센서 수집과 회로 worker의 scheduler/GIL/GC/디바이스 대기를 계측하는 재현 가능한 오프라인 시험을 먼저 만들고, 실제 정지 shadow에서 같은 신선도 기준으로 지연 원인과 개선을 검증한다. 이후 움직이는 시각·몸 시각 정합, 단일 회로 조향·가감속/변속·셋업, 유효 완주와 라인·랩타임 검증을 순서대로 진행한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 공통 조향을 억제한 코너 쌍 방향 — screen 실패:** 후보 FlyVis와 단일 재귀 회로에서 학습 합성 장면의 좌우 코너 출력차 기울기를 공통 출력 기울기에 직교화하고 원소 변화 최대 0.02의 단일 방향을 적용했다 (`scripts/fit_t4d_orthogonal_pair.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-t4d-orthogonal-pair-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-orthogonal-pair-20261102-01/`). 이는 optimizer 갱신 0회지만 공간 투영은 실제 변경한 후보이며 actor 적격이 아니다. 학습 장면을 별도 전체 이력으로 재생하면 쌍 차는 0.00292918→0.00293046 (+1.289e-6), 공통 출력 변화는 +1.86e-9였다 (`D:/CodexResearchRuns/flybrain-t4d-orthogonal-fit-audit-20261102-01/`). 다른 질감이지만 이미 본 합성 평가에서 공통 변화 +1.17e-6, 여섯 쌍 5% 이상 악화 없음, 평균 쌍 오차 개선은 0.00334%로 사전 1% 미달 및 screen 실패 (`D:/CodexResearchRuns/flybrain-t4d-orthogonal-pair-audit-20261102-01/`). 공통 출력을 유지하는 제한 방향이 존재하지만 이 방향의 코너 구분 개선은 작고 일반적 원인이나 실제 폐루프 성능을 증명하지 않는다. 학습 장면 감사 결과 파일은 저장됐으나 실행 핸들은 확인 시 살아 있으며 이전 두 추적 프로세스도 여전히 살아 있다. 후보 미채택·실제 차량 제어 0, 합성 고정 몸·생리 미검증. 다음은 실효 방향과 동역학을 더 넓은 장면에서 확인하고 시간·몸 감각 정합 및 주행 학습을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 코너별 공간 변화와 회로 상태 추적:** 직전과 같은 후보/기준 및 고정 몸 이력의 합성 좌우 코너에 대해 41결정의 공간 투영, 적용 전류, 회로 상태, intent를 기록했다 (`scripts/trace_t4d_differential_state.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-differential-trace-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-differential-trace-20261102-01/`, 감사 `D:/CodexResearchRuns/flybrain-t4d-differential-trace-audit-20261102-01/`). 조향은 직전 저장 재생과 기준·후보 모두 일치했다. 후보 변화의 좌우 평균 RMS와 좌우 차이의 RMS는 공간 포트 각각 1.4874e-4/1.5643e-5, 회로 상태 전체 9.3580e-7/1.2070e-7, intent 두 축 2.4640e-6/2.5579e-7이다. 서로 다른 크기·단위의 신호이며 전류와 상태 RMS는 9728원소 전체를 포함하므로 단계 간 전달률로 비교하지 않는다. 이 두 이력에서 후보 변경이 좌우를 구분하는 성분보다 공통 성분을 크게 만들었음을 보이나 근본 원인·일반성을 입증하지 않는다. 결과 파일은 저장됐으나 이 프로세스와 직전 포화 검사의 프로세스는 확인 시 여전히 살아 있어 종료 여부를 별도 추적한다. 실제 생리·주행 미검증, optimizer/actor/차량 제어 0. 다음은 학습 손실의 쌍 차이 기울기가 공통 성분을 줄이고 코너별 차이를 키울 수 있는지 제한된 공간 포트 방향을 별도로 검증하고 신규 장면 및 폐루프에 연결한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 공간 투영 포화 검사:** epsilon 대조의 기준·후보 투영을 동일 단일 재귀 회로와 기존에 본 합성 평가 질감의 zero-yaw 직선/양 코너 세 이력에 적용하고 공간 projection의 tanh 전 출력을 기록했다 (`scripts/trace_t4d_spatial_saturation.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-spatial-trace-plan-20261102-01/`, 기록 `D:/CodexResearchRuns/flybrain-t4d-spatial-trace-20261102-01/`, 감사 `D:/CodexResearchRuns/flybrain-t4d-spatial-trace-audit-20261102-01/`). 저장 결과는 직전 평가 조향과 기준·후보 모두 최대 차 0, tanh 입력 절댓값 최대 0.9212, |입력|≥2 비율 0, 0.5×tanh의 최소 국소 미분 0.2361이다. 따라서 이 세 이력에서 해당 tanh가 강하게 포화돼 있다고 볼 근거는 없다. 후보의 tanh 이후 공간 포트 변화 RMS는 0.0001505이며, 최종 조향 변화는 세 조건 +4.30e-6~+4.46e-6로 거의 공통이고 좌우 코너 차 변화는 1.66e-7이었다. 이 수치는 차원이 달라 단계별 전달률로 단정할 수 없다. 원인은 후속 회로·상태·조건 및 공간 변화 방향의 공통 성분일 수 있으나 아직 분리되지 않았다. 실행 결과 파일은 완료 상태로 저장됐으나 실행 종료 핸들은 확인 시 여전히 살아 있어 종료 여부를 별도 확인해야 한다. optimizer/actor/차량 제어 0, 합성·세 이력·고정 몸이며 생리·실제 주행 미검증. 다음은 동일 몸 이력에서 좌우 코너 간 공간 포트 차이와 회로 내 반응을 직접 추적하고, 차별 신호와 공통 신호를 분리해 학습과 폐루프를 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 학습기 epsilon 대조 — 연결 변화 증가, screen 실패:** 동일 초기 단일 회로·전체 41결정 역전파·한 번 갱신과 공간 투영 상한 0.02를 유지하고 Adam epsilon만 1e-8에서 1e-12로 변경했다 (`scripts/train_t4d_adam_epsilon.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-adam-epsilon-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-adam-epsilon-20261102-01/`, 감사 `D:/CodexResearchRuns/flybrain-t4d-adam-epsilon-audit-20261102-01/`). 같은 초기 손실 0.000702569, 기울기 절댓값 중앙 1.431e-9로 기존 epsilon보다 작았다. 투영 갱신 L2는 0.04567→0.15445, 원소 최대값은 0.002502→0.003006으로 증가했다. 그러나 기존에 본 합성 평가 질감에서 쌍 평균 제곱오차 개선은 약 0.00741%에 그쳐 사전 1% 기준에 미달, 여섯 쌍 5% 이상 악화 제한은 통과했으나 전체 screen 실패다. 조향 쌍 변화가 매우 작아 epsilon만이 유일 원인이라 할 수 없다. 후보 미채택, optimizer 한 번, actor/차량 제어 0; 고정 몸 이력·생리·실제 주행 미검증이다. 다음은 공간 투영 전후 tanh 입력·포화와 회로 상태/출력의 실제 민감도를 인과적으로 측정하고, 유효 신호가 있는 조건에 학습을 연결해 새 질감과 폐루프에서 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 전체 이력 쌍 차이 학습 — screen 실패:** 이전과 같은 T4d 후보 시각·합성 코너×yaw 자료와 재귀 속도 경로에서, 각 독립 41결정의 숫자 상태를 유지하고 중간 gradient detach 없이 공간 투영만 한 번 갱신했다 (`scripts/train_t4d_fullhistory_paired.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-t4d-fullhistory-paired-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-fullhistory-paired-20261102-01/`). 여섯 합성 쌍의 공학 목표는 이전과 동일하며, 이미 본 평가 질감에서 평균 쌍 제곱오차는 약 0.00378% 개선돼 사전 1% 기준에 미달했다. 각 쌍 5% 이상 악화는 없었으나 전체 screen 실패 (`D:/CodexResearchRuns/flybrain-t4d-fullhistory-paired-audit-20261102-01/`). yaw 쌍은 최대 3.13e-6, 코너 쌍은 최대 5.24e-7 변화했다. 한 번의 갱신과 이전 두 번의 잘린 갱신은 직접적인 동량 비교가 아니며, 전체 이력 기울기만으로 이 후보의 방향 문제를 해결하지 못했다. 합성 고정 이력, 이미 본 평가 질감·가설 목표이며 생리·독립 일반화·실제 폐루프는 미검증, actor/차량 제어 0, 후보 미채택이다. 다음은 공간 투영에서 회로 출력까지의 포화·전달과 정규화/동역학을 추적하고, 새로운 시각·몸 폐루프에서 실제 기여를 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 재귀 코너 쌍의 잘린 기울기와 전체 이력 비교:** 직전 공간 투영 학습의 동일 합성 zero-yaw 좌우 코너 쌍에서, 전방 계산을 유지하고 36번째 결정까지 매번 기울기만 절단한 자동미분을 계산했다 (`scripts/audit_t4d_recurrent_pair_gradient.py`, 결과 `D:/CodexResearchRuns/flybrain-t4d-pair-gradient-20261102-01/`). 학습/추론 쌍 출력 차는 0, 투영 기울기 L2 1.027e-6이고 최대 원소로 정규화한 방향의 잘린 방향 미분은 8.004e-6이다. 같은 파라미터 방향을 ±0.01/±0.02 바꿔 전체 재귀 이력을 각각 재실행한 중심 유한차분은 6.370e-5/5.928e-5로 잘린 값의 약 7.96/7.41배였다 (`D:/CodexResearchRuns/flybrain-t4d-pair-gradient-20261102-02/`, 교차 감사 `D:/CodexResearchRuns/flybrain-t4d-pair-gradient-crosscheck-20261102-01/`). 두 방법은 기울기 절단 이력과 유한 변화량이 달라 일치해야 한다는 뜻은 아니며, 한 합성 쌍·한 방향·float32 비교로 일반적인 유일 병목이나 실제 주행을 증명하지 않는다. 02 계획의 program_sha256에는 원래 스크립트 해시가 남았고 실제 별도 래퍼 해시는 manifest에 기록됐다; 래퍼 변경은 계획 경로뿐이나 계획의 프로그램 해시와 실행 일치를 주장하지 않는다. optimizer/actor/차량 제어 0. 다음은 전체 이력에서 메모리를 제한하는 구간별 재계산 또는 유한차분 학습을 사전 설계하고 코너·회전 쌍 및 새 질감·폐루프 효과를 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 T4d 후보·재귀 속도 경로의 쌍 차이 학습 실패:** 합성 코너×yaw 자료와 후보 FlyVis 출력을 사용해 단일 회로의 실제 기존 `experimental_recurrent_speed_training_decision` / 평가 경로를 맞췄다 (`scripts/train_t4d_paired_recurrent.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-t4d-paired-recurrent-fit-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-paired-recurrent-fit-20261102-01/`). 조향 기준을 고정하고 공간 투영만 2회 갱신하며 세 코너에 대한 ±yaw 출력 차 목표 +0.024, 세 yaw에 대한 양 코너 차 목표 +0.04를 손실에 직접 넣었다. 이는 합성 공학 가설이지 생리·실제 차량 정답이 아니다. 이미 본 다른 질감의 여섯 쌍에서 사전 지정 평균 제곱오차 개선은 약 0.00177%로 요구 1% 미달, 각 쌍 5% 악화 제한은 충족하여 전체 screen 실패 (`D:/CodexResearchRuns/flybrain-t4d-paired-recurrent-audit-20261102-01/`). 코너 차는 대략 +0.005325/+0.006613/+0.010563에서 최대 2.34e-7만 변했고 yaw 차도 최대 9.32e-7 변했다. 공간 투영은 2678개 요소가 바뀌고 최대 변화 0.001610이므로 단순히 갱신이 0인 것은 아니지만, 작은 출력 영향의 정확한 원인은 아직 분리되지 않았다. 후보 미채택·actor/차량 제어 0, 합성 고정 이력·기존 노출 장면이며 독립 생리·실제 주행은 미검증이다. 다음은 실제 재귀 경로에서 코너 쌍 출력에 대한 공간 투영 기울기·포화와 목표 범위/동역학을 진단하고 인과적으로 유효한 변경을 찾은 뒤 새로운 질감·폐루프에서 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 T4d 후보 포함 코너 보존 학습 — 좁은 screen 통과, 반응 개선 미확인:** 합성 코너×회전 3×3 장면 두 질감의 저장 retina를 원본 및 T4d 후보 FlyVis로 재생했다 (`scripts/replay_t4d_candidate_road_yaw.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-road-yaw-retina-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-road-yaw-retina-20261102-01/`). 원본 신경 이력은 기존 저장값과 최대 4.33e-7 이내다. 기존 단일 회로 공간 투영 및 조향 참조만 제한 학습하며 합성 목표 0.02×bend−0.1×yaw와 zero-yaw 코너 출력 보존 항을 사용, 두 번 갱신했다 (`scripts/train_t4d_corner_yaw_candidate.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-t4d-corner-yaw-fit-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-corner-yaw-fit-20261102-01/`). 다른 질감이지만 과거에 이미 본 합성 평가 장면에서 목표 MSE가 0.0036962→0.0035420 (4.17% 개선), zero-yaw 출력 변화는 최대 0.001316으로 사전 1%/0.002 screen 충족 (`D:/CodexResearchRuns/flybrain-t4d-corner-yaw-fit-audit-20261102-01/`). 그러나 9조건 출력이 모두 음수이고 후보 변화가 거의 일정한 +0.001312~+0.001317이며 좌우 코너 차이 변화는 최대 2.21e-6, 회전 차이 변화는 최대 4.93e-6이다. 따라서 통과의 대부분은 공통 출력 치우침 보정이며 질감에 따른 방향 반응 개선이나 주행 향상을 보여주지 않는다. 학습/평가는 `sensory_decision` 경로로 직전 회전 실험의 `experimental_recurrent_speed_decision`과도 다르다. 후보 actor 미적격·실제 제어 0, 생리·독립 일반화·폐루프 미검증. 다음은 공통 편향과 별도로 코너 및 회전의 짝 차이에 직접 손실과 평가를 두고 실제 사용할 동일 재귀 경로와 새로운 질감에서 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 T4d 후보에서 공간·운동 투영 인과 개입:** 앞선 두 질감×두 회전 속도 동일 후보 시각과 고정 몸 이력에서 상대운동 투영, 공간 대비 투영을 각각·동시에 0으로 한 16개 조건을 실행했다 (`scripts/ablate_t4d_candidate_factorial_yaw.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-t4d-factorial-ablation-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-factorial-ablation-20261102-01/`). 개입 없는 후보 출력은 직전 저장 결과와 네 조건 모두 최대 차 0 (`D:/CodexResearchRuns/flybrain-t4d-factorial-ablation-audit-20261102-01/`). 마지막 5결정의 -yaw와 +yaw 조향 차는 질감 879601의 속도 0.13/0.23에서 기본 -0.003262/-0.004343, 운동 투영 차단 -0.005526/-0.009028, 공간 투영 차단 +0.002923/+0.005586, 둘 다 차단 +0.000785/+0.001003이다. 질감 879602에서는 각각 +0.003087/+0.004443, +0.001274/+0.001053, +0.002115/+0.004058, +0.000294/+0.000690이다. 이 네 조건에서 공간 투영 차단은 질감 간 부호 불일치를 없애며, 운동 투영만 차단하면 불일치가 남는다. 이는 해당 공간 투영이 불일치에 인과적으로 기여함을 보이지만, 투영 차단은 비생리적이고 공간 경로는 코너 선행 정보에도 관여하므로 제거를 정당화하지 않는다. 이미 본 합성 질감·고정 몸 이력, 생리·독립 일반화·실제 폐루프 미검증, actor/optimizer/차량 제어 0. 다음은 코너 정보 보존을 명시적으로 제약하면서 질감에 따른 잘못된 yaw 반응을 줄이는 같은 단일 회로의 학습과 새로운 질감 및 폐루프 평가를 수행한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 T4d 후보의 질감×회전 속도 교차 비교:** 기존 합성 구면 장면의 두 질감 879601/879602와 두 회전 속도 ±0.13/±0.23rad/s를 교차한 4개 입력을 동일 원본·후보 FlyVis로 재생하고 4시야 신경 이력을 단일 회로에 각각 넣었다 (`D:/CodexResearchRuns/flybrain-t4d-yaw-factorial-retina-plan-20261102-01/`, `D:/CodexResearchRuns/flybrain-t4d-yaw-factorial-retina-20261102-01/`, `D:/CodexResearchRuns/flybrain-t4d-factorial-unified-plan-20261102-01/`, `D:/CodexResearchRuns/flybrain-t4d-factorial-unified-20261102-01/`). 원본 신경 이력은 이전 원본과 부동소수점 오차 내 일치하며 원본 단일 회로 출력은 과거 factorial 정상 조건과 네 조건 모두 최대 차 0 (`D:/CodexResearchRuns/flybrain-t4d-factorial-unified-audit-20261102-01/`). 마지막 5결정의 -yaw와 +yaw 조향 intent 차는 질감 879601에서 원본 -0.001262/-0.000289, 후보 -0.003262/-0.004343, 질감 879602에서 원본 +0.005108/+0.009572, 후보 +0.003087/+0.004443 (순서 속도 0.13/0.23)이다. 같은 속도에서도 질감별 부호가 다르므로 두 장면에서 속도 차이만이 원인은 아니며 T4d 후보도 불일치를 해소하지 못했다. 이미 검토한 두 합성 질감과 고정 몸 이력으로 질감의 어떤 특징이 원인인지, 생리·실제 주행 일반화는 증명하지 않는다. 후보 actor 미적격, optimizer/차량 제어 0. 다음은 기존 공간/운동 경로의 인과 분해를 후보에도 적용하고 코너 정보 보존과 독립 장면, 실제 시각·몸 시각 정합을 확인한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 T4d 후보의 단일 회로 반사실 비교:** 앞서 조향 부호가 달랐던 합성 회전 장면 879604/879605의 저장된 동일 전후좌우 retina를 원본 000과 비음수 보간 후보에 각각 넣고 시야별 상태를 매 경우 초기화해 전체 시간을 재생했다 (`scripts/replay_t4d_candidate_yaw_retina.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-yaw-retina-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-yaw-retina-20261102-01/`). 원본 신경 출력의 과거 저장값 대비 최대 오차는 각각 3.58e-7/3.73e-7, 후보-원본 신경 출력 RMS 차이는 0.1611/0.1563이다. 동일 단일 비행/몸 회로·고정 몸 이력·독립 recurrent reset으로 두 시각 출력을 재생했다 (`scripts/compare_t4d_candidate_unified_yaw.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-unified-yaw-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-t4d-unified-yaw-20261102-01/`, 저장 결과 감사 `D:/CodexResearchRuns/flybrain-t4d-unified-yaw-audit-20261102-01/`). 원본 회로 출력은 기존 재생 저장값과 최대 차 0이다. 마지막 5결정의 -yaw와 +yaw 조향 intent 차는 879604에서 +0.005538→+0.004123, 879605에서 -0.010909→-0.011803으로, 후보에서도 질감 간 반대 부호가 유지됐다. 두 장면은 속도와 질감도 다르므로 단일 원인·물리적 정답·생리 일반화를 결론 내릴 수 없고, 이미 본 합성 입력·고정 몸 이력은 폐루프 주행이 아니다. 후보 actor 미적격, 실제 차량 제어·optimizer 0. 다음은 같은 질감과 속도를 교차해 후보 효과의 원인을 분리하고 시각/몸 동기와 생리 기능 및 통합 폐루프를 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


> **2026-11-02 전체 flash 이력 학습과 사후 보간:** 비음수 T4d 입력 10개에 전체 1초 flash 이력의 저자 지표로 유한차분 25회와 제한 선탐색을 적용했다 (`D:/CodexResearchRuns/flybrain-t4d-fullhistory-fit-20261102-01/`). T4d 지수는 전체 -0.3120→+0.0798, 학습에 넣지 않은 radius6 -0.3119→+0.0793이었지만 T4c radius6 변화 0.101000으로 사전 상한 0.1을 넘어 15/16, 후보 미채택 (`D:/CodexResearchRuns/flybrain-t4d-fullhistory-flash-gate-20261102-01/`). 이를 본 뒤 두 비음수 후보를 0.9 비율로 보간한 탐색 후보는 원본 대비 strength 10개만 바뀌고 최대 변화 0.006425, 음수 0개다 (`D:/CodexResearchRuns/flybrain-t4d-interpolation-exploratory-20261102-01/`). 같은 flash 재검사에서 T4d 전체 +0.07067, radius6 +0.07014, T4c radius6 변화 0.09382로 16/16 (`D:/CodexResearchRuns/flybrain-t4d-interpolation-flash-gate-20261102-01/`), 저자/독립 numpy 지표 최대 차 0. 두 반경은 이미 관찰한 결과를 반영해 선택한 탐색 자료이므로 독립 일반화·생리 검증이 아니다. 저자 이동 경계 144조건 및 8세포×6속도 원본 비교를 완료했다 (`D:/CodexResearchRuns/flybrain-t4d-interpolation-edge-20261102-01/`, 분석 `D:/CodexResearchRuns/flybrain-t4d-interpolation-analysis-20261102-01/`, 비교 `D:/CodexResearchRuns/flybrain-t4d-interpolation-comparison-20261102-01/`). T4d의 기대 밝기 부호는 0/6→6/6, 최대 방향 오차 9.55°→3.67°, 최소 방향 선택성 0.061→0.181이다. 나머지 7세포는 밝기 부호 6/6을 유지했으나 최대 방향 오차는 T5a 6.66°→12.18°, T5b 12.78°→19.38° 등 일부 악화했다. 합성 부품 반응으로 실측 생리·주행 검증이 아니며 후보의 독립 일반화와 단일 회로 영향은 미검증이다. 실제 AC 주행·actor/optimizer 배포는 하지 않았다. 다음은 움직이는 경계의 방향·밝기별 반응과 다른 세포 보존을 원본과 비교하고 독립 자극 및 단일 회로 통합을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


## 목표와 현재 위치

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 상대 바람, 변속, 차량·타이어·셋업 변화도 원래 목표에 포함한다. 전체는 **하나의 연속 감각운동 시스템**이며 별도의 운동 뇌를 만들지 않는다. 생물학적 실측과 기능 근사, 자동차 대응 가설, 학습한 경계는 구분한다.

현재는 **1단계의 약 8초 정지 host 경로와 원본 출력 재생을 확인했고, 움직이는 자료의 학습 자격과 2단계 검증이 남았다**. 네 시야·차체 감각과 FlyVis·단일 회로의 정지 실행을 확인했다. 실제 주행 기록의 두 방향 중 한쪽으로 조향이 치우치는 문제가 남았다. 실제 주행 자료 일부는 `training_admissible=false`이며 평가·진단에만 사용한다. 물리 노출/몸 시각 정합, 지속 실시간 회로, 실제 회로 제어·유효 랩은 아직 없다.

## 작업 원칙 — 반복 제한

- 기존 결과·원본·실패를 보존한다. 같은 데이터와 같은 조건에서 통과 기준이나 의사결정을 바꾸지 않는 진단은 다시 실행하지 않는다.
- 각 실험 전에 **변경할 항목 하나, 기준 모델, 허용 입력, 학습/검증 분리, 성공·실패 기준, 다음 행동**을 기록한다. 첫 결과가 나오면 분석 횟수를 최소화하고 성공·실패에 따라 다음 단계로 이동한다.
- 학습은 합성 자료 또는 명시적으로 자격을 통과한 실제 TRAIN만 사용한다. `training_admissible=false`는 경로 이름이 TRAIN이어도 학습에 쓰지 않는다. 지도·절대 위치·정답 라인은 교사/보상/평가 전용이다.
- 매 실행은 고유 경로·설정·소스/체크포인트 해시·seed·종료 이유를 남긴다. 기존 저장 기록을 삭제하지 않는다. 대용량 신규 결과는 D 공간을 우선 사용하되 기존 경로·참조가 깨지지 않게 한다.
- AC 창 입력·활성화 없이 검증된 CSP 수명주기와 백그라운드 센서/제어만 사용한다. 카메라나 제어 결과가 불명확하면 같은 핸들을 확인하고 무작정 재시작하지 않는다.

## 1. 데이터와 실시간 감각 경로 확보

**할 일:** 현재 네 시야 분리 프로세스의 실제 픽셀·몸 감각 경로에 FlyVis와 단일 회로를 연결한 무조작 shadow를 구축한다. 같은 회로 상태를 유지하고, 화면/몸 결측·오래된 입력을 명시적으로 거부한다. 카메라의 원본 노출 시간과 몸 샘플 발생 시간은 확인 가능한 근거가 있을 때만 동기라고 부른다.

**통과:** 동일 세션에서 네 시야 ID·해시와 몸 패킷/시간이 보존되고, 원본 속도로 지속 처리하며 현재 프로젝트의 host freshness 제한과 실패 시 중단을 충족한다. 원본 재생이 일치하고 종료 영수증이 확인되어야 한다. 물리 시각이 미검증이면 그 한계를 그대로 기록하고 실시간 통합 성공으로만 표시한다.

**다음:** 조작 없는 정지 shadow 다음에 허용된 교사 주행 자료를 수집한다. fit에 사용할 수 있는지는 영상·몸 시각, 교사 조작의 출처/권한, 누락, 차량 상태를 개별 검증하고 사전 분리한 TRAIN/DEV/TEST 단위로 결정한다. 기존 진단 자료는 소급 승격하지 않는다.

## 2. 좌우 시각 판단과 비행 감각운동 기여

**할 일:** 현재 좌우 후보의 실제 기록 실패를 만든 시각 경계·중립 기준·출력 gain을 한 번의 사전 지정 대조로 분리한다. 이미 실행 중이던 8조건 반복 진단은 사용자 지시에 따라 중단했다. 기존 결과에서 바로 결정할 수 없는 최소 대조만 다시 설계한다. 서로 다른 곡률·속도·질감의 합성 학습/미사용 평가를 먼저 분리하고, 진단용 실제 네 시야 이력은 평가에만 쓴다. 영상 가림·좌우 교환·몸 감각 제거/교란으로 시각과 비행 유래 몸 경로의 기여를 검증한다.

**통과:** 사전에 정한 좌우·직선·시각 교란 조건 모두에서 기존 대비 개선하고, 한쪽만 좋아지거나 직선이 크게 악화되지 않아야 한다. 실제 평가에서는 양방향 표본을 개별 기록 단위로 보고, 단순 한 방향 출력보다 나아야 한다. 미래 경로 chord는 사후 대리 지표이며 정답 조향이나 레이싱 성능으로 취급하지 않는다. 결과가 실패하면 학습 설정만 반복하지 않고 해당 시각·회로 연결/출력 편향으로 돌아간다.

## 3. 학습과 조작 연결

**할 일:** 검증된 단일 회로를 사용해 시각 선행 예측, 자세·회전·하중·타이어/바람 되먹임, 조향·가속·제동을 함께 학습한다. 교사 정답과 경로 정보는 손실/평가에만 둔다. 변속과 셋업은 명시적인 단위·상태·행동 계약과 변형 조건을 마련해 순차적으로 추가한다. 학습은 이전 checkpoint/최고 checkpoint·학습 곡선·독립 검증을 남긴다.

**통과:** 미사용 조건의 좌우 코너와 직선·가감속에서 기준 모델 대비 개선하고, 감각 제거/교란이 판단에 미치는 영향이 타당해야 한다. 결측 감각을 정답으로 보정하지 않고 명시적으로 거부하거나 검증된 대체 정책을 사용한다. 실제 학습 가능 자료가 없으면 합성 단계는 진행하되 실제 학습 완료라고 말하지 않는다.

## 4. AC 주행과 최종 목표

**할 일:** 모델 출력의 단위·부호·제어권·종료 조건을 확인하고 짧은 구간부터 실제 회로 제어를 검증한다. 정지/재출발, 좌우 코너, 제동과 이탈 복구를 확인한 뒤 전체 서킷으로 늘린다. 그 다음 유효 완주 반복, 레이싱 라인 및 랩타임, 타이어·셋업과 변속 조건을 평가·최적화한다.

**통과:** 교사 없이 허용된 영상·몸 감각만 사용하는 단일 회로가 반복적으로 유효 완주해야 한다. 코스 이탈·충돌·보조 사용·실패 원인을 기록하고, 같은 차량·조건의 기준 랩과 분리된 조건의 랩타임 및 라인을 비교한다. 부품이 실행됐다는 사실이나 짧은 구간 성공은 완주나 최종 목표 달성으로 대체하지 않는다.

## 바로 실행할 작업

1. 중단한 8조건 분석의 부분 진행은 보존하되 결과로 간주하지 않는다.
2. 기존 코드에서 **실제 카메라 픽셀 → FlyVis → 단일 회로**의 live shadow 구성과 측정된 지연을 확인한다. 중복 구현 없이 가장 작은 연결 부분을 수정하고 실제 정지 상태에서 한 번 검증한다.
3. 그 결과로 1단계의 성공 또는 구체적 병목을 판단하고, 동시에 2단계의 학습/평가 자료 자격과 최소 실행 계획을 고정한다. 한 번의 확인으로 결정되지 않는 세부 분석은 다음 행동에 필요한 경우에만 추가한다.

## 2026-11-02 첫 실행 결과

`scripts/record_isolated_unified_stationary_shadow.py` 및 `D:/CodexResearchRuns/flybrain-isolated-unified-shadow-20261102-01/`: 정지 상태에서 별도 카메라의 네 시야 픽셀→FlyVis→기존 단일 회로를 실행하여 shadow 출력 454개와 시각 장면 129개를 기록했다. 카메라 145묶음의 보관 원본 검증 및 같은 요청 terminal stopped 확인. 그러나 마지막 회로 출력에서 몸 수신 후 약 162.6ms로 50ms 제한을 초과했고 시각 계산 간격도 150ms를 넘어 중단했다. 원본 결과는 `stopped_on_error`이며 1단계 지속 실시간 통과가 아니다. 물리 노출/몸 발생 시각·실제 차량 제어·생리 기능은 검증하지 않았다. 다음 우선 작업은 이 원본의 계산 구간과 스레드 대기 중 어느 부분이 50ms 초과에 기여했는지 한 번 분석하고, 실제로 통합된 경로에서 검증된 프로세스 분리/계산 변경을 적용하는 것이다. 단순히 freshness 한도를 늘리지 않는다. 관련 테스트 50개 통과.

### 미리 계산 제거 대조

동일 코드에서 held-prefix만 끈 `D:/CodexResearchRuns/flybrain-isolated-unified-shadow-no-prefix-20261102-01/`도 479개 출력 후 몸 입력 50ms 제한으로 중단됐다. 마지막 회로 처리 154.43ms와 FlyVis 처리 구간이 겹쳤다. prefix 사용 시험에서는 몸 회로 157.81ms와 prefix 165.22ms가 겹쳤다. 각각 1회이며, host 구간 겹침만으로 CPU·GPU·스케줄 원인을 확정하지 않는다. 2개 카메라 요청 각각 terminal stopped 및 전체 145/149묶음 archive 검사 확인, 차량 조작·학습0. 미리 계산을 끄는 것만으로 1단계가 통과되지 않았으므로 더 반복하지 않는다. 다음은 기존 원본 회로의 상태와 시각 입력을 유지한 별도 프로세스 경로를 먼저 오프라인 동등성 검증한 뒤 한 번의 정지 shadow로 확인한다.

## 2026-11-02 단일 회로 프로세스와 수신 공백 계측

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 목표이며 하나의 연속 회로다.

기존 회로를 별도 프로세스로 실행하도록 구현하고 `D:/CodexResearchRuns/flybrain-unified-process-parity-20261102-04/result.json`에서 기록된 365개 결정의 상태·출력 최대 차이 0을 확인했다. 재생은 실제 주행 성능이나 생리 기능 검증이 아니다. 첫 실제 연결 `D:/CodexResearchRuns/flybrain-unified-process-live-20261102-02/result.json`은 shadow 486개 뒤 출력 시 몸 감각 50ms 초과로 중단됐다. 카메라 154묶음 보존·해시 검증, worker와 카메라 정상 종료. 이전 카메라 heartbeat 오류와 stopped terminal 불일치는 원본을 보존하고 해당 오류만 별도 검증 후 복구 조건으로 인정했다. Windows monotonic 저해상도로 실패한 시계 감사 01/02와 정밀 perf_counter로 성공한 03을 보존한다.

추가 계측을 넣은 `D:/CodexResearchRuns/flybrain-unified-process-live-20261102-03/result.json`은 shadow 475개 뒤 ticker의 몸 감각 stale 및 시각 처리 간격 초과로 중단됐다. 회로 호출 476회의 부모 왕복 최대 7.593ms, 자식 계산 최대 6.106ms. `native-copy.json`의 마지막 오류 직전 인접 수신 사이 147.418ms 공백이 관측됐다. `worker-timings.json`, `ticker-events.json`, `events.json`과 원본 149묶음을 보존하며 camera/worker terminal stopped 및 archive 해시를 확인했다. 이 시험은 수신 공백을 확인하지만 프로세스/GIL/GPU/OS 중 원인을 확정하지 않는다. 학습 업데이트·actor 배포·조작은 0이며 물리 노출 동기와 실제 주행은 미검증. 관련 회귀 16개 통과.

다음 의사결정: 시각 계산과 몸 수신/회로 소유의 독립 프로세스 경계를 설계해 원본 시간·결측·재귀 상태·종료 영수증을 유지하고, 오프라인 동등성과 실패 경로를 먼저 확인한 뒤 하나의 정지 shadow에서 동일한 freshness 기준으로 검증한다. 기준 완화나 같은 조건 재반복은 하지 않는다. 이 단계 통과 뒤 자료 자격과 좌우·몸 감각 기여를 검증하고, 조향·가감속·변속·셋업 및 라인/랩타임 학습으로 이어간다.

## 2026-11-02 시각 소유 프로세스 구현 및 미해결 종료

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응은 원래 목표이고 별도 운동 뇌가 아니다.

`src/flybrain_drive/flyvis_process.py`에 네 시야 프레임 순서와 반복 FlyVis 상태를 소유하는 GPU 프로세스를 추가하고 `scripts/record_separate_visual_stationary_shadow.py`를 준비했다. 기존 회로 50ms·시각 150ms 제한, 원본 body context와 단일 회로 상태는 유지한다. 단, 새 경로는 정확한 소스 해시가 일치하는 합성 GPU 출력 동등성 및 stopped 증거가 없으면 pause 전에 중단하도록 잠갔다. 실제 session에서 아직 시험하지 않았다. 테스트 `tests/test_flyvis_process.py`와 기존 회로·ticker·guard를 합쳐 18개 통과는 프로토콜 단위 검증이며 GPU/실시간 성공이 아니다.

`D:/CodexResearchRuns/flybrain-separate-visual-parity-20261102-01/`과 `...-02/` 합성 비교는 worker stopped 응답 후 프로세스 종료 확인 실패로 결과 동등성을 산출하지 못했다. 자식 각각 PID 17952와 13520에서 Windows GetExitCodeProcess=0이었으나 WaitForSingleObject=WAIT_TIMEOUT이며 부모도 살아 있다. 03도 종료 대기를 60초로 늘렸으나 동일하게 실패했다. 자식 PID 32176은 GetExitCodeProcess=0이지만 WaitForSingleObject=WAIT_TIMEOUT이며 부모 PID 33972도 실행 상태였다. 종료 응답과 프로세스 종료는 구별하며 원인이나 GPU 드라이버 책임을 단정하지 않는다. 신규 시각 작업자를 무작정 재시작하지 않는다. 후속: 기존 실행 핸들과 자식의 종료 상태를 확인하고, 신뢰할 수 있는 정리·출력 동등성 증거를 얻기 전까지 실시간 경로를 열지 않는다. 학습·차량 조작·실제 완주·물리 동기 검증 없음.

## 2026-11-02 CPU 종료 대조와 증거 보존 보강

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·바람·셋업 적응도 전체 목표이며 단일 감각운동 체계를 유지한다.

기존 GPU 시험 PID 17952/13520/32176은 exit code 0임에도 Windows wait timeout, 부모 16676/45400/33972도 여전히 종료되지 않았다. 추가 GPU 프로세스를 만들지 않았다. CPU 전용 대조 `D:/CodexResearchRuns/flybrain-cpu-process-exit-20261102-01/result.json`에서 일반 Python과 CPU PyTorch import가 정상 종료했고, `D:/CodexResearchRuns/flybrain-cpu-spawn-exit-20261102-01/result.json`에서 CPU PyTorch 객체를 Windows spawn/pipe로 넘겨 계산·stopped 응답 후 정상 종료했다. 이로써 일반 Python/CPU spawn 자체만으로 재현되지 않지만 GPU 드라이버가 원인이라고 입증하지는 않는다.

`scripts/verify_separate_flyvis_process_parity.py`는 이후 고유 --output으로 프레임별 원본 합성 픽셀·시각·출력을 worker 종료 전에 보존하고, 실패 시 result에 PID·생존 상태를 기록한다. `src/flybrain_drive/flyvis_process.py`의 시작 실패는 worker PID·생존 상태를 예외로 전달하며 실제 세션 준비 스크립트도 이를 기록한다. 정확한 시각 프로세스 동등성 및 정상 종료 증거가 없어 live gate는 여전히 닫혀 있다. 새 GPU 비교를 반복하기 전 기존 실행 핸들의 종료 원인을 확인하고 자원 복구 여부를 판정한다. 새로운 학습/실제 운전은 없으며 관련 18개 테스트 통과.

## 2026-11-02 CPU FlyVis 부분 재생

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응 역시 전체 목표이며 별도 운동 뇌가 아니다.

별도 시각 소유자의 CPU 실행을 명시적으로 지원했고, FlyVis가 기본 GPU 난수 장치를 캐시해 CPU 생성 시 실패하는 문제를 해당 초기화 범위에서만 수정했다. 실시간 스크립트의 gate는 CUDA 동등성 증거만 받는다. 원본 체크포인트와 입력 순서, 상태, 150ms 제한을 바꾸지 않았다. CPU 시험 01은 장치 불일치로 시작 실패했고 02는 모듈 import 오류로 시작 실패했으며 각각 원본을 보존했다. 수정된 03 `D:/CodexResearchRuns/flybrain-separate-visual-parity-cpu-20261102-03/result.json`은 합성 영상 2프레임 처리 후 시각 간격 초과로 중단됐다. 부분 입력·출력·시각은 해시 검증된 `partial-observations.npz`에 있다. worker PID 42788은 중단 시 생존으로 기록됐으나 이후 OS 핸들이 없음을 확인했다.

`scripts/replay_partial_flyvis_cpu_parity.py`는 위 기록의 실제 두 프레임, 간격 0.0398576초, 동일 시각망·pool 가중치와 hold 계산을 별도 CPU 모델에 재생했다. `D:/CodexResearchRuns/flybrain-separate-visual-partial-replay-cpu-20261102-01/result.json`: 2프레임 최대 차이 0. 합성 부분 결과이며 원래 03을 성공으로 바꾸지 않고 GPU 동등성, 물리 동기, 시각 생리 또는 실시간 통과로 취급하지 않는다. 기존 GPU 자식 17952/13520/32176은 재확인 시 신호 미완료였다. 다음은 기존 GPU 종료 원인과 자원을 확인하고 정확한 CUDA 증거 및 종료 계약을 확보한 뒤 동일 제한의 live shadow를 실행한다. 조작·학습 없음; 관련 18개 테스트 통과.

## 2026-11-02 GPU 자원 복구, 시각 분리 동등성 및 실제 shadow

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응을 포함하고 하나의 감각운동 체계를 유지한다.

기존 GPU worker 17952/13520/32176은 종료 코드 0임에도 핸들이 신호되지 않고 3초 동안 각각 CPU 약 3.2초를 소비했다. 소유 작업의 종료 응답과 상태를 재확인하고 Win32 TerminateProcess로 각각 정리했다. 결과 `D:/CodexResearchRuns/flybrain-gpu-exit-recovery-20261102-01/` 및 `...-02/`; 자식과 부모 전원 OS 핸들 소멸 확인. GPU 동등성 04는 시각 worker 정상 종료·최대 차이 3.73e-8이었으나 기준 계산 부모가 종료 중 멈춰 해당 부모만 별도 정리했다 (`...exit-recovery-20261102-03/`).

FlyVis engine에 종료 전 GPU tensor 제거·synchronize/cache 반환을 추가하고, baseline도 외부 state 참조를 해제하도록 수정했다. `D:/CodexResearchRuns/flybrain-separate-visual-parity-20261102-05/`는 합성 3프레임의 기존 계산 대비 최대 차이 4.47e-8, worker `device_released`와 stopped, 기준 프로세스 자원 반환 및 OS exit 0을 확인했다. 별도의 `exit-receipt.json`은 결과 해시와 원래 실행의 종료 상태를 연결한다. 이는 생리 또는 실제 운전 성능 증거가 아니다.

이 증거 이후 `D:/CodexResearchRuns/flybrain-separate-visual-live-20261102-01/result.json`에서 실제 정지 네 시야·몸 감각·시각/단일 회로 별도 프로세스 연결을 실행했다. 출력 135개, 시각 41개 후 몸 수신 50ms 제한 초과로 중단. 거부된 마지막 결정에서 IPC 전체 89.997ms, 회로 내부 3.995ms, 회로 종료 후 부모 수신 85.643ms였고 native 인접 수신에 87.544ms 공백이 겹쳤다. 이 사실만으로 OS/GIL/하드웨어 원인을 확정하지 않는다. 원본 카메라 52묶음 해시 검증·camera terminal과 GPU/회로 worker 정상 종료 및 두 PID 소멸을 확인했다. 최종 세션 일시정지 해제, brake1, AI/카메라 off; optimizer/actuator/UI 입력 0. 물리 노출 동기·운전·생리 미검증, 1단계 미통과. 다음은 몸 수신과 회로 상태의 독립 실행 경계를 설계하고 실제 시간·결측·state 연속성·종료를 오프라인 검증한 뒤 같은 제한으로 검증한다.

## 2026-11-02 몸 수신·단일 회로 소유 분리

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응을 포함하고 별도 운동 뇌가 아니다.

`src/flybrain_drive/body_circuit_process.py`는 동일한 `UnifiedLiveBodyTicker`, `UnifiedHostGuard`, 단일 회로를 소유한 별도 CPU 프로세스가 독립 native 원본을 읽고, 원본 시간·과거 시야 몸 컨텍스트와 영상 feature만 부모에서 받는다. 조작 없음. 합성 영상·실제 정지 몸 패킷 `D:/CodexResearchRuns/flybrain-body-circuit-readonly-20261102-03/`은 priming 1, 출력 14, 79개 native, 정상 종료. 호스트 시각만 다루며 물리 동기는 미검증이다.

실제 네 시야 `D:/CodexResearchRuns/flybrain-body-owner-live-20261102-01/`: 원본 55묶음 archive hash 검증, priming 1·출력 145 후 영상 render 하한이 150ms를 넘어 ticker가 중단됐다. 성공 출력 시 body age 최대 9.733ms, 자식 native 인접 복사 최대 간격 3.170ms였고 이전 회로 결과 부모 IPC 지연은 결정 경로에서 제거했다. 이는 한 번의 짧은 정지 실험으로 지속 실시간 통과가 아니다. 상위 result는 `Owner unavailable`·worker close unknown으로 끝났지만 자식 원본 `body-owner/result.json`에는 실제 stale view 오류, events 146, copies 718, cleanup errors 없음이 기록돼 있다. 자식 receipt sequence 0은 예외 때 로컬 sequence 갱신이 유실된 로깅 결함이며 실제 accepted count가 아니다. camera terminal stopped, visual worker stopped/device released; optimizer/actuator 0.

오류 시 마지막 수신 순번을 보존하고 worker 종료 후 원본 events/native hash를 검증해 상위에 failure receipt를 수집하도록 수정했다. 순서·실패·증거 변조 포함 회로/시각 관련 21개와 추가 failure receipt 테스트 통과. 이 수정으로 이전 03 read-only proof의 body-owner 소스 해시는 더 이상 일치하지 않으므로 새 read-only 증거 없이 live gate는 열리지 않는다. 다음은 새 소스의 read-only 증거를 확보한 뒤 원본 화면 수신·시각 계산·게시 간 150ms 공백의 원인을 밝혀 시각 경로를 개선한다. 실제 시각 생리·주행·물리 동기·유효 랩은 미검증이다.

## 2026-11-02 영상 전달과 컨텍스트 지연 구간

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 목표이며 별도 운동 뇌가 아니다.

몸 소유자 오류 영수증 보강 뒤 read-only `D:/CodexResearchRuns/flybrain-body-circuit-readonly-20261102-04/`는 실제 정지 몸 수신·합성 영상에서 priming 1/출력 14 및 정상 종료했다. 실제 정지 `D:/CodexResearchRuns/flybrain-body-owner-live-20261102-02/`는 출력 157개 후 영상 render 하한 150ms 초과로 중단. worker 결과가 실제 stale 오류·수신 순번 44·native/event 해시를 상위에 전달했다. camera 69묶음 해시 검증, visual stopped 및 GPU 해제, 원본 worker failure와 cleanup errors 없음, 두 PID 소멸. AI/카메라 off·brake1·pause 해제. 조작·학습0.

마지막 capture 53: camera received 62565.8346209, owner selected 약 1.95ms 뒤, 부모 packet received 약 0.89ms 뒤, unpack 약 0.26ms. unpack 종료 62565.837719부터 body context decision 62565.9228038까지 약 85.08ms. 기존 기록은 이미지의 nonzero 검사와 원본 body-history drain 중 어디에서 지연됐는지 분리하지 못하며 원인 확정 불가. 157개 성공 출력의 body age 최대 9.343ms, 자식 native 인접 복사 최대 간격 3.061ms.

중복 camera status IPC를 하나로 줄였고 `scripts/isolated_camera_diagnostic.py`에 owner 선택·전달 계측을 더했다. 이후 `online_body_shadow.py`에 native drain 시작/완료와 live script에 영상 검사 시작/완료 시각을 더했으며, publication 실패 뒤 status를 다시 조회해 실제 이유를 덮어쓰는 문제를 수정했다. 관련 테스트 19개 통과. 다음 한 번의 정지 시험에서 두 구간을 분리해 실질 수정 대상을 결정한다. 현재 시각 신선도 기준 미통과, 실제 시각 생리·물리 동기·실제 주행·유효 랩은 미검증이다.

## 2026-11-02 몸 이력 지연 대조 및 8초 정지 host 검증

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 목표이고 전체는 하나의 감각운동 체계다.

추가 계측 `D:/CodexResearchRuns/flybrain-body-owner-live-20261102-03/`은 출력 161개 뒤 영상 stale로 중단됐다. 마지막 native 이력 drain wall 102.73ms, 검증 0.25ms, 처리한 신규 native 약 14개. 같은 구간 부모 native 복사 104.34ms 공백이었으나 독립 몸 소유자 복사 41개는 진행했다. `...-04/`도 158개 뒤 stale; 마지막 drain wall 88.20ms, Windows thread/process CPU 시간 각각 93.75ms였다. 이 계측만으로 지연 유일 원인은 확정하지 않는다. 두 기록·영상 원본·실패 유지.

부모의 자동 GC만 8초 이내 관찰 동안 중단하고 이전에 전체 수집·종료시 원래 상태 복원하는 대조를 구현했다. 몸/영상 50/150ms 기준은 변경하지 않았다. `...-05/`는 662개 출력, 189개 영상, 약 7.52초에서 카메라 200장 상한에 먼저 닿아 원본 상태 failed; GC 복원·전체 200개 원본 해시와 종료는 확인했다. 신규 `...-06/`은 카메라 상한 240장 및 기존 256MiB 예산, 8초 observation 완료, 201개 영상, priming1·출력710, 출력 span 7.98598초. camera 221묶음 해시 검증, 정상 stopped worker와 카메라, GPU 반환, 두 PID 소멸, GC 원복, 조작·학습0. `independent-host-audit.json`: current/held body 최대 11.8585/24.0338ms, view 최대 90.849ms, body 원본 hash·first receipt·scene 링크 오류 0; camera 동기 물리 시각·회로 full output 재생·생리·실제 주행이나 랩타임은 검증하지 않았다. GC 중단에 따른 개선은 이 조건의 관찰이며 인과 원인 확정은 아니다.

`bounded_gc_suspension.py` 복원/실패, 기존 body·ticker·camera 테스트 16개 통과. 다음은 06 원본 영상과 각각의 native/게시 시각으로 전체 회로 출력을 재생하고 독립 일치를 확인해 1단계의 남은 통과 조건을 검증한다. 그 뒤 실제 자료의 TRAIN 자격, 좌우·가감속·몸 피드백 인과 검증과 사용자 목표인 변속·셋업·유효 완주/라인/랩타임 단계로 이어간다.

## 2026-11-02 정지 시각·몸·단일 회로 전체 원본 재생

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응을 포함한 하나의 감각운동 체계다.

`scripts/reconstruct_body_owner_live_visual.py`는 `D:/CodexResearchRuns/flybrain-body-owner-live-20261102-06/`의 원본 ZIP·manifest와 선택된 각 BMP의 해시를 검증하고 선택된 전후좌우 영상 201개를 당시 visual snapshot 간격·동일 GPU FlyVis checkpoint·held-input state로 재구성했다. `D:/CodexResearchRuns/flybrain-body-owner-visual-reconstruction-20261102-01/`에서 201개 retina/T4/T5 결과와 SHA를 남겼으며 GPU 자원 반환·프로세스 exit 0 확인. 이 결과만으로 회로 출력이 맞다고 판단하지 않았다.

`scripts/replay_body_owner_recorded_output.py`는 재구성된 영상 feature와 실제 body-owner의 최초 native packet/receipt, 원래 각 시야 몸 관측·게시/결정 시각을 동일 단일 회로에서 순차 재생했다. `D:/CodexResearchRuns/flybrain-body-owner-full-replay-20261102-01/result.json`: 결정 711, 준비 1, 출력 710, 실제 기록 intent와 최대 절대차 1.862645e-7; 모든 출력 배열 및 해시 보존. `independent-arithmetic-audit.json`은 배열별 오차를 재계산한 별도 산술 감사이며 두 번째 독립 모델 실행은 아니다. 재생 guard 시계는 기록된 결정 시각에 고정한 이상적 zero-compute 시계이고 실제 시간 제한은 원본 host audit에서 별도 검증했다.

따라서 1단계의 **8초 정지 host 감각 연결과 원본 수치 재생**은 확인됐다. 물리 노출/샘플 시각 동기, 움직이는 장면 및 학습 자료 자격, 생리 반응·인과 기여, 조향·가감속·변속/셋업 적응과 실제 주행/유효 랩/레이싱 라인·랩타임은 확인되지 않았다. 현 자료 `training_admissible=false`, optimizer/vehicle control 0. 다음은 사전 분리된 움직이는 데이터 수집 조건과 교사 조작·시각/몸 시간·누락·권한을 검증해 TRAIN 자격 여부를 판정하고, 기존 좌우 실패의 최소 사전 지정 대조 및 시각/몸 개입 검증으로 2단계를 진행한다.

## 2026-11-02 움직이는 후보 자격 및 봉인 경계

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응을 포함하는 하나의 감각운동 회로다.

`D:/CodexResearchRuns/flybrain-moving-fit-qualification-20261102-01/result.json`은 기존 `20261031-pace-train-native-copy-candidate-00` 728결정과 preassigned development 729결정의 후보·기록·raw/authority/native 감사 SHA와 일치 여부를 확인한다. 양쪽 원본 역할·전체 이력은 보존되고 raw actor array 오류 0, same-copy AI true 각각 728/729다. 하지만 모두 원래 fit_allowed=false 및 recording training_admissible=false, raw continuous_40ms_training_admissible=false이고 원본 감사의 expert codec 검증과 labels가 없으며 물리 camera/body 샘플 시각 동기·독점 교사 권한은 입증되지 않았다. raw 검증 성공이나 AI true를 학습 가능·최적 교사로 바꾸지 않는다. DEV는 이미 본 개발 자료이고 별도 TEST가 아니다.

확인 중 `scripts/seal_ac_engine_ai_receipt_dataset.py`와 `scripts/seal_ac_moving_handoff.py`가 원본의 학습 불허를 확인하지 않고 결과 dataset을 training_admissible=true로 쓰는 경계가 발견됐다. `training_source_guard.py`를 두 경로에 연결해 recording·raw 연속 시간 감사·label 감사 각각 명시적으로 true일 때만 봉인을 진행하도록 수정했다. 이 검사는 부재·false·정수 1도 거부한다. 관련 단위/기존 aid 테스트 13개 통과; 3개 true 표시는 필요 조건일 뿐 증거의 신뢰성·경로 분리·생리·최종 모델 검증을 대체하지 않는다. 기존 datasets/run/실패 기록은 바꾸지 않았다.

다음은 기존 학습 불허 자료를 진단에만 두고, 원본 영상 및 몸/조작의 각 발생시각 경계와 검증 가능한 교사 authority·단위/표적 provenance를 확보하는 새 수집/판정 계약을 구현한다. 새 TRAIN/DEV/TEST 지리와 조건은 시작 전에 확정하고 기존 개발 자료를 TEST로 재지정하지 않는다. 수집 후에도 검증 전 fit 금지다. 자료 부족을 이유로 합성 모델을 생물학적 회로 검증이나 실제 주행 성공으로 대체하지 않는다.

## 2026-11-02 물리 시각의 불확실성 계약

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응을 포함하고 전체는 하나의 감각운동 체계다.

현재 CSP `render_started_clock`은 렌더 호출 원본 시계이고, native repeated-copy 시간은 호스트가 동일 바이트를 읽은 시간이다. 둘 다 그 자체로 물리 노출이나 physics source 발생 시각이 아니다. `car.timestamp`는 이전 진단 후보에서 관측했으나 그래픽 상태 캐시·단위·원자성/노출 관계가 완전히 검증되지 않았다. 이를 정확한 물리 시간으로 승격하지 않는다.

`src/flybrain_drive/physical_timing_intervals.py`는 **외부에서 별도로 입증된** 공통 host QPC의 닫힌 시간 범위만 입력받아, 결정 시각 대비 네 view 최대 150ms/몸 최대 50ms, 미래 교사 적용 40ms±25ms의 최악 경계를 계산한다. 4개 시야·몸·행동 중 하나라도 물리 시간 범위가 없으면 unresolved, 가능한 미래 센서나 임계 초과는 rejected다. 증거 정당성·권한·학습 허용을 판정하지 않으며 training_admissible은 언제나 false다. 임계값은 현재 코드의 예시적 predeclared task 기준이지 물리 상수나 보편적인 안전 보증이 아니다.

`scripts/audit_existing_physical_timing_bounds.py`는 원본 행사 SHA와 authority 감사·fit 금지를 확인하고 기존 TRAIN 728/known development 729를 입력했다. `D:/CodexResearchRuns/flybrain-physical-timing-interval-review-20261102-01/`에서 전부 물리 노출·몸 샘플·행동 적용 경계 부재로 unresolved이며, 호스트 수신시각을 대체해 추정하지 않았다. 미확인은 실제 시간 차이가 크다는 증거도 아니다. 범위 계산·누락·미래 센서·서로 다른 clock·기존 clock/원본 guard 관련 29개 검사 통과.

다음은 producer 이벤트의 timestamp 의미와 영상 exposure·native 몸/명령 적용 시간 사이를 독립적으로 교차 검증할 수 있는 계측을 설계·실제 확인하고, 유효하면 그 근거를 원래 source hash·episode/role·제어 권한과 연결한다. `car.timestamp` 또는 host receipt 하나만으로 위 physical bounds를 채우지 않는다. 근거가 없으면 원본·mask를 유지한 진단에만 사용한다. 새로운 TRAIN/DEV/TEST 수집은 사전 역할·조건을 봉인하고 실제 수집 후 별도 자격 검사한다. 학습·차량 조작은 이번에 하지 않았다.

## 2026-11-02 단일 비행 회로의 상대 바람 경로

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응이 포함되며 별도 운동 뇌가 아니다.

기존 `relative_airflow_sensor.py`는 검증된 바람/차량 속도를 몸 축 상대풍으로 투영하고 미검증·오래된 입력은 결측 마스크로 나타냈지만 `UnifiedFlightSensorimotor`에 연결되지 않았다. 새 `flight_airflow_feedback_adapter.py`는 front/left/up 상대풍을 30m/s 공학 정규화·0 초기화 투영으로 같은 회로의 기존 body port에 더한다. 사용자 선택형이며 측정된 초파리 수용기 배선이나 검증된 비행 생리 모델이라는 뜻이 아니다. 활성화 시 동일 decision의 명시적 `AirflowObservation`이 필요하고 미검증 입력을 valid로 만들지 않는다. `experimental_airflow_sensory_decision`과 training variant는 동일 재귀 상태를 사용하며 기존 미선택 경로/출력 포트/그래프를 바꾸지 않는다. 실측 바람 source는 아직 미자격이며 현재 live actor에 켜지 않았다.

`scripts/replay_unified_airflow_intervention.py` / `D:/CodexResearchRuns/flybrain-unified-airflow-intervention-20261102-01/`: 해시 확인된 정지 영상/몸 711결정(priming1·출력710)을 원래 순서로 재생했다. 결측 바람의 출력과 이전 원본 차 최대 1.862645e-7. 별도의 동일 초기상태 조건으로 ±10m/s 가상 상대풍·모든 기존 몸 port에 임의 고정 가중치 0.1을 적용했을 때 원본 대비 조향/전진 평균 절대차는 양방향 각각 [0.0005643,0.0004715]/[0.0005541,0.0004657]였다. 숫자는 해당 인공 개입의 연결 영향이며 생리 반응, 유의미한 운전 개선, 실제 바람의 사용 또는 가중치 학습을 뜻하지 않는다. 원본은 fit 불가, optimizer/차량 조작0; 산술 독립 감사는 저장 배열 재계산이며 두 번째 모델 실행이 아니다. 동일 recurrent state 경로·결측/오래된 source·합성 기울기/기존 guard 포함 34개 테스트 통과.

핵심 `unified_flight_sensorimotor.py` 소스가 변경됐으므로 이전 8초 실제 host 및 기존 replay는 당시 버전 증거로 보존한다. 새 버전의 바람 비활성 baseline 동등성은 위 710출력에서 확인했으나 live source/current worker proof를 새로 검증해야 한다. 다음은 바람 source의 단위·방향·차량 위치·샘플 시각을 독립 검증하고 영상/몸/바람 인과 개입과 학습 가능한 합성 조건·미사용 평가를 분리한다. 실제 source가 검증되지 않은 동안은 결측 상태로 유지한다. 가감속·변속·셋업 및 실제 유효랩/라인·랩타임 목표는 미완이다.

## 2026-11-02 바람 경계 추가 후 현 소스 기본 경로 검증

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 목표이며 별도의 운동 뇌는 없다.

바람 경계 추가로 `unified_flight_sensorimotor.py`가 바뀌었지만 이전 live gate가 worker 소스만 비교하고 단일 회로 소스는 확인하지 않던 불일치를 고쳤다. 현 unified/airflow 소스 SHA를 process parity와 read-only proof에 기록하고 live gate에서 두 해시를 확인한다. 기존 원본 evidence는 수정하지 않았다. `D:/CodexResearchRuns/flybrain-unified-process-parity-20261102-05/`는 바람 비활성, 과거 움직임 진단 원본 365결정의 process/부모/기록 상태·출력 최대 차 0, 정상 종료. `D:/CodexResearchRuns/flybrain-body-circuit-readonly-20261102-05/`는 합성 영상·실제 정지 몸, priming1/출력14·원본 80 copy, 정상 종료다.

실제 정지 `D:/CodexResearchRuns/flybrain-body-owner-live-20261102-07/`: 카메라 선택 201, priming1/출력709, output span 7.97366초, current/held body 최대 나이 11.752/23.013ms, view render 하한 최대 89.450ms. 독립 host audit은 원본 hash/first receipt/scene link 오류0. 카메라 222묶음 archive 검증·camera 및 body/visual 정상 종료·두 PID 소멸·GC 원복, optimizer/actuator0. 이것은 바람을 끈 8초 정지 host 시험이다.

원본 BMP 201개를 같은 FlyVis와 당시 snapshot 간격으로 재구성한 `D:/CodexResearchRuns/flybrain-body-owner-visual-reconstruction-20261102-02/`과 원본 몸/시야별 이력·결정시각을 사용하는 `D:/CodexResearchRuns/flybrain-body-owner-full-replay-20261102-02/`는 priming1/출력709 전부에서 원래 intent 최대 절대차 1.862645e-7. 산술 감사는 저장 배열 재계산이지 두 번째 모델 실행이 아니다. 이전 소스 06 결과와 새 07은 각각 고유한 독립 기록이며 학습/실제 운전으로 승격하지 않는다.

다음은 실제 바람의 단위·축·차량 위치·원본 시각 자격 및 새로운 분리된 움직임 자료 수집 계약을 검증하고, 생물학적 시각/비행 경로의 기능·인과 기여를 평가한다. 실제 바람 사용·물리 exposure 동기·실차/회로 운전·유효랩·레이싱 라인/랩타임은 미검증이다.

## 2026-11-02 단일 회로 합성 상대풍 반응 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 전체 목표이며 별도 운동 뇌가 아니다.

이전 실제 엔진 바람 기록은 반대 풍향에서 후보 상대풍 부호 차이를 보였으나 물리 시각·풍원·독립 기준이 미검증이므로 fit하지 않았다. `D:/CodexResearchRuns/flybrain-synthetic-airflow-plan-20261102-01/`에 합성 fit seed 879701/879702, 평가 seed 879715/879716, 좌우 코너·가상 ±5m/s 측풍, target delta `-0.001*wind` 및 power 유지, 8 Adam 업데이트, 성공 조건을 미리 봉인했다. 이 부호와 수치는 검증되지 않은 공학적 기능 시험 목표이며 생리/실제 조향 정답이 아니다. 각 seed와 생성기는 과거 프로젝트에서 이미 사용됐기 때문에 학습 입력으로부터 분리된 합성 평가이지만 전역 미노출 독립 holdout이나 실환경이 아니다.

`D:/CodexResearchRuns/flybrain-synthetic-airflow-fit-20261102-01/`은 기존 50ms held receipt 경계의 부동소수점 초과로 optimizer0 실패, `...-02/`는 inference-only Numba CSR에서 역전파 거부로 optimizer0 실패했다. 두 기록을 보존했다. 합성 receipt에 여유를 두고 같은 그래프의 미분 가능한 cached CSR을 사용한 `...-03/`은 단일 회로·기존 visual/body/graph/readout을 고정하고 새 airflow projection만 8회 Adam 학습했다. 마지막 다섯 결정에 걸쳐 gradient를 유지하고 앞선 상태는 수치 유지한 채 detach했다. 실제 data fit/actuator/UI0, actor 미자격. 학습 loss는 약 1.0→0.0810이며 일부 중간 상승이 있었다.

학습에 포함되지 않은 두 합성 seed에서 baseline MSE 2.5e-5 대비 각각 4.965e-7/4.967e-7, -5/+5 측풍 조향 변화 각각 약 +0.005706/-0.005703, 전진 출력의 평균 절대 변화 약 1.62e-5로 사전 기준을 통과했다. `synthetic-airflow-feedback.pt` SHA `398af23716c59fcbd0ed18e2d6ccf1f51605c85607293cbfd10b62b7b142b338`, actor_qualified=false·source_verified=false·synthetic_only=true. 독립 산술 감사는 저장된 분리된 seed/배열을 재계산한 것으로 별도 모델 재실행이 아니다. 실제 바람·생리·주행이나 안전성을 보여 주지 않고 이 checkpoint를 live actor에 적용하지 않았다.

다음은 위 가상 부호와 달리 실제 바람 원천의 단위/부호/시간을 검증하고, 새로운 바람·곡률·속도 조건 및 시각/몸/바람 제거 대조를 독립적으로 설계한다. 자격이 없는 기존 실제 데이터는 평가/진단만 하며 학습에 승격하지 않는다. 가감속·변속·셋업 적응 및 실제 유효 랩·라인/랩타임은 계속 미완이다.

## 2026-11-02 합성 학습 바람 경계의 기록 후보 반사실

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응을 포함하는 하나의 감각운동 체계다.

`D:/CodexResearchRuns/flybrain-airflow-recorded-fixture-plan-20261102-01/`에 checkpoint와 세 wind 원본 및 개입 방식을 먼저 고정했다. 실제 과거 엔진 진단 90°·270°·무풍 각 run의 result, metadata 및 프레임 hash를 확인하고 각 이동 중인 바람−차량속도 후보의 각자 차체축 투영 35개를 추출했다. 풍원/부호/물리 시각은 검증되지 않았고 세 자료의 시각과 실제 정지 07 시각·몸 이력은 동시 주행이 아니다. 35개를 별도 정지 이력의 상대 진행률에 따라 반복하는 수치 fixture이며, 계산 내부의 source_verified=true 마스크는 인위적 개입을 허용하기 위한 값이지 실제 원천을 승인하는 표시가 아니다.

`D:/CodexResearchRuns/flybrain-trained-airflow-recorded-fixture-20261102-01/`에서 합성 학습 checkpoint를 고정한 동일 비행 graph, 07 네 시야/몸/시각과 각 조건별 독립 recurrent state로 결측·무풍·양/음풍을 순차 재생했다. 결측 baseline은 기존 709출력 대비 최대 차 1.862645e-7. 무풍 대비 양풍 평균 조향/전진 변화 -0.0126213/+0.00003479, 음풍 +0.0138309/-0.00003845. 최대 절대 변화는 각각 [0.0133929,0.00003764]/[0.0142932,0.00003982]. 보존 배열의 별도 산술 감사는 두 번째 모델 실행이 아니다. 이 결과는 고정 모델의 수치적 민감도·상대 부호만 보여 주며 실제 풍속의 물리성, 최적 조향·생리·안정성 또는 주행 개선을 증명하지 않는다. source_verified=false, physical_sync=false, training_admissible=false, optimizer/차량 조작0.

다음은 동일 실행에서 발생한 영상·몸·바람의 원본 시각/축/단위를 독립 검증할 수 있는 관측과 서로 분리된 이동 조건을 확보하고, 영상·몸·바람을 각각 제거하는 단일 상태의 대조 및 실제 폐루프 기여를 평가한다. 기존 후보는 학습에 소급 승인하지 않는다. 변속·셋업 적응과 빠른 유효랩·라인/랩타임 목표도 미완이다.

## 2026-11-02 합성 측풍 폐루프의 실제 궤적 기여

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 포함하며 별도 운동 뇌가 아니다.

기존 `synthetic_planar_vehicle.py` slip-free bicycle에는 바람 외력이 없다. `synthetic_crosswind_vehicle.py`에서 기존 차량의 yaw/x/속도 업데이트를 보존하고, 명시적 가상 world-left 바람에 비례하는 횡위치 drift `0.02*wind*dt`를 더했다. 이는 공학적 부호/민감도 실험이지 AC 공력·횡력·타이어·관성 모델이 아니다. zero-wind 기존 모델 정확 일치 및 반대 바람 대칭·범위 거부, unified/airflow 관련 25개 테스트 통과.

`D:/CodexResearchRuns/flybrain-crosswind-loop-plan-20261102-01/`은 평가 전에 현재 소스·학습 checkpoint SHA, synthetic texture seed 882901/882902, 직선 road, 3m/s 고정, 81×40ms, -5/0/+5m/s wind, missing/trained 12조건, 각 ±wind·seed 모두 final 절대 횡오차가 줄고 zero 효과 ≤0.05m라는 기준을 봉인했다. 도로 geometry는 actor에 주지 않고 렌더된 전후좌우 영상·몸 감각을 통해서만 단일 graph에 전달한다. 실제 바람 source는 미검증이므로 각 trained 풍속은 synthetic fixture다. power를 제어하지 않고 속도를 고정한 조향 한정 실험이다.

`D:/CodexResearchRuns/flybrain-synthetic-crosswind-loop-20261102-01/`은 12조건 전부 81단계 완료, 사전 기준 통과. -5 조건에서 결측/학습 final |y|는 seed별 0.35994/0.32887m, 0.35997/0.32919m, +5는 0.29392/0.26028m, 0.29401/0.26044m다. 무풍 차이는 약 1.02e-8/4.78e-8m. 저장된 전체 궤적과 hash·산술 독립 감사는 보존했으며 산술 감사는 두 번째 모델 실행이 아니다. 이 조건에서 약 3cm 개선됐으나 새로운 비행 생리, 자동차 실제 공력, 코너·가감속·속도/타이어/셋업 일반화나 실주행/유효랩/랩타임을 증명하지 않는다. 결과는 actor_qualified=false, optimizer/실제 차량 제어0.

다음은 더 다양한 바람·곡률·속도와 타이어/하중 및 감각 가림에 대한 사전 분리된 대조, 실제 바람의 단위·부호·시간/source와 움직이는 실제 자료의 학습 자격 검증이다. synthetic checkpoint를 실제 actor에 적용하지 않는다.

## 2026-11-02 합성 코너·가감속·측풍 폐루프: 사전 기준 실패

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 바람·변속·셋업 적응도 목표이며 별도 운동 뇌가 아니다.

직선 고정속도에서 약 3cm 개선됐던 합성 airflow checkpoint의 범위를 확인하기 위해 `unified_airflow_speed.py`에서 기존 같은 `UnifiedFlightSensorimotor`의 직전 muscle state 기반 속도 경계와 선택형 바람 감각 경계를 한 번의 동일 재귀 상태 업데이트로 결합했다. 별도 상태/actor가 없다. 바람 결측 시 기존 속도 경로의 출력과 상태가 정확히 같다는 테스트 및 unified/crosswind 관련 20개 테스트 통과. 이전 core source는 변경하지 않았다.

`D:/CodexResearchRuns/flybrain-corner-wind-speed-plan-20261102-01/`에 기존 graph·synthetic airflow checkpoint 및 실행 소스 SHA, 새 synthetic 질감 seed 883201, 좌우 곡률 ±0.018, 초기 3/6m/s, 양쪽 ±5m/s 측풍, 결측/학습 16조건과 x=16m까지 비교를 사전 봉인했다. 모델 출력을 기존 공학적 축 매핑을 통해 가속·제동 및 조향에 사용하되 센서는 렌더된 네 시야와 허용된 synthetic body/airflow만 받는다. 속도와 궤적은 간이 slip-free bicycle, 횡풍은 0.02*wind*dt의 가상 횡위치 drift이며 AC 공력/타이어나 실제 가속 응답이 아니다. 목표 x=16에서 보간한 도로 중심선은 사후 평가 전용이다.

`D:/CodexResearchRuns/flybrain-corner-wind-speed-evaluation-20261102-01/`은 16조건 모두 목표 x에 도달했지만, 학습 경계가 같은 조건의 결측 대비 |오차|를 낮춘 것은 8쌍 중 5쌍이라 사전 기준에 실패했다. 악화 3개: 초기3·좌코너·+풍 0.02090→0.04286m, 초기6·우코너·-풍 0.09058→0.10968m, 초기6·좌코너·+풍 0.15245→0.17149m. 값은 합성 절대 중심선 오차이고 최적 레이싱 라인이나 actual lap이 아니다. 16개 원본 궤적 hash, x=16 보간과 판정 산술을 별도 감사했으며 두 번째 모델 실행은 아니다. synthetic-only, 기존 checkpoint 비배포·optimizer/실제 차량 제어0. 이전 직선 성공 범위와 충돌시키지 않고 그대로 둔다.

결론: 풍향만으로 고정 보정을 학습한 현 가상 바람 경계는 코너/속도 조건에서 일관된 개선을 보이지 않으므로 실제 actor에 적용하지 않는다. 다음은 코너 편향·측풍 외란과 시각/몸 상태의 상호작용을 하나의 회로에서 사전 지정 대조로 분리하고, 학습 가능한 자료 및 실제 source·시간을 검증한다. 조향·가감속·변속·셋업 적응과 실제 유효완주/라인/랩타임은 계속 미완이다.

## 2026-11-02 코너 오차 분해와 공간 시각 경로 제거

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

- 기존 좌우 코너·3/6m/s·양방향 측풍의 중심선 오차 개선 기준은 5/8로 실패한 상태를 유지한다. 기존 결과를 보고 추가한 무풍 비교에서 같은 모드의 무풍 궤적 대비 측풍 변화량은 8/8 감소했다. 이는 사후 탐색이며 원래 기준을 대체하지 않는다. 무풍 절대 중심선 오차는 0.227–0.293m다. `D:/CodexResearchRuns/flybrain-corner-wind-reference-20261102-01/`의 산술 감사는 독립 시뮬레이션이 아니다.
- 사전 계획된 합성 무풍 4조건에서 학습된 공간 시각 경계만 제거했다. 직접 및 시간 시각 경로는 유지했다. 네 조건 모두 중심선 절대 오차가 0.0250–0.0367m 증가했다. 계획의 2cm 진단 한계는 넘지만 큰 기저 코너 오차가 남는다. 원본·사전 계획·저장 궤적 산술 감사는 `D:/CodexResearchRuns/flybrain-corner-spatial-lesion-20261102-01/`, `D:/CodexResearchRuns/flybrain-corner-spatial-lesion-plan-20261102-01/`.
- 위는 간이 합성 차량과 도로에 한정되며 생리 타당성·실제 물리 시간 동기·AC 운전·유효랩·actor 자격을 증명하지 않는다. optimizer 업데이트·차량 제어·배포는 없었다. 정답 중심선/위치는 평가용이다. 다음에는 시각 선행 정보가 코너 전에 회로 입력과 출력에 표현되는지, 직접·시간·공간 경로별 반응을 검증한다.

추가 사후 궤적 보간: x=10m에서 좌우 코너의 조향 차이는 공간 경계 사용 0.00629/0.00476rad, 제거 -0.000075/+0.000006rad (초기 3/6m/s)였다. 폐루프 상태도 달라 시각 선행 예측을 인과적으로 입증하지 않는다. 근거 `D:/CodexResearchRuns/flybrain-corner-anticipation-trajectory-audit-20261102-01/result.json`. 다음은 동일한 몸·회로 상태에서 코너 영상만 바꾸어 비교하는 통제된 반사실 실험이다.

## 2026-11-02 동일 몸 이력에서 코너 영상 개입

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 음의 코너 무풍 궤적을 고정해 동일한 몸 감각·회로 이력으로 재생하고 마지막 한 프레임 또는 다섯 프레임의 합성 코너만 -1/0/+1로 바꿨다. 비교는 기존 결과를 본 이후 계획한 탐색으로 사전 성능 합격 기준이 없다. 원본 이력은 평가에만 쓰며 actor 입력으로 지도·절대 좌표를 제공하지 않는다. 시각 프레임 생성에 사용하는 환경 좌표와 actor 입력을 구별한다.

한 프레임 교체 직후 조향 차이는 최대 2.2e-8rad로 거의 없었다 (`D:/CodexResearchRuns/flybrain-fixed-state-frame-swap-20261102-01/`). 다섯 프레임 교체에서 좌우 조향 차이는 공간 경로 사용 시 초기 속도 3/6m/s에서 각각 0.001779/0.001704rad, 제거 시 0.00000144/0.00000067rad였다. 좌우 최종 망막 영상 최대 차이는 0.334이고 교체 전 출력 차이는 최대 9.1e-8이다. 원본 배열 재계산은 `D:/CodexResearchRuns/flybrain-fixed-history-window-swap-20261102-01/arithmetic-audit.json`, 사전 계획은 `D:/CodexResearchRuns/flybrain-fixed-history-window-swap-plan-20261102-01/plan.json`. 수치 재계산은 독립 모델 실행이 아니다.

공간 경로와 시간 상태를 거친 합성 영상 변화가 동일 몸 이력의 조향에 인과적으로 작용한다는 제한된 증거다. 실제 시각·생리·주행·유효랩 검증이나 모델 배포가 아니며 원래 코너 중심 오차 평가 실패는 유지한다. optimizer/차량 제어/배포 0. 다음은 실제 영상의 물리 시각·몸 감각 정합 자격을 확보하면서 시간 시각 경로 기여와 실제 영상 반응을 검증한다.

## 2026-11-02 후보 물리 시계 상대 경과시간 확인

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

과거 원본 영상의 메타데이터 해시 250개를 확인하고 후보 차량 물리 timestamp와 진단 CSP 완료 시계의 상대 경과시간을 첫 프레임에 맞춰 비교했다. 250프레임 중 78은 후보 속도 1m/s 초과다. 상대 차 범위 -18.91~+29.81ms, 프레임별 증가량 차 범위 -44.95~+27.61ms다 (`D:/CodexResearchRuns/flybrain-candidate-clock-drift-audit-20261102-01/`). 기준 프레임과 읽기 구간의 영향이 섞였으며 이 결과만으로 실제 drift, 물리 노출 시간, 두 시계 절대 오프셋 또는 동기 오차를 규명할 수 없다. 과거 source는 진단 전용, actor/학습 불허 상태다.

카메라 `shot:update` 호출 전후의 시각·동일 그래픽 프레임/캐시 car timestamp만으로 픽셀이 어느 물리 tick을 표현하는지 증명할 수 없다. 다음에는 사용 가능한 CSP 렌더/물리 소스의 이벤트 의미와 공통 시계 변환·보수적 불확실도를 실험적으로 확인해야 한다. 원본 결과는 유지하며 새 수집·optimizer·차량 제어는 없다.

## 2026-11-02 CSP 시간 의미와 비배포 계측 후보

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

설치된 CSP `ac_apps/lib.lua`의 API 문서는 `car.timestamp`가 마지막 물리 상태 기록의 밀리초 시각이며 `sim.time`과 원점이 같다고 설명한다. `sim.time`은 AC 시작 이후 밀리초다. 이는 문서에 명시된 의미이며, 실측 데이터와 화면이 같은 물리 상태라는 증거는 아니다. 과거 원점/시각 의미 미검증 표기는 당시 지식이다.

기존 프레임 시계 진단 후보를 보존하고 `integrations/ac_csp/candidates/FlyBrainCamera-sim-clock-observation-v1.lua`에 렌더 호출 전후 `sim.time`/`car.timestamp` 및 CSP 읽기 시작·끝 시각을 추가했다. 차이는 삽입 22줄뿐이다. `src/flybrain_drive/diagnostic_sim_clock_pairs.py`는 네 시야와 읽기 순서를 검사하고 값 차이를 보고하되 물리 노출 동기·학습을 허용하지 않는다. 3개 검사가 통과했다. 해시와 범위 `D:/CodexResearchRuns/flybrain-sim-clock-candidate-inspection-20261102-01/result.json`. 실제 CSP에 설치하거나 실행하지 않았고 Lua 런타임 문법 검사도 하지 않았다.

다음은 사용자 선호에 맞는 CSP lifecycle과 비조작 경로에서 후보 설치·복구/충돌 안전 조건과 Lua 실행을 확인한 뒤, 별도 정지 진단으로 시계쌍을 모으는 것이다. 그 결과만으로 노출 동기 또는 학습 자격이 부여되지는 않는다. 실주행/유효랩 및 생리 검증은 미완료다.

## 2026-11-02 정지 CSP 시계쌍 실측 및 설치 원복

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

실제 설치본은 기존 보관 MMF 소유 재개 기능을 가진 `FlyBrainCamera-owned-mmf-resume-v4.lua`, 해시 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`이었다. 앞서 만든 후보의 기반과 달라 그 후보를 직접 설치하지 않았다. 설치본을 기반으로 시계쌍만 추가한 `integrations/ac_csp/candidates/FlyBrainCamera-sim-clock-owned-v2.lua`를 Lua 문법 로드 검사하고 별도 candidate-pinned 정지 probe를 준비했다. 문법 검사는 Lua 5.5 구문 검사이며 CSP 실행의 대체 증거가 아니다.

CSP lifecycle API로 차량 정지·브레이크 유지와 유휴 카메라를 확인하고 pause 후 후보를 백업/해시 확인해 임시 설치했고 새 generation을 확인했다. 배경 엔진에서 영상 4그룹×4시야를 수집하고 원본·사본 해시를 검증했다. 각 시야 `car.timestamp - sim.time` 렌더 전 값은 16개 모두 0ms, 읽기 구간 최대 0.0028ms, 역행 0이다. 원본 `D:/CodexResearchRuns/flybrain-simclock-stationary-probe-20261102-01/`; 분석 `D:/CodexResearchRuns/flybrain-simclock-stationary-analysis-20261102-01/`.

후보를 제거하고 정확한 기존 해시의 설치본과 이전 unpaused 상태로 복구했다 (`D:/CodexResearchRuns/flybrain-simclock-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-simclock-resume-20261102-01/`). 후속 heartbeat에서 camera_active=false, AI=false, paused=false 확인. lifecycle pause/resume 각 1, UI·actuator·optimizer·actor 0. 정지에서 시계값이 같다는 것은 움직임이나 렌더 GPU 노출/몸 샘플 동기가 아니다. 다음은 움직이는 시점의 추가 표식과 물리 상태/픽셀 관계 및 호스트 변환의 독립 증거를 설계한다.

## 2026-11-02 움직이는 진단 CSP 시계쌍

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 설치본 기반 계측 후보와 소유 카메라 + CSP expert-motion-probe를 연결했다. 첫 시도 `D:/CodexResearchRuns/flybrain-simclock-motion-probe-20261102-01/`은 CSP가 `Expired, changed, active or nonstationary session`으로 거부했고 AI 출발 전이다. 카메라 deadline 잔여 부족이 가능하지만 거부 사유의 어느 조건인지 단정할 증거는 없다. 실패와 원본 98프레임을 보존했다. 최초 설치본과 이전 세션 상태를 원복한 후 유한 deadline 30초, 최대 300프레임으로 수정해 별도 계획을 봉인했다.

두 번째 `D:/CodexResearchRuns/flybrain-simclock-motion-probe-20261102-02/`는 3초 AI 주행 후 성공적으로 정지·브레이크 유지했고 최대 속도 16.5986km/h, 원본 네 시야 157묶음과 host 관찰 해시를 보관했다. 628개 시야의 전후 두 시계 차이는 모두 0ms, 그중 가장 가까운 AI telemetry가 ±50ms 이내이며 속도 >1km/h인 시야 152개도 같다. 읽기 구간 최대 0.036ms, 한 렌더 호출 내 시계 역행 0 (`D:/CodexResearchRuns/flybrain-simclock-motion-analysis-20261102-01/result.json`). telemetry 연관은 진단상 근접 매칭이며 동시 물리 시점 증명이 아니다.

CSP 시계값의 동등은 같은 캐시일 수 있고, 렌더된 픽셀이 어느 물리 시점을 담았는지, 호스트 몸 패킷이 같은 tick인지 증명하지 않는다. 학습과 student actor는 불허를 유지한다. 원래 카메라 해시 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`로 원복, lifecycle로 이전 unpaused 상태 복구 후 카메라 off·AI off·정지·브레이크 확인. UI 조작 0. 다음은 카메라 렌더 프레임과 몸 패킷의 공통 물리 tick에 대한 독립 검증이나 명시적인 불확실도 상한을 확보한다.

## 2026-11-02 실제 AI 진단 네 시야 FlyVis 재생

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 CSP AI 진단 영상 157그룹의 메타데이터·PNG를 원본 host 관찰 해시와 대조했다. 당시 dashboard 전방 투영 설정과 4시야 FlyVis를 사용해 CSP script frame 시간 간격으로 신경 상태 2,572 tick을 이어가며 최초 영상을 고정한 대조와 비교했다. 각 시야 운동 활동 최대 차이는 1.386. 전체 저장 활동 및 재계산 `D:/CodexResearchRuns/flybrain-real-motion-flyvis-replay-20261102-01/`, 사전 계획 `D:/CodexResearchRuns/flybrain-real-motion-flyvis-plan-20261102-01/plan.json`.

가장 가까운 CSP AI telemetry ±50ms를 이용한 탐색적 구분에서 속도 >1km/h 인접 38그룹의 전/좌/우/후 RMS 평균 0.190/0.135/0.140/0.141, <0.1km/h 인접 11그룹은 0.175/0.104/0.116/0.114였다. 집단은 시간·영상 내용이 달라 속도 효과를 독립적으로 분리하지 못하고 밝기/시야 변화도 포함한다. telemetry 근접은 물리 동기 증거가 아니다. 첫 화면 고정도 현실적인 정지 영상 대조가 아니다. 실제 움직임 영상이 회로 입력으로 전달돼 시간 경과에 따라 출력이 달라지는지에 관한 제한적 증거이며 FlyVis 생리·AC 자율주행 성능·학습 자격을 증명하지 않는다. 새 optimizer·actor·차량 제어는 없다. 다음은 실제 영상 변화와 몸 시각 관계의 정합을 입증할 독립 기준을 마련하고 통합 단일 회로의 동시 입력 반응을 평가한다.

## 2026-11-02 실제 영상과 합성 몸을 단일 회로에 재생

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 AI 움직임 진단의 해시 확인된 네 시야 FlyVis 이력과 첫 화면 유지 대조를 동일한 단일 비행 회로에 따로 연결했다. 몸 입력은 실제 동기된 관측이 아니라 요 0rad/s·속도 3m/s·바퀴 3/0.29rad/s로 고정된 합성 가정이며 나머지는 결측으로 명시했다. CSP 영상 시각의 간격이 회로 제한을 넘는 곳은 이전 시각 입력을 유지하며 최대 40ms 중간 결정을 넣었다. 독립된 두 episode에서 회로 상태는 결정 사이에 유지했다.

157개 영상 시각에서 실제 영상 경로와 첫 화면 고정 경로의 intent 최대 절대 차는 조향 0.0031568, 전진 0.00008467이다. 보존된 원본과 산술 검증 `D:/CodexResearchRuns/flybrain-real-motion-unified-20261102-03/`, 계획 `D:/CodexResearchRuns/flybrain-real-motion-unified-plan-20261102-03/`. 01은 필수 yaw 결측, 02는 원래 영상 간격이 80ms 회로 제한을 넘어서 중단됐다. 두 실패를 보존하고 각각 명시적 합성 yaw와 시간 구간 분할로 수정했다. 산술 검증은 독립 모델 재실행이 아니다.

이 차이는 실제 영상이 합성 몸 조건의 통합 회로 출력에 미치는 제한적 반사실이며 실제 몸-영상 동기, 초파리 생리, 실차량 조향 적합성이나 랩 성능 검증은 아니다. 원본은 학습 불허, optimizer·actor·차량 제어 0이다. 다음은 실제 몸 감각 데이터의 원본 수신 시간과 영상 상한을 보존한 연속 입력 경로를 구성하되, 물리 동기 미검증을 계속 명시하고 학습 자격을 임의로 부여하지 않는다.

## 2026-11-02 실제 영상·몸 호스트 수신 순서 통합 재생

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

원본 AI 진단에서 native 패킷 133개 전체의 바이트 해시와 카메라 메타데이터/host 해시 로그를 확인했다. 원본 FlyVis 157프레임 중 패킷 수신이 겹치는 103–155번 53프레임을 사후 선택했다. 각 영상 host 파일 읽기 시점과 40ms 이하 중간 결정에서 이미 수신된 최신 원본 패킷만 사용하고, 최초 수신 시각 기준 50ms를 넘으면 중단한다. 116결정/조건의 몸 최대 수신 경과 31.835ms. 원본 몸 값과 마스크, 동일 몸 이력을 사용해 실제 영상과 첫 프레임 고정의 두 통합 회로를 독립 reset 후 연속 재생했다.

157프레임 중 53프레임에서 조향 intent 최대 절대 차 0.0031424, 전진 0.00008404 (`D:/CodexResearchRuns/flybrain-real-motion-host-body-unified-20261102-01/`). 계획은 `D:/CodexResearchRuns/flybrain-real-motion-host-body-plan-20261102-01/`; 저장 배열 산술 검증은 독립 회로 실행이 아니다. 원본 영상과 패킷은 같은 진단에서 왔으나 파일 읽기와 패킷 수신은 물리 상태가 생성된 시각이 아니다. host 렌더 하한은 미측정이므로 0을 자명한 보수 하한으로 넣고 네 시야의 과거 몸 문맥은 모두 결측으로 처리했다. 이 숫자는 영상 신선도 통과·물리 동기·운전 효과를 뜻하지 않는다. 학습/actor/새 차량 제어 0; 몸 tilt/접지/타이어·상대풍의 유효성도 별도 점검이 남는다. 다음에는 동기 근거와 비행 회로의 실제 몸 피드백 인과 기여를 검증한다.

## 2026-11-02 실제 몸 yaw의 통합 회로 기여 대조

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

앞서 원본 영상·몸을 host 수신 순서로 재생한 53영상/116결정 조건에서 같은 영상과 모든 몸 값·마스크를 유지하고 유효 yaw 값만 반사실적으로 0으로 바꿨다. 원본 yaw는 -0.13954~+0.58833rad/s이며 최대 출력 차는 조향 intent 0.048109, 전진 intent 0.0003392다. 원본 yaw 조건은 직전 독립 실행에서 저장된 원본 조건의 출력과 정확히 일치했다 (최대 차 0). 결과·배열 산술 비교 `D:/CodexResearchRuns/flybrain-real-motion-yaw-counterfactual-20261102-01/`; 계획 `D:/CodexResearchRuns/flybrain-real-motion-yaw-counterfactual-plan-20261102-01/`.

이것은 해당 3초 진단의 단일 회로가 몸 회전 입력에 반응한다는 인과적 반사실이지 그 방향이 바른 균형 제어나 초파리 생리라는 증명은 아니다. 카메라/몸 물리 시각·감각 등록·차량 피드백 효과·빠른 유효랩은 미검증이다. host 파일 읽기 시점 이전 패킷 사용, host render 하한 미측정, 영상-몸 과거 문맥 결측; 학습/actor/새 차량 제어 0. 다음은 영상 조건 및 몸 부하/기울기 마스크를 확인하고 반응의 방향·시간 관계를 실제 차량 반응과 비교할 독립 조건을 갖춘다.

## 2026-11-02 실제 몸 기울기·하중 인과 대조와 누락 발견

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

동일 원본 영상·몸 이력 53프레임/116결정의 단일 회로에서 기울기만 0으로 만들거나 네 정규화 바퀴 하중을 0.25씩 균등화한 반사실을 별도 실행했다. 기울기 원본 범위 pitch -0.02071~0.00188rad, roll -0.00910~0.00421rad에서 0으로 바꾸면 조향·전진 intent 최대 차 0.00156695/0.00000545다. 하중 균등화에서는 회로 상태와 intent 차이가 모두 0이다 (`D:/CodexResearchRuns/flybrain-real-motion-tilt-load-20261102-01/`). 원본 조건은 직전 원본 이력의 저장 출력과 정확히 일치했다.

해당 실행의 `20261101-speed-wheel-turn-control-fit/speed-body-adapter.pt`에서 body index 8–11의 하중값 projection 열은 모두 정확히 0이다 (`arithmetic-and-weights-audit.json`). 회로 기본 몸 경계는 각속도 2–4만 처리하므로 이 checkpoint의 이 하중값 경로는 중립이다. 이는 하중 센서 원본이 없다는 뜻이 아니라, 관찰된 하중을 이 회로가 아직 활용하지 못한다는 구현상 결함이다. 기존 가중치가 0인 이유는 추가 확인이 필요하다. 설정 및 회로가 하중을 처리한다고 생리·기능 검증을 대체하지 않는다.

계획된 다음 작업은 native 바퀴 순서/좌우 부호의 출처와 결측 mask를 검증하고, 비행 유래 균형 회로에 대한 공학적 하중 대응 가설을 명시해 단일 회로 내부 몸 포트의 합성 반응을 학습·좌우 반전/결측/원본 baseline을 독립 검증하는 것이다. 실제 원본의 물리 시각 동기 미검증 상태에서 fit 자격을 임의로 주지 않는다. 실제 차량·랩·생리는 미검증이며 이번에는 학습/actor/차량 제어 0이다.

## 2026-11-02 합성 하중 경로 학습 실행

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

현재 adapter에서 하중 네 열만 훈련하며, 행별 합을 0으로 투영하여 균등 하중일 때 원본 기준을 유지하고 다른 열은 동결한다. 좌우 하중에서 조향 ±0.003, 전후에서 전진 ±0.002는 가설로 지정한 합성 반응 목표이며 파리 생리나 실제 최적 운전을 뜻하지 않는다. 기존 합성 장면 fit 및 분리된 합성 validation source 해시를 선봉인했다. 계획 `D:/CodexResearchRuns/flybrain-synthetic-load-plan-20261102-01/plan.json`.

`D:/CodexResearchRuns/flybrain-synthetic-load-fit-20261102-01/`은 확인 시 Python PID 22456에서 6회 중 1회 완료, 최종 checkpoint/검증 미완료였다. 종료와 결과를 다시 확인하기 전 성공으로 판정하지 않는다. 최종 checkpoint는 synthetic_only·actor 미자격이어야 하며, 균등·대칭·결측·실제 비학습 진단 반사실과 기존 baseline 보존을 확인해야 한다. 실제 영상·몸의 물리 시각, 비행 생리·AC 운전·유효랩은 미검증이다.

후속 확인: 실행 PID 22456의 progress.json에 6회 중 3회 완료, 해당 손실 0.4278이 기록됐다. 결과 checkpoint는 아직 없다. 후보 생성 후 `scripts/audit_synthetic_load_candidate_ports.py`로 원본 열 보존, 균등 하중 기준, 좌우·전후 반전 및 16개 하중 결측 조합의 포트 영향을 검사하도록 준비했다. 실제로 검사하기 전 통과로 판정하지 않는다.

합성 하중 학습 확인 당시 6회 중 4회 완료, 손실 0.2179. 별도로 `complete_wheel_load_feedback.py`에 네 바퀴가 모두 유효할 때만 하중값을 전달하는 실험용 경계를 추가했다. 유효성 표시는 유지하며 완전 유효 시 기존 출력과 일치한다. 16가지 마스크 조합과 다른 채널 보존 검사 2개 통과. 학습 checkpoint에 적용하거나 생리·실제 주행 검증을 한 것은 아니다.

## 2026-11-02 합성 하중 학습 및 결측 포트 검증 결과

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

6회 합성 학습 완료; 손실 0.5→0.589→0.428→0.218→0.333→0.369. 분리된 합성 seed 2개×좌우 코너×하중 방향 4개 총 16조건에서 좌우 하중 조향 평균 약 ±0.001529(임의 목표 ±0.003), 전후 하중 전진 평균 약 ±0.000313(목표 ±0.002). 저장 배열 산술 재계산의 정규화 MSE 0.238, 무반응 baseline 0.5; 사전에 합격 기준을 정하지 않았으므로 통과 판정이나 독립 현실 성능을 주장하지 않는다. `D:/CodexResearchRuns/flybrain-synthetic-load-fit-20261102-01/` checkpoint SHA `97f0f1831e7d5485d551da626b206d0c7a15761fe5f9aa98a486d0dea3161528`, synthetic_only·actor 미자격.

포트 감사에서 비하중 가중치 정확히 동일, 균등 하중 포트 차 최대 7.45e-9, 행별 하중 가중치 합 최대 1.49e-8. 부분 결측에서는 후보의 포트가 기존 기준과 최대 0.02839 차이나 안전하지 않다 (`D:/CodexResearchRuns/flybrain-synthetic-load-port-audit-20261102-01/`). 모든 하중이 유효할 때만 하중값을 전달하는 `CompleteWheelLoadFeedback` 적용 시 16가지 mask에서 완전 유효는 후보와, 부분/전체 결측은 이 checkpoint의 원본 기준과 최대 차 0 (`D:/CodexResearchRuns/flybrain-synthetic-load-gate-audit-20261102-01/`). 포트 단계의 시험이며 통합 회로 및 폐루프 자격을 뜻하지 않는다.

다음은 학습된 후보+결측 경계를 동일 통합 회로의 실제 진단 이력에 오프라인 연결해 기준 보존, 실제 하중에 대한 인과 차이, 결측, 기존 시각·yaw 반응을 검증한다. 실제 입력은 fit 금지다. 물리 노출 동기·생리·가감속/차량 주행 기여·유효랩은 미검증, actor/차량 제어 0이다.

## 2026-11-02 하중 후보와 결측 경계의 실제 진단 이력 대조

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

합성 학습된 하중 후보와 네 바퀴 완전성 경계를 기존 하나의 통합 회로에 임시 연결해 학습 불허 AI 진단의 53프레임/116결정을 재생했다. 6개 독립 reset 이력에서 동일 영상·몸 원본 및 균등 하중/한 바퀴 결측을 각각 baseline과 후보로 비교했다. 기존 baseline은 이전 저장 출력과 정확히 동일했다. 실제 하중일 때 후보 대 baseline 조향·전진 intent 최대 차 0.0006911/0.0006428, 상태 최대 차 0.0025298. 균등 하중은 출력 차 0, 상태 최대 차 2.24e-8; 한 바퀴 결측은 출력·상태 차 모두 0. 원본 배열 재계산 `D:/CodexResearchRuns/flybrain-load-real-history-20261102-01/`, 계획 `D:/CodexResearchRuns/flybrain-load-real-history-plan-20261102-01/`.

이는 해당 합성 후보의 하중 이력 기여와 한 가지 결측 대조를 같은 회로에서 확인한 제한적 증거다. 실제 이력에 fit하지 않았으며 호스트 수신 순서만 보증, 물리 영상/몸 동기·생리·차량 운동 기여·빠른 유효랩은 미검증이다. 원래 시각·yaw 반응이 후보에서 유지되는지도 별도 인과 대조가 필요하다. optimizer·actor·새 차량 제어 0. 다음은 실제 하중의 교란·몸 회전·시각과의 상호작용을 독립 조건에서 평가하고, 이후 합성 폐루프 개선 여부 및 결측을 검증한다.

## 2026-11-02 하중 후보 적용 시 기존 시각·yaw 경로 대조

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

동일 실제 AI 진단 영상 53프레임/몸 116결정에 합성 하중 후보와 결측 경계를 유지하고 각기 첫 영상 고정 및 유효 yaw=0 개입을 했다. 해당 후보가 적용된 원본 조건은 앞선 하중 대조에서 저장한 해시 확인 결과다. 시각 고정의 조향/전진 intent 최대 차 0.00314259/0.00008407, yaw 제거는 0.04811580/0.00033882. 후보 없는 이전 결과에 비해 각 개입 효과의 시계열 최대 차는 시각 조향/전진 1.034e-6/8.94e-8, yaw 6.892e-6/4.768e-7이다. `D:/CodexResearchRuns/flybrain-load-visual-yaw-20261102-01/`, 계획 `D:/CodexResearchRuns/flybrain-load-visual-yaw-plan-20261102-01/`. 저장 배열 비교는 독립 실행이나 생리 증명이 아니다.

이 기록에서 기존 시각·몸 회전 반응이 유지되는 증거이나, 하나의 사후 선택 기록과 임의 가설 학습일 뿐 일반화·물리 시간 동기·주행 개선은 검증하지 않았다. 원본은 fit 금지, optimizer·actor·차량 제어 0. 다음은 하중-차량 동역학이 명시된 합성 폐루프에서 baseline과 후보의 이탈·안정성·속도 변화를 사전 지정된 조건으로 비교하고, 실제 회로 관련 생리적 근거와 차량 대응 가설을 분리한다.

## 2026-11-02 합성 하중 이동 폐루프 결과

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

`src/flybrain_drive/synthetic_wheel_loads.py`는 이전 가감속과 자전거 모형의 횡가속으로 전후·좌우 하중 분율을 즉시 변화시키는 가설이다. 좌우 거울, 가감속 방향, 정규화, 잘못된 입력과 결측 게이트 관련 검사 4개 통과. 이는 완전한 하중·서스펜션·타이어 모델이나 실제 AC 물리가 아니다. 합성 좌우 코너×초기 3/6m/s×기존/후보/첫 바퀴 결측 12조건, seed 883301을 사전 계획 `D:/CodexResearchRuns/flybrain-synthetic-load-loop-plan-20261102-01/`에 지정했다.

모두 x=16m에 도달했다. 후보 적용 시 절대 중심 오차 변화는 3m/s 음/양 코너 -0.022/-0.002mm, 6m/s 음/양 코너 -0.065/+0.012mm다. 기저 오차 0.227–0.292m에 비해 변화는 극히 작다. 센서 하나 결측 시 기준 오차와 차이 최대 2.04e-8m다. 관찰된 하중 분율 범위 0.2327–0.2675; 범위와 실험 모델은 현실 보장이 없다. 저장 궤적 12개 보간 산술 검증 `D:/CodexResearchRuns/flybrain-synthetic-load-loop-20261102-01/`. 사전 지정 수치 합격 기준은 없으며, 유효한 코너 추종 개선을 주장하지 않는다.

하중 반응을 만드는 것만으로 코너/속도 최적화가 되지 않는다. 기존 시각 경로의 큰 코너 오차, 하중 대 목표 방향·크기 및 더 근거 있는 동역학/생리 검증이 남는다. 실제 영상-몸 물리 시간·AC 운전·유효랩 미검증, optimizer/actor/차량 제어 0이다. 다음은 기존 코너 실패의 시각 선행과 하중 학습의 공학적 목표를 별도로 점검하고 실제 차량 물리 자료와 비교 가능한 검증 설계를 마련한다.

## 2026-11-02 코너 조향 규모 진단 및 미분 가능 합성 차량

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

보존된 합성 baseline 4조건의 x=15.8m에서 실제 조향은 -0.00472~-0.00422rad(음 코너), +0.00240~+0.00290rad(양 코너)였다. 같은 위치 중심선에 정확히 접하는 무슬립 자전거 모형의 기하학적 유지 조향은 ±0.08731rad다. 이미 이탈한 차량의 정답 조향값이나 실차 물리가 아니다. 중심선·world pose는 사후 평가 전용이며 actor 입력에 주지 않는다. 원본·산술 `D:/CodexResearchRuns/flybrain-corner-kinematic-audit-20261102-01/`. 과거 단순 이득 학습은 한 방향을 개선하고 직선 및 다른 방향을 악화시켰다 (`20261102-joint-turn-gain-loop-1`, `20261102-straight-balanced-varied-1`), 동일 실험 반복을 피한다.

기존 합성 자전거 모형의 상수 조향·가속 정확 적분을 자동 미분 가능하게 구현했다 (`src/flybrain_drive/differentiable_planar_vehicle.py`). 0 곡률, 곡선, 정지 포함 기존 수치 일치, 조향 gradient 유한차분, 잘못된 상태 거부 3개 테스트 통과. 이는 실제 타이어/서스펜션 모델이 아니며 학습이나 주행 검증은 아니다. 다음은 고정된 영상 이력을 교사 구간으로만 사용하는 차분 가능 궤적 손실을 설계하고 직선·양 코너 및 다양한 곡률의 오프폴리시 한계를 분리해 검증한 뒤, 새 시각 영상이 차량 상태와 같이 바뀌는 폐루프를 별도로 평가한다. 고정 이력에서 나온 성적을 폐루프로 오인하지 않는다.

## 2026-11-02 고정 이력 궤적 손실 및 합성 폐루프 평가

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 합성 직선·좌우 영상/몸 이력과 기록 가속을 고정한 채, 미분 가능한 합성 차량의 중심선 오차와 진행 방향 오차를 손실로 삼아 단일 회로의 시각 projection과 기존 조향 readout gain을 6회 업데이트했다. 지도·world pose는 손실에만 쓰고 actor에 넣지 않았다. TBPTT 9결정 경계에서 그래프만 분리하고 수치 상태를 유지했다. 과거에 이미 본 fit/eval seed이며 고정 이력 실험은 폐루프가 아니다. 계획 `D:/CodexResearchRuns/flybrain-trajectory-objective-plan-20261102-01/`.

후보 저장 직후 numpy 배열 JSON 직렬화 오류로 원 실행은 실패했다 (`D:/CodexResearchRuns/flybrain-trajectory-objective-fit-20261102-01/result.json`). 6회 업데이트 checkpoint `c21269e78d420a1980807a150199abcb45353ce1db905845588a51b3b947efa8`를 재학습 없이 별도 재생한 결과 (`D:/CodexResearchRuns/flybrain-trajectory-objective-eval-20261102-01/`) 고정 이력 코너 손실은 0.493→0.348 및 0.654→0.610, 직선은 0.00359→0.01198로 악화됐다. 실패 원본은 성공으로 바꾸지 않았다.

이 결과 확인 후 질감 seed 883401/883402의 새 합성 직선·양 코너×기존/후보 12조건의 폐루프를 지정했다. 차량이 움직일 때 시각도 재생성한다. 81결정 종점에서 기존→후보 절대 중심 오차는 음 코너 0.578/0.574→0.505/0.495m, 양 코너 0.648/0.650→0.624/0.628m, 직선 0.0337/0.0298→0.0525/0.0390m였다. 저장 trajectory 12개의 종점 산술 재계산 `D:/CodexResearchRuns/flybrain-trajectory-objective-loop-20261102-01/`. 종점 x는 조건마다 약간 다르므로 해당 수치는 각 종점 기준이며 고정 위치 비교가 아니다. 큰 코너 오차가 남고 직선도 악화돼 후보를 채택하지 않는다. 사전 수치 합격 기준은 없고 선행 결과를 보고 계획한 합성 실험이다.

실제 데이터 학습·차량 제어·actor 배포 0, 생리/AC/유효랩 미검증. 다음에는 직선 안정성과 좌우 코너를 함께 만족하는 시각 표현을 갖춘 독립 조건을 우선 설계하고, 물리 시각 정합 및 몸/운동 회로의 실제 기여 검증을 계속한다.

## 2026-11-02 T4d 불일치의 사전 지정 공개 모델 비교 실행

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 공개 모델 `flow/0000/000`의 T4d는 저자 깜박임과 6속도 이동 경계에서 기대 대비 극성이 반대였다. 같은 패키지의 001·002를 결과 전에 연속 ID로 지정하고 동일 저자 자극·분석으로 조사한다. 가중치 선택/교체가 아니라 해당 한 모델 불일치가 작은 사전 지정 집합의 다른 모델에서도 관찰되는지 확인하기 위한 탐색이다. 000 실패, 저자 원 모델 및 모든 원본은 보존한다. 3개는 50개 전체 대표 표본이나 생체 반응의 증명이 아니다. 계획·모델/프로그램 해시 `D:/CodexResearchRuns/flybrain-t4d-prespecified-models-plan-20261102-01/plan.json`.

실행 당시 001 flash 완료, edge 25/144, PID 20928이 살아 있었고 002는 같은 순차 작업에서 대기 중이었다. `D:/CodexResearchRuns/flybrain-t4d-model-001-flash-20261102-01/`, `...-001-edge-...`, `...-002-flash-...`, `...-002-edge-...`. 후속 컨텍스트는 PID/원본 terminal 결과를 확인하고 완료 후 각 model에 대응한 flash+edge 분석을 수행한다. optimizer/actor/차량 제어 0이며 실제 뇌 생리·비행/주행 통합과 유효랩은 미검증이다.

## 2026-11-02 사전 지정 FlyVis 001 분석과 002 실행

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

사전 지정 001의 flash/edge 저자 분석을 완료했다 (`D:/CodexResearchRuns/flybrain-t4d-model-001-analysis-20261102-01/`). T4d 깜박임 지수 +0.29856, 6개 속도의 이동경계 대비 부호 6/6이 기대와 같다. 기존 000은 깜박임 -0.31202, T4d 부호 0/6이다. 그러나 001 T5b는 부호 0/6이고 T5a는 6속도 중 방향 오차 최대 121.53°다. 000 T5b는 6/6이고 T5a 최대 방향 오차 6.66°였다. T4d 한 지표만으로 좋은 모델을 고르거나 생리 타당성·운전 개선을 주장하지 않는다. 공개 모델의 동일 인공 자극 결과이며 예측 전에 선정된 모델 3개가 전체 population을 대표하지 않는다.

002 flash 완료, 이동경계 후속 확인 시 36/144로 동일 순차 실행 중. 종료 후 해당 원본 flash/edge를 저자 분석해 모든 T4/T5의 대비와 방향을 함께 비교한다. 최종 결과는 실제 process/PID와 terminal 자료를 다시 확인한다. optimizer·차량 제어·actor 0, 실제 물리 동기 및 주행 성능 미검증이다.

## 2026-11-02 사전 지정 공개 FlyVis 세 모델 비교 완료

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

000·001·002의 원본 checkpoint와 저자 자극 분석 입력 해시를 사전 계획에 대해 확인했다. 저자 깜박임과 12방향×6속도×2대비 이동 경계의 결과 `D:/CodexResearchRuns/flybrain-t4d-prespecified-models-comparison-20261102-01/result.json`. T4d 깜박임 지수 000/001/002 -0.3120/+0.2986/+0.3313, 이동경계 기대 대비 부호 일치 0/6, 6/6, 6/6이다. T4d만 보면 001·002가 다르지만 001은 T5b 대비 부호 0/6 및 T5a 방향 최대 오차 121.53°였고, 002는 T4b/T5a/T5b 방향 최대 오차 각각 153.70°/174.76°/179.80° 및 T5c 대비 부호 5/6이었다. 000의 T5a/b 최대 방향 오차는 6.66°/12.78°, 대비는 각 6/6이다.

이는 단일 000의 T4d 대비 불일치가 조사한 세 모델 모두에 똑같이 나타나는 것은 아님을 보여주지만 각 모델에 다른 불일치가 있다. 동일한 인공 저자 자극, 모델 세 개, 기대 방향표와 최대 오차 요약에 한정되며 전체 집단·생체 실제 반응·비행 회로 통합·AC 주행을 증명하지 않는다. 단일 지표로 모델이나 채널을 뒤섞어 교체하지 않고 원본 000과 기존 실패를 유지한다. optimizer/actor/차량 제어 0. 다음은 모델 출처별 생리 근거와 검증된 유형/방향 경계를 정리한 뒤, 동일 자극에서 각 모델의 코너 관련 운동 시각 통합 기여를 사전 지정 조건으로 평가한다.

## 2026-11-02 동일 실제 진단 영상과 세 FlyVis 모델의 통합 회로 반응

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

같은 학습 불허 AI 진단 원본 157묶기 네 시야 이미지를 000/001/002 전체 FlyVis checkpoint에 각각 같은 CSP 프레임 간격·독립 상태로 재생했다 (각 2,572 tick). 원본 PNG와 모델 해시를 확인했고 3모델의 망막 입력·시간 배열이 같음을 확인했다. 몸 패킷과 host 수신 이력이 겹치는 프레임 103–155, 53프레임/116결정에 대해 기존 단일 비행 회로와 동일 몸 이력을 사용했다. `D:/CodexResearchRuns/flybrain-three-model-unified-20261102-01/`.

000 대비 조향/전진 intent 최대 절대 차는 001 0.0030967/0.0001252, 002 0.0091704/0.0003125; 000 출력은 직전 원본 저장 이력과 최대 차 0. 모델 교체로 다르게 나올 수 있음을 보이는 값이며 교사/정답의 정확도나 생리·코너 주행 개선 증거가 아니다. 이전 방향 검사를 먼저 본 뒤 계획한 동일 진단 장면이고 downstream 회로는 000에 맞춰진 고정 구성이다. 채널 선택적 혼합·실제 학습·모델 선택 없음. 원본 물리 노출/몸 시각 미검증, host 파일 읽기 순서만 사용, actor/차량 제어 0.

다음은 모델별 운동 대비/방향 차이를 장면·움직임·밝기 통제 조건에서 같은 회로 입력/결과와 연결하되 실제 코너 주행 및 몸 감각 물리 시간 검증을 별도로 수행한다. 조향·가감속·변속·셋업 및 유효랩은 미완료다.

## 2026-11-02 FlyVis 전체 모델 대체 합성 폐루프

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

000으로 구성된 기존 통합 회로·body·readout을 고정하고 시각 network만 001 또는 002 전체 checkpoint로 바꿨다. 이전에 확인된 질감 seed 883401/883402×직선/양 코너, 각 81결정의 합성 폐루프 12조건. 원본 000 baseline은 같은 조건의 기존 `D:/CodexResearchRuns/flybrain-trajectory-objective-loop-20261102-01/`에서 보존한 결과다. 후행 계획과 해시 `D:/CodexResearchRuns/flybrain-flyvis-model-loop-plan-20261102-01/`, 원본 `D:/CodexResearchRuns/flybrain-flyvis-model-loop-20261102-01/`.

저장 궤적에서 위치 x=16.5m 보간: 001은 기존 대비 절대 중심 오차가 음 코너 0.0211~0.0253m 줄었고 직선 0.0181~0.0215m, 양 코너 0.0178~0.0232m 늘었다. 002는 음 코너 0.0457~0.0528m 줄었으나 직선 0.0478~0.0537m, 양 코너 0.0466~0.0559m 늘었다. 이는 각 모델 교체의 혼합 효과이며 독립 물리 실행이 아닌 저장 배열 산술 감사 `fixed-x-arithmetic-audit.json`이다. 모델별 downstream을 재학습하지 않아 시각망의 본질적 우열을 뜻하지 않는다. 사후 선택, 인공 자극/차량과 2질감에 한정되며 생리·실제 AC·유효랩·시각몸 물리 동기는 미검증이다. 모델 교체/배포·optimizer·차량 제어 0. 다음은 직선 중립 편향과 좌우 시각 의미의 상호작용을 학습 데이터와 구조적 검증에서 해결하고, 000 의존 시각 경로와 실제 감각 시간 근거를 유지한다.

## 2026-11-02 조향 이득 고정·직선 중시 합성 궤적 학습 실패

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 gain 증가가 직선 편향을 악화시킨 점을 반영해 gain은 고정하고 시각 projection만 6회 업데이트했으며 직선의 합성 궤적 손실을 5배로 설정했다. 기존에 본 합성 영상·몸 이력을 고정하고 world pose는 손실에만 사용했다. TBPTT 분리는 수치 상태를 초기화하지 않는다. 평가 전 기준은 x=16.5m에서 직선/양 코너가 baseline보다 5mm 초과 악화되지 않고 적어도 한 코너가 10mm 이상 개선되는 것. 사전 계획 `D:/CodexResearchRuns/flybrain-neutral-trajectory-plan-20261102-01/`.

고정 이력 결과 `D:/CodexResearchRuns/flybrain-neutral-trajectory-fit-20261102-01/`: 직선 절대 오차 0.0272781→0.0272614m, 음 코너 0.2937415→0.2937421m, 양 코너 0.3494683→0.3494586m. 최대 코너 개선 0.00972mm는 10mm 기준에 부족해 `predeclared-screen.json`에서 실패다. 따라서 사전 규칙대로 새 폐루프 후보 시험을 진행하지 않았다. projection 최대 변화 0.009806, 초기 손실 0.3786765→여섯 번째 업데이트 직전 0.3786664; 가중치 변화만으로 실패의 유일 원인은 알 수 없다. 같은 기존 synthetic scene family 평가, 실시간 시각·생리·실제 운전/랩 검증은 아니다. optimizer 6, actor/차량 제어 0.

다음은 이 조건에서 학습 신호의 시간적 신용 할당과 시각 입력의 좌우 분별/중립 독립성을 분리해, 동일한 gain 반복 증가 대신 결함을 측정 가능한 표현·교사 신호 개선으로 연결한다. 실제 영상-몸의 물리 시각과 비행 유래 회로의 실제 운전 인과 관계도 계속 미검증으로 둔다.

## 2026-11-02 차량 전체 시간 기울기 점검과 두 단계 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

이전 고정 이력의 저장 조향을 동일하게 넣은 합성 차량 모델에서 81결정 전체 미분 대비 9결정마다 차량 상태를 분리한 기울기 norm은 음/직선/양 코너 각각 2.75%/1.73%/2.57%, 벡터 코사인 0.126/0.352/0.149였다. 수치 궤적은 동일하며 float32 저장값과 차이는 최대 4.86e-6 (`D:/CodexResearchRuns/flybrain-trajectory-credit-audit-20261102-01/`). 차량-only 사후 대조이지 신경 회로 전체 기울기/실제 물리의 증거는 아니다.

`fixed_history_trajectory_gradient.py`는 고정 신경 출력에서 전체 차량 궤적 손실의 출력별 미분을 계산한다. `train_full_vehicle_credit.py`는 이를 두 번째 신경 재생의 벡터-야코비안 곱으로 전달하며 신경 상태 미분만 9결정마다 분리하고 수치 값은 유지한다. 미니 모형의 두 단계 기울기와 전체 미분 및 기존 동역학 등 4개 검사 통과. 동일 합성 이력·고정 gain·직선 가중 5, 6회 학습에서 두 pass 출력의 최대 차는 전부 0, loss 0.3786765→마지막 업데이트 전 0.3786536 (`D:/CodexResearchRuns/flybrain-full-vehicle-credit-fit-20261102-01/`).

사전 5mm 직선/코너 악화 제한과 한 코너 10mm 개선 기준에서 x=16.5m 음 코너는 0.0162mm 악화, 직선 0.0661mm 개선, 양 코너 0.0324mm 개선으로 실패다 (`predeclared-screen.json`). 따라서 후보 폐루프를 실행하지 않았고 actor 자격도 없다. 전체 차량 기울기 복원은 이 설정에서 충분하지 않았으며 그 이유가 시각 표현/단일 신경 가중치/학습량/고정 이력 중 무엇인지는 아직 확정하지 않는다. 합성 제한, actual 노출/몸 시간·생리·AC·유효랩 미검증, optimizer 6·차량 제어 0. 다음은 시각 projection에서 좌우 방향/중립을 분리할 수 있는지와 비용 기울기가 실제 바뀐 시각 입력에 얼마나 전달되는지 확인한다.

## 2026-11-02 직선·양 코너 기울기 충돌 및 공통 방향 한 단계

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

이전 고정 이력 full-vehicle-credit 학습의 기존 가중치에서 세 조건별 시각 projection 기울기를 따로 계산했다. 당시 직선 손실은 5배 가중, 신경 gradient는 9결정 절단, 차량은 전체 이력이다. 음 코너–직선 cosine -0.606, 직선–양 코너 +0.614, 양 코너 -0.411, 합 norm/개별 norm 합 0.516. 저장 기울기와 산술 감사 `D:/CodexResearchRuns/flybrain-full-credit-projection-audit-20261102-01/`. 단일 합성 질감·기존 checkpoint의 국소 기울기이며 실패의 유일 원인이나 실제 생리 증거는 아니다.

3개 기울기의 simplex 최소 norm 공통 하강 방향 계산 `src/flybrain_drive/common_descent.py`, 3개 단위 검사 통과. 저장 기울기 상 단순 합은 음 코너와 내적 -2.88e-11인 반면 공통 방향은 세 조건 모두 약 +4.21e-11로, 음의 방향으로 작은 update를 하면 각각 1차 손실이 줄 것으로 예측한다 (`D:/CodexResearchRuns/flybrain-common-descent-diagnostic-20261102-01/`). 계산 오차 허용 1e-7은 수치 최적화 잔차에 대한 상대 검사다.

최대 projection 변화 .005의 한 번 update를 사전 계획하고 이전에 본 fit/eval 두 합성 질감의 고정 이력을 재생했다 (`D:/CodexResearchRuns/flybrain-common-descent-step-20261102-01/`). fit 세 손실이 각각 3.62e-6/1.00e-6/3.23e-6 감소해 사전 국소 수치 기준을 통과했지만, x=16.5m 중심 오차 변화는 모든 조건에서 0.001~0.004mm 개선에 불과하다. 이는 이전 10mm 주행성 기준 통과나 폐루프 검증이 아니다. gain 및 기존 단일 회로 구조는 고정, optimizer 1, 실제 학습·actor·차량 제어 0. 다음은 여러 합성 이력에서 공통 기울기를 재계산해 다중 단계 학습의 크기·안정성과 독립 직선/좌우 조건을 평가하고, 사전 기준을 통과해야 폐루프로 진입한다. 실제 물리 시간·생리·AC 주행·유효랩은 여전히 미검증이다.

## 2026-11-02 두 질감 여섯 조건 공통 기울기 학습 실패

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

이전 국소 공통 방향의 매우 작은 효과를 확인해 두 합성 질감×직선/양 코너 6조건의 차량 전체 시간 기울기를 매 갱신 계산하고, 시각 projection 및 기존 단일 회로 조향 gain에 대해 정규화된 공통 하강 방향으로 8회 갱신했다. 회로 상태는 이어가며 신경 미분만 9결정에서 분리한다. 여섯 조건과 범위 제한을 사전 지정, 회로 수치의 두 pass 재생 차는 0. 계획 `D:/CodexResearchRuns/flybrain-multiscene-common-plan-20261102-01/`; 결과 `D:/CodexResearchRuns/flybrain-multiscene-common-fit-20261102-01/`. 두 질감 모두 fit 자료이며 독립 평가가 아니다.

사전 기준은 x=16.5m에서 6조건 어느 것도 5mm 초과 악화되지 않고 각 질감에서 적어도 한 코너가 10mm 개선되는 것이다. 두 질감의 음 코너 변화는 각각 -0.0042/+0.0147mm, 양 코너는 -0.0557/-0.0731mm; 직선은 -0.1494/-0.1705mm였다 (음수는 오차 감소). 10mm 개선에 못 미쳐 실패, `predeclared-screen.json` 보존. 고정 영상·몸 이력의 한계와 작은 gradient/제약 등 어느 것이 주된 원인인지는 별도 구분이 필요하다. 사전 규칙대로 새 폐루프·actor 배포는 없다. optimizer 8·실제 fit/차량 제어 0. 물리 동기·시각 생리·실제 운전 및 유효랩 미검증. 다음은 기존 고정 이력 기반 gradient 접근을 반복 확대하지 말고, 실제 시각과 몸 변화에 반응하는 폐루프 학습 데이터 및 명시적 teacher/actor 분리를 우선 설계한다.

## 2026-11-02 모델 구동 합성 이력 수집과 첫 데이터 집계 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

기존 모델과 직선 악화로 미채택했던 궤적 손실 후보가 각각 실제로 움직이는 합성 차량을 구동하게 하고, 움직임에 따라 네 시야/FlyVis와 몸이 변화하는 2질감×2모델×직선/양 코너 총 12이력, 각 81결정을 수집했다 (`D:/CodexResearchRuns/flybrain-dynamic-history-20261102-01/`). 파일 해시, 형상, 유한성 및 차량 상태 연결을 감사했다. 후보 사용은 이력 확보용이며 후보 채택이 아니다. 선봉인된 883501은 fit, 883502는 holdout. 신경 actor 입력은 영상/유효 몸 감각이며 합성 pose·중심선은 교사/평가에만 쓴다.

두 모델이 만든 fit 질감의 6개 이력을 모아 시각 경계·기존 조향 gain을 6회 학습하는 첫 데이터 집계 라운드 `D:/CodexResearchRuns/flybrain-dynamic-dagger-fit-20261102-01/`. 학습 당시에는 수집 이력을 고정해 재생했으며 candidate의 새 폐루프가 아니다. fit 교사 MSE는 매 업데이트 전 0.16251→0.15973. 사전에 보류 질감의 각 6이력 교사 MSE가 5% 초과 악화되지 않고 전체 평균이 1% 이상 개선돼야 후보 폐루프로 진입하도록 지정했다 (`D:/CodexResearchRuns/flybrain-dynamic-dagger-plan-20261102-01/`).

보류 평균 MSE 0.16258→0.15928, 코너 네 이력은 약 1.2~3.1% 감소했으나 직선 baseline/candidate 이력은 각각 15.1%/12.1% 악화돼 사전 기준 실패 (`predeclared-screen.json`). 규칙에 따라 신규 후보 폐루프·actor 배포를 하지 않는다. 보류 질감은 fit에서 제외했지만 같은 합성 도로 생성기이므로 독립 환경·실제 물리 일반화가 아니다. optimizer 6·실제 데이터 fit/차량 제어 0, 영상·몸 물리 시각과 생리·실제 AC·유효랩 미검증. 다음은 직선 악화 원인을 교사-입력 분포와 중립 편향으로 분석하고, 고정 이력 평가를 통과한 이후에만 새 폐루프를 실행한다.

## 2026-11-02 모델 구동 이력의 공통 기울기 학습 실패

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 최종 목표다.

앞서 수집한 모델 구동 합성 이력 중 fit 질감 6조건의 teacher MSE 기울기를 각각 계산하고 시각 projection·기존 gain의 최소 norm 공통 방향을 매회 계산해 8번의 유한한 update를 했다. scene/body는 수집 이후 고정되며 정확한 폐루프 학습은 아니다. 해시 선봉인 계획 `D:/CodexResearchRuns/flybrain-dynamic-common-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-dynamic-common-fit-20261102-01/`.

fit 평균 MSE는 업데이트 전 0.16250946→마지막 업데이트 전 0.16251103, 보류 질감 평균 0.16258059→0.16258280. 직선 두 이력 MSE는 약 0.13%/0.07% 감소, 네 코너는 0.0004~0.0032% 증가했다. 사전 각 이력 5% 악화 제한은 통과하나 평균 1% 개선 요건을 채우지 못해 실패 (`predeclared-screen.json`). projection 최대 0.0200005, gain 변화 1.14e-5. 각 계산의 최소 task 내적은 약 -2.9e-16으로 solver의 허용 오차 내 음수였으며, 제한 범위 투영과 유한 스텝도 있어 모든 조건에서 엄밀한 공통 하강이었다고 주장하지 않는다.

고정 이력에서 교사 모방의 두 번째 변형도 실제로 유용한 개선을 내지 못했다. 후보 폐루프·actor 배포는 사전 규칙대로 하지 않는다. 합성 체계·가설 teacher, 실제 물리 시각/생리·AC·유효랩 미검증, optimizer 8·차량 제어 0. 다음에는 시각 정보의 분리 가능성, 기준 영상/외형 편향 및 신경 경로의 제어 권한을 측정하고 조건별 경로 설계가 필요한지 검증한다.


### 2026-11-02 — 모델 구동 이력의 시각 정보 탐침

**2026-11-02 모델 구동 이력의 시각 정보 탐침:** 합성 폐루프 12개 이력에서 한 질감으로 선형 비교기를 맞추고 다른 질감의 직선/좌우를 평가했다. 코너 시작 전 x=9~11.5m와 이후 x=13~16m 모두 망막 공간 대비는 보류 66.7%였으나 직선은 전부 오분류, 운동 시각은 33.3%로 한 클래스로만 예측했다. 조향 intent 선형 탐침도 직선 0%였다. 사후 점검에서 망막·운동 특징 학습 정확도는 100%였지만 학습 기준 표준화 후 보류 RMS가 각각 3.21~8.20, 3.54~93.56으로 달랐다. 이는 질감 변화에 취약한 **이 탐침**의 결과이지 회로의 정보 부재나 실패 원인을 입증하지 않는다. 각 보류 조건은 단일 이력이고 프레임은 상관되어 있다. 입력·계획·결과 `D:/CodexResearchRuns/flybrain-dynamic-visual-probe-20261102-01/`; 사후 확인 `posthoc-train-shift.json`. 위치·굽음은 오프라인 구간/라벨에만 사용했고 actor 입력이 아니다. 다음은 독립 질감·다양한 차체 상태와 과도한 표준화의 영향을 분리하는 감각 반응 및 회로 인과 비교다. 생리·물리 동기·실제 주행·actor는 미검증이다.



### 2026-11-02 — 합성 시각 탐침의 정규화 민감도

**2026-11-02 합성 시각 탐침의 정규화 민감도:** 앞서 확인한 동일 합성 이력·동일 학습/보류 질감을 사후 재분석했다 (`D:/CodexResearchRuns/flybrain-dynamic-scale-diagnostic-20261102-01/`). 운동 시각 특징의 채널별 표준화에서 보류 정확도는 코너 전/후 각각 33.3%/33.3%, 특징 전체에 동일한 RMS 척도를 적용하면 67.5%/100%였다. 후자의 코너 전 클래스별 정확도는 음/직선/양 11.9%/90.5%/100%였다. 학습 질감에서는 모두 100%였고 보류는 질감 하나·모델별 조건 하나씩의 상관된 프레임이다. 이미 보류 결과를 본 뒤 지정한 사후 비교이므로 독립 검증이나 모델 선택 기준이 아니다. 척도와 ridge의 상대적 규제가 함께 바뀌어 이전 비교가 전처리에 민감함을 확인하며, 회로 자체에 운동 정보가 없다는 결론을 지지하지 않는다. 분석기는 actor가 아니고 실제 회로 입력이나 학습을 변경하지 않았다. 다음은 새로운 질감과 차체 상태에서 시각 반응의 선행 정보 및 단일 회로 개입 효과를 사전 지정 검증한다. 물리 동기·생리·실제 주행은 미검증이다.



### 2026-11-02 — 새 질감 모델 구동 이력과 사전 지정 평가

**2026-11-02 새 질감 2개 모델 구동 이력·사전 지정 평가:** 기존 회로와 이전 미채택 후보를 새 합성 질감 883601/883602에서 직선/좌우 각 81결정씩 실행했다 (`D:/CodexResearchRuns/flybrain-dynamic-holdout-20261102-01/`). 기존 두 질감으로 선형 비교기를 학습하고 새 질감 조건별 정확도 80% 이상을 사전 요구한 평가 (`D:/CodexResearchRuns/flybrain-dynamic-holdout-eval-20261102-01/`, 계획 `D:/CodexResearchRuns/flybrain-dynamic-holdout-eval-plan-20261102-01/`)는 48개 중 34개만 충족해 실패했다. 공통 RMS 척도에서 코너 전 망막 36.1%, 운동 특징 83.3%였으나 각각 최악 이력 0%; 코너 후 망막 76.8%, 운동 특징 100%였다. 사후 운동 특징에는 차량 상태 변화가 반영될 수 있어 시각 선행 예측의 증거가 아니다. 후보의 최종 코너 절대 오차는 기준 0.573~0.652m에서 0.493~0.631m로 줄었지만 직선 절대 오차가 0.030~0.032m에서 0.041~0.046m로 늘어 기존 실패가 재현됐다. 합성 도로·두 질감·조건별 상관된 프레임이며 actor 적격성/생리/실제 AC·물리 동기는 미검증이다. 다음은 같은 차체 상태에서 새 질감별 시각 선행 입력을 통제해 감각 차이를 분리하고 직선 중립을 보존하는 회로 학습을 설계한다.



### 2026-11-02 — 동일 차체 상태의 시각 선행 반응

**2026-11-02 동일 차체 상태의 시각 선행 반응:** 직전 두 합성 질감의 기존 회로 직선 이력을 기준으로, 12개 재생에서 몸 감각·자세 이력은 같게 고정하고 도로 굽이만 바꿨다 (`D:/CodexResearchRuns/flybrain-matched-pose-replay-20261102-01/`; 분석 `D:/CodexResearchRuns/flybrain-matched-pose-analysis-20261102-01/`). 코너 시작 전 x=9~11.5m, 두 질감×기존/후보×좌우 8조건의 21프레임 모두 조향 차이는 해당 굽이 방향이었다. 조건별 평균 차 절대값은 기존 0.00337~0.00409rad, 이전 미채택 후보 0.00616~0.00749rad였다. 사후 비교에서 직선 조향 평균은 기존 -0.00131~-0.00143rad, 후보 -0.00240~-0.00261rad로 후보가 같은 몸 상태에서 한쪽 편향을 늘렸다 (`posthoc-straight-bias.json`). 원본 직선 이력 재생은 scene 최대 차 1.20e-7, 조향 intent 최대 차 1.83e-7, 굽이 조건 간 몸 이력 차 0. 이 실험은 합성 화면 변화의 선행 조향 영향과 편향을 보여주지만 직선 악화의 유일한 원인, 생리 기능, 실제 AC 또는 폐루프 성능을 증명하지 않는다. 후보는 미채택·actor 불허. 다음은 이 동일 몸/시각 쌍을 써 직선 중립을 보존하는 비행 회로 경계 학습을 설계하고 새로운 질감·자세에서 검증한다.



### 2026-11-02 — 동일 차체 이력 직선 중립 학습

**2026-11-02 동일 차체 이력의 직선 중립 학습: 기준 실패.** 기존 미채택 후보를 시작점으로, 같은 몸 이력에서 직선 출력은 기존 회로에 맞추고 좌우 굽이와 직선의 후보 출력 차이는 유지하는 목표로 시각 경계 projection만 6회 학습했다 (`D:/CodexResearchRuns/flybrain-matched-neutral-fit-20261102-01/`, 계획 `D:/CodexResearchRuns/flybrain-matched-neutral-fit-plan-20261102-01/`). 883601 질감으로 fit, 이전에 결과를 살펴본 883602를 보류했으며 합성 고정 이력이다. x=9~11.5m 보류 직선 intent의 기존 회로 대비 MAE는 0.00339157→0.00340782로 악화해 사전 요구 50% 감소에 실패했다. 좌우 응답 차이 변경은 각각 0.00000736/0.00001347로 원래 차이 0.01761/0.02139의 10% 제한 이내였지만 전체 기준은 실패다. 학습 질감 전체 평균 MSE는 2.4586e-5→2.4541e-5(각 업데이트 전 기록), 마지막 업데이트 후의 독립 fit 점수는 아니다. 사전 replay 출력 차 0, 최대 가중치 변화 0.01174. 후보 폐루프·배포 불허, 물리 동기·생리·실제 AC 미검증. 다음은 이전 후보에서 단순 projection 갱신이 직선 편향을 일반화하지 못한 이유를 확인하고 새로운 자세/질감을 포함한 직선 중립 회로 설계를 검토한다.



### 2026-11-02 — 코너 진입 전 구간 중립 학습

**2026-11-02 코너 진입 전 구간에 맞춘 중립 학습: 기준 실패.** 직전 학습이 전체 프레임을 사용한 점을 수정해 x=9~11.5m의 손실만 6회 갱신했다 (`D:/CodexResearchRuns/flybrain-matched-neutral-window-20261102-01/`, 계획 `D:/CodexResearchRuns/flybrain-matched-neutral-window-plan-20261102-01/`). 학습 질감의 해당 구간 MSE는 업데이트 전 1.1180e-5→1.1051e-5로 내려갔으나 보류 질감의 직선 intent MAE는 0.00339157→0.00341124로 늘어 사전 50% 개선 기준 실패다. 좌우 반응 변화는 허용 이내이고 전체 81프레임 직선 MAE도 0.00240827→0.00241719로 5% 악화 제한 이내지만 전체 기준은 실패다. fit/보류 각 질감 하나이며 보류는 이전에 본 합성 이력이다. 이전 후보에서 출발한 시각 projection만 갱신, 최대 변화 0.01184; 학습과 판정 구간 불일치만으로 보류 악화를 설명할 수 없다. 후보 폐루프·배포 불허. 다음은 단일 질감의 미세 조정 반복을 중단하고 다양한 질감/차체 상태를 가진 학습 자료와 실제 감각 시간 동기/회로 인과 검증을 우선한다. 생리·실제 AC·물리 동기는 미검증이다.



### 2026-11-02 — CSP 시계 변환 직접 호출 검증 보강

**2026-11-02 CSP 시계 변환 직접 호출 검증 보강:** 실제 움직임 진단의 카메라 시계쌍과 몸 패킷을 재확인했다. 렌더 호출/호스트 수신은 각각 픽셀에 반영된 물리 시점/몸 발생 시점이 아니므로 기존 자료의 물리 동기 오차는 여전히 미확정이며 `training_admissible=false`다. `CSPClockBounds.render_start_lower()`는 같은 generation 및 유한값만 검사해 직접 호출 시 교환이 없거나 폭이 너무 넓어도 하한을 반환할 수 있었다. 최소 두 번의 상호 일치한 bounded 교환과 음수가 아닌 원본 시각을 직접 요구하도록 수정했다. 교환 0개·넓은 교환·음수·정상 경계 회귀 포함 관련 18개 검사 통과. 이 변경은 호스트 렌더 시작 추정의 입력 검증이며 시계 단위율 가정, 물리 노출/몸 샘플 동기를 검증하거나 실데이터 학습을 허용하지 않는다. 원본 데이터·기존 실행은 변경하지 않았다. 다음은 독립적으로 식별 가능한 렌더 물리 프레임과 몸 생성 tick 또는 검증된 상한을 확보하는 계측이다.



### 2026-11-02 — CSP 시간·몸 상태 동시 진단 후보

**2026-11-02 CSP 시간·몸 상태 동시 진단 후보:** 기존 검증된 카메라 진단 후보를 기반으로 각 시야 렌더 호출 전후 `car.timestamp`와 같은 읽기 구간에 CSP `speedKmh`, `localAngularVelocity`, `acceleration`을 보존하는 후보 `integrations/ac_csp/candidates/FlyBrainCamera-state-clock-owned-v1.lua`를 만들었다 (SHA-256 `94b23290e7fd6787659f6f67f6b5170f43f821afb5be2f97073f4c237177dd56`). 로컬 CSP SDK는 car 상태가 그래픽 주기 캐시일 수 있음을 명시하므로 이는 픽셀이나 native 패킷의 물리 tick 동기가 아니다. `src/flybrain_drive/diagnostic_cached_body_clock.py`는 원본 값과 읽기/렌더 순서를 검증하고 시간·상태 차이를 진단하되 동기·native 일치·학습 자격을 부여하지 않는다. 결측·비유한·순서 역전·학습 허용 거부 포함 관련 13개 검사와 Lua 문법 load 확인; Lua 문법 검사는 CSP 실행 검증이 아니다. 후보는 설치하거나 실제 실행하지 않았다. 다음은 기존 설치본 해시 및 lifecycle/원복 조건을 검증하고 정지 진단에서 상태 원본을 수집해 native와의 일치 가능성을 별도로 검토한다. 물리 노출·몸 발생 시각·생리·실제 AC 자율주행은 미검증이다.



### 2026-11-02 — 실제 정지 CSP 시계·몸 상태 진단

**2026-11-02 실제 정지 CSP 시계·몸 상태 진단 및 원복:** 해시 고정 lifecycle pause 후 `FlyBrainCamera-state-clock-owned-v1.lua`를 원본 백업·검증해 임시 설치했고, 기존 probe를 후보 해시에 맞춘 `scripts/probe_state_clock_owned_v1.py`로 네 시야 4묶음을 수집했다 (`D:/CodexResearchRuns/flybrain-stateclock-probe-20261102-01/`). 아카이브 전 파일 해시와 frame identity를 재검사하여 16개 시야의 전후 시각·속도·각속도·가속도가 존재함을 확인했다 (`D:/CodexResearchRuns/flybrain-stateclock-analysis-20261102-01/`). 전후 CSP 속도는 각 0.0005164km/h로 불변; 설치 전 별도 native 수신은 0.0005233km/h였지만 동시 샘플이 아니고 정지라 tick 동기 증거가 아니다. 기존 카메라 SHA-256 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`로 원복 (`D:/CodexResearchRuns/flybrain-stateclock-restore-20261102-01/`), lifecycle resume 뒤 paused=false, camera_active=false, AI=false, brake=1 확인. UI·actor·차량 제어·optimizer 0. 이 정지 수집은 후보 CSP 실행과 원본 보존을 확인할 뿐 물리 노출/몸 발생 동기·생리·실제 주행 학습을 증명하지 않는다. 다음은 동일 캡처 구간의 native 패킷을 원본 수신 시각·바이트와 함께 수집하는 움직임 진단을 설계해 속도·회전의 시간 변화와 식별 가능성을 검증한다.



### 2026-11-02 — 움직이는 CSP 상태·native 동시 수집

**2026-11-02 움직이는 CSP 상태·native 동시 수집:** lifecycle과 제한된 CSP expert-motion-probe로 3초 진단 주행 중 카메라 150묶음/600시야와 native 원본 바이트 복사 5,194개를 기록했다 (`D:/CodexResearchRuns/flybrain-stateclock-motion-20261102-01/`). 카메라 픽셀/메타데이터 host 관찰 해시, 모든 native prefix 해시·copy 시각, 전후 CSP↔host 시계 교환을 보존하고 재검사했다. 시계 오프셋 폭 3.281ms는 교환과 단위율 가정에 한정된다. 비교 분석 (`D:/CodexResearchRuns/flybrain-stateclock-motion-analysis-20261102-01/`)은 각 렌더 호출의 추정 host 시각 중간값에 가장 가까운 native 복사(진단상 미래 허용)를 선택했다. CSP 속도 >1km/h인 168시야에서 속도 절대 차 중앙 0.0237, 95백분위 0.493, 최대 0.864km/h; 최대 CSP 속도 16.292km/h. 시야/프레임은 상관되어 독립 표본이 아니다. 렌더 호출·native 복사는 픽셀 노출/물리 발생 시점이 아니어서 이 숫자는 물리 동기 오차나 같은 tick의 증거가 아니다. 원래 카메라 해시 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`로 원복하고 paused=false·camera_active=false·AI=false·brake=1 확인했다. AI 진단 외 student 제어·UI·optimizer 0; 실제 자료 학습/actor 불허. 다음은 원본 몸/영상의 움직임 시점 관계를 외부 표식 또는 명시된 소스 이벤트 증거로 검증한다.



### 2026-11-02 — 실제 픽셀 변화와 native 속도

**2026-11-02 실제 픽셀 변화와 native 속도의 탐색적 비교:** 움직임 진단의 원본 영상 150묶음·native 5,194복사의 해시를 재검증하고 시야별 인접 프레임 그레이스케일 절대차/렌더 간격을 native 속도와 비교했다 (`D:/CodexResearchRuns/flybrain-pixel-native-timing-20261102-01/`, 사전 방법 `D:/CodexResearchRuns/flybrain-pixel-native-timing-plan-20261102-01/`). 렌더 추정 시각 대비 -0.4~+0.4초에서 가장 높은 Pearson 상관은 전/좌/우/후 각각 0.661/0.796/0.706/0.695, 해당 shift는 -0.08/+0.16/+0.08/+0.08초였다. 0초 값은 각각 0.404/0.787/0.702/0.691. 단일 짧은 움직임·상관된 프레임이며 도로 무늬, 회전, 조명 및 노출 영향이 있어 shift는 동기 오차나 물리 노출 지연이 아니다. 그레이스케일 픽셀차는 광학 흐름도 아니다. 단위율 가정된 렌더 호출과 native host 수신만 사용하며 진단상 미래 샘플도 포함한다. 실제 픽셀 변화와 속도 사이 관찰 관계를 확인했지만 물리 동기·생리·학습/actor 자격은 여전히 미검증이다. 다음은 독립적인 표시 이벤트나 검증된 소스 프레임 계약을 확보한다.



### 2026-11-02 — 정지 카메라 렌더 콜백 표식 실증

**2026-11-02 정지 카메라 렌더 콜백 표식 실증:** CSP GeometryShot의 opaque callback에서 시야별 색상을 프레임마다 적/녹으로 바꾸고 콜백 source clock·car.timestamp를 기록하는 후보 (`integrations/ac_csp/candidates/FlyBrainCamera-render-marker-diagnostic-v1.lua`)를 만들었다. 후보는 paused의 `allow_ideal_line_probe=true`, 최대 8장, 파일 전용만 허용하고 actor/학습을 금지한다. 기존 카메라 백업·해시 고정으로 임시 설치, 정지 4묶음×4시야 수집 후 원본 해시 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`로 원복·resume했다 (`D:/CodexResearchRuns/flybrain-marker-probe-20261102-01/`, `...-restore-...`, `...-resume-...`). 원본 파일 해시·16개 callback이 각 render 호출 내부이며 인덱스/시야에 일치함을 검증했다 (`D:/CodexResearchRuns/flybrain-marker-analysis-20261102-01/`). 사후 정한 픽셀 색차 기준에서 네 시야 모두 적/녹 교대 영역이 관찰됐다(전/좌/우/후 900/263/261/258픽셀). 이는 해당 정지 진단에서 이미지와 렌더 콜백의 연관을 실증하지만, 콜백에서 읽은 캐시 car.timestamp가 그려진 세계의 물리 tick임을 증명하지 않고 native 생성 시각도 알 수 없다. 생리·실제 학습·actor·물리 동기는 미검증. 다음은 콜백 표식과 원본 native 상태를 함께 수집할 수 있는 제한적 이동 진단을 별도로 설계하고, 표식에 기록된 캐시와 실제 렌더 물리 관계를 독립적으로 검증한다.



### 2026-11-02 — 움직이는 렌더 표식과 native 원본 동시 진단

**2026-11-02 움직이는 렌더 표식과 native 원본 동시 진단:** 정지 전용 표식 후보를 별도 bounded motion 전용 후보 `integrations/ac_csp/candidates/FlyBrainCamera-render-marker-motion-v1.lua`로 확장했다. 정지·브레이크·오프라인에서 명시적 요청으로만 시작하고 파일 출력·진단 `aid_probe_only=true`, actor/학습 불허를 강제한다. CSP lifecycle AI 3초 진단 중 네 시야 155묶음과 native 원본 5,380복사·전후 host/source 시계 교환을 수집했다 (`D:/CodexResearchRuns/flybrain-marker-motion-20261102-01/`). 카메라 원본을 해시 고정 복구하고 resume 후 paused=false, AI=false, camera_active=false, brake=1 확인. 독립 정지 표식 영상에서 얻은 영역·색 기준으로 실제 주행 620시야 모두 표식 색/프레임 인덱스가 일치했고, 620개 콜백은 모두 대응 render 호출 안에 있었다 (`D:/CodexResearchRuns/flybrain-marker-motion-analysis-20261102-01/`). 시각 교환 폭은 3.549ms(단위율 가정)다. 168개 움직이는 시야에서 콜백 host 추정에 가장 가까운 native 복사 속도와 CSP 캐시 속도의 차이 중앙 0.0229, 95백분위 0.503, 최대 0.866km/h. 진단상 미래 복사도 허용한 비교로 원본 생성 tick/픽셀의 세계상태 시각이나 물리 동기 오차는 알 수 없다. 한 짧은 주행·시야 상관, actor/optimizer/실제 학습 없음. 다음은 표식 콜백에 기록한 car.timestamp와 화면 속 차량/세계 기하의 시간 관계를 독립적으로 평가하고, native 발생 시각 상한 근거를 확보한다.



### 2026-11-02 — 실제 전방 영상의 정적 무늬·카메라 기하

**2026-11-02 실제 전방 영상의 정적 무늬·카메라 기하 비교:** 렌더 표식 이동 진단에서 전방 영상의 표식 밖 고정 격자 520개 patch를 추적해 세 시점의 대응 127개를 찾고 앞 두 시점 camera metadata로 삼각측량했다 (`D:/CodexResearchRuns/flybrain-landmark-geometry-20261102-01/`, 방법 `D:/CodexResearchRuns/flybrain-landmark-geometry-plan-20261102-01/`). 양의 유한 깊이를 가진 83개 추적점의 세 번째 영상 위치 재투영 오차 중앙 0.539px, 95백분위 9.06px, 최대 194.16px. 사후 대조 (`D:/CodexResearchRuns/flybrain-landmark-pose-shift-20261102-01/`)는 세 번째 카메라 자세만 3프레임 뒤로 바꾸어 두 방법 모두 투영 가능한 82점에서 원래/변경 오차 중앙 0.526/0.998px, 원래가 작은 점 74개였다. 선택된 점과 겹친 10개 시간 구간은 독립 표본이 아니고 잘못된 추적·이동 물체가 있을 수 있다. 자세 metadata와 렌더러가 같은 원본을 공유할 수 있어 화면 기하와 상대 자세의 제한적 일치 증거일 뿐, native 물리 tick이나 절대 동기 오차가 아니다. 사후 대조는 사전 등록 검정이 아니다. 원본 이미지·해시 보존, actor/학습/차량 제어 0. 다음은 실제 몸 패킷의 생성 tick을 확인할 독립적 소스 증거 또는 물리적 지연 상한을 얻는다.



### 2026-11-02 — CSP/native 상태 수치 일치

**2026-11-02 움직이는 렌더 콜백의 CSP/native 수치 일치:** 기존 3초 표식 진단의 영상·메타데이터·5,380 native 복사 원본 해시를 검증하고, 콜백 CSP speed와 3축 localAngularVelocity를 native speed/3축 각속도와 binary32로 정확 비교했다 (`D:/CodexResearchRuns/flybrain-cached-native-value-match-20261102-01/`). CSP 속도 >1km/h인 168시야 모두 일치하는 패킷이 있었고, 각 시야에 대해 기록된 표본 안에서는 일치 패킷 ID가 하나였다. 가속도 3축까지도 정확 일치한 것은 152시야다. 해당 값의 가장 가까운 native host 복사와 콜백 host 추정 시각 차는 -21.23~+0.89ms(중앙 -9.26ms)이며 source↔host 시계 교환 폭 3.549ms 및 단위율 가정이 있다. 이 값은 소급 분석이고 하나의 짧은 주행·중복된 시야/프레임이다. 같은 수치가 같은 physics tick을 입증하지 않으며 host 복사/콜백은 생성·세계 렌더 물리 시각이 아니다. 영상/몸 시간 동기, native packet identity, 생리·실제 학습 자격은 여전히 미검증. 다음은 값 일치를 개별 시각 증거와 함께 활용할 수 있도록 반복 수집에서 유일성·패킷 누락·시간 불확실도를 검증한다.



### 2026-11-02 — 렌더 표식·native 수치 반복 진단

**2026-11-02 렌더 표식·native 수치 반복 진단:** 같은 차·트랙에서 사전 계획 후 별도 3초 CSP AI 주행을 수집했다 (`D:/CodexResearchRuns/flybrain-marker-motion-20261102-02/`, 계획 `D:/CodexResearchRuns/flybrain-marker-motion-repeat-plan-20261102-01/`). 네 시야 161묶음/644화면 및 native 5,605복사를 보존했고 644시야 표식 색상·콜백 인덱스가 기존 정지 캘리브레이션과 일치했다 (`D:/CodexResearchRuns/flybrain-marker-motion-analysis-20261102-02/`). 속도 >1km/h 168시야 모두 속도·3축 각속도의 binary32 값과 일치하는 native 패킷이 있었으며, 수집 표본에서 각각 유일했다. 가속도까지 일치한 것은 152시야 (`D:/CodexResearchRuns/flybrain-cached-native-value-match-20261102-02/`). 두 주행의 불일치 각 16시야는 출발/정지 부근 4프레임×4시야였고, 첫 출발 프레임 CSP 가속도는 세 축 모두 약 1e-30이었다. 원인은 미확정이며 native와 무조건 동등 취급하지 않는다 (`D:/CodexResearchRuns/flybrain-marker-repeat-gap-audit-20261102-01/`). 샘플된 서로 다른 native 패킷 ID는 각 1,393개; 첫 주행 범위에는 ID 공백이 없고 두 번째는 ID 하나가 관찰되지 않은 구간 1개가 있다. 이는 샘플링 누락이지 엔진 전송 손실 증명이 아니다. 동일 환경의 두 짧은 주행·상관된 시야이며 공통 물리 tick/영상 노출과 몸 생성 시각의 인증이 아니다. 기존 카메라 해시로 복구, paused=false·카메라/AI off·brake=1 확인. 실제 fit/actor 및 생리 검증 불허 유지. 다음은 가속도 수신과 CSP 캐시 불일치를 마스크 처리하는 경계 및 독립 물리 시각 증거를 검토한다.



### 2026-11-02 — 실제 native 가속도 마스크 검증

**2026-11-02 실제 native 가속도 마스크 검증:** 현재 단일 회로의 `VirtualBodyEncoder`는 CSP 가속도를 입력으로 쓰지 않으며, native `accG`도 축/중력 등록이 완료되지 않아 linear acceleration 3채널을 0과 invalid로 유지한다. 두 실제 3초 진단의 원본 패킷 해시를 검증해 중복 ID 첫 수신만 각각 1,393개 재생했다 (`D:/CodexResearchRuns/flybrain-real-acceleration-mask-audit-20261102-01/`). 두 run 각각 1,384개에 비영 가속도가 있었지만 가속도 3채널 유효/값 누출은 0건, 속도 채널은 전부 유효했다. 이 오프라인 encoder replay는 계약의 신원값을 제공한 것이며 실제 생리·물리 시각을 인증하지 않는다. 별도 단위 테스트에서 상반된 큰 native 가속도 입력에도 몸 특징/마스크가 같음을 확인했고 관련 11개 검사 통과. 마스크는 안전 경계이면서 실제 몸 가속도 되먹임이 목표 대비 미구현임을 뜻한다. 다음은 native 축·부호·중력과 시각 시간 정합을 독립 검증한 뒤 단일 비행 회로에 도입한다. 실제 fit/actor/주행 미검증.



### 2026-11-02 — native 가속도 축의 속도 미분 비교

**2026-11-02 native 가속도 축의 속도 미분 비교:** 두 기존 3초 진단 원본 prefix 해시를 검증하고, 첫 주행에서 호스트 수신 기준 중심 150ms의 속도 크기 선형 미분과 native accG 3축을 비교해 절대 상관 최대 축을 선택했다 (`D:/CodexResearchRuns/flybrain-native-accel-axis-20261102-01/`, 사전 방법 `D:/CodexResearchRuns/flybrain-native-accel-axis-plan-20261102-01/`). 속도 >5km/h 및 창 내 수신 간격 ≤20ms 조건에서 native index 2가 선택되었고 첫 run 상관 0.9868, 고정된 관계의 둘째 run 상관 0.9866 및 예측 RMSE 0.609m/s²였다. 비교 대상은 서로 겹치는 996/995개 창이며 양/음 속도 미분을 모두 포함한다. 단위 변환은 표준 중력 9.80665m/s²를 가정했다. 이 결과는 한 차·트랙의 짧은 움직임에서 해당 축이 속도 미분과 연관된다는 진단이며, 수신 시각을 물리 샘플 시각으로 인증하지 않고 미분에 미래 패킷을 쓰며 횡/상하 축·노면 기울기·중력·슬립·실시간 성능을 검증하지 않는다. 가속도 회로 마스크는 유지한다. 다음은 서로 다른 회전·경사·가감속 조건에서 나머지 축과 중력 분리를 독립 검증한다. 실제 학습/actor·생리·물리 동기 미검증.



### 2026-11-02 — 실제 코너 진단 횡방향 가속도 축 비교

**2026-11-02 실제 코너 진단의 횡방향 가속도 축 비교:** 기존 AI pace 원본 두 기록은 학습 불허를 유지하고 진단에만 사용했다. 첫 기록의 속도≥5m/s, |yaw|≥0.02rad/s 조건에서 speed×기존 body yaw(좌회전 양)를 횡방향 구심 가속도의 단순 근사로 삼아 native accG 3축과 비교했으며 절대 상관 최대인 index 0을 선택했다 (`D:/CodexResearchRuns/flybrain-native-lateral-axis-20261102-01/`, 사전 계획 `D:/CodexResearchRuns/flybrain-native-lateral-axis-plan-20261102-01/`). 첫/둘째 기록은 각각 3,685/4,801개 상관된 고유 패킷이고 좌우 yaw를 모두 포함했다. 선택 축과 근사의 상관은 첫 0.970, 둘째 0.987, 부호 일치 비율은 각각 99.6%/99.1%였다. 9.80665m/s²/g 가정이며 yaw와 속도의 native 수신 상태 및 작은 횡슬립/횡속도 변화 가정에 의존한다. 노면 기울기·중력·타이어 슬립·physically aligned 시각은 분리하지 못해 축 등록·생리·실제 주행 능력의 증거가 아니다. 두 run은 동일 차/트랙, 패킷은 서로 상관된다. 실제 가속도 입력은 계속 0/invalid이며 actor/실제 fit/차량 제어 0. 다음은 경사·정지·회전 조건에서 중력 및 횡미끄러짐 영향을 분리한다.



### 2026-11-02 — 실제 정지 native 가속도 관찰

**2026-11-02 실제 정지 native 가속도 관찰:** 처음 계획한 3초 이동 진단 원본에서는 첫 기록에 연속 1초의 고유 native 패킷 정지 창이 없어 해당 계획을 `insufficient_qualifying_stationary_windows`로 보존하고 중단했다 (`D:/CodexResearchRuns/flybrain-native-stationary-gravity-20261102-01/`). 이후 과거 정지 shadow 원본 `D:/CodexResearchRuns/flybrain-unified-process-live-20261102-02/`의 native 복사 해시를 검증하고 별도 계획으로 재분석했다 (`D:/CodexResearchRuns/flybrain-stationary-shadow-gravity-20261102-01/`). speed<0.1km/h·세 축 각속도<0.01rad/s·앞뒤 각 0.5초 호스트 수신 간격≤20ms 조건의 1,621개 고유 패킷은 accG 세 축이 모두 0이었다. 당시 pitch -0.007616~-0.007615rad, roll +0.003796~+0.003798rad로 기울기 변화가 거의 없다. 원래 shadow run은 stale로 `stopped_on_error`였고 이 분석으로 성공 처리하지 않는다. 차량 물리 sleep, 게임의 중력 처리, 필터링 및 실제 물리 시각을 구별하지 못하므로 0이 무중력 또는 중력 보정의 증거가 아니다. 가속도 축/중력 등록과 actor/실제 학습은 미허용이다. 다음은 다른 기울기와 더 다양한 회전 상태의 실측 대조를 확보한다.



### 2026-11-02 — 실제 world velocity 미분·카메라 축 비교

**2026-11-02 당시 분석 — 좌측 외적 방향 오류, 후속 정정 참조:** 기존 움직임 진단 두 run에서 전방 콜백과 속도/3축 yaw-rate가 정확히 일치하는 고유 native 패킷을 찾고, 해당 패킷 앞뒤 150ms host 수신의 world velocity를 선형 미분해 기록된 카메라 forward, cross(forward,up), up으로 투영했다 (`D:/CodexResearchRuns/flybrain-world-accel-20261102-01/`, 사전 방법 `D:/CodexResearchRuns/flybrain-world-accel-plan-20261102-01/`). 각 run 전방 37프레임에서 native index2와 투영 전방 미분의 상관은 0.989/0.976이었다. 그러나 index0과 기하상 left 투영의 상관은 -0.666/-0.629로, 이전 speed×등록 yaw 근사와 index0의 양의 상관(0.970/0.987)과 부호가 충돌한다. 두 실험은 주행 기록·운동 범위와 참조가 달라 정확한 원인은 미확정이며 카메라/물리 좌표, yaw 부호·횡슬립을 재검증해야 한다. 이번 3초 데이터의 left 투영 표준편차가 0.028/0.031m/s²에 그쳐 상관이 불안정하다. index1과 up 투영 상관은 0.898/0.948이지만 up 변화도 작고 중력 분리 증거가 아니다. host 시간 미분에는 미래 샘플이 포함되고 콜백-패킷 수치 일치는 공통 물리 tick 증명이 아니다. 차체 world 방향은 오프라인 비교에만 사용했으며 actor 입력이 아니다. 가속도 axis 미등록·0/invalid, 실제 fit/actor/생리·물리 동기 미검증 유지. 다음은 같은 코너 원본에서 카메라·yaw·world velocity의 부호 및 좌표 관계를 비교한다.



### 2026-11-02 — 좌측 벡터 계산 오류 정정

**2026-11-02 정정 — 좌측 벡터 계산 오류 및 코너 원본 재검증:** 직전 world velocity 분석의 `cross(forward,up)`을 좌측으로 해석한 것은 오류였다. 같은 기록의 실제 좌측 카메라 시선은 `cross(up,forward)`에 일치한다. 잘못된 분석 원본을 보존하고, 좌측 카메라 시선과 내적≥0.99인지 매 프레임 확인하는 별도 정정 분석을 실행했다 (`D:/CodexResearchRuns/flybrain-corner-axis-corrected-20261102-02/`, `D:/CodexResearchRuns/flybrain-world-accel-corrected-20261102-02/`). 코너 두 기록 262/345프레임에서 native accG index0과 올바른 좌측 투영의 상관은 +0.996/+0.997이며, speed×native yaw 근사와도 +0.972/+0.987로 같은 방향이다. 앞서 보고한 좌우 부호 충돌은 본 분석의 좌표계 오류로 철회한다. 세 번째 프레임 위치를 쓰는 사후 yaw 대조는 native yaw와 카메라 방위 변화가 같은 부호임을 보였으나 시간/카메라 basis 조건이 남는다 (`D:/CodexResearchRuns/flybrain-corner-camera-yaw-sign-20261102-01/`). 정정 계산은 결과 확인 후 수행한 탐색적 분석이며 기존 회로/계약을 바꾸지 않았다. 기록된 3초 직선의 좌측 변화는 작고, 코너는 같은 차/트랙의 상관된 프레임이다. 경사·중력·미끄러짐·물리 시각·생리는 미검증이므로 가속도 입력은 0/invalid, 실제 fit/actor 불허를 유지한다.



### 2026-11-02 — 합성 가속도 단일 회로 경계·학습

**2026-11-02 합성 가속도 단일 회로 경계·학습:** 기존 `CenteredSpeedBodyFeedbackAdapter`의 base projection을 보존하고 21차원 몸 입력의 가속도 3채널을 별도 마스크·초기 0 projection으로 같은 비행 회로 감각 포트에 더하는 실험 후보 `src/flybrain_drive/synthetic_acceleration_feedback.py`를 추가했다. 기존 출력의 bitwise 동일성, 결측 축 차단, 범위/비유한 검사 포함 관련 13개 테스트 통과. 실제 native 가속도는 여전히 invalid이며 후보는 생리 측정 회로가 아닌 공학 가설이다. 합성 장면 2개×좌우, 전후/좌우 ±3m/s²의 가설 안정화 목표로 새 projection만 2회 학습 (`D:/CodexResearchRuns/flybrain-synthetic-accel-fit-20261102-01/`). 학습 손실은 각 업데이트 전 0.5→1.713으로 증가했다. 보류 합성 장면의 16조건 모두 사전 지정 목표 부호·크기 10% 이상이었고 결측 출력 차 0으로 좁은 사전 screen은 통과했다. 그러나 사후 교차축 검사 (`D:/CodexResearchRuns/flybrain-synthetic-accel-crosstalk-20261102-01/`)에서 전후 가속도에 의도치 않은 조향 intent 약 ±0.00694가 발생했고 주목표 전진 intent는 약 ±0.000337이었다. 사전 screen이 교차축을 제한하지 못해 이 통과는 후보 품질 증명이 아니다. 후보 미채택·실제 actor/fit 불허, 생리·실제 시각 동기·폐루프 미검증. 다음은 교차축 억제를 사전에 포함하고 실제 물리/생리 근거와 합성 폐루프 개선을 검증한다.



### 2026-11-02 — 교차축 제한 합성 가속도 학습

**2026-11-02 교차축 제한 합성 가속도 학습: 기준 실패.** 기존 단일 회로의 합성 가속도 projection만 GPU에서 2회 갱신하고, 기존 0.02 대신 학습률 0.001, 보류 16조건의 목표 축 부호/크기와 다른 출력축 25% 상한을 사전에 지정했다 (`D:/CodexResearchRuns/flybrain-accel-decoupled-fit-20261102-01/`, 계획 `D:/CodexResearchRuns/flybrain-accel-decoupled-plan-20261102-01/`). 업데이트 전 손실 0.5→0.419; 보류 16조건 중 8개만 통과했다. 좌우 가속도에 대한 조향 intent는 평균 ±0.00103으로 목표 ±0.003의 약 34%였고, 전후 가속도의 불필요한 조향은 ±0.000274로 사전 교차축 상한 이내였으나 전진 intent 변화는 ±0.0000183으로 목표 ±0.002의 약 0.9%에 그쳐 전후 8조건이 실패했다. 결측 시 출력 차 0, base body projection 불변. 기존 GPU/CPU·학습률도 달라 이전 후보와의 차이를 단일 원인으로 단정하지 않는다. 보류 장면은 합성이며 이전 실험에서 이미 본 장면이다. 실패 후보 폐루프/배포 불허; 실제 가속도 마스크, 물리 시각, 생리·실제 AC 검증은 유지/미완료다. 다음은 전후 신호가 비행 회로 전진 출력에 미치는 기울기와 조향 교차 영향을 원인별 분리해 연결을 수정한다.



### 2026-11-02 — 가속도 감각 경로와 합성 폐루프

**2026-11-02 가속도 경로 민감도·동시 입력·합성 폐루프:** 가속도 projection 0에서 같은 비행 회로의 전진/조향 출력에 대한 국소 미분을 계산한 결과 첫 합성 장면의 조향 gradient norm은 전진의 41.5배였지만 두 gradient cosine은 -0.0268로 거의 직교했다 (`D:/CodexResearchRuns/flybrain-accel-port-sensitivity-20261102-01/`). 첫 장면에서 전진 gradient를 조향 gradient에 직교화한 방향의 최대 가중치 크기를 0.15로 적용해 실제 회로를 재생했다 (`D:/CodexResearchRuns/flybrain-accel-orthogonal-eval-20261102-01/`). 기존에 본 보류 합성 장면 8조건에서 전후 가속도 반응은 전진 intent 약 ±0.00066, 조향 약 ±0.00018, 결측 차 0으로 사전 screen 통과. 이를 이전 미채택 후보의 좌우 가속도 열과 같은 단일 회로에 합쳐 전후×좌우 16조건을 재생한 결과 사전 기준 통과 (`D:/CodexResearchRuns/flybrain-accel-combined-eval-20261102-01/`); 비선형 상호작용과 결측도 범위 이내. 이것은 고정 합성 이력의 공학 목표이며 기존 방향은 사후 선택, 생리/실제 actor 적격성은 아니다.

새 합성 질감·자전거 차량의 3/6m/s×좌/직선/우×기준/후보/결측 18폐루프에서 x=16m 오차를 비교했다 (`D:/CodexResearchRuns/flybrain-synthetic-accel-loop-20261102-01/`). 기존 절대 중심 오차는 코너 약 0.221~0.289m였고 후보 변화는 -0.162~+0.132mm 범위; 사전 요구 코너 5mm 개선 미달로 실패다. 결측 대비 기준 차 최대 0.0000962mm. 후보 실제 배포 불허, 물리/생리/실제 AC/가속도 등록 미검증. 시각 선행·차체/하중/타이어·전진/감속/변속·셋업·라인/랩타임 목표는 유지하며, 다음은 왜 열린 이력의 반응이 폐루프 성능에 기여하지 않는지 차량-회로 상호작용과 속도 지표를 확인한다.



### 2026-11-02 — 가속도 폐루프 크기와 교란 검증

**2026-11-02 합성 가속도 경로의 약한 폐루프 효과·교란 검증:** 이전 x=16m 합성 궤적을 공통 x=8.5~15.5m에서 사후 비교했다 (`D:/CodexResearchRuns/flybrain-accel-loop-dynamics-20261102-01/`). 후보에 들어간 전후 가속도 절대값 중앙은 조건별 0.171~0.210m/s², 횡가속도는 0.0036~0.0578m/s²로 고정 이력 학습의 ±3m/s²보다 작았다. 같은 위치에서 기준 대비 최대 조향 차 2.33e-5~2.98e-5rad, 최대 속도 차 0.000189~0.000588m/s였으며 이 분포 차이는 작은 효과의 가능한 설명이지 유일한 원인 증명은 아니다.

새 질감의 slip-free 합성 직선에서 3/6m/s×조향 외란 -0.08/0/+0.08rad(0.5초)×기준/후보/가속도 결측 18조건을 비교했다 (`D:/CodexResearchRuns/flybrain-accel-disturbance-20261102-01/`, 사전 계획 `D:/CodexResearchRuns/flybrain-accel-disturbance-plan-20261102-01/`). 가속도는 직전 실제 적용 조향·가감속에서 생성했다. 외란 4조건의 기준 x=16m 중심선 절대 오차는 약 0.218~0.329m였고 후보는 각각 0.117~0.449mm 악화해 사전 5mm 개선 기준에 실패했다. 결측과 기준 차는 최대 0.000036mm 이내. 한 합성 질감·단순 차량·인위적 조향 외란이며 실제 AC/생리 검증이 아니다. 가속도 후보 미채택·actor/실제 fit 불허. 다음은 단순 신호 확대/튜닝을 반복하지 않고 시각 선행과 비행 유래 몸 제어의 인과 기여, 고유한 회로 연결 및 다양한 동역학/실제 센서 근거를 우선한다.



### 2026-11-02 — 전·좌우·후 시야 시간 변화의 합성 폐루프 기여

**2026-11-02 전/좌우/후 시야 시간 변화의 합성 폐루프 기여:** 새 질감 883901/883902 × 직선·좌/우 코너에서 동일 원래 단일 회로로 24개 합성 폐루프를 실행했다. 각 대조는 전방, 좌우, 후방 수용기 입력 중 하나의 그룹을 첫 프레임에 고정하고 나머지는 갱신하며 모든 시야는 유효로 표시한다 (`D:/CodexResearchRuns/flybrain-view-lesion-20261102-01/`, `D:/CodexResearchRuns/flybrain-view-lesion-analysis-20261102-01/`). x=16m에서 전방 고정은 네 코너의 절대 오차를 7.10~10.16mm 늘렸다(기준 오차 0.221~0.279m). 좌우 고정의 코너 오차 변화는 -0.55~+2.53mm이고 직선은 4.02~7.89mm 증가했다. 후방 고정 변화는 전체 -0.71~+1.07mm였다. x=10m 조향 최대 변화는 전방/좌우/후방 각각 0.000514/0.000254/0.0000878rad였다. 저장된 수용기 고정·초기 시야·차량 상태 연속성 차 0 (`D:/CodexResearchRuns/flybrain-view-lesion-integrity-20261102-01/`); 별도 신경망 재실행 검증은 아니다. 인위적 시야 고정과 폐루프에 따른 몸 상태 변화가 함께 존재하며 단일 합성 도로/두 질감 결과는 시야별 생리·실제 AC 정확도나 일반 성능 증명이 아니다. actor/실제 학습/조작 0. 전후좌우 겹눈·운동시각과 단일 비행 회로 목표는 유지하고, 다음은 좌우 시야의 변화를 회로가 약하게 사용하는 원인을 동일한 몸 상태·시야 교란으로 분리한다.



### 2026-11-02 — 동일 몸 이력의 시야 고정 반사실

**2026-11-02 동일 몸 이력의 전·좌우·후 시야 고정 반사실:** 직전 합성 폐루프의 기준 차체 위치·속도·yaw 몸 이력을 그대로 고정하고 원래 단일 회로를 별도 reset해 기준/전방·좌우·후방 수용기 첫 프레임 고정으로 각각 재생했다 (`D:/CodexResearchRuns/flybrain-fixed-body-view-20261102-01/`, 분석 `D:/CodexResearchRuns/flybrain-fixed-body-view-analysis-20261102-01/`). 두 질감×직선/양 코너, 각 81결정이다. 전/좌우/후 고정 시 조향 intent 절대차의 조건별 평균을 다시 평균하면 코너 전 x=9~11.5m에서 0.001087/0.000349/0.000116, 코너 후 x=13~16m에서 0.005554/0.002419/0.000548이다. 좌우를 함께 고정했으므로 좌·우 개별 기여는 분리하지 못한다. 몸 이력 차 0, 고정 수용기 차 0, 나머지 시야 장면 최대차 1.05e-7, 기준 원본 intent 재생 최대차 1.84e-7. 같은 몸·같은 다른 시야에서 좌우 시야 시간 변화가 이 회로의 출력에 영향을 준다는 합성 반사실 증거이며, 폐루프 성공·방향 정확성·실제 AC/생리 기능을 뜻하지 않는다. 첫 프레임 고정은 카메라 결측과 다르고 고정 입력에도 FlyVis 내부 상태는 시간에 따라 변한다. optimizer·actor·차량 제어 0; 미검증 실제 동기·가속도 마스크는 유지. 다음은 좌·우 시야와 운동/공간 경로를 독립적으로 개입해 시간적 운동 시각 기여를 분리한다.



### 2026-11-02 — 좌·우 시야 및 수용기/운동 특징 개입

**2026-11-02 동일 몸 이력에서 좌/우·수용기/운동 특징 개입:** 첫 실행 `D:/CodexResearchRuns/flybrain-side-view-path-20261102-01/`은 시야 인덱스 변수와 유효 마스크 이름 충돌로 `UnboundLocalError`, optimizer 0으로 실패했으며 원본을 보존했다. 변수 이름만 고친 재생 `D:/CodexResearchRuns/flybrain-side-view-path-20261102-02/`은 두 질감×직선/좌우 × 8조건의 기존 단일 회로를 같은 몸 이력으로 각각 reset해 재생했고 기준 출력 원본 차 0, 양쪽 동시 고정 이전 실험과 최대 차 1.83e-7이었다. 좌·우 각각 전체 시야 고정, 수용기 1채널만 고정, T4/T5 8채널만 고정한 계산의 6조건별 조향 intent 평균 절대차를 비교했다 (`D:/CodexResearchRuns/flybrain-side-view-path-summary-20261102-01/`). 코너 전 x=9~11.5m 좌/우 전체 시야는 0.000317/0.000131, 수용기만 0.000313/0.000129, 운동 특징만 0.0000239/0.0000335였다. 코너 후 x=13~16m 좌/우 전체 0.000979/0.001707, 수용기만 0.000948/0.001732, 운동 특징만 0.0000470/0.0000440였다. 양쪽 동시 효과는 개별 합과 완전히 같지 않으며 최대 조건별 평균 상호작용 0.000698이다. 이는 이 두 합성 질감·단일 도로·현재 checkpoint에서 수용기 교체 효과가 운동 특징 교체보다 큼을 보일 뿐 운동 시각 회로의 일반적 무용성이나 정상 생리를 입증하지 않는다. 수용기/운동 특징을 따로 합친 입력은 자연 신경 상태에 없을 수 있고 화면 고정은 센서 결측이 아니다. optimizer·actor·차량 제어 0; 실제 동기/생리/주행 미검증. 다음은 운동 특징의 방향·속도 자극과 회로 내 전달을 실제 보존 checkpoint 기준으로 교란하고 결측 시 분리를 검증한다.



**2026-11-02 좌우 운동 특징의 단일 회로 경로 추적:** 기존 합성 차체 이력과 보존 checkpoint를 사용한 두 질감×직선/좌우 코너×기준/좌·우 운동 특징 고정/좌·우 수용기 고정의 30개 재생에서, 운동 특징 입력·상대운동 projection·공간 projection·시각 포트·순환 상태·출력을 추적했다 (`D:/CodexResearchRuns/flybrain-side-motion-trace-20261102-01/`, 사후 감사 `D:/CodexResearchRuns/flybrain-side-motion-trace-audit-20261102-01/`). 모든 30개 intent는 앞선 독립 저장 재생과 최대 차 0이다. 코너 전 x=9~11.5m의 좌/우 운동 특징 개입에서 상대운동 projection 변화 RMS의 6조건 평균은 0.002016/0.005113, 시각 포트는 0.000118/0.000254, intent 전체는 0.0000189/0.0000282였다. 수용기 개입에서 상대운동 projection 변화는 0이고 운동 특징 개입에서 공간 projection 변화는 0이다. 신호는 해당 projection과 통합 회로 출력까지 존재하지만 단위와 차원이 달라 단계별 크기로 전달률이나 유일한 병목을 추론할 수 없다. 고정/갱신 혼합은 인공 입력이며 방향·속도를 통제한 자극, 실제 생리·차량 성능 검증이 아니다. optimizer·actor·차량 제어 0. 다음은 운동 방향·속도를 통제한 자극과 회로 경로별 인과 개입을 사용하고, 검증된 신호만 학습·실차 시뮬레이터로 연결한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 상대운동 projection 차단 인과 실험:** 이전의 동일 합성 장면·몸 이력에서 기준/좌·우 운동 특징 고정과 상대운동 projection 유지/출력 0을 교차한 36개 재생을 수행했다 (`scripts/ablate_opponent_projection_fixed_history.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-opponent-ablation-plan-20261102-01/`, 실행 `D:/CodexResearchRuns/flybrain-opponent-ablation-20261102-01/`, 분석 `D:/CodexResearchRuns/flybrain-opponent-ablation-analysis-20261102-01/`). 차단 안 한 18개 출력의 이전 추적 대비 최대 차 0, 차단한 projection 출력 최대 0이다. 두 질감×세 도로 조건의 조향 intent 평균 절대 변화는 코너 전 x=9~11.5m에서 좌/우 운동 특징 고정 시 정상 0.00002385/0.00003351, 차단 시 0.00000430/0.00000126이다. 코너 후 x=13~16m에서는 정상 0.00004697/0.00004401, 차단 시 0.00002711/0.00000360이다. 시간별 차이의 차이 절대값 평균은 전 0.00002127/0.00003309, 후 0.00002371/0.00004572다. 상대운동 경로가 이 합성 장면의 운동 특징 민감도에 인과적으로 기여하지만, 경로 차단 자체가 기준 조향을 바꾸며(전 평균 절대 0.001243) 비선형 순환 효과와 인위적 혼합 입력이 포함된다. 뉴런별 생리 기능·방향 정확성·현실 주행 성능의 증명이 아니며 기존 장면의 사후 실험이다. optimizer·actor·차량 제어 0, 실제 동기와 가속도는 미검증. 다음은 방향·속도 통제 자극과 통합 회로 응답을 확인하고, 기능 결함을 수정해 독립 장면과 실제 센서에서 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 현재 통합 회로의 통제 시각 yaw 진단 — 부호 불일치:** 과거 평가에 사용된 합성 질감 879604의 시각 yaw 0/±0.18rad/s와 질감 879605의 0/±0.13rad/s를 현재 단일 회로에 넣고 몸 속도 5m/s·yaw 0 등은 고정하여 각각 독립 recurrent reset으로 재생했다 (`scripts/evaluate_unified_controlled_yaw.py`, 계획 `D:/CodexResearchRuns/flybrain-unified-yaw-test-plan-20261102-02/`, 결과 `D:/CodexResearchRuns/flybrain-unified-yaw-test-20261102-02/`). 두 신호는 시각과 몸이 물리적으로 일치하는 주행이 아니다. 마지막 5결정의 조향 intent 평균에서 -yaw와 +yaw의 차는 각각 +0.005538과 -0.010909로 부호가 반대였다. 상대운동 projection 차단 시에도 +0.002982와 -0.012884로 반대였다. 자극 전 처음 11결정의 양 방향 출력 최대차는 각각 9.15e-8과 0이고 입력 retina는 동일하다. 두 질감과 속도가 동시에 달라 원인이 질감인지 속도인지 다른 상호작용인지 분리할 수 없다. 해당 실험은 이미 본 합성 장면의 사후 진단이며 생리·독립 일반화·실제 주행 검증은 아니다. 현재 통합 후보의 방향 일관성은 이 조건들에서 성립하지 않으므로 actor 적격으로 인정하지 않는다. 첫 실행 `D:/CodexResearchRuns/flybrain-unified-yaw-test-20261102-01/`은 CPU/GPU 입력 장치 불일치로 조건 재생 전 실패했고 원본 보존 후 수정해 02를 실행했다. optimizer·차량 제어 0. 다음은 같은 질감에 양 속도를 교차하고 상대운동·공간 경로를 분리해 부호 역전 원인을 규명한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 동일 질감·속도 교차 및 시각 경로 차단:** 이전 학습에 사용한 합성 질감 879601/879602 각각에서 시각 yaw ±0.13/±0.23rad/s를 교차하고, 같은 단일 회로의 상대운동 projection과 공간 contrast projection을 각각 유지/0 차단하는 48회 이력(4 자극원×4 경로 조건×3 방향)을 재생했다 (`scripts/evaluate_unified_yaw_factorial.py`, 계획 `D:/CodexResearchRuns/flybrain-yaw-factorial-plan-20261102-01/plan.json`, 결과 `D:/CodexResearchRuns/flybrain-yaw-factorial-20261102-01/`; 저장 출력 사후 재계산 `D:/CodexResearchRuns/flybrain-yaw-factorial-audit-20261102-01/`). 마지막 5결정의 -yaw와 +yaw 조향 intent 차는 원래 회로에서 879601의 0.13/0.23에 -0.001262/-0.000289, 879602에 +0.005108/+0.009572였다. 상대운동만 차단하면 각각 -0.005450/-0.008789, +0.001301/+0.001111이고, 공간만 차단하면 +0.004982/+0.009641, +0.004153/+0.009191로 네 조건 모두 양수였다. 두 경로 모두 차단 시에도 +0.001056/+0.001365, +0.000302/+0.000748이다. 따라서 이 제한된 조건의 질감 간 부호 차이에 공간 경로가 인과적으로 관여한다. 공간 경로는 코너 시각에도 기여하므로 이 결과만으로 제거하거나 일반 주행 개선을 주장하지 않는다. 기존 학습 질감·인위적 차단·몸 yaw 고정이며 실제 생리/차량 동기·독립 일반화가 아니다. 원본 16행 수치 재계산 차 0, 자극 전 양 방향 출력 최대차 1.83e-7; 재계산은 회로 독립 재실행이 아니다. optimizer·actor·차량 제어 0. 다음은 공간 경로의 질감 의존성을 줄이되 코너 선행 정보를 보존하는 구조/학습을 설계하고 독립 질감·실제 데이터에서 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 코너 특징 보존 가중치 해석적 수정 실패:** 합성 회전 ±0.13/±0.23rad/s의 2개 학습 질감과 합성 도로 질감 883901의 좌·우 코너 대 직선 공간 특징 4개 평균 차를 이용해, 같은 단일 회로의 공간 projection에서 도로 차이를 유지하면서 회전 특징 성분을 제거하는 최소 변화 직교 투영 후보를 계산했다 (`scripts/fit_spatial_yaw_invariance.py`, 사전 상한 최대 가중치 변화 0.15, 계획 `D:/CodexResearchRuns/flybrain-spatial-invariance-fit-plan-20261102-02/plan.json`). 계산상 도로 특징 제약 차 최대 4.67e-16이고 공간 회전 projection norm은 6.816→4.240이지만 최대 가중치 변화 32.38로 사전 상한을 크게 넘어 실패·후보 저장/적용/actor 배포 0 (`D:/CodexResearchRuns/flybrain-spatial-invariance-fit-20261102-02/`). 01 실행도 같은 제약 실패였으나 수치를 기록하지 못해 실패 원본을 보존하고 같은 상한에서 기록만 추가해 02를 재실행했다. 이는 지정한 선형 직교 투영이 실패했다는 결과이지 다른 비선형/제약 학습 불가능의 증명이 아니다. 입력 특징 수준의 4개 평균 제약으로 코너 예측이나 폐루프가 보존된다고 주장할 수 없다. 기존 학습 질감·인공 yaw·고정 몸, 실제 생리/주행·물리 시각 미검증, optimizer 0이다. 다음은 같은 회로 안에서 코너 보존과 회전 민감도를 함께 제한하는 정규화/제약 학습을 별도 후보로 설계하고, 기존과 보류된 질감·코너 이력·실제 데이터에 통과한 후에만 채택한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 가중치 제한 공간 회전 후보 적합·순환 회로 평가 실패:** 이전 직교 투영의 최대 변화 32.38을 피하기 위해 공간 projection 변화량을 요소별 0.15로 제한하고, 합성 회전 특징 억제와 도로 특징 변화 벌점(10), 정규화(0.01)를 고정한 최소제곱 후보를 계산했다 (`scripts/fit_spatial_bounded_yaw.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-spatial-bounded-fit-plan-20261102-01/`, 후보 `D:/CodexResearchRuns/flybrain-spatial-bounded-fit-20261102-01/`). 최대 변화 0.15, 입력 단계 회전 projection norm 6.8159→6.7501, 도로 평균 특징 변화 최대 0.00581이었다. 이전에 본 학습/보류 합성 회전 6조건 및 두 도로 질감×직선/코너의 고정 몸 이력을 동일 단일 순환 회로에서 기준/후보 재생했다 (`scripts/evaluate_bounded_spatial_candidate.py`, 계획 `D:/CodexResearchRuns/flybrain-spatial-bounded-eval-plan-20261102-02/`, 실행 `D:/CodexResearchRuns/flybrain-spatial-bounded-eval-20261102-02/`, 감사 `D:/CodexResearchRuns/flybrain-spatial-bounded-eval-audit-20261102-01/`). 원래 출력은 이전 저장 재생과 최대차 0. 도로 12개 구간의 후보 조향 평균 절대 변화는 사전 상한 0.0002 이내, 최댓값 0.0000469였다. 하지만 회전 부호는 사전 6/6 기준 대비 3/6만 맞아 실패했다. 특히 기존 보류 질감 879605는 -0.010858로 여전히 역전한다. 후보 미채택·actor/실제 학습 불허다. 평가 스크립트 첫 01은 실행 전 들여쓰기 오류로 run이 생성되지 않아 구문 수정 후 02를 실행했다. 최소제곱은 실제 차량 학습이나 생리 신경망 재학습이 아니며 기존에 본 합성 영상·고정 몸 이력이고 폐루프/실제 동기/생리 미검증이다. 다음은 소규모 가중치 제한만으로 부호 문제를 해결하지 못한 사실을 반영해 공간·운동 정보 통합 구조와 원천 방향 선택성을 재검토하고, 코너 보존과 실제 센서 검증을 함께 설계한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 운동 특징으로 공간 회전 영향 보정하는 합성 탐침 실패:** 같은 단일 회로의 공간 시각 logits에서 합성 yaw로 생긴 차이를 운동 특징으로 예측하는 ridge 선형 보정기를 2개 기존 학습 질감의 360개 샘플로 적합했다 (`scripts/fit_motion_compensation_probe.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-motion-compensation-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-motion-compensation-probe-20261102-01/`). 입력은 에피소드 시작 대비 각 시야 운동 특징이며, 시간대가 같은 zero-yaw 장면은 오프라인 목표에만 사용했다. 이전에 살펴본 보류 질감 2개×양 방향의 보정 후/전 MSE 비율은 31.76/0.662/0.236/1.329로 사전 4개 모두 <0.8 기준에서 2/4 실패다. 별도 보류 도로 고정 이력에서 보정 logits RMS/기존 공간 logits RMS는 좌/직선/우 0.196/1.199/0.155로 직선이 사전 0.25 상한을 초과했다. 목표 자체가 아주 작은 조건에서는 상대 오차 비율이 커지므로 절대 MSE도 원본에 남겼다. 탐침 가중치는 회로에 적용하지 않았고 actor/차량 제어 0, 최적화는 해석적 ridge로 신경망 학습 아님. 이미 본 합성 질감·인위적 정지 기준·시각/몸 불일치로 실제 생리/차량·신규 환경이나 코너 보존을 입증하지 않는다. 다음은 시작 프레임 기준의 취약성과 000 모델의 T4d 불일치를 해결할 방향 선택성 및 몸-시각 시간 정합을 우선 재검토하고, 교차 장면에서 코너를 보존하는 단일 회로를 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 공개 FlyVis 003~006 사전 지정 검증:** 기존 000~002에 이어 연속 4개 ID를 결과 전에 지정하고 저자 flash 2반경×2밝기를 같은 GPU 조건에서 검사했다 (계획 `D:/CodexResearchRuns/flybrain-flash-prescreen-plan-20261102-01/`, 원본 `D:/CodexResearchRuns/flybrain-flash-prescreen-003-20261102-01/`부터 006, 분석 `D:/CodexResearchRuns/flybrain-flash-prescreen-analysis-20261102-01/`). 사전 기준인 T4 네 종류 ON 및 T5 네 종류 OFF가 두 반경 모두 부호 일치한 모델은 006만이었다. 003은 T5a/T5c, 004는 T4a/T5a/T5d, 005는 T5a/T5b/T5c에서 적어도 한 반경이 반대였다. 저자 flash 분석과 독립 numpy는 최대 차 0, 기존 001 분석 재현 최대차 0이다. 사전 계획대로 006에 저자 이동 경계 12방향×6속도×2명암 144조건을 추가 실행·분석했다 (`D:/CodexResearchRuns/flybrain-flash-prescreen-006-edge-20261102-01/`, `D:/CodexResearchRuns/flybrain-flash-prescreen-006-full-analysis-20261102-01/`). 006의 T4d는 기대 명암 6/6·최대 방향 오차 16.4°였지만 T5b는 기대 명암 0/6·최대 방향 오차 179.3°, T5a는 명암 6/6이나 최대 방향 오차 175.9°, T5d 명암 3/6·최대 오차 90°였다. 낮은 DSI 조건의 방향 각도는 불안정할 수 있으므로 원본별 DSI와 함께 해석한다. 006 역시 정상 방향 시각 또는 000 대체로 인증하지 않았고 가중치 교체·optimizer·actor·차량 조작 0이다. 제한된 공개 모델·인공 자극 검증이며 실제 생리/운전 일반화는 미검증이다. 다음은 원천 생리 불일치의 훈련/모델 원인과 사용자 목표에 맞는 통합 대안을 검토하고 검증된 모델만 회로에 연결한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 실험용 하중 입력 경계 구현:** 기존 실제 기록에서 하중 projection 4열이 0인 문제에 대응해 `SyntheticLoadBodyFeedbackAdapter`를 추가하고, 기존 동일 `UnifiedFlightSensorimotor`가 명시적으로 이 실험용 경계를 받을 수 있게 했다 (`src/flybrain_drive/synthetic_load_feedback.py`, `src/flybrain_drive/unified_flight_sensorimotor.py`). 네 native 하중 fraction 전체가 유효할 때만 각 값의 균등 0.25 대비 차이를 별도 0 초기화 projection으로 기존 몸 감각 포트에 더한다. 부분 결측 시 새 하중 경로는 0, 완전 유효한 fraction이 음수·1초과 또는 합계 1과 1e-4 초과 차이면 오류다. 기존 속도/몸 및 합성 가속도 경로를 보존하며 네 하중이 모두 유효하고 신규 가중치가 0일 때 부모 출력은 bitwise 동일하다(부분 결측의 후속 수정은 최신 기록 참조). 바퀴의 좌우/전후 순서는 해석하지 않았으며 이 경계는 공학 가설이지 파리 수용기 생리나 실제 actor가 아니다. 단위·마스크·기존 출력 보존·native 순서 구분·gradient 및 같은 단일 회로 결합을 포함한 32개 관련 테스트 통과. 신규 하중 projection은 아직 0이고 학습/실제 감각 동기·하중 방향/물리/주행 검증은 수행하지 않았다. 다음은 가상 차량에서 하중의 원인과 좌우/전후 wheel registration을 근거로 고정하고, 교차축·결측·독립 폐루프 기준을 사전 지정해 새 경로만 학습한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 하중 연구 이력 정정 및 실제 패킷 탐색:** 직전 설명은 기존 하중 실험을 충분히 반영하지 못했다. 별도의 기존 합성 하중 후보는 이미 6회 학습·실제 이력 대조·합성 폐루프를 거쳤으며, 폐루프 중심 오차 변화는 -0.065~+0.012mm로 유용한 개선을 증명하지 못해 미채택이었다 (`D:/CodexResearchRuns/flybrain-synthetic-load-fit-20261102-01/`, `D:/CodexResearchRuns/flybrain-synthetic-load-loop-20261102-01/`). 직전에 만든 별도 0 초기화 경계는 이를 대체하는 학습 완료 경로가 아니다. AC 공유 메모리 공개 원문 재게시본은 `wheelLoad[4]` 필드는 명시하나 인덱스 순서는 명시하지 않는다 ([공유 메모리 참조](https://www.assettocorsamods.net/threads/doc-shared-memory-reference.58/)). 저장된 native 패킷 3개 기록의 전체 원본 SHA와 개별 패킷 해시를 확인하고 0/1/2/3=FL/FR/RL/RR 가설에서 우-좌와 전-후 하중 비율을 같은 패킷의 미등록 native 가속도 축 0/2와 비교했다 (`scripts/audit_native_wheel_load_order.py`, 계획 `D:/CodexResearchRuns/flybrain-native-load-order-plan-20261102-03/`, 결과 `D:/CodexResearchRuns/flybrain-native-load-order-audit-20261102-03/`). 원래 사전 지정 속도 ≥5m/s에는 충분한 패킷이 없어 02는 실패했고, 0.5m/s로 내린 03은 사후 탐색이다. 세 기록의 상관은 우-좌 대비 축0 각각 +0.099/+0.041/+0.692, 전-후 대비 축2 각각 -0.878/-0.878/-0.859였다. 각 약 3초·1095/1094/94개 중복 제거 패킷은 시간 상관이 높고 같은 차량/환경이며, 가속 축·중력·물리 시각·좌우 독립 접지 기준이 미검증이다. 전후 가설과 일치하는 경향은 있으나 바퀴 순서·부호를 등록하거나 실제 학습에 쓰기에 부족하며 좌우는 일관되지 않다. 01은 긴 teacher prefix를 짧은 decoder에 전달해 실패했고 수정 후 02/03을 실행했다. actor/optimizer/차량 제어 0. 다음은 독립적인 좌·우 및 가감속 자극과 바퀴 위치의 별도 기준을 확보해 순서를 검증하고, 기존 실패 후보와 새 경계를 비교해 중복 학습을 피한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 실험용 하중 경계 부분 결측 수정:** 직전 `SyntheticLoadBodyFeedbackAdapter`는 새 load projection만 네 값 중 하나라도 결측일 때 차단했고, 기존 base projection의 하중 열이 0이 아닌 checkpoint에서는 나머지 부분 하중이 새어 들어갈 수 있었다. 기존 checkpoint의 하중 열이 0인 경우에는 드러나지 않는 결함이다. 이제 한 바퀴라도 결측이면 base와 새 projection의 하중값 4개를 모두 0으로 차단하면서 원래 각 바퀴 validity 표시·다른 몸 감각·가속도 경로는 유지한다 (`src/flybrain_drive/synthetic_load_feedback.py`). 네 값 모두 유효하고 새 projection이 0이면 이전 부모 출력은 bitwise 동일하지만, 일부 결측에 기존 하중 열이 0이 아닌 경우에는 안전한 차단을 위해 부모와 출력이 달라질 수 있다. 무작위 비영(非零) base 가중치에서 15개 불완전 mask 각각 하중 수치를 바꿔도 출력이 불변임을 추가했고 전체 관련 47개 테스트 통과 (`tests/test_synthetic_load_feedback.py`). 새 하중 가중치는 미학습·실제 바퀴 순서 및 몸 시각 동기 미검증, actor/차량 제어 0. 기존의 별도 하중 후보는 이미 합성 폐루프에서 유효한 개선을 입증하지 못했다. 다음은 좌우·전후 바퀴 순서의 독립 근거와 기존 후보/새 경계의 인과 성능을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 정차 중 SDK/native 하중 순서의 읽기 전용 대조:** 실행 중인 AC의 기존 파일 텔레메트리 `ac.getCarState(Load)`와 native `wheelLoad[4]`를 같은 host에서 20회 비제어 수집하고 24개 인덱스 순열을 비교했다 (`scripts/audit_live_sdk_native_wheel_order.py`, 계획 `D:/CodexResearchRuns/flybrain-sdk-native-wheel-plan-20261102-02/`, 결과 `D:/CodexResearchRuns/flybrain-sdk-native-wheel-audit-20261102-02/`). SDK 게시 코드는 네 반환값에 FL/FR/RL/RR 이름을 붙이며, native 원래 순서 [0,1,2,3]가 최선으로 평균 절대차 0.144, 최대차 0.799, 두 번째 순열 평균 11.205다(원본 값 단위). 서로 다른 publisher step 20개, 파일 mtime 기준 최대 나이 36.6ms, 최대 native 속도 0.000118m/s다. 01은 서로 다른 Python 실행 환경의 perf_counter 기준점을 혼용해 유효 표본 0으로 실패했으며 02에서는 파일 갱신 시각과 증가 step을 사용했다. 본 대조는 publisher가 붙인 이름과 native 배열의 일치만 보인다. 게시 파일 해시는 현재 디스크 파일의 해시이고 실행 중 로드된 코드의 독립 증명은 아니며 두 경로가 같은 내부 데이터를 사용할 수 있다. 실제 바퀴 접지 좌표·시뮬레이션 tick·동적 조건을 독립 확인하지 못해 바퀴 순서 등록, 물리 동기 및 실제 학습 적격성은 여전히 거부한다. 옵티마이저·배포·차량 조작 0. 다음은 AC SDK의 타이어 접지 좌표/차체 기준 방향을 별도 읽기 진단으로 수집해 이름과 실제 위치를 검증하고, 기존 하중 후보의 동적 유용성을 재평가한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 정차 CSP 바퀴 위치·하중 대조:** 기존 설치 Lua 원본에서 진단 코드만 추가한 후보와 읽기 전용 모듈을 만들고 (`integrations/ac_csp/candidates/FlyBrainCamera-wheel-geometry-diagnostic-v1.lua`, `integrations/ac_csp/candidates/WheelGeometryDiagnostic.lua`), 차체 forward/up과 바퀴 좌표·outside 벡터로 원래 순서를 분류하는 독립 분석을 작성했다 (`src/flybrain_drive/wheel_geometry_diagnostic.py`). 모의 Lua·기하·기존 센서 관련 28개 테스트 통과, 후보에서 진단 부분을 제거하면 설치 원본과 텍스트 일치 (`D:/CodexResearchRuns/flybrain-wheel-geometry-candidate-review-20261102-01/`). 세션 API로 정차·오프라인·브레이크 조건에서 pause하고 hash 확인 후 임시 후보를 배치해 1프레임을 수집했다 (계획 `D:/CodexResearchRuns/flybrain-wheel-geometry-live-plan-20261102-01/`, 캡처 `D:/CodexResearchRuns/flybrain-wheel-geometry-live-20261102-01/`). 앞/뒤 중심 기준 좌표는 약 +1.255/+1.255/-1.255/-1.255m, 좌측 기준 좌표는 +0.810/-0.810/+0.833/-0.833m로 CSP 인덱스는 FL/FR/RL/RR 분류였고 outside 방향 정합은 0.9986 이상이다. CSP 하중과 native 동순서의 평균 절대차 0.143·최대차 0.561, 다음 순열 평균 11.818이었다. 캡처 전후 native packet ID는 같았지만 CSP timestamp와 native tick의 동시성은 검증되지 않았다. 한 차·한 트랙·정차 1프레임의 CSP geometry/하중 및 두 API 관계가 정확하다는 독립적 보장은 없으므로 바퀴 순서·물리 시각 등록이나 actor 적격을 아직 선언하지 않는다. 원래 카메라 source SHA 78863113…를 복구하고 자동 resume으로 기존 비일시정지 상태를 복원했다 (`D:/CodexResearchRuns/flybrain-wheel-geometry-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-wheel-geometry-resume-20261102-01/`). 데스크톱 입력·조향/가감속·optimizer 0. 다음은 다른 차량/재시작 및 독립 SDK 접지 위치·동적 좌우/가감속을 포함한 재현과 native 일치 검증 후에만 순서를 등록한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 제한 AI 이동 중 바퀴 기하·하중 대조:** 현재 설치된 원본 Lua에 기존 airflow 제한 진단과 바퀴 위치·하중 수집만 덧붙인 임시 후보를 만들고 원본 재구성 일치·Lua 및 관련 13개 테스트를 확인했다 (`integrations/ac_csp/candidates/FlyBrainCamera-wheel-motion-diagnostic-v1.lua`, `scripts/record_ac_wheel_timed.py`, 계획 `D:/CodexResearchRuns/flybrain-wheel-motion-plan-20261102-01/`). 검증된 lifecycle API와 기존 owned AI motion3 진단으로 오프라인에서 잠시 이동·정지했으며, 단일 회로 actor/학습/데스크톱 입력은 사용하지 않았다. 원본 162프레임, native 반복 복사 5221개, 화면 호출 648/648 host bracket을 보존했다 (`D:/CodexResearchRuns/flybrain-wheel-motion-live-20261102-01/`). 사후 해시 대조 후 CSP 진단 시작 이전의 host native 수신을 보수적으로 선택한 결과 162/162프레임이 시간상 짝지어졌고 최대 host age 6.085ms였다 (`scripts/audit_wheel_motion_geometry.py`, 결과 `D:/CodexResearchRuns/flybrain-wheel-motion-analysis-20261102-01/`). 162프레임 모두 바퀴 위치의 기하 분류가 FL/FR/RL/RR이었고 속도 ≥1m/s 프레임은 39개(최대 4.519m/s)다. CSP·native 하중 배열의 24개 순열 중 원래 [0,1,2,3]이 평균 절대차 33.03으로 가장 낮았고 다음 순열은 52.48이었다. 이동 39프레임만 보면 각각 72.12/117.60이며 동일 순서 차이 95백분위 311.83, 최대 467.48이었다. 이는 순서 일치의 추가 근거이나 값 차이가 커질 수 있고 host 수신은 물리 tick이나 노출 동기 증명이 아니다. 같은 차·같은 트랙의 짧은 AI 주행과 사후 선택된 분석이라 독립 생리·일반화·실제 학습 적격성은 없다. 원래 Lua SHA 78863113…와 시작 전 비일시정지 상태를 복원했다 (`D:/CodexResearchRuns/flybrain-wheel-motion-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-wheel-motion-resume-20261102-01/`). 다음은 물리 tick/하중 변화 차이를 독립적으로 해석하고 좌우·전후 자극과 다른 차량에 대한 순서 검증을 진행한 후, 기존 실패 하중 후보의 폐루프 효용을 재검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 이동 중 하중 host 시차 탐색:** 직전 162프레임·5221 native 복사 원본의 해시와 각 패킷/프레임 해시를 검증하고, 속도 ≥1m/s인 39프레임에서 CSP 캡처 중간시각을 host 시계 경계의 중간으로 환산한 뒤 native 복사 시각을 -40~+40ms로 10ms 간격 비교했다 (`scripts/audit_wheel_host_lag.py`, 계획 `D:/CodexResearchRuns/flybrain-wheel-host-lag-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-wheel-host-lag-audit-20261102-01/`). 앞 19프레임에서 네 하중값 평균 절대차가 최소인 -20ms를 선택하자 평균 차는 무시차 106.80→22.80, 뒤 20프레임에서도 53.41→9.00이었다(원본 하중 단위). 같은 짧은 주행을 양분한 시간상 상관된 구간의 사후 탐색이며 CSP/source clock과 native 수신에 내재한 지연, 정수 프레임 및 smoothing 등이 섞일 수 있다. 일부 탐색 시차는 미래 패킷을 사용하므로 actor 입력이 아니며 -20ms를 고정 물리 tick 보정이나 동기 증거로 등록하지 않는다. 다른 속도·방향·차량의 독립 반복 및 원천 physics timestamp 확인 필요. 원본, 후보, 실패와 센서 계약을 보존했고 optimizer·추가 차량 제어·배포 0. 다음은 이 시차 가설을 사전 고정한 별도 기록에서 평가하고, 실제 물리 시각/바퀴 순서 등록 후 하중 감각의 폐루프 기여를 확인한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 하중 host -20ms 가설 별도 기록 재현:** 앞선 기록에서 선택한 -20ms를 새 데이터 전에 고정하고, 동일 차·트랙에서 기존 제한 AI motion3를 별도 실행했다 (사전 계획 `D:/CodexResearchRuns/flybrain-wheel-host-lag-repeat-plan-20261102-01/`, 원본 `D:/CodexResearchRuns/flybrain-wheel-lag-repeat-live-20261102-01/`). ≥1m/s인 38프레임을 순서대로 19/19로 나누어 무시차 대비 -20ms의 네 하중 평균 절대차를 비교했다. 전반은 20.690→4.179, 후반은 54.448→6.712로 각각 원래의 20.2%/12.3%이며 사전 지정 80% 이하 화면 진단 기준을 두 구간 모두 만족한다 (`D:/CodexResearchRuns/flybrain-wheel-lag-repeat-analysis-20261102-01/`, `D:/CodexResearchRuns/flybrain-wheel-lag-repeat-screen-20261102-01/`). 비교 기준은 host 수신시각이며 같은 차·트랙의 인접한 짧은 AI 주행과 시계 단위율 가정에 제한된다. 원천 physics tick/노출 시각이나 고정 지연이 검증됐다는 뜻이 아니므로 -20ms를 actor 정합에 적용하거나 실제 학습을 허용하지 않는다. 원래 카메라 Lua와 비일시정지 상태를 복구했다 (`D:/CodexResearchRuns/flybrain-wheel-lag-repeat-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-wheel-lag-repeat-resume-20261102-01/`). 파리 actor·optimizer·데스크톱 입력 0, 게임 제한 AI 진단만 수행했다. 다음은 독립 차/주행 조건과 source physics timestamp를 검증하고, 하중 순서·시간 계약이 충분해지면 기존 단일 회로의 하중 경로 성능을 평가한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 CSP 물리 상태 시각의 상대 변화:** 설치된 CSP SDK는 `car.timestamp`를 마지막 물리 상태 기록 시각(ms, `sim.time`과 같은 기준)으로 설명한다. 이전 두 AI 진단의 모든 162/178프레임 원본 해시를 확인하고 이 값과 진단 호출 `os.preciseClock`의 진행을 비교했다 (`scripts/audit_csp_physics_timestamp_stability.py`, 계획 `D:/CodexResearchRuns/flybrain-csp-physics-time-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-csp-physics-time-audit-20261102-01/`). 두 timestamp는 프레임별 증가하고 시간 진행 기울기는 각각 1.000091/1.000030이었지만, 두 시각 차이의 범위는 38.26/37.86ms, 연속 프레임 간 차이 변화 절댓값의 95백분위는 4.22/3.80ms였다. 시계 원점·호출 타이밍·상태 기록 지연이 섞이므로 이것이 센서 지연 자체를 뜻하지 않는다. native packet과 동일 물리 tick을 식별할 증거는 없고 정적 -20ms를 실제 동기값으로 적용하지 않는다. 같은 차·트랙의 기존 기록 사후 분석, optimizer·actor·차량 제어 0. 다음은 CSP `sim.time`과 물리 timestamp를 같은 소스 호출에서 함께 기록하고 native packet 및 시각 노출의 원천 시간 계약을 확인해 유효 프레임만 학습·검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 CSP 시뮬레이터·차량 상태 시각 동시 관측:** 기존 read-only wheel 진단에 `sim.time`과 `car.timestamp`를 수집 전후에 읽는 별도 후보를 추가하고 Lua·진단 관련 12개 테스트를 통과했다 (`integrations/ac_csp/candidates/WheelGeometryClockDiagnostic.lua`, `integrations/ac_csp/candidates/FlyBrainCamera-wheel-clock-diagnostic-v1.lua`, 계획 `D:/CodexResearchRuns/flybrain-wheel-clock-plan-20261102-01/`). 검증된 세션 API로 일시정지, 임시 후보 배치 후 기존 제한 AI motion3를 실행하고 원본을 복구·기존 비일시정지 상태로 되돌렸다 (`D:/CodexResearchRuns/flybrain-wheel-clock-live-20261102-01/`, restore/resume `D:/CodexResearchRuns/flybrain-wheel-clock-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-wheel-clock-resume-20261102-01/`). 원본 해시 확인 결과 165프레임 전부 `sim.time - car.timestamp`가 0ms였고 동일 호출 전후에도 두 값 변화가 0이었다. 속도 ≥1m/s 후보 39프레임·최대 4.532m/s였다 (`scripts/audit_wheel_source_clock.py`, 결과 `D:/CodexResearchRuns/flybrain-wheel-clock-analysis-20261102-01/`). 같은 값은 공통/복사 시계일 수 있어 마지막 물리 상태의 실제 시점, native packet tick, 이미지 노출 동기화의 독립 증거가 아니다. -20ms 보정은 미등록, 실제 학습·배포 불허 유지. 게임 AI 진단만 수행했고 파리 actor·optimizer·데스크톱 입력 0. 다음은 native 쪽 원천 물리 tick/시간을 얻거나 별도의 사건 기반 상한을 검증하고, 단일 회로의 시각·몸 감각과 주행 학습을 이어간다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 CSP 시각 원천의 갱신 영역 확인:** 설치된 CSP SDK의 `ac.getCar()`와 `ac.getSim()` 선언은 둘 다 그래픽 프레임 단위로 갱신됨을 명시하며, `car.timestamp`는 물리 상태 기록 시각이라는 설명이 있어도 이를 읽는 구조는 그래픽 측 자료다. 별도의 car-physics SDK에는 `ac.accessCarPhysics()`가 있지만 현재 확인한 background SDK stub에는 없다. SDK 두 파일 SHA와 선언 근거 `D:/CodexResearchRuns/flybrain-local-csp-clock-semantics-20261102-01/`; API 선언만으로 실제 호출 가능성·native packet 대응·카메라 노출 시각은 입증되지 않는다. 따라서 앞선 165프레임의 `sim.time == car.timestamp`는 독립적인 physics/native tick 검증이 아니다. 기존 `UnifiedHostGuard`는 원래부터 physical_synchronization_verified=false와 actor_qualified=false를 유지했다. 새 wheel-clock 진단 후보 v2에는 상태와 sim의 갱신 영역이 `csp_graphics_frame`이며 native tick 연결은 미검증이라는 메타데이터를 명시했다 (`integrations/ac_csp/candidates/WheelGeometryClockDiagnostic.lua`, `integrations/ac_csp/candidates/FlyBrainCamera-wheel-clock-diagnostic-v2.lua`); 이전 실측 v1 결과와 후보는 보존했다. 진단 Lua와 기존 host guard 13개 테스트 통과, v2 설치·차량 조작·optimizer 0. 다음은 physics-rate read-only 접근의 실행 가능성/격리와 native packet 연결을 검증하며 그래픽 시각을 물리 시간으로 사용하는 것을 금지한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 physics 주기 읽기 후보 준비:** CSP의 [차량 physics 스크립트 문서](https://cup.acstuff.club/docs/csp/cars/physics-scripts)는 확장 physics 활성화와 차량 data 폴더의 `script.lua`를 요구하고 `ac.accessCarPhysics()` 예시를 제공한다. 현재 BMW Z4 GT3는 `data.acd`가 있고 풀린 `data` 폴더가 없어 기존 차량 설정을 바꾸지 않고 이 API를 실행할 수 있다는 증거가 없다 (`D:/CodexResearchRuns/flybrain-physics-tick-probe-review-20261102-01/`). 차량 데이터나 게임 설정은 변경하지 않았다. 별도 실험 후보 `integrations/ac_csp/candidates/PhysicsTickWheelProbe.lua`는 허용된 호출에서만 `ac.accessCarPhysics()`의 네 native 순서 하중·속도·미등록 가속/회전을 읽고 최대 2048개의 상대 sequence와 누적 dt를 메모리에 보관하며 제어 쓰기·파일/네트워크 I/O를 하지 않는다. 잘못된 dt/하중·용량 초과 시 실패를 고정하고 명시적 reset이 필요하다. Lua 모의와 기존 관련 테스트 16개 통과 (`tests/test_physics_tick_wheel_probe.py`); 첫 테스트 실행은 의도적 Lua 오류 검사 중 Windows faulthandler 예외 덤프를 출력했으나 pytest exit 0/16 통과였고, faulthandler를 끈 재실행에서도 16개 통과했다. 이는 모의 검사이며 실제 physics callback·부하·전송 bridge는 미검증, 상대 sequence/dt는 native packet ID나 절대 물리 시간이 아니다. actor·optimizer·게임 설치/차량 제어 0. 다음은 기존 차량 data를 보존할 수 있는 physics 접근·격리와 비차단 전송 방법을 검증하고 native packet 및 시각 tick의 관계를 증명한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 physics→진단용 공유 구조 후보:** 설치된 CSP car-physics와 background SDK stub에 모두 `ac.connect`와 동일 layout을 통한 Lua 간 공유 선언이 있음을 확인하고 원본 SHA를 보존했다 (`D:/CodexResearchRuns/flybrain-physics-wheel-bridge-review-20261102-01/`). 진단 전용 `integrations/ac_csp/candidates/PhysicsWheelLatestBridge.lua`는 차량별 키, 명시적 generation, 상대 sample sequence/dt, 네 하중의 최신 값을 담고 쓰기 중 홀수·완료 시 짝수 commit으로 두 번 읽을 때의 일부 찢어진 읽기와 세대 불일치를 거부한다. `PhysicsTickWheelProbe`의 physics 값만 받는 후보이며 자동차의 물리 필드에 쓰지 않는다. Lua 모의에서 순서·세대 불일치·중간 변경·무효 하중·키 분리 등 관련 15개 테스트 통과 (`tests/test_physics_wheel_latest_bridge.py`). 두 번 읽기는 CPU 메모리 순서나 실제 공유 구조의 원자성을 증명하지 않으며 producer 독점·신선도·CSP context 간 동작·native packet ID·카메라 노출 연계도 미검증이다. 실제 게임에는 설치·실행하지 않았고 actor/optimizer/차량 제어 0. 다음은 기존 차량 data 보존 가능한 physics 실행과 격리된 consumer를 확보하고, 구조 일치·누락/재시작·원자성·시간 계약을 실제 측정해 합격 전 학습 입력으로 사용하지 않는다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 실제 CSP 동일 Lua 영역 공유 구조 시험:** 설치 차량 중 풀린 data 폴더가 있는 차량은 0개였다. physics 스레드를 바꾸지 않고 기존 정차 진단 후보에 격리된 임시 공유 채널 시험을 추가하여 Lua 문법·마스크 관련 13개 검사를 통과했다 (`integrations/ac_csp/candidates/FlyBrainCamera-wheel-bridge-diagnostic-v1.lua`, `scripts/record_csp_bridge_diagnostic.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-csp-bridge-live-plan-20261102-01/`). 검증된 세션 API로 정차·오프라인 조건에서 pause, hash 고정 후보를 임시 배치하고 한 프레임에서 같은 카메라 Lua 영역의 `ac.connect` 참조 2개로 시험했다. 지정한 generation 73, sequence 11, 하중 값 [101.25,202.5,303.75,404]가 정확히 복원됐으며 원본 프레임 해시를 재확인했다 (`D:/CodexResearchRuns/flybrain-csp-bridge-live-20261102-01/`). 이는 실제 CSP typed shared 구조의 동일 영역 시험이지 physics/graphics 스레드 간 공유, 메모리 원자성, native packet ID·물리 tick·노출 시각이나 센서 적격성 증명이 아니다. 임시 키는 독립 ID이며 테스트 값만 사용했고 실제 차량 물리/조작값을 쓰지 않았다. 원래 Lua와 비일시정지 세션을 복구했다 (`D:/CodexResearchRuns/flybrain-csp-bridge-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-csp-bridge-resume-20261102-01/`). actor/optimizer/데스크톱 입력 0. 다음은 차량 data를 변경하지 않는 격리된 physics-rate 실행을 찾아 실제 서로 다른 스레드의 누락·재시작·원자성 및 native/시각 시간 관계를 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 CSP 물리 스레드 통과 이벤트의 비제어 진단:** 설치된 CSP SDK는 `ac.onTrackPointCrossed`의 트랙 지정 지점 통과 시간이 물리 스레드에서 위치를 보간하고 `sim.time`과 같은 기준이라고 설명한다. 기존 차량 data를 수정하지 않고 요청 소유된 오프라인 AI motion3에만 적용되는 4개 임시 구독과 명시적 정리 후보를 만들었다 (`integrations/ac_csp/candidates/TrackPhysicsMarkerDiagnostic.lua`, `integrations/ac_csp/candidates/FlyBrainCamera-physics-marker-diagnostic-v1.lua`, 테스트 16개 통과). 출발 전 현재 정차 위치 앞 네 progress를 진단용으로 고정하고 제한 주행에서 4/4 이벤트가 발생했다 (`D:/CodexResearchRuns/flybrain-physics-marker-plan-20261102-01/`, 원본 `D:/CodexResearchRuns/flybrain-physics-marker-live-20261102-01/`, 해시 감사 `D:/CodexResearchRuns/flybrain-physics-marker-analysis-20261102-01/`). 이벤트 콜백에서 읽은 그래픽 sim.time과 물리 통과 시각 차이는 +9.929/-2.934/+14.081/-1.866ms였고, 콜백부터 최초로 이벤트가 포함된 휠 진단 프레임까지는 77.78/23.64/78.63/34.29ms였다. 그래픽 sim.time의 음수 차이는 콜백 지연을 직접 계산할 수 없음을 보여준다. 문서상 보간된 물리 통과 시각의 관측이지 원천 native packet ID/카메라 노출 시점이나 일반적인 물리 tick 정합의 증명은 아니다. track progress는 진단에만 사용하고 actor 경로 판단 입력이 아니다. 원래 Lua와 시작 전 비일시정지 상태를 복구했으며 파리 actor/optimizer/데스크톱 입력 0, 게임 제한 AI 진단만 수행했다 (`D:/CodexResearchRuns/flybrain-physics-marker-restore-20261102-01/`, `D:/CodexResearchRuns/flybrain-physics-marker-resume-20261102-01/`). 다음은 관측 이벤트에 대한 원본 native 신호 및 이미지 변화를 별도로 대조해 인과 시간 상한을 평가하고, 실패 시 정확한 동기 주장은 보류한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 물리 통과 이벤트와 그래픽/native 보간 대조:** 직전 4개 물리 통과 이벤트의 원본 프레임, AI 진단 progress 이력, native 복사·CSP/host 시계 자료의 SHA를 고정해 사후 분석했다 (`scripts/audit_physics_marker_native_bracket.py`, 계획 `D:/CodexResearchRuns/flybrain-physics-marker-native-plan-20261102-01/`, 결과 `D:/CodexResearchRuns/flybrain-physics-marker-native-audit-20261102-01/`). 그래픽 progress가 같은 표시 지점을 지난 시각을 인접 진단 표본으로 선형 보간하고 그래픽 `sim.time`으로 환산했을 때 물리 이벤트 시각과의 차이는 각각 -25.30/-22.88/-24.08/-23.38ms였다. 진행도 표본 간격은 38.30/42.95/37.42/38.68ms다. 그래픽 시각으로부터 추정한 host 구간을 전후 native 복사가 감싸는 간격은 7.68/4.86/7.24/7.51ms였으나 이 구간은 보간·clock 가정에 조건부이고 native 생성 시각의 증명은 아니다. 기존 -20ms host 가설과 수치상 가깝지만 양쪽 모두 graphics 측 영향을 공유하며 4개 연속 이벤트·같은 차/트랙의 사후 분석으로 독립 검증이나 카메라 노출·실제 물리 tick 동기화로 해석하지 않는다. route progress는 오직 비교 진단에만 사용하고 actor 입력 금지, optimizer/추가 차량 제어/배포 0. 다음은 physics event와 식별 가능한 native 또는 시각 표식을 연결해 원천 시각 불확실성을 제한하고 검증된 시간/감각에서 통합 학습을 수행한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 FlyVis 000 T4d 입력 연결 제한 학습 — 전체 flash 검증 실패:** 기존 공개 000 네트워크의 구조·edges_sign 텐서와 T4d 이외 연결을 고정하고(후속 감사에서 음수 strength에 따른 유효 부호 반전 발견) T4d로 들어가는 10개 shared synapse strength만 최대 절대 변화 0.05로 제한해 전체 화면 ON/OFF 합성 flash의 onset 이후 40 frame에서 3회 Adam 갱신했다 (`scripts/fit_flyvis_t4d_contrast_candidate.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-bounded-fit-plan-20261102-01/`, 결과/후보 `D:/CodexResearchRuns/flybrain-t4d-bounded-fit-20261102-01/`). 1초 steady 및 200 frame grey를 no-grad로 연속 warm하고 onset에서만 gradient를 절단했다. 학습 창 평균 ON-OFF 차이는 -0.3105→+0.0965였지만 갱신 전 손실은 0.1027→0.8241→0.0350으로 비단조였다. 후보 state dict를 원본과 별도 비교해 `edges_syn_strength`의 10개 값만 변했고 최대 변화 0.02656임을 확인했다. 동일 저자 전체 600 frame flash, 학습에 쓰지 않은 radius6를 포함해 사전 정한 부호·비대상 보존 기준을 검증했다 (`scripts/probe_flyvis_declared_checkpoint.py`, 사전 계획 `D:/CodexResearchRuns/flybrain-t4d-full-validation-plan-20261102-01/`, 원본 `D:/CodexResearchRuns/flybrain-t4d-bounded-flash-20261102-01/`, 감사 `D:/CodexResearchRuns/flybrain-t4d-bounded-flash-gate-20261102-01/`). T4d flash index는 전체 -0.3120→-0.0484, radius6 -0.3119→-0.0501로 개선됐으나 둘 다 잘못된 음수였다. 나머지 7개 세포×2반경은 사전 부호·변화량 상한을 만족해 14/16, 전체 gate 실패다. 저자와 독립 numpy 계산 차 0. 사전 계획대로 이동 경계/actor/차량 통합 미진행, 기존 000 checkpoint 무변경, 합성 전용·생리/실제 주행 미검증이다. 다음은 truncated onset 손실과 전체 이력 index 불일치 원인을 반영해 생리 지표를 직접 포함하는 제약 학습을 설계하고, 비대상/보류 자극·원래 단일 회로 유지 검증을 수행한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.


**2026-11-02 정정 및 비음수 제약 T4d 재학습:** 기존 000 설정은 `edges_syn_strength`에 `non_negative`를 요구한다. 앞선 두 후보를 원본 state dict와 대조한 결과 첫 bounded 후보는 strength 1개(인덱스 375, -0.003727), peak 후보는 2개(199/375, 최솟값 -0.000993)가 음수였다. `edges_sign` 텐서가 고정돼도 음수 strength는 유효 시냅스 부호를 뒤집을 수 있으므로 앞선 ‘연결 부호 보존’ 설명을 철회한다. 두 결과·checkpoint는 실패 원본으로 보존하고 생물학적 적격/actor 사용 금지다. 원래 000의 strength 음수는 0개. 이후 후보마다 비음수·유한·원래 대비 bound를 명시적으로 적용하고 평가 전에 검사하는 공통 경계를 추가했다 (`src/flybrain_drive/flyvis_parameter_integrity.py`, `scripts/probe_flyvis_nonnegative_checkpoint.py`, 테스트 `tests/test_flyvis_parameter_integrity.py` 5개 통과). 원본에서 T4d 진입 10개 strength만 첫 40 flash frame의 peak 차이를 목표로 10회 재학습했다 (`scripts/fit_flyvis_t4d_nonnegative_candidate.py`, 계획 `D:/CodexResearchRuns/flybrain-t4d-nonnegative-fit-plan-20261102-01/`, 실행 `D:/CodexResearchRuns/flybrain-t4d-nonnegative-fit-20261102-01/`). 학습 창 ON-OFF peak 차는 -0.1703→-0.0346으로 개선됐으나 음수였다. 후보 state dict는 원본 대비 지정 10개 strength만 다르고 최댓값 변화 0.00491, 음수 strength 0개다. 전체 600 frame 저자 flash 및 미학습 radius6 사전 gate에서 T4d index는 -0.3120→-0.0474, radius6 -0.3119→-0.0496으로 여전히 음수; 다른 7세포×2 반경은 기준을 만족해 전체 14/16 실패 (`D:/CodexResearchRuns/flybrain-t4d-nonnegative-flash-gate-20261102-01/`). 전 실험의 6회 peak 학습은 초기 후보보다 손실/실제 index가 악화했으며 그 후보 역시 음수 strength 위반이라 무효다 (`D:/CodexResearchRuns/flybrain-t4d-peak-fit-20261102-01/`, `D:/CodexResearchRuns/flybrain-t4d-peak-flash-gate-20261102-01/`). 모든 결과 보존, 후보 미채택, 생리·실제 AC·actor 미검증. 다음은 nonnegative 제약하 전체 이력 목표의 안정적 최적화와 별도 moving-edge 자극 검증이다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.
