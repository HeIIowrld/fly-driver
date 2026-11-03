# 네 시야 수집 구간 접근 — 사전 선언

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


이 실행의 역할은 **transit**, 학습·검증·평가에 사용하지 않는다. 최종 목표는 초파리 겹눈/운동 시각과 비행 유래 균형·운동 회로를 차량 몸에 연결해 시각·몸 감각과 검증된 바람으로 조향·가감속을 학습하고 빠른 유효 랩까지 검증하는 것이다. 지도·spline은 이 이동의 사후 진단에만 쓰며 actor 입력이 아니다.

2026-10-31 현재 fresh GET-only 관측은 AC PID 32912, generation `593846ULL:0.0036658`, epoch 0, Monza, paused·AI off·brake 1·vJoy FREE·background, lap0 spline 0.94050109다. 기존 `configs/flight-spatial-exterior-front-corpus-20261029.json`의 지리적 TRAIN 0.27–0.63, known DEV 0.67–0.84와 `next_lease`는 이 위치에서 30초 transit을 제안한다. 기존 계획의 오래된 프로세스/소스 봉인을 현 세션의 자격으로 재사용하지 않는다.

기존 검증된 CSP `expert-collect30-speed60`과 네 시야 MMF/source timing/same-copy native 기록을 이용한다. 이동 제한·AI 종료·브레이크·pause는 기존 CSP API 영수증으로 확인한다. protected 구간을 포함해 어떤 시점의 데이터도 fit하지 않는다. 실패·경로 이탈·알 수 없는 자식 상태는 원본을 남기고 자동 재시도하지 않는다. 결과를 본 뒤 role을 바꾸지 않으며, train/독립 평가 수집은 별도 사전 선언·소스/시각/라벨 자격 후 실행한다.

## 첫 이동 결과와 다음 이동 사전 선언

첫 이동 `20261031-spatial-transit-01/`은 종료 코드 0, spline 0.940501→0.026888, 최고 60.91km/h, 변위 495.91m, 전체 telemetry에서 트랙 바깥 바퀴 0이었다. 경계선 통과가 보이지만 lap 유효성은 false이며 어떤 데이터도 fit하지 않는다. 별도 GET-only `20261031-spatial-transit-01-after-observation/`은 동일 정지 위치·paused/full-brake/FREE/background를 확인했다.

현재 spline 0.0268878에서 동일 기존 지리적 계획의 `next_lease` 결과는 다시 **30초 transit**이다. 다음 실행 `20261031-spatial-transit-02/`도 위와 동일한 사전 조건과 60km/h 제한을 적용하고 역할을 transit·fit 금지로 고정한다. 완주나 경로 품질을 주장하지 않는다.

두 번째 이동은 종료 코드 0, spline 0.026888→0.113329, 최고 60.92km/h, 변위 497.77m, 전체 telemetry에서 트랙 바깥 바퀴 0이었다. `20261031-spatial-transit-02-after-observation/`에서 같은 위치 및 paused/full-brake/FREE/background를 확인했다. 해당 지점은 과거 평가 보호 구간 0.08–0.125에 있으므로 두 이동 모두 학습에 쓰지 않는다. 학습 시작 구간 0.27–0.63에 도달하지 않았고 B2의 시작 구간·카메라·소스 재적격화도 별도다.

## 세 번째 이동 사전 선언

현재 0.1133289에서 기존 지리적 계획의 `next_lease`는 **30초 transit**이다. `20261031-spatial-transit-03/`은 기존과 같은 60km/h 유한 AI·네 시야 기록을 사용하되 보호 구간을 포함한 이력 전체를 transit·fit 금지로 유지한다. 종료 후 실제 경로·정지와 원본 상태를 확인한다. 외부 모델이나 학생 제어는 없다.

세 번째 이동의 차량 제어는 정상 종료되어 spline 0.199108, paused·AI off·brake 1·camera off에 도달했으나 수집 프로세스는 **실패/종료 코드 1**이다. GPU 네 시야 그룹 기한 40ms가 초과돼 969그룹을 저장한 뒤 producer는 970그룹에서 정지했다. `20261031-spatial-transit-03/` 원본, 실패 슬롯과 복사 실패 trace를 보존했고 어떤 프레임도 학습에 쓰지 않는다. 별도 `20261031-spatial-transit-03-after-observation/`은 paused/full-brake/FREE/background 및 같은 위치를 확인했다. 기존 역할은 바꾸지 않고 실패 실행을 자동 재시도하지 않았다.

`scripts/audit_transit_gpu_stall.py`가 앞선 성공 두 기록과 실패 기록의 원본 zip 해시·GPU host 시간 흔적을 대조했다. 성공 그룹의 group-start→Flush 반환은 세 기록에서 중앙 0.142–0.160ms, 95백분위 0.275–0.289ms, 최대 2.59–2.78ms였다. 실패 그룹은 Flush 반환이 시작 후 **146.90ms**, 첫 map 진입은 146.91ms로 기록됐으며 40ms 기한을 그 전에 넘었다. `20261031-transit-gpu-stall-audit/`에 증거를 보존했다. 드라이버 실행이나 운영체제 스케줄링 중 구체적 원인은 이 기록만으로 확정할 수 없다. 시간 제한을 늘리거나 낡은 프레임을 학습 데이터로 허용하지 않았다.

## 2026-10-31 정지 네 시야 전송 재현 진단

사전 paused·offline·full-brake·AI off·vJoy FREE·background를 확인하고 `scripts/qualify_stationary_exterior_mmf.py`로 기존과 동일한 외부 시야 설정, MMF, batched shared copy, async persistence 및 source-timing의 정지 카메라 **900그룹**을 수집했다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-stationary-mmf-stall-reproduction/`은 프로세스 종료0, 누락·오류 없이 900그룹과 원본 6,312파일 해시 일치, 최종 세션 paused/full-brake/AI off다. 차량 조작과 학생·학습은 0이다. `20261031-stationary-vs-moving-gpu-audit/`에서 정지 그룹 start→Flush 반환 중앙 0.165ms, 95백분위 0.274ms, 최대 2.611ms였으며 직전 실패의 146.90ms는 재현되지 않았다.

관측 당시 이전 두 실험의 결과 파일은 저장됐으나 Python child PID 19188/18176의 프로세스 핸들이 살아 있었고 GPU 프로세스 메모리는 각 약 742MiB로 관측됐다. 프로세스들이 지연을 유발했다고 입증되지 않았고 임의 종료하지 않았다. 정지 900그룹 성공은 드문 지연의 부재, 주행 중 안정성 또는 물리 노출·센서 동기를 증명하지 않는다. 타임아웃·새 프레임 판정 기준을 바꾸지 않았다. 다음은 GPU 부하와 주행 조건의 기록을 유지하면서 별도 제한 진단으로 간헐적 지연 원인을 좁히고, 새 학습용 수집 자격을 확립하는 일이다.

## 2026-10-31 네 번째 이동 사전 선언

직전 실패의 종료 및 새 정지 관측 위치 spline 0.1991077에서 기존 지리 계획의 `next_lease`는 30초 transit이다. 실패 이력은 재시도하거나 이어붙이지 않고, `20261031-spatial-transit-04/`는 별도 시각·원본을 갖는 60km/h 유한 이동 진단으로 수행한다. 역할은 사전 **transit/fit 금지**이며 보호 구간의 어떤 기록도 학습에 사용하지 않는다. 40ms GPU 그룹 기한·freshness·소유권은 변경하지 않는다. Windows GPU engine/프로세스 메모리 카운터를 독립적으로 1초 간격 수집해 시간상 부하를 살펴보되 GPU 지연의 인과 원인이나 순간 부하를 증명하는 것으로 해석하지 않는다. camera 원본 및 정상/실패·CSP 정지 영수증은 보존하고 자동 재시도하지 않는다.

`20261031-spatial-transit-04/`는 종료0, spline 0.199108→0.285681, 최고 속도 61.04km/h, 변위 459.94m, 전 telemetry 바퀴 이탈 0, 네 시야 유효 731결정·camera miss 0이었다. 새 GET-only `20261031-spatial-transit-04-after-observation/`은 0.285681 및 paused/full-brake/FREE/background를 확인했다. 이 위치는 기존 지리적 TRAIN 구간 0.27–0.63 안이나 B2 제동 수집의 시작 0.49–0.54는 아니다. 이동은 사전 transit이며 fit 금지다.

`20261031-transit-04-gpu-timing-audit/`에서 성공 1,075그룹 start→Flush 반환 중앙 0.167ms/95백분위 0.272ms/최대 2.474ms였다. 독립 Windows 1초 표본 55개 `20261031-transit-04-counter-summary/`에서 AC 단일 GPU 엔진 최대 약 8.68%, 이전 두 평가 PID 각각 GPU 전용 메모리 약 742MiB·표본된 각 GPU 엔진 최대 0%였다. 이는 1초 샘플이며 실패 시 147ms의 원인을 설명하거나 지연 부재를 보증하지 않는다. 기존 timeout을 바꾸지 않았다.

## 별도 학습 후보의 사전 선언과 원본 감사

현재 지리적 TRAIN 구간에 도달한 후 기존 transit을 재분류하지 않고, 새 `20261031-prospective-visual-train-plan.json`(SHA-256 `21dbb2f66a6d8b27df047393007521d78fb94b505c2193b3a873dcca40bc14b2`)에 **train_candidate·fit 금지·자동 재시도 금지**, 동일 session/PID, 시작 0.2856812, camera 및 주요 recorder/observer/control/vision/route 소스 해시를 기록했다. `scripts/collect_prospective_visual_train_candidate.py`는 수집 전 독립 GET-only 실제 위치와 기존 구간 계획을 재확인하고 새 30초 AI 기록을 수집했다. 전체 원본 telemetry/정지 상태는 구간 0.285681–0.372268, 바퀴 이탈 0 및 같은 lap 조건을 통과했다. 실행 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-prospective-visual-train-candidate-01/`, child 종료 코드 0, 여전히 fit 금지다. 기존 train 구간의 새로운 이력이지 독립 blind test가 아니다.

별도 `20261031-prospective-visual-train-candidate-01-raw-audit/`는 원본 7,537파일·1,075그룹·731결정, MMF와 영상/몸 감각 재생 오차 0, aid off, 최대 결정 간격 43.319ms 및 80ms 초과 0을 확인했다. `20261031-prospective-visual-train-candidate-01-receipt-age/`의 시야 render-start 호스트 하한→native 수신까지 중앙 상한 29.54–30.47ms, 95백분위 65.93–66.92ms다. 이는 실제 노출/물리 발생 시차가 아니다. same-copy native AI 표식과 원본 actor 입력은 보존했지만 교사 조작 codec·카메라/물리 실제 시각 정합·labels는 미검증이며 기록은 fit 불가다. 새 collector와 GPU counter 요약 Ruff 및 기존 지리 경계 검사 14개 통과. 이후 시각/몸 동기와 교사 표적 검증, 독립 평가를 거쳐야 학습할 수 있다. 종료 후 paused·AI off·brake 1·camera off를 확인했다.

## 같은 native 복사본 권한 감사 및 사후 정지 조향 보정 계획

`scripts/audit_ac_same_copy_teacher.py`의 `20261031-prospective-visual-train-candidate-01-authority-audit/`는 각 결정의 원본 420바이트와 body 308바이트 접두부, 해시·packet·AI flag·수신 시각을 검증했다. 731/731은 같은 복사본의 native AI flag와 기존 좁은 조건을 통과했으나 물리 발생 시각·카메라/교사 동기·엔진의 원자성·최적 교사는 검증하지 않는다. 학습 권한은 false다.

조향 단위 검토를 위해 같은 generation·동일 항목 controls.ini의 현재 해시에 묶인 별도 설정으로, `calibrate_ac_sdk_steering.py` 기존 정지·브레이크 조건에서 0/±0.6/0의 유한 보정을 수행한다. 이 보정은 위 수집 **이후**에 수행하므로 앞서 수집 시점의 설정 불변과 동적 주행의 조향 전달, 60km/h 교사 표적의 타당성은 별도로 증명해야 한다. 기존 teacher label 계약은 15/30km/h만 허용하므로 60km/h 기록을 우회해 학습하지 않는다. 실패·비활성 publisher·시각 불일치는 그대로 보존하고 자동 재시도하지 않는다.

보정 시도 `20261031-post-visual-training-steering-calibration/`는 세션 resume 후 사전 검사에서 CSP AI off이나 native AI flag true여서 **장치 획득 전 실패**, 샘플·actuator 요청 0이었다. `20261031-steering-calibration-pause/`는 CSP pause 성공이다. 이를 단순 paused frozen packet으로 해석할 수 있는지 별도 확인하기 위해 다시 CSP resume 후 `scripts/audit_live_native_ai_idle.py`가 2초 이내 정지/full-brake 구간에서 같은 native 420바이트 복사본과 전후 CSP AI-off 상태를 읽었다. 기존 기준 packet 171304 이후 새 packet 174727–175026의 300/300에서 native AI flag true였고 CSP는 AI off였다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-native-ai-live-idle-audit/`의 raw bytes·hash를 보존했으며, 후속 `20261031-native-ai-idle-audit-pause/`에서 pause 성공·UI/actuator 0이다. native flag의 의미나 양 보고 중 실제 독점 제어를 어느 쪽이 증명하는지는 아직 미확정이다. 진행된 packet이므로 이번 값은 단순히 일시정지 중 같은 frozen packet을 다시 읽은 것만으로 설명할 수 없다. 조향 보정·새 교사 표적·학습은 하지 않았고 충돌 검사를 완화하지 않았다.

## 2026-10-31 AI 표식 불일치에 대한 세션 재시작 사전 계획

사용자는 필요시 재시작을 명시적으로 허용했다. 보존된 후보 데이터와 역할은 변경하지 않는다. 현재 offline·paused·CSP AI off·full-brake·카메라 off·vJoy FREE를 확인하고 기존 `control_ac_session.py --command restart --execute`의 CSP lifecycle 계약을 한 번 사용한다. 이는 엔진 재설치나 UI 자동화가 아니라 기존 시뮬레이터 세션 재시작이다. 세션 epoch/메뉴·terminal을 확인하고 새로운 세션의 native AI/body를 별도로 점검한다. 기존 spline 위치는 바뀔 수 있으며 평가/학습 역할은 새로 검토한다. 상태 또는 요청 결과가 불확실하면 재시도·조작하지 않고 원본을 보존한다. 재시작만으로 native flag 의미, 단독 제어 또는 학습 자격을 주장하지 않는다.

세션 restart와 start는 각각 `20261031-native-ai-conflict-session-restart/`, `20261031-native-ai-conflict-session-start/`에 성공 영수증이 남았고 epoch 1로 전환됐다. 이후 확인한 native packet 178587을 기준으로 진행 패킷 감사를 실행했으나 `20261031-native-ai-after-session-restart/`는 기준보다 진행한 패킷과 정지/full-brake 조건을 요구하는 검사에서 실패했다. 실패 결과만으로 어느 조건이 실패했는지 판정할 수 없다. 별도 관찰에서는 동일 packet이 반복됐으나 원본 연속 기록이 없어 지속 정체나 새 세션의 AI 상태를 확정하지 않는다. `20261031-native-ai-after-session-restart-pause/`에서 pause 목표 상태 관측, UI·actuator 요청 0을 확인했다. 재시작 후 AI 충돌 해소는 미확인이고 보정·학습·모델 배포는 하지 않았다. 다음에는 일시정지 상태를 유지하며 restart 전후 native 매핑과 packet 검사 실패 조건을 원본 기반으로 조사하고, 재개가 필요하면 별도 제한 진단을 설계한다.

## 재시작 후 native packet 구분 진단 사전 선언

현재 paused·offline·CSP AI off·camera off·native brake 1인 동일 generation에서 CSP resume을 한 번 사용한 뒤 `diagnose_native_restart_packets.py`로 최대 3초 동안 native 420바이트 반복 복사와 각 검사 조건을 저장하고 CSP pause한다. 원본 기준 packet 178587을 포함해 중복 관측도 기록하되 신선한 물리 표본으로 취급하지 않는다. 속도·제동·CSP 조건 이탈 시 진단을 즉시 종료하고 pause를 요청한다. actuator/UI·학습·자동 재시도는 없다. 실패 시 원본과 실패 조건을 보존한다.

재시작 후 구분 진단 `20261031-restart-packet-distinction-diagnostic/`은 59개 독립 packet ID 180476–181466, 전체 native AI false와 CSP AI off·정지·brake 조건 통과를 기록했다. 직전 실패 원인은 결과만으로 확정할 수 없으나 새 live 구간에서 packet 갱신과 native AI off가 관측됐다. 뒤이어 `20261031-restart-packet-distinction-pause/` 성공이다. 실제 독점 제어는 여전히 미입증이다.

별도 사전 선언: 같은 profile SHA-256을 재확인한 사후 보정 설정 `20261031-after-restart-steering-config.json`으로 다시 CSP resume 후 정지 0/+0.6/-0.6/0 조향 보정을 1회 시도하고 CSP pause한다. 매 샘플 전후 native/CSP AI off·정지·full-brake·차량/프로필 소유권 조건을 유지한다. 실패 시 기록하고 재시도하지 않는다. 이는 재시작 이후 정지 조향 전달 확인이며 과거 60km/h 자료의 동적 표적·시간 정합 자격을 부여하지 않는다.

정지 보정 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-after-restart-steering-calibration/`은 종료0·32샘플, 요청 0/+0.6/-0.6/0에 SDK 중앙값 0/+269.99997/-269.99997/0, 비율 449.99995였다. 32샘플 전후 native AI off, 최대 속도 0.015797km/h, 최소 brake 1.0000, 최대 gas 0.0000; background 기록 32/32다. 조향·가속·브레이크 0/0/1 종결 요청이 driver에 승인됐으나 그 자체가 게임 적용 증거는 아니다. 후속 `20261031-after-restart-steering-pause/`는 목표 상태 확인·UI/actuator 0, 별도 CSP 관측 paused/offline/AI off/camera off다. 재시작 후 정지 조향 결과이며 독립적인 단위·주행 중 전달·과거 수집 당시 설정의 연속성·교사/영상/물리 시간 정합·60km/h label 자격은 입증하지 않는다. 후보 학습이나 배포는 없었다.

## 정지 조향 원본 독립 감사

`scripts/audit_stationary_steering_calibration.py`는 `20261031-after-restart-steering-independent-audit/`에 32개 native 308바이트 원본 SHA 및 재복원 필드·packet/수신 순서·브레이크·가속·속도·조향 전후 AI off·background를 별도 검증했다. 정지 native 중앙값은 0/+0.84374994/-0.84374994/0, 양 끝 계산 gain 1.40624990/요청, SDK 중앙값 0/+269.99997/-269.99997/0이다. 중간 요청 선형성·실제 물리 단위·주행 중 전달·과거 60km/h 수집 시 설정 불변·교사/영상 시간 정합은 검증하지 않았다. 과거 수집은 재시작 전 epoch 0, 보정은 재시작 후 epoch 1이며 같은 generation만으로 이 간극을 메울 수 없다. 제안 라벨 생성기는 출력에 학습 불가를 명시하고, `seal_ac_engine_ai_receipt_dataset.py`의 `common_expert_settings`가 기존 15/30km/h 과제만 허용하므로 60km/h를 승인하지 않는다. 스크립트 Ruff와 기존 라벨·same-copy 검사 22개 통과, 실제 학습·차량 조작은 없다.

## 재시작 후 학습 구간 접근 1 사전 선언

GET-only `20261031-after-restart-position-observation/`에서 PID 32912, generation `593846ULL:0.0036658`, epoch 1, paused/offline/CSP AI off/full brake/vJoy FREE/background, lap0 spline 0.8564310074를 확인했다. 기존 구간 계획의 `next_lease`는 transit 30초다. `20261031-after-restart-transit-01/`은 기존 네 시야 60km/h 유한 AI 수집 경로로 수행하되 전체 기록을 transit·fit 금지로 사전 고정한다. 보호 구간을 통과할 수 있으며 기록은 분할·재분류하지 않는다. UI·학생 제어 없이 원본과 실패를 보존하고 자동 재시도하지 않는다. 현 free C 공간 약149GiB로 기존 최소 40GB를 넘는다. 종료 후 API/route/정지 상태를 별도 확인한다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-after-restart-transit-01/`은 프로세스 종료0·기록 completed, spline 0.856431→0.941895, 최대 61.1003km/h, 시작점 변위 최대 266.38m, 네 시야 유효 731결정/1,075그룹 및 camera miss 0이다. receipt telemetry 976개에서 wheels_outside 최대 0이나 lap_valid는 전부 false; 유효 완주 근거가 아니다. 사전 transit 및 fit 금지는 유지한다. `20261031-after-restart-transit-01-observation/` 독립 GET-only 관측에서 epoch1 같은 위치·paused/full-brake/FREE/background 확인. 현재 위치는 TRAIN 0.27–0.63 밖이고 다음 접근도 보호 구간을 지나는 transit이다. 학습·학생 제어 없음.

## 재시작 후 학습 구간 접근 2 사전 선언

`20261031-after-restart-transit-02-pre-observation/`은 epoch1 같은 엔진의 spline 0.94189453125, paused·full-brake·FREE·background를 다시 확인했다. 기존 구간 계획은 30초 transit을 지정한다. `20261031-after-restart-transit-02/`는 전과 같은 60km/h 네 시야 유한 이동이며 전체 이력 fit 금지, 보호 구간 이동 전용이다. UI·학생 제어·자동 재시도 없이 종료 상태와 원본을 보존한다.

두 번째 이동 `20261031-after-restart-transit-02/` 종료0, spline 0.941895→0.028391, 최대60.8875km/h, 731결정·camera miss 0, telemetry 975개 중 최대 바퀴 이탈 0; lap valid false다. `20261031-after-restart-transit-02-observation/`에서 같은 epoch/위치 및 paused/full-brake/FREE/background를 확인했다. 이력 전체는 transit·fit 금지다.

## 재시작 후 학습 구간 접근 3 사전 선언

현재 spline 0.0283909477에서 기존 계획은 30초 transit이다. `20261031-after-restart-transit-03/`은 같은 60km/h 네 시야 유한 이동이다. 과거 평가 보호 구간 0.08–0.125에 진입할 수 있으므로 전체 이력은 transit·fit 금지이며 분할/재분류하지 않는다. UI·학생 제어·자동 재시도는 없다.

세 번째 시도 `20261031-after-restart-transit-03/`는 프로세스 종료1. CSP terminal은 `rejected`, 사유는 여러 조건을 합친 `Expired, changed, active or nonstationary session`이다. 반환된 before는 epoch1, paused, AI off, speed 0.008675, gas 0, brake 1, wheels outside 0, 요청된 카메라 active다. AI API 호출 목록은 비어 있고 결정 0, 카메라는 378그룹 후 stop됐다. 요청 만료값 4581.5517119와 before script clock 4578.585034은 기록됐으나 최종 판정 시각·카메라 소유 freshness 등 개별 조건은 남지 않아 거부 원인을 특정할 수 없다. 요청을 자동 재시도하지 않았다. 후속 `20261031-after-restart-transit-03-observation/` GET-only에서 같은 spline 0.028391 및 paused/full-brake/FREE/background를 확인했다. 실패 원본과 transit fit 금지 역할 유지. 다음에는 복합 조건의 실패 원인을 기록할 수 있는 진단을 마련하고 새 독립 계획으로 진행한다.

추가 원본 분석: 세 번째 시도의 보존된 camera.zip 내 `source-timing-v1.json` SHA `c9aeca8319901a2ae461c1382693f47157a72218db4df7963fb795101bcea979`에서 카메라 source timing 시작 4563.4423532, 거부 before의 script clock 4578.585034는 해당 `pollSession` 구간 4578.5848246–4578.5855946 안에 있다. 원본 manifest의 소스 해시가 현재 녹화기·vision·Lua 해시와 일치하며, 그 소스 경로의 1200프레임×0.04초+5초 카메라 deadline은 53초다. Lua는 60km/h 30초 probe 진입에 카메라 deadline까지 39초 이상을 요구한다. timing 시작은 active 생성 직후이므로 해당 거부 시점 남은 시간의 상한은 약 37.857초로 조건에 미달한다. 이 계산은 원본 시간과 고정 코드 경로에서 나온 추론이며 terminal이 개별 실패 조건을 내보내지는 않았다. 직전 capture는 같은 poll 전 약32ms에 실행됐다. 다음 수정은 사전 준비/카메라 시작 지연 예산을 분리하고 기존 39초 안전 조건과 1200프레임·MMF 한도를 유지하는 방향으로 검토한다. 타임아웃을 무작정 완화하거나 거부된 요청 ID를 재발행하지 않는다.

## 새 카메라 주기 transit 진단 사전 선언

거부된 요청은 재발행하지 않는다. recorder에 선택형 `--camera-period-seconds .045`를 추가했다. 공유 30초 기록에서만 허용하며 기본값 .04, 1200프레임·60초 미만 MMF source timing 제한·30초 제어·39초 남은 camera 소유 검사·40ms GPU 기한·신선도는 그대로다. .045에서는 카메라 deadline 59초, 프레임 주기 45ms이므로 53초 설정보다 준비 여유가 6초 늘지만 결정 누락/실제 영상 동기 개선을 보장하지 않는다. 관련 테스트 68개와 Ruff 통과. `20261031-camera45-transit-pre-observation/`은 epoch1 spline 0.02839095, paused/full-brake/FREE/background 확인. 별도 `20261031-camera45-transit-diagnostic/`은 보호 구간을 통과하는 새 60km/h transit·fit 금지 진단이다. 원본/실패·카메라 간격을 보존하며 학생·UI·자동 재시도는 없다.

`20261031-camera45-transit-diagnostic/`은 종료0, spline 0.028391→0.114856, 최대 60.9055km/h·변위 최대 497.91m, telemetry 978개 바퀴 이탈 최대0, lap valid false다. 네 시야 유효 732결정, camera wait miss 0, 전체 958그룹이다. 단 events의 camera capture ID는 732결정에 유일 581개로 인접 재사용 151회다. 직전 40ms transit 02는 731결정 중 유일 633개·인접 재사용 98회였다. 두 이력은 독립·시간적 조건이 달라 인과 비교는 아니며 재사용은 새 시각 노출이 아니다. 결정 간격 최대43.456ms·80ms 초과0; source exposure/물리 동기 및 이 주기의 운동 시각 학습 자격은 검증하지 않았다. `20261031-camera45-transit-observation/`에서 보호 구간 spline 0.114856, paused/full-brake/FREE/background 확인. 전체 이력은 사전 transit·fit 금지다.

## 45ms 이동 2 사전 선언

`20261031-camera45-transit-02-pre-observation/`이 epoch1 spline 0.1148560122, paused/full-brake/FREE/background를 재확인했다. 기존 계획의 다음 역할은 30초 transit이며 과거 평가 보호 구간에서 출발한다. 새 `20261031-camera45-transit-02/`는 앞서 진단한 45ms 주기·60km/h 네 시야 조건의 이동 전용 기록이고 전체 fit 금지다. 학생/UI·자동 재시도는 없고 실패와 원본을 보존한다.

45ms 이동 2 `20261031-camera45-transit-02/` 종료0, 0.114856→0.200675, 최대 60.9108km/h·변위 최대 465.75m, 네 시야 유효 733결정·camera miss 0, telemetry 976개 바퀴 이탈 최대0, lap valid false. `20261031-camera45-transit-02-observation/`에서 같은 위치·paused/full-brake/FREE/background를 확인했다. 전체 transit·fit 금지다.

## 45ms 이동 3 사전 선언

현재 spline 0.2006751에서 기존 지리 계획은 30초 transit이며 과거 평가 보호 구간 0.20–0.25를 통과한다. `20261031-camera45-transit-03/`은 동일 45ms·60km/h 유한 이동이고 기록 전체를 transit·fit 금지로 고정한다. UI·학생 제어·자동 재시도 없이 원본과 정지 상태를 확인한다. 도착 후 TRAIN 여부와 새 수집 자격은 독립 판단한다.

45ms 이동 3 `20261031-camera45-transit-03/` 종료0, 0.200675→0.287229, 최대61.0361km/h·변위 최대459.24m, 네 시야 유효732결정·camera miss 0, telemetry 976개 바퀴 이탈 최대0, lap valid false. `20261031-camera45-transit-03-observation/`에서 동일 위치·paused/full-brake/FREE/background 확인. 보호 구간을 포함한 이력 전체는 transit·fit 금지다.

## 재시작 후 30km/h 학습 후보 사전 선언

같은 engine PID32912·generation `593846ULL:0.0036658`·epoch1, GET-only spline 0.2872285843은 기존 TRAIN 0.27–0.63 안이다. `20261031-epoch1-30kmh-train-candidate-01/`은 기존 40ms 네 시야 MMF/source timing/same-copy native authority, AI speed hint 30km/h·steer multiplier .9·30초로 새로 수집한다. 역할은 사전 train_candidate, fit 금지·자동 재시도 금지다. 30km/h 과제는 기존 후보의 60km/h를 재분류한 것이 아니며 이력 전체 경로가 TRAIN을 벗어나거나 보호 구간에 들면 후보 전체를 거부한다. 앞서 epoch1의 정지 ±0.6 조향 보정은 시작 전 확보됐지만 주행 전달·시각과 물리 실제 시각 동기·교사 표적·원본 감사는 별도다. 카메라 40ms group 기한·freshness·actor 입력 경계·AI 안전 계약을 바꾸지 않는다. 기존 실패·transit·평가 원본은 보존하며 학생·UI 조작 없이 실패 시 재시도하지 않는다.

새 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-epoch1-30kmh-train-candidate-01/`은 종료0, 전체 telemetry 경로 0.287229–0.331417, 바퀴 이탈 최대0, 최고30.9493km/h, 네 시야 유효732결정·camera miss0, lap valid false다. `20261031-epoch1-30kmh-train-candidate-01-observation/`에서 종료 spline 0.331407, paused/full-brake/FREE/background 확인. `...-raw-audit/`는 7,523파일·1,073 camera 그룹의 aid off·MMF 원본과 732결정 시각/몸 actor 재생 오차0, 최대 결정 간격44.156ms·80ms 초과0을 확인했다. `...-authority-audit/`는 같은 native420/body308 복사본 AI true 732/732 및 20,537검사를 보고한다. 이는 실제 발생시각 동기·원자성·최적 교사 검증이 아니다.

같은 epoch 사전 정지 ±0.6 보정과 원본 감사에 기초한 `...-label-proposal/`은 미래40ms 수신값 표적 제안으로 732결정 중 578행 세 축 유효, 범위 밖 조향0·clip0이다. 유효 조향 >0.05는 17행, <-0.05는 6행, brake는 578행 모두0이라 코너·제동 학습을 충분히 대변하지 못한다. 최종 데이터 봉인·fit·배포 없고, 동적 조향 전달·카메라/물리 실제 발생 시간·독립 코너/제동 평가와 생리 검증이 남았다. lap 유효성 false는 자율 완주 실패/성공으로 해석하지 않는다.

## 제동 후보 구간 접근 1 사전 선언

`20261031-braking-approach-pre-observation/`에서 epoch1 spline 0.33140731, paused/full-brake/FREE/background 확인. 제동 자료 후보 위치 0.49–0.54까지 기존 60km/h·45ms 카메라 경로로 30초 유한 이동 `20261031-braking-approach-01/`을 한다. 전체 transit·fit 금지, UI·학생 제어·자동 재시도 없음. 도착 위치와 새 제동 수집 자격은 별도 확인하며 과거 B2 봉인을 재사용하지 않는다.

접근1 `20261031-braking-approach-01/` 종료0, spline 0.331407→0.418208, 최대61.1358km/h·733결정·camera miss0, telemetry 978개 바퀴 이탈 최대0, lap valid false. `...-observation/`에서 같은 위치·paused/full-brake/FREE/background 확인. 전체 transit·fit 금지다.

## 제동 후보 구간 접근 2 사전 선언

현재 spline 0.4182077에서 `20261031-braking-approach-02/`는 동일 60km/h·45ms·30초 유한 이동이다. 전체 이력은 transit·fit 금지이며 제동 수집 후보 시작 구간 0.49–0.54에 도달하는지 별도 GET-only 관측한다. UI·학생 제어·자동 재시도는 없다. 도착만으로 기존 B2 봉인/적격성을 재사용하지 않는다.

접근2 `20261031-braking-approach-02/` 종료0, spline 0.418208→0.505160, 최대61.2395km/h·732결정·camera miss0, telemetry 976개 바퀴 이탈 최대0, lap valid false. `...-observation/` GET-only에서 같은 위치·paused/full-brake/FREE/background 확인. 전체 transit·fit 금지다. 기존 B2는 PID·Lua·source 봉인이 달라 현재 위치에 도달한 것만으로 수집 자격이 생기지 않는다.

## epoch1 제동 구간 pace 진단 사전 선언

현재 spline 0.50516027에서 기존 `expert-pace60-100-stop30`으로 새 네 시야·몸·같은 native AI 복사본을 수집한다. `20261031-epoch1-brake-region-pace-diagnostic/`은 명시된 외부 60→100→0→60km/h 과제이며 전 이력 diagnostic·fit 금지, 자연 레이싱 교사·B2 학습 자료로 재분류하지 않는다. 같은 45ms MMF/source timing·기존 제어/속도/제동·camera freshness 안전 계약을 유지하고, 전체 원본 경로/표본·성공/실패·정지 상태를 기록한다. 이전 B2의 다른 Lua나 봉인을 재사용하지 않고 UI·학생 제어·자동 재시도는 없다. 수집 후 moving-brake 원본·동일 복사본 AI·네 시야 raw를 별도 감사한다.

`20261031-epoch1-brake-region-pace-diagnostic/`은 종료0, 전체 경로 0.505160–0.584101, 최고100.9855km/h·시작점 대비 최대 변위453.34m, telemetry978개 바퀴 이탈 최대0·lap valid false, 네 시야 유효731결정·camera wait miss0이다. `...-pace-observation/` GET-only에서 spline 0.584092 및 paused/full-brake/FREE/background를 확인했다. `...-pace-raw-audit/`은 원본 영상·몸 actor 재생 오차0을 보고했다. `...-moving-brake-audit/`는 원본 native bytes/hash에서 speed>1km/h·brake>0.05인 연속 62결정, 속도 1.965–99.568km/h, 같은 복사본 AI true 62/62를 확인했다. 이들은 독립 표본이 아니며 camera/물리 발생 시각·독점 제어·최적 제동은 미검증이다. 역할은 외부 속도 과제 diagnostic·fit 금지로 보존한다. B2 새 소스 자격·페달 연속성·학습용 별도 사전 선언 및 fit/독립 평가가 여전히 필요하다.

## 네 시야 렌더 전 몸 수신 경계 감사

`scripts/audit_pace_causal_body_receipts.py`는 원본 native 수신 바이트/hash·packet과 원본 raw replay 감사에 묶어, 각 네 시야 render-start host 하한의 최솟값보다 엄격히 먼저 도착한 마지막 native 수신을 고른다. 각 시야의 실제 render 상한은 추정하지 않고 그룹의 host camera 수신 시각으로 보수적으로 묶으며 해당 상한과 몸 수신 간 차가 50ms 초과면 거부한다. 새 30km/h 후보 732/732의 중앙 host 차 상한은 25.88ms, fit 금지 pace 진단 731/731은 29.35ms였다. 각 경우 기존 actor가 기록한 native packet과는 0개만 같았다. 이는 저장 host 수신 순서와 보수적인 나이 경계일 뿐 physics 발생·카메라 노출 시간의 동기 증명이 아니다.

전 수신 후보와 기존 actor 기록의 절대 속도 차 중앙/95백분위/최대는 30km/h에서 0.00153/0.3215/3.0067km/h, pace 진단에서 0.03145/2.0369/4.5738km/h였다. 브레이크 차가 0.05를 넘는 행은 각각 1/732, 3/731이다. 관측은 연속이고 packet의 원 발생 시각은 알려지지 않아 정확한 시간 오차·인과 관계를 뜻하지 않는다. 원본과 역할을 바꾸지 않았으며 새 입력 배열·학습·배포는 없다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-epoch1-30kmh-causal-body-receipt-audit-v2/` 및 `20261031-epoch1-brake-pace-causal-body-receipt-audit-v2/`에 각 pairing과 원본 hash를 보존했다. 기존 선택기 검사 4개 및 Ruff 통과. 이후 실제 시각/물리 표본 시점 검증과, 선행 수신을 사용한 별도 actor 후보의 유효성·기존 후보 대비 평가가 필요하다.

## 선행 몸 수신 반사실과 결정 시점 유효성 (2026-10-31)

`scripts/build_prior_receipt_body_counterfactual.py`는 각각의 이전 수신 audit 및 native 원본 hash를 검증하고, 원래 actor 몸과 렌더 전 마지막 수신 몸을 같은 21채널 NavigationBodyEncoder에 원래 결정 시각으로 각각 입력했다. 원본 episode·시간·마스크를 유지한 별도 반사실 arrays는 `20261031-epoch1-30kmh-prior-body-counterfactual/`(732행), `20261031-epoch1-brake-pace-prior-body-counterfactual/`(731행)에 저장했다. 영상/물리 발생 시각 증명·actor 입력 교체·fit은 아니다.

기존 속도/yaw 필수 회로에 전체 쌍을 재생한 첫 시도는 기존 모델의 필수 유효 입력 검사에서 거부돼 종료1, 출력/optimizer/차량 명령 0이다. 렌더 시작 전 수신의 50ms 상한은 camera receipt 시점까지이고 최종 decision 시점 신선도와 다르다. 별도 `scripts/audit_prior_body_freshness.py` 원본 마스크 감사에서 30km/h 원래 몸 yaw/속도 유효 732/732, 선행 몸 440/732·무효292, 연속 유효168구간·최대12행이다. pace는 원래731/731, 선행337/731·무효394, 연속 유효192구간·최대9행이다. 결과 각각 `20261031-epoch1-30kmh-prior-body-freshness-audit/`, `20261031-epoch1-brake-pace-prior-body-freshness-audit/`에 기록했다. 무효를 0으로 속이거나 원본 유효 마스크를 덮어쓰지 않았다. 모델은 해당 결측을 허용하지 않으므로 현 시점 전체 연속 비교·학습 불가다. 시각 이전 몸과 결정 시점 신선 몸을 서로 다른 시각의 입력으로 명시하는 모델/결측 계약을 설계하고, 완전 이력 재귀 상태와 비교를 검증해야 한다. Ruff 통과.

## 시각 맥락과 결정 시점 몸 감각 분리 (2026-10-31)

`src/flybrain_drive/temporal_body_context.py`는 렌더 시작 host 하한 전의 원래 몸 수신, 실제 camera group host 수신 시점에 평가한 맥락, 별도의 결정 시점 몸과 그 원래 수신/결정 시각·episode·sensor 계약을 검사한다. 동일·역전 시각, 다른 episode/계약, 과거 몸을 결정 시각의 관측이라고 바꿔치기하는 입력은 거부한다. 몸 21채널의 원래 freshness/마스크는 각 평가 시각에 적용하며 실제 노출이나 물리 샘플 발생 시각이라고 주장하지 않는다.

`scripts/build_temporal_body_context.py`가 같은 원본·prior receipt 감사에서 두 입력을 별도로 만들었다. `20261031-epoch1-30kmh-temporal-body-context/`는 732/732, `20261031-epoch1-brake-pace-temporal-body-context/`는 731/731에서 결정 몸과 camera-host 맥락 몸 각각의 yaw/속도가 자기 평가 시점에서 유효했다. 그러나 맥락 몸의 원래 수신부터 최종 결정까지 중앙/95백분위/최대는 각각 45.58/81.66/103.60ms와 51.45/90.18/111.16ms여서 이 맥락을 현재 몸이라고 사용하지 않는다. 배열·마스크·다섯 host 시각과 원본 hash 보존; no fit·actor 배포·회로 통합이다. 순서·미래·시각 바꿔치기·episode/계약 테스트 포함 14개 및 Ruff 통과. 다음은 결정 몸을 현재 제어에 그대로 쓰고, 별도로 보존된 시각 맥락을 운동 시각/시각 안정화에 연결하는 시간 명시적 회로를 설계하여 원본 대비 인과 개입과 recurrent state를 검증하는 것이다.

## 시간 분리 몸 맥락의 운동 시각 포트 연결 (2026-10-31)

`src/flybrain_drive/flight_visual_body_temporal_context.py`에 기존 네 시야 T4/T5 방향 대립 fast-minus-slow 흔적과 camera-host 맥락 몸 yaw의 곱을 기존 HS/VS 시각 포트에 투영하는 연구 경계를 추가했다. yaw ±1rad/s clamp와 시간 필터·선형 투영은 공학적 가설이며 실측 신경 tuning/연결이 아니다. 가중치는 0 초기화해 기존 시각 경계를 유지하고, yaw 무효 또는 운동 차이0이면 이 추가 경로는 0이다. 현재 결정 몸은 기존 joint body/airflow 경로에 그대로 넣고 과거 맥락은 추가 시각 포트에만 사용한다.

`TemporalContextOpponentFlightStream`은 원래 episode/receipt/render/camera/decision 시각 계약을 매번 재검사하고 기존 전체 drive·방향 시간 흔적·신경 상태를 원래 간격의 이전 입력 hold로 유지한다. 예외는 새로운 상태를 commit하지 않고 episode reset만 명시적으로 한다. 합성 mock 시험에서 0 초기화, yaw 부호/결측·운동 부재, 현재 몸 전달, 기존 시간 전개, 유효하지 않은 시간의 롤백·reset을 확인했다. 관련 검사9개 및 Ruff 통과. 실제 FlyVis/비행 graph 전체에서의 중립 동등성·과거 몸 개입·새 경로 학습, 영상/물리 발생시각 동기, 실제 바람·자율 주행과 생리 검증은 아직 없고 actor 미배포다.

## 저장 FlyVis와 과거 몸 맥락의 고정 그래프 재생 (2026-10-31)

과거 역할 TRAIN `episode-00`의 원본 이벤트·네 시야 저장 FlyVis 반응·원래 결정 몸 741개를 그대로 사용했다. 원본 raw audit와 별도 host 수신 감사에서 렌더 전 몸 479개가 50ms 경계 내, 261개는 오래됨, 1개는 선행 수신이 없었다. 원래 데이터 역할을 바꾸거나 새로운 후보의 몸을 다른 영상에 합치지 않았다. 새 시간 맥락 builder는 결측262개를 값0·마스크false·별도 결측 이유로 유지하고 현재 몸을 그대로 둔다. 결측 경계 출력은 항상0이며 생략된 신호를 센서 관측으로 삼지 않는다.

`scripts/replay_temporal_body_context_flight.py`가 `20261031-historical-train00-temporal-context-flight-replay/` 결과를 기록했다. 원본 cached body와 재구성 현재 몸 최대 차 1e-6 이내·mask/결정 시각 일치, 같은 FlyVis features, 고정 원래 graph와 기존 학습 방향 시간 경계, 독립 재귀 상태와 이전 drive hold를 사용했다. 0 초기화 새 몸-시각 경계는 741개 출력에서 원래와 최대 오차0. 임의의 사전 고정 가중치 0.01/256 probe는 기존 대비 조향/전진 평균 절대 차 3.06e-8/1.29e-9, 최대 2.24e-7/5.22e-8이다. 이는 하나의 작은 공학 probe와 이미 사용한 TRAIN 기록에서의 매우 작은 반응이며 생리 기능·학습 결과·시각/물리 발생시간 동기·코너 주행을 입증하지 않는다. 원본 role·결측·상태를 보존했고 optimizer/차량 제어/배포 0이다.

이 결과 파일은 저장됐으나 원래 process handle 54799, Python parent PID22664/child PID5308은 기록 시점에 살아 있어 종료 코드 미확정이다. 재시작·임의 종료하지 않는다. 관련 결측·시간 경계 mock 검사8개 및 Ruff 통과. 다음 작업은 같은 핸들의 종료를 확인하고 추가 경로의 실제 기울기·학습 가능성과 알려진 개발 이력의 인과 대조를 평가하는 것이다.

## 시간 분리 몸-운동 시각 경계의 단일 TRAIN 학습 사전 선언

원래 역할 TRAIN `episode-00.npz` 한 이력과 그 기록에서 엄격한 host 수신 순서로 구성한 몸 맥락만 사용한다. 결측 맥락은 마스크로 유지하며 이전 진단/DEV/TEST를 optimizer에 열지 않는다. 기존 FlyVis features·원래 현재 몸·목표·시간·교사 표적과 유효 마스크를 유지하고 기존 비행 그래프·기존 opponent 경계는 고정한다. 신규 context projection만 Adam 0.001, TBPTT64에서 수치 상태·이전 held drive·방향 흔적은 이어가며 한 이력 한 차례 제한 갱신한다. 손실은 기존 유효 조향/전진 표적의 정규화 출력 제곱오차이며 이는 응용 교사 근사로 신경 생리 검증이 아니다. 학습 전후 같은 TRAIN 고정 재생과 결측 여부를 기록하고 별도 known DEV에 대한 실제 독립 평가는 아직 없으므로 배우·차량에 배포하지 않는다. 소스/데이터 hash·결과·실패를 보존한다.

실제 저장 TRAIN00 네 시야 FlyVis·시간 분리 몸 context를 fixed flight graph에 연결한 `20261031-historical-train00-temporal-context-gradient-probe/`는 16결정 중 역사 yaw 유효10개에서 출력 제곱 기반 신규 projection gradient L2 2.9294e-6, 최대 2.1838e-7, 프로세스 종료0·optimizer 0을 기록했다. 연결성 검사일 뿐 교사 오차 개선이나 생리 타당성은 아니다. 앞선 원본 전체 replay 프로세스 handle 54799/PID22664·5308은 이 시점에도 살아 있고 종료 코드는 미확인이다.

사전 선언대로 `scripts/train_temporal_body_context_onepass.py`를 실행한 `20261031-historical-train00-temporal-context-onepass/`는 기존 episode00 TRAIN 741관측, 각 유효 표적 578개, TBPTT64·Adam .001 신규 projection 12회 갱신·종료0이다. 그래프·기존 temporal 경계는 고정, 결측 body context·상태 및 이전 입력을 보존했다. 동일 TRAIN 재생 조향 RMSE 0.043957185→0.043955881, 전진 0.103331298→0.103331298; 출력 절대 변화 평균 3.48e-6/3.84e-8, 최대 3.42e-5/3.87e-7이다. 초기 출력은 이전 3조건 원본 재생과 최대 오차0이다. 이 수치는 학습에 쓴 단일 이력으로 독립 성능이 아니며 매우 작은 변화다. 제동 표적 없음, DEV 미평가·실제 발생시각 동기/생리/주행·랩타임 미검증, 배우 미배포다. 다음은 기존 개발 이력의 시각/몸 시간 맥락을 별도 역할로 구성하고 중립·학습·결측/부호 개입을 전체 비행 그래프에 검증하는 것이다.

## 시간 분리 몸·시각 경계 known DEV 평가 (2026-10-31)

기존 개발 역할 episode02/07의 원래 영상·FlyVis·몸과 native receipt를 각각 원본 hash 및 raw 감사와 결합했다. 시각 맥락 선행 몸 유효는 497/742·506/741이고 나머지245/235는 명시 결측·추가 경로0이다. 학습에는 열지 않은 두 이력이지만 이미 여러 번 평가한 known DEV이며 독립 blind test가 아니다. `scripts/evaluate_temporal_body_context_development.py`는 원래 graph·기존 방향 경계·고정 신규 TRAIN00 checkpoint에서 기존/학습/신규 경계 부호반전 세 조건을 recurrent state를 보존해 재생했다.

조향 RMSE 기존/학습/반전은 episode02 0.06817935/0.06817781/0.06818090, episode07 0.09332076/0.09332078/0.09332077이며 무조향 0.05257177/0.09318612보다 둘 다 나쁘다. 양 조향 target>0.01 부호 일치는 세 조건 모두 각각 0/13, 7/15다. 전진 RMSE는 각각 기존/학습 0.13374032/0.13374031, 0.17454366/0.17454368이다. 학습 효과는 매우 작고 부호 반전도 실질 개선을 보이지 않는다. 이 결과는 물리/영상 발생시각 동기·생리·제동·주행 검증이 아니며 학습 후보는 배포하지 않는다. 원본 `20261031-historical-dev02-temporal-context-evaluation/`, `20261031-historical-dev07-temporal-context-evaluation/` 및 source를 보존했다. 평가 결과 파일은 생성됐지만 원래 handles 94311/44720은 확인 시 살아 있어 종료 코드 미확정이며 재실행/종료하지 않는다. 이전 전체 재생 handle 54799도 아직 살아 있다. Ruff 통과. 다음은 동일 데이터의 미세 튜닝 반복보다 네 시야 방향 선택성과 비행 유래 회로의 인과 기여, 제동/바람/시간 정합 및 새로운 독립 평가에 우선순위를 둔다.

## 시간 분리 몸·시각 경계의 실제 중간 신호 (2026-10-31)

`scripts/audit_temporal_context_signal.py`는 이전 checkpoint의 저장 normalization, 실제 저장 FlyVis scene과 원래 decision 간격의 fast/slow 방향 흔적 및 시간 분리 yaw를 재생했다. 기존 TRAIN00과 known DEV02/07에서 각각 맥락 유효479/497/506개만 집계하며 label·optimizer·차량 제어를 열지 않았다. yaw 절댓값 중앙/95백분위는 TRAIN 0.03792/0.11026, DEV02 0.005265/0.02446, DEV07 0.02560/0.03456 rad/s였다. yaw×방향 대립 흔적 L2 중앙은 0.08286/0.008075/0.05054, 학습 projection의 시각 포트 delta L2 중앙은 0.0005494/0.00005676/0.0003223이다. 유효 신호 중 최대 포트 delta L2는 0.03497/0.04476/0.05568로 희소하게 큰 값도 있었다. 저장된 learned 가중치 L2는 0.16997이다.

DEV02 맥락 yaw 및 상호작용의 중앙 크기가 TRAIN보다 작다는 관측과 앞선 미세한 최종 조향 변화는 경로 스케일 차이를 가리킬 수 있으나 비선형 회로·시간 상태·분포 차이가 섞여 있으므로 성능 저하의 유일 원인을 입증하지 않는다. 경로는 생리 실측이 아닌 공학 가설이고 host receipt의 물리 동기·독립 주행은 미검증이다. `20261031-historical-{train00,dev02,dev08}-temporal-context-signal-audit/`에 원본 신호/해시 및 실행 소스 보존, 세 프로세스 종료0·Ruff·관련 테스트14개 통과. 신규 학습 및 배포 없다.

바람 상태 재확인: 기존 CSP `ac.getWindVelocity` 후보와 차량 속도/축의 진단 수집은 존재하나 차량 위치에서의 바람 부호/좌표/물리 동기 검증이 완료되지 않았고, 기존 네 시야 기록에 승인된 실제 상대풍 입력은 없다. 무효 마스크는 유지한다. 다음은 독립 코너·속도 변동에서 입력 시점/생리와 실제 상대풍 원천을 검증하고 추가 경로의 인과 기여를 평가하는 것이다.

## 보존된 두 주행의 바람 후보와 몸 축 비교 (2026-10-31)

`scripts/audit_airflow_cross_heading.py`가 기존 `20261030-airflow-motion-owned-probe/`와 `20261030-corner4-clock-recording/`의 프레임 index·원본 hash·진단 전용/미검증 flags를 확인했다. 각각 248/422프레임 중 속도>1m/s 78/348개, 몸 전방 후보와 차량 속도 후보의 내적은 이동 중 78/78·348/348 양수였고 세 몸 축 최대 직교 오차 2.01e-7/1.82e-7, 행렬식은 약1이었다. 같은 API 집합의 기하학적 일관성으로 실제 측면 부호나 물리 좌표 등록을 독립 검증한 것은 아니다.

몸 전방 후보의 각 기록 내 수평 각도 변화는 약0.499°/4.018°에 그쳤다. 날씨 풍속 및 풍속 벡터 후보가 두 기록 전부0인데 `ac.getWindVelocity()` 크기는 0.04753–0.08578/0.07329–0.08497m/s이고 전방 투영은 -0.06531–0.04690/-0.08441–0.08143m/s로 부호가 변했다. 풍향/풍속을 의도적으로 바꾼 독립 자극이 없고 기록은 연속적이므로 두 API 의미 차이나 유효한 바람 부호·단위를 여기서 구별할 수 없다. 차속이 반영된 계산상 상대 전방 속도는 이동 구간 -4.633–-1.010/-17.060–-0.964m/s이나 wind 후보의 실제 출처를 검증하지 못해 actor 바람으로 사용하지 않는다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-cross-heading-airflow-audit/`·실행 소스·원본 hash 보존, Ruff 통과·종료0·optimizer/제어 0. 현재 SDK 정적 조회에서 풍속 조회는 확인했지만 검증된 현 세션 풍향/풍속 변경 제어 계약은 확인하지 못했다. 향후 안전한 별도 설정과 독립 자극으로 좌표·부호·시간을 검증한다.

## 독립 바람 자극 설정의 격리 준비 (2026-10-31)

설치된 CSP `ac_apps/lib.lua`의 `ac.WindParams` 주석은 풍향 degrees 및 최소/최대 속도 km/h를 명시하고, 기본 `extension/weather-controllers/base/controller.lua`는 `ac.getInputWind()`를 읽어 조건에 넣는다. `ac.setConditionsSet2`는 확인된 WeatherFX controller API이고 `ac.overrideReplayConditions`는 replay 전용이므로 현재 AC app에서 풍향을 즉시 바꿀 수 있다는 증거는 아니다. 원본 race.ini의 WIND는 방향 -1·속도 최소/최대0이다. 원본을 수정하거나 실행 세션에 적용하지 않았다.

`scripts/prepare_wind_stimulus_configs.py`와 `wind_stimulus_config.isolated_wind_copy`로 원래 WIND 섹션 세 값만 바꾸는 새 복사본 다섯 개(무풍; 36km/h의 0/90/180/270°)를 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-isolated-wind-stimulus-preparation/`에 만들고 각 hash·원본 사전/사후 hash 불변을 보존했다. 누락/중복 WIND·비유한/범위 밖 입력을 거부한다. 합성 원본 보존 및 거부와 상대풍 경계 관련 검사17개·Ruff 통과. 실제 게임이 이를 읽는 경로, 효과·방향/부호·차체 위치·physics 시간·원본 설정 복원 및 UI 없는 session lifecycle은 검증되지 않았다. 36km/h는 실험에 지정한 자극이지 측정된 바람이 아니다. 이어서 현재 engine/session 계약을 보존하면서 별도 설정의 적용·관측·복원 경로를 검증해야 하며 그 전에는 live 파일 변경/재시작·유효풍 입력·학습을 하지 않는다.

## 바람 자극 전 native AI 상태 재검사 사전 선언

현 세션 epoch1은 paused/offline/CSP AI off·brake1·camera off이나 native frozen packet285707에서 AI true로 보인다. 정지 표식의 진행 여부를 구분하기 위해 기존 CSP resume 1회 후 3초 이내 `diagnose_native_restart_packets.py`로 같은420바이트·CSP 상태·full-brake를 기록하고 pause한다. 기준 packet285707, UI·vJoy 조작·학습·자동 재시도 0. native AI 실제 제어 의미 및 바람 source 자격은 이 진단만으로 확정하지 않는다.

`20261031-pre-wind-native-ai-check-diagnostic/`은 기준285707 이후 live packet287731–288720의 서로 다른 59개에서 native AI true 59/59, 전후 CSP AI off·정지/full-brake 조건을 기록했다. `...-pause/` 목표 상태 관측, 진단/일시정지 종료0, UI·actuator 0이다. 이는 정지 packet 반복만으로 설명되지 않지만 native/CSP 중 실제 독점 제어를 증명하지 않는다. 이전 epoch0 충돌 후 session restart로 잠시 native off였던 사실이 이번 epoch1 충돌 해소를 보장하지 않는다.

설치 SDK는 `ac.restartAssettoCorsa(raceIni)`의 선택적 INI 병합을 설명한다. 그러나 현재 검증된 `FlyBrainCamera.lua`는 restart-engine에 `request.race_ini == nil`을 요구하므로 이 경로에 임의 설정을 넣거나 조건을 완화하지 않았다. 순수 `wind_restart_merge`는 0–359° 및 0–36km/h의 고정 세 WIND key만 포함하는 INI 문자열을 만들며 재시작 호출·파일 변경·원본 복원 기능은 없다. 경계/relative airflow 테스트18개와 Ruff 통과. 실제 engine merge 동작·복원·AI off 자격과 풍향/부호 독립 검증은 미완료다. 이후 운영 전에는 별도 versioned 후보·중단/복원 영수증·실제 수신 관찰/부작용 검증 및 동시 변경 보호가 필요하다.

## 바람 자극 재시작 소스 후보의 격리 시험 (2026-10-31)

`scripts/build_wind_restart_lua_candidate.py`는 현재 운영 Lua SHA `475e085d…`에서 별도 진단 command를 생성한다. 풍향 0/90/180/270° 및 36km/h 또는 0°·0km/h만 허용하고, 임의 INI를 받지 않으며 내부에서 WIND 세 항목만 문자열로 생성한다. 기존 paused/offline/CSP AI off·full brake·카메라/제어 상태·process 및 generation/epoch guard와 요청 ID별 durable intent 기록 후 1회 호출 구조를 유지한다. terminal에는 지정 풍향/속도와 원본 race hash를 남긴다. 기존 command는 임의 race_ini를 계속 거부하며 운영 Lua는 교체하지 않았다.

LuaJIT 모의에서 정상 wind-only payload·중복 reload 미실행·상태/속도/풍향/임의 INI/host attest 거부·terminal 기록 실패 시 API 미호출을 기존 engine restart 회귀와 함께 20개 통과했다. Ruff 통과. 최종 별도 후보 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-wind-restart-lua-candidate-v3/` SHA `75a94e4d85bfc29c5f4877075c6fcf39a723f5ef0751272bc402c5357423217f`, 실행 source·원본 hash 보존. 다만 native AI off와 원본 race hash는 host의 진술에 불과하고 Lua가 독립 검증하지 않는다. 실제 engine 재시작/merge, 기존 race.ini 동시 변경 보호와 자동 복원, weather controller가 풍속을 받는지, source 좌표/시간/생리·주행 영향은 미구현 또는 미검증이다. 현재 native AI 충돌 상태에서 후보 설치·실행·학생 제어는 없었다. 다음은 host 독립 native 상태와 race.ini 백업/원자적 복원·unknown outcome 회복을 구현하고 검증하는 것이다.

## 풍향 설정의 제한적 복원 dry run (2026-10-31)

`wind_stimulus_config.restore_wind_values`는 파일을 직접 열거나 쓰지 않는 순수 byte 변환이다. 현재 WIND 세 값이 소유한 실험 설정의 정확한 값과 일치할 때만 원본 세 값으로 바꾸고 그 밖의 바이트는 현재 상태대로 유지한다. 외부 소유자가 풍속을 바꿨거나 WIND 섹션/항목이 중복·누락·비수치면 거부한다. 값의 텍스트가 다르면 수치상 같아도 보수적으로 거부한다. 이 함수 자체는 파일 잠금 또는 검사와 쓰기 사이 동시 변경 방어를 하지 않는다.

`20261031-wind-restore-offline-dry-run/`은 실제 race.ini 원본 bytes를 메모리에서만 읽어 90°·36km/h 변경과 다른 섹션의 외부 변경을 모의했다. wind만 원래 값으로 복원하고 외부 변경을 보존했으며, 충돌 wind 180°를 거부하고 실제 원본 hash 불변을 확인했다. 소스 기록 hash와 스크립트 보존. `tests/test_wind_stimulus_config.py`, 상대풍·Lua lifecycle 관련 41개 통과, Ruff 통과. 실 파일 잠금·원자적 교체·프로세스 실패시 복구 및 실제 CSP merge 적용/복원·native AI 충돌 해결은 아직 없으므로 라이브 파일·카메라 설치·재시작은 하지 않았다. 다음은 실제 파일 동시 변경을 방어하고 unknown engine outcome을 보존하는 host transaction/복구를 별도 구현·시험하는 것이다.

## Windows 배타적 잠금 복원 원시 함수 (2026-10-31)

`src/flybrain_drive/wind_locked_restore.py`는 기존 파일을 Windows `CreateFileW`에서 read/write/delete 공유를 모두 금지해 열고, 같은 handle 아래 현재 WIND 값과 소유 실험값을 대조한다. 일치하면 전부 원래 세 값으로 바꾸되 다른 현재 바이트는 보존한다. 대상에 쓰기 전에 별도 신규 journal에 pre-write bytes 및 대상/전후 hash intent를 생성·flush하고, handle 쓰기 후 fsync·readback을 확인한 뒤 receipt를 저장한다. 충돌은 거부하고 실패 기록은 남긴다.

Windows 임시 파일 시험에서 잠금 중 외부 읽기·삭제 거부와 해제, 다른 섹션 변경 보존, 풍향 충돌 거부, intent 저장 실패 전 대상 불변, receipt 저장 실패시 pre-write/intent 보존을 확인했다. wind 및 airflow 검사25개·Ruff 통과. 이는 아직 live 호출자가 없는 저수준 실험 함수다. 같은 파일의 in-place 쓰기는 원자적이지 않고 프로세스 중단 시 부분 파일이 가능하며 pre-write journal은 수동 복구 자료일 뿐 자동 recovery가 아니다. directory journal의 시스템 충돌 내구성, OneDrive 동기화/외부 handle 영향, 엔진을 정지한 상태에서의 backup/재시작/restore, native AI 충돌 및 CSP 바람 반영은 미검증이다. 현재 실제 race.ini·앱·엔진은 변경하지 않았다. 다음은 임시 프로세스 중단 주입과 기존 파일 주체/재개 판정·engine lifecycle을 결합한 별도 host 후보를 검증한다.

## Windows 풍향 복원 중단 상태의 보수적 판정 (2026-10-31)

복원 쓰기 전에 journal에 before.bin과 after.bin 및 target·각 hash intent를 저장하도록 확장했다. `inspect_interrupted_restore`는 journal hash/대상 경로를 검사하고 대상에 Windows 배타 handle을 얻어 현재 bytes가 pre-write와 정확히 같은지, 의도한 결과와 정확히 같은지, 어느 쪽도 아닌지 읽기 전용으로 판정한다. 세 번째 상태는 부분 쓰기뿐 아니라 다른 주체의 변경일 수도 있으므로 원인을 추정하거나 자동 복구/덮어쓰지 않는다. 판정은 lock 보유 중 한 시점의 파일 상태일 뿐 lock 해제 후 불변이나 엔진 동작을 보장하지 않는다.

Windows 임시 파일을 대상으로 별도 child가 복원 bytes 일부만 쓰고 fsync한 직후 강제 종료하는 시험에서 종료86, before/after/intent 보존, 대상이 두 원본과 다른 부분 상태이며 검사기가 unknown으로 분류하고 bytes를 수정하지 않는 것을 확인했다. receipt 기록 실패 후 대상이 의도 결과와 일치하는 경우, 잘못된 대상 경로와 journal 변조 거부도 검증했다. wind/relative airflow 관련 총27개 테스트와 Ruff 통과. 실제 race.ini·게임은 건드리지 않았다. OneDrive 동기화/충돌, Windows 강제 재부팅·저장장치 flush, 실제 engine restart와 입력 적용·원래 풍향 복원, native AI 불일치는 여전히 미검증이다. 결과를 자동 복구 허가로 승격하지 않는다.

## 설치된 WeatherFX base controller의 합성 풍향 전달 (2026-10-31)

`scripts/probe_base_weather_wind_contract.py`는 설치된 `extension/weather-controllers/base/controller.lua` bytes/hash를 읽어 LuaJIT의 mock `ac.getInputWind()`와 `ac.setConditionsSet2()` 환경에서 변경 없이 실행했다. 무풍 및 36km/h의 0/90/180/270° 다섯 독립 mock runtime에서 해당 값이 조건 setter에 정확히 전달됐고 온도/압력/습도 등 mock 비풍향 항목은 유지됐다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-base-weather-wind-synthetic-probe/`에 설치 파일 hash, 실행 소스, 각 조건 보존, Ruff 통과·종료0이다.

이는 설치된 base controller가 입력 객체를 전달하는 로직의 합성 실행으로, 현재 세션이 그 controller를 쓴다는 증명·실제 race.ini 재시작 merge 또는 물리 바람/축/부호/시간을 검증하지 않는다. 현 세션·설치 파일·설정 변경, optimizer·actor/차량 제어는 없었다. 검증된 host 원본 백업/동시 변경 보호·native AI 충돌 해소·새 session 생명주기와 바람 센서 관측을 연결한 뒤 실제 0/반대 방향 자극을 검증해야 한다.

## 바람 실험 준비를 위한 AI 상태 충돌 session reset 사전 선언

현재 PID32912/generation `593846ULL:0.0036658` epoch1, paused·offline·CSP AI off·brake1·camera off이나 frozen native packet290835 AI true다. 앞서 사용자가 재시작을 허용했으며 별도 CSP lifecycle `restart`를 한 번 수행해 session만 초기화한다. 종료 영수증을 확인한 경우 `start` 후 pause를 복원하고, 별도 새 live/동일 generation·epoch에서 짧은 원본 packet 진단 후 다시 pause한다. 각 요청 실패/불확실하면 다음 요청·자동 재시도 없이 원본을 보존한다. reset은 native 표식 의미·독점 제어·wind 자격을 입증하지 않으며 설정·앱 파일·UI·actuator는 변경하지 않는다.

CSP `20261031-pre-wind-session-reset-epoch2/`, `...-session-start-epoch2/`, `...-session-start-pause-epoch2/`는 각각 target observed·종료0이고 epoch2로 전환됐다. 이후 `20261031-pre-wind-epoch2-native-ai-diagnostic/`의 새 packet292664–293653, 59/59에서 native AI false 및 CSP AI off·정지/full-brake 조건을 관측했고 `...-pause/` 목표 관측·종료0이다. 이는 해당 시점의 보고 일치이며 독점 제어/원자성 또는 미래 AI 상태를 보증하지 않는다. UI·actuator 0.

별도 WIND 재시작 source 후보 builder에 기존 `RelativeAirflowDiagnostic.lua`를 포함해 `wind-observe-diagnostic` 요청의 원본 바람 후보·날씨·속도·몸 축·physics timestamp·source clock을 terminal에 저장하는 정지 read-only 경로를 추가했다. paused/offline/CSP AI off/full brake·camera off 요구 및 관측 `actor_admissible=false`, `source_verified=false`를 유지한다. 0/36·정해진 풍향 이외의 INI 주입을 허용하지 않으며 native AI off 증거는 host 검증 책임이다. LuaJIT mock에서 풍속 관측이 control/restart API를 부르지 않고 live/AI/속도 조건을 거부하며 기존 restart 실패/재실행 규칙을 유지함을 포함 24개, Ruff 통과다. 격리 candidate `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-wind-restart-observation-lua-candidate-v4/` SHA `00a971aef548910806914ac570eaf8e4deac2b1cd1ba3b3a774e387e3b732c81`; 기존 v3/운영 파일은 보존·미설치다. 실제 host preflight·원본 race.ini backup/잠금·재시작 결과/복원 transaction과 실제 풍향 관측은 아직 미구현/미검증이며 생리·주행·풍향 부호 자격은 없다.

## 정지 풍속 관측 후보의 일회성 설치·복원 사전 선언 (2026-10-31)

기존 운영 Lua SHA `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119` 및 CSP/native AI off·정지/full-brake·offline·camera off·백그라운드·vJoy FREE를 확인한 경우에만 격리 후보 SHA `00a971aef548910806914ac570eaf8e4deac2b1cd1ba3b3a774e387e3b732c81`를 설치한다. 이 실험은 정지 상태에서 `wind-observe-diagnostic`을 최대 한 번 발행하며 풍향 설정 변경이나 엔진 재시작을 하지 않는다. 요청 ID 및 intent와 terminal을 보존하고, 결과가 확인된 뒤 같은 안전 조건에서 원래 Lua를 복원한다. 불확실한 요청을 재발행하지 않으며 상태가 불명확할 때는 확인 없이 복원/재시도하지 않는다. UI 조작·actuator 명령·학습은 없다. 관측치는 독립 자극/시간 동기 검증 전 actor 입력으로 사용하지 않는다.

호스트 관측 guard 및 Lua 모의 회귀 30개 통과, Ruff 통과. 이 사전 선언은 실행 성공을 뜻하지 않는다.

실행 결과: `20261031-wind-observer-preflight/` 확인 및 native AI false, speed 0.00048 km/h·gas 0·brake 1 후 `...-candidate-install/` 종료0, 후보 generation `593846ULL:0.0014031`. `...-wind-observer-live/`는 단일 요청의 terminal 및 원본 자료를 보존했고 날씨 속도 0 km/h, `wind_velocity_candidate_m_s=[-0.0045557, 0, 0.0276553]`를 기록했다. source_verified·actor_admissible·물리 동기는 false/미검증이다. 관측의 불일치는 원인 미확정이며 무풍 기준만으로 축·부호를 확정할 수 없다. `...-candidate-restore/` 종료0, 원래 운영 SHA `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119`, 새 generation `593846ULL:0.0013702`를 확인했다. 두 설치 모두 UI·actuator·AI API 호출 0, 엔진 재시작과 풍향 설정 변경 없음. 현재 실행 중 학습/운전 성과를 의미하지 않는다. 다음은 엔진 재시작과 race.ini 복원 transaction의 실제 안전성 및 독립 방향 자극 검증이다.

## 풍향 transaction 원본 보존과 읽기 전용 판정 (2026-10-31)

`wind_locked_restore.py`에 배타 handle 아래 고정 SHA의 원본 bytes와 후보 WIND bytes를 신규 journal에 저장하는 준비 함수와, 저장된 hash·대상 경로를 재검증하고 현재 풍향을 원본/후보/충돌 또는 원본=후보 모호 상태로 나누는 읽기 전용 검사를 추가했다. 원본 외 항목의 변경 가능성과 별개로 풍향만 분류하며, lock은 반환 후 해제되므로 이 판정은 이후 쓰기나 재시작의 허가가 아니다. 충돌·부분 변경을 자동으로 덮어쓰지 않고 엔진 상태도 추정하지 않는다. 임시 파일을 통한 wind 관련 23개 테스트와 Ruff 통과.

실제 `race.ini`는 잠금 읽기로 원본 SHA `6a1225d6c8ecc9b3828980e7397e43f4e8fb92c447b83ea0b1fe1d1342ef368f`를 확인하고 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-wind-transaction-real-readonly-snapshot/`에 원본·90°/36 km/h 후보·manifest·검사 기록을 보존했다. 재검사에서 전체 파일이 원본과 일치했다. 실제 파일 쓰기, 풍향 변경, 엔진 재시작이나 학습은 없었다. 독립 자극을 실행하려면 lock 해제 후 재검사와 engine request/unknown outcome/복원 전체를 연결하고 실패 경로를 검증해야 한다.

## 풍향 실험 종료 시 조건부 복원 (2026-10-31)

`finalize_wind_transaction`은 호출자가 새 엔진을 확인했다고 명시한 경우에만 저장 원본·후보 및 대상 identity를 확인하고, 배타 파일 handle에서 현재 WIND 세 값을 재판정한다. 이미 원래 값이면 쓰지 않으며, 정확히 실험값이면 다른 설정을 유지하고 WIND만 원래 값으로 복원한다. 다른 값이나 원본=후보 모호성은 거부한다. 전후 bytes와 intent를 쓰기 전 보존하고 결과를 readback한다. 동일 finalization 디렉터리의 재실행은 거부하며 부분 쓰기 실패는 기존 read-only 검사로 unknown으로 남긴다. 해당 엔진 확인은 이 함수 안에서 검증하지 않는 호출자의 주장이다.

임시 파일에서 원상태/후보/충돌·다른 섹션 보존·unknown 요청 거부·중복 실행 거부·부분 쓰기 실패를 포함하여 풍향 검사 28개, Ruff 통과. 실제 race.ini나 엔진에 이 함수를 호출하지 않았고 engine restart 발행/원래 process 종료와 신규 engine 검증/풍향 관측/복원 orchestration과 OneDrive 동기화는 미검증이다. 현재 session 점검 당시 PID32912, generation `593846ULL:0.0013702`, paused·offline·AI off·camera off, native AI false·brake 1이었다. 이는 점검 시점 상태다.

## 풍향 재시작 host 후보의 일회성 설치 및 읽기 전용 사전 검사 (2026-10-31)

기존 세션의 안전 상태를 다시 확인하고 풍향 관측 때 썼던 격리 후보 SHA `00a971aef548910806914ac570eaf8e4deac2b1cd1ba3b3a774e387e3b732c81`를 설치한다. 새 `restart_ac_engine_wind_candidate.py`는 실행 플래그 없이 실제 설정 원본, AI·브레이크·카메라·vJoy·프로세스·producer를 읽기만 검사한다. 검사 후 원래 Lua SHA `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119`를 복원한다. 이번 선언에서 엔진 재시작, 풍향 설정 변경, 조작, 학습을 수행하지 않는다. 검사 실패나 reload 결과 불명이면 상태를 확인하고 자동 재발행하지 않는다.

Host 영수증·새 process/producer 식별·풍향 계약과 기존 Lua 및 파일 복원 회귀 42개, Ruff 통과. 이는 실 엔진 재시작 결과나 복원 연계를 증명하지 않는다.

첫 host preflight는 request 0개로 실패했다. 원인은 이전 `configs/ac-axis-probe.json`의 controls SHA `4e2cbe…`와 실제 `8265c2…` 차이였다. 같은 변경의 44개 INI 섹션·항목 일치와 실제 정지 축 시험은 `20261031-live-controls-equivalence/`, `20261031-axis-current-stationary-test/`에 기존 기록이 있다. 후보 설치를 종료0으로 원래 SHA `475e085d…`로 복원했고 UI/actuator 0이다. host에 명시적 `--axis-config`를 추가해 이전에 실제 시험한 `20261031-axis-current-profile-config.json`과 현 profile hash를 다시 비교한다. 해당 변경 검사 42개·Ruff 통과. 다른 계약 변경이나 해시 검사를 생략하지 않는다. 후보를 다시 일시 설치해 읽기 전용 preflight 1회 후 복원하며 재시작 요청은 하지 않는다.

두 번째 `20261031-wind-restart-host-preflight2/`는 현재 시험된 axis config를 사용해 `read_only_preflight`, request 0, UI·actuator·race 쓰기 0, 원래 PID32912 handle active를 기록했다. `...-candidate-restore2/`에서 운영 Lua 원래 SHA 및 새 generation `593846ULL:0.001355` 확인, 종료0. 실제 engine restart는 실행하지 않았다. host는 단일 request의 durable intent와 원래 process handle의 signaled 종료, 새 PID·generation·window·producer를 확인하되 타임아웃을 성공이나 재시도 허가로 취급하지 않는다. 검사 42개는 helper/Lua/복원 시험이며 host 전체 실행·실제 PID 전환·wind 적용을 검증하지 않는다. 다음은 재시작 후 세션이 메뉴/비정지로 시작할 때도 관측과 운영 Lua 복원이 가능한 lifecycle을 검증하고, 결과 불명 시 원본 request를 계속 추적하는 통합을 마련하는 것이다.

## 실제 90°·36 km/h 풍향 merge 한 번의 검증 선언 (2026-10-31)

사용자의 재시작 허용에 따라 원본 race.ini hash 및 paused/offline/background·camera/AI off·native AI off·정지/full-brake·vJoy FREE와 pinned 후보 source를 재확인한다. WIND만 허용한 후보를 일시 설치하고 host에서 사전 준비된 90°·36 km/h 한 조건의 engine restart 요청 ID를 최대 한 번 발행한다. 원래 process handle의 종료와 새 PID/window/generation/producer 및 원본 terminal intent를 별도 확인하며 timeout/오류는 unknown으로 남기고 재발행하지 않는다. 새 엔진 상태에 맞는 기존 CSP API만 사용하여 정지 관측이 가능하면 풍속 후보를 기록한다. 실제 race.ini는 저장된 원본/실험값/충돌을 잠금에서 분류하고 원본이면 유지, 소유 실험값일 때만 WIND를 복원하며 충돌·unknown은 자동 덮어쓰지 않는다. 운영 Lua는 신규 세션이 검증된 안전 정지 상태일 때 원 SHA로 복원한다. UI·vJoy 조작·actor/학습 0. 단일 자극은 방향 부호나 독립 생리·주행 성능의 검증이 아니다. 실패·불확실 상태와 원본 기록은 보존한다.

실제 실행 중 새 session start 뒤 native brake가 0으로 바뀌어 pause가 요청 발행 전 거부됐다. 속도는 약 0.00061km/h·gas0이었다. 이는 위 선언의 vJoy 0 조건에서는 자동 복구할 수 없으므로 변경을 기록한다. 앞서 같은 현상에서 정지 축 시험을 수행한 현재 controls/profile 및 DirectInput 식별을 재확인하고, FREE vJoy를 잠시 취득해 조향0·가속0·제동1만 설정한 뒤 native의 제동1을 관측하고 장치를 해제한다. 이어 CSP pause를 한 번 발행하고 정지 관측/운영 Lua 복원을 계속한다. 시험 실패면 추가 조작 없이 남긴다. 창 클릭·키 입력은 하지 않는다.

90° 실행 결과: 원래 PID32912 종료0·새 PID26180/generation `1511102ULL:0.002739`·동일 요청 terminal 및 새 producer 확인. 새 엔진이 자동 foreground/menu로 열렸으나 창 입력 0. race.ini의 WIND가 실험값으로 변경됐고 나머지 파일은 엔진이 변경한 상태여서 `...-race-finalization/`에서 WIND만 원래 값으로 복원했다. session start 뒤 brake0으로 첫 pause 발행 전 실패, `...-brake-recovery/`에서 기존 검증된 vJoy에 조향0·가속0·제동1 두 요청 후 native brake1/FREE, CSP pause 성공. `...-observation/`에서 weather speed35km/h, weather vector `[34.99995,-0.06094]` km/h, wind candidate `[10.03632,0,-0.60329]` m/s를 관측했다. UI 입력 0. 운영 Lua 복원 후 새 generation `1511102ULL:0.001332`, paused/offline/AI off/camera off/native brake1. actor 입력과 단위/축/물리 시간 검증은 여전히 불가. 관련48검사·Ruff 통과.\n
설정 파일의 WIND는 원래 값이지만 이미 실행된 엔진에는 90° 자극이 남는다. 원래 파일을 현재 바이트 그대로 다시 보존하고 별도 0°·0km/h merge를 정확히 한 번 적용, 새 엔진 확인 후 persistent WIND 원래 값(-1/0/0)만 복원한다. 별도 후보 설치·원래 프로세스 및 새 producer 확인은 앞과 동일하며 결과 불명일 때 자동 재시도하지 않는다. 새 엔진의 안전 상태와 가능하면 무풍 관측, 운영 Lua 복원을 수행한다. 엔진 자동 foreground 외의 창 조작 없음. 엔진 내부 풍향=원본 설정을 물리적으로 완전 입증하는 것은 아니다.

무풍 복원 실행: `...-wind-clear-restart/`에서 원래 PID26180 종료0·새 PID30324 및 새 producer 확인, 0°/0km/h 요청 한 번. `...-wind-clear-race-finalization/`은 실험 WIND만 원본 -1/0/0으로 복구했고 다른 엔진 변경은 유지했다. 새 session 시작 후 brake0이 재현돼 별도 `...-wind-clear-brake-recovery/`에서 가상장치 두 full-brake 요청 후 FREE·native brake1을 확인하고 CSP pause했다. `...-wind-clear-observation/` weather vector/speed 0, wind candidate `[0.040027, 0, 0.074238]` m/s였다. 이 0/90° 비교는 자극에 따른 변화의 실 엔진 증거이나 미세 바람 모델, 36→35 차이, 좌표 부호·물리 시각 동기·상대풍·기능 기여는 확인하지 않는다.

`...-wind-clear-candidate-restore/` 종료0, 운영 SHA `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119`. 마지막 검사 당시 PID30324/generation `13832124ULL:0.0013864`, paused/offline/AI off/camera off, native AI false·speed 0.00048km/h·brake1, vJoy FREE, WIND 원본 및 무풍 후보 관측. 원래 시작 전 전체 race.ini hash와 다르나 실험 전후 엔진의 WIND 외 변경을 유지한 결과이며 새 clear 준비 직전 전체 hash `fbc34386…`와는 같다. 관련 48개 테스트 및 Ruff 통과. 다음은 상반 풍향 및 몸 축 변화 실험과 물리/영상 시각 정합, 상대풍 source qualification 후 감각·회로 인과 검증이다.

## 반대 방향 풍향 및 무풍 복원 실행 선언 (2026-10-31)

현재 PID30324/generation `13832124ULL:0.0013864`를 재확인한 후 원래 WIND(-1/0/0)와 전체 race hash를 배타 잠금에서 보존하고 270°·36km/h 자극을 한 번 발행한다. 이전과 동일하게 각각 원 프로세스 종료·새 PID/producer·terminal을 확인하고, race.ini의 WIND만 원래 값으로 복원하며, CSP session start 뒤 필요하면 검증된 vJoy로 조향0·가속0·제동1만 복구하고 pause하여 풍속 후보를 관측한다. 이후 별도 0°·0km/h 요청 1회로 엔진을 무풍으로 되돌리고 persistent WIND와 운영 Lua를 복원한다. 각 원래 요청의 결과가 불명확하면 재발행하지 않고 증거를 확인한다. UI/키 조작 없음. 두 방향과 무풍은 정지 자극 검증이며 이동 상대풍·물리 동기·학습/주행 증명이 아니다.

270° 실제 자극 결과: `...-wind-opposite-restart/` 원래 PID30324 종료0, 새 PID29984·요청 terminal/producer 확인, 요청1회. 원본 WIND 복원 후 session start·조향0/가속0/제동1 두 요청·FREE·pause로 정지하고 `...-wind-opposite-observation/`을 기록했다. weather 속도35km/h·벡터 `[-34.9999466,0.0609283]` km/h, 풍속 후보 `[-11.9671497,0,2.2823308]` m/s다. 90°와 270°의 풍속 후보 방향 cosine -0.991766이나 크기는 각각 10.0544/12.1828m/s다. 한 시점씩 서로 다른 엔진 재시작에서 관측했으므로 크기 차이의 원인·통계적 안정성·좌표 의미를 확정하지 않는다. 원본 세 result hash와 비교는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-wind-opposite-comparison.json`에 보존했다.

`...-wind-opposite-clear-restart/` 0°/0km/h 한 요청으로 PID29984 종료0·새 PID23092와 새 producer를 확인, persistent WIND 원복 및 session start 뒤 full-brake 두 요청·FREE·pause를 완료했다. 무풍 관측 weather 0, wind 후보 `[0.042497,0,0.069542]` m/s. `...-wind-opposite-clear-candidate-restore/`에서 운영 SHA `475e085d…` 복원. 마지막 검사 당시 PID23092/generation `7212754ULL:0.0014077`, paused/offline/CSP 및 native AI off·camera off·brake1·vJoy FREE, race.ini는 270° 실험 직전 원본 전체와 hash 일치. 두 재시작 각각 한 요청, 창 조작 0. actor에 연결하거나 학습하지 않았다. 다음은 조건별 반복 시계열과 몸 회전/차속 개입으로 각 센서의 시간·단위·좌표·상대풍 신뢰도를 검증하고 이후 비행 회로 기여를 평가한다.

## 풍속 후보 시간 변화 반복 관측 선언 (2026-10-31)

현재 무풍·정지·AI off 상태와 원본 WIND를 재확인한 뒤 270°·36km/h를 기존 단일 재시작 경로로 한 번 적용한다. 원본 WIND를 복원하고 정지/full-brake/FREE에서 pause하여 관측한다. 이 상태를 유지하면서 기존 CSP resume→pause를 두 번 추가해 각각 새로운 물리 진행 뒤 총 세 풍속 후보를 기록한다. 각 lifecycle 요청과 관측 요청은 새 ID를 한 번만 발행하고, 속도·가속·제동·AI 조건이 맞지 않거나 결과가 불명이면 즉시 중단한다. 진단은 actor 입력이나 학습에 쓰지 않는다. 이후 원래 WIND와 실행 엔진을 무풍으로 복원하고 운영 Lua를 되돌린다. 부득이 새 엔진 시작 뒤 brake0이면 앞서 검증한 vJoy 정지 brake hold만 복구한다. UI 입력은 없다. 세 표본은 조건별 시간 변화의 탐색이며 통계적 안전성·좌표·물리 시간 정합을 확정하지 않는다.

동일 풍향 반복 결과: `...-wind-series-restart/`에서 원 PID23092 종료0, PID19164 및 새 producer 확인; 270° 요청1회, persistent WIND 복원. session start 후 full-brake 두 요청·FREE·pause, 관측0 이후 두 차례 CSP resume/pause와 관측1/2 모두 완료했다. weather 벡터는 세 번 모두 `[-34.9999466,0.0609283]` km/h이고 wind 후보 벡터는 각각 `[-11.61087,0,-2.51335]`, `[-10.97161,0,-2.82085]`, `[-10.74728,0,-2.88799]` m/s, 크기 11.8798/11.3284/11.1285m/s였다. 원본 source clock 41.060/69.029/98.508초·frame 34498/58716/84090이며 각 원본 result hash와 계산은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-wind-series-comparison.json`. 동일 설정·정지 상태의 시간 차이에 따른 후보 차이를 관측했으나 난류 여부/시간 동기·통계 분포를 확정하지 않는다.

`...-wind-series-clear-restart/` 무풍 요청1회 뒤 PID19164 종료0·PID23448/producer 확인, 설정 WIND 원복, start 뒤 full-brake 두 요청·FREE·pause 후 weather speed0 및 wind 후보 `[0.073937,0,-0.032683]`m/s를 관측했다. `...-wind-series-clear-candidate-restore/` 운영 SHA `475e085d…` 복원. 최종 점검 PID23448/generation `13831178ULL:0.0014227`, paused/offline/AI off·camera off·native brake1·vJoy FREE, race.ini는 실험 전 전체 hash `fbc34386…` 일치. UI 입력·actor 사용·학습 없음. 다음은 속도/차량 자세를 독립 조작하는 실제 상대풍 및 감각운동 인과 실험이다.

## 실제 바람과 저속 이동의 결합 진단 선언 (2026-10-31)

기존 무풍 이동 기록과 비교하기 위해 현 설정 원본·엔진 정지/AI off/full-brake·장치 FREE를 확인하고 90°·36km/h WIND를 기존 1회 CSP engine restart로 적용한다. 원본 WIND를 보존·조건부 복원한 뒤 일시정지 상태에 도달하면 기존에 실행 검증한 `FlyBrainCamera-airflow-motion-diagnostic-v1.lua`를 고정 hash로 일시 설치하고, 기존 owned camera·CSP engine AI의 최대 3초/15km/h 진단을 1회 실행한다. 카메라/AI 종료와 pause, 진단 raw를 확인하고 운영 Lua를 복원한다. 이후 0°·0km/h engine restart 1회로 무풍 복원하고 원래 WIND와 운영 Lua를 복원한다. 각각 결과 불명 또는 안전 조건 실패는 재시도하지 않으며 기존 데이터와 영수증을 보존한다. 새 엔진 start 뒤 brake 복구에 필요한 가상장치의 조향0·가속0·제동1만 허용한다. UI 입력·학생 actor·학습 없음. 영상/바람/물리의 원자적 동기 및 비행 생리 증명이 아니며 진단 이력을 훈련에 사용하지 않는다.

이동 진단 실패 원본: `...-wind-motion-recording/` control request ID `817afa8e5bdd48cfab9aa5268f396e3a`는 CSP rejected·AI API 0, camera는 stop 후 원본 180프레임을 보존했다. 첫 프레임 source clock 9.2784358, 거부 시각 16.4561222, 원래 camera deadline 15초에서 남은 시간은 시작이 첫 프레임보다 늦을 수 없으므로 최대 7.822314초다. Lua owned camera는 3초 이동 + 5초 여유인 최소 8초를 요구하므로 이 조건은 반드시 실패했다. 다른 조건의 동시 실패 가능성은 배제하지 않는다. 정지 180프레임의 weather는 35 km/h, 풍속 후보 크기는 모두 12.9231m/s였으며 physics가 정지된 이 표본을 독립 시계열이나 이동으로 취급하지 않는다. frame SHA와 산식 `.../20261031-wind-motion-failure-forensics.json`.\n
무풍 복원 `...-wind-motion-clear-restart/` 종료0·PID17988 신규/원 PID12660 종료, WIND 원복, full-brake 복원과 pause, weather 0 관측 후 운영 SHA `475e085d…` 복원했다. 다음 신규 실험을 위해 recorder의 camera 설정을 250×40ms/deadline15초에서 후보 한도 내 300×80ms/deadline30초로 변경했다. AI 운전 시간·속도·안전 경계는 변경하지 않는다. 관련 Lua 회귀 42개와 Ruff 통과. 아직 새 입력 조건은 실증하지 않았다.

## 수정된 카메라 예산의 새 바람·이동 진단 선언 (2026-10-31)

실패한 요청은 재발행하지 않는다. 300그룹·80ms·deadline30초 신규 camera 및 신규 CSP 요청으로, 이전과 동일한 90°/36km/h 자극과 최대3초/15km/h engine AI 진단을 별도 한 번 수행한다. 기존 소유·신선도·brake·AI·대상 guard, 실패 시 중단 및 원본 보존, WIND와 무풍 엔진/운영 Lua 복원은 그대로다. 입력과 결과는 진단 전용이며 학습·학생 제어·UI 입력 없음.

수정된 `...-wind-motion-v2-recording/`은 별도 신규 camera 요청으로 종료0·146프레임·owned AI control 종료0, 학생 actuator/UI 0. 이후 camera 정지·CSP paused/AI off/full-brake 상태에서 무풍 복원용 `...-clear-restart/`는 native same-copy AI true로 lifecycle 요청 발행 전 거부됐다. frozen native 표식과 CSP AI off의 의미는 미확정이다. 기존에 성공한 CSP session restart/start로 새 epoch를 만든 뒤 정지/full-brake·native AI off를 확인하고, 그때만 무풍 복원용 신규 엔진 요청을 발행한다. 재시작 실패/표식 불일치 지속 시 원본을 남기고 다음 요청을 하지 않는다. UI 입력 없음.

새 예산 실험 결과 `...-wind-motion-v2-recording/` 종료0·owned engine AI 3초 종료0·camera stopped·원본 146그룹. `...-wind-motion-v2-strength-audit/`은 source frame 및 hash 검사 후 35개 이동 표본(차량 속도 후보 >1m/s), 몸 전방속도 1.0736–4.6052m/s, weather speed 35km/h, wind 후보 크기 12.1336–12.2371m/s, wind−vehicle의 몸 전방 후보 -1.9148–1.3310m/s를 기록했다. `...-wind-motion-v2-telemetry-audit/`에서 source clock 근접 38쌍의 속도 차 중앙 0.00342km/h·최대 0.5254km/h, 최대 시각 간격 21.42ms다. 근접 시각 비교는 동일 물리 snapshot·카메라 노출 정합이 아니며 풍향의 차량 위치 물리 의미·측면 부호·상대풍 생리 및 모델 기여는 미검증이다. 원본 진단은 actor/학습 금지, UI·학생 actuator 0.

AI 종료 후 CSP off/native true여서 첫 무풍 복원 preflight는 요청 발행0으로 중단됐다. CSP session restart/start 후 frozen packet은 여전히 true였으나 resume 상태의 새 packet 59개는 native false 59/59였다. pause 복원 뒤 새 0°/0km/h 요청1회로 원 PID7344 종료0·새 PID11528/producer 확인, WIND 원복 및 start 뒤 full-brake/FREE·pause, weather 0 관측과 운영 Lua SHA `475e085d…` 복원했다. 마지막 점검 generation `9177494ULL:0.0013521`, paused/offline/CSP 및 native AI off·camera off·brake1·vJoy FREE, race.ini 전체 hash는 실험 전 `fbc34386…`와 일치. native/CSP 표식의 독점 제어 의미는 여전히 미검증이다. 다음은 동일 조건 무풍 대조와 반대 풍향·다른 몸 방향의 추가 기록을 시간 계약에 맞춰 비교한 뒤 비행 회로 입력의 인과 기여를 검사한다.

## 무풍 저속 이동 대조 선언 (2026-10-31)

앞선 90° 자극 이력과 같은 초기 track·car 및 시작 위치를 확인하고, 현재 무풍 weather 관측·정지·AI off/full brake·vJoy FREE 및 운영 Lua hash를 재검사한다. 기존 진단 후보 SHA `906a4be9d202df921a064303b0a3a28df0d9ed8b215e6cbca879032a55b518ed`를 일시 설치하여 300그룹·80ms·deadline30초, 최대3초/15km/h owned AI 진단 신규 요청 1회를 수행한다. 실패·불명확 시 재발행하지 않으며, 종료 후 카메라/AI 및 full-brake 확인하고 운영 Lua를 복원한다. 진단 이력은 훈련과 actor에 사용하지 않고, 두 연속 이력은 서로 독립 엔진 재시작이며 정확한 trajectory/시각 매칭이나 생리 검증이라고 가정하지 않는다. UI 입력 없음.

기존 GET-only observer는 자동으로 열린 foreground를 이유로 요청 전 거부했다. 창의 focus를 변경하지 않고 같은 native graphics 264bytes를 repeated-copy 및 기존 decode로 읽은 별도 시점의 spline은 0.8564317822·lap0이며 바람 이력의 AI receipt 시작 spline 0.8564317822와 일치한다. 현 CSP paused/AI off/camera off, native AI false·brake1, vJoy FREE 및 운영 Lua hash를 별도로 관측했다. 이는 한 시점 읽기이며 기존 observer의 두 graphics 증가 검증을 통과했다는 뜻이 아니다. 창을 클릭하거나 focus를 변경하지 않는다.

무풍 진단 이후 CSP paused/AI off이나 frozen native AI true가 다시 관측됐다. 앞서 같은 현상에서 효과를 본 기존 CSP session restart/start 후 짧은 resume 중 새로운 native packet을 관측하고 pause하여 일치 여부를 확인한다. 각 원래 요청 불확실 시 재발행하지 않으며 UI/가상장치 조작 없음. 이 표식 검사는 독점 제어 증명이 아니다.

무풍 대조 `...-wind-zero-motion-recording/` 종료0·136그룹, 이동35개·후보 전방속도 1.3283–4.6058m/s. weather 0, wind 후보 크기 0.0763–0.0848m/s. 같은 시작 spline 원본 `...-wind-zero-motion-position-check/`; 두 주행의 실제 궤적 일치나 동시는 미검증이다. `scripts/audit_wind_speed_matched_diagnostics.py`는 두 원본 metadata와 모든 frame SHA를 검증하고 속도 차 ≤0.2m/s의 34쌍을 만들었다. 무풍 표본 27개를 사용하므로 표본 재사용이 있다. (wind−vehicle)의 각 이력 자체 몸 축 투영을 뺀 바람−무풍 차의 전방/좌측/상방 중앙은 `[3.7154,11.4524,0.0831]`m/s이고 최대 속도 차 0.1833m/s다. 독립 34개 표본이나 부호 검증된 물리 상대풍으로 해석하지 않는다. `...-wind-speed-matched-audit/` 원본 및 소스 보존, Ruff 통과. 무풍 38 source-clock 근접 telemetry 쌍 최대 간격 16.37ms·속도 차 중앙 0.00168km/h이나 동일 물리시각은 아니다. 훈련·학생 actor 0, UI 입력0.

카메라 운영 SHA `475e085d…` 복원 후 CSP AI off/native frozen true여서 session restart/start를 수행했다. 시작 후 첫 59회는 모두 기존 packet13123/native true로 반복됐으므로 새 표본이나 AI off로 판정하지 않았다. pause 후 resume 별도 조사에서 packet15996–16985, 59/59 native false를 관측하고 pause했다. CSP API의 종료0과 native 표식 독점 제어는 별개다. 당시 PID11528/generation `9177494ULL:0.0014011`, epoch1, 무풍 설정과 운영 Lua, pause/full-brake/FREE. 다음은 다른 차체 방향/반대 풍향과 물리 시각 검증 및 회로의 인과 개입이다.

## 반대 풍향 저속 대조 선언 (2026-10-31)

동일 시작점·정지/full-brake/FREE·native/CSP AI off와 원래 WIND를 재확인하고 270°·36km/h를 신규 단일 engine restart 요청으로 적용한다. WIND만 원본에 복구한 뒤 기존 소유 카메라의 300×80ms/deadline30초 및 최대 3초/15km/h AI 진단을 신규 요청 1회 수행한다. raw·terminal/종료를 보존하고 무풍 엔진·원본 WIND·운영 Lua를 복원한다. native frozen AI 표식 충돌 시 이전처럼 session restart와 새 진행 packet 관측 후에만 복원 재시작을 발행한다. 모든 결과 불명은 중단·확인, 같은 발행 ID는 재발행하지 않는다. UI/학생 actor/학습 없음, 진단은 훈련 금지다.

`...-wind-negative-motion-recording/`은 270° 조건 3초 AI 제어 종료0, camera stopped, 129그룹, 이동35개. `...-wind-negative-motion-strength/`는 전방속도 후보 1.1472–4.6026m/s, weather 35km/h, wind 후보 크기 12.8674–12.9758m/s를 기록했다. `...-wind-negative-speed-matched-audit/`는 무풍 기록과 속도 차 ≤0.2m/s인 35쌍(무풍 24개 재사용)을 비교해 각 기록 몸 축 상대풍 후보 차 중앙 `[-2.2075,-12.6590,-0.0930]`m/s를 계산했다. 앞선 90° 비교의 좌우축 +11.4524와 반대 부호다. 다른 엔진/시간·중복 대조·각자 몸 축·후보 자료여서 독립 표본, 정확한 물리 wind/시간 동기, 비행 회로 생리 또는 모델 성능 입증이 아니다. 현 airflow adapter는 0 초기화이며 이 실험을 학습하지 않았다.

AI 중지 뒤 frozen native true여서 새 session epoch restart/start/pause/resume 후 59개 진행 packet에서 native false 59/59, pause했다. `...-wind-negative-motion-clear-restart/`에서 원 PID28712 종료0·새 PID12432/producer 확인 후 WIND 원복, start 뒤 full-brake 두 요청·FREE·pause 및 weather0 관측, 운영 Lua SHA `475e085d…` 복원했다. 최종 PID12432/generation `15533904ULL:0.001457`, paused/offline/CSP 및 native AI off·camera off·brake1·vJoy FREE, race.ini 전체 hash `fbc34386…` 유지. UI 입력·학생 제어·학습0. 다음은 원본 wind 자격/시간 정합과 별개로 실험 값의 오프라인 비행 회로 인과 기여를 검토하고, 학습 가능한 풍속 경로와 독립 평가를 준비한다.

## 실제 후보 값의 비동기 회로 개입 사전 선언 (2026-10-31)

기존 60km/h known DEV04의 저장 네 시야 FlyVis·몸 이력과 학습 완료된 기존 비행 그래프/시각 시간 경계를 고정한다. 별개 90°/270°/무풍 3초 진단의 해시 검증 원본에서 이동 상대풍 후보(바람−차속을 각 원본 몸 축에 투영한 공학 가설)를 가져와 다른 DEV 이력의 상대 순서에 맞춰 늘려 넣는다. 두 기록은 동시·동일 주행이 아니며 이런 시간 확장은 반사실 fixture다. 모델 내부 검사용으로만 가짜 유효 mask를 주고 진짜 sensor가 검증됐다는 뜻으로 쓰지 않는다. 동일 시각·몸·시각 입력을 고정한 상태로 결측, 기존 0 초기화 투영에서 90° 후보, 인위적인 측풍→몸 포트 단위 가중치에서 90°/270°/무풍 후보를 각각 별도 recurrent state로 비교한다. 0 가중치 조건은 결측과 동일해야 한다. 임의 가중치는 실측 뉴런/학습·생리가 아니고 입력 간 인과적 수치 민감도만 확인한다. optimizer·배포·실차 제어0, 실제 풍원/시간 자격·독립 운전 검증과 구별한다.

`.../20261031-recorded-wind-counterfactual/`은 독립 개발 DEV04 739관측의 FlyVis 네 시야·원래 몸/시간과 기존 고정 비행 그래프를 재생하면서 별도 세 진단에서 1m/s 초과 moving 후보를 순서상 늘려 가정 입력으로 넣었다. source frame/metadata hash와 원래 모델·데이터 계보를 검사했다. 실제 결측과 0 초기화 바람 경로에 90° 후보를 넣은 대조의 최대 출력 차는 정확히 0, 기존 baseline 재현 오차 2.24e-7이었다. 임의 unit 측풍→몸 포트 계수를 넣었을 때 동일 무풍 후보 대비 90°/270° 가정 입력의 평균 signed 조향 변화는 +0.006644/−0.006290, 전진 변화는 −0.004959/+0.004756이었다. 이는 반대 방향 수치 민감도이며 계수는 합성, source-valid mask도 fixture용이다. 원래 영상·몸과 풍속은 다른 주행으로 물리 동기 없음, 현 모델이 바람을 학습했거나 실제 생물학적 회로/주행 이득을 증명하지 않는다. optimizer·차량 제어0, 종료코드0, 관련 테스트8개·Ruff 통과. 예측 SHA `334e201d5ca01450b212ab32edb57734338e141b1cb231c440b6bacf17ba5354`. 다음은 독립 원천의 시간/좌표 검증과 동일 주행 감각을 확보한 뒤 실제 학습 및 반대풍 제거·전환의 폐루프 기여를 검사한다.

## 일시정지 물리 시각 후보 대조 선언 (2026-10-31)

기존 실패 camera 180프레임은 AI API가 발행되지 않았고 후보 진단 조건상 정지 중이었다. 그 기간 source clock은 7.159초, `car.timestamp` 후보는 7.195초 진행했다. 이 두 clock의 유사성은 timestamp가 물리 발생 시각이라는 증명이 아니므로 현재 무풍 paused/offline/CSP 및 native AI off/full-brake/FREE에서 wind 진단 후보를 일시 설치하고, 서로 다른 두 read-only 풍속 관측 전후에 native physics packet ID와 exact bytes hash를 보존한다. 같은 frozen physics인지 확인하고 timestamp/source clock 변화를 계산한 뒤 운영 Lua 원본으로 복원한다. 창 입력·actuator·학습·주행 0, 관측은 actor에 사용하지 않는다.

현재 paused 진단 `...-wind-pause-clock-observation0/`, `...-observation1/` 및 사후 native snapshot에서 동일 packet11787과 동일 native 420-byte SHA `ba0a66a9…`를 확인했다. 두 CSP 읽기 사이 `os.preciseClock`은 9.4072727초, `car.timestamp` 후보는 9.407125초 증가했고 세 native 수신은 서로 다른 host 시각이다. 따라서 이 구간에서 해당 값 증가가 native physics 새 표본을 뜻하지 않는다. 두 인터페이스가 동일 표본을 가리킨다는 증거도 아니며 clock 자체의 정확한 정의는 미확정이다. 원본 두 result SHA 및 계산 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-wind-pause-clock-comparison.json`. 앞선 실패 camera 180프레임의 paused 구간에서도 source clock 7.159초, car.timestamp 7.195초가 진행했다. UI/actuator 0, actor/학습0. 운영 Lua `475e085d…` 복원 후 PID12432/generation `15533904ULL:0.0014004`, paused/CSP 및 native AI off/camera off/brake1 확인.

기존 `audit_airflow_motion_axes.py`와 `audit_airflow_motion_strength.py`의 새 결과 출력에서 `physics_step_ms_min` 등 오해 소지가 있는 이름을 `car_timestamp_candidate_step_ms_min`으로 고치고 이 값은 물리 표본 시각으로 미검증임을 명시했다. 과거 원본 JSON은 변경하지 않는다. 새 `...-wind-motion-v2-clock-naming-audit/`에서 기존 146개 source 검사를 재실행해 종료0, Ruff 통과. 다음은 생산자 physics 원본의 실제 시간 경계 또는 새로운 동기 계약을 확보해 영상·몸·바람의 인과적 유효성을 검증하는 것이다.

## native 패킷 발행 시각 조건부 추정 (2026-10-31)

`scripts/audit_native_publication_windows.py`는 앞서 감각 수신을 영상보다 선행시킨 기존 두 raw 이력의 prefix bytes/hash, packet ID, 원래 event/receipt audit를 대조하고, 선택 패킷 수신 직전 다른 packet 수신과 해당 수신 사이 간격을 계산한다. 이 구간이 발행을 감싼다는 해석에는 mapping이 일관되고 단조 발행되며 이전 복사본이 복사 후 기록된 수신 시각까지 최신이었다는 미검증 가정이 필요하다. 원본에는 이전 복사 시작 시각이 없고 발행 전 물리 샘플 생성 지연도 모르므로 발행 또는 실제 물리 나이 상한을 증명하지 못한다. 기존 카메라/물리 동시성 주장으로 승격하지 않는다.

30km/h `...-30kmh-native-publication-window-v3/`은 732개 선택의 연속 수신 간격 p50/p95/max 31.858/32.787/33.464ms, 결정−선택 수신 45.581/81.662/103.603ms. 위 가정 아래 결정−이전 수신은 77.467/113.088/134.730ms, 50ms 초과707/732다. pace `...-pace-native-publication-window-v3/`은 각각 수신 간격 31.817/32.709/34.290ms, 결정−선택 수신 51.452/90.177/111.157ms, 조건부 값 83.425/122.250/142.828ms 및 725/731 초과다. 이것은 선택 입력의 receipt 기반 신선도와 실제 source age가 다를 수 있음을 보여 주는 조건부 민감도이며, 707/725개 물리 표본이 실제로 오래됐다는 주장이 아니다. 원본 hash·행별 계산 보존, 순서 경계 검사 포함 9개 테스트·Ruff 통과, 학습/actor/차량 제어0. 다음은 producer-side 직전/현재 copy 시작·종료 및 source update 계약을 기록해 신뢰 가능한 시각 구간을 얻고 영상·몸·바람의 같은 실행 이력을 검증하는 것이다.

## native copy 시작/종료 직접 기록 선언 (2026-10-31)

현재 무풍·정지/full-brake·offline·AI off·camera off·FREE 및 운영 Lua를 재확인한다. 기존 CSP resume을 1회 발행하고 가속0·브레이크1·속도<0.1km/h를 유지하는 동안 3초간 동일 native 420-byte mapping을 두 번 복사하며 각각 시작/중간/종료 host `perf_counter`와 정확한 bytes/hash·packet ID를 보존한다. CSP 상태는 주기적으로 재검사하고 native AI true 또는 정지 조건 위반이면 즉시 중단한다. 이후 기존 CSP pause를 확인한다. UI/가상장치/AI/학습 제어0. 서로 다른 packet의 전후 copy 시작~종료 구간은 일관된 단조 mapping 관찰이라는 조건에서만 publication 전환 구간 후보이고, 물리 표본 발생 시각이나 카메라 노출 동기화는 아니다. 같은 바이트 두 번 읽기는 engine seqlock이 아니다.

새 `timed_teacher_snapshot`은 native 420 bytes를 각각 읽기 전·두 읽기 사이·완료 후 host `perf_counter`를 기록하고 같은 bytes일 때만 반환한다. 기존 독자/actor 동작은 변경하지 않았으며 clock 역행과 중간 복사 불일치 재시도 등 12개 테스트·Ruff 통과. 두 copies가 같은 것은 engine seqlock/생성 시각이 아니다.

실제 `...-native-copy-window-resume/` 이후 `...-native-copy-window-diagnostic/` 3초에 1,209관측, native AI true 0, 서로 다른 packet 전환 997, 같은 packet에서 byte 변화 0. packet14950→15948에서 996 전환은 ID+1, 한 번은 ID+2여서 중간 ID 하나를 관측하지 않았다. 이중 복사 시간 p50/p95/max 0.0057/0.0106/0.0294ms, 이전 copy 시작부터 다음 다른 packet copy 끝까지의 구간 p50/p95/max 2.526/2.781/3.491ms다. 일관된 단조 mapping 발행 가정 아래 보이는 전환 구간일 뿐 샘플이 물리 계산된 시각, engine 원자성, 영상 노출 또는 빠진 packet의 개별 시각을 보장하지 않는다. rows와 hash·packet 감사 보존. 이후 `...-native-copy-window-pause/` target 관측·종료0, UI/actuator0. 이 정지 무풍 실험은 학습·주행 성능이나 움직임 중 부하에서의 측정이 아니다. 다음은 이 복사 경계를 실제 네 시야 renderer와 함께 기록하여 인과 수신 범위를 좁히고, 별도 물리 생성·노출 timestamp 계약을 검증한다.

## 네 시야 렌더와 native copy 구간의 동시 host 관측 선언 (2026-10-31)

운영 Lua SHA `475e085d…`, 현재 engine·offline/paused·native 및 CSP AI off/full-brake·vJoy FREE와 background를 재확인한다. CSP resume 1회 뒤 기존 source clock ping과 2그룹 네 시야 camera probe를 실행하는 동안 native 420-byte copy 시작/중간/종료 및 원본 bytes/hash를 최대 20,000회, 약 2ms 간격으로 기록한다. same-generation CSP와 native 정지/AI off/brake guard를 유지하고 원래 camera child 결과를 추적한다. 이후 pause 1회. 각 view render-start host clock 범위 앞에 끝난 마지막 copy와 범위 뒤에 시작한 첫 copy를 비교한다. 두 clocks의 offset은 짧은 ping 전후 unit-rate 가정에 한하며 copy 중 publication 일관성/물리 표본 발생·영상 실제 노출/동시성은 미검증이다. UI·actuator·학습0, 새 감각은 actor에 공급하지 않는다. failure/unknown child는 원래 handle을 확인하고 재발행하지 않는다.

`...-camera-copy-window-resume/` CSP resume 후 `scripts/audit_live_camera_copy_windows.py`는 기존 camera probe의 2개 네 시야 그룹과 native 420-byte 이중 복사 2,562회를 동일 host perf_counter 상에서 수집했다. probe child 종료0·원본/복사본 저장, source generation 동일 및 전후 CSP clock 교환 교집합 2.5874ms다. 8개 view 모두에서 render-start host 범위 전체가 이전 native 두 번째 copy 종료와 다음 첫 copy 시작 사이에 있었고 그 바깥 copy 사이 간격의 최소/중앙/최대는 4.2709/4.81785/5.2951ms였다. native 원본 2,562개 hash·각 copy 시간 순서와 AI false를 독립 확인한 `.../independent-copy-check.json` 보존. 관측 중 offline·정지·gas0·brake1, UI/actuator0. 이후 `...-camera-copy-window-pause/` target 관측·종료0이다.

이는 두 번의 정지 진단에서 source clock 변환의 unit-rate 가정에 따른 render 호출과 host copy 경계이며, 이전 copy가 영상의 물리 순간을 포함했다는 증거나 실제 exposure/physics 생성 시각·모든 주행에서의 지연 보장이 아니다. 중복/빠진 packet과 두 복사의 원자성 미검증을 계속 구분한다. 데이터는 actor/학습에 미사용. 새 코드 `scripts/audit_live_camera_copy_windows.py`, native timing 유닛 12개·Ruff 통과. 다음은 동일 copy 구간을 실제 주행 기록 경로에 격리해 적용하고, 센서 수집 부하 및 반복 데이터를 비교하며 source-side/노출 계약을 계속 검증한다.

## 저속 주행의 네 시야·native copy 시간 동시 기록 선언 (2026-10-31)

현재 무풍·정지/full-brake/offline·CSP/native AI off·카메라 off·vJoy FREE 및 운영 Lua를 확인하고 기존 진단 camera 후보 `906a4be9…`를 일시 설치한다. 새 `record_ac_airflow_timed_diagnostic.py`는 기존 소유 camera 및 최대3초/15km/h CSP engine AI를 별도 단일 요청으로 실행하면서 제어 child와 독립적으로 native 420-byte 2회 복사의 시작/끝·원본 bytes와 네 시야 source-clock을 수집한다. camera 전후 ping으로 짧은 구간의 단위 속도 clock 가정하 host 범위를 계산하고, 각 렌더 시작 범위 이전에 끝난 copy와 이후 시작한 copy만 연결한다. 읽기 실패나 child timeout은 원래 핸들을 추적하고 재발행하지 않는다. 원본/terminal, full-brake·AI off를 확인한 뒤 운영 Lua 복원. UI 입력·학생 actor·학습0이며 물리 생성/카메라 노출 시각 및 비행 생리는 미검증, 이력은 학습 금지다. 경계 검사 14개와 Ruff 통과.

주행 뒤 CSP AI off이고 camera off·pause/full-brake이나 frozen native packet28605 AI true가 나타났다. 이 상태를 off로 간주하지 않고, 기존 절차대로 CSP session restart/start/pause/resume 후 3초 새로운 native packet의 AI 표식을 진단하고 pause한다. 각 요청은 별도 한 번이며 실패/불확실 시 재발행하지 않는다. UI/actuator0, 진단은 독점 제어를 증명하지 않는다.

신규 `scripts/record_ac_airflow_timed_diagnostic.py`는 기존 camera 300×80ms/deadline30초·3초/15km/h engine AI 소유 절차를 유지하면서 control child 실행 중 native 420-byte 두 copy와 각 host 시작/종료를 기록한다. camera 전후 CSP ping, 원본 metadata/hash·copy 전체·각 시야 render-start host 범위 및 그 앞뒤 copy를 보존한다. 원본 `...-moving-copy-window-recording/` 종료0·camera stopped·control 종료0, 138그룹/552시야·native 4,318개(copy 중 native AI true 1,722개). 532시야의 host render 범위 앞뒤 copy 간격 최소/중앙/최대 4.1641/5.17375/8.2855ms, clock 교환 불확도 3.4593ms다. 나머지 20시야는 그룹0–3 및137로, control child 실행 중에만 native 수집하는 현 설계에서 전후 관측이 없었다. 이동 후보 속도 >1m/s인 35그룹의 140시야는 140/140 host 경계가 있다. `.../independent-copy-check.json`에서 전체 4,318 raw SHA/시간 순서를 재검사했다. 이는 한 저속 이력의 host render 호출·메모리 copy 순서이며 실제 physics 샘플 발생이나 노출 동기·다른 부하의 성공을 보장하지 않는다. 진단 이력은 actor/학습 금지, UI/학생 actuator0. 경계/reader 테스트14개·Ruff 통과.

종료 후 CSP off/native frozen true가 반복되어 운영 Lua 원본 SHA `475e085d…` 복원 뒤 CSP session restart/start/pause/resume으로 새 epoch를 만들고 진행 packet32238–33229에서 native AI false 59/59를 관측했다. `...-moving-copy-native-reset-audit-pause/` pause target·종료0, 당시 PID12432/generation `15533904ULL:0.0013834`. 독점 제어·물리/노출 시간은 여전히 미검증이다. 다음은 camera 전후 구간을 덮는 native sampling lifecycle과 빠진 view를 명시한 입력 마스크를 마련하고 고속/코너·부하에서 별도 검증한 후 학습 자격을 판단한다.

## 전 구간 native copy 기록의 저속 진단 선언 (2026-10-31)

native sampler를 별도 제한 thread로 camera 요청 전 시작·준비 확인하고 camera terminal을 확인한 뒤 정지하도록 수정했다. 최대 20,000개·약 2ms 간격, 오류와 부분 원본을 보존하며 제어 child의 종료를 기다리는 동안 복사 실패를 이유로 임의로 child를 중단하지 않는다. 기존 3초/15km/h engine AI·카메라·CSP gate와 UI 금지는 그대로다. 현재 무풍·정지/full-brake/offline·CSP/native AI off·카메라 off·FREE·운영 Lua 확인 후 진단 후보를 일시 설치해 신규 요청1회, 종료 후 기존 Lua 복원 및 native AI 표식 검사를 수행한다. 오류/불명 outcome은 원래 handle을 추적하고 재발행하지 않는다. 경계/reader/sampler 검사15개·Ruff 통과. camera 호출 전후 host copy의 존재는 물리 발생·실제 노출 동기나 학습 자격을 뜻하지 않는다.

첫 전 구간 결과: `...-moving-copy-full-recording/` control 종료0·132그룹·528/528시야 host bracket이지만 native sample 689개, copy 시작 간격 중앙/95백분위/최대 15.373/16.218/17.019ms, 시야 bracket 간격 중앙/95/최대 15.507/31.016/32.622ms. packet 전환 270회에서 중간 ID 1,126개 미관측이다. 별도 같은 머신 thread 대기 30회에서 `Event.wait(0.002)` 중앙15.616ms, `time.sleep(0.002)` 중앙2.266ms를 측정해 시간 양자화가 유력하나 다른 부하의 영향은 배제하지 않는다. 원본 hash 감사 `.../independent-copy-check.json`, 운영 Lua 복원. 현 CSP AI off/native frozen true는 실제 독점 제어 여부가 미확정이다.\n
수정된 sampler는 동일 stop flag를 검사하되 wait 대신 2ms sleep을 사용한다. 단위15개·Ruff 통과. 기존 CSP session restart/start/pause/resume 뒤 새로운 packet에서 native AI off를 확인해 pause한 경우에만, 수정된 전체 camera 범위 sampler와 기존 3초/15km/h AI 진단을 신규 요청으로 한 번 실시한다. 종료 후 운영 Lua 복원 및 반복 표식 확인; 필요시 같은 절차로 현장 pause를 복원한다. UI 입력·학생 actor·학습0. 기존 기록을 덮어쓰지 않고 source/exposure 물리 동기는 미검증으로 둔다.

수정 후 `...-moving-copy-fast-recording/` camera stopped·control 종료0, 137그룹/548시야·native 4,526개. 548/548 view host 범위에 앞뒤 copy를 찾았고 bracket gap p50/p95/max 5.163/7.688/8.183ms, source clock 교환 불확도 3.463ms, copy 시작 간격 2.509/2.725/3.321ms. packet 전환 1,396회는 모두 +1로 그 사이 빠진 ID 0개이나 이는 이 이력에서만 해당한다. 이동 그룹35, native AI true copy1,719, 전체 raw SHA/시각 감사 `.../independent-copy-check.json`. 이전 Event.wait 이력의 p50 15.373ms 및 중간 ID1,126개 누락과 구분해 원본 둘 다 보존한다. 15개 unit 및 Ruff 통과. 데이터는 진단 전용이며 actual physics 발생·노출 시각 및 actor/학습 자격을 뜻하지 않는다.

운영 Lua 원 SHA `475e085d…` 복원 후 CSP AI off/native frozen true가 보여 session restart/start/pause/resume 뒤 새 packet51779–52768에서 native false 59/59를 확인하고 pause했다. 당시 PID12432/generation `15533904ULL:0.0013857` epoch1, UI 입력/학생 actuator0. 이는 독점 제어 보증이 아니다. 다음은 이 sampler를 실제 역할 분리 수집의 시간계약에 통합하고 더 긴 코너/가감속 데이터에서 처리량·원본 시간 경계를 검증한 후 생물학적 회로 기여와 학습 자격을 확인한다.

## 시야별 선행 native copy와 명시 결측 (2026-10-31)

`scripts/audit_causal_copy_body_context.py`는 `...-moving-copy-fast-recording/`의 전체 native 420-byte 원본 hash, packet 순서·동일 ID 바이트 일치, camera metadata/hash 및 각 view의 host render 범위를 확인한다. 각 시야에서 render 하한보다 엄격히 먼저 끝난 최초 수신 copy 중 가장 최근 packet을 택하고, 중복 copy는 receipt를 갱신하지 않는다. sampler가 처음 본 packet은 이전 전환을 관측하지 못해 나이 불명이므로 결측(값0/mask false)으로 둔다. 기존 21채널 body encoder는 view host 범위 상한에서 평가한다. 출력은 diagnostic-only, camera/physics time 검증이나 실제 actor 교체가 아니다.

`...-moving-copy-causal-body-context/`는 548시야 중 이전 전환을 확인한 212시야에서 yaw와 속도 mask 모두 유효, 나머지 336은 초기 packet 나이 불명으로 결측이다. 움직임 후보 >1m/s인 35그룹 140시야는 140/140 전환 이후이며 전체 최초 유효 view는 group84다. 선택된 수신과 render 상한 간 host 시간 차 p50/p95/max 5.039/7.852/44.388ms는 source 발생 시각의 상한이 아니다. 원래 이력 전체를 삭제·초기화하거나 결측을 실측 0으로 취급하지 않았고 output NPZ/hash와 view별 증거를 보존했다. 중복 시각 재설정 거부/동일 ID 바이트 변경 거부·시각 경계와 기존 copy tests 17개·Ruff 통과. 고속·기울기·제동 구간, source 물리 시각·실제 exposure, 통합 회로 인과 기여·배포/학습은 아직 없다. 다음은 명시된 per-view 마스크를 방향 시간 시각 안정화 경계에 연결해 동일 원본의 유효 구간/결측 대응을 검증한다.

## 동일 저속 진단의 네 시야 망막 및 시야별 몸 입력 재구성 선언 (2026-10-31)

`...-moving-copy-fast-recording/` 원본 frame metadata와 `...-moving-copy-causal-body-context/`의 각 view별 선행 packet/결측 mask를 결합한다. 원본 카메라 layout2/FOV·투영 행렬·네 view ID/PNG 형식을 확인하고 등록 파일의 FlyVis 수용기 721좌표와 원래 `FlyvisCameraProjector`를 사용해 각 view의 픽셀을 투영한다. 원본 이미지 SHA는 캡처 당시가 아니라 이번 재생 시점에 계산하므로 원래 노출 또는 캡처 중 변조 부재 증명이 아니다. 결과를 별도 diagnostic NPZ로 보존하며 학습·actor·액추에이터0. 이는 FlyVis 네트워크의 운동 반응, 초파리 생리, physical time/exposure 동기나 지도 라인 입력의 증명이 아니다.

## 동일 진단 네 시야 FlyVis 고정 모델 재생 선언 (2026-10-31)

망막 투영·body mask가 보존된 동일 137그룹을 원저자 고정 FlyVis 네트워크에 오프라인 재생한다. 모델/registration/원본 metadata SHA를 확인하고, 네 시야는 독립 신경 상태 4개를 하나의 batch로 유지한다. 그룹의 네 원본 이미지 저장 완료 중 가장 늦은 CSP source clock을 그 그룹의 보수적 가용 시각으로 정해 그때까지 이전 그룹을 유지하고 5ms 이하 원래 시간 간격으로 적분한다. 최초에는 회색 steady state를 한 번 초기화하며 그룹 사이 상태를 reset하지 않는다. 네 영상의 실제 노출은 서로 다른 시각이고 이 공학적 그룹 가용 규칙은 실제 노출·물리 동기나 초파리 광학을 검증하지 않는다. T4/T5 8종×721개 공간 반응 및 중앙 반응을 별도 저장하고 원저자 가중치 불변을 확인한다. 기존 diagnostic data는 training 금지, optimizer·actor·차량 제어0. 화면 색상은 LDR 강도이며 실제 광자량과 다르다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-moving-copy-retina-body-projection/`은 137×4×721 망막 및 시야별 몸 값·mask를 보존한다. `.../20261031-moving-copy-flyvis-replay/`는 네 독립 상태, 2,235 내부 tick, 그룹 간 reset 0, 고정 가중치 불변을 기록했다. T4/T5 출력은 137×4×8×721이다. 원본 이미지는 캡처 후 재생 시 hash했으므로 캡처 당시 무결성을 증명하지 않는다.

## 시야별 몸 회전과 운동 시각 포트 진단 (2026-10-31)

`PerViewVisualBodyContextBoundary`는 네 시야 각각의 유효한 yaw와 운동 시각 시간차를 결합한다. 제로 초기화 및 결측 mask를 유지하는 공학 가설이며 측정된 시냅스가 아니다. `scripts/replay_per_view_visual_body_diagnostic.py`는 위 동일 원본과 기존 정규화·시간 경계 체크포인트를 확인하고 이전 영상 유지, 5ms 이하 적분으로 전 이력의 상태를 보존한다. 고정 가중치 0.01/256의 인위적 탐침과 0 가중치를 비교하며 학습이나 actor를 사용하지 않는다.

첫 실행 `.../20261031-per-view-visual-body-port-probe/`은 NPZ 키 이름 오류로 실패했고 원본과 실패 기록을 보존했다. 수정 실행 `.../20261031-per-view-visual-body-port-probe-v2/`는 137그룹·2,235 tick, reset 0, yaw 결측 84그룹 및 유효 53그룹을 기록했다. 결측 그룹과 제로 경계의 출력은 정확히 0, 고정 탐침의 최대 절댓값은 0.0001012이고 49그룹에서 0이 아니었다. 이것은 시각 포트 입력 변화의 존재만 보여 주며 실제 비행 생리나 회로 출력·차량 제어·자율주행 성능의 인과적 개선은 아직 검사하지 않았다. 물리 표본/실제 노출의 동기 미검증이므로 이 진단은 학습 금지다. 다음에는 감각 발생 시각 계약과 결측을 유지한 채 비행 회로 내부 및 출력 개입을 검증하고, 실제 코너·제동·하중 변화 이력으로 확대한다.

## 포트 탐침의 비행 회로 전파 실험 선언 (2026-10-31)

동일 137그룹의 운동 시각 시간 상태는 처음부터 유지한다. 필수 몸 채널의 수신 근거가 없는 최초 84그룹에는 회로 입력을 만들지 않고, 필수 몸이 모두 유효한 연속 구간 시작점에서만 별도 진단 회로 상태를 0으로 초기화한다. 각 그룹의 첫 시야에 선행한 body를 회로 입력의 과거 body 대용값으로 사용하며 실제 현재 결정 시점의 몸이라고 주장하지 않는다. 네 시야 개별 body mask로 계산된 기존 고정 탐침값을 시각 포트에 0·원래 부호·반대 부호로 주고, 동일 시각 경과 동안 이전 입력을 유지해 각 회로 상태·출력을 비교한다. 공통 시각 시간 경계와 입력, 시작 상태는 동일하다. 경로·정답 지도·절대 위치는 넣지 않으며, legacy 시각 여섯 채널과 바람은 결측, 목표는 명시적 고정 실험값이다. 이는 비행 회로의 공학적 개입 전파 검사이며 폐루프·생리·성능 및 물리/노출 시간 증거가 아니고 학습·actor·액추에이터0이다.

float32 첫 실험에서 출력 차가 작아 동일 저장 입력·고정 가중치를 float64로 승격해 독립 재계산한다. 이미 float32로 만들어 저장한 입력과 탐침의 잃어버린 정밀도는 복구하지 않으며, float64 결과도 생리·운전 유효성 검증이 아니다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-per-view-body-probe-graph/` 및 `.../20261031-per-view-body-probe-graph-float64/`는 그룹84–136의 53그룹(가용 시각 18.2974–22.4614초)을 비교했다. 앞 84그룹의 시각 시간 상태는 유지하지만 회로 상태는 84에서 별도 초기화하므로 전 에피소드 회로 재생이 아니다. float64에서 중립 대비 ±탐침의 회로 상태 최대 차는 각각 1.4724e-5, 조향 의도는 3.3362e-7, 전진 의도는 1.4672e-8, 근육 출력 최대는 8.6693e-10이다. 부호 반전 시 양쪽 조향 차의 합 최대는 1.31e-11로 작아 단순 float32 출력 반올림만은 아니지만 효과 크기 자체가 매우 작다. 고정 탐침을 인위적으로 선택한 것이며 학습된 기능, 운동 안정화, 실제 driving 또는 성능 개선을 증명하지 않는다. 두 실행은 result/NPZ 기록 후 Python 프로세스가 즉시 종료되지 않아 종료 코드는 아직 미확인이다. 원본 해시, 코드 스냅샷, 각 결과를 보존한다. 다음은 입력 sensor 발생 시간과 결측 계약을 개선하고 독립적인 생리 반응 및 더 긴 코너·가감속 이력에서 회로 기여를 검사한다.

## 네 시야 가용 시점의 선행 body 재구성 선언 (2026-10-31)

위 회로 진단의 앞 시야 render 시점 body 대용 한계를 줄이기 위해, 저장된 네 시야 PNG가 모두 준비된 CSP clock 시점의 host 시간 범위를 기존 ping 교집합으로 재구성한다. 범위 하한보다 먼저 완전히 수신한 가장 최근 원본 native packet을 택하고 최초 관측 packet은 나이 불명으로 결측 처리한다. 범위 상한에서 body encoder의 기존 나이 제한을 검사하며 원본 render 시점 body와 packet/yaw 차이를 비교한다. 이는 같은 이력의 host 수신 순서 및 단위 clock 속도 가정에 한정된다. group 이미지 저장 완료는 실제 actor 결정이 아니고, camera 노출·physics 생성 시점 동기 검증이나 학습 자격이 아니다. 원래 `clock-after.json`은 기록 당시 hash가 봉인되지 않아 교환값과 기존 hash된 render 범위로만 교차 검사하고 이번 읽기 시 hash를 기록한다. 학습·actor·제어 0.

`.../20261031-group-availability-body-audit/`에서 137그룹 중 관측된 전환 이후 53그룹에 대해 선행 packet을 복원했고 최초 84그룹은 결측이다. 유효 53그룹 중 52그룹의 packet이 앞 시야 render 시점과 다르며 yaw의 최대 절대 차는 0.07368rad/s다. 가용 시점의 host 수신 나이 p50/p95/max는 5.221/7.752/48.721ms지만 물리 생성 나이는 아니다. 따라서 같은 0/±탐침, 시간 이력, 초기 상태와 float64를 유지하고 회로의 공통 몸 입력만 앞 시야 packet에서 가용 시점 선행 packet으로 교체해 다시 재생한다. 시야별 yaw 탐침은 각 원래 시야의 선행 값을 계속 사용한다. 두 이력은 동일한 관측이므로 독립적인 주행 성능 평가가 아니다.

가용 시점 몸 재생 `.../20261031-availability-body-probe-graph-float64/`은 동일 53그룹에서 ±탐침의 조향 최대 차 3.3346e-7을 기록했다. 봉인된 출력을 비교한 `.../20261031-availability-vs-front-body-graph/`에서 공통 0 탐침 출력은 앞 시야 몸 기준 대비 52그룹이 달랐고 조향/전진 최대 차 0.001883/0.006610, 평균 절대 차 0.000625/0.000533이다. 몸 시점 선택 효과가 탐침 효과보다 이 이력에서 훨씬 크다. 어느 쪽이 실제 actor 시점의 정답인지, 물리 발생 시각이나 주행 성능은 이 실험으로 판정할 수 없다. 경계·기존 body tests 9개 통과. 가용 시점 재생은 결과 기록 후 PID9624/25100이 아직 살아 있어 종료 코드를 확인하지 못했다. 앞 재생 PID34048/33756, 12616/25808도 재확인 시 살아 있었다. 재시작하지 않으며 상태는 다음 관찰 때 다시 확인한다.

## 시야별 과거 몸과 결정 몸을 분리한 재귀 회로 경계 (2026-10-31)

`PerViewTemporalBodyContext`는 전후좌우 각 시야의 선행 수신 body와 하나의 결정 시점 body를 별도로 보관한다. 네 시야마다 render 하한보다 엄격히 이른 원본 수신, 카메라 수신·결정 순서, 동일 에피소드·계약과 21개 유효 mask를 검사하며 과거 body가 없는 시야는 명시적 결측이다. 시야별 과거 값은 운동 시각 포트에만, 결정 body는 기존 몸 회로에만 들어간다. `PerViewTemporalContextOpponentFlightStream`은 원래 결정을 최대 150ms 간격까지 받아 최대 5ms 내부 tick에서 이전 시각 입력과 이전 회로 입력을 유지하며 재귀 상태를 episode reset까지 보존한다. 영상, 몸 또는 바람이 물리적으로 동기화됐다는 주장이나 액추에이터 연결은 없다. 단위 11개는 미래/다른 에피소드 거부, 결측과 결정 몸 분리, 기존 drive 유지 및 100ms 원본 간격 내 이전 영상 유지 등을 확인한다. 실제 AC 폐루프는 아직 미실행이다.

실제 저장된 저속 진단 53그룹에 신규 stream을 연결한 `.../20261031-per-view-stream-diagnostic/`은 종료 코드0이다. 각 시야의 render 선행 packet을 그룹 가용 시점에 다시 encode해 freshness/mask를 검사하고, 현재 body는 그룹 가용 선행 packet으로 사용했다. 212시야 yaw mask는 모두 유효했고 중립 대 고정 가설 탐침의 출력 최대 차는 조향 5.07e-7, 전진 1.79e-7(float32)이다. 그룹84에서 시각 시간 상태와 비행 회로 상태를 모두 새로 시작했으므로 앞 84그룹까지 포함한 전체 이력 재생과 비교하면 안 된다. 그룹 가용 시점은 실제 actor 결정이 아니며 물리/노출 동기, 초파리 생리, 차량 성능 검증 및 학습은 아니다. 독립 진단 프로세스는 정상 종료했지만 앞선 세 graph replay 프로세스 쌍은 이후 확인 시에도 살아 있어 종료 코드는 계속 미확인이다. 다음은 실제 결정 시점 수집 경로에 이 계약을 연결하고, 원본 source 타이밍을 확인한 뒤 더 긴 코너·제동 이력의 회로 기여와 폐루프를 검증한다.

## 원본 native 수신 이력의 실시간형 선택기 (2026-10-31)

`PerViewBodyHistory`는 기존 `NativePhysicsSample`의 원본 바이트/수신 시각을 받아 제한된 이력을 유지한다. 첫 packet은 이전 전환을 관측하지 않아 결측이며 동일 packet 재수신은 첫 시각을 바꾸지 않는다. 각 시야에는 render 하한보다 엄격히 선행한 packet, 현재 body에는 결정 시각 이하의 packet을 고르고, 각 시점에서 별도의 body encoder로 유효 기간과 mask를 적용한다. 용량 초과로 지워진 과거는 결측이고 역행·reset·동일 ID 다른 바이트는 거부한다. lock 및 원자적 encoder 상태 갱신을 사용하며 이 클래스가 thread·센서·액추에이터를 직접 열지는 않는다.

`.../20261031-per-view-online-history-replay/`에서 기록된 native 복사 4,512건을 그룹 가용 하한까지 순차 투입한 뒤 가용 상한을 결정 시각의 오프라인 대용으로 삼았다. 137그룹 중 초기 84그룹은 current body 결측, 나머지 53그룹의 current packet과 212개 시야별 packet은 기존 별도 감사와 일치했다. 시야별 yaw는 카메라 수신 대용 시점에서 212/212 유효했다. 종료0, 관련 단위 11개 통과. 이는 저장 이력 재생이며 실제 실시간 actor, 실제 camera receipt/decision 및 물리 발생·노출 동기를 보증하지 않는다. 원본 이력은 여전히 학습 금지다. 다음은 기존 수집기의 실제 host 결정 시각과 원본 카메라 receipt를 연결해 shadow-only로 계측하고 원본 source 시간 계약과 지연을 검증한다.

## 실제 카메라 수신·몸 이력 shadow 계측 선언 (2026-10-31)

**측정 명칭 정정:** 아래 기존 `.../20261031-online-body-shadow-audit/`의 `ingestion_and_build_ms_p50_p95_max`는 이름과 달리 camera metadata read 완료에서 **body 선택용 시각을 찍을 때까지**만 측정한다. `history.build()`는 그 시각을 찍은 다음에 실행되므로 encoding/build 완료까지의 전체 지연이 아니다. 기존 결과 파일은 보존하고 이후 shadow 이벤트에 `context_ready`를 추가해 실제 완료 시각을 기록한다. 앞선 1.173/1.471/5.299ms를 전체 body 처리 지연으로 사용하지 않는다.

독립적인 online-body shadow 소비자를 기존 3초/15km/h 제한 진단 수집기에 선택적으로 연결한다. 동일 native sampler가 이미 기록한 원본 420-byte copy를 순차 소비하고, camera metadata 파일을 실제로 읽은 뒤 host receipt, 소비 완료 뒤 별도 host decision을 기록한다. 각 render 하한은 요청 전 ping에서 확정한 CSP clock offset 하한으로 계산한다. 이 소비자는 `PerViewBodyHistory`의 최초 packet 결측·중복 시각 유지와 시야별 과거/current 선택을 적용하며 actor나 액추에이터에 연결하지 않는다. 각 원본 파일·결과와 시점 및 source hash를 보존하고 기존 진단 camera/control 종료를 그대로 추적한다. 실제 물리 생성/노출 시각 및 clock 단위 속도는 검증되지 않아 학습 금지다. 단위9개 통과. 정지/full-brake/offline·CSP/native AI off·camera off·vJoy FREE를 확인한 뒤 기존 hash-pinned 배포기로 후보를 잠시 설치하고 신규 요청1회만 수행한다. 종료 후 camera Lua 원본을 복원하고 native AI 표식을 확인한다. 불명 시 원 핸들을 추적하고 같은 요청을 재발행하지 않는다. UI 입력·학생 제어·학습0.

사전 `.../20261031-online-body-shadow-admission/`은 별도의 unpaused 전용 보정 검사에서 paused 상태로 거부됐다. 당시 AI false·브레이크1·vJoy FREE·background는 기록됐고 아무 동작도 하지 않았다. paused 전용 검사를 갖춘 기존 배포기로 후보 `906a4be9…`를 설치하고 한 번 요청했다. `.../20261031-online-body-shadow-recording/`은 camera stopped, control 종료0, 138그룹/552시야, native copy 4,519회, shadow 138건·error null이다. 모든 metadata hash가 원본과 일치했고 552/552 host render bracket이 있다.

독립 감사 `.../20261031-online-body-shadow-audit/`는 각 선택 시각 전에 실제 소비한 native 범위만 재생하고 packet 최초 수신, 시야별 선행/현재 packet 및 모든 값과 mask를 비교했다. 초기 84그룹은 current 결측, 54그룹의 context 중 yaw 유효 212/216시야다. camera metadata read 완료부터 원본 native 소비 후 선택 시각까지 중앙/95/최대 1.173/1.471/5.299ms이며 이후 body build는 이 수치에 포함되지 않는다. 추정 save lower부터 실제 file read 완료까지 3.477/4.533/4.830ms다. 한 선택에서 최대 168 copy를 처리했다. 이는 동작한 shadow 소비자의 host 계측이며 영상 신경망 inference나 실시간 actor 성능이 아니다. `clock-before.json`은 원본 작성 당시 hash 봉인이 없어 감사 시점 hash로 구분했다. 9개 단위 통과, 원본 동기·초파리 생리·학습 자격과 자율 제어는 미검증이다.

복원 첫 시도 `.../20261031-online-body-shadow-restore/`는 일시적인 status 파일 부재로 교체 전 실패했고 candidate hash가 그대로임을 확인했다. 별도 재시도 `.../20261031-online-body-shadow-restore-retry/`는 운영 Lua `475e085d…` 복원 및 generation 교체를 확인했다. 직후 CSP AI off이나 frozen native AI true여서 기존 단일 CSP restart/start/pause/resume 절차 후 `.../20261031-online-body-shadow-native-reset-audit/`에서 새로운 packet 63075–64063의 AI false 59/59를 확인하고 pause했다. 현 PID12432/generation `15533904ULL:0.0013711` epoch1, paused/offline/CSP AI off/camera off/full-brake, UI/학생 actuator0. 독점 제어를 증명하지는 않는다. 세 이전 graph replay 프로세스 쌍은 같은 시점 재확인 시 계속 살아 있으며 종료 코드는 미확인이다.

## 실제 shadow의 기록 시각으로 FlyVis·비행 회로 오프라인 재생 (2026-10-31)

원래 shadow 카메라 metadata·native 원본을 다시 감사해 `.../20261031-online-shadow-causal-body/`와 `.../20261031-online-shadow-retina-projection/`에 138그룹의 네 시야 망막을 투영했다. 이미지 해시는 캡처 시점이 아닌 이번 투영 시점에 계산했다. `.../20261031-online-clock-flyvis-replay/`는 기존 shadow의 host body 선택 시각을 각 영상 가용 시각으로 사용해 원저자 고정 FlyVis의 독립 4상태를 초기화 한 번 후 2,249 tick 동안 유지했고 고정 가중치는 변하지 않았다. 원래 관측이 없던 프레임은 만들지 않았으며 neural live inference가 아니다.

`.../20261031-online-clock-flight-replay/`는 동일 시각의 FlyVis 출력과 당시 기록한 시야별/현재 body 값을 비행 회로 stream에 전달했다. 결정 body 필수 mask가 유효한 연속 그룹84–136의 53그룹에서 0 및 고정 가설 탐침을 비교하고 137의 stale body는 제외했다. FlyVis는 앞 84그룹의 시각 상태를 유지했으나 비행/시간 경계 상태는 그룹84에서 별도로 시작했다. 조향/전진 탐침 출력 최대 차는 3.73e-7/1.49e-7로 작다. 오프라인 **회로 두 조건**만의 호출 시간 중앙/95/최대는 66.78/77.62/606.79ms이며 이미지 투영·FlyVis·body build 및 프로세스 초기화는 제외된다. 두 회로 조건을 순차 측정했으므로 단일 actor의 지연이나 전체 실시간 성능으로 해석하지 않는다. 기존 카메라 선택 시각은 `history.build()` 전이므로 실제 online neural decision도 아니다. source physics/노출, 생리·주행 효능·학습 자격은 모두 미검증이며 actuator와 optimizer 0이다. 관련 단위9개 통과. 다음은 체감 bottleneck인 두 회로 처리의 내부 tick 비용을 분리하고 단일 회로·FlyVis·body build를 하나의 동시 경로로 계측하여 실제 지연과 frame backlog를 검증한다.

## 동일 5ms 단계 유지한 비행 회로 반복 계산 최적화 (2026-10-31)

`.../20261031-flight-replay-cprofile.pstats`의 53그룹·두 조건에서 `joint.advance` 1,716회 누적 2.326초, temporal `step` 1,718회 1.678초였다. 프로파일러 비용 및 GPU 동기화가 포함되므로 각 값은 배타적 GPU kernel 시간은 아니다. 동일 held 입력의 매 5ms 반복 중간 단계에서는 불필요하게 같은 입력의 유효성 검사와 읽지 않는 중간 readout/시각 projection을 반복한다. `PerViewTemporalContextOpponentFlightStream(..., optimize_held=True)`는 첫/마지막 단계에서 원래 검사를 수행하면서 내부 지수 누적 및 원래 회로의 각 5ms 갱신은 그대로 수행한다. 기본값은 기존 방식이다. exact 기존 joint/temporal 클래스와 명목 dt≥5ms 조건에서만 허용하며, 값이 맞지 않는 입력이나 실패를 무시하도록 사용하지 않는다.

`.../20261031-flight-held-optimization-comparison/`은 위 동일 이력 53그룹 및 중립/가설 탐침 두 조건에서 매 결정의 모든 회로 상태·held drive·세 시각 trace를 기준 방식과 비교했다. 모든 텐서의 최대 차 0(float32 bitwise equality), 단위 시간 간격 3/100/148ms를 포함한 별도 회귀와 관련 총9개 검사도 통과했다. 순차 두 조건의 오프라인 실행 시간 기준 p50/p95/max 67.53/78.48/567.00ms에서 최적화 25.36/30.16/30.84ms로 감소했다. 이는 같은 실행에서 기준 조건 다음 최적화 조건을 측정한 값이며 초기 컴파일·캐시 효과 및 측정 순서 영향이 가능하다. 이미지·FlyVis·body build를 포함하지 않고, 단일 모델 또는 live actor 처리량을 보증하지 않는다. 실제 body/영상 동기·생리·폐루프 운전 및 학습 자격도 검증되지 않았다. 다음은 warmup과 조건 순서를 통제한 단일 통합 경로 처리량·backlog와 원본 source 시간 계약 검증이다.

## 단일 경로 이미지·body·FlyVis·회로 오프라인 처리량 실험 선언 (2026-10-31)

실제 shadow 138그룹의 저장 PNG 4개를 원본 재생 시 hash 검증·디코딩하고, 원본 native copy 중 해당 shadow 소비 범위를 차례로 투입해 body history를 재구성한다. 이후 실제 망막 투영·고정 원저자 FlyVis의 네 독립 상태·scene pool·최적화된 중립 단일 비행 회로를 같은 순서로 실행하고 각 단계와 GPU 완료까지 경과를 잰다. 각 그룹의 원래 host 시각은 재생에만 쓰며 이미 저장된 입력이므로 실제 동시 수집은 아니다. FlyVis는 별도 폐기용 상태로 warmup하고 실제 시각 에피소드는 reset하지 않으며, 필수 body가 유효한 그룹84부터 회로 상태를 한 번 초기화한다. 보존된 원래 retina·FlyVis·비행 출력과 tolerance 이내 일치를 확인한다. 측정한 각 처리 시간과 원본 도착 간격으로 단일 대기열의 이상화된 지연을 계산하되, 그 대기열에서 stale해질 수 있는 원본 body/영상 재선택은 시뮬레이션하지 않으므로 live 정확성/제어 성능을 뜻하지 않는다. 원본 PNG hash는 캡처 후에 계산된 것, 물리/노출 시간과 actor·훈련 자격은 미검증이며 actuator·학습0.

첫 실행 `.../20261031-integrated-sensory-throughput/`과 `.../20261031-integrated-sensory-throughput-audit/`은 138그룹, 비행 유효53그룹을 처리했다. retina 최대 차0, FlyVis motion 최대 차5.22e-7, 비행 intent 최대 차5.82e-7로 앞선 원본 재생과 일치 허용 범위다. 유효 그룹 전체 p50/p95/max 48.61/53.87/428.39ms, 그중 첫 비행 그룹의 회로 단계가 396.68ms였다. 첫 그룹을 제외한 유효 그룹 전체는 48.60/53.24/137.42ms이고 원래 관측 간격 p50/p95/min 79.87/82.67/41.23ms다. 모든 138그룹의 각 단계 p50은 PNG 읽기/디코딩/hash 4.28ms, body 이력/encode 1.41ms, 망막 투영 3.39ms, FlyVis/scene 23.03ms이며 회로 p50은 body 결측 85그룹을 포함하므로 별도로 유효53그룹 회로 단계 p50/p95/max 15.70/18.86/396.68ms를 사용한다. 오프라인 측정 기반 이상화 queue 최장 응답 485.74ms(그룹85), 마지막 32.86ms이며 stale 재선택·실제 live 수집 부하는 미반영이다.

첫 비행 단계의 지연을 분리하기 위해 동일 입력과 모델에서 합성 2결정의 폐기용 flight stream을 시작 전에 실행한 후 실제 회로 stream을 새로 만들어 재생한다. 기존 실제 episode state·weights는 warmup에 사용하지 않는다. 첫 비교와 동일 이미지/hash·출력 검증·시간 정의를 유지하며 warmup 시간 자체는 제외해 별도 기록한다. 순서·부하 차이가 있으므로 이 한 쌍만으로 live 성능을 보장하지 않는다.

`.../20261031-integrated-sensory-throughput-flight-warm/`과 별도 `...-flight-warm-audit/`는 동일 138그룹·53회로 유효그룹, retina 차0, FlyVis 최대 차4.77e-7, intent 최대 차4.84e-7로 기존 재생 tolerance를 통과했다. 폐기용 준비 뒤 첫 비행 그룹 전체 39.72ms·회로 단계 7.19ms이며, 53 유효그룹 전체 p50/p95/max 47.73/50.69/50.99ms다. 모든 그룹의 stage p50은 PNG 4.27ms, body 이력/encode 1.50ms, 망막3.61ms, FlyVis/scene 22.96ms, 유효53그룹의 회로 p50/p95/max 15.17/16.51/18.07ms다. 원래 도착 간격을 이용한 이상화 queue의 최대 응답은 50.99ms였으나 실제 동시 수집·body stale 시 재선택·frame drop을 모델링하지 않았다. FlyVis/scene이 이 오프라인 한 번의 주요 stage였고, 원본 이미지 hash는 촬영 후 측정된 값이다. 초기 model load, FlyVis/flight warmup은 시간에서 제외했다. 같은 실험의 준비 전 첫 비행 회로 396.68ms와 비교할 근거가 생겼으나 순서·환경 차이 및 한 번의 반복이라 일반적인 live 결과는 아니다. source physics/노출, 생리, 폐루프와 학습은 미검증, actuator/optimizer0. 다음은 이 별도 준비 방식을 실제 shadow 시작 전에 통합하고 지연·backlog·결측 및 이상 종료를 실시간으로 직접 계측한다.

## FlyVis·비행 회로 live shadow 준비와 진단 선언 (2026-10-31)

`OnlineNeuralShadow`는 카메라 요청 전에 등록된 망막, FlyVis 원저자 가중치와 기존 비행/시간 경계 체크포인트를 검증하고 폐기용 visual/flight 상태를 준비한 뒤 실제 상태를 별도로 만든다. 이후 online body shadow의 원본 각 시야 PNG와 인과적 body context를 한 그룹씩 받아, 원래 선택 시각 간격 동안 이전 영상과 회로 drive를 유지해 비행 의도만 기록한다. 신경 출력은 actuator에 연결하지 않으며 상대풍은 명시 결측이다. 필수 current body 결측 후에는 회로를 자동 재시작하지 않고, 허용 간격을 넘거나 추론 오류 발생 시 shadow를 실패로 남긴다. 실제 화면·몸 표본의 물리/노출 시각 검증이나 학습 자격은 없다. 각 이미지의 해시는 신경망이 읽는 시점에 계산되며 실제 producer capture 시점 무결성 증명은 아니다.

기존 원본 138그룹에 대한 첫 검증 `.../20261031-neural-shadow-offline-verification/`은 config 경로 처리 오류로 실패했고 원본은 보존했다. 수정한 `...-verification-v2/`는 138그룹·53개 의도 출력, 기존 기준 대비 최대 오차5.37e-7, 종료0, GPU 초기화13.03초다. 관련 테스트11개 통과. live 수집기는 이 성공한 검증의 neural source/연관 코드·config·checkpoint hash를 일치시킨 경우에만 model을 준비하며 준비를 마친 뒤 현재 AC 상태를 다시 검증한다.

운영 Lua·CSP/native AI off·pause/full-brake·camera off·vJoy FREE를 새로 확인한 뒤 동일 hash-pinned 진단 camera를 잠시 설치하고 기존 최대3초/15km/h AI 진단을 신규 한 번 실행한다. 실제 파일 수신, body snapshot과 context 준비, neural 출력 완료 및 원본 시야 age와 body age, backlog/결측을 남긴다. 기존 model 학습·학생 actuator·UI 입력은 0이다. 실패·timeout 시 원래 핸들과 부분 로그를 보존하고 같은 제어 요청을 재발행하지 않으며, camera Lua를 원본으로 복원하고 native/CSP AI 상태를 확인한다.

`.../20261031-neural-live-shadow-recording/`은 camera stopped·control 종료0, 139그룹·556시야·native 4,504 copy, neural shadow 139건·오류0이다. 필수 몸 결측 86그룹 동안에도 독립 시야의 FlyVis 상태를 유지했고, 유효53그룹에서 비행 의도를 기록했다. `.../20261031-neural-live-body-audit/`는 실제 소비한 native copy의 최초 수신, body 선택·값·mask를 재검증했고, `.../20261031-neural-live-shadow-audit/`는 neural read 시점 이미지 hash 및 host 단계 순서를 검증했다. 전체 139그룹의 metadata receipt→context 준비 p50/p95/max 1.49/2.21/4.52ms, neural 처리 35.54/57.95/62.40ms, receipt→출력 36.97/60.08/64.17ms. 의도 출력이 있던 53그룹만의 receipt→출력은 57.29/61.27/64.17ms다. 두 그룹의 마지막 저장 시각 상한보다 앞 그룹 output이 늦은 경우는 0/138이지만 metadata가 실제 대기열에 들어온 시각을 증명하는 것은 아니다.

**현재 제어 차단 근거:** 비행 의도 53개 중 52개에서 output 시점의 current body 최초 host receipt 나이가 기존 50ms body 계약을 넘었다. 수신 나이 p50/p95/max 58.92/63.95/87.23ms이며 실제 물리 발생 나이는 더 알 수 없다. 선택 시점에는 중앙3.01ms였지만 이후 visual/flight 계산으로 오래됐다. 전방 포함 시야 render lower 기준 최대 age는 의도 출력에서 p50/p95/max 78.75/84.08/86.42ms, 150ms 초과 0이나 실제 exposure 시각은 아니다. 따라서 이 신경 shadow를 차량 제어에 쓰거나 최종 목표를 달성했다고 판정하지 않는다. 다음은 visual 계산 뒤 현재 body를 다시 원본 receipt에서 선택하고 이후 출력까지 50ms freshness를 검사하는 역할 분리 설계·비교를 우선한다. 시야별 과거 body는 원래 render 선행 이력을 유지하며, late body로 visual history를 재기록하지 않는다. 누락/오래된 body는 결측 또는 제어 중지이고 oracle/재시작은 없다.

**출력 인과 관계 정정/추가:** 현재 `PerViewTemporalContextOpponentFlightStream.decision()`은 이전 held drive로 경과 구간의 상태를 전진한 뒤 그 상태에서 intent를 읽고, 방금 선택한 body를 포함한 새 drive는 다음 구간에 보관한다. 따라서 직전 문단의 52/53은 새로 선택한 body의 출력 시각 수신 나이에 대한 관측이지, 방금 선택한 body가 동일 출력에 인과적으로 반영됐다는 증거가 아니다. 현재 출력에 영향을 주는 이전 held body의 receipt 나이를 별도 기록·검증해야 하며, 첫 출력에는 직전 held body가 없다. 이를 50ms 계약 검사에서 숨기거나 새 body로 대체하여 통과시키지 않는다.

## 시각 이후 결정 몸 재선택과 held-body 이중 freshness 검증 선언 (2026-10-31)

새 `LateBodyNeuralShadow`는 원본 camera receipt의 시각 snapshot에서 이전 영상을 유지해 FlyVis/scene을 완료하고 GPU 동기화한 뒤, 그때까지 sampler가 실제로 받은 새 native copy를 순차 소비해 current body를 결정 시각에서 선택한다. 시야별 과거 body는 여전히 각 render 하한 이전 packet을 원본 camera receipt에 평가한다. 새 body가 바뀌더라도 이미 진행된 영상 이력을 다시 쓰지 않는다. output에는 새로 선택한 body receipt와 **실제로 방금 완료한 구간에서 held했던 직전 body receipt**를 둘 다 기록한다. 전자는 다음 구간 drive에 반영되며 둘 중 하나라도 부재·50ms 초과거나 시야 render age가150ms 초과면 host freshness를 false로 남긴다. 이 가드는 진단일 뿐 제어 권한이나 source 물리 시각 증명이 아니다. 기존 고정 회로 상태를 즉시 재설정하거나 시간을 앞당기지 않고 actor/actuator/학습0을 유지한다.

`.../20261031-late-body-shadow-offline-verification-v3/`은 기존 138그룹에 대해 visual snapshot과 body selection을 원래 동일 시각으로 둔 수치 동등성 검사로, 53개 intent 최대 차 5.37e-7·종료0이다. 이 가상의 동일 시각 재생은 늦은 선택의 live 속도를 증명하지 않는다. 시각 후 새 packet을 넣어도 render 선행 view packet은 유지하는 검사 및 새/held body를 각각 검사하는 경계 등 관련 테스트13개 통과. 다음 신규 단일 live 진단은 동일 3초/15km/h 조건에서 앞선 shadow와 비교하되 먼저 pause/offline/full-brake·native/CSP AI off·camera off·vJoy FREE와 운영 Lua를 확인하고, 기존 배포기 및 정확한 검증 source hash로만 후보를 일시 설치한다. 두 종류의 output body receipt age, 전체 latency, freshness 및 원본을 보존한다. 종료 후 운영 Lua와 pause/AI 상태를 복원한다. UI 입력·학생 actuator·학습0, 감각 물리 발생/노출 시각은 계속 미검증이다.

`.../20261031-late-body-live-recording/`은 camera stopped·control 종료0, 129그룹·516시야, native copy 4,248건과 shadow129건·오류0을 남겼다. 별도 body 및 neural 감사 `...-live-body-audit/`, `...-live-neural-audit/`는 원본 packet·각 view 선행 순서·이미지 hash, GPU visual 완료→최신 body 선택→context→output 시간 순서와 직전 held body receipt를 확인했다. 초기 current body 결측76그룹, 의도53그룹이다. 의도 output에서 **새로 선택한 body**의 최초 host receipt age p50/p95/max 22.07/25.30/45.40ms, 50ms 초과0/53이다. 그러나 output 구간에 실제 적용된 **직전 held body**는 첫 output에 없고 나머지52개 모두 50ms를 초과했으며 p50/p95/max 102.20/110.08/111.50ms다. 두 나이와 시야 age를 함께 평가한 output host freshness 통과는 0/53이다. 이는 새 body가 다음 구간에만 전달되는 현재 상태 갱신 방식과 관측 frame 주기의 결과이며, 선택 지연만으로 해결되지 않는다. source 물리 발생/노출 시각도 미검증이므로 실제 actor·학습에 쓰지 않는다.

전체 129그룹 camera receipt→output p50/p95/max 36.20/56.50/60.03ms이고 시야 render lower→의도 output 상한 p50/p95/max 73.86/78.20/79.62ms다. 다음 그룹 image 저장 상한보다 앞 output이 늦은 경우 0/128이지만 metadata queue 부재의 증거는 아니다. 다음은 영상 준비와 독립된 **더 빠른 body receipt 기반 비행 상태 전진**을 설계해 원본 시간 내에 held body가 회로를 실제로 갱신하도록 하고, 기존 회로 연결·상태 연속성 및 출력 시점 freshness를 개입 실험으로 검증한다. 실제 actor·생리·폐루프 및 학습은 계속 미검증이다.

진단 뒤 운영 Lua `475e085d…` 복원, frozen native AI true를 확인해 기존 CSP restart/start/pause/resume 단일 요청 후 `.../20261031-late-body-live-native-reset-audit/`에서 새 packet85015–86003의 AI false 59/59를 관측하고 pause했다. 당시 PID12432/generation `15533904ULL:0.0013847` epoch1, offline·full-brake·camera/CSP AI off. 이는 독점 제어 증명이 아니며 UI 입력·학생 actuator·optimizer0이다.

## 원본 body receipt 기반 다중 속도 회로 재생 선언 (2026-10-31)

같은 live 진단의 129개 영상에 대해 실제 visual snapshot host clock으로 고정 FlyVis 네 시야 상태를 처음부터 재생한다. `MultirateFlightStream`은 해당 영상의 시야별 render 선행 body와 그 영상의 사용 가능 시각을 보관하고, 새 영상이 없는 동안 같은 scene을 유지한 채 새 원본 native packet의 최초 host receipt에서 기존 비행 회로 상태를 반복 전진한다. 원본 packet을 최소20ms 간격으로 선택하며 중간 데이터 보간·재타임·상태 reset은 없다. 실제 camera context가 준비된 시각에는 새 영상과 이미 수신된 가장 최근 body를 같이 적용한다. 영상/몸이 허용 host 나이를 넘으면 실패하며 현재 결정의 output은 직전 held body로 생겼음을 별도 기록한다.

비교는 같은 이력·기존 회로의 영상마다 body 갱신하는 조건과, 같은 빠른 body 이벤트에서 current yaw만 0으로 둔 명시적 오프라인 개입이다. 빠른 조건과 영상 조건의 차이는 몸 입력뿐 아니라 회로 내부 적분 분할도 다르므로 몸 감각의 단독 인과 효과로 부르지 않는다. yaw 0은 교사나 실제 센서가 아닌 회로 개입이다. 각 이벤트의 body 나이는 원본 모델 이벤트 시각에서만 계산하며 재생 처리 지연·실제 동시 수집의 backlog를 포함하지 않으므로 output 실시간 freshness 증명이 아니다. source 물리/노출 시각·생리·actor 자격 미검증, 액추에이터·학습0.

`.../20261031-late-visual-flyvis-replay/`는 원래 visual snapshot 129그룹의 네 시야와 2,101 tick을 reset 없이 고정 모델에 재생했다. `.../20261031-multirate-flight-original-replay/`는 필수 몸이 유효한 53개 frame 이벤트와 사이의 원본 body packet 158개를 사용했다. 최초 회로 상태는 구간 시작 시 한 번 초기화, 이후 211개 이벤트 사이 reset0이다. 빠른 조건의 직전 held body는 모델 이벤트 시각 기준 p50/p95/max 21.88/26.21/32.65ms, 50ms 초과0/210이다. 영상마다 갱신한 조건은 83.55/89.82/91.35ms, 50ms 초과52/52다. 실시간 output 나이가 아니다.

같은 빠른 시각 이력·이벤트에서 현재 yaw만0으로 둔 개입과 원본의 최대 회로 상태 차는 0.00766, 의도 차는 조향/전진0.04989/0.01388이다. 이는 해당 공학적 회로에서 yaw 입력이 영향을 주는 오프라인 개입 근거이며 초파리 생리·주행 향상의 증거는 아니다. 빠른 조건과 영상 조건 의도 최대 차0.00356/0.00315는 내부 적분 시간 분할도 다르므로 감각 단독 효과로 해석하지 않는다. 빠른 단일 stream 호출만의 오프라인 p50/p95/max는 9.48/10.94/604.08ms, 첫 준비 비용·시각/인코딩·다른 조건 및 실제 동시 스케줄링은 분리하지 못했다. 관련 단위8개 통과. 다음은 독립 body ticker와 FlyVis의 실제 GPU 경쟁·시각 교체 순서·output 시점 body age/backlog를 shadow에서 측정하고, stale/누락 시 fail-closed 처리한 뒤 폐루프 자격을 판단한다.

## 독립 body ticker·FlyVis live shadow 진단 선언 (2026-10-31)

`MultirateLiveShadow`는 원저자 FlyVis의 네 시야 producer와 기존 비행 회로의 단일 body ticker를 분리한다. FlyVis를 끝낸 원본 scene·render 선행 body context만 thread-safe mailbox에 올리고, ticker는 독립적으로 원본 native copy를 차례로 확인해 첫 수신이 보존된 최신 packet을 최소20ms 간격으로 고른다. ticker만 비행 재귀 상태를 변경하며 전 단계의 held scene/drive를 원래 시간만큼 전진하고, 새 scene은 실제 준비·공개 뒤에만 적용한다. 미소비 scene을 다른 scene으로 덮거나 native 중복 ID가 다른 바이트를 갖거나 시각/몸이 stale하면 shadow를 중지하고 오류를 남긴다. output의 새 body와 직전 held body가 모두 50ms, view render 하한 age가150ms 이내인지 별도 기록하며 첫 held 없음은 실패로 남긴다. 이 지표도 physics 발생/실제 노출 시각이나 자율 제어 자격이 아니다.

source hash가 일치하는 직전 138그룹 원본의 neural 동등성 증거와 별도의 211-event multirate 오프라인 증거를 모두 확인하고, mailbox 덮어쓰기 거부·body freshness·상태 보존 등 관련 단위12개를 확인했다. 이는 새 thread 간 GPU 경쟁의 live 검증은 아니다. 현재 paused/offline/full-brake·CSP/native AI off·camera off·vJoy FREE 및 운영 Lua를 확인한 뒤, 기존 hash-pinned 진단 후보를 일시 설치하고 최대3초/15km/h engine AI 진단을 신규 한 번 수행한다. 전체 camera·native 원본, vision event, body ticker event/error/stop reason을 보존하고 body 회로가 일찍 멈추면 재시작하지 않는다. 종료 후 Lua와 pause/AI 상태를 복원한다. UI/학생 actuator/학습0, 생성 시각·노출 동기/생리·폐루프는 미검증이다.

`.../20261031-multirate-live-shadow-recording/`은 control 종료0·camera stopped, 130그룹/520시야·native copy4,294건, visual 130건과 ticker157건을 남겼다. 53개의 body 유효 visual scene이 모두 순서대로 ticker에 소비됐으며 덮어쓰기나 ticker error는 없었다. 다만 진단 말미 current body receipt가 50ms를 넘어 ticker가 `current_body_receipt_stale`로 중단됐고 자동 재시작하지 않았다. 별도 `...-live-body-audit/`와 `...-live-neural-audit-v3/`는 원본 packet/최초 수신·image hash·scene publication 순서 및 실제 output 시각을 감사했다.

157개 출력 중 새 body와 직전 held body 양쪽 및 시야의 host freshness를 모두 만족한 것은57개뿐이다. 첫 출력은 직전 held body가 없어 실패했고, 나머지156개 중 held body 수신 age가50ms 초과99개(p50/p95/max 52.75/62.34/68.08ms)였다. 새 body age는 50ms 초과1개, 시야 render lower age 상한은150ms 초과1개다. body ticker 의사결정→output은 p50/p95/max 13.50/39.07/42.31ms, ticker 간격은21.89/39.61/42.87ms였다. 여전히 source 물리/노출 나이는 알 수 없다.

진단적 시간 중첩 분석에서 FlyVis host 실행 구간과 겹친 ticker 58개의 의사결정→output 중앙/95/최대는37.34/40.89/42.31ms, 겹치지 않은99개는12.80/15.31/16.79ms였다. 이는 겹침과 지연의 관찰상 연관이며 GPU contention을 단독 원인으로 증명하지 않는다. 현 구현의 두 thread 모두 전체 GPU 동기화를 사용하므로 다음은 분리된 CUDA stream 및 필요한 event 의존성만 사용한 버전을 수치 동등성과 오류 전파를 검증한 뒤 같은 shadow 조건에서 비교한다. 실제 제어·학습은 여전히 금지이며 output freshness 합격만으로 생리/폐루프를 입증하지 않는다.

## 시각·몸 CUDA stream 분리 진단 선언 (2026-10-31)

`SplitCudaMultirateLiveShadow`는 별도 시각/body CUDA stream을 사용하며 각 worker는 전체 GPU 대신 자기 stream의 완료만 기다린다. model 초기화·폐기용 warmup이 끝난 뒤 stream을 만들고, 시각 worker는 scene GPU 사본까지 자기 stream에서 완전히 끝낸 뒤에만 thread-safe mailbox에 게시하므로 ticker는 미완료 scene을 받지 않는다. 기존 native first receipt·20ms 이상 몸 tick·view 결측/나이 제한·미소비 scene 덮어쓰기 거부·원래 재귀 state 보존·stale 시 중단 및 actuator0은 유지한다. 이 구현이 실제 GPU에서 동시 실행될 때 결과나 지연이 같을 것이라고 가정하지 않는다.

`.../20261031-split-cuda-multirate-original-replay/`에서 기존 원본 53 frame·158 body event를 default CUDA와 별도 CUDA stream으로 각각 실행해 211개 이벤트의 전 회로 state·held drive·세 temporal trace·intent가 bitwise 일치했다. 이는 두 신경망을 동시에 경쟁시키는 live 시험은 아니다. stream별 동기화와 mailbox 결측/덮어쓰기 등 단위10개 통과. 현재 운영 Lua와 paused/offline/full-brake·CSP/native AI off·camera off·vJoy FREE를 재확인하고 기존 hash-pinned 후보를 잠시 설치한 뒤 최대3초/15km/h AI 진단을 새로 한 번 실시한다. 동일 50ms body/150ms view host output freshness, 누락·stop reason 및 GPU 작업 중첩 구간을 기록하고 독립 감사한다. 각 run 조건/난수·부하는 달라 한 번의 전후 비교로 일반 성능을 보장하지 않는다. 종료 후 Lua·pause/AI 상태 복원, UI/학생 actuator/학습0, source 물리/노출·생리/폐루프 미검증.

`.../20261031-split-cuda-live-recording/`은 camera stopped·control 종료0, visual146그룹·584시야, native copy4,774건·ticker170회, shadow/ticker error null이다. ticker는 마지막에 `current_body_receipt_stale`로 중단됐고 재시작하지 않았다. 별도 `...-live-body-audit/`, `...-live-neural-audit/`는 packet 최초 수신, 원본 이미지 hash, 53개 공개 scene의 53개 순서 소비 및 실제 output 시각을 재검사했다. 최초 held body가 없는1회를 포함해 두 body/시야의 host freshness는109/170회 통과했다. 직전 held body age는 p50/p95/max 45.76/61.34/65.37ms, 50ms 초과60/169이고 새 body 50ms 초과0/170, 시야 render lower 기준 150ms 초과0/170이다. 이전 별도 비분리 실험은 freshness57/157, held body 초과99/156이었다. 조건이 다른 각 1회 관측이므로 일반적인 개선·원인을 확정하지 않는다.

body tick decision→output p50/p95/max 13.49/37.81/41.15ms, ticker 간격21.71/38.35/41.69ms다. 시각 host 구간과 겹친68회의 출력 지연은29.51/39.67/41.15ms, 안 겹친102회는12.52/14.64/22.84ms였다. 시간 중첩은 GPU 커널 경쟁의 인과 증거가 아니며 여전히 60개 held body가 stale이므로 실제 제어/학습에 부적합하다. 다음은 영상 작업과 body tick의 GPU 실행·동기화 구간을 직접 계측하고, 원본 시간 유지와 stale 중단을 보존하면서 body 경로의 반복 동기화·직전 input 보유 시간을 줄이는 방법을 검증한다. physical source/exposure·생리·폐루프 및 레이싱 성능은 계속 미검증이다.

## 분리 stream CUDA 이벤트와 body 준비 시간 계측 선언 (2026-10-31)

기존 분리 CUDA live shadow에 선택적 계측을 추가한다. 각 시각 scene 처리와 body 회로 갱신의 해당 CUDA stream에 시작/끝 이벤트를 기록하고 완료 후 elapsed span을 저장하며, body encoder의 host 시작·종료도 남긴다. CUDA 이벤트 구간은 kernel 실행 중 공백·스케줄 대기·CPU 제출 지연을 포함할 수 있으므로 순수 GPU 점유율이나 단독 원인으로 해석하지 않는다. 계측 비용 자체가 실행에 영향을 줄 수 있고 실제 모델 load/warmup은 별도다. native receipt, scene publication/소비, 지연·stale 실패 기준 및 재귀 상태 계약은 바꾸지 않는다.

변경된 source에 대해 `.../20261031-cuda-timing-split-original-replay/`는 실제 이력211개 이벤트의 default/split state·drive·trace·intent bitwise 일치, 관련 단위11개 통과다. 실제 timing mode는 새 live 시험에서만 계측하며 이 수치 동등성 검증이 동시 GPU 실행의 결과 동일성을 보장하지 않는다. 운영 Lua와 pause/offline/full-brake·CSP/native AI off·camera off·vJoy FREE를 확인하고 기존 후보를 잠시 설치해 기존 3초/15km/h 진단을 신규 한 번 수행한다. camera·native·ticker·시각/event timing과 stop/error를 보존, 실패 시 재시작하지 않고 운영 Lua·pause/AI를 복원한다. UI·학생 actuator·학습0, source 물리/노출·생리·폐루프 미검증.

`.../20261031-cuda-timing-live-recording/`은 camera stopped·control 종료0, visual143그룹·native copy4,640건·ticker170건, shadow/ticker error null이다. body receipt가 stale해 ticker는 멈췄고 이후 공개된 마지막 scene1개는 소비되지 않았다(공개54·소비53). 이는 정상 소비로 계산하지 않는다. `...-live-body-audit/` 및 `...-live-neural-audit/`는 이미지 hash, 원본 native 최초 수신, 실제 scene 공개/소비 순서, output freshness와 추가 시간값을 검사했다. host freshness는110/170회 통과, 직전 held body 50ms 초과59/169, 첫 출력은 직전 body 없음이다.

전체 body CUDA event span p50/p95/max 13.04/37.64/43.07ms, vision stream span24.51/27.37/28.23ms, body encoder host0.160/0.202/0.241ms다. 시각 host 구간과 겹친 body event span은29.29/38.69/43.07ms, 겹치지 않은 경우12.17/15.12/15.95ms다. 각각 CUDA stream 양 끝 이벤트 간 시간으로 CPU 제출 공백·대기·동시 실행 영향이 포함되고 순수 GPU kernel occupancy가 아니다. 시간 중첩과 지연은 연관되지만 이 수치만으로 정확한 병목 원인을 단정하지 않는다. output decision→완료13.55/38.17/43.47ms, ticker 간격21.71/38.77/44.08ms이며, source physics/노출 시각과 actor/학습 자격은 미검증이다. 다음은 body 회로의 내부 GPU 동기화·kernel 제출을 세분화한 오프라인 프로파일과 수치 동등한 batched/captured 경로를 검증하고, stale·미소비 scene 중단을 유지한 실제 shadow에서 반복 비교한다.

## body 갱신 경로 시도와 10ms 최소 간격 비교 선언 (2026-10-31)

원본 오프라인 multirate 프로파일 `.../20261031-multirate-body-cprofile.pstats`는 475개 세 조건 update 중 joint drive, original rate advance, temporal step이 모두 비용을 차지했다. 순수 kernel 점유율이 아닌 CPU 측 cumulative 비용이다. 원래 희소 CSR 곱까지 graph capture한 `.../20261031-graphed-body-rate-probe/`는 현재 HIP runtime에서 capture invalidated로 실패했다. 원본은 보존하고 희소 곱을 eager로 둔 pointwise capture `.../20261031-graphed-body-rate-pointwise-probe/`는 원본 상태 211건에서 차0이나 isolated rate의 p50이 기준1.35ms보다1.83ms로 늘었고 전체 재귀 stream·동시 GPU는 미검증이다. 따라서 어느 capture도 live에 적용하지 않는다.

대신 기존 코드·체크포인트·시간 및 검증 규칙은 유지하고 최소 body tick 간격만20→10ms로 지정할 수 있게 했다. `.../20261031-multirate-10ms-original-replay/`은 실제 원본 이력53 frame과334 body event에서 원래 timestamp를 유지해 모델 시각상의 held age p50/p95/max10.22/15.89/32.65ms를 관측했다. `.../20261031-multirate-10ms-split-original-replay/`는 동일387개 전체 state/drive/trace/intent가 default/split CUDA에서 bitwise 일치한다. 이는 미래의 live 처리 속도나 실제 output 나이 검증이 아니다. 관련 단위13개 통과.

20ms와 마찬가지로 stale body/시야, 미소비 scene, 오류 시 재시작하지 않는 동일 조건을 유지해, 실제 native receipt 기반 최소10ms ticker와 FlyVis를 별도 stream에서 동시에 실행하는 제한 3초/15km/h shadow를 신규 한 번 요청한다. 운영 Lua·paused/offline/full-brake·CSP/native AI off·camera off·vJoy FREE를 먼저 확인하고 기존 hash-pinned camera를 일시 설치한다. output freshness, 실제 결정/출력 나이, ticker 처리량·GPU 중첩과 종료 이유를 독립 감사하며 부하가 커져 실패하면 원본을 보존하고 재발행하지 않는다. 종료 후 운영 Lua·pause/AI 상태 복원, UI/학생 actuator/학습0. 단일 각 run 비교의 조건 차이·물리/노출/생리/폐루프 미검증을 유지한다.

`.../20261031-multirate-10ms-live-recording/`은 camera stopped·control 종료0, 136그룹·544시야·native copy4,469건·ticker257회, shadow/ticker error null이다. ticker는 이후 current body receipt stale로 멈추고 재시작하지 않았다. 별도 `...-live-body-audit/`, `...-live-neural-audit/`에서 native 최초 수신과 원본 이미지 hash·scene 순서·output clock을 확인했으며 공개 scene54개 중53개를 소비하고 마지막1개는 미소비로 남았다. output host freshness 통과245/257, 첫1개는 이전 held body 없음, 이전 held body 50ms 초과10/256, view render lower age 상한150ms 초과1/257, 새 body 50ms 초과0/257이다. 직전 held body age p50/p95/max29.95/49.40/52.64ms, ticker 간격13.00/33.07/35.70ms, decision→output12.72/32.63/35.09ms다. FlyVis host 구간과 겹친108회 output 지연 중앙18.51ms, 안 겹친149회11.81ms로, 시간 중첩은 kernel 경쟁 원인의 증거는 아니다.

별도 20ms 최소 간격 1회(109/170)보다 이번 10ms run(245/257)의 통과 비율이 높았지만 별도 주행·부하의 단일 관측이라 일반 개선으로 확정할 수 없다. 12회는 여전히 freshness를 통과하지 못했고 물리 sample/실제 exposure, 생리·폐루프와 학습 자격은 없다. 다음은 같은 조건의 독립 반복과 더 긴 코너·제동 상황에서 failure/no-output 비율·scene stale·실제 센서 출처를 검증하고 회로 계산을 더 효율화한다. 기준을 완화하거나 stale 출력으로 차량을 제어하지 않는다.

운영 Lua `475e085d…` 복원 후 native frozen AI true라 기존 CSP restart/start/pause/resume을 각각 한 번 수행했다. `.../20261031-multirate-10ms-live-native-reset-audit/`에서 새 packet133170–134157의 AI false59/59를 확인하고 pause했다. 당시 PID12432/generation `15533904ULL:0.0013628` epoch1, offline/full-brake·camera/CSP AI off. UI/학생 actuator/optimizer0, 독점 제어는 미검증이다.

운영 Lua `475e085d…` 복원 후 frozen native AI true라 기존 CSP restart/start/pause/resume 각각 단일 요청을 수행했다. `.../20261031-cuda-timing-live-native-reset-audit/`에서 새 packet117916–118904의 AI false59/59 확인 후 pause했다. 당시 PID12432/generation `15533904ULL:0.0013685` epoch1, offline/full-brake·camera/CSP AI off. UI/학생 actuator/optimizer0이며 독점 제어는 미검증이다.

운영 Lua `475e085d…` 복원 후 frozen native AI true라 기존 CSP restart/start/pause/resume 각각 단일 요청을 수행했다. `.../20261031-split-cuda-live-native-reset-audit/`에서 새 packet106530–107518의 AI false59/59를 확인하고 pause했다. 당시 PID12432/generation `15533904ULL:0.0013763` epoch1, offline·full-brake·camera/CSP AI off. 독점 제어는 미검증이며 UI/학생 actuator/optimizer0이다.

운영 Lua `475e085d…` 복원 후 frozen native AI true를 확인해 CSP restart/start/pause/resume을 각각 단일 요청했다. `.../20261031-multirate-live-native-reset-audit/`에서 새 packet95783–96771의 AI false 59/59를 확인하고 pause했다. 당시 PID12432/generation `15533904ULL:0.0013627` epoch1, offline·full-brake·camera/CSP AI off. 독점 제어는 미검증, UI/학생 actuator/optimizer0.

운영 카메라 Lua `475e085d…` 복원 후 CSP AI off/native frozen AI true가 관측돼 기존 CSP restart/start/pause/resume을 각각 단일 요청으로 수행했다. `.../20261031-neural-live-native-reset-audit/`에서 새 packet73774–74761의 AI false 59/59 관측 후 pause했다. 확인 당시 PID12432/generation `15533904ULL:0.0013945` epoch1, paused/offline/full-brake·camera off·CSP AI off, UI/학생 actuator/optimizer0. 이는 실제 독점 제어 증명이 아니다. 원래 clock-before 파일의 최초 봉인 hash 및 source 물리/노출 시간은 없다.

### 사전 등록 — 10ms 독립 반복 (2026-10-31)
기존 저속 Monza 3초 expert-motion-probe, 10ms body ticker 및 분리 CUDA stream을 동일하게 독립 반복한다. 원본 수신 시각, 출력 직전 보유 몸 감각과 영상의 host freshness, 중단 사유와 미소비 scene을 독립 감사한다. 물리 센서 생성·카메라 노출 시각이나 폐루프 주행 성능의 검증으로 해석하지 않는다. stale 중단 시 자동 재시작하지 않는다.

**반복 결과:** 134 영상 그룹, ticker 264회 중 host freshness 246회 통과. 직전 보유 몸 감각 50ms 초과 16/263, 영상 150ms 초과 2/264, 최초 직전 감각 없음. scene 54개 공개·53개 소비, current body stale로 안전 중단. 출력까지 중앙값은 시각 처리 host 구간과 겹칠 때 22.22ms, 아닐 때 11.06ms이며 인과관계는 미확정이다. 첫 시험 245/257과 함께 여전히 기한 위반이 반복된다. 물리 시각·생리·실제 제어는 미검증이며 학습 업데이트는 없다.

원본: `C:\Users\jeff4\AppData\Local\flybrain\runtime-runs\20261031-multirate-10ms-repeat-recording`; 독립 감사: `C:\Users\jeff4\AppData\Local\flybrain\runtime-runs\20261031-multirate-10ms-repeat-body-audit`, `C:\Users\jeff4\AppData\Local\flybrain\runtime-runs\20261031-multirate-10ms-repeat-neural-audit`. 운영 Lua SHA256 `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119` 복원. 새 native packet 59개 AI false, pause 확인(PID 12432, generation `15533904ULL:0.0013382`, epoch 1).

후속: 30초 제동 명령이 CSP에 있으나 현재 수집기는 3초 명령을 고정하고 있어 그대로는 제동 검증이 아니다. 별도 명시적 모드와 시간 제한·증거·안전 종료 조건을 검토하고, 원본 물리/노출 시각과 코너·하중 변화 검증을 갖춘 뒤 실행한다.

### 사전 등록 — 30초 제동 원본 수집 (2026-10-31)
정지·오프라인·AI 해제·카메라 유휴 상태에서 기존 CSP expert-brake-pulse30(.4초 펄스, 30km/h profile)의 단일 소유 요청을 실행한다. 진단 카메라 최대 600그룹·55초, native 40000표본·child 47초 제한, 10ms 분리 stream shadow를 기록한다. 펄스 확인은 CSP terminal 및 독립 원본으로 판단하고 미발생·시간 초과·stale은 실패 증거로 보존하며 자동 재시도하지 않는다. 이는 teacher 진단이며 actor 운전·코너 성공·학습 또는 물리 노출 시각 검증이 아니다.

**제동 수집 결과 — 실패 보존:** `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-brake30-shadow-recording/`의 control terminal은 `api_returned_false`, `owned camera ended or became stale`. 펄스 API는 1회 반환 true, terminal의 peak brake 0.99969, 펄스 전 30.943km/h, 이후 최소 0.044km/h가 기록됐지만 전체 probe 검증은 false이다. 실제 camera frame 384→385 script clock 간격 297.21ms (57.35930→57.65651), control 종료 clock 57.65659로 시간상 일치한다. 원인 인과는 미확정. camera 408그룹 보존, shadow 104그룹·ticker 111회에서 stale scene 오류와 미소비 scene 오류, native 13134회 보존. 이 run은 30초 완수·actor 제동·학습 증거가 아니다.

운영 Lua SHA256 `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119` 복원 후 신규 native packet 59개 AI false, pause (PID 12432, generation `15533904ULL:0.0013473`, epoch 1). 변경된 후보 SHA256 `08d1887f6abe484f0acbd8634fdc9071db8207247d7ca97dbe80c8af29541c1b`는 제동 진단에만 600그룹·55초를 허용하며 기존 한도와 안전 게이트를 유지한다. 수집 profile 및 Lua 관련 7개 테스트 통과. 다음에는 프레임 간 297ms 지연과 stale scene 오류의 원인을 원본 시각·큐·연산 관점에서 추적하고, 게이트를 완화하지 않은 상태에서 제동·코너를 재검증한다.

### 실패 원본 독립 감사 — 2026-10-31

408개 그룹의 메타데이터와 봉인된 shadow/ticker 해시, CSP terminal을 감사했다. 그룹 102 전방의 render 종료→save 종료 72.20ms, 전체 1,632개 시야 중앙 3.70ms, 두 번째 최대 5.49ms다. 마지막 ticker는 그룹 101을 19677.54057에 판단해 19677.55045에 출력했으며 최초 시야 출력시 나이는 151.91ms; 그룹 102는 19677.61794에 공개됐다. 이어진 ticker stale scene 오류와 미소비 오류의 시간순 맥락이다. 저장 구간 지연의 구체적 원인은 계측되지 않았다.

더 나중 그룹 384→385 시작 간격 297.21ms, disable_ai clock 57.65659는 그룹 385 시작 57.65651 직후다. 이 공백은 frame 시작 사이이므로 그룹 102 저장 지연과 동일 원인이라고 단정할 수 없다. 408개 메타데이터 해시는 감사 시점에 새로 계산했으며 원 수집 시 봉인된 해시가 아니다. 부분 기록의 결과와 제한을 보존한다. 독립 감사: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-brake30-failed-camera-audit-v2/result.json`; 감사 코드 `scripts/audit_failed_camera_capture.py`. 관련 테스트 8개 통과.

다음은 파일 저장/발행과 렌더 이전 공백을 별도 계측하고 기존 MMF 경로와 단일 소유 shadow 통합을 검토한다. source 물리·노출 시간, 신경 생리와 폐루프 제어 및 실전 랩타임은 미검증이다.

### MMF 불변 네 시야 스냅샷 — 2026-10-31

`scripts/verify_immutable_camera_group.py`로 보존된 실제 `20261031-epoch1-30kmh-train-candidate-01`의 해시 검증된 MMF camera.zip과 원본 결정 중 동일 간격 8개 그룹을 읽었다. 네 시야 32장의 파일·RGB 해시, 원래 receipt·render 하한·capture/generation을 확인하고 원본 RGB에서 기존 투영과 불변 byte-backed 사본 투영의 최대 차이 0을 확인했다. 결측·위조 픽셀·stale·다른 capture는 거부, 원본 배열의 후속 변경은 사본에 전파되지 않는다. 관련 10개 테스트 통과. 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-immutable-mmf-group-original-replay/result.json`.

이것은 기존 기록의 8개 선택 재생이며 FlyVis 신경망, 실시간 MMF 전달/처리 지연, 물리 센서 시각, 생리, 폐루프 제어 또는 학습 검증이 아니다. 다음은 이 불변 스냅샷을 단일 소유 MMF consumer와 원래 FlyVis shadow에 연결하고 전체 이벤트 순서·생략·backpressure·출력 freshness를 원본 재생에서 검증한 뒤 실시간 시험한다.

### MMF와 shadow 카메라 계약 검사 — 2026-10-31

이전 exterior 기록의 8그룹 투영 동등은 그 기록 자신의 카메라 설정으로 동일 픽셀을 두 번 투영한 결과다. 기존 shadow 설정과 호환된다는 증거가 아니다. 실제 exterior 설정에는 각 시야의 `origin_body_m`과 `exterior_front_v3` 장착 계약이 있으나 기존 shadow 설정에는 이 항목이 없다. 위치가 실제로 다르다고 단정할 수는 없지만 동등하다는 근거도 없다. 전체 view/order/mount 계약 비교를 `require_matching_camera_geometry`에 추가했고 이 기록은 거부된다.

기존 shadow와 설정 계약이 동일한 보존 MMF 기록 `20261029-t1521-live-mmf-v3-preflight`를 찾아 원본 decision 전체 74개·296시야를 해시 검증해 불변 사본과 기존 망막 투영 차이 0을 확인했다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-immutable-mmf-all-geometry-replay/result.json`. 이 중 capture ID 인접 차이는 1이 69회, 2가 3회, 0이 1회다. 원래 source capture 전체를 복원한 것이 아니며 빠짐을 채우지 않았다. 위치 계약 동등은 설정 비교일 뿐 실제 카메라 기하·노출 시각 자격이 아니다. FlyVis state/출력, 실시간 shadow, 학습, 폐루프는 아직 미검증. 기존 테스트 11개 통과. 다음은 원본 중복·건너뜀을 명시한 전체 camera stream의 인과 처리와 신경망 연속 상태 검증이다.

### 불변 MMF 영상의 FlyVis 연속 상태 비교 — 2026-10-31

카메라 설정 계약이 같은 `20261029-t1521-live-mmf-v3-preflight`의 실제 보존 MMF 원본 74결정을 한 episode로 순서대로 재생했다. `OnlineNeuralShadow`의 고정 author 가중치·망막 투영·fractional 그래프 backend와 scene_pool을 사용해 파일 RGB 경로와 불변 RGB 경로의 상태를 각각 연속 갱신했다. 기존 입력을 실제 결정 간격으로 hold하며 branch당 657 내부 tick, 재설정 0. 중복 capture 1회와 중간 미관측 3개는 그대로 보고했으며 없는 이미지를 만들지 않았다. 망막 최대 차이 0, 전체 FlyVis 활동 state 최대 차이 9.5367e-7, scene 출력 최대 차이 8.9407e-8 (허용 2e-6). 동일 GPU에서 두 branch를 순차 실행해 bitwise 동일을 주장하지 않는다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-immutable-flyvis-continuous-replay-v2/result.json`, 코드 `scripts/verify_immutable_flyvis_state.py`. 관련 11개 테스트 및 해당 스크립트 Ruff 통과. 이 offline 비교는 결정 시각의 기록된 subset만 사용하며 전체 카메라 stream, 실제 동시 수집·지연·물리 노출 시각, 몸/비행 회로 출력·생리 및 폐루프 주행을 검증하지 않고 학습 업데이트가 없다. 다음은 MMF 단일 consumer의 누락·중복 기록과 shadow producer의 실제 연결 및 body ticker 통합을 별도 자격검사한다.

### 불변 MMF 입력의 실제 shadow producer 연결 — 2026-10-31

`MultirateLiveShadow.configure_immutable`는 초기화에 사용된 config 및 카메라 hash, view/order/mount 계약을 고정한다. `process_immutable`은 원본 receipt·네 view 하한·RGB hash를 검증한 불변 사본을 기존 FlyVis producer에 연결한다. 같은 capture가 바뀌지 않은 경우 중복으로 기록하고 새 시각 event로 재처리하지 않으며, 같은 ID의 이미지/시각 변경, 역순 및 generation 전환은 실패한다. 건너뛴 capture 개수는 명시하고 없는 영상은 생성하지 않는다. 새 경로는 아직 실제 MMF consumer에 연결되지 않았다.

원본 `20261029-t1521-live-mmf-v3-preflight`의 decision 74개를 원래 결정 시각을 대입한 오프라인 재생에서 실제 새 producer 함수로 실행: 고유73·중복1·미관측 중간3·656 내부 tick, 기존 파일 RGB 참조 대비 FlyVis 전체 state 최대 오차 9.5367e-7. body context 없이 scene 미발행, 실제 지연·camera/MMF 실시간 소비·몸 회로·제어·학습 검증이 아니다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-immutable-shadow-bridge-original-replay-v2/result.json`.

변경된 source의 기존 split CUDA 숫자 검사는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-multirate-immutable-source-split-proof/result.json`에서 원본 387개 전부 state/drive/trace/intent 오차 0이며 동시 시각 실행은 미검증이다. 이전 split proof는 새 source hash에 맞지 않는다. 관련 테스트 14개와 새 검증 코드 Ruff 통과. 다음은 단일 MMF 소비자에서 capture 전체와 body histories를 전달하고 중단·backpressure·output freshness를 원본/실시간에서 검증한다. 물리 발생·노출 시각과 생리·폐루프 주행·랩타임은 계속 미검증.

### MMF 단일 소유자와 몸 감각 shadow 연결부 — 2026-10-31

`src/flybrain_drive/immutable_body_shadow.py`는 이미 시작된 기존 MMF camera owner, 동일 native rows와 episode를 가진 기존 body ticker, 원본 `PerViewBodyHistory`를 묶는 비제어 소비자다. 선택 시점의 원본 후보와 pixels를 기존 immutable producer에 전달하고, 영상 처리 후 native 원본을 drain해 각 시야와 현재 시점의 몸 맥락을 선택한다. camera와 ticker는 이 연결부가 새로 열거나 제어하지 않는다. owner thread, 공통 perf_counter, 정확히 같은 native 배열·episode를 요구한다. ticker 실패·stale·camera 만료 등은 해당 이벤트와 오류를 남기고 자동 재개하지 않는다.

단위 검사에서 원래 bytes/hash/receipt로 만든 서로 다른 native packet을 실제 per-view history에 전달해 decision packet 2와 최초 packet 유효성 결측을 확인했고, 타 native 배열 거부, ticker 오류 후 재개 금지와 mock camera 결합도 확인했다. 관련 13개 테스트와 Ruff 통과. 실제 FlyVis/몸 ticker가 함께 도는 MMF live 연결·frame backlog·freshness·물리/노출 시각·생리 및 폐루프는 아직 검증되지 않았다. 다음은 source/hash 고정과 실제 record/종료 프로토콜을 갖춘 제한 shadow runner를 구축하고 정지·이동 실험으로 독립 감사한다. 학습 및 actuator 0이다.

### 사전 등록 — 단일 MMF 시각·몸 shadow 실시간 진단 (2026-10-31)
기존 운영 Lua와 일치하는 카메라 계약, 단일 MMF owner·같은 native 원본 sampler·10ms 분리 CUDA body ticker를 사용한다. CSP expert-motion-probe 단일 요청은 3초·15km/h 제한, 자동차 제어는 엔진 AI만 담당하고 shadow actuator·학습은 0이다. 원본 camera/센서/시각·ticker/오류·종료 영수증을 보존한다. 영상 결측·stale·consumer 실패 및 부정확한 물리/노출 시각을 숨기지 않으며 실패와 알 수 없는 요청을 재시작하지 않는다.

**실시간 결과 및 독립 감사 — 2026-10-31:** `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-mmf-immutable-shadow-live-first/`에서 원 운영 Lua, 단일 MMF camera 276그룹·native 4362 copy·engine AI child 종료0, archive 1944파일 hash 확인. bridge 1873호출: 고유 capture 269, duplicate 1604, 몸 맥락 없음 164, 시각 scene 공개105, 관찰된 capture ID 중간 누락0. ticker 134회 중 output host freshness 7회만 통과, 직전 held body 50ms 초과 126/133, 4 view 상한 150ms 초과0. 몸 held 나이 중앙/95/최대 63.52/78.73/85.63ms, decision→output 31.03/39.11/41.65ms. current body receipt stale에서 ticker 종료, 최종 `incomplete_diagnostic`.

독립 감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-mmf-immutable-shadow-live-first-audit-v3/`는 공개 시각 105개 4시야의 보존 MMF RGB와 실제 신경 입력 해시, 원 native 최초 receipt·held·output clock을 대조했다. 단일 시험이며 지연의 원인과 source 물리/노출 시각은 미확정이다. 현 수치는 폐루프·학습 자격 실패이며 actor/학생 제어와 optimizer update는 없다. 실행기·감사 코드 Ruff, 관련 테스트 13개 통과.

종료 후 paused CSP AI false였으나 native 공유 메모리는 직전 정지 packet AI true로 남아, 동일 운영 Lua에서 restart/start/pause/resume 후 새 native packet 59개 AI false 확인하고 pause했다. 현재 PID 12432, generation `15533904ULL:0.0013473`, epoch2, paused·camera off·brake1. 독점 제어/물리 시각은 별도 미검증. 다음은 과도한 중복 polling·시각 producer와 body stream의 경쟁 및 body 회로 latency를 계측해 개선하되 freshness 기준을 완화하지 않는다. 전체 겹눈/비행회로의 생리적 기여, 가감속·변속·셋업 적응 및 실제 유효 완주/라인/랩타임 최적화는 미완료다.

### 사전 등록 — 중복 RGB 복사 제거 후 MMF shadow 반복 (2026-10-31)
직전과 같은 운영 Lua, MMF 40ms, body ticker 10ms, 동일 checkpoints 및 3초 15km/h engine AI를 사용한다. 동일 capture의 metadata가 정확히 같으면 RGB 재복사·해시·신경 재처리를 하지 않고 중복을 기록한다. 다른 metadata의 동일 capture, source 오류와 stale은 중단한다. 기존 조건·host freshness와 원본 보존을 유지하며 사용자 actor·학습은 0이다. 두 독립 단일 시험의 차이는 부하/출발 시점 차이를 포함하므로 단독 원인 증거가 아니다.

**중복 최적화 반복 결과 — 2026-10-31:** `ImmutableBodyShadow`는 동일 capture 후보의 generation·receipt·4시야 RGB 해시·render 하한이 같을 때 `samples_for`와 4장 복사/재해시를 건너뛴다. 메타데이터 변화는 중단한다. 실제 `20261031-mmf-immutable-shadow-skip-repeat/`에서 카메라 285그룹·native copy 4527·engine AI child 종료0, scene 106개 공개, duplicate 1760회를 복사 없이 기록했다. ticker 138회 중 host freshness 24회, held body >50ms 113/137, view >150ms 0; held p50/p95/max 61.88/82.71/86.16ms, decision→output 31.82/39.72/42.13ms. stale body로 중단해 전체 결과는 incomplete다. 원본 및 2007파일 hash 보존.

독립 감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-mmf-immutable-shadow-skip-repeat-audit/result.json`에서 실제 106 scene × 4 RGB, native 최초 receipt 및 출력 시각을 재확인했다. 앞선 7/134와 다른 단일 시험 하나를 비교한 것으로 최적화 효과의 인과나 통계적 유의성은 입증하지 못한다. 결정 latency 중앙은 전 31.03ms→후 31.82ms이고 여전히 부적합하다. 기존 조건·freshness 기준을 바꾸지 않았다. 테스트 14개와 해당 코드 Ruff 통과.

종료 후 CSP off였으나 정지 native packet AI true가 남아 restart/start/pause/resume 후 새 packet 59개 AI false 확인 및 pause(PID12432, generation `15533904ULL:0.0013473`, epoch3). 물리/노출 시간·생리·폐루프·학습은 미검증. 다음에는 네 시야 처리와 ticker 사이 GPU event/host 구간 및 native decode/queue 경합을 분리 계측해 원인을 찾고, 조향·가감속/변속·하중/타이어 적응과 실제 완주/라인/랩타임 목표를 유지한다.

### 사전 등록 — MMF shadow GPU 구간 계측 (2026-10-31)
이전 동일 조건 3초 15km/h 엔진 AI·MMF 40ms·body ticker 10ms와 중복 복사 생략을 유지하고 기존 split CUDA 계측만 명시적으로 켠다. 시각 producer host/GPU event, body encode host/GPU event, 출력 host freshness 및 원본을 함께 보존한다. CUDA event 구간에는 큐 대기와 host 제출 간격이 포함될 수 있으며 GPU kernel 점유율이나 인과관계로 부르지 않는다. 계측 오버헤드는 별도로 인정하고 학습·actor 제어·자동 재시도는 0이다.

**GPU 구간 계측 결과 — 2026-10-31:** 운영 Lua·3초 15km/h engine AI 단일 요청 종료0, MMF 278그룹·native 4398 copy. ticker 135회 중 host freshness 8회, held body >50ms 126/134, 시야 상한 >150ms 0. body 인코더 host 중앙/95/최대 0.132/0.168/0.206ms, body CUDA event 구간 32.35/39.44/43.17ms, 시각 CUDA event 구간 13.99/15.23/16.24ms, 시각 host 구간 20.50/23.99/26.24ms. 시각 host 구간과 겹치는 body 133회의 decision→output 중앙 32.89ms, 겹치지 않는 2회 중앙 13.95ms. 작은 비겹침 표본과 미통제 부하·계측 overhead 때문에 인과관계/커널 점유율은 미확정이며 GPU event에는 대기와 host 제출 공백이 포함된다. 실제 source/노출 시간과 학습·actor 운전은 미검증이다.

원본/시간 독립 감사: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-mmf-shadow-cuda-event-audit/`, 구간 감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-mmf-shadow-cuda-event-timing-audit-v2/`. 단일 실험, 유효 완주·라인·랩타임 시험이 아니다. 운행 후 native 정지 packet AI true가 남아 운영 API restart/start/pause/resume, 새로운 packet 59개 AI false 확인 후 pause(PID12432/generation `15533904ULL:0.0013473`, epoch4).

다음은 GPU event 구간 안의 제출/동기 경계와 CPU/GPU 경합을 더 분리하고, 생물학적 방향·시간 시각과 비행 유래 회로의 인과성, 가감속/변속/하중/셋업 적응과 실제 완주/라인/랩타임을 검증한다. 관측된 시간 기준을 완화하거나 임의로 생략한 영상으로 해석하지 않는다.

### 몸 회로 CPU/GPU 원본 재생 비교 — 2026-10-31

기존 10ms body-rate 원본 사건 387개(영상53·몸334)를 동일 그래프·학습 checkpoint·입력과 recurrent state 유지 조건으로 GPU CSR 및 CPU CSR에서 나란히 재생했다. `scripts/replay_multirate_flight.py --compare-cpu-body` 결과 GPU 대비 CPU 전체 state/drive/traces/intent 최대 절대 오차 5.1782e-7 (허용 2e-6), 숫자 동등은 해당 이력 범위에서 통과했다. 회로 단독 host 시간 p50/p95/max는 CPU 32.36/48.32/152.69ms, GPU 11.65/17.40/548.60ms이다. GPU 최댓값은 초기 구간을 포함한다. CPU 시간은 GPU scene→CPU 복사·실시간 영상 동시 부하를 제외한 수치이며 live scheduler 자격이 아니다. 원본 파일 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-multirate-cpu-body-original-replay/`, 관련 테스트 11개 통과. 9728노드·1311145엣지 그래프를 보존했으며 Dense 근사나 GPU→CPU 교체는 배포하지 않았다.

본 실험은 학습·actor 제어·생리 및 물리/노출 시각·실제 코너/제동 완주를 검증하지 않는다. CPU에서도 기준을 만족할 근거가 부족하므로 CPU 이동을 해결책으로 채택하지 않는다. 다음은 원래 희소 회로의 host 동기/검증 구간을 각 단계에서 계측하고, 수치 및 결측/보호 경계를 유지하는 구간 최적화를 원본·실시간으로 확인한다.

### 사전 등록 — 네 시야 몸 입력 묶음 전송의 live 검증 (2026-10-31)
기존 3초 15km/h engine AI·40ms MMF·10ms split CUDA body ticker·duplicate RGB 생략을 유지한다. 각 시야 몸 값/mask를 CPU에서 원래대로 조립한 뒤 2개 tensor로 전송한다. 현 소스·checkpoints 및 기존 387사건 출력 일치와 immutable bridge proof를 고정한다. 실시간 host freshness·원본 RGB/native와 오류를 감사하며 actor 제어·학습·기준 완화·자동 재시도는 없다.

### 시야별 몸 감각 전송 묶음 — 2026-10-31

원본 387 사건 replay의 cProfile (`20261031-body-offline-profile-summary-v2`)은 세 비교 branch를 합친 827 decision에서 drive_observed 누적2.61초, advance 누적2.54초를 보였다. host inclusive 측정으로 내부 구간이 겹치고 비동기 GPU 시간/실시간 경합 원인을 나타내지 않는다. 각 4시야의 몸 값/mask를 원래 구조대로 CPU에서 조립해 GPU에 각각 1회 전송하는 `packed_view_body.py`를 적용했다. 원본 53 context에서 기존과 tensor/mask가 정확히 일치했고 동기화된 격리 microbenchmark 159쌍 두 번째 실행은 기존 중앙1.179ms, 묶음0.245ms였다. 첫 별도 실행은 0.473→0.115ms로 부하 차이가 커 실제 live 개선량으로 일반화하지 않는다.

수정 후 원본 387사건의 events 및 frame-comparison SHA가 변경 전과 같고 split CUDA 387/387 state·drive·trace·intent 차이0 (`20261031-packed-view-body-full-original-replay`). 별도 138그룹 late-body replay 53 intent 최대오차 5.36e-7 및 immutable bridge replay도 통과. 수정 소스와 묶음 helper에 대한 proof 의존성 확인을 두 live runner에 추가했다. 테스트 12개 및 대상 코드 Ruff 통과.

**실시간 결과:** `20261031-packed-view-mmf-shadow-live/` 단일 engine AI 요청 종료0·camera278그룹·native4533 copy, ticker 4출력은 모두 host freshness 불통과였다. 원본 시각 재감사에서 producer의 unconsumed scene 오류가 먼저였고, 이후 ticker가 stale view로 중단한 것으로 정정했다. `...-live-audit/`에서 실제 공개 scene2개, held body >50ms 3/3, view >150ms 1/4, native 최초 receipt와 실제 RGB를 대조했다. 마지막 두 scene ID171/172, 172의 마지막 출력 시야 상한152.2ms; 원인 단정 불가. 기존 단일 시험들과 직접 인과 비교 불가하며 경로는 미완성, actor/학습 0이다. 운영 Lua는 유지했고 이후 새 packet AI false 59/59 확인 후 pause(PID12432, generation `15533904ULL:0.0013473`, epoch5).

다음은 시각 producer와 body ticker의 stale/미소비 전환을 원본 순서·명시적 결측·회로 state 보존 조건에서 처리하고, 세부 GPU/host 경합을 개선해 재검증한다. 실제 물리/노출 시각·생리 및 가감속/변속/셋업 적응과 유효 완주/라인/랩타임은 여전히 미검증이다.

### 사전 등록 — 제한된 시각 mailbox 대기 실험 (2026-10-31)
직전 기록 원본 시간에서 frame173의 공개 시도 실패는 21746.2413에 발생했고, pending frame172의 소비 시작은 21746.2547로 이후다. 앞선 stale→미소비 순서 설명을 정정한다. 동일 MMF 40ms·body10ms·3초 15km/h AI 조건에서 새 영상은 이전 장면이 소비될 때까지 최대50ms 대기하고, 대기 후 영상 상한 150ms 초과 또는 ticker 중단이면 오류로 중단한다. overwrite·가짜 영상·임의 상태 초기화·actor 제어·학습·자동 재시도는 없다. frame별 mailbox wait와 원본 시각/센서/출력 및 종료 영수증을 보존한다.

**제한 대기 live 결과 — 2026-10-31:** `20261031-bounded-visual-wait-live/`에서 엔진 AI 3초 child 종료0·MMF 266그룹·native 4308 copy, scene105개 공개·ticker105회. 이전 시야를 덮어쓰지 않는 bounded wait는 105공개 중 103회가 1ms 초과, 중앙/95/최대 14.28/19.53/21.07ms였다. 하지만 독립 원본 감사 `20261031-bounded-visual-wait-live-audit/`는 host freshness 0/105, held body >50ms 104/104, 시야 >150ms 23/105, held 중앙81.30ms·decision→output 중앙39.43ms를 확인했다. ticker는 current body receipt stale로 중단, producer는 기다리던 ticker 종료를 명시했다. 0/105는 이 단일 시험이며 대기가 원인이라고 단정하지 않되, 현재 경로는 목표 운전에 부적합하다. 중단 전 출력도 진단일 뿐 actuator·학습에 사용하지 않았다.

첫 실패의 순서는 원본 확인상 producer frame173 미소비 오류가 21746.2413에 먼저, frame172 소비 시작 21746.2547 이후 ticker stale 순서였다. 네 시야 몸 전송의 float64 입력 precision 누락 가능성도 수정했고 회귀 검사로 확인했다. 변경 source의 387사건 split state/drive/trace/intent 차이0, immutable 원본 74결정 replay 통과, 관련 테스트 16개 통과. 시뮬레이터 종료 후 native AI false 새 packet 59개를 확인하고 pause(PID12432/generation `15533904ULL:0.0013473`, epoch6).

다음은 대기 방식의 지속 사용을 성능 성공으로 간주하지 않고 GPU 시각·body 회로 간 우선순위/자원 경합을 통제한 비교와 더 빠른 원본 보존 계산 경로를 검토한다. 신경 생리, 실제 물리/노출 시간, 조향·가감속·변속·셋업 적응과 유효 완주/라인/랩타임은 미검증이다.

### 사전 등록 — 몸 회로 GPU stream 우선순위 비교 (2026-10-31)
HIP 조회 priority 범위 least 1, greatest -1, runtime은 요청 -1을 보고한다. 앞선 MMF40ms·body10ms·3초 15km/h engine AI·CUDA event 계측·bounded mailbox 대기 조건을 유지하고 body stream만 priority -1로 변경한다. 387 원본 사건의 split state/drive/trace/intent 일치 검증과 exact source 증거를 요구한다. 스케줄 우선순위는 kernel 선점이나 실제 순서 보장을 의미하지 않으며 event 및 output freshness를 독립 감사한다. 비행 회로·시각 구조·원본 센서·기준을 변경하지 않고 actor 제어·학습·자동 재시도는 없다.

**GPU 우선순위 결과 — 2026-10-31:** HIP `hipDeviceGetStreamPriorityRange` 성공, least=1/greatest=-1; PyTorch stream은 요청한 body priority -1을 보고했다. 원본 387 사건의 high-priority split과 기본 stream의 state/drive/trace/intent 최대 차이0, 동시 시각 작업은 이 오프라인 검증에 포함되지 않는다. immutable bridge 원본 재생과 관련 테스트16개도 통과했다.

실시간 `20261031-body-priority-mmf-live/`는 engine AI child 종료0·MMF273그룹·native4386 copy·scene103개·ticker103회다. 독립 원본 감사 `...-live-audit/`에서 host freshness 0/103, held >50ms 102/102, view >150ms 40/103, held 중앙82.21ms, decision→output 중앙39.93ms. GPU event 감사 `...-gpu-audit/`에서 body event 중앙39.19ms, visual event 중앙16.40ms이며 queue·host 제출 공백 및 계측 overhead 포함이다. 우선순위 이전 bounded-wait 단일 run 0/105와 더불어 성공 근거가 없지만 단일 run 비교로 인과를 확정하지 않는다. 실험은 shadow stale 중단, training/actor 제어0이며 우선순위 변경은 선택형 실험으로 남긴다.

이후 운영 API로 재시작·새 native packet59개 AI false 확인 후 pause(PID12432, generation `15533904ULL:0.0013473`, epoch7). 물리/노출 시각·생리·자율주행 성능 미검증. 다음은 9,728노드/1,311,145연결의 원 희소 계산과 검증의 host 동기 비용을 분리하고, 신경 상태와 감각 결측을 보존하는 더 빠른 계산·스케줄 경로를 추적한다. 가감속·변속·셋업 적응과 유효 완주·레이싱 라인/랩타임은 전체 목표로 유지한다.

### 사전 등록 — 원본 body-rate 회로의 기울기·하중 개입 (2026-10-31)
기존 검증된 저속 원본 53시각 그룹·334 몸 사건의 재생 시간/상태/카메라/시야별 과거 몸 감각을 유지한다. 현재 결정 몸의 pitch·roll(0,1) 또는 네 바퀴 하중(8–11)만 각각 0으로 설정하고 원본 mask는 유지해 intact와 전 회로 state/조작 intent를 비교한다. 각 개입의 원본 유효값 크기를 기록하고, 해당 저속 단일 이력의 공학적 반사실험으로만 해석한다. 실제 생리, 차량 주행 개선·독립 성능·학습으로 주장하지 않는다.

**기울기·하중 개입 결과 — 2026-10-31:** 동일 저속 원본에서 387 body-rate 사건(53시각·334몸)을 각각 재생했다. pitch/roll 현재값만0으로 둔 경우 유효 원본 값774개·최대 0.02043rad, intact 대비 조향/전진 intent 평균 절대차 0.0000959/0.002036, 최대 0.000234/0.003631, 전체 신경 state 최대차 0.002417. 네 바퀴 하중값만0으로 둔 경우 유효 원본 값1548개·최대 하중 비율0.40196, 평균 절대차 0.0000704/0.003663, 최대 0.000175/0.004497, state 최대차0.006406. validity mask·카메라·시야별 이전 몸 정보·원래 시각/상태를 그대로 두고 해당 현재 몸 입력만 개입했다. 0으로 둔 하중은 실제 주행에서 비현실적일 수 있다.

독립 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-body-counterfactual-original-audit/`가 두 실험의 같은 원본/model/387개 packet·시각·intact output 완전 일치 및 조건별 숫자를 확인했다. 기존 yaw 개입과 합쳐 이 공학적 경로에서 몸 감각이 출력에 영향을 준다는 증거이나, 생리학적 반사 재현·접지 인식·학습 기여·성능 개선·실제 주행의 증거는 아니다. 구간은 단일 저속 원본이고 source 물리/노출 시간 미검증, optimizer/actor0. 관련5개 테스트 및 감사 코드 Ruff 통과. 다음은 실제 코너·제동 원본의 몸·시야 동기 및 독립 이력에서 반응/폐루프 기여를 검증하고, MMF body freshness 실패를 해결한다.

### 사전 등록 — 실제 pace 제동 기록의 네 시야·몸 재생 (2026-10-31)
기존 20261031-epoch1-brake-region-pace-diagnostic 731결정 전체를 원본 순서와 저장 host 수신 시각으로 재생한다. 차량 외부 카메라의 별도 위치 계약을 유지해 원래 yaw 카메라 actor와 같다고 가정하지 않는다. 기존 외부 60→100→0→60 과제는 진단 전용·fit 금지이고 시각/물리 발생 시각, 생리 및 폐루프 제동 성능은 미검증이다. 원본 및 실패를 보존하고 actuator·optimizer·자동 재시도는 0이다.

### 사전 등록 — pace 제동 기록의 영상·몸 공동 회로 개입 (2026-10-31)
이전 외부 카메라 pace 원본 731결정 전체의 FlyVis4시야·현 시점 body21·가장 이른 시야 전 마지막 수신 body를 사용한다. 이 단일 prior를 4시야에 공통 적용하는 보수적 맥락이지 각 시야의 별도 센서 표본이 아니다. 고정 실험 목표 15km/h는 원래 60→100→0→60 과제가 아니며, 네 시야 외부 mount는 기존 yaw-camera 모델과 다르므로 domain shift로 표시한다. intact·현재 tilt0·현재 wheel load0을 원래 mask와 시간 및 재귀 상태로 비교하고 62개 이동 제동 표본을 사전 감사 결과로 식별한다. 하중0은 비물리적 분포 밖일 수 있다. 전 과정 진단·fit/actuator0, 노출/물리 시각·생리·폐루프 성능을 주장하지 않는다.

**Pace 제동의 실제 시각·몸 회로 개입 — 2026-10-31:** 보존된 외부 60→100→0→60 과제의 731개 전 결정에서 legacy/navigation body replay를 완성하고, 네 시야 저자 FlyVis를 연속 6570 tick·handoff reset0으로 재생했다(`20261031-pace-diagnostic-flyvis-replay/`, fit/actuator0). 원본 decision/current body와 네 시야 모두보다 먼저 수신된 한 과거 body를 per-view temporal 경계에 넣어 동일 시각/상태로 intact·현재 tilt0·현재 load0을 731개 전체 재생했다(`20261031-pace-visual-body-intervention/`). 사전 검증된 이동 제동62개에서 intact 대비 조향/전진 intent 평균 절대차는 tilt0 0.00005096/0.010595, load0 0.00002254/0.003366이다. 이 출력은 조향/전진 intent 두 채널이며 브레이크 명령이 아니다. 전체 731개 mean은 각각 0.0001165/0.008166, 0.00008684/0.002442다.

`20261031-pace-visual-body-intervention-audit/`에서 731 native raw hash/packet/decision과 speed>1km/h·brake>0.05인 62행 및 저장 결과 통계를 독립 대조했다. 이 62개는 연속 표본이며, 네 view에 하나의 보수적 공통 prior를 사용했다. body yaw·tilt·load 등 원본 유효 mask는 유지하고 roll/pitch angular rate와 가속도 결측은 0인 유효값으로 위장하지 않았다. 기존 모델 카메라와 exterior mount가 다르며 고정 모델 목표 15km/h는 원래 속도 과제가 아니다. 하중0 개입은 비물리적일 수 있다. 따라서 이는 고정 입력 공학 회로에서의 감도 증거에 그치며 센서 발생/노출 동기, 비행 생리, 실제 브레이크·독립 주행/라인/랩타임 개선은 증명하지 않는다. 원본 변경·fit/optimizer·차량 제어0, 관련8개 테스트와 실행/감사 코드 Ruff 통과. 다음은 실제 동기와 각 시야 독립 과거 몸 감각을 검증하고, 제동/가속·변속의 실제 출력 경로와 실시간 freshness를 회복해야 한다.

### 사전 등록 — pace 4시야별 선행 body receipt (2026-10-31)
이전 단일 공통 prior를 개선하기 위해 동일 731결정·원본 native 수신의 최초 packet 이력에서 front/left/right/rear 각 render-start 호스트 하한보다 엄격히 앞선 마지막 수신을 독립 선택한다. 원래 camera 수신까지의 보수적 나이 50ms를 초과하면 해당 시야를 명시 결측 처리한다. 동일 packet을 재날짜하지 않고 원본 21값·mask와 네 시야의 다른 packet 비율을 보존한다. 이 분석은 host 수신 순서이며 실제 발생/노출 동기·actor 입력 자격·학습/성능 증거가 아니다.

### 사전 등록 — 공통/시야별 prior의 고정 탐침 비교 (2026-10-31)
같은 pace 731결정·동일 camera/현재 몸/temporal/회로 checkpoint·고정 15km/h 가상 과제를 사용해, 시각 전 몸 맥락만 공통 prior 대 각 시야별 원본 prior로 비교한다. 해당 시야·yaw 상호작용 projection은 현재 학습값이 아닌 0.01/256 고정 공학 탐침으로 명시한다. 원본 mask·시각과 recurrent state를 유지, 생리 재현·성능·학습·actor 배포 근거로 해석하지 않는다. 제동 62행은 원래 native 감사로만 구분하며 실제 노출/물리 발생 시각과 외부 mount domain shift는 미검증이다.

**Pace 네 시야별 native prior — 2026-10-31:** 원본731결정의 2924시야에서 각각 해당 render 시작 host 하한보다 엄격히 앞서고 camera receipt까지 50ms 이내인 마지막 native 최초 수신을 선택했다. front/left/right/rear 각각 731/731 matched·yaw valid731, 기존 earliest-view 공통 prior 대비 packet 차이 각각 0/9/12/15개, 서로 다른 prior가 포함된 장면15개였다. 같은 packet인 2888시야에서는 기존 값/mask가 정확히 같았다. `20261031-pace-per-view-body-original-audit/`에 각 원본 packet·receipt·값/mask와 증거 hash 보존. 실제 센서 발생/노출 시간은 미검증이다.

공통 prior 대 시야별 prior를 731결정의 같은 FlyVis·현재 몸·회로·고정15km/h 조건에서 고정 yaw 탐침 weight 0.01/256으로 전 회로에 전달했다. `...-pace-common-yaw-probe/`, `...-pace-per-view-yaw-probe/` 결과를 `...-pace-per-view-yaw-probe-audit/`가 lineage·731시각/packet/제동62행을 재확인했다. 조향/전진 intent 최대 절대차 3.35e-8/0, 평균 2.50e-10/0, 수치 허용 2e-6보다 작고 제동62행 최대차0이므로 관측 가능한 시야별 맥락의 회로 기여를 입증하지 못했다. 6개의 비트 차는 부동소수점 잡음과 구분되지 않는다. 탐침은 임의 고정 공학 가중치이며 측정·학습된 생물학 회로가 아니다. 외부 mount domain shift·15km/h 고정 과제·공기/물리/노출 시각 한계 유지. fit/optimizer·차량 제어0, 관련8개 테스트 및 새 audit 코드 Ruff 통과.

다음은 동적 시야별 차이가 충분한 독립 코너/회전 이력과 방향별 motion neuron 생리 반응, 검증된 감각 발생 시각을 확보하고, 학습/인과 기여가 있는 경로인지 비교한다. 동기·생리·브레이크/변속 제어·셋업 적응 및 실제 완주/레이싱 라인/랩타임은 여전히 미검증이다.

### 사전 등록 — 기존 transit 코너의 네 시야 몸 맥락 탐색 (2026-10-31)
보존된 원본 중 20261029-t1526-spatial-corner-mmf-corpus/episode-01-transit는 60km/h 제한의 transit·fit 금지 기록이며 원래 yaw-camera 계약과 일치한다. 기존 여러 기록의 yaw 크기를 살펴본 뒤 큰 회전 구간으로 선택한 탐색 이력이므로 독립 성능/holdout으로 부르지 않는다. 원본 전체739결정의 raw 영상·센서/aid, 4시야별 host 선행 packet과 실제 FlyVis 연속 상태를 검증한다. 물리 센서 발생·노출 시각, 생리 및 실전 제어는 증명하지 않으며 actor/optimizer0, 원본 역할을 변경하지 않는다.

### 사전 등록 — 코너 transit 공통/시야별 prior 고정 탐침 비교 (2026-10-31)
원본739결정의 카메라 yaw 계약·raw 감사·네 FlyVis state·기존 body/current와 각 시야의 인과적 host prior를 연결한다. 공통 earliest-view prior 대 개별 prior 외 모든 입력·시각·모델·상태를 동일하게 두고 0.01/256의 비학습 고정 yaw 탐침이 출력에 미치는 차이를 비교한다. 원래 60km/h 과제가 아닌 15km/h 고정 목표이며 2e-6 수치 기준을 명시한다. 회전량을 본 뒤 선택한 transit은 탐색 자료·fit 금지이며 독립 성능, 생리, 물리/노출 동기 및 폐루프를 주장하지 않는다.

**코너 transit 시야별 prior와 회로 비교 — 2026-10-31:** 탐색적으로 회전 큰 `20261029-t1526-spatial-corner-mmf-corpus/episode-01-transit` 선택. 원래 transit/fit 금지를 유지하고 `20261031-yaw-corner-transit-raw-audit/`에서 전체739결정·7003파일·aid off와 기존 actor packing 차이0을 재확인했다. `...-causal-audit/`, `...-common-context/`, `...-per-view-body-v2/`에서 4시야 각739 선행 receipt/yaw 유효, 공통 prior와 packet이 다른 시야57개·장면28개. 같은 packet의 2899시야 값/mask는 원본 공통과 정확히 같다. 달라진 57시야의 yaw 절대차 평균0.00424·최대0.04389rad/s. host 수신 전후이며 물리/노출 시각 동기 증거가 아니다.

`...-transit-flyvis/`는 전체739결정·독립 4시야·6642 internal tick·reset0로 replay. `20261031-yaw-corner-per-view-probe/`에서 사전 선언한 고정 비학습 yaw probe 0.01/256과 실제 회로를 사용해 공통 대 시야별 prior를 동일 현재 몸·시각·state·고정 목표15km/h로 비교했다. 원본 yaw 절대 p50/p95/max 0.00554/0.69245/0.95844rad/s, 회로 조향/전진 intent 최대 차2.6822e-7/0, 2e-6 초과0/739. 별도 `...-probe-audit/`에서 원본과 output/mask/패킷 비교 확인. 이 조건에서 회로 출력으로 분명한 per-view 기여는 입증되지 않는다. 해당 probe는 측정·학습된 생리 회로가 아니며 이 이력은 회전량을 본 뒤 택한 탐색 transit이지 독립 성능 평가가 아니다. 현재 목표15km/h는 원래 엔진 AI 과제와 다르다.

관련12개 테스트·새 replay/audit Ruff 통과. fit/optimizer·actor0. 다음은 모델 내 시야별 yaw 경계가 실제로 중립 초기화 상태임을 고려해 합성 방향 운동의 부호·반응을 생리 근거와 분리 검증하고, 별도 학습된 경계가 인과적으로 기여하는지 독립 데이터를 구성한다. 실제 센서 발생/노출·풍향·가감속/변속·셋업·폐루프 유효 완주/라인/랩타임은 계속 미검증이며 실시간 freshness 병목도 해결되지 않았다.

### 사전 등록 — 네 시야 운동 신호 개입 (2026-10-31)
같은 탐색 transit 739결정과 고정 15km/h 과제를 사용한다. 네 FlyVis 시야의 pooled 특징 중 방향별 8개 운동 채널을 학습 정규화 평균으로 각각 대체하는 네 이력과 네 시야 모두 대체하는 한 이력, 원본 두 반복을 독립 recurrent state로 처음부터 비교한다. 밝기 특징·시야 유효 mask·시야별 선행 몸 감각은 유지하고 미학습 yaw projection은 0으로 둔다. 차이는 전체 이력에 대한 회로 출력 영향이며 실제 카메라 결손, 생리적 방향 선택성이나 폐루프 성능을 의미하지 않는다. 동일 원본 반복의 수치 차와 사전 기준 2e-6을 함께 보고하고 학습·차량 제어는 수행하지 않는다.


### 2026-10-31 네 시야 운동 신호 개입
탐색적으로 선택한 코너 transit 739결정의 원본 FlyVis 출력과 몸 감각을 7개 독립 recurrent 이력으로 재생했다. 원본 두 반복은 조향/전진 intent 차이 0이었다. front/left/right/rear 운동 채널을 각각 학습 평균으로 대체한 경우 평균 절대 변화는 조향 0.03050/0.02395/0.01112/0.01859, 전진 0.06600/0.04608/0.09283/0.06394였다. 각각 738/739결정에서 적어도 한 출력 차이가 2e-6을 초과했다. 네 방향을 모두 대체하면 평균 변화는 0.04885/0.15060이다. 이는 각 채널이 이 고정 회로 출력에 영향을 준다는 이력 수준의 관측이며, 방향 판별 정확도·파리 생리·시각 선행 예측·운전 성능을 증명하지 않는다. 밝기는 유지했고 평균 대체는 실제 카메라 결손이 아니다. 고정 15km/h 목표와 fit 금지 transit, 물리/노출 동기 미검증 제약을 유지한다. 첫 실행은 다른 구조 checkpoint 선택으로 초기 중단했고 원본을 보존했다. 성공 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-yaw-corner-motion-view-ablation-v2/`, 원본 사건 대조·통계 재계산 `.../20261031-yaw-corner-motion-view-ablation-audit/`(모델 독립 재실행은 아님). 학습/차량 제어 0, 테스트 12개 통과. 다음은 방향/속도별 입력에 대한 운동 시각 반응과 회로 기여를 생리 근거 및 독립 데이터와 대조하고, 실시간 freshness·가감속/변속/셋업 적응을 검증한다.


### 2026-10-31 몸 상태 검증 GPU 동기화 감소
`ExperimentalVisualBodyAirflow.advance`의 기존 state 유한성·drive 유한성·state [0,1] 범위 조건은 유지하고 GPU scalar 판정을 합쳐 host 대기를 줄였다. 에피소드 state·시간 적분·결측 처리·보호 경계는 바꾸지 않았다. GPU 단독 200쌍 교차 순서 격리 실험에서 검증 중앙/95/최대는 기존 0.2405/0.6095/0.8967ms, 변경 0.1648/0.2924/0.5636ms였다 (`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-body-validation-sync-benchmark/`). 시간은 해당 조건식 단독이며 동시 시각 처리나 실제 운전 지연 개선을 뜻하지 않는다. NaN/무한값/범위 이탈과 형상/시간 오류 거절 등 관련 19개 테스트 통과.

원본 53장면+334몸 사건 재생 `.../20261031-combined-body-validation-original-replay/`의 events와 frame-comparison SHA는 변경 전과 동일, split CUDA 387건 state/drive/trace/intent 최대차 0이다. `.../20261031-combined-body-validation-bridge-proof/`에서 이전 카메라 원본 74결정 재생 및 state 최대차 9.54e-7; 실제 물리/노출 시각과 동시 실시간 부하는 미검증이다. 실시간 증거 검사에 변경된 몸 회로 소스 해시를 추가하고 새 proof 조합을 통과시켰으며 해당 소스 해시만 고친 임시 proof는 거부했다. 감사 첫 시도는 시스템 기본 문자 인코딩으로 실패해 UTF-8로 수정했다. AC 조작·학습 0. 직전 live shadow freshness 0/103 병목은 아직 해결되지 않았고, 이 수십 마이크로초 차이를 전체 해결로 주장하지 않는다. 다음은 원본 full 경로의 host/GPU 동기 및 동시 GPU 경합을 줄이고 변경된 경로를 실제 shadow에서 자격 검증한다. 조향·가감속·변속·셋업 적응 및 유효 완주/레이싱 라인·랩타임 목표 유지.


### 2026-10-31 운동 시각 고정 채널 추출
시간 경계의 정규화된 4×9×16 입력에서 움직임 4×8×16 채널을 GPU boolean mask 대신 같은 순서의 고정 슬라이스로 추출한다. 저장된 mask가 원래 계약과 같은지 최초 및 tensor 변경 때 확인하고 불일치는 거부한다. 값과 입력 gradient의 정확한 일치, 변경 mask 거부를 포함한 관련 15개 테스트 통과. GPU 단독 200쌍 교차 실험에서 이 추출 연산 중앙/95/최대는 mask 0.1749/0.5546/0.6864ms, slice 0.0952/0.2720/0.4051ms였다 (`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-temporal-motion-index-benchmark/`). 다른 GPU 작업과 함께 측정한 실시간 이득은 아니다.

원본 387사건 `.../20261031-static-motion-index-original-replay/`에서 기존 events/frame-comparison SHA와 동일, split CUDA state/drive/trace/intent 차이0, reset0이다. `.../20261031-static-motion-index-bridge-proof/`의 원본 카메라 74결정 재생은 state 최대차 9.54e-7이다. 실시간 proof의 시간 경계 소스 의존성을 추가했고 변경 전 proof는 거부, 두 새 proof 조합은 통과했다. 이전 학습/실험의 정확한 소스 해시를 요구하는 도구는 변경된 현재 소스를 과거와 같다고 취급하지 않으며 과거 원본 증거를 유지한다. 실제 실시간 freshness 개선·생리·폐루프 운전은 미검증, 학습/차량 제어 0이다. 다음은 전체 동시 GPU 경로와 host 대기를 계측하고 충분한 개선 뒤 실시간 shadow를 다시 검증한다. 목표의 가감속·변속·셋업 적응, 유효 완주와 레이싱 라인/랩타임 최적화는 그대로다.

### 사전 등록 — 개선 코드의 동시 시각·몸 shadow 검증 (2026-10-31)
새 body 검증 및 운동 채널 추출 코드의 387사건 split proof와 74결정 bridge proof를 고정한다. 기존 bounded mailbox 조건의 default GPU priority, 40ms 카메라·10ms 몸 주기·3초 15km/h 엔진 AI 진단과 CUDA event 계측을 사용한다. 학생 명령·학습·UI 조작은 없고 오류/오래된 감각은 완화 없이 기록한다. 직전 단일 진단과 비교하되 실행 간 부하 및 계측 차이가 있어 인과적 속도 개선으로 단정하지 않는다. CSP 상태 및 native AI off와 수집 소유권은 실행 직전 재검증하고 원본 및 종료 상태를 보존한다.


### 2026-10-31 변경 코드 동시 시각·몸 live shadow — 신선도 실패
새 코드와 새 proof 조합으로 기존 bounded mailbox·default GPU priority·40ms 영상/10ms 몸·3초 15km/h 교사 조건을 실행했다. 엔진 AI child 종료0·camera275그룹·native4423 copy이며 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-optimized-body-visual-live/`에 원본과 실패를 보존한다. ticker는 현재 몸 receipt stale로 중단됐고 pending visual 소비 전 ticker 중단 오류가 뒤따랐다. 학생 actuator·학습·UI0이다.

독립 raw 감사 `.../20261031-optimized-body-visual-live-audit/`: 104 ticker 출력 중 host freshness 통과 0, 이전 held body >50ms 103/103, 시야 >150ms 44/104. held 중앙/95/최대 82.07/94.99/105.92ms, decision→output 39.66/46.49/53.50ms다. `.../20261031-optimized-body-visual-live-gpu-audit/`: body event 중앙 38.90ms, visual event 중앙 16.44ms. GPU event는 제출 공백·대기·계측 오버헤드를 포함한다. 이 단일 실험과 이전 bounded-wait 0/105는 독립 실행이라 변경 효과나 원인을 인과적으로 확정하지 않는다. 실시간 운전 자격은 실패이며 물리/노출 시각 및 생리·폐루프·제동/변속/셋업·유효 완주/랩타임은 미검증이다.

시험 종료 직후 CSP paused/AI off였지만 frozen native packet 222987은 AI true여서 그대로 두지 않았다. CSP API restart/start/pause/resume 후 `.../20261031-optimized-live-reset-native-audit/`에서 기준 이후 새 packet 59/59 AI false·정차/브레이크/오프라인을 확인하고 `.../20261031-optimized-live-reset-final-pause/`로 pause했다. 최종 확인 PID12432, generation `15533904ULL:0.0013473`, epoch8, paused·AI/probe/camera off·offline. 다음은 원본 시각 모델과 몸 회로가 GPU를 공유할 때 내부 host/GPU 대기와 kernel 제출을 분리하는 제어된 오프라인 동시 부하 시험 및 정확한 계산 경로 개선이다. 초파리 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차 몸에 연결하고 몸 기울기·회전·하중·타이어 되먹임을 바탕으로 조향·가감속과 빠른 레이싱 라인/랩타임을 학습하는 목표를 유지한다.

### 사전 등록 — 원본 FlyVis와 몸 회로의 오프라인 GPU 동시 부하 (2026-10-31)
원본 387개 몸 사건을 순서·recurrent state·모델·시야 특징 그대로 재생한다. 동일 실행 코드의 무부하 대조와, 별도 CUDA stream/worker에서 원본 기록의 망막 영상을 반복하며 고정 FlyVis 상태를 5ms×8 tick씩 40ms 목표로 전진시키는 동시 부하를 차례로 실행한다. 각 몸 update와 시각 workload의 host 시작/완료를 기록해 실제 중첩 수를 산출하고 몸 처리시간 중앙/95/최대를 비교한다. 두 결과의 신경 의도·원본 사건 hash가 대조와 같아야 한다. 반복 영상의 시각 이력은 처리 부하용이며 실제 카메라 시간·host freshness·생리·차량 성능 재현이 아니다. GPU 대기/제출을 포함한 벽시계 비교이며 커널 점유율이나 CPU/GPU 원인 분리로 해석하지 않는다. AI·차량 제어·학습0.


### 2026-10-31 원본 FlyVis 동시 부하 대조
동일 원본 53영상·334몸 사건/387 update를 순서와 원래 state 그대로 재생했다. 몸 단독→기록 망막 영상을 반복하는 FlyVis 5ms×8 tick/40ms 목표 부하와 동시 실행→몸 단독 반복 순서이며, 세 원본 events/frame-comparison SHA는 동일하다. 각 update의 host 구간과 시각 worker 구간은 별도 저장했고 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-contention-independent-audit-v2/`에서 해시·사건·중첩을 독립 재계산했다. 결과는 `.../20261031-contention-body-alone/`, `.../20261031-contention-body-plus-flyvis/`, `.../20261031-contention-body-alone-after/`다.

몸 update 중앙/95ms는 단독 전 8.18/13.34, 동시 14.70/25.68, 단독 후 9.41/13.53이다. 동시 실행의 시각 workload 309회 중 몸 구간과 겹친 몸 update는 160/387이고 겹친 경우 중앙/95 23.85/25.99ms, 겹치지 않은 경우 12.72/15.60ms다. 단독 최대 약480ms는 최초 cold 시작을 포함한다. 이는 제한된 한 순서의 오프라인 비교다. 시각 worker는 CPU 제출과 GPU를 함께 사용하므로 GPU 커널 점유율이나 GPU 단독 원인을 증명하지 않는다. 반복 망막 영상은 계산 부하용이며 원래 시간의 실제 시각 이력 또는 live freshness가 아니다. 이 결과는 동시 부하가 실제 원본 몸 재생을 지연시킨다는 직접 근거이며, 직전 live 중앙 39.66ms의 전체 원인·실제 운전/생리 자격을 설명하지는 못한다. 최적화·학습·차량 제어0. 다음은 동일 부하 조건에서 host 제출/동기와 GPU 실행을 분리하고 원래 회로·결측·state를 유지하는 계산/스케줄 개선을 검증한다. 조향·가감속·변속·셋업 적응, 유효 완주·빠른 라인/랩타임 목표 유지.

### 사전 등록 — 동시 부하 body CPU 프로파일 (2026-10-31)
같은 원본 387사건의 기존 무부하·FlyVis 동시 부하 replay를 Python cProfile로 각각 실행한다. 프로파일러가 보는 body 호출의 inclusive/self 비용·호출 수와 host scalar 반환/동기 경계를 비교한다. cProfile은 주 스레드만 보며 GPU 대기·다른 스레드 영향·계측 오버헤드가 섞인다. 설치된 torch profiler의 지원 활동은 CPU만이므로 GPU kernel 실행시간/점유율 또는 CPU와 GPU의 순수 시간 분리는 주장하지 않는다. 원본 events hash·부하 중첩·오류를 보존하고 실제 제어/학습은 하지 않는다.


### 2026-10-31 시간 경계 유효성 동기 묶음
기존 원본 이력 무부하/시각부하의 Python cProfile을 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-contention-profile-alone.pstats` 및 `.../20261031-contention-profile-loaded.pstats`에 보존했다. 시각 부하 시 초기화·추가 warmup 및 torch device wrapper가 함께 기록되어 두 전체 통계는 동등한 순수 CPU/GPU 비교가 아니다. 설치 torch profiler는 CPU activity만 지원한다. 시간 경계에서 GPU scene/시야 유효성 두 판정과 세 과거 trace 유한성 판정을 각각 묶어 host scalar 대기를 줄였다. 기존 검사는 유지하고 잘못된 scene, 각 trace의 NaN, 시야 결측과 gradient 등 관련 19개 테스트를 통과했다.

같은 387개 원본 사건의 동시 FlyVis 부하 replay `.../20261031-temporal-batched-checks-contention/`은 변경 전 events/frame-comparison SHA와 같다. 변경 전 단일 동시 시험 중앙/95 14.70/25.68ms, 변경 후 단일 동시 시험 13.16/24.19ms; 실제 host 구간이 겹친 경우 전 160개 중앙23.85ms, 후143개 중앙22.39ms다. 실행 간 overlap·부하·환경이 달라 수치를 인과적 향상이나 실시간 통과로 판정하지 않는다. 새로운 split proof `.../20261031-temporal-batched-checks-split-proof/`는 387건 state/drive/trace/intent 차이0, bridge proof `.../20261031-temporal-batched-checks-bridge-proof/`는 74결정 state 최대차9.54e-7이며 두 proof 조합을 검사했다. source-exact 과거 proof는 새 소스에 재사용하지 않는다. 실제 live freshness·물리/노출 시간·생리·가감속/변속/셋업·완주/라인/랩타임은 미검증, 학습/차량 제어0이다. 다음은 GPU 실행과 host 제출을 직접 구분할 수 있는 계측 경로를 찾거나, 동일 회로를 보존한 계산 스케줄 변경 후 동시 부하와 실제 shadow에서 검증한다.

### 사전 등록 — 시각 GPU 묶음 크기/몸 회로 대기 (2026-10-31)
바로 전 오프라인 동시 부하를 같은 실행 코드로 반복한다. 네 시야의 FlyVis 계산은 원본 5ms 단계 여덟 개와 동일한 held 영상·연속 상태를 유지하고 GPU 제출 묶음만 8틱 또는 2틱으로 나눈다. 각 묶음 완료를 기다리며 두 경우 같은 초기 상태·영상에 대한 전체 노드 activity가 2e-6 이내인지 확인한다. 몸 회로는 같은 387사건을 보존하고 실측된 host 중첩과 몸 시간·8틱 visual 완료 시간·실제 workload 횟수를 함께 보고한다. 이것은 반복 영상의 오프라인 부하, 단일순서 실험이며 actual camera latency·GPU 점유율·폐루프 성능이 아니다. 실패하면 변경을 실제 경로에 적용하지 않는다.


### 2026-10-31 FlyVis 묶음 크기 동시 부하 비교 — 채택하지 않음
기록 망막 영상의 반복 부하에서 원래 FlyVis 5ms×8단계를 통째로 제출한 경우와 2단계씩 네 번 제출한 경우를 같은 몸 원본 387사건으로 비교했다. 먼저 하나의 같은 초기 시각 state/영상에서 8단계 최종 전체 node activity를 대조했으며 8 대 8 반복의 최대차 7.15e-7, 8 대 2×4 최대차 4.77e-7이다. 이는 단일 8단계 비교이며 긴 시각 이력의 전체 동등성 증명이 아니다. 두 몸 사건 결과 SHA는 동일하다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-visual-chunk8-contention/`, `.../20261031-visual-chunk2-contention/`, 독립 시각/몸 host 시간 계산 `.../20261031-visual-chunk-contention-audit/`.

몸 처리 중앙/95ms는 8단계 묶음 13.99/24.82, 2단계 묶음 14.16/25.79다. 실제 host 구간이 겹친 몸 148 대 186건의 중앙은 22.77 대 22.08ms다. 시각 8단계 전체 완료 중앙/95는 각각 11.61/12.48 대 16.60/18.24ms다. 단일 순서·실행의 차이로 인과적 수치 확정은 불가능하지만, 이 결과는 몸 지연의 뚜렷한 개선 없이 시각 완료가 늦어진 사례다. 2단계 분할을 실제 경로에 채택하지 않았다. CPU 제출·GPU 대기는 분리되지 않고 반복 영상 부하는 실제 카메라/physiology/live freshness가 아니다. 학습·차량 제어0. 다음은 동일 상태·결측을 유지하면서 GPU 반복 제출/동기화와 실시간 소비 구조의 다른 병목을 검토한다. 조향·가감속·변속·셋업 적응과 유효 완주/빠른 라인·랩타임 최적화 목표는 유지한다.

### 사전 등록 — 동일 희소 몸 회로 CPU thread 수 비교 (2026-10-31)
24 logical core 장비에서 같은 원본 387사건·9728노드/1311145엣지·원래 신경 state와 시간 적분을 유지하며 GPU 기준과 CPU CSR 경로를 비교한다. 기존 실험의 CPU intra-op thread 1을 기준으로 2·4·8을 차례로 실행하고 전체 상태·drive·trace·intent GPU 최대차 2e-6 이하, CPU 경로 중앙/95/최대 처리시간을 보고한다. 전체 원본 events hash와 실제 thread 설정을 보존한다. 기존 구현대로 GPU scene→CPU 전송과 동시 시각 부하는 CPU 처리시간에서 제외되므로 실시간 자격이나 실제 제어 개선 판정에 사용하지 않는다. GPU·CPU 계산 순서가 고정된 단일 시행이며 무작위 순서 인과 비교가 아니다. 학습/차량 제어0.


### 2026-10-31 CPU 희소 회로 thread 수 비교 — CPU 이전 미채택
동일 9728노드/1311145엣지 회로·원본 387사건·연속 상태를 GPU 및 CPU CSR로 재생하며 CPU intra-op thread 2→4→8→1 순서로 시험했다. CPU-only body 경로 중앙/95ms는 각각 thread1 33.84/48.85, thread2 38.89/51.24, thread4 38.95/50.91, thread8 39.14/49.98이다. 각 실험에서 GPU 대비 CPU state/drive/trace/intent 최대차 5.18e-7, 원본 events SHA 동일이다. 실행별 증거 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-cpu-body-threads1-after/`, `.../20261031-cpu-body-threads2/`, `...threads4/`, `...threads8/`. `.../20261031-cpu-body-thread-audit/`는 설정·결과·사건 hash를 확인했으며 개별 CPU 시간 원자료는 저장되지 않아 요약 수치를 독립 재계산하지 않았다.

이는 단일 비무작위 순서의 오프라인 측정이며 CPU 시간에 GPU scene 복사 및 실제 동시 시각 부하는 포함되지 않는다. 이 범위에서는 thread 증설로 CPU 이전의 이득을 보여주지 못했고 live 경로에 적용하지 않았다. 기전·생리·실시간 freshness·폐루프 운전은 미검증, 학습/차량 제어0이다. 다음은 원래 희소 연결·시간 상태를 유지한 별도 CPU 커널 또는 GPU/host 동기 구조 개선을 계측한다. 초파리 겹눈/운동 시각과 비행 유래 균형·운동 회로, 몸 기울기·회전·하중·타이어 피드백, 조향·가감속·변속·셋업 적응 및 빠른 유효 완주/레이싱 라인/랩타임 목표를 유지한다.

### 사전 등록 — 고정 비행 CSR CPU 커널 비교 (2026-10-31)
기존 checkpoint에서 서명·정규화된 고정 희소 행렬을 추출해 PyTorch CPU CSR과 SciPy CPU CSR의 같은 입력 행렬곱을 비교한다. 동일 유한 합성 상태·입력에서 100회 원래 활성화/시간 적분 전체 state 최대차 2e-6 이하를 요구하고, 단일 thread의 행렬곱 200쌍을 교차 측정해 원본 timing을 보존한다. 학습/gradient·실제 감각/전체 스트림·동시 시각 및 차량 제어는 포함되지 않으며, 성공해도 실시간 적용 전에 원본 사건 이력과 결측/시간/모델 버전 계약을 검증한다.

### 사전 등록 — SciPy CPU CSR 원본 이력 검증 (2026-10-31)
위 합성 미세측정 후 원래 387사건·동일 checkpoint·네 시야/몸·시각·recurrent state를 GPU 기준과 추론 전용 CPU SciPy CSR로 재생한다. 고정 그래프의 모든 연결과 가중치, 활성화·적분·마스크를 유지하고 모든 사건의 state/drive/trace/intent 최대차 2e-6 이내를 요구한다. CPU 처리시간 원자료를 새로 보존하며 scene GPU→CPU 복사와 동시 FlyVis 부하는 시간에서 제외한다. gradient 활성화·학습·GPU 입력은 해당 backend가 거절한다. 단일 원본 성공은 실제 live 신선도·독립 주행 또는 생리 검증이 아니다.


### 2026-10-31 고정 비행 회로 SciPy CPU CSR 추론 후보
원래 9728노드/1311145연결·signed weights를 그대로 가진 별도 `scipy_csr_cpu` 추론 backend를 선택형으로 추가했다. CPU float32·고정 가중치·no_grad만 허용하고 gradient 활성화, gradient 요구 state/drive, 다른 장치/형식을 거부한다. matrix 변경 시 cache를 갱신하며 기존 GPU/CPU backend와 학습 기본값은 유지한다. cache 변경, 배치 수치, gradient 거부 등 관련 28개 테스트 통과. 합성 입력의 100단계 전체 state 차이0; 격리 행렬곱 중앙/95ms PyTorch 7.85/11.83, SciPy 0.68/0.84 (`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-frozen-cpu-csr-kernel-comparison/`). 합성 반응만으로 전체 회로 성능을 주장하지 않는다.

원래 53영상+334몸 사건의 GPU 기준과 각 CPU 경로를 같은 코드에서 독립 비교했다. SciPy `.../20261031-scipy-cpu-original-replay/`와 PyTorch CPU `.../20261031-torch-cpu-original-replay-comparable/`는 각 387사건 GPU 대비 state/drive/trace/intent 최대차 5.18e-7, events/frame-comparison SHA 일치다. CPU 처리 중앙/95/최대ms는 SciPy 5.26/15.37/203.46, 기존 29.72/51.08/170.16. `.../20261031-scipy-cpu-replay-audit/`가 양쪽 원본 timing 387개와 hash에서 수치를 다시 계산했다. 최대값은 초기 cold 및 실행 중 지연을 포함한다. 단일 순서·오프라인 측정이고 GPU scene→CPU 복사·동시 FlyVis 부하·실시간 scheduler는 제외한다. 따라서 배우 자격/실시간 freshness 개선/생리/폐루프 주행을 증명하지 않으며 아직 live에 연결하지 않았다.

수정된 회로 source를 실시간 GPU proof 의존성에 추가하고 `.../20261031-scipy-backend-gpu-split-proof/` 387사건 state 차이0과 `.../20261031-scipy-backend-gpu-bridge-proof/` 74결정 최대차 9.54e-7을 재검증했다. 학습·차량 제어0. 다음은 원본 시각 network GPU와 별도 CPU body가 동시 동작할 때 scene 복사·스케줄·마스크 및 actual output freshness까지 검증하고, 이후 조향·가감속·변속·셋업 적응과 유효 완주/레이싱 라인·랩타임 목표를 계속한다.

### 사전 등록 — CPU SciPy 몸 회로와 FlyVis 동시 부하/scene 전송 (2026-10-31)
원본 387사건의 동일 GPU 기준과 SciPy CPU 몸 회로를 재생하며 새 장면이 올 때의 GPU→CPU feature 복사를 몸 update 시간에 포함한다. 같은 코드로 시각 부하 없음과 원래 FlyVis 8×5ms의 별도 GPU stream 부하를 비교하고, 각 몸 update·복사·시각 workload host 구간과 전체 state/drive/traces/intent 차이를 보존한다. 이 진단은 각 사건의 GPU 기준/반사실도 순차 실행하므로 실제 CPU-only live 대비 GPU 작업이 더 많다. 기록된 이미지를 반복하는 부하와 원래 사건 시각 사용은 실제 live scheduler/카메라 freshness가 아니다. 원본 데이터·결측·신경 시간·state를 보존하고 학습/제어0.


### 2026-10-31 SciPy CPU 몸 회로와 FlyVis GPU 동시 부하
동일 원본 387사건, body state·원래 시간·4시야 mask를 유지하고 새 scene GPU→CPU 복사를 CPU update 시간에 포함했다. 원래 FlyVis 모델의 8×5ms 시각 부하를 별도 GPU stream에서 반복한 오프라인 재생 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-scipy-cpu-transfer-flyvis-load/`에서 CPU update 중앙/95/최대 6.55/10.97/17.79ms, 53개 새 scene 복사 중앙/95/최대 0.285/0.489/0.580ms였다. 시각 host 구간과 겹친 CPU update는 69/387, 그 중앙/95 7.23/16.70ms; 나머지 중앙/95 6.50/7.41ms다. 같은 코드의 시각부하 없는 `.../20261031-scipy-cpu-transfer-no-load/` 중앙/95 5.18/15.66ms이고 cold 포함 최대205ms다. 각 조건의 CPU 대 GPU state/drive/trace/intent 전체 387개 차이 최대5.18e-7, 원본 events SHA 일치. `.../20261031-scipy-cpu-visual-contention-audit/`가 저장된 시각·전송·몸 원시 시간과 차이 배열·해시를 독립 계산했다.

두 단일 실행은 부하·overlap이 다르고 GPU 기준/반사실도 순차 실행하므로 실제 CPU-only 시스템의 정확한 시간이나 인과적 개선으로 단정하지 않는다. 원래 망막을 반복한 별도 GPU 계산은 실제 camera timing이 아니고 결정시간은 저장된 host 이력이다. 95백분위가 주기 10ms를 넘는 구간도 있다. 실제 live output freshness, 물리/노출 동기, 센서→행동 폐루프·생리·가감속/변속/셋업·유효 완주/랩타임은 미검증이며 CPU backend는 여전히 비배포, 학습/차량 제어0이다. 다음은 CPU body와 GPU visual의 실제 소유권·복사·결측·상태·freshness 계약을 분리 구현하고 원본 비교 후 shadow로 검증한다. 전체 초파리 겹눈·운동 시각 및 비행 유래 몸 제어와 몸 기울기·회전·하중·타이어 되먹임, 빠른 유효 주행·라인/랩타임 목표는 그대로다.

### 사전 등록 — 실제 CPU body shadow 원본 이력 검증 (2026-10-31)
GPU 시각 producer와 CPU SciPy 고정 회로를 소유하는 연구용 shadow를 만든다. 새 scene은 GPU producer 완료 뒤 CPU ticker가 한 번 복사하고 복사 시간을 결정→출력에 포함한다. 네 시야·원본 몸 정보·결측·원래 recurrent 상태를 유지하며 다른 장치/잘못된 scene 및 CUDA body timing 요청은 거부한다. 독립 387 원본 이력에서 실제 shadow의 CPU stream과 기존 GPU stream의 state/drive/trace/intent를 비교하고 max2e-6을 요구한다. 이 시험은 별도의 실제 실시간 camera/ticker가 아니며 actor·학습·제어0. 결과와 source를 묶어 통과하기 전 live 사용하지 않는다.

### 사전 등록 — CPU body / GPU FlyVis 실제 shadow (2026-10-31)
원본 387사건의 CPU shadow proof와 새 GPU split/bridge proof 및 각 소스·checkpoint 해시를 확인한 뒤 기존과 같은 단일 3초 15km/h 교사·카메라40ms·body10ms·bounded mailbox 조건을 실행한다. GPU 시각은 기존과 같고 실제 ticker의 CPU scene 복사와 전체 계산을 출력 시각에 포함한다. body CUDA event는 CPU에서 불가능해 계측하지 않는다. 이전과 같은 원본 native first receipt·held age·네 render upper age를 출력 완료 기준으로 감사하며, stale·결측·오류를 완화하거나 재시도하지 않는다. 학생 actuator·학습·UI0. 물리/노출 원시 시각·생리·폐루프 운전은 이 시험으로 증명하지 않는다. 종료 후 CSP 및 독립 native AI 상태를 확인한다.


### 2026-10-31 GPU 시각·CPU body shadow 실시간 시험 — 부분 신선도, 미자격
GPU FlyVis producer와 독립 CPU SciPy 추론 회로를 가진 `CpuBodyLiveShadow`를 구현했다. 새 scene만 GPU에서 복사하고 복사·몸 계산·출력까지의 host 시간을 기준으로 기존 시야/몸 freshness를 평가한다. CPU body CUDA event 사용·잘못된 scene 거부, 복사 독립성 등 관련 18개 테스트 통과. 실제 shadow 객체의 387 원본 사건 재생 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-cpu-body-shadow-original-proof/`: GPU 대비 state/drive/trace/intent 최대차5.18e-7, events SHA 기존과 동일, CPU 처리(복사 포함) 중앙/95/최대7.14/9.45/17.61ms. 시각 부하 동시 재생은 포함되지 않는다. 최신 GPU split/bridge proof와 CPU source·checkpoint·이력 proof를 정확히 확인한 뒤 시험했다.

3초 15km/h 엔진 AI live `.../20261031-cpu-body-live-shadow/`는 child 종료0·카메라276그룹·native4407 copy·학생 actuator/학습/UI0. ticker 250출력 뒤 current body receipt stale로 중단해 run 상태는 incomplete다. 독립 `.../20261031-cpu-body-live-shadow-audit/`에서 output host freshness 242/250 통과, 첫 출력은 이전 held 감각이 없어 기준 불충족, 나머지 7개는 held >50ms, 네 시야 >150ms는0이다. held 중앙/95/최대35.09/44.21/70.93ms, decision→output 중앙/95/최대16.20/23.90/36.57ms; 106 scene을 원본 RGB/packet/시각으로 검증했다. 앞선 GPU live 0/104와 각각 단일 실행·계측 차이가 있어 인과적 속도/일반화 결론은 내리지 않는다. 모든 신선도 통과는 아니며 정지 후 stale 종료까지 포함해 배우 자격은 실패다. 센서 물리 발생/노출 동기, 실제 폐루프 운전·생리·가감속/변속/셋업 및 유효 완주/레이싱 라인·랩타임은 미검증이다.

종료 후 CSP는 paused/AI off였으나 정지된 native packet232012은 AI true였다. 소유 CSP restart/start/pause/resume 후 `.../20261031-cpu-shadow-reset-native-audit/`에서 새 packet59/59 AI false·정차/브레이크/오프라인 확인, `.../20261031-cpu-shadow-reset-final-pause/`로 pause. 마지막 확인 PID12432/generation `15533904ULL:0.0013473`, epoch9, paused·AI/probe/camera off·offline이다. 다음은 7회 held stale 및 최종 current stale의 원본 시간·lifecycle 원인을 분리하고 기준 완화 없이 연속 신선도를 개선해 재시험한다. CPU 경로는 shadow만이며 실제 배우 운전에 미승격이다. 초파리 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결하며 몸 기울기·회전·하중·타이어를 되먹임해 조향·가감속·변속·셋업 적응 및 빠른 유효 완주/라인/랩타임을 학습하는 대전제를 유지한다.

### 사전 등록 — CPU body의 thread CPU/벽시계 시간 및 stale 종료 (2026-10-31)
바로 앞 CPU shadow와 같은 3초15km/h 교사·GPU 시각·CPU body·40ms 영상/10ms 몸 조건을 단일 반복한다. 매 ticker update의 Windows thread_time과 host 벽시계를 함께 보존하고 stale 중단의 최초 receipt·정지 감지 시각/packet을 기록한다. thread CPU는 해당 thread 실행만 세며 OS clock 분해능·함수 측정 오버헤드를 포함하고 다른 thread/GPU 총 사용량이 아니다. 두 시간의 차이를 스케줄/GIL/전송 등 대기의 합으로만 보고 단일 원인으로 부르지 않는다. 기준 완화·학습·학생 제어·UI0, 원본 실패와 종료 상태를 보존한다.


### 2026-10-31 CPU shadow stale 원인 감사와 thread CPU 계측
직전 `.../20261031-cpu-body-live-shadow/`의 held stale 7건은 선택시점 기준 최신 packet을 모두 선택했고 선택 receipt 나이는 최대4.85ms였다. 결정→출력은 각24.78–36.57ms, 바로 이전 결정→출력도22.04–36.57ms였다. 전체 결정 간격 중앙/95/최대16.49/24.36/37.23ms (`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-cpu-shadow-stale-timing-audit/`). 따라서 7건을 단순히 더 최신 packet 미선택 탓으로 돌릴 근거가 없고, 계산/스케줄 중 이전 입력의 보유 시간이 길어진 것이 관측된다. 첫 출력은 이전 held 부재로 별도다. 기존 기록은 ticker stop 정확 시각이 없어 최종 stale와 pause 선후를 확정하지 못했다.

CPU ticker의 thread CPU 시계와 stale stop 시각·packet을 기록하는 계측을 추가했다. 관련16검사와 새 원본 CPU 387사건 GPU 최대차5.18e-7·GPU split 차이0·bridge74결정 최대차9.54e-7 재확인. 같은 조건의 새 live `.../20261031-cpu-body-threadtime-live/`에서 교사 child 종료0, 출력263개 중 freshness260 통과·첫 held 부재1·held>50ms 2·시야>150ms0, current-body stale 중단·actor/학습/UI0 (`.../20261031-cpu-body-threadtime-live-audit/`). 이 단일 반복은 기준을 모두 만족하지 못했으며 이전 242/250과의 차이를 개선 인과로 판정하지 않는다. stop 시각26340.1756557, 동일 packet242010 최초 receipt26340.1246342, 감지 나이51.02ms이며 lifecycle native-after도 packet242010이었다. 정지와 관련된 원본 관측은 있으나 정확한 CSP pause의 host 시각은 확인되지 않아 인과 확정하지 않는다.

Windows thread_time의 실제 관측은 263개 중 0ms 101개, 15.625ms 143개, 31.25ms 19개로 양자화됐고 98개에서 CPU 숫자가 해당 벽시계 구간보다 컸다. 선언된 API 해상도와 달리 해당 짧은 update에서 계산/대기 시간을 개별 분리하기에 부적합하다 (`.../20261031-cpu-threadtime-independent-audit/`). 이 값을 뺄셈해 대기 원인을 주장하지 않는다. 종료 후 frozen native AI true였으므로 API restart/start/pause/resume, `.../20261031-cputime-reset-native-audit/`에서 새 packet59/59 AI false 확인 후 `.../20261031-cputime-reset-final-pause/`로 pause. 마지막 PID12432/generation `15533904ULL:0.0013473`, epoch10 paused·AI/probe/camera off·offline. 다음은 더 적절한 고해상도 스레드 계측 또는 별도 프로세스 분리로 GIL/CPU/GPU 대기와 ticker 지연을 구별하고 실시간 기준을 유지해 검증한다. 생리·실제 폐루프·가감속/변속/셋업 적응 및 유효 완주·라인/랩타임은 미완료다.

### 사전 등록 — 독립 CPU 몸 프로세스의 원본 이력 검증 (2026-10-31)
고정 SciPy CPU 비행 회로를 단일 소유 별도 프로세스로 옮긴 연구용 경로를 만든다. 부모는 원래 4시야 feature·현재/선행 몸 관측 및 원본 host 시각을 하나씩 순서대로 전달한다. worker는 state를 원래 episode 동안 유지하고 event sequence 불일치·결측/유효성 오류·timeout을 중단하며 자동 reset/retry를 하지 않는다. 원본 387사건에서 각 단계의 GPU 기준 state/drive/traces/intent 전체 차이 2e-6 이하를 요구한다. scene GPU→CPU 복사와 직렬 IPC 왕복을 시간에 포함하고 별도 검사 snapshot 왕복은 제외한다. FlyVis 동시 부하·실시간 scheduler·source 물리/노출 시각·폐루프 및 차량 제어는 검증하지 않는다. 성공 시 별도 프로세스가 종료됐는지 확인하고 원본/오류/해시를 보존한다.


### 2026-10-31 별도 CPU body 프로세스의 원본 이력 검증
기존 고정 SciPy 비행 회로를 소유하는 단일 CPU worker 프로세스를 추가했다. 부모는 기존 4시야 feature·현재/선행 몸 관측·원래 host 결정시각을 순서대로 보내며, worker는 원래 episode의 recurrent state를 유지한다. 이벤트는 한 번에 하나만 처리하고 sequence 불일치·timeout·연결 종료는 명시 실패, 자동 재실행/reset을 하지 않는다. 실제 episode 전에 별도 임시 상태로 원래 계산 cache를 준비하며 새 episode state는 따로 생성한다. 프로세스는 simulator/actuator에 접근하지 않는다. timeout·연결 EOF·순서 불일치와 회로 관련 16개 테스트 통과.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-cpu-process-original-replay-warm/`에서 원본 387사건 모두 GPU 기준 state/drive/trace/intent 최대차5.18e-7, events/frame-comparison SHA 일치, worker PID34464 정상 종료0을 확인했다. scene GPU→CPU 복사 및 동기 IPC 왕복 포함 시간 중앙/95/최대7.16/7.89/18.08ms이며, 각 단계 진단 snapshot 왕복과 동시 시각 부하는 시간에서 제외했다. `.../20261031-cpu-process-original-replay-audit/`가 저장된 387개 차이·시간과 hash를 다시 계산했다. warm-up 이전 첫 후보 `.../20261031-cpu-process-original-replay/`, 오류 전송 처리 수정 이후 `.../20261031-cpu-process-original-replay-v2/`도 보존한다.

이 경로는 오프라인 동기 원본 재생만 검증했고 아직 실시간 shadow의 카메라/ticker/신선도 계약에 연결하지 않았다. 별도 프로세스가 GIL 경합을 해결하거나 실제 주행을 개선했다고 판정하지 않는다. 물리/노출 시각·생리·폐루프 조향/가감속/변속·셋업 적응·유효 완주 및 라인/랩타임은 미검증이고 학습/차량 제어0이다. 다음은 독립 CPU process에 실제 GPU 시각 부하를 함께 주고 IPC 시간/순서·종료·stale를 계측한 뒤, 원본 proof와 정확한 source 해시를 묶어 shadow에 연결한다.

### 사전 등록 — 별도 CPU worker와 GPU FlyVis 동시 부하 (2026-10-31)
원본 387사건을 동일 process worker·IPC·scene 복사·연속 신경 상태로 재생한다. 같은 코드의 무부하와, 원래 FlyVis GPU 8×5ms 단계 부하가 별도 stream/worker로 동시 실행되는 조건을 순서대로 비교한다. IPC 부모 시작/scene 복사 완료/자식 처리 시작·완료/응답 수신과 시각 부하 구간을 기록하고, 각 사건 GPU 기준 state/drive/traces/intent 차이와 원본 SHA 및 자식 정상 종료를 검사한다. 진단 snapshot 왕복은 시간에서 제외하며 GPU 기준/반사실도 같은 프로세스에서 순차 실행하므로 실제 CPU-only live 비용과 다르다. 반복 영상은 실제 live 노출·scheduler가 아니고 학습/차량 제어0이다.

### 사전 등록 — process shadow adapter 원본 이력 (2026-10-31)
별도 CPU worker를 기존 ticker와 호환하는 adapter에 연결한다. 명시된 task·시각/결측 mask를 검사하고 원래 관측·시각을 순서대로 IPC 전송하며 body 응답 최대 대기는 50ms다. timeout·순서/입력 오류는 중단하며 재시도·state 재설정하지 않는다. 387 원본사건의 실제 adapter 출력을 GPU 기준 state/drive/traces/intent와 비교하고 source hash와 worker 정상 종료를 보존한다. 이때는 실제 live camera/ticker가 아니고 비동기성·freshness·차량 제어/학습은 검증하지 않는다.


### 2026-10-31 CPU 독립 프로세스 + FlyVis 부하 및 shadow adapter
같은 387개 원본 사건과 독립 CPU process를 원래 FlyVis GPU 부하와 비교했다. 새 scene 복사·IPC 왕복 포함 중앙/95/최대ms는 부하 없음 7.12/7.80/18.80, 부하 있음 7.38/8.28/18.41이다. 시각 host 구간과 겹친 CPU update64개 중앙/95 7.45/17.42ms, GPU feature 복사53개 중앙0.297ms다. 자식 실제 계산 중앙6.89ms, 부모 전체 시간과 자식 시간 차 중앙0.477ms(복사·IPC·스케줄 포함). 양 조건 387개 GPU 기준 state/drive/trace/intent 최대차5.18e-7, 원본 events SHA 동일·소유 process 종료0. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-cpu-process-visual-no-load/`, `.../20261031-cpu-process-visual-load/`, 원시 host 시각과 차이 배열 재계산 `.../20261031-cpu-process-visual-audit/`. 단일 비무작위 순서, 반복 망막 부하 및 GPU 기준/반사실 동시 계산으로 실제 live 동등 시간이나 GIL 해소를 증명하지 않는다.

별도 process를 기존 연구용 ticker에 연결하는 `ProcessBodyAdapter`와 `ProcessBodyLiveShadow`를 추가했다. 고정 task·mask를 검증하고 50ms 응답 timeout·sequence·결측 실패를 감추지 않는다. 실제 adapter 387원본사건 `.../20261031-process-body-adapter-original-proof/`는 GPU 대비 전체 state/drive/trace/intent 최대차5.18e-7, 원본 events SHA 동일, process 종료0; scene 복사·IPC 포함 중앙/95/최대7.26/8.28/18.15ms. `.../20261031-process-body-adapter-proof-audit/`가 해시·차이/시간을 재계산했다. 입력 계약·timeout·종료·기존 shadow 회귀 7검사 통과. 실제 카메라/ticker에서 process adapter를 아직 실행하거나 실시간 proof gate를 만들지는 않았다. 따라서 실시간 감각 freshness·물리/노출 시각·생리·폐루프 주행·가감속/변속/셋업 및 유효 완주/라인/랩타임은 미검증, 학습/차량 제어0이다. 다음은 정확한 source/checkpoint/adapter 원본 proof를 검증하는 독립 live shadow runner 및 실패 시 소유 process 종료 계약을 완성한 뒤 실시간 시험한다.

### 사전 등록 — 독립 CPU process body 실제 live shadow (2026-10-31)
최종 process adapter 387원본사건 proof·GPU split/bridge proof 및 정확한 소스/checkpoint hash를 요구한다. 기존 CPU thread shadow와 같은 단일 3초15km/h 엔진 AI·GPU FlyVis·40ms 영상/10ms 몸·bounded mailbox·동일 freshness 기준으로 측정한다. CPU process는 scene 복사·원본 몸 관측과 host 시각을 순서대로 IPC 받고, 50ms 응답 한도를 넘거나 오류가 나면 실패·중단한다. 준비 실패·종료 실패 시 소유 worker만 bounded 종료하며 재시작하지 않는다. 학생 명령·학습·UI0, 원본 camera/native/중단·worker 종료를 보존하고 별도 raw 감사 및 시뮬레이터 종료 확인을 수행한다. 물리/노출 시각·생리·폐루프 운전은 주장하지 않는다.


### 2026-10-31 독립 CPU process body live shadow — 출력 신선도 개선 관측, 종료 미자격
`ProcessBodyLiveShadow`용 독립 runner가 기존 GPU split/bridge proof와 정확한 process adapter·worker source/checkpoint/387사건 proof를 검사한다. 오래된 proof는 거부한다. 준비 실패나 timeout 뒤 소유 worker만 제한 대기 후 종료하고 재시작하지 않으며, 실패 시에도 ticker 원본을 보존한다. 관련6개 테스트 통과; 최종 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-process-body-preflight-final-proof/`에서 원본387사건 GPU 대비 state/drive/trace/intent 최대차5.18e-7·worker 정상종료0이다.

같은 3초15km/h 엔진 AI·GPU FlyVis·CPU process·영상40ms/몸10ms 조건 `.../20261031-process-body-live-shadow/`는 교사 child 종료0·camera280그룹·native4609 copy·worker 정상종료0/강제종료false·학습/학생 actuator/UI0이다. 원본 RGB·native 최초 receipt·clock 독립 감사 `.../20261031-process-body-live-shadow-audit/`에서 출력372개 중 host freshness371 통과; 첫 출력은 이전 held 부재로 불통과, 이후371/371은 통과했다. held>50ms 0, 네 시야>150ms 0; held 중앙/95/최대20.48/24.90/33.20ms, decision→output 중앙/95/최대7.84/9.87/18.92ms. 직전 CPU thread 260/263과 이 process 371/372는 각각 단일 실행이며 조건/시스템 부하 차이가 있어 차이를 인과나 반복 일반화로 확정하지 않는다.

최종 출력27156.5957 뒤 ticker는 packet250087의 최초 receipt27156.5827706이 52.32ms 지난 27156.6350944에 current-body stale로 중단했고 이후 시각 producer도 ticker 중단을 기록해 run status는 shadow_stopped다. 독립 control native-after도 같은 packet250087이지만 CSP pause의 정확한 host 시각은 확인되지 않아 이 종료를 정상 lifecycle이라고 확정하거나 실패를 지우지 않는다. 물리 발생/노출 동기·실제 배우 제어·생리·유효 완주/라인/랩타임은 미검증이다. 종료 직후 CSP AI off지만 frozen native는 AI true여서 API restart/start/pause/resume 후 `.../20261031-process-shadow-reset-native-audit/` 새 packet59/59 AI false·정차/브레이크 확인, `.../20261031-process-shadow-reset-final-pause/`로 pause. 마지막 PID12432/generation `15533904ULL:0.0013473`, epoch11 paused·AI/probe/camera off·offline. 다음은 실제 lifecycle 종료와 unexpected stale를 원본 증거로 구분하는 명시적 종료 계약 및 반복 조건 검증이며, actor/학습에는 승격하지 않는다. 목표는 초파리 겹눈·운동 시각 및 비행 유래 균형/운동 회로와 몸 기울기·회전·하중·타이어 되먹임으로 조향·가감속·변속·셋업 적응 및 빠른 유효 완주·라인/랩타임 최적화다.

### 사전 등록 — lifecycle 관찰 및 process shadow 반복 (2026-10-31)
이전과 같은 process body·GPU FlyVis·3초15km/h 교사·40ms 영상/10ms 몸을 한 번 반복한다. 제어 요청 전부터 child 종료 관측까지 검증된 CSP status를 10ms 목표로 읽고 각 read host 시작/종료·wall mtime·freshness·generation·pause/AI 상태·읽기 실패와 canonical hash를 저장한다. 각 값은 관측 시점의 증거이며 이전/다음 관측 사이의 정확한 물리 시각을 만들지 않는다. ticker stale 시각과 첫 fresh paused 관측의 선후 및 관측 공백을 감사한다. 기존 freshness/중단을 바꾸거나 종료를 성공으로 재분류하지 않고 학습/학생 actuator/UI0이다.

### 사전 등록 — 관찰된 owned pause에서 process shadow 종료 (2026-10-31)
동일 generation의 fresh status에서 실제 owned expert active/unpaused를 먼저 관찰한 뒤 paused/AI off/probe off를 관찰하면 host 관찰 시각과 shutdown 요청 시각을 보존하고 더 이상의 새 scene 소비를 중단, ticker를 협조 종료한다. 초기 pause·다른 generation·오래된 status·읽기 실패는 종료 신호가 아니다. 기존 stale 감지와 50ms/150ms 기준을 완화하지 않고, pause 이전의 예상 밖 stale나 bridge 오류는 실패로 보존한다. 같은 3초15km/h 교사·GPU 시각/CPU process·40ms/10ms 조건을 단일 실행하고 원본 scene/native/결정시각·observer·worker 종료를 독립 감사한다. 1회 통과만으로 학습/배우/생리·물리 노출·폐루프 주행을 주장하지 않으며 학생 명령/UI0이다.


### 2026-10-31 lifecycle 관찰·owned pause 종료와 process shadow 반복
직전 반복 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-process-body-lifecycle-observed/`는 raw 감사 `...-audit/`에서 372출력 중 첫 held 부재1·이후371/371 host freshness 통과, held>50ms0·view>150ms0, decision→output 중앙/95/최대7.97/9.35/19.21ms다. 하지만 status는 current-body stale로 중단된 incomplete다. `.../20261031-process-lifecycle-order-audit/`가 671건 fresh 동세대 status 관측과 원본 시각을 확인해 실제 owned active 관측 후 paused/AI off/probe off 관측이 stale stop보다 35.84ms 앞섰음을 확인했다. 정확한 pause 물리 순간과 source 발생/노출 시간은 확인하지 못했고, 이 증거로 기존 run을 성공으로 소급 변경하지 않는다. 종료 후 frozen native AI true를 API restart/start/pause/resume 후 `.../20261031-observed-process-reset-native-audit/` 새 packet59/59 AI false로 확인하고 pause했다.

이를 근거로 동일 세대의 fresh owned active가 먼저 관측된 이후 fresh paused/AI off/probe off가 관측될 때만 새 영상 소비를 멈추고 ticker에 협조 종료를 요청하도록 했다. 초기 pause·오래된/다른 세대·읽기 실패는 종료 신호가 아니며 stale/freshness 기준은 그대로다. 단위 검증9개 통과. 사전 등록한 단일 재시험 `.../20261031-process-body-owned-pause-live/`: 교사 child 종료0·camera260그룹·native4271 copy·ticker/bridge 오류0·worker 종료0/강제종료false·run completed_diagnostic, 학생 actuator/학습/UI0. 독립 raw 감사 `...-audit/`에서 365출력 중 첫 held 부재1·이후364/364 host freshness 통과, held>50ms0·view>150ms0; held 중앙/95/최대20.91/27.40/44.26ms, decision→output 중앙/95/최대8.00/10.78/21.61ms. `.../20261031-owned-pause-shutdown-audit/`는 fresh active 선행, pause 관찰 뒤0.46ms shutdown 요청, 관찰 시각·마지막 tick·child 종료·worker0 및 652개 관측/읽기 오류0을 확인했다. 첫 출력의 held 결측은 통과로 둔갑하지 않았다. 단일 교사 shadow이며 실제 actor·신경 생리·물리/노출 동기·유효 완주/라인/랩타임의 자격은 아니다.

이후 frozen native AI true를 API restart/start/pause/resume 후 `.../20261031-owned-pause-reset-native-audit/` 새 packet59/59 AI false·정차/브레이크로 확인하고 `.../20261031-owned-pause-reset-final-pause/`로 pause. 최종 PID12432/generation `15533904ULL:0.0013473`, epoch13 paused·AI/probe/camera off·offline. 다음은 다른 주행 조건과 반복에서 process shadow freshness·출력의 원본 replay를 검증하고, 필요한 시각·비행 제어의 인과적 기능 및 조향·가감속/변속/셋업 적응, 빠른 유효 완주와 레이싱 라인/랩타임 최적화로 진행한다.

### 사전 등록 — 완료된 process shadow 원본 영상·몸 출력 재생 (2026-10-31)
`20261031-process-body-owned-pause-live`의 독립 원본 감사와 소스/checkpoint·BMP 해시를 확인한다. 시각망은 최초 처리된 영상부터 몸 정보 부재 장면을 포함한 모든 원래 unique group을 visual_snapshot host 시각과 원래 held input으로 연속 재생한다. 실제 publication을 tick별 선택한 camera ID에 연결하고 원본 native packet의 최초 receipt·body encoder 및 시야별 prior/mask를 유지한다. 원래 body decision 시간 전체와 process recurrent state를 리셋 없이 재생해 각 live intent와 비교한다. 사전 허용 절대 오차 2e-6, 전 365결정의 차이와 source SHA를 보존한다. 이는 저장된 관측의 수치적 재현이지 물리/노출 동기, 생리 기능, 실제 actor나 랩타임 검증은 아니다. 학습/차량 제어0.


### 2026-10-31 완료된 process shadow의 원본 시각·몸 연속 재생
`20261031-process-body-owned-pause-live/`의 최초 처리부터 몸 context 부재148개를 포함한 네 시야 영상253그룹을 원본 visual_snapshot 시각·이전 영상 유지·동일 FlyVis state로 재생했다. 원본 archive의 모든 처리된 RGB 해시를 확인했고, 실제 공개105그룹·내부 시각 tick2135개를 보존했다. 이 105그룹과 원본 native 최초 receipt·body decision을 소유 process에 순서대로 전달해 365출력 전체를 비교했다. 조향/전진 intent 최대 절대오차 2.24e-7/1.94e-7, 2e-6 초과0/365, 원본 episode 중 reset0·worker 정상종료0이다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-process-shadow-live-original-replay/`. `.../20261031-process-shadow-live-original-replay-audit/`가 저장된 live 원본 packet·시각·frame과 365개 수치 오차를 독립 재계산했다(모델 자체를 두 번째 독립 재실행한 것은 아니다).

별도 `.../20261031-process-shadow-visual-context-raw-audit/`에서 원본 native copy 이력을 각 영상의 실제 drain 수까지 순서대로 다시 읽어 현재 몸과 네 view prior의 packet·최초 receipt·21값·유효 mask를 검증했다. 처리253그룹 중 현재 몸 부재148, 공개105그룹의 prior 결측4시야는 그대로였고 원본과 모두 일치했다. 이는 host 수신과 기록된 렌더 하한의 관계이며 물리 센서 발생·실제 노출 시각의 동기 증거가 아니다. 본 시험은 teacher 주행 중의 shadow 출력 재현으로 학생 제어, 시각/비행 생리 기여, 실제 제동·변속·셋업 적응, 유효 완주/라인/랩타임 최적화를 입증하지 않는다. fit/optimizer·차량 제어0. 다음은 원래 이력과 별도 조건에서 시각·몸 개입의 기능적 기여 및 process shadow의 반복 freshness를 검증하고 실제 가감속 출력과 폐루프 학습으로 진행한다.

### 사전 등록 — 기존 TRAIN00의 네 시야별 선행 몸 감각 (2026-10-31)
기존 역할 TRAIN episode00의 원본 raw 영상/actor packing 감사 및 기존 earliest-view 몸 맥락과 동일한 native 최초 receipt·render 시작 host 하한을 사용한다. 각 view별로 엄격히 선행한 마지막 receipt를 독립 선택하고 camera receipt까지의 나이 50ms 초과는 결측으로 남긴다. 다른 이력·DEV를 합치거나 원본 timestamp/role을 바꾸지 않는다. 값·mask·packet·host ordering과 기존 공통 prior를 감사하며, 이는 물리 생성/노출 동기나 생리·주행 성능이 아니다. raw 진단의 continuous_40ms_training_admissible=false 및 실제 제동 표적 부재를 그대로 유지하고 이 단계 optimizer/차량 제어0.

### 사전 등록 — 원래 TRAIN 여섯 이력의 시야별 몸 수신 다양성 조사 (2026-10-31)
원래 dataset manifest가 TRAIN으로 봉인한 여섯 source event/native 기록만 확인한다. actor 또는 teacher label NPZ를 열지 않고 native raw 해시·최초 receipt 순서·각 카메라 render 하한 및 camera receipt 50ms 상한으로 네 선행 packet/yaw 유효 범위를 계측한다. 동일 장면 내 서로 다른 packet 및 yaw 차이·시야별 결측을 기록하고 원래 역할을 유지한다. 영상 전체 재감사/물리 발생·노출 동기 검증이 아니며 어떤 학습 자격도 부여하지 않는다. optimizer/차량 제어0.


### 2026-10-31 TRAIN 이력 및 고빈도 진단의 시야별 몸 수신 감사
기존 TRAIN00의 741결정×4시야를 원본 native 최초 receipt, 카메라 render host 하한과 50ms 나이 기준으로 독립 재구성했다. 1,956시야는 매칭, 1,004는 오래됨, 4는 선행 packet 부재였다. 부분적으로 시야별 유효성이 다른 장면은 18개다. 원본과 값·mask·packet·receipt가 일치했다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-train00-per-view-independent-audit/`; 생성 결과 `.../20261031-train00-per-view-body/`. 원래 기록의 continuous_40ms_training_admissible=false와 제동 표적 부재는 유지한다.

원래 TRAIN 여섯 이력의 4,449결정을 조사하고 별도 NumPy 구현으로 재확인했다. 같은 장면에서 유효한 시야별 prior packet이 서로 다른 경우는 0개, 일부 시야만 유효한 장면은 119개다. 저장된 native 표본 간격 중앙 약 31.7ms, 같은 장면의 render host 시각 범위 중앙 약 0.75ms는 기록 간격이지 물리 센서 주기나 노출 시각이 아니다. 결과 `.../20261031-train-six-per-view-yaw-survey/`, 독립 감사 `.../20261031-train-six-view-yaw-independent-audit/`. 전체 영상 재감사나 학습 자격 판정은 아니다. 공통 yaw와 네 시야 운동의 결합 학습은 가능하나 시야별로 다른 과거 packet의 효과를 이 자료만으로 분리 검증할 수 없다.

기존 저속 teacher shadow의 고빈도 native copy를 원본 drain 순서로 재구성한 `.../20261031-process-shadow-visual-context-raw-audit-v2/`에서는 공개 105장면 중 39개가 서로 다른 유효 시야 prior packet을 가졌다. 장면 내 yaw 범위 중앙/95%/최대는 0/0.00335/0.00977 rad/s다. 253개 처리 장면 중 현재 몸 부재 148개와 prior 결측 4시야도 보존하고 값·mask·receipt 일치를 확인했다. 이 기록은 사전 지정된 진단이며 학습 자료로 승격하지 않는다. 물리 센서 발생과 카메라 노출 동기, 학습된 생리 기능, 실제 제동/변속 및 폐루프 성능은 여전히 미검증이다. 이번 조사에서 optimizer 및 학생 차량 제어는 실행하지 않았다.

다음 수집은 TRAIN/DEV 역할을 사전에 분리하고 고빈도 native 최초 receipt와 네 화면 원본·결측을 함께 저장해 원본 감사한다. 코너 회전과 제동을 포함한 다양한 조건 및 라벨·제어 권한·시간 적합성을 별도로 검증한 뒤 시야×몸 경계의 학습 및 독립 개입 실험을 진행한다. 공통 회전 감각 결합 가능성과 시야별 수신 차이의 검증을 구별한다.


### 2026-10-31 학습용 수집기의 고빈도 native 이력 연결
`record_ac_engine_ai_probe.py`에 선택형 `--native-copy-history`를 추가했다. 카메라 준비 뒤, 제어 활성화 전에 기존 2ms 목표 bounded native copy sampler를 시작하고, 수집 종료 시 원본 authority/copy 시간 이력과 SHA-256을 보존한다. 오류 또는 빈 이력은 전체 기록 실패로 처리하며 부분 이력은 남긴다. 이는 host 수신 이력이고 실제 발생/노출 시각이나 학습 적합성을 보증하지 않는다. `record_pace_task.py`의 사전 계획에 명시적 `native_copy_history: true`와 수집기·sampler·native decoder 소스 hash가 있을 때만 옵션을 전달한다. 기존 계획의 기본 동작과 데이터 역할은 바뀌지 않는다. 관련 테스트 12개 통과; 실제 새 수집·물리 시간·학습·차량 제어는 아직 검증하지 않았다. 다음은 사전 역할 지정 및 최신 카메라/권한 조건을 확인한 새 수집, 원본 이력과 부하·누락 감사다.


### 사전 등록 — 고빈도 native copy 수집기 단일 실시간 진단 (2026-10-31)
현재 동일 generation의 fresh paused·AI off·background·vJoy released·정지/브레이크 조건을 확인하고, 기존 3초 15km/h engine AI 프로필에서 네 시야와 새 native copy history를 동시에 기록한다. 원본 camera/native 및 host ordering, 기록 공백과 부담을 독립 감사한다. 역할은 diagnostic 고정, 사용자 actuator·학습·GUI 조작 0이며 학습으로 승격하지 않는다. 원래 실패와 누락을 보존하고 물리 sensor 발생/노출 시간 또는 actor 주행의 증거로 간주하지 않는다.


### 2026-10-31 고빈도 native 수집기 실시간 진단 결과
사전 지정한 3초15km/h engine AI 단일 진단 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-native-copy-collector-live-diagnostic/`은 종료코드0·74결정·camera wait miss0·고빈도 copy 3,995개를 기록했다. `...-raw-audit/`에서 원본 영상·패킹 전 항목 오차0, aid off, decision 최대 간격42.27ms·80ms 초과0을 확인했다. `...-native-audit-v2/`는 원본 raw audit의 manifest/event 결속과 각 copy base64 hash·packet 순서·최초 수신을 재검증했다. 고유 packet 1,393개, copy 간격 중앙/95/최대 2.50/2.83/5.67ms, 74장면 중 27개에서 시야별 유효 선행 packet이 달랐다. 엄격 선행 부재0시야·50ms 초과4시야는 결측으로 남겼고, 유효 prior 나이 중앙/95/최대9.36/11.81/14.02ms다. 이는 host 수신 시각이지 물리 발생·노출 시각이 아니며 단일 저속 진단, no fit/학생 제어/GUI다. 새 고빈도 이력은 저속 진단에서만 실측했으며 pace30, TRAIN/DEV 자격·학습·시각/비행 생리 기능 및 실제 완주 성능은 미검증이다. 종료 뒤 같은 generation PID12432에서 fresh CSP paused·AI/camera/probe off·정지 브레이크 및 background/vJoy released를 확인했다(해당 관측 시점 기준). 다음은 사전 역할을 부여한 pace 조건의 수집과 원본·권한·제동·시간 검증이다.


### 사전 등록 — 고빈도 native copy의 pace30 진단 (2026-10-31)
저속 원본 감사 뒤 현재 paused 위치 spline 약0.85836은 기존 TRAIN 시작 구간0.275–0.44 밖이다. 따라서 role은 diagnostic으로 고정하고 `configs/flight-pace30-native-copy-diagnostic-20261031.json`의 기존 60→100→0→60 pace30을 한 번만 실행한다. 기존 camera qualification은 이전 엔진 세대의 증거이므로 사후 원본 카메라/결측·native first receipt·실제 제동·ownership 및 기록 부하를 새로 감사한다. 샘플 2ms는 목표값이며 물리 발생/노출 증거가 아니다. 실패나 중단은 그대로 보존하며 fit/학생 actuator/GUI 0, 역할 변경0이다.


### 2026-10-31 고빈도 pace30 진단 실패와 원본 부분 조사
사전 등록 diagnostic `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-native-copy-diagnostic/`에서 30초 control child는 종료0, native 최고속 약101.04km/h·API pace brake peak 약1.0, 729결정·copy 15,236개를 남겼다. 그러나 카메라 worker가 `Shared four-view GPU group deadline exceeded`로 실패해 원본 전체 수집은 failed다. 원본 camera raw audit `.../20261031-pace-native-copy-failed-raw-audit/`는 모든 생산 frame의 aid-state 수량 검증에서 중단됐다. `.../20261031-pace-native-copy-failed-native-inspection/`에서 보존 zip의 frame metadata 952개와 완료 group 카운트 951개의 불일치를 확인했다. 마지막 미완료 group과 count 불일치를 삭제하거나 정상 처리하지 않는다.

부분 조사에서 원본 고유 packet 10,713개, copy 간격 중앙/95/최대2.50/2.81/7.00ms, 729장면 중 209개에서 시야별 선행 packet이 달랐고 50ms 초과4시야가 있었다. 카메라 원본 감사 실패 상태이므로 이 값은 예비 관측이다. 반복 copy를 포함한 native 기록에서 AI flag true·속도10km/h 초과·brake>0.05인 copy 1,410개와 최고 제동값1.0을 관찰했으나 독립 제동 시각/라벨 자격은 아니다. GPU 감사 `.../20261031-pace-native-copy-gpu-stall-audit/`에서 완성 951그룹의 flush 중앙/95/최대0.149/0.308/2.690ms, 실패 group의 flush0.158ms 후 map에서 40ms 기한 초과를 확인했다. GPU 근본 원인이나 sampler와의 인과관계는 미확정이다. 학생 제어·fit/GUI 0, training false·role diagnostic 유지. 종료 후 동일 PID12432/generation의 paused·AI/camera/probe off 및 background/vJoy released, 속도0.01km/h·brake1 확인(당시). 다음은 lifecycle 종료와 GPU map 기한 초과의 원본 순서를 분석하고 카메라 종료 안전성을 개선한 후 별도 사전 등록 수집으로 검증한다.


### 사전 등록 — AI 종료 관측 후 camera stop pace30 (2026-10-31)
이전 실패의 GPU group 시작은 동일 세대 clock bounds에서 AI disable 후2.3868–2.3902초, pause 후0.2088–0.2122초였다(`20261031-pace-native-copy-end-order-audit/`). 물리 시각 및 원인 확정이 아니며 이전 실패는 유지한다. 새 독립 diagnostic에서 기존 fresh same-generation 상태로 owned AI active를 관찰한 뒤 처음 inactive를 관찰했을 때만 카메라 producer에 stop을 요청하고 동일 control child는 끝까지 기다린다. 카메라 그룹을 자르거나 오류를 성공으로 바꾸지 않는다. 원본 camera·native·stop host 시각·control 결과를 다시 감사한다. 소스 hash는 새 plan에 고정, 학습/학생 제어/GUI0, 역할 변경0이다.


### 2026-10-31 AI 종료 관측 후 camera stop의 pace30 실시간 진단
이전 실패의 GPU copy 시작이 AI disable 후2.3868–2.3902초, pause 후0.2088–0.2122초였음을 clock bounds로 확인했다(`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-native-copy-end-order-audit/`). 원인은 확정되지 않았고 실패를 성공으로 변경하지 않았다. 선택형 종료 방식은 동일 generation에서 AI active 관측 후 inactive를 처음 관측하면 카메라 stop만 요청하고 기존 control child 종료를 기다린다. 다시 active가 되면 실패 처리한다. 기존 gate·timeout과 raw 데이터는 유지하며 관련 12테스트 통과.

새 diagnostic `.../20261031-pace-native-copy-ai-stop-diagnostic/`의 기존 pace30 교사 child 종료0, 728결정·camera wait miss0, 카메라 911프레임 정상 종료, moving-brake 원본 receipt 표본85개, fit/학생 actuator/GUI0. AI inactive host 관측에서 stop 요청까지0.458ms. `.../20261031-pace-native-copy-ai-stop-raw-audit/`는 원본 카메라 911프레임 aid off, raw MMF 및 이미지/몸 packing 차이0, 결정 간 최대44.90ms·80ms 초과0을 확인했다. `.../20261031-pace-native-copy-ai-stop-native-audit/`에서 copy15,487개·고유 packet10,697개, 728장면 중214개에 서로 다른 선행 packet, 50ms 초과4시야를 확인했다. copy 간격 중앙/95/최대2.50/2.81/33.59ms; host 수신이지 물리 발생/노출 시간이 아니다. 원본 raw 감사의 continuous_40ms_training_admissible=false와 label 미생성을 그대로 유지한다. 이 단일 진단은 종료 개선의 반복성·TRAIN/DEV·생리·학생 주행·제동/변속 학습 또는 랩타임을 증명하지 않는다. 종료 후 같은 PID12432/generation에서 paused·AI/camera/probe off, background/vJoy released, 정지/브레이크를 확인했다(당시). 다음은 독립 라벨/authority 및 사전 지정 TRAIN/DEV의 역할·구간을 검증하고 학습 및 인과 실험을 진행한다.


### 사전 등록 — pace 원본의 native flag·제어값 상관 감사 (2026-10-31)
정상 종료 diagnostic의 전체 raw camera 감사 및 420-byte 고빈도 copy hash·packet 순서를 먼저 고정하고, 각 decision의 308-byte native event와 같은 packet·동일 body bytes인 이전 고빈도 copy를 찾는다. 그 copy의 AI flag·host receipt와 원 event의 조향/가감속 및 moving brake 분포를 계산하되 서로 다른 copy에서 얻은 정보는 원 event와 same-copy 권한으로 바꾸지 않는다. 일치 실패·시간 조건 실패를 분리하고 actor camera와 교사 제어의 물리적 동기·독점 제어·학습 자격은 부여하지 않는다. diagnostic 역할과 40ms training false를 유지하며 fit/학생 제어0이다.


### 2026-10-31 pace 원본의 native flag·제어값 상관 감사
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-native-authority-correlation/`에서 원본 728결정 각각의 308-byte body를 고빈도 420-byte 최초 packet 이력과 대조했다. 421개는 동일 packet·동일 body bytes·이전 receipt의 AI true, 305개는 첫 고빈도 copy가 이벤트 수신보다 나중, 2개는 해당 packet 자체가 없었다. moving brake(speed>1km/h, brake>0.05) 61결정 중 35개는 이전 일치 AI true, 26개는 copy가 나중이었다. phase는 accelerate60 292, accelerate100 242, stop0 194다. 이 상관은 별개의 read를 같은 copy의 권한으로 바꾸지 않으며 영상-교사 물리 동기, 독점 제어 및 fit을 입증하지 않는다. diagnostic 유지·optimizer0.

이를 반영해 `record_pace_task.py`가 사전 계획의 `same_copy_native_authority: true`에 한해 기존 420-byte same-copy reader를 선택하게 연결하고 필수 source hash를 검사한다. 고빈도 copy 옵션과 독립이며 기본값은 기존 동작이다. 관련 46테스트 통과. 아직 이 조합의 새 live 수집이나 authority 감사는 하지 않았다. 현 paused spline 약0.02020은 기존 TRAIN 시작 0.275–0.44 밖이다. 다음은 현재 generation에 맞는 transit을 별도 지정하고 전체 이력을 보존한 뒤 TRAIN 구간에서 same-copy·고빈도 수집, 원본/권한/시간 검증 및 분리된 DEV 실험으로 진행한다.


### 사전 등록 — 현재 엔진의 TRAIN 구간까지 transit (2026-10-31)
새 read-only baseline에서 PID12432/generation `15533904ULL:0.0013473`, paused spline0.020201, background·vJoy free·정지/브레이크를 확인했다. 과거 axis config의 profile SHA와 현재 controls.ini가 달라 최초 baseline은 실패했다. 기존 `20261031-live-controls-equivalence`의 44개 INI 섹션/필드 동일 및 현 profile SHA를 가리키는 기존 config를 사용해 read-only baseline을 재검증했으며 runtime axis response 확인이나 학생 제어 자격은 아니다. 원본 controls.ini와 구 설정은 수정하지 않았다.

`configs/flight-pace-transit-current-20261031.json`은 현재 baseline·source hash·동일 60km/h 3/30초 한정 API·최대8회 및 기존 역할/위치 검증을 고정한다. 각 단계는 transit·fit false이고 종료와 정지 상태를 검증한 후 다음 단계만 실행한다. TRAIN 시작 0.275–0.44에서 활성화 없이 중지한다. 우회/자동 재시도/역할 변경 없고 관측 불확실성·실패는 보존한다. 이 transit은 학습 데이터나 학생 주행 검증이 아니다.


### 사전 등록 — 같은 copy 권한·고빈도 native의 단일 TRAIN 후보 (2026-10-31)
`20261031-pace-transit-current/`의 세 차례 transit은 각각 종료0이고 spline0.02020→0.10604→0.19021→0.27670에서 일시정지했으며 transit 역할·fit false를 유지한다. 기존 TRAIN 시작0.275–0.44에 해당하므로 새 `configs/flight-pace-train-native-copy-20261031.json`으로 전체 pace30 기록 한 건을 TRAIN 후보로 사전 지정한다. 고빈도 원본 copy, same-copy native AI/body, AI 종료 관측 후 camera stop을 함께 사용하며 이전 진단/이동을 TRAIN에 재분류하지 않는다. 원본 camera·authority·제동·행동 표적·물리 시간·전체 경로를 각각 감사하기 전 fit false; 실패 시 기록하고 자동 재시도하지 않는다. 분리 DEV는 별도 수집한다. 학생 actuator·GUI0.


### 2026-10-31 현 엔진 transit 및 TRAIN 후보 원본 감사
기존 controls.ini와 등록 SHA 차이로 첫 read-only baseline은 실패했다. 이미 보존된 `20261031-live-controls-equivalence/`가 이전/현재 44개 INI 섹션과 항목이 같음을 확인했으며, 현재 SHA를 명시한 기존 config로 재검증했다. runtime axis 반응 자격은 별도이며 controls는 수정하지 않았다. 새 v2 transit은 계획에 고정한 PID/generation·baseline을 매 단계 대조하고 기존 3/30초60km/h 한도·역할·구간을 유지한다. 관련 테스트4개 통과. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-transit-current/`의 세 30초 단계는 모두 종료0, spline0.02020→0.10604→0.19021→0.27670에서 paused했고 모든 단계는 transit/fit false다.

사전 등록 단일 TRAIN 후보 `.../20261031-pace-train-native-copy-candidate-00/`는 child 종료0, route 원본1502개 전체가 지정 TRAIN 구간 조건을 통과하고 728결정·camera miss0였다. `...-raw-audit/`: 카메라 915프레임 aid off, 이미지/몸 재구성 오차0, 결정 간 최대43.67ms·80ms 초과0. `...-authority-audit/`: 같은 native 420-byte copy의 AI true 728/728·제한된 native-body 권한 조건728/728, 원본 receipt age 최대0.448ms. 별도 camera/교사 물리 동기·독점 제어·최적 교사·원자성은 검증하지 못했다. `...-native-audit/`: copy15,556·고유 packet10,703, 시야별 prior packet 차이226장면, prior 50ms 초과4시야 보존, copy 간격 중앙/95/최대2.50/2.81/31.86ms. `...-brake-audit/`: 이동 중 제동59결정, 전부 same-copy AI true; 인위적 pace stop 명령이며 자연 제동 label은 아니다.

이 후보는 TRAIN 역할을 유지하되 raw 감사의 continuous_40ms_training_admissible=false, label 미생성, fit false를 그대로 유지한다. 지도 신호의 시간/행동 표현 가능성·실제 영상과 teacher 권한 동기 및 독립 DEV는 미검증이고 학습/학생 제어0이다. 종료 후 PID12432 같은 generation paused·AI/camera/probe off·정지/브레이크 및 background/vJoy released를 확인했다(당시). 다음은 전체 후보의 원래 packet/mask를 보존해 시간·행동 label 자격을 검증하고 분리된 DEV 후보를 수집한 후 시각·비행 회로 인과 실험/학습을 진행한다.


### 사전 등록 — TRAIN 후보의 고빈도 네 시야별 과거 몸 입력 재구성 (2026-10-31)
원래 TRAIN 후보의 전체 728결정, 독립 raw camera 및 native copy 감사 hash를 고정한다. 각 시야 render host 하한보다 엄격히 앞선 가장 최근 최초 native receipt를 찾고 camera receipt 기준 50ms를 초과하면 결측으로 둔다. 각 시야별 기존 21채널 encoder 상태와 event의 원래 현재 몸 상태를 유지하며 packet·시각·값·valid mask를 기록한다. 원본을 변경하거나 DEV/진단을 TRAIN으로 바꾸지 않는다. 이는 학습용 후보 입력의 재구성이지 물리 발생/노출 시각, actor 자격 또는 label/fit 자격이 아니다. optimizer/차량 제어0.


### 2026-10-31 고빈도 TRAIN 후보의 시야별 몸 입력 재구성
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-train-high-rate-per-view-body/`는 raw camera/native 감사에 묶인 원본 728결정의 각 시야에 대해 최초 receipt가 해당 view render host 하한보다 엄격히 앞선 마지막 native packet을 선택한다. 21채널 body 값·valid mask 및 각 수신 시각, 원본 현재 body·결정 시각을 보존했다. 시야별 matched는 각727개, 동일 장면에서 시야 prior가 다른 경우226, 50ms 초과 결측 총4시야다. 원본을 수정하지 않았다. `.../20261031-pace-train-high-rate-per-view-body-independent-audit/`는 별도 NumPy searchsorted와 원본 420-byte 복원, encoder 재계산으로 2,908개 유효 view 및 4개 결측, 현재 몸 값·mask·시각 일치를 확인했다.

이 결과는 offline 호스트 수신 기준의 후보 입력으로 fit/optimizer0이며, 실제 물리 발생·카메라 노출 동기, 독립 DEV, 조향/가감속 행동 표적의 시간 및 표현 자격은 여전히 미검증이다. 원본 raw audit의 continuous_40ms_training_admissible=false를 바꾸지 않는다. 다음은 same-copy teacher 활동 구간과 원본 event/camera 시간 경계, 행동 표적의 보정/표현 가능성을 감사하고 분리 DEV에서 시각 및 몸 회로 기여를 비교한다.


### 사전 등록 — 분리된 DEV 구간으로 transit (2026-10-31)
원본 TRAIN 후보의 종료 후 fresh idle과 spline0.35783을 확인했다. 기존 보호 DEV 구간0.67–0.84의 시작0.67–0.72까지 동일 60km/h 3/30초 유한 transit(최대8단계)을 실행한다. 각 단계 full original 보존, transcript/권한/fit false 및 학생/GUI0. 기존 TRAIN/DEV/evaluation 이력은 변경하지 않으며 목표 초과 또는 원본 오류 시 중단하고 재시도하지 않는다. 도착 후 별도 사전 지정된 DEV 후보의 전체 구간과 원본 자격을 감사한다.


### 2026-10-31 분리 DEV transit 단계 예산 소진
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-dev-transit-current/`의 전체 상태는 `failed`, 사유는 사전 지정 8단계 예산 소진이다. 각 child는 종료0이며 spline0.35783→0.44469→0.52451→0.61142→0.61859→0.62571→0.63273→0.63972→0.64659에서 멈췄다. 3초 단계의 실제 진행은 약0.0069–0.0072여서 시작0.67에 도달하지 못했다. 모든 이력은 transit/fit false로 보존했고 DEV 후보는 수집하지 않았다. 이미 정한 '새 batch 자동 시작 금지'를 준수해 추가 동작은 하지 않았다. 종료 후 같은 generation PID12432의 paused·AI/camera/probe off·정지/브레이크 및 background/vJoy released를 확인했다(당시). 계획 추정이 과도하게 보수적이었던 문제는 측정값에 근거해 별도 검토해야 하며 실패를 성공으로 바꾸지 않는다. 관련 기존 및 DEV target 테스트5개 통과. 다음은 실패 증거를 이용해 별도의 bounded 이동 계획과 보호 DEV 전체 구간·독립성 계약을 검토한 뒤 DEV 후보를 사전 지정한다.


### 사전 등록 — 단계 소진 후 검토한 DEV 접근 (2026-10-31)
이전 실패 원본과 8개 child 종료0·최종0.64658785 및 fresh same-generation idle을 재확인했다. 다섯 3초 구간의 실제 이동0.00687–0.00718을 반영해 별도 plan `configs/flight-dev-followup-transit-20261031.json`을 작성한다. 최대5개의 기존 3초60km/h transit, 매 단계 동일 PID/generation·paused 위치·source hash를 재검증하고0.67–0.72 도착 시 즉시 중단한다. 목표 초과, 실패, 불확실 또는 예산 소진은 그대로 중단·보존하며 기존 실패를 재분류하지 않는다. 이 작업 전체는 transit/fit false이고 DEV 기록은 도착 후 별도 등록한다.


### 2026-10-31 DEV 후속 transit 시작 전 카메라 소유권 불일치
사전 등록 후속 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-dev-followup-transit/`의 첫 child는 failed, `control/result.json`에는 lifecycle 요청0·actuator0, 이유는 카메라 소유권 확인 실패다. `control/before.json`은 camera_active false/요청 ID 부재였으나 recorder는 이미 네 시야 영상을 수신해 해당 camera request ID를 넘겼다. 이후 recorder는 정해진 대기시간 종료·원본 archive를 남겼고 이 child는 0결정이며 차량 이동/학습에 사용하지 않는다. camera frame 준비와 세션 heartbeat 반영 사이의 관측 불일치가 확인됐지만 정확한 지연 원인은 확정하지 않는다. 원본 실패를 유지하고 재시도하지 않았다.

`record_ac_engine_ai_probe.py`는 이제 제어 child를 한 번만 시작하기 전에 동일 generation·fresh paused AI off의 heartbeat에서 정확히 자기 camera request ID가 active로 표시될 때까지 최대2초 기다린다. 다른 camera·세대·오류·불명확 상태는 실패시키고 기존 제어 guard/timeout을 완화하지 않는다. ownership 확인 뒤 기존 lease 잔여시간도 다시 확인한다. test fixture도 새로운 필수 사전 조건을 반영했고 39테스트 통과. 새 경로의 live 검증은 아직 없다. 종료 후 PID12432 같은 generation paused·AI/camera/probe off·정지/브레이크·background/vJoy released 확인(당시), DEV 미수집·fit/학생제어0이다. 다음은 소스 hash와 새로운 사전 등록 조건으로 이 수정의 live 검증 및 DEV 수집을 별도 진행한다.


### 사전 등록 — camera ownership 확인 수정 후 DEV transit (2026-10-31)
직전 child가 카메라 heartbeat 소유권 불일치로 control 요청0에서 실패한 원본과 수정된 수집기 source를 고정한다. fresh idle spline0.64658785, PID12432 같은 generation을 확인했다. 기존 3초60km/h transit을 최대5회, 각 단계 완전 종료 및 paused 확인 후에만 시행하고 0.67–0.72에서 중지한다. 자기 camera ID heartbeat가 확인되지 않거나 오류/초과 시 중단하며 이전 실패를 보존한다. 원본은 전부 transit, fit/학생제어/GUI0이다.


### 사전 등록 — 보호 DEV 구간의 단일 pace 후보 (2026-10-31)
직전 camera heartbeat 수정 후 `20261031-dev-owned-camera-transit/` 네 3초 단계가 종료0이고0.64659→0.67440에서 paused하여 protected development 시작0.67–0.72에 도달했다. 이전 두 실패는 유지하며 모든 이동은 transit/fit false다. `configs/flight-pace-development-native-copy-20261031.json`은 pace30 전체를 기존 보호 DEV0.67–0.84에서 한 번 수집하도록 역할을 development로 사전 지정한다. 기존 알려진 DEV 지형이므로 전역적으로 처음 보는 지형이라는 주장은 하지 않는다. native 고빈도 및 같은 copy AI/body, 카메라 소유권 사전 확인과 AI 종료 후 camera stop, 전체 route 검사, 원본·authority·시간 감사 전 fit false다. 실패 시 자르거나 역할을 바꾸지 않고 보존한다. 학생 actuator/GUI0.

첫 DEV plan 실행은 사전 입력 hash 검사에서 변경된 transit 테스트 파일의 오래된 hash로 중단됐고 run 생성·카메라/제어 요청0이었다. 원본 plan을 보존하고 해당 실제 테스트 hash만 고정한 v2 plan으로 첫 실제 후보 수집을 등록한다.


### 2026-10-31 카메라 ownership 수정 후 DEV 후보 수집
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-dev-owned-camera-transit/`의 3초4단계가 모두 종료0, spline0.64659→0.67440에서 정지했다. 수정된 camera ownership 확인을 live 통과했으나 이전 실패를 재분류하지 않는다. 첫 DEV plan은 오래된 transit test hash 때문에 run/카메라/제어 실행 전 중단; 현재 테스트 hash만 고정한 v2로 후보 하나를 실행했다.

`.../20261031-pace-development-native-copy-candidate-00/`: 기존 보호 DEV0.67–0.84에서 시작0.67440, 전체 route 원본1484 표본 통과, child 종료0·729결정·camera miss0, 역할 development·fit false. 알려진 DEV 지형이며 전역적으로 미관측 지형이라는 주장은 없다. `...-raw-audit/`: 729결정 raw 카메라/body packing 차이0, aid off, 최대 간격43.35ms·80ms 초과0, continuous_40ms_training_admissible=false. `...-authority-audit/`: native 같은 copy AI true 및 제한된 자격729/729, 물리 exposure/독점 권한 자격 아님. `...-brake-audit/`: 이동 제동60결정 모두 AI true, scripted stop이지 자연 제동 label 아님. `...-native-audit/`: native copy15,445·고유 packet10,683·226장면의 상이 prior·4시야 stale, copy 간격 중앙/95/최대2.50/2.82/26.97ms. `.../20261031-pace-development-high-rate-per-view-body/`와 `...-independent-audit/`에서 네 시야 body 2,912 match·4 결측, 원본 값/mask/receipt 일치를 확인했다. 호스트 시간만 확인했으며 물리 발생·노출, 행동 label/학습 적합성 및 학생 폐루프/랩타임은 미검증. optimizer/학생제어/GUI0. 종료 후 같은 PID12432/generation paused·AI/camera/probe off·정지/브레이크·background/vJoy released 확인(당시). 다음은 TRAIN/DEV의 원본 시간·행동 표적 및 학습 가능성을 별도 감사하고 시각/비행 회로 인과 학습을 진행한다.


### 사전 등록 — TRAIN/DEV 원본 교사 활동 시간·조작값 감사 (2026-10-31)
사전 할당된 TRAIN/DEV 원본, raw camera 및 same-copy authority 감사 hash를 고정한다. 각 원본 카메라 metadata의 CSP render call 범위, 시작/끝 API call 범위 및 원래 clock-final ping에서 복원한 native host receipt bounds를 사용해 기존 보수적 activity/phase 검사를 두 이력 전체에 적용한다. 원래 몸/AI 같은 copy와 held task를 사용하고 애매한 결정은 남기되 별도 이유를 기록한다. native steering/gas/brake 원래 분포와 .02 pedal floor 아래 값, 동시 페달을 집계한다. 조향의 움직이는 차량 전달 보정과 물리 exposure/선생 독점 제어는 미검증으로 남긴다. 이 평가는 제어 label 생성·fit 자격·물리 동기 증명이 아니며 TRAIN/DEV 역할 보존, optimizer/학생 제어0이다.


### 2026-10-31 TRAIN/DEV 시간·원본 조작값 진단
`scripts/audit_pace_candidate_temporal_axes.py`는 각각 원본 camera metadata hash·same-copy420 bytes 및 원래 CSP clock-final ping과 API 시작/끝 bounds로 기존 보수적 activity/phase 판정을 적용했다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-train-temporal-axes-audit/`: TRAIN 728결정 중 시간 사유 없음725, 활동 창 외2, phase 경계 불명확3(사유 중복 가능). `.../20261031-pace-development-temporal-axes-audit/`: DEV729 중 사유 없음723, 활동 창 외3, phase 경계 불명확6(중복 가능). 이 수는 saved CSP render call과 host receipt의 투영 범위에 한정되며 물리 sensor 생성·실제 exposure 또는 독점 teacher 제어를 입증하지 않는다. 각 결정의 사유·원본 native axes는 별도 기록하며 사건 삭제·label 생성0이다.

원본 native steering 절대 중앙/95/최대 TRAIN0.0133/0.8082/0.8082, DEV0.0410/0.6522/0.6529; pedal 값의 (0,0.02) 구간은 각 축에서 두 이력 모두0, gas/brake 동시 양수는 각3결정이었다. 추가 `.../20261031-pace-train-axis-thresholds-audit/`, `.../20261031-pace-development-axis-thresholds-audit/`: raw steering 절대값0.5 초과 TRAIN149/728·DEV128/729. 이는 native 원시 단위의 기술통계이지 보정된 actuator 한계 또는 실현가능 판정이 아니다. 기존 정지 보정은 움직이는 차량 조향 전달을 검증하지 않았으며 yaw 교사 기준도 없다. 관련 eligibility/native tests101개 통과. 두 이력 모두 원래 fit false·continuous_40ms_training_admissible=false, optimizer/학생 제어0. 다음은 moving 조향 보정 및 teacher 독점성·행동 표적 자격을 해결하면서 시각·비행 부품의 생리/인과 기능을 별도 검증한다.


### 사전 등록 — 새 TRAIN/DEV 원본 네 시야 FlyVis 연속 재생 (2026-10-31)
원본 raw camera·몸/same-copy 감사가 완료된 사전 역할 TRAIN과 protected DEV 각 전체 episode를 원래 결정시각·프레임·결측 그대로 순차 재생한다. 먼저 legacy16과 navigation21 몸 encoder의 전체 일치를 검증하고, 동일 카메라 기하와 고정 pretrained FlyVis 네 독립 상태를 이전 입력 zero-order hold·최대5ms 내부 간격으로 사용한다. 원본 frame/RGB hash 및 graph/checkpoint를 확인하고 episode 내 state reset 없이 전체 retina·운동 세포 응답을 기록한다. 역할 변경·gradient/optimizer·차량 제어0; 저장된 감각 반응만으로 물리/노출 동기·생리 기능·학습/주행을 주장하지 않는다.


### 2026-10-31 TRAIN/DEV 원본 네 시야 FlyVis 전체 연속 재생
원본 각 episode의 legacy16 및 navigation21 body replay에서 전체728/729결정을 보존하고 원래 legacy16 값을 exact 비교했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-train-flyvis-replay/`와 `.../20261031-pace-development-flyvis-replay/`는 원본 frame/RGB hash, 동일 exterior camera 기하, 네 독립 pretrained FlyVis state 및 이전 retina hold로 각각728/729결정, 6543/6552 내부 tick을 재생했다. episode 중 reset0, 미래 frame 소급 사용0, 가중치 변경0이다.

각각 `.../20261031-pace-train-flyvis-full-audit/`와 `.../20261031-pace-development-flyvis-full-audit/`가 원래 전체 시각·held input을 독립 순서의 batch로 다시 실행했다. TRAIN central/motion 최대 차5.96e-7/4.34e-7, DEV 5.96e-7/4.77e-7, 전체 결정과 tick 일치·state reset0; 네 시야 permutation 및 상태 분할 검증도 통과했다. 결과는 원본 기록 기반 계산 재현이며 optical resampling은 공학적이고 T4d 기존 contrast 불일치, 물리 발생/노출 동기, 생리 기능, 비행 회로 결합과 독립 주행은 미검증이다. DEV는 알려진 보호 지형, fit/optimizer/학생 제어0이며 원본 continuous_40ms_training_admissible=false 유지. 다음은 두 역할의 시야별 몸 맥락과 시각 회로를 비행 유래 회로에 연속 결합하고 공학적 개입 반응 및 학습 가능한 표적의 자격을 검증한다.


### 사전 등록 — TRAIN/DEV 네 시야·몸 감각의 비행 회로 연속 개입 (2026-10-31)
독립 raw/시야별 몸 감사와 전체 FlyVis chronological audit을 통과한 원본 TRAIN728·DEV729결정을 각 episode 처음부터 그대로 사용한다. 이전에 사용한 고정 비행 유래 회로 및 opponent temporal 경계 checkpoint, 원본 네 시야 prior mask·현재 몸, FlyVis 운동 state와 15km/h 고정 공학 진단 목표를 사용한다. 동일 초기화의 독립 recurrent state로 intact, 현재 pitch/roll 0, 현재 타이어 하중0, FlyVis scene feature0을 각각 전체 연속 재생해 조향/전진 intent 차이를 측정한다. 개입값0은 분포 밖일 수 있고 비행 생리, 원래 pace 목표 또는 제동 명령이 아니다. per-view yaw 경계는 기존 zero 초기화 상태로 학습되지 않아 수신 차이의 인과 효과를 이 시험이 입증하지 못한다. optimizer/차량 제어0, 물리 발생/노출 동기·주행/랩타임 주장0.


### 2026-10-31 TRAIN/DEV 원본 시각·몸 비행 회로 개입
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-train-multisensory-interventions/`와 `.../20261031-pace-development-multisensory-interventions/`에서 전체728/729결정의 네 시야 FlyVis state, 현재 body21·원본 view별 prior/mask를 고정 비행 유래 회로와 opponent temporal 경계에 순서대로 넣었다. 네 조건은 독립 recurrent state이며 episode 중 reset0. 고정 목표15km/h는 원래 pace 목표가 아니고, per-view yaw projection은 학습되지 않은 zero이므로 시야별 yaw 시간차 기여를 증명하지 않는다.

intact 대비 조향/전진 intent 평균 절대차는 TRAIN에서 현재 기울기0 시 0.0000433/0.004863, 현재 하중0 시 0.0000783/0.001776, scene feature0 시 0.12813/0.30188이다. DEV는 기울기0 0.0001451/0.003008, 하중0 0.0001063/0.003799, scene feature0 0.14791/0.81331이다. 0 개입은 분포 밖일 수 있고 이 두 채널은 브레이크/변속 명령이나 실제 성능이 아니다. 각 `...-multisensory-interventions-audit/`는 전 원본 event ID/시간과 저장 통계 재계산을 확인했으며 모델을 독립 재실행한 것은 아니다. 감도 확인이지 생리 기능·유효 완주/라인/랩타임·개별 시야별 yaw 기여나 학습 효과의 증거가 아니다. 고정 모델·fit/optimizer/차량 제어0, 물리 발생/노출 동기 미검증. 다음은 동일 입력·독립 DEV에서 시야별 시각/몸 경계의 학습 가능성과 결측·방향/시간 인과 기여 및 조작 표적 자격을 검증한다.


### 사전 등록 — 과거 학습된 공통 yaw 경계의 시야별 전이 및 인과 시험 (2026-10-31)
과거 TRAIN00 한 이력에서12회 업데이트된 `20261031-historical-train00-temporal-context-onepass/` 공학 common-yaw projection을 원본 hash·source lineage 보존해 per-view 동일 shape 경계에 복사한다. 네 시야 yaw가 같은 경우 양 경계의 출력 동등성 및 yaw mask 결측 시 0을 unit test로 검증했다. 새 TRAIN/DEV는 원래 역할·전체 시간/몸 mask를 유지하고 학습 또는 label 접근 없이 고정 과거 checkpoint를 적용한다. 동일 15km/h 고정 목표로 independent recurrent state의 원래 per-view yaw, projection0, view yaw0, 전 시야 history 결측을 비교한다. 과거 경계는 동일 TRAIN 내 미미한 개선만 확인됐고 독립 검증 전·물리 시각/생리 불명이며 적용은 명시적 실험일 뿐 배포 아님. optimizer/차량 제어0.


### 2026-10-31 과거 공통 yaw 학습 경계의 시야별 전이 시험
과거 TRAIN00 한 이력으로12 업데이트된 common-yaw 경계 `20261031-historical-train00-temporal-context-onepass/`의 원래 projection weight를 동일 shape per-view 경계에 명시적으로 복사했다. 동일 yaw에서는 기존/전이 출력이 일치하고 yaw valid가 false이면 해당 입력이 0이 되는 unit test를 통과했다. 별도 파라미터 학습은 하지 않았다. 원본 TRAIN728과 알려진 DEV729결정을 네 시야 FlyVis·prior body 원본 순서로 고정 비행 회로에 넣고, 동일 시각의 독립 state로 학습 projection 유지 vs projection0/yaw0/시야 prior 결측을 비교했다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pace-train-transferred-context-interventions/`와 `.../20261031-pace-development-transferred-context-interventions/`: projection 제거의 조향/전진 intent 평균 절대차 TRAIN 2.55e-6/2.60e-8, 최대4.45e-5/1.15e-6; DEV 평균1.40e-5/7.70e-8, 최대1.97e-4/9.24e-7. yaw0 및 모든 과거 context 결측은 이 projection 구조에서 동일한 제거 결과였다. 각 `...-audit/`는 원본 event identity·수치 통계를 대조했으나 모델 재실행은 아니다. 이는 저속 고정 목표와 두 이력에서의 작은 공학적 출력 감도이며 고유 시야 수신 차이의 효과, 운동시각 생리 기능 또는 실제 주행/랩타임 향상을 입증하지 않는다. 과거 checkpoint의 학습/근거 한계를 그대로 유지하고 새 TRAIN/DEV label 개봉·optimizer·차량 제어0, 물리 exposure 동기 미검증이다. 다음은 실제 속도0–100km/h 목표·유효 조작 범위와 별도의 독립 자격을 갖춘 학습 경로를 마련하고 per-view vs common 인과 효과를 검증한다.


### 정정 및 사전 등록 — 기존 고속 물리 운동 경로의 새 시각 입력 (2026-10-31)
이전 요약의 30km/h 제한은 구 `flight.speed_goal.v1` 경로에만 해당한다. 이미 존재하는 `flight_maneuver.py`의 물리 단위 yaw(rad/s)·속도(m/s)와 `racing_goal.py`의 외부 최고속도 계약은 100km/h를 표현한다. 과거 checkpoint의 30km/h 의미를 바꾸지 않으며 기존 물리 경로 역시 시각 판단 정확도·제동·생리·폐루프 자격은 없다. 기존 `visual_preview_speed.py`는 학습된 20m chord 예측으로 yaw/속도를 만드는 공학 가설이다.

새 TRAIN/DEV의 원본 FlyVis 전체를 같은 checkpoint·원래 episode clocks로 재생하며 기존 8개 T4/T5에 추가 L1/L2/Mi1/Tm3/Tm1/Tm9의 전체 시야 column을 저장한다. 원래 선행 replay의 retina·motion·central을 2e-6 이내로 비교해 상태/광학 처리 변화를 감지한다. 각 역할과 결측·원본 hash를 보존하고 데이터 fitting, 학생 제어 및 actor 배포0이다. 추가 세포 실행은 생리 재현·고속 주행 기능 증명이 아니다.


### 2026-11-01 사용자 재확인: 하나의 통합 감각운동 회로
사용자는 별도 운동회로가 아니라 하나의 회로로 가는 목표를 재확인했다. 현 구현은 구30km/h 시각-몸 비행 회로와 물리 단위 고속 경로가 같은 비행 연결망 유래임에도 개별 실험 경로로 나뉜 상태다. 고속 경로는 FlyVis를 공학적 별도 preview 예측기로 변환해 같은 운동 연결망에 넣고, 이전 시야별 몸 맥락 경계를 아직 함께 통합하지 않는다. 이를 완성된 하나의 end-to-end 생물 회로라고 부르지 않는다. 다음 구현은 원래 FlyVis의 전후좌우 운동 시각·몸 및 고속 목표→기존 비행 연결망→차량 조작이 하나의 연속 상태로 이어지도록 연결하고, 각 공학 경계와 결측을 분명히 하여 인과·폐루프를 검증한다. 기존 경로/실험 원본은 보존한다.

직전 14종 재생의 process handle이 사라졌고 result 없이 manifest만 남아 종료 상태 불명으로 보존했다. 이후 별도 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pace-train-flyvis-14types-replay/`, `.../20261101-pace-development-flyvis-14types-replay/`에서 TRAIN728·DEV729 전 결정을 재생했다. 각 4시야·6543/6552 tick·state reset0, 기존 retina 최대 차0, motion 최대 차4.06e-7/4.17e-7, central 최대 차5.96e-7/7.15e-7, 학습/제어0. 기존 8운동 세포 외 L1/L2/Mi1/Tm3/Tm1/Tm9의 시야별 원본 response를 보존했다. 이는 감각 재생이지 통합·생리·주행 검증은 아니다.


### 사전 등록 — 단일 비행 그래프의 고속 감각 입력 원본 재생 (2026-11-01)
동일 화학 graph sparse indices/weights, body/vision/steering/goal ports 및 보호 body 경계의 동일성을 검사한 뒤 하나의 physical flight graph state에 기존 FlyVis 네 시야 투영·opponent 시간 경계·원본 시야별 body prior·현재 몸 및 물리 단위 외부 속도 과제를 함께 연결한다. source graph/weights와 이전 경로는 보존하고 새로운 body feedback은 zero 초기화 공학 경계다. TRAIN/DEV 원본의 외부 지시 속도 0/60/100km/h를 사용하고 yaw 명령은 0으로 명시한다. 이는 영상으로 yaw를 예측한 시험이나 실제 배우가 아니며 world pose/map/teacher steering을 actor에 주지 않는다. episode 전체 원래 clocks와 held 입력 및 결측을 유지, 실제 조작·optimizer0. 시각 motion/yaw 경계의 경로와 하나의 state에 대한 감도·연속성을 확인하되 생리, 실제 제동/완주 및 학습 효과 주장0이다.


### 2026-11-01 하나의 비행 graph에 시각·몸·물리 속도 결합
`src/flybrain_drive/unified_flight_sensorimotor.py`는 기존 v5 physical graph와 v7 시각 경계의 화학 sparse graph 값/방향, 몸·시각·조향·속도 port, 시간 상수 및 보호 body weight가 일치할 때만 연결한다. 실제 두 기존 checkpoint에서 graph/ports/body exact를 확인했다. 하나의 physical flight graph state에 네 시야 FlyVis 시각 포트, opponent 시간 traces, 각 시야 body prior의 기존 공학 projection, 현재 body feedback 및 물리 단위 command를 연결한다. 입력 보조 경계가 실측 생리 회로라는 뜻은 아니다. 타이어/기울기용 새 body projection은 zero 초기화·미학습, 상대풍은 아직 이 경로에 연결되지 않았다. command의 actor 출처를 모듈만으로 증명할 수 없으며 호출 측이 영상·허용 몸 감각 유래인지 검증해야 한다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-flight-train-replay-v2/`, `.../20261101-unified-flight-development-replay-v2/`: 원본 TRAIN728·DEV729 전 결정, 외부 목표0/60/100km/h 및 명시적 yaw0으로 각 하나의 state를 연속 재생했다. 시각 feature0 개입에 조향/전진 intent 평균 차 TRAIN0.003848/0.0000181, DEV0.005987/0.0000179; 0은 분포 밖일 수 있다. 이는 도로를 보고 yaw/속도를 자율 예측한 것이 아니며 원래 외부 speed task와 yaw0으로 통합 경로를 시험한 것이다. GPU source replay v2는 명령을 상태 갱신 전에 검증한 버전이고, 이전 v1은 별도 원본·통계 감사만 완료했다. v2는 v1과 각 `intents.npz` SHA가 동일하다. 이후 오류 발생 시 state/traces를 부분 갱신하지 않도록 commit을 원자적으로 하는 변경을 추가했고 synthetic fixture 포함 관련15테스트 통과; 이 마지막 변경은 아직 실제 원본 재생하지 않았다. 모델 독립 재실행·physical sensor/exposure, 생리 및 폐루프 주행/라인/랩타임 검증 전이다. 모든 실험 fit/optimizer/차량 제어0. 다음은 current source로 원본 재현을 재검증하고 학습된 시각 예측을 동일 graph의 physical command에 연결해 교사/map 없이 방향·속도를 결정하도록 발전시킨다.


### 사전 등록 — 시각 예측→단일 비행 graph의 100km/h 상한 진단 (2026-11-01)
기존 TRAIN05/06으로만 학습한 공학 14종 시각·몸 preview fold를 해시 고정해 새 TRAIN/known DEV의 네 시야·유효 body에서 방향 및 속도 명령을 산출한다. 고정 외부 상한 100km/h만 제공하고 원본 teacher 조향/속도, map/pose/정답 라인을 명령으로 사용하지 않는다. 사전 존재하는 curvature→yaw/speed 변환은 가설이며 정확도·생리·운전 자격이 없다. 원본 clock·mask를 유지하고 T4/T5 motion 평균 제거 개입을 별도 기록한다. 명령은 같은 physical flight graph state에만 전달하며 고정 모델/no optimizer/차량 제어0. 전체 독립 DEV·주행 향상이나 신경 구현 검증으로 해석하지 않는다.


### 2026-11-01 상태 보호 원본 재현과 시각 예측→단일 graph
최종 상태 commit 보호 변경 후 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-flight-train-replay-v3/`, `.../20261101-unified-flight-development-replay-v3/`는 각각728/729결정을 재생했고 이전 v2 `intents.npz` SHA와 각각 완전히 일치했다. 원래 clock·zero state reset 및 외부 speed0/60/100·yaw0 조건은 동일하다. 이것은 보존된 결과와 같은 source 이후 재실행이지 독립 모델 구현이 아니다.

고정 TRAIN05/06 출처의 14종 시각/몸 preview는 입력 검증 후 계산됐으나 첫 TRAIN/DEV 시도는 결과 JSON의 numpy.float32 직렬화 오류로 실패해 원본을 보존했다. 형식만 수정한 `.../20261101-pace-train-visual-preview-commands-v2/`, `.../20261101-pace-development-visual-preview-commands-v2/`는 각각728/729결정의 전후좌우 FlyVis·몸에서 회전/속도를 산출했다. 외부 ceiling100km/h, 미리 정한 gain100m; 원래 teacher 속도/조향·map/pose를 predictor 명령으로 넣지 않았다. 예상 속도 범위 TRAIN61.91–100.00, DEV54.28–99.88km/h. 이는 모델 예측이지 적정 속도나 제동 증거가 아니다.

`.../20261101-unified-flight-train-preview-replay/`와 `.../20261101-unified-flight-development-preview-replay/`는 이 예측과 원본 네 시야·몸 prior/current를 동일 physical flight graph의 단일 recurrent state에 전달했다. preview에 들어가는 motion 성분만 평균으로 대체하면 조향/전진 intent 평균차 TRAIN0.00696/0.05584, DEV0.00345/0.04034. graph의 별도 시각 포트는 유지한 개입이며 반사실 분포 밖 가능. 각 `...-preview-replay-audit/`는 전 원래 clock·출력 통계만 재계산했고 모델 독립 재실행은 아니다. 모형은 engineering readout이고 원래 fly 생리나 독립 학습/자율 배우 자격이 아니다. 기존 30kmh 제한을 고속 모형 전체의 제한으로 표시한 것은 정정한다. 물리 sensor/exposure, 모델 예측 정확도, 상대풍·body adapter 학습, 변속·셋업, 폐루프 주행/랩타임 미검증. fit/optimizer/차량 제어0, known DEV 역할 보존. 다음은 teacher/map을 actor에 주지 않고 실제 preview 정확도와 motion/body 반사실 기여, 0–100km/h 조작 표적을 검증하고 적격 TRAIN에서 학습한다.


### 사전 등록 — 시각 예측의 원본 교사 주행 경로 비교 (2026-11-01)
이미 고정된 TRAIN/known DEV 예측 명령을 변경·재학습하지 않고, 같은 episode 원본 camera metadata 및 API telemetry에서 원래 `future_path_preview`의 20m 이동 경로 chord만 평가 표적으로 생성한다. driving/AI/wheels/연속 간격 및 실제 20m 미래 구간 부재는 기존 mask 그대로 둔다. 예측은 저장된 yaw=-body speed×chord를 speed>1m/s인 경우에만 역산하고 클립 여부를 확인한다. 방향 부호는 |chord|≥0.001/m에서만 비교하고 전체 RMSE와 zero 예측 기준도 기록한다. 실제 최적 racing line, 도로 중심 또는 독립 physical timing으로 부르지 않으며 원본 경로/pose를 actor에 제공하거나 DEV로 optimizer를 실행하지 않는다. 반복 프레임/단일 구간의 상관을 인정해 통계 유의성·폐루프 주행 주장은 하지 않는다.


### 2026-11-01 고정 시각 preview의 원본 교사 경로 평가
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pace-train-preview-teacher-path-evaluation/`, `.../20261101-pace-development-preview-teacher-path-evaluation/`는 고정 prediction을 바꾸지 않은 채 원래 camera metadata·script clock 및 실제 교사 이동 telemetry에서 20m chord curvature를 계산했다. 기록의 경로·pose는 평가 전용이고 actor 입력/optimizer에는 주지 않았다. TRAIN728 중 20m valid698·속도/마스크 포함 비교553, DEV729 중 valid698·비교552. curvature RMSE(1/m)는 TRAIN0.008331 vs 항상 직선0.002894, DEV0.008220 vs 직선0.006247로 현재 예측이 두 기록 모두 더 나빴다. |교사 chord|≥0.001/m의 방향 일치율은 TRAIN329개에서71.4%(좌55/우274), DEV406개에서44.6%(좌293/우113)다. 좌우 불균형 및 연속 frame 상관이 있어 독립 시행 또는 확률적 일반화 성능으로 해석하지 않는다. 20m 실제 교사 주행 경로는 최적 racing line이나 객관적 트랙 중심이 아니다.

별도 `...-preview-teacher-path-evaluation-audit/` 두 건은 FutureTrace의 다른 순회 방식으로 전 원본 target/유효성 및 RMSE·부호 계산을 재구성해 최대 차 TRAIN5.08e-14·DEV1.27e-14를 확인했다. predictor 독립 재실행은 아니다. 물리 sensor/exposure 동기, 자연 제동, 생리, 학생 폐루프 성능은 미검증이고 예측기는 주행 자격 실패로 남긴다. DEV는 이미 알려진 개발 구간이며 이 평가에 사용됐으므로 최종 테스트로 부르지 않는다. 새 optimization/차량 제어0. 다음은 직선 기준보다 못한 방향 예측을 고치기 위해 원본 TRAIN의 인과적 시각/몸 features, 양방향 코너 커버리지와 시각 경계 gradient를 점검하고 명시적 학습·독립 평가 기준을 사전 고정한다.


## 2026-11-01 기존 TRAIN 시각 예측 방향 분포

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-train-preview-direction-coverage/`에 원본 TRAIN 세 이력과 target SHA256, 계산 결과를 기록했다. 20m 교사 경로 곡률에서 |값|≤0.001/m를 직선으로 묶으면 유효 2,121프레임 중 좌 254, 직선 1,083, 우 784다. 임계값 기준 연속 구간은 각각 6, 13, 9개이며 연속 프레임과 구간을 독립 코너로 간주할 수 없다. 시계 간격은 약 40.19–43.57ms이나 물리 노출 동기 적합성을 보장하지 않는다. TRAIN 원본만 열었고 DEV payload 및 새 후보 학습은 사용하지 않았다. 단일 감각운동 회로 목표는 그대로이며, 이 분포만으로 기존 예측 실패의 원인을 확정하거나 새 모델의 성능을 주장할 수 없다. 다음에는 TRAIN 이력별 학습·검증과 시각/몸 개입, 자격이 확인된 새 이력 확보를 통해 방향별 일반화를 평가한다.


## 2026-11-01 TRAIN fold의 방향과 시야 비교

기존 t1692 저장 점수를 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-train-preview-direction-fold-summary/`에 재정리하고 앞선 원본 TRAIN 표적 감사의 유효·방향별 행 수와 일치시켰다. 해당 세 fold는 각기 다른 두 TRAIN 이력으로 학습하고 나머지 이력을 보류했다. 20m 곡률 RMSE (1/m)는 04: 시각+몸 0.009883, 몸-only 0.004560, zero 0.007064; 05: 0.011998, 0.008668, 0.013504; 06: 0.005102, 0.002918, 0.006019다. 몸-only는 별도로 학습한 비교군이다. 동일 시각+몸 모델의 전방 특징 제거는 각각 0.013302, 0.020781, 0.006759로 증가했지만, 전체 신경 특징을 평균으로 바꾸는 개입은 0.009865, 0.012225, 0.004909로 효과가 일관되지 않았다. 개입은 분포 밖일 수 있고 유효한 운전 기여나 생리학 입증이 아니다. 이력별 좌·우·직선 score와 시야별 개입 수치는 결과 JSON에 기록했다. 세 이력은 인접 구간이며 프레임은 상관돼 독립 주행 검증이 아니다. 기존 결과 요약이므로 추론 재실행·독립 점수 재계산·새 최적화는 하지 않았다. 기존 저자망 시간학습 t1696도 몸-only 기준을 통과하지 못한 사실을 유지한다. 다음 시각 학습은 단순 프레임 가중치만으로 해결을 가정하지 않고, 시야·시간·몸 개입의 인과 비교와 별도 구간을 사전 지정해야 한다.


## 2026-11-01 통합 상태의 학습 가능 경계

기존 `UnifiedFlightSensorimotor.decision`은 no_grad 및 매 결정 state/drive/traces detach로 동작하여 연속 회로의 시간 gradient를 보존하지 않았다. 기존 실행 API를 유지하고 같은 계산에 연구 전용 `training_decision`과 `detach_training_state`를 추가했다. 후자는 수치 상태·시간을 유지하면서 학습 구간 경계에서 계산 그래프만 끊는다. body/visual 포트 갱신은 미분 가능한 index_copy를 사용한다. 합성 8노드 fixture에서 실행 경로의 출력과 학습 경로가 정확히 일치하고, 이전 몸 포트에서 다음 상태까지 유한 비영 gradient, detach 후 수치 보존과 다음 구간 gradient, 합성 목표에 대한 8회 SGD 손실 감소를 확인했다. 관련 17개 검사 통과. 이 fixture는 실측 연결망이 아니며 실제 fly graph의 backward/GPU 메모리·원본 전체 replay 동등성·학습 가능한 적격 시계와 몸/조작 표적·실제 운전은 아직 확인하지 않았다. 새 후보는 fit_allowed=false 그대로다. 비행 유래 통합 회로, 가감속·변속·타이어/셋업 및 랩타임 목표는 유지한다.


## 2026-11-01 원본 재생 동등성과 비행 graph 역전파

수정 후 `20261101-unified-flight-train-preview-autograd-parity`에서 새 TRAIN 후보 원본 728결정과 두 기존 시각 예측 조건을 재생했다. `intents.npz` SHA256 `b93cecbfd343b16e0fdbca1e0549f6247dd141fe2c8139b8de6bfe67535070d2`는 수정 전 `20261101-unified-flight-train-preview-replay`와 동일하다. 원본은 fit_allowed=false이며 학습에 쓰지 않았다. 두 파일의 직렬화까지 일치한 범위에 한한 증거다.

`20261101-unified-flight-original-graph-backward-diagnostic`은 같은 원본 physical graph/visual/temporal checkpoint를 사용하지만 입력과 손실은 합성 fixture다. 8결정 연속 backward 결과 몸 감각 adapter 기울기 norm 0.09213183, 비영 3,120개, peak GPU allocated 250,192,384 bytes; optimizer 0회, 제어 0회다. 짧은 합성 진단은 실제 관측·장기 구간·생리 반응·실주행 성능 검증이 아니다. 다음은 적격 기존 TRAIN 이력과 시계/표적을 확인하여 사전 역할 분리된 실제 graph 학습을 수행하고 원본 외 검증을 해야 한다. 목표는 동일 통합 회로의 시각·몸 감각 되먹임, 조향·가감속·변속·셋업 적응과 빠른 유효랩이다.


## 2026-11-01 원본 감각 경로 gradient 진단

`20261101-unified-flight-recorded-train-backward-probe`는 `20261031-pace-train-native-copy-candidate-00`의 선두 16결정, 해당 FlyVis 4시야 및 시야별 body 이력을 원본 시간·결측·해시로 검증하여 실제 단일 graph에 연속 입력했다. candidate는 fit_allowed=false로 유지했다. 목표 명령은 임의 고정 yaw 0/속도 10m/s이며, 마지막 출력 제곱합은 연결성 측정용 임의 목표다. optimizer/조작 0회. 몸 adapter gradient norm 0.0998089 (비영 5460), 운동시각 projection 0.00337727 (비영 10752), peak GPU allocated 256,475,648 bytes다. 이는 시각·몸 경로로 수치 gradient가 도달한다는 짧은 진단이고 학습·생리·방향 예측·제동·폐루프 성공을 뜻하지 않는다. 물리 노출 시간과 독점 권한은 여전히 미검증이다.

별도 과거 `20261029-t1625-spatial-publisher-async-corpus/episode-04-train/result.json`은 training_admissible=false, expert_labels_available=false, physical_exposure_sync_verified=false라고 명시한다. 기존 교사 경로 보조 실험의 역할을 그대로 남기되 해당 기록을 자격 있는 조작 표적으로 승격하지 않는다. 다음 실제 통합 학습은 시각·몸의 원본 시간 및 동시 조작 표적 자격이 입증된 TRAIN 이력을 확보하거나 기존에 명시 승인된 별도 실험 범위 안에서 설계해야 한다.


## 2026-11-01 교사 조작의 host receipt 창 변화

`20261101-pace-train-control-receipt-window-drift-v2`에서 기존 독립 몸 감사의 events/native SHA와 15,556개 420바이트 copy별 hash를 확인했다. 각 TRAIN 장면의 네 렌더 시작 하한 중 가장 이른 시각 직전 수신값부터 결정시각까지의 native steerAngle/gas/brake 범위를 원래 수신 시간 순으로 집계했다. 728개 창 모두 적어도 한 packet 변화가 있었고 범위 0.01 초과는 각각 35/78/3개, 중앙값 0.000339/0.000381/0, 95백분위 0.009439/0.083339/0, 최댓값 0.449020/0.800000/1.000000이다. 임계값은 진단용이며 raw steer는 보정되지 않았다. 시작 하한 직전 수신을 포함한 host 창이지 물리 노출이나 실제 제어 발생의 동기 오차 측정이 아니다. 일부 큰 변화는 인위적 제동 과제 경계일 수 있으며 원래 phase를 삭제하거나 보정하지 않았다. 기존 fit 금지, 표적 미생성, optimizer/제어 0. v1 증거도 보존했다.

확인 시 AC 프로세스가 없었으며 session 상태 파일은 약 11시간 전에 갱신돼 현재 상태의 증거가 아니다. 실제 수집은 AC 및 CSP lifecycle 가용성 확인 후 진행해야 한다. 새로운 동기 자료와 이동 조향 보정·권한 검증 전 해당 기록을 조작 학습에 승격하지 않는다. 단일 겹눈/비행 유래 몸 회로와 가감속·변속·셋업·빠른 랩 목표는 유지한다.


## 2026-11-01 기존 학습 몸 경계의 단일 graph 전이

`scripts/replay_unified_flight_candidate.py`에 명시적 `--body-feedback` 비교만 추가했다. 고정 checkpoint SHA `4030ac7e53c9175a517adb3d588d9c5be4746f818ddc09762c6ee524771e2315`는 이전 1909 TRAIN의 가속/제동과 기존 조향 보존 손실로 학습된 공학 몸 감각 adapter다. 당시 가속 MSE는 감소했으나 제동 MSE는 0.193343→0.195367로 악화해 actor 자격이 없다. checkpoint 원래 schema·미배포·감각 채널/척도·포트 형상과 원본 graph/ports를 검사했다. 학습 당시 사용한 pedal command 파라미터는 현재 전이 실험과 다르므로 동등한 전체 모델이 아니다.

같은 원본 영상·몸·고정 미검증 시각 preview 명령을 사용하고 각 조건의 단일 graph 재귀 상태를 별도 유지해 adapter 전이와 zero를 비교했다. `20261101-unified-flight-train-anchored-body-transfer` 728결정의 두 intent 평균 절대 차 0.0128341/0.0121164, 최대 0.0185908/0.0170869; `20261101-unified-flight-development-anchored-body-transfer` known DEV729결정 평균 0.00875195/0.0112139, 최대 0.0108267/0.0172486다. 두 neutral 조건은 동일한 이전 preview replay의 intact 출력 및 clock과 정확히 일치했다. 전이 조건과 중립의 차이만으로 특정 실제 몸 변수에 대한 인과 기여, 생리 타당성, 제동 또는 주행 개선을 주장하지 않는다. 유효성 indicator 기여도 포함된다. 두 원본은 fit_allowed=false, optimizer/실제 제어 0, DEV는 이미 알려진 개발 기록이다. 17개 관련 단위 검사 통과. 향후 각 tilt/load/tyre 값과 결측을 개별 개입하고 물리 시간·조작 자격 및 폐루프를 검증한다. 단일 회로의 가감속·변속·셋업과 빠른 유효랩은 여전히 미완이다.


## 2026-11-01 통합 graph 몸 감각 값 개입

`20261101-unified-flight-{train,development}-body-value-interventions/`에서 기존 학습된 1909 adapter를 고정하고 원본 TRAIN728/known DEV729를 각각 동일한 5조건으로 연속 재생했다. intact 외 기울기(인덱스0–1), 하중(8–11), 온도(12–15), 바퀴회전(17–20)의 값만 매 결정마다 adapter 입력에서 0으로 했다. 해당 원본 valid mask와 몸 직접 경로·시야별 맥락·시각 preview·나머지 값은 그대로며 조건마다 별도 graph 상태를 갖는다. 상태/시각 개입에 해당하지 않는다. 결과 파일의 intents 순서는 intact, tilt, load, temperature, wheel이다. 정상 조건은 직전 각 전이 실험과 원래 clock 및 output이 정확히 같았고, 출력 배열에서 모든 평균 차를 별도 재계산했다.

| adapter 값 치환 | TRAIN 두 intent 평균 절대 차 | known DEV 두 intent 평균 절대 차 |
|---|---|---|
| 기울기 | 0.002882 / 0.0000573 | 0.001855 / 0.0000310 |
| 하중 | 0.000675 / 0.000412 | 0.000680 / 0.000376 |
| 온도 | 0.001259 / 0.000683 | 0.001440 / 0.000709 |
| 바퀴 | 0.001731 / 0.001810 | 0.001740 / 0.001711 |

값 0은 관측 분포 밖일 수 있으며 실제 고장이나 생리적 반응을 의미하지 않는다. 상대편은 원시 intent이고 조향/페달 제어 성능이 아니다. 기존 adapter는 제동이 악화한 미배포 후보이며 다른 pedal command에서 학습됐다. known DEV는 독립 최종 시험이 아니며 물리 시각·자율주행·랩은 미검증이다. 최적화·차량 명령0, fit 금지 원본 유지. adapter 입력 한정 및 원본 불변·mask 유지 4검사와 통합 상태 5검사 통과. 다음은 적격 시계/제어 표적 확보와 회로 자체의 감각 생리·폐루프 기여를 검증한다.


## 2026-11-01 시야별 반대 운동 채널 교환

`20261101-unified-flight-{train,development}-view-motion-reversal/`에서 앞·좌·우·뒤 중 한 시야씩 T4/T5의 각 ON/OFF 상대 방향 채널쌍 0↔1, 2↔3을 pooled scene에서 교환해 TRAIN728과 known DEV729 전체를 연속 재생했다. 망막 강도와 다른 시야·몸 입력은 유지했고 시각 preview 명령도 원본 intact로 고정했다. 따라서 반사실은 직접 시각 경로만 바꾸며 실제 반대 방향으로 움직인 영상을 새로 렌더하거나 preview까지 바꾼 것이 아니다. 조건별 graph 상태는 독립 유지했다. 저장 배열 순서는 intact/front/left/right/rear이며 각 intact는 이전 1909 adapter 전이의 clock·출력과 정확히 일치했고 평균 차는 배열에서 재계산했다.

| 시야 | TRAIN 두 intent 평균 절대 차 | known DEV 두 intent 평균 절대 차 |
|---|---|---|
| 앞 | 0.000777 / 0.00000628 | 0.001189 / 0.00000255 |
| 좌 | 0.000464 / 0.00000277 | 0.000974 / 0.00000222 |
| 우 | 0.001068 / 0.00001527 | 0.000870 / 0.00001203 |
| 뒤 | 0.000858 / 0.00001543 | 0.001638 / 0.00001117 |

이것은 방향 정보의 교환에 대한 수치 민감도이며 운동 방향의 정확한 부호·시간 선택성, 실제 파리 생리나 운전 이득을 입증하지 않는다. 채널 교환은 실제 활동 분포 밖일 수 있다. 학습된 body adapter와 preview는 기존처럼 미검증이고 fit 금지 기록에서 optimizer/차량 제어0. 네 시야 교환의 상대 채널·다른 시야/망막 보존 및 2회 적용 복원 검사 등 관련 13검사 통과. 다음에는 제어된 실제 이동 자극에서 방향 부호·시간 지연과 억제의 예상 반응을 사전 지정하고, 몸 감각 폐루프·가감속/변속/셋업 및 빠른 유효랩 검증으로 연결해야 한다.


## 2026-11-01 FlyVis 721열과 단일 비행 graph의 이동 경계 자극

이전 중앙 뉴런값을 모든 공간 bin에 복제한 자극의 한계를 줄이기 위해 `20261101-flyvis-author-edges-full-columns/`에서 기존 원저자 실험의 동일한 밝음/어두움 및 0°/180°, 속도19 자극을 원래 pretrained FlyVis와 원래 5ms 간격으로 재실행했다. 각 485프레임·721 수용기 위치에 대응하는 8종 T4/T5 전체 공간 출력을 보존했다. 저장된 원저자 중앙 전체 뉴런과 최대 차 4.77e-7~9.54e-7로 사전 허용 2e-6 이내이며 GPU peak allocated 1,437,450,240 bytes다. 새 최적화0.

`20261101-unified-flight-author-edges-full-columns/`에서 각 자극을 8프레임 간격(40ms)으로 표본화해 네 시야 중 한 시야씩 원래 상태를 유지하는 동일 비행 graph에 입력했다. 나머지 세 시야는 해당 조건의 최초 시각 반응을 고정했고, 가상 몸과 명령 yaw0/속도10m/s는 고정했다. 따라서 동시에 움직이는 실제 4시야나 AC 영상이 아니다. 각 조건의 시작 전 t<1초 반대방향 intent 차는 최대 9.69e-8/0이었고 t≥1초 첫 intent의 0°−180° 평균은 어두운 자극 앞/좌/우/뒤 -0.001401/+0.0000938/-0.000948/+0.000862, 밝은 자극 +0.000924/-0.000205/+0.000569/-0.000442였다. 전체 원본 배열에서 최대 및 평균을 재계산했다. 각 방향의 실제 올바른 조향 부호는 사전 검증하지 않았으며 밝기 반전 부호·평균 차만으로 생리학적 방향 선택성이나 제어 유용성을 보증하지 않는다. 원래 몸 adapter/시각 경계는 공학적 미배포 후보며 40ms sampling 및 고정 나머지 시야가 제한이다. actor/차량 제어·optimizer0, 실제 서킷 완주·가감속/변속·셋업·랩타임 미검증.


## 2026-11-01 네 시야 동시 가상 yaw의 시각 경로

`20261101-flyvis-coherent-four-view-yaw/`는 실제 회로가 아니고 반지름20m 고정 질감 구와 현재 exterior 카메라 기하에서 시드863101로 렌더한 진단이다. 정지/−0.23/+0.23rad/s yaw, 0.4초 정지 후 회전, 40ms×41프레임을 사용했다. 각 조건과 네 시야의 FlyVis 상태는 독립이며 기존 저자 checkpoint, 각 프레임 8×5ms, 전체721열 T4/T5 반응을 보존했다. 시각 입력의 회전 전 11프레임은 세 조건이 정확히 같고 운동 반응 최대 차는 4.18e-7이다. 회전 후 양방향 retina 차이는 네 시야 모두 존재했다. GPU peak 5,519,787,520 bytes, 학습/차량 제어0.

`20261101-unified-flight-coherent-four-view-yaw/`에서 한 번에 네 시야를 같은 graph state에 넣었다. 몸 yaw는 의도적으로 0, 차속10m/s 및 명령 yaw0/속도10m/s를 유지해 시각 효과를 분리했으며 실제 몸/시각 일치나 회전 제어를 흉내 내지 않았다. 반대 방향 두 출력 차는 시작 전 정확히 0, 전체 최대 0.0009597/0.00001824, 시작 후 첫 intent 차 평균(+yaw−−yaw)은 −0.00030979다. 정지 조건 대비 0.4초 이후 첫 intent 평균은 −yaw +0.00031488, +yaw −0.00000524로 비대칭이다. 이는 한 질감·한 속도·짧은 자극에서의 값이며 회전 안정화의 올바른 부호나 일반화 증명이 아니다. 양방향에서 시각 기여가 비대칭인 원인을 반복 질감/속도 및 몸 yaw 일치·반대 조건의 개입으로 검토하고, 이동 중 안전한 조작 표적과 실제 폐루프·변속/셋업/빠른 랩 검증을 계속한다.


## 2026-11-01 동시 네 시야와 몸 yaw의 일치/불일치

같은 `20261101-flyvis-coherent-four-view-yaw/`를 `20261101-unified-coherent-yaw-body-matched/` 및 `...-body-opposite/`에 재생했다. 세 시각 조건 각각에 t>0.4초부터 body21 yaw index4를 시각 회전 0/−0.23/+0.23rad/s와 같게 또는 정반대로 주고, 이전 실험 body yaw0와 비교했다. 시야별 과거 몸 context는 명시적으로 결측, 현재 body 나머지 값과 명령 yaw0·속도10m/s는 고정이다. 정지 조건과 t≤0.4초의 세 모드 출력은 정확히 일치한다.

회전 후 t>0.4초 +시각 yaw−−시각 yaw의 첫 intent 평균은 body 고정 −0.0003201, 일치 −0.0405233, 반대 +0.0398833이다. 원시 intent이며 제어 부호를 확정하지 않는다. 동일한 +body yaw에서 양쪽 시각 조건의 몸 효과 차이 `(matched +시각 − fixed +시각) − (opposite −시각 − fixed −시각)`는 첫 intent 평균 7.63e-7·최대 2.94e-6, 반대 몸 yaw에서도 평균 5.41e-7·최대 1.61e-6이었다. 이는 본 합성 fixture와 고정 명령에서의 사후 차이의 차이이며 일반적 감각 통합 부재·생리 타당성·안정화/폐루프를 증명하지 않는다. 몸 감각 효과가 시각 단독 차이보다 커 방향성 회로 훈련 때 시각 무시를 별도 감시해야 한다. 두 graph 실행 optimizer/차량 제어0, 사전 대상/최종 검증 아님. 단일 회로 가감속·변속·셋업·빠른 유효랩 목표는 유지한다.


## 2026-11-01 사전 고정: 합성 yaw 안정화 시각 경계 학습

기존 질감 seed 863101의 합성 정지/±0.23rad/s yaw 네 시야와 새 seed 863102를 동일 프로토콜로 만들고, 전자는 fit, 후자는 개발 평가에만 사용한다. AC 원본/fit 금지 자료는 사용하지 않는다. 모든 조건의 몸 yaw는 0, 과거 body context는 결측, 고정 yaw 명령0 및 속도10m/s다. 원래 단일 graph·FlyVis·기존 몸 경계를 고정하고 운동 시각의 opponent projection만 학습한다. 목표는 기존 readout의 양의 raw intent가 좌회전이라는 공학 계약에 따라 시각 회전에 반대되는 방향 반응을 만들어내는 가설적 안정화 신호다. +yaw에는 정지 기준 intent −0.01, −yaw에는 +0.01을 부과한다. 크기 0.01은 진단적 공학 선택이며 측정 생리나 최적 차량 제어가 아니다. 손실은 마지막 10프레임의 세 조건 평균에서 방향별 기준 차 MSE와 정지 출력을 원본에 유지하는 항이다. optimizer Adam lr 0.01, 고정 12회 전개, seed863009; 전 이력 상태는 보존하고 optimizer update 전에만 각 독립 에피소드를 새로 시작한다. 결과 선택/조기 종료에 개발 자료를 쓰지 않고 마지막 checkpoint 한 번만 평가한다. 최대 메모리/gradient 유한성과 graph 불변·0 command를 검사한다. 정지·좌우의 선형부호가 새 질감에서도 맞는지와 실제 출력 차이를 보고한다. 이는 폐루프, 생리 또는 유효랩 검증이 아니며 조작·배포하지 않는다.


## 2026-11-01 합성 yaw 안정화 경계 학습 결과 — 새 질감 부호 실패

사전 기록된 계획대로 `20261101-unified-synthetic-visual-yaw-fit/`에서 질감863101의 3개 전체 이력과 4시야·기존 FlyVis·단일 graph state를 유지하고 opponent projection만 Adam 0.01로 12회 갱신했다. 몸 yaw0·명령 yaw0이므로 합성 회전 표적을 actor 입력으로 주지 않았다. 원래 graph 및 몸 경계는 requires_grad false, 시야별 body 맥락은 결측이다. 정지 대비 목표 −yaw/+yaw raw intent 변화는 +0.01/−0.01이라는 공학적 안정화 가설이며 실제 조향·파리 생리의 측정값은 아니다. 최종 checkpoint `f57288a9eac2f3789d043532596503411375bf49b0f6bb0296a8c1471b4061af`, actor_qualified=false; 그래프/몸/기존 시각 가중치는 기존 파일 보존.

Fit 마지막10프레임 평균 변화는 이전 +0.000412/−0.000031, 이후 +0.011350/−0.011540이고 목표 MSE 9.566e-5→2.096e-6이다. 학습에 열지 않은 질감863102의 변화는 이전 +0.000633/+0.000279, 이후 −0.000139/−0.004245로 첫 방향이 목표와 반대다. 두 방향이 모두 맞아야 하는 사전 부호 기준에 실패했다. 개발 손실이 9.670e-5→6.796e-5로 줄어도 방향 일반화 또는 운전 성공으로 판정하지 않는다. 원본 배열에서 평균과 MSE를 별도 재계산했다. GPU peak allocated 323,473,920 bytes, optimizer12, 차량 명령0. 추가 질감·속도·개별 시야 인과 평가와 몸 회전이 있는 경우의 분리된 검사, 적격 AC 시각/조작 기록 및 폐루프·제동·변속·셋업·랩 검증이 남아 있다.


## 2026-11-01 사전 고정: 다중 질감 합성 시각 학습

앞의 seed863101 한 질감 학습 실패를 계기로 새로운 독립 초기화의 fit 질감은 863101/863103/863104 세 개, 새 사전 지정 개발 질감은 863105다. 이전 개발 863102는 역할을 바꾸거나 fit하지 않는다. 동일 41프레임 정지/±0.23rad/s, 네 시야, 몸 yaw0, graph·FlyVis·몸 경계 고정, 정지 대비 ±0.01 공학 목표와 정지 anchor 및 Adam 0.01을 유지한다. 각 질감의 세 전체 이력 손실을 동일 가중 평균하고 epoch마다 1회, 총 12회 갱신한다. 각 질감별 초기 정지 출력을 anchor로 저장하며 마지막 checkpoint만 지정 개발 질감에 평가한다. 양 방향 부호·정지 기준 차·정량 오차를 전부 기록하고 actor 배포·실제 조작0, 생리·실주행·폐루프 자격을 부여하지 않는다.


## 2026-11-01 다중 질감 통합 시각 학습 — 보류 질감 방향 실패

사전 고정한 seed863101/863103/863104 세 합성 질감의 3조건·네 시야 전체 41프레임에서 `20261101-unified-synthetic-yaw-multitexture-fit/`은 단일 비행 graph·FlyVis·몸 경계를 고정하고 opponent projection만 독립 원래 초기값에서 Adam 0.01로 12회 갱신했다. 각 질감의 같은 원래 정지 평균을 anchor로 삼고, 세 loss를 동일 가중 평균했다. 이전 개발 seed863102는 사용하지 않았다. 신경 상태는 각 이력 안에서 유지하고 이력 시작에만 reset했다. 시각 회전 표적은 loss 전용, 몸 yaw·명령 yaw0, 센서 이력/graph에 주입하지 않았다. ±0.01 target은 공학적 안정화 가설이지 실측 비행 반응이 아니다.

Fit의 세 질감 모두 정지 대비 양방향 부호를 맞췄다. 863101은 +0.009833/−0.009977, 863103은 +0.013050/−0.011365, 863104는 +0.011586/−0.012588이다. 사전 지정 보류 질감863105는 학습 전 +0.000543/−0.000613에서 학습 후 −0.002668/−0.005114로 한쪽 방향이 반대로 바뀌었고 목표 MSE 8.877e-5→9.218e-5로 악화했다. 마지막 model만 평가했고 기존 checkpoint와 실패 보존, 새 checkpoint SHA `2704392c21a43f4fa91b5174a1f654d4d1f5e53c78060818021e0e7dff8b584f`는 actor_qualified=false다. 결과 arrays에서 학습 전후 8개 점수 독립 산술을 확인했고 GPU peak allocated 477,154,816 bytes, 실제 제어0. 세 질감만으로 분포 일반화를 기대할 수 없고 시각/비행 생리·몸 피드백·실제 트랙·제동/변속/셋업·유효랩은 미검증이다. 다음은 heldout 역할 보존 아래 광류 방향의 원인과 실제 자극 구조를 살피고, 인과적 시각 기능과 시간 정합된 자율 폐루프 경로를 개선한다.


## 2026-11-01 사전 고정: 기하 운동 부호의 새 질감 확인

이미 본 합성 질감 다섯 개의 원래 T4/T5·721열에 기존 카메라 기하 회전 template를 적용한 사후 진단에서 정지 대비 yaw feature는 −회전에서 음수, +회전에서 양수였다. 이 관찰로 같은 절차·0.23rad/s·마지막10프레임 평균·정지 조건 대비 차의 부호를 아직 생성하지 않은 독립 seed863106에 사전 지정한다. 기하 template는 공학적 가설이며 label/pose를 배우 입력에 넣지 않는다. 원래 전체 5ms FlyVis 이력과 네 시야·카메라 등록을 유지하고 fit/최적화·차량 제어0. 이 검사는 실제 트랙·생리·closed-loop나 서로 다른 속도/환경 일반화를 판정하지 않는다.


## 2026-11-01 기하 운동 축의 새 질감 검사

`20261101-geometric-yaw-texture-diagnostic/`은 이미 본 fit863101/103/104와 known DEV863102/105에서 원본 4시야 T4/T5 721열과 카메라 방향 등록을 `VisualSelfMotionPool.from_registration`으로 검사하고, 고정 기하 template의 yaw 축을 마지막10프레임 평균했다. 각 같은 질감의 별도 정지 자극 대비 −yaw/+yaw 변화는 5개 모두 음/양이었다. 이는 사후 관찰이다. 이후 사전 기록된 새 합성 seed863106을 4시야 FlyVis 전체 이력으로 생성하고 `20261101-geometric-yaw-fresh-texture-diagnostic/`에서 동일하게 계산했다. 정지 대비 −0.0917743/+0.0674371로 지정 부호가 맞았다. 기하 template는 FlyVis 운동 채널과 카메라 광선의 공학적 기능 대응이며 측정된 HS/VS 연결이 아니다. 정지 기준은 같은 장면의 별도 반사실 전체 이력에서 얻었으므로 이 수치 자체를 실시간 actor가 사용할 수 있다고 주장하지 않는다. 한 속도·합성 구형 질감이고 새로운 seed 하나는 통계적 일반화나 생리/차량 검증이 아니다. optimizer/차량 제어0. 다음은 실제 독립적인 정지 기준 계약 또는 인과적 시각 시간 경계와 함께 이를 단일 graph 포트에 넣고, 비운동/온전한 이전 시간·결측·속도 변화 및 held-out 자극을 비교한 다음 적격 실주행 데이터에서 폐루프를 검증한다. 가감속·변속·셋업 및 빠른 유효랩은 계속 미완이다.


## 2026-11-01 사전 고정: 인과적 정지 기준의 기하 시각 포트 학습

고정 FlyVis T4/T5와 camera rays의 기존 6축 기하 template는 공학적 기능 근사다. fit 질감863101/103/104, 새 보류 질감863107; 이전 DEV102/105/106은 사용하지 않는다. 각 41프레임 이력 자체의 사전 지정 t≤0.4초 11프레임만 정지 기준으로 평균하고 그 이후의 현재 특징에서 뺀다. 이 합성 자극에서 정지는 프로토콜로 보장되지만 실제 actor에서의 정지 판정은 별도 검증·인증이 필요하다. 첫 11프레임은 기준 계산 완료 전 0 전달, 후속에만 기준 대비 6축을 전달한다. ground truth 회전은 손실에서만 사용한다. geometry 6→기존 visual port bounded projection만 0 초기값부터 학습하며 기존 temporal·몸·graph는 고정한다. 목표는 이전과 같은 각 질감의 정지 대비 ±0.01, 정지 anchor, Adam 0.01, 12번 세 질감 평균 업데이트, 마지막 checkpoint 하나만 보류 질감 평가다. 저자망을 재훈련하거나 시간/몸을 리셋하지 않으며 생리/실주행/배포 성공이라 부르지 않는다.


## 2026-11-01 인과 정지 기준 기하 시각 포트 학습 결과

`20261101-unified-synthetic-geometry-yaw-fit/`에서 사전 고정한 seed863101/103/104 fit, 863107 holdout을 사용했다. 각 이력 자신의 처음 11개 정지 프레임만으로 camera geometry 가중 운동 6축 기준을 계산하고 11개 관측 동안 경로는0, 이후 현재값−기준을 bounded linear projection으로 같은 graph의 기존 시각 포트에 주입했다. 이 정지 보장은 합성 프로토콜에 한정되며 현실의 검증된 정지 감각/누락/시각 동기는 구현하지 않았다. 원래 temporal·몸·graph·FlyVis를 고정하고 새6축 투영만 0에서 Adam 0.01로 12회, 세 질감 동등 손실로 학습했다. 원본 graph와 초기 출력은 기존 실험과 최대2e-6 이내다.

Fit 3질감은 정지 대비 −yaw/+yaw 출력 변화가 각각 +0.000577/−0.000173, +0.000351/−0.000396, +0.000082/−0.000872로 부호는 맞지만 목표 ±0.01에 비해 작다. 처음/마지막 손실은 0.00019175→0.00018476이다. 새 seed863107은 학습 전 +0.000554/+0.000434, 학습 후 +0.000758/+0.000266으로 +yaw 방향 부호가 틀렸으며 목표 MSE 9.905e-5→9.540e-5다. 저장 배열에서 모든 지표를 재계산했고 단일 graph/새 port 초기중립 등 관련 6검사가 통과했다. 새 checkpoint SHA `a347b4c4eebb0230c2bfef2499ba80cef11ef84494dbbffbec9a1b3a1c71890f`, actor_qualified=false, GPU peak allocated 476,115,968 bytes, 차량 제어0. 생리·동시 몸 yaw·고속/셋업·실제 폐루프와 유효랩은 미검증이다. 같은 정지 baseline을 쓸 수 없는 상황 및 장기 지속·독립 속도와 질감에서 반응을 입증하고, 경로의 작은 gradient/출력 부호와 원인, 미검증 조작 타이밍을 해결해야 한다.


## 2026-11-01 사전 고정: 기하 yaw 단일 축 학습과 다른 속도 평가

직전 6축 실험의 작은 기울기와 보류 질감 부호 실패 후, 동일 단일 graph/원본 초기 가중치/정지 11프레임 기준을 유지하되 기하 feature 중 yaw 축만 기존 시각 포트로 보낸다. fit은 같은 seed863101/103/104와 ±0.23rad/s, loss 정지 대비 ±0.01 및 원본 정지 anchor; 새 독립 초기화에서 Adam 0.1을 30회, 매회 세 이력의 동일 평균으로 업데이트하고 마지막 모델만 평가한다. 새로운 질감 seed863108, 다른 회전 속도 ±0.13rad/s를 보류 평가로 사전 지정하며 target 절댓값 0.01×0.13/0.23은 공학적 선형 가설이다. 이전 DEV102/105/106/107은 fit에 쓰지 않는다. 마지막 프레임 10개의 양방향 부호와 목표 오차를 보고한다. 이는 통제된 합성 자극에서의 방향 경계 학습이며 실제 안정화, 생리, 차량 조작, 유효랩 자격을 주지 않는다.


## 2026-11-01 기하 yaw 단일 축 학습 — 합성 보류 부호 통과

사전 고정한 `20261101-unified-synthetic-geometry-yaw-axis-fit/`에서 기하 6축 중 yaw 하나만 기존 단일 graph의 visual port로 주고 나머지 5축은0으로 고정했다. 이전과 같은 세 fit 질감과 각 이력의 최초 11개 정지 프레임 인과 기준, yaw±0.23rad/s 및 정지 상대 목표 ±0.01, 동일 graph/몸/FlyVis 고정, 새 projection 0 초기값, Adam0.1, 전체 이력의 손실을 평균해 30회 갱신했다. 학습 신호는 공학적 방향 안정화 가설이다. 첫/마지막 학습 손실 0.00019175→0.00008792. 세 fit 질감 모두 마지막 양방향 부호는 맞았으나 목표 크기에는 미달했다.

학습에 사용하지 않은 사전 지정 seed863108, ±0.13rad/s에서 정지 대비 −yaw/+yaw 출력은 학습 전 −0.000110/+0.000288(두 부호 반대), 학습 후 +0.002738/−0.001540으로 예상 부호가 맞았다. 보류 속도에 비례한 가설적 목표 ±0.005652 대비 MSE는 3.424e-5→1.270e-5다. 이는 한 개 합성 보류 질감·한 보류 속도에서의 결과이며 진정한 독립 서킷/자율 제어 검증이 아니다. 출력 배열에서 모든 fit 및 speed-scaled DEV 지표를 재계산했다. checkpoint SHA `a9ed4f224b7dd00731ad51adcbd91f3bbe10812c83dd6cdfc31db950834927d8`, actor_qualified=false, optimizer30, GPU peak allocated 476,115,968 bytes, 차량 조작0. 현실에서 정지 기준을 어떻게 검증할지, 시각과 실제 몸 yaw의 공동 반응·다양한 환경 및 속도, 가감속·변속·셋업·유효랩을 검증해야 한다.


## 2026-11-01 고정 기하 yaw 경계의 기존 개발 질감 평가

`20261101-unified-geometry-yaw-known-texture-eval/`은 앞의 최종 checkpoint SHA `a9ed4f224b7dd00731ad51adcbd91f3bbe10812c83dd6cdfc31db950834927d8`를 고정하고 학습에 쓰지 않은 이전 known DEV102/105/106/107의 4시야·±0.23rad/s 전체 이력을 재생했다. 각 이력의 앞 11개 정지 프레임으로만 기준을 정하고 나머지 graph/몸 yaw0/명령을 유지한다. 추가 optimizer0, 각 이력은 별도 상태. 정지 대비 −yaw/+yaw raw intent 차는 학습 후 각각 +0.003673/−0.002812, +0.004951/−0.004252, +0.003853/−0.003494, +0.003825/−0.002329로 4/4에서 공학적 목표 부호를 보였고 목표 ±0.01 MSE도 초기 대비 모두 감소했다. 원래 결과 arrays에서 8개 score를 재계산했다. 이들 질감은 이전 개발 중 이미 관찰했으므로 최종 독립 검증이 아니다. 합성 구형 장면·정지 시작·몸 yaw0이며 미검증 실제 이동/몸 감각 정합·생리·제동·변속·셋업/랩 결과를 대체하지 않는다. checkpoint 미배포, 차량 제어0. 이후에는 정지 기준이 실제로 인과적으로 확보된 기록과 몸 yaw 일치/충돌·방향 외 이동·실제 폐루프를 검증한다.


## 2026-11-01 기하 시각 후보와 현재 몸 yaw 합성 비교

고정 checkpoint `a9ed4f...`를 동일 known DEV 4개에 넣은 `20261101-unified-geometry-known-textures-body-matched/` 및 `...-body-opposite/`와 기존 body-yaw0 결과를 비교했다. 0.4초 후 몸 yaw index4를 시각 ±0.23rad/s와 같게 또는 반대로 지정했고 각 조건의 그래프 상태는 따로 유지했다. 시야별 과거 몸 맥락은 명시적으로 결측, 명령 yaw0/속도10m/s, 처음11개 자신의 정지 기준만 사용한다. 세 조건의 정지 이력과 t≤0.4초는 bitwise 같다.

시각과 현재 몸 yaw가 일치할 때 정지 대비 첫 intent는 4질감 각각 +0.026121/−0.025142, +0.027400/−0.026578, +0.026301/−0.025820, +0.026273/−0.024658이었다. 시각과 반대 몸 yaw에서는 각각 −0.018671/+0.019628, −0.017395/+0.018186, −0.018490/+0.018943, −0.018518/+0.020111로 몸 신호 쪽 부호가 우세했다. 각 동일 몸 조건에서 학습된 시각 경계를 0과 비교한 효과는 body yaw0의 같은 시각 경계 효과와 평균 절대 차 첫 intent 약 2.95e-6(일치)/2.79e-6(반대)였다. 합성 한 속도와 정지 onset·고정 몸 값의 조건부 관찰이며 두 입력의 일반적 통합 방식이나 옳은 생리/안정화 반응을 확정하지 않는다. 기존 몸 경로 자체가 회전 피드백을 갖고 있어 반대 자극은 감각 불일치 실험이지 실제 대응 관계가 아니다. optimizer/제어0, known DEV, 미배포. 이후 실제 시야별 몸 수신 이력·실제 정지 계약·감각 결측과 독립 조건 및 폐루프를 검증하고 가감속/변속/셋업·빠른 유효랩 목표를 유지한다.


## 2026-11-01 인과 정지 감각 기반 기하 포트 구현

`CausalGeometryVisualPort`는 기존 4시야 motion·등록된 카메라 기하와 검증 가능한 native body prefix를 받아 기존 시각 context에 bounded projection을 더하는 비배포 경계다. native bytes와 디코딩된 값이 동일하고 receipt age가 0–50ms이며 속도≤0.1km/h, 네 바퀴 회전≤0.05, 각속도≤0.02이고 서로 다른 프레임/packet으로 0.4초 이상의 11개 정지 관측이 있을 때만 자신의 정지 기준을 만든다. 11번째는 기존 학습과 같이 보정0이고 다음 결정부터 yaw 기하 차를 쓰며, 기준은 에피소드에 고정한다. 시야 누락·body 없음/오래됨은 해당 결정 보정0, 초기 기준 수집 중이면 불완전 이력을 버린다. reset_episode는 caller의 episode/source-generation 경계에서 명시적으로 호출해야 한다. 이 host receipt 계약은 센서 물리 시각이나 카메라 exposure 동기를 입증하지 않으며 실제 생리 tuning도 아니다. 기존 StationaryMotionReference의 기본 마지막3 평균을 유지하고 이 후보만11 평균을 명시한다.

`20261101-causal-geometry-reference-synthetic-equivalence/`에서 7개 합성 질감×3조건×41결정의 원본 FlyVis 반응과 합성으로 인코딩한 정지/이동 native packet을 넣고 이전 학습 스크립트의 인과11프레임 기하 입력과 비교했다. 모든 결정의 최대 차는 1e-6 미만이었다. 합성 packet은 실제 센서 증거가 아니며 이 구현은 아직 전체 graph의 자동 연결·원본 live timing·조작/운전 검증이 없다. 15개 기존·신규 관련 테스트 통과, optimizer/차량 명령0. actor_qualified=false checkpoint를 실제 actor에 배포하지 않는다. 다음은 원본 정지 body·네 시야에서 reference 수집/결측/타이밍과 graph 포트의 후보 전후 동일성을 검증한 뒤 실제 폐루프와 가감속·변속·셋업·랩 목표에 연결한다.


## 2026-11-01 원본 후보에 인과 정지 기준과 학습 시각 포트 재생

`scripts/audit_causal_geometry_candidate.py`로 원본 카메라/몸·FlyVis audit과 bytes, 등록/시계/role을 확인해 `20261101-{train,development}-causal-geometry-reference-audit/`를 실행했다. TRAIN728 및 known DEV729에서 정지 기준 후 신호 가능은 각각285/286결정, 최초 index441(0-based)이다. 시작 후 바로 이동하므로 초반에는 정지 기준을 만들지 못했고 나중의 정차 구간에서 형성됐다. 각각 반복 카메라 캡처4/3회는 해당 결정 보정0으로 유지한다. 이전 구현의 반복 capture 예외를 명시적 unavailable로 수정해 동일 프레임을 새 관측으로 재사용하지 않도록 했다.

`20261101-unified-flight-{train,development}-causal-geometry/`에서 같은 원본 네 시야/몸·고정 미검증 preview·과거 학습 몸 adapter와 기존 비행 graph에 고정된 합성 학습 geometry checkpoint `a9ed4f...`를 연결하고 zero 가중 대조를 독립 상태로 재생했다. 대조 output은 이전 동일 몸 adapter 결과와 bitwise 같고, 독립 readiness 행과 저장 mask가 정확히 일치한다. 준비 전 output 차이는 정확히0; index441 이후 두 intent 평균 절대 차는 TRAIN0.000122265/0.000000534, DEV0.000054656/0.000000201이다. 이 차이는 출력 민감도이며 더 정확한 코너 예측·조향/가감속 개선이나 생리 기능을 증명하지 않는다. 마지막 정지 감각 기준을 이후에 계속 보유하며 조명/환경 변화나 실제 exposure 동기 안정성은 미검증이다. 원본은 fit_allowed=false, optimizer/차량 조작0; 현 센서 자료는 host receipt 기반, known DEV는 최종 테스트가 아니다. 관련 15검사 통과. 다음은 정지 기준 전 시각 경로의 명시적 결손을 설계하고 실제 시각 예측/몸 감각의 기여와 시계·조작/실제 폐루프 및 가감속·변속·셋업·랩타임을 검증한다.


## 2026-11-01 원본 감각 yaw와 인과 기하 출력 사후 진단

`scripts/audit_causal_geometry_yaw_response.py`와 `20261101-{train,development}-geometry-native-yaw-diagnostic/`은 직전 원본 후보·independent reference readiness·네 시야 body·고정 graph output의 SHA와 결정시각 일치를 확인하고, 준비된 결정 중 몸 speed>1m/s 및 |yaw|>0.02rad/s인 행만 사후 집계했다. TRAIN46행 중 native yaw 양36/음10, known DEV20행 중 양14/음6이다. 기하 feature와 같은 결정 몸 yaw의 부호 일치는 42/46 및18/20, Pearson r은 0.623/0.570이다. 학습 geometry port 유무의 첫 intent 차가 같은 결정 몸 yaw와 반대 부호인 수는 42/46 및12/20이고 r은 −0.608/−0.154다. 각 부호 정수는 저장 배열에서 별도 재계산했다.

동일 index 비교이며 graph는 이전 입력을 통합하므로 순간적인 동시 반응이 아니다. host receipt는 물리 발생/노출 동기 검증이 아니고 native yaw는 독립 실측 정답이 아니다. 행은 연속되고 좌우 수·속도/구간이 불균형하다. 게다가 모델 출력의 부호와 body yaw의 반대 관계는 단순 가상 안정화 가설이지 실제 조향 적합성/바퀴 움직임·생리학을 보장하지 않는다. DEV에서는 기하 부호의 일치보다 그래프 output 변화의 일치가 낮아 시각 부품의 회로 전달을 더 검증해야 한다. fit 금지 기록 보존, 추가 optimizer/실제 차량 제어0; 원래 목표인 단일 감각운동 회로의 가감속·변속·셋업 및 빠른 유효랩은 미완이다.


## 2026-11-01 기하 시각 경로의 시간 비교와 단일 펄스

`20261101-{train,development}-geometry-fixed-lag-audit/`에서 같은 결정들의 신호를 기준으로 현재 및 이전1–5결정(중앙 간격 약0.041–0.206초)의 기하 신호/몸 yaw와 현재 graph intent 차를 비교했다. 모두 같은 행 TRAIN41·known DEV20을 사용했고 DEV에서 intent와 이전 기하 신호가 반대 부호인 비율은 lag0 14/20, lag5 18/20이었다. 이는 연속 기록에서의 사후 상관이지 정밀 지연 추정이 아니다. 몸 센서와 exposure의 물리 동기 미검증·자기상관·양방향 불균형이 있다.

후속 `20261101-unified-{train,development}-geometry-pulse-500/`에서 원본 카메라/몸/기존 고정 preview 등 다른 경로를 보존하고 geometry 보정 입력을 매 결정0으로 두되 실제 reference가 준비된 index500 한 결정에만 +0.005 합성 신호를 넣었다. 조건별 별도 graph 상태에서 동일 가중치/zero 대조와 비교했다. 양쪽 기록 모두 index500까지 차이는 정확히0, 첫 intent 차 >1e-7은 다음 index501의 0.04147/0.04123초, 절대 최대는 index505의 0.20772/0.20525초이며 크기 −2.902e-5/−2.817e-5였다. zero 조건은 기존 body adapter 정상 결과와 bitwise 동일했다. 이는 그 기록의 해당 graph 상태·한 크기/부호에서의 오프라인 신경 상태 충격 반응이며 물리 camera exposure, 실제 조향 지연, 동물 생리 또는 주행 성능을 나타내지 않는다. 합성 개입·미검증 preview·기존 몸 adapter를 사용했고 optimizer/차량 제어0, fit 금지 유지. 앞으로 더 빠른 반응의 생리 근거와 새로운 독립 조건, 시각 선행 예측·제동/변속/셋업 및 폐루프 랩 최적화를 검증해야 한다.


## 2026-11-01 기존 graph와 교사 20m 경로 방향의 사후 비교

`20261101-{train,development}-unified-geometry-teacher-path-audit/`는 원본 role/event SHA, 독립 검증된 실제 교사 경로 target, graph output 및 causal readiness 시계·hash를 확인했다. 같은 결정에서 정지 기준 ready와 기존 20m 평가의 valid·속도>1m/s·|chord curvature|≥0.001/m를 충족한 TRAIN54/known DEV27개는 모두 화면 왼쪽 target이며 오른쪽 유효 사례는0이다. 기존 정의에서 path 오른쪽 양수·raw flight intent 왼쪽 양수이므로 target 부호를 반대로 하여 raw intent 부호를 사후 비교했다. geometry port 적용 전/후 모두 일치는 0/54 및0/27, 부호가 바뀐 행0이다. 각 raw intent 평균 전/후 TRAIN −0.055154/−0.055285, DEV −0.050810/−0.050688이다. 전체 반복 연속 프레임이며 교사 실제 20m chord는 최적 라인도 직접 필요한 즉시 조향 정답도 아니다. 원본 camera/몸 물리 시각·실제 steering 전달은 미검증이므로 이 숫자를 운전 정확도로 부르지 않는다. 그래도 이번 정지 기준 이후 원본에서는 새 기하 경계가 회로의 지배적 방향을 뒤집지 못했고 좌우 다양성도 부족하다는 점을 유지한다. 위치/교사 경로는 오직 사후 평가, actor 입력0, optimizer/차량 제어0. 기하 시각 안정화 경로만으로 선행 예측·가감속/변속·셋업·빠른 유효랩 달성을 주장하지 않는다.


## 2026-11-01 기존 graph와 등록 quiet readout의 반사실

기존 v5 graph의 readout은 이미 reference_calibrated=true이며 lateral_reference=[0.0002581037,−0.0001217412]다. 별도 원래 t1734/t1741 CPU quiet 등록 기준은 [0.0001908990,−0.0001509463]이며 gain/mix가 정확히 같은 것을 `bind_registered_reference`와 원본 SHA로 확인했다. 이 등록 기준은 영상 없는 다른 physical quiet 상태에 대한 것이어서 현재 4시야/학습 몸/geometry 경로의 유효 중립으로 자동 승계할 수 없다. `20261101-{train,development}-unified-quiet-reference-counterfactual/`은 원래 상태/출력을 바꾸지 않고 첫 intent의 atanh 좌표에 해당 중립 차이 +0.0342103을 더해 출력만 반사실로 계산했다. 대표 좌우 근육 차 3조건을 직접 readout으로 계산한 결과와 최대 오차 8.2e-8이다.

직전 교사 20m chord 비교의 TRAIN54/known DEV27결정에서 원래/반사실 raw 첫 intent 평균은 각각 −0.055285/−0.021131, −0.050688/−0.016520이며 교사 경로 방향 부호 일치는 둘 다 0이다. 이는 별도 quiet 기준을 가정하면 중립 치우침이 일부 줄 수 있음을 보여주지만, 올바른 통합 graph 중립이라는 보증도 즉시 조향의 정답도 아니다. 이력 자체·기존 reference·모델 checkpoint를 변경하지 않았고 optimizer/차량 제어0이다. 실제 같은 조건·동일 graph의 정지 reference와 생리/시간/폐루프 검증, 시각 선행·가감속·변속·셋업·유효랩은 계속 필요하다.


## 2026-11-01 같은 통합 graph의 원본 정지 구간과 zero-task 반사실

`20261101-unified-{train,development}-zero-command-diagnostic/`은 각 원본 728/729결정의 네 시야·몸/상태와 합성 학습 geometry port/과거 body adapter를 유지하되, 전체 이력에 목표 yaw0·속도0을 주는 오프라인 반사실이다. 실제 기록의 이동 구간에도 0명령을 주므로 당시 차량에 실제로 적용된 명령이나 동일한 물리 폐루프를 뜻하지 않는다. 원본 native 308바이트를 다시 decode한 `20261101-unified-{train,development}-stationary-zero-task-audit/`에서 가장 긴 정지 구간은 index430–557, 각각 5.2478/5.2401초다. 정지 시작 뒤1초 이상 지난103결정의 첫 raw intent 평균/범위는 TRAIN −0.025622/[−0.025629,−0.025614], DEV −0.029522/[−0.029556,−0.029499]이며 두 번째 intent 평균은 −0.338368/−0.338312다. 속도0 목표의 가감속 intent는 별도의 제동 codec/정지 과제이므로 이것을 자동으로 중립이라고 해석하지 않는다.

같은 구간의 첫 intent atanh 평균을 다른 preview-command 재생 결과에서 사후 차감하는 가상 출력 비교도 수행했다. 다른 명령의 회로 상태가 동일한 것은 아니므로 실제 중립 교정이나 actor에 설치한 값은 아니다. 이후 교사 20m 경로 방향과의 부호 일치는 TRAIN0/54, DEV0/27로 변하지 않았다. 해당 교사 chord는 직접 steering 정답이 아니며 원본 물리 시각/정지 판정·폐루프도 미검증이다. optimizer/실제 차량 제어0. 통합 중립을 확정하려면 동일 실제 actor 조건에서 고정된 sensory/command 계약·반복 정지/출발과 시각 장면 변화를 따로 확인해야 하고, 사용자 목표의 시각 선행·몸되먹임·가감속/변속/셋업 및 빠른 유효랩은 계속 남아 있다.


## 2026-11-01 AC 비GUI 재기동 시도와 종료

확인 시 AC와 Steam 프로세스가 없고 기존 CSP status의 mtime은 약12시간 전이었다. Windows의 실제 Documents 경로 아래 기존 race.ini를 읽어 RACE model=bmw_z4_gt3, track=monza, REMOTE active=0을 확인했고 파일은 변경하지 않았다. 요청된 재시작의 일환으로 기존 `acs.exe`를 `Start-Process -WindowStyle Hidden`으로 실행했으나 PID11724는 곧 종료했다. Steam을 `-silent`/hidden으로 시작한 뒤 다시 실행한 PID13960도 종료했다. 직접 exit 관측을 위한 마지막 hidden 실행 PID24768은 exit code53으로 종료했다. Steam PID38796은 확인 시 실행/응답, 현재 게임 프로세스 없음; stale status의 값은 현 session 상태로 사용하지 않았다. CSP 로그는 초기 라이브러리 진입만 기록했고 코드53의 응용 프로그램 원인은 확인되지 않았다. 현재 live session·제어/학습 미시작, GUI activation/input·actuator0. `20261101-ac-hidden-start-attempt/`에 원본 실행 결과를 남겼다. 인증되지 않은 임의 race 설정 변경이나 UI fallback을 하지 않고, 지원되는 게임 세션 생성 경로·Steam/CSP 상태를 확인해야 한다. 전체 사용자 대전제와 시각/몸 feedback·가감속/변속·셋업·빠른 유효랩 목표는 미완으로 유지한다.


## 2026-11-01 세션 생성 경계와 통합 상태 reset 수정

앞선 최초 숨김 acs PID11724가 Steam PID38796을 시작했고, Steam이 AssettoCorsa launcher PID27272 및 child21544를 시작했다는 Windows process parent/creation 기록을 후속 확인했다. 확인 시 launcher와 Steam은 살아 있으나 주행 엔진 `acs.exe`는 없고 CSP session status는 12시간 이상 오래됐다. 따라서 과거의 AC 프로세스 없음은 당시 `acs.exe`/주행 엔진 관찰이며 이후 launcher process가 등장했다는 점을 구분한다. 설치된 launcher 화면 코드의 시작은 내부 `ac://start` 요청을 거쳐 설정을 구성한다. 외부에서 검증된 비GUI 세션 생성 API로 확인된 것은 아니며 사용자 금지에 따라 launcher/AC 화면 입력을 하지 않았다. exit53의 원인도 확정하지 않는다. 새 학습/제어 없음.

동시에 `UnifiedFlightSensorimotor.reset_episode()`가 추가한 상태형 `CausalGeometryVisualPort`의 정지 reference를 자동으로 비우지 않아 caller가 이를 별도 호출하지 않으면 이전 episode의 기준이 남을 수 있는 결함을 수정했다. 명시적 episode reset 때 reset_episode 메서드를 가진 visual context를 함께 reset하며, 일반 기존 context와 결정 간 연속 상태/gradient detach 동작은 유지한다. 신규 회귀 포함 관련24검사 통과. 원본 TRAIN728결정을 다시 재생한 `20261101-unified-train-causal-reset-verification/`의 output SHA `c0f582d5c295a5c635b7434197d12c540209f192d0fea618d69d13b870d1732e`와 readiness SHA는 이전과 동일하다. live timing·실제 actor/생리·시각 선행·가감속/변속·셋업 및 빠른 유효랩은 미검증이다.


## 2026-11-01 시각 관측의 중복 사용 방지

상태형 `CausalGeometryVisualPort`의 이전 구현은 정상 observe 이후 caller가 다음 프레임 observe를 빠뜨리면 동일 current를 다음 forward에서 다시 사용할 수 있었다. 이는 누락을 명시해야 하는 요구와 맞지 않아, 한 observe의 기하 보정은 첫 forward에서만 소비하고 재관측 없이 다음 forward가 호출되면 원래 visual context만 통과하도록 수정했다. 명시적 결측/중복/stale 경로와 episode reset은 유지한다. 신규 테스트는 비영 projection에서도 재관측 없이는 보정이 반복되지 않음을 확인하며 기존 관련24검사 통과. `20261101-unified-train-causal-single-use-verification/`의 원본 TRAIN728결정 출력 SHA `c0f582d5c295a5c635b7434197d12c540209f192d0fea618d69d13b870d1732e` 및 readiness SHA는 이전과 동일하다. 이는 모든 결정에 실제 observe가 제공된 오프라인 재생의 동등성일 뿐 live 동기·배포나 생리 검증이 아니다.

확인 시 Steam과 AC launcher는 실행 중이지만 주행 엔진 acs와 신선한 CSP heartbeat는 없다. 사용자가 금지한 AC 창 조작 없이 새 session을 만드는 검증된 API는 아직 확인되지 않았다. 실제 주행/학습 및 가감속·변속·셋업·빠른 유효랩은 계속 미완이다.


## 2026-11-01 몸 packet 반복 시 최초 receipt 보존

인과 기하 시각 경계에서 동일 native packet이 다시 복사돼 늦은 receipt로 전달될 때 기존 구현은 그 늦은 시간으로 body age를 계산할 수 있었다. 동일 packet의 최초로 관측한 immutable bytes/receipt를 episode 내 보존해 재복사는 최초 시각을 쓰도록 했고, 동일 packet bytes 변경·packet 역행·decision 시간 역행은 명시적으로 거부한다. 기존 50ms stale 처리·중복 camera 보정0·episode reset을 유지한다. 단위 테스트에서 늦은 시간으로 동일 packet을 재발행해도 stale로 판정하고 다른 bytes는 오류인 것을 확인했으며 관련25검사 통과. 기존 TRAIN/known DEV 원본 event는 결정 간 native packet 반복/역행이 각각0회였다. 원본 TRAIN728 전체 재생 `20261101-unified-train-native-first-receipt-verification/` output SHA는 이전과 동일 `c0f582d5c295a5c635b7434197d12c540209f192d0fea618d69d13b870d1732e`다. 새로운 실제 packet의 생성시각/카메라 exposure 동기·실시간 actor·생리·폐루프는 여전히 미검증이다. AC 주행 엔진 acs는 확인되지 않았고 CSP status는 오래됐다. 단일 감각운동 회로의 시각 선행·가감속/변속·셋업과 빠른 유효랩 목표는 계속 미완이다.


## 2026-11-01 인과 기하 포트의 영상 host 수신시각

통합 회로의 실험용 기하 시각 포트는 원본 네 시야 묶음의 host 수신시각을 명시적으로 받는다. 결정시각보다 미래이거나 경과 50ms 초과면 보정을 사용하지 않는다. 정지 기준을 수집 중이면 부분 수집을 초기화하며, 완성된 기준은 유지한다. 수신시각이 비유한수거나 누락되면 거부한다. 이 50ms는 해당 연구 경계의 조건이며 카메라의 물리 노출 시각·각 시야 동기·나머지 기본 시각 경로 전체의 신선도를 증명하지 않는다. 동일 capture는 기존대로 먼저 거부된다.

원본 TRAIN 728건의 host 수신 경과는 0.534~43.589ms이고 known DEV 729건은 0.312~60.940ms, 50ms 초과 1건이다. DEV index 175의 해당 capture 337은 이미 반복 capture로 거부됐다. DEV 원본 감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-development-camera-age-audit/`에서 준비 286, 최초 441이며 제어/학습 0이다.

TRAIN 원본의 통합 회로 전체 재생 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-train-camera-age-verification/`에서 output SHA256 `c0f582d5c295a5c635b7434197d12c540209f192d0fea618d69d13b870d1732e`, readiness SHA256 `865e821936f8cf82697741e2e78d29c62ef8b0223aed28fa78bd9eb030ec7aff`는 변경 전과 동일하다. 관련 검사 26개 통과. 기존 명령은 미검증 공학적 preview이며 actor의 완전한 시각 판단과 생리·물리시간·실제 폐루프/완주/랩타임은 미검증이다. 당시 AC launcher는 실행 중이나 acs 주행 엔진은 확인되지 않았다. 다음은 단일 회로에서 외부 preview 의존을 줄이면서 영상·몸 감각의 인과 기여와 실제 시간 계약을 검증한다.


## 2026-11-01 고정 영 명령에서 통합 회로 감각 개입

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-train-zero-task-sensory-interventions/`은 원본 TRAIN 728결정을 하나의 동일 비행 graph 상태로 재생하며, 외부 yaw·속도 명령을 0으로 고정했다. 고정 racing goal(100,1)은 그대로여서 모든 외부 목표가 없는 실험이 아니다. 네 조건은 원본, pooled scene 0, 기하 시각 보정 0, 과거 학습 몸 adapter 가중치 0이다. 각 조건은 동일 초기상태에서 전체 이력을 재생했으며 회로의 다른 감각 경로는 남는다. 0 대체는 분포 밖일 수 있다. 학습/차량 제어0.

기존 `20261101-unified-train-zero-command-diagnostic`과 결정시각 및 첫 조건의 모든 intent가 완전히 같았다. 두 intent 각각의 평균 절대 변화는 scene 0.003829·0.00000650, 기하 0.0000506·0.000000226, 몸 adapter 0.013024·0.006978이다. 이는 감각 경로의 수치적 인과 반응이며 생리 타당성·유용한 도로 예측·폐루프 운전 성능 증거가 아니다. scene 제거에도 기하 포트와 몸 입력은 남으며 몸 adapter 제거에도 다른 몸 경로는 남는다.

동일 기록의 기존 preview 명령 재생과 영 명령 재생의 두 intent 평균 절대 차이는 0.02742·0.57436이다. 두 명령 모두 독립적인 최적 조작 정답이 아니므로, 이 차이는 외부 명령에 대한 의존의 크기를 보여줄 뿐 운전 우열을 뜻하지 않는다. 단일 이력과 연속 표본으로 일반화할 수 없다. 원본 result의 `motion_intervention_scope`는 기존 라벨만 기록해 scene·몸 개입을 누락했으므로 상세 `zero_task_sensory_interventions`와 이 설명이 정확한 범위다. 후속 스크립트 라벨은 수정했다. 관련 회로 검사18개 통과. AC 주행 엔진 미확인, 물리 시각·생리·실제 자율주행 및 가감속/변속/셋업 적응·유효 완주/라인/랩타임 미검증. 다음은 레이싱 목표 입력의 의미와 영상·몸 기반 명령 생성을 통합 회로 안에 연결하고 독립 조건에서 검증한다.


## 2026-11-01 새 엔진 세션과 racing goal 경로 확인

이전에는 AC 주행 엔진이 없었으나 이후 PID34992가 실행됨을 관측했다. CSP read-only `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-engine-return-status/`에서 generation `19007968ULL:0.003981`, epoch0, Monza/BMW Z4 GT3, offline, live, menu, AI off를 확인했다. `.../20261101-engine-return-start/`의 검증된 API start 1회가 menu 밖 전환 성공, UI/actuator0이다.

`.../20261101-engine-return-native-sample/`은 native 100 copy, packet43508–43856, `.../20261101-engine-return-camera-probe/`는 전후좌우 각8장, hash 일치, render frame 증가·영상 비상수를 기록했다. 둘은 독립 수집이며 actor나 학습 입력으로 인증되지 않고 카메라 exposure/물리 시각 동기·센서 축·차량 응답은 미검증이다. camera probe 초기 장면의 밝기가 달라지는 것도 기록상 보이나 원인을 확정하지 않았다. `.../20261101-engine-return-post-sensors/` 마지막 확인에서 같은 generation live·offline·menu 밖·AI/camera off, 속도0.00061km/h, gas/brake0이다. 현재 실행 상태는 후속 작업에서 다시 확인한다.

`ManeuverCurrents.forward`에서 racing goal 첫 항은 0만 허용, 둘째 항은 원하는 속도의 외부 상한 검증에만 사용하며 조향 및 power 전류 계산에는 직접 들어가지 않는다. 따라서 이전 zero-command 실험의 고정 goal은 허용 경계이지 그 자체가 추가 신경 drive를 제공한 것은 아니다. 명령 0은 몸 속도와의 차이를 통해 power에 영향을 주므로 명령 경로가 제거된 상태는 아니다. 기존 상한 100/200 불변·위반 거부를 포함한 `tests/test_flight_maneuver.py` 11개 통과. 독립 센서 기반 목표 생성, 비행 생리, 폐루프 제동/변속/셋업과 유효 랩은 미검증이다. 다음 단계는 살아 있는 세션을 재확인하고 시간/제어 권한을 갖춘 시각·몸 실험을 수행한다.


## 2026-11-01 새 세션 카메라 초기 밝기 변화

`scripts/audit_camera_probe_brightness.py`는 `20261101-engine-return-camera-probe/`의 각 PNG를 result 및 archive 양쪽 SHA256과 대조하고 네 시야 RGB 평균 및 인접 프레임의 평균 절대 픽셀 차이를 계산한다. 원본/result/script hash와 결과는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-engine-camera-probe-brightness-audit/`에 있다.

8개 묶음에서 전방 RGB 평균은 0.6211→0.4942→0.4293→0.3292→0.3001이며 이후 0.3001 근처였다. 전방 마지막 세 인접 차이는 0.000322·0.000480·0.000236이다. 좌/우/후도 초기에 큰 밝기 변화를 보였다. 화면별 밝기 평균은 시각 운동 뉴런 출력이 아니며 실제 광학 노출, 정지 동시 근거, 조명 적응이나 카메라 원인을 독립 확인하지 않았다. 8묶음·한 세션이고 학습 불가 진단이다. 이 초기 변화가 운동 시각 및 정지 기준에 미치는 영향을 고정영상 대조와 함께 검증해야 하며 자동으로 프레임을 지우거나 신호를 보정하지 않는다.

이전 AI 교사 shadow는 현재 통합 회로가 아니고 paused·full-brake 등의 선행조건을 요구하므로 이번에 통합 actor 실험으로 실행하지 않았다. 마지막 session read-only `.../20261101-shadow-preflight-session/`: 동일 새 generation, live·offline·menu 밖·AI/camera off, 약0.00062km/h, gas/brake0. 실시간 물리 동기·통합 자율판단·가감속/변속/셋업 및 유효 완주/라인/랩타임은 미검증이다.


## 2026-11-01 새 카메라 초기 영상의 FlyVis 고정영상 대조

`scripts/replay_camera_probe_visual_contrast.py`는 원본 probe의 32개 PNG와 8개 metadata hash를 result 및 archive로 확인해 고정 저자 FlyVis의 4시야를 재생한다. 동일 초기 grey steady-state에서 원본 영상과 첫 화면 유지 대조를 8 batch로 실행하고 이전 영상 zero-order hold 및 기록된 CSP script clock 간격을 보존했다. 원본 간격은 0.1969~0.1998초로 기존 실시간 입력 기준 밖이며 물리 노출 시각이 아니다. retina projection은 기존 공학 근사다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-camera-initial-brightness-flyvis-contrast/`: 8묶음·내부 280 tick, T4/T5 개별 활동 최대 절대차 0.89052, 6축 기하 pool 변화 최대 [0.01569, 0.00798, 0.04006, 0.01843, 0.00705, 0.03396], 마지막 yaw 특징 차 0.02808이다. 11프레임 정지 기준을 만들기에는 표본이 부족하며, 밝기와 그 밖의 영상 변화가 함께 있는 단일 기록이므로 원인이나 실제 몸 회전과의 불일치를 입증하지 않는다. 대조군은 가상이며 physiology 검증이 아니다. 학습·차량제어0, 관련 visual/self-motion 검사11개 통과. 다음은 고정 장면의 독립 시간·정지 근거와 밝기만 바꾸는 개입을 확보하여 정지 reference 오염 및 통합회로 출력 기여를 검증한다. 기존 센서 물리 시각, 통합 actor·조향/가감속/변속/셋업·빠른 유효 완주/라인/랩타임 미검증.


## 2026-11-01 첫 망막 영상의 균일 밝기 개입

`scripts/replay_camera_probe_brightness_counterfactual.py`는 동일 원본 8묶음·CSP 간격·고정 FlyVis 및 첫 화면 유지 대조에 세 번째 조건을 추가한다. 시야마다 첫 721개 망막값 전체를 현재 원본 망막 평균/첫 평균 비율로 균일 곱하고 [0,1] 범위로 제한한다. 이 합성 조건은 장면 이동 없이 시야별 전역 밝기만 조정한 공학적 반사실이며 실제 카메라 노출/물리 조명 모델이 아니다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-camera-brightness-only-counterfactual/`: 균일 밝기 조건과 고정 영상의 T4/T5 최대 절대차 0.669423, 기하 yaw 특징 최대 절대차 0.039780 및 마지막 차 0.026467이다. 원본 변화 대 고정은 각 0.890521·0.040056·0.028079다. 최댓값은 시점/뉴런이 다를 수 있고 차이는 비선형이므로 비율을 실제 밝기의 기여율로 해석하지 않는다. 이전 8배치 결과와 새 12배치에서 원본시각·망막은 정확히 같고 공통 motion 최대차 4.21e-7, 특징 최대차 7.82e-9다. 원본/합성 공통의 전역 변화만으로도 해당 운동 특징이 달라질 수 있음을 보여주지만 이 기록에 실제 생리·물리 정지·노출 원인·실주행이 확인된 것은 아니다. 8개 표본으로 정지 기준 준비도 검증할 수 없다. optimizer/차량제어0. 다음은 11개 이상 실시간 기준에 대한 시간적으로 정합된 정지 원본과 밝기 단독·기하 대조를 확보해 통합 회로의 실제 출력 기여를 평가한다. 빠른 유효완주·레이싱 라인·랩타임 및 가감속/변속/셋업 적응 목표는 미완료다.


## 2026-11-01 정지 네 시야·몸 기록의 동기 수집 병목

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-fourview-body-capture/`에서 CSP 카메라 80ms·24묶음 및 각 native/SDK sample을 수집했다. 24묶음 중 네 시야 유효는 2개, 이후 원본 view age upper가 누적되어 마지막 전방 1.152초다. native 속도 0.000608~0.000620km/h이나 이미지 exposure와 물리 sensor 생성 동기는 검증되지 않았다. 요청 및 완성 원본은 보존했고 training_admissible=false, 모델 추론/제어/학습/UI0이다.

`scripts/audit_sensory_host_stage_timing.py` 및 `.../20261101-stationary-fourview-host-stage-audit/`는 원본 events 해시를 기록하고 시간순 원본을 감사한다. camera receipt→native receipt 중앙/95/최대15/29.6/47ms, native→SDK receipt 109/125/125ms, camera receipt→decision 125/140.85/141ms, decision 간격125/141/141ms다. 이 수집기에서 직렬 SDK 대기가 영상 80ms 생산보다 큰 구간이며 누적 backlog와 일관된다. 단일 run·호스트 수신 기준이므로 SDK 자체 단독 원인이나 physical exposure/발생 타이밍을 확정하지 않는다. 기존 고빈도 native/MMF collector의 분리 경로를 이 실험의 source/time 계약에 맞춰 사용하고, 결측과 최초 수신시각을 보존한 독립 감사로 다시 검증해야 한다. 실시간 기준을 완화하거나 이 실패 데이터를 actor 학습으로 승격하지 않는다.


## 2026-11-01 명시적 SDK 생략 native-only 진단

`scripts/record_ac_csp_sensory.py --native-only`는 SDK receive를 실행하지 않고 이벤트 SDK를 null로 남기며 actor_stream_created=false·training_admissible=false다. 기본 모드는 기존 SDK 경로를 유지한다. `scripts/audit_ac_csp_sensory.py`도 명시적 `--allow-native-only`에서만 SDK 누락을 승인하여 원본 bytes·camera archive·clock·전체 감각 pack을 재구성하고 기본 호출은 거부함을 확인했다.

동일 새 AC 세션의 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-native-only-fourview-capture/`: 24묶음 중 네 시야 유효23, SDK null24, camera receipt→decision 중앙/95/최대16/16/47ms, decision 간격63/109/110ms, 가장 오래된 view age 중앙/95/최대85.90/110.05/155.33ms, native 속도0.000608~0.000621km/h. 초기 clock uncertainty 46.25ms다. 기존 SDK 직렬 기록은 각각 유효2/24·receipt→decision 중앙125ms였지만 단일 비무작위 전후 실행이라 인과적 성능 일반화는 금물이다. `.../20261101-stationary-native-only-fourview-independent-audit/`에서 24개 원본 pixels/native/clock·감각 pack 재계산 오차0. 실제 노출/몸 물리시간, 1개 불합격 원인 및 독립 차량 제어 권한은 미검증이다.

관련 clock/native/multiview 묶음24검사에서 `test_runner_reset_and_checkpoint_restore_preserve_independent_source_state`가 회로 state 불변으로 1회 실패했고 단독 재실행은 통과했다. 실패의 원인은 확인되지 않았으며 원본 결과를 지우지 않는다. 다음은 해당 검사 불안정 원인 및 1개 지연을 확인하고, 실제 시간 계약 하에 통합 시각·비행 회로 정지 reference 검증으로 연결한다. 학습·차량 제어/UI0, 빠른 유효 완주/라인/랩타임 및 가감속/변속/셋업 미검증.


## 2026-11-01 native 정지 기록의 FlyVis와 인과 시각 reference

`scripts/replay_stationary_native_visual_port.py`는 `20261101-stationary-native-only-fourview-capture/`의 전체 archive·clock·pixel/native 감각 pack을 독립 재구성한 다음, 기록된 host decision 간격으로 FlyVis를 재생하고 원본 native bytes, camera receipt 및 네 시야 validity를 인과 기하 포트에 전달한다. index0 영상은 유효성 실패로 버리고 첫 유효 영상 전까지 회색을 유지한다. 고정 영상 대조는 첫 유효 영상 유지이며 입력 신선도 결측 시 이전 입력을 유지하는 오프라인 가정이다. 포트의 50ms host receipt 기준과 11개 정지 reference는 완화하지 않았다.

첫 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-native-flyvis-geometry-replay/`은 numpy scalar decision clock이 엄격한 파이썬 float 계약에 거부되어 실패했고 보존했다. 명시적 float 변환 후 `.../20261101-stationary-native-flyvis-geometry-replay-v2/`에서 24묶음·356 visual tick, index0 camera_unavailable, 1~11 reference 수집, index11 기준 성립 후 index12~23 포트 준비12개다. 최초 index11도 기존 규칙에 따라 보정0이다. 준비 이후 원본 yaw feature-reference 평균 0.002779 범위 [0.002750,0.002806]; 별도로 동일 1~11 평균을 뺀 첫 유효영상 고정 대조는 평균0.009545다. 고정대조도 시각망 시간상태가 변하므로 잔차의 원인을 장면이나 밝기로 단정하지 않는다.

실행 result의 `csp_intervals`라는 항목명은 잘못이며 실제 값은 manifest에 명시된 host decision 간격이다. 후속 스크립트 이름을 `host_decision_intervals`로 고쳤고 기존 원본은 수정하지 않았다. 시각 생리·물리 exposure/회전 시각·차량 조향 연결 및 전체 통합 회로 폐루프는 검증하지 않았고 학습/제어0이다. 다음은 고정영상 상태의 시간 안정화와 reference 선택의 영향을 통제하며 통합 graph 출력을 비교한다. 조향·가감속·변속·셋업과 빠른 유효 완주/레이싱 라인/랩타임 목표는 미완료다.


## 2026-11-01 장시간 정지 후보의 시각 reference 반복

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-native-only-long-capture/`에서 새 48묶음 수집, 네 시야 유효47·SDK 미수집·actor/학습 자격 없음이다. `.../20261101-stationary-native-long-flyvis-geometry-replay/`는 원본 archive/native/clock 감각 재구성 후 실제 FlyVis와 인과 포트 재생: 746 tick, 첫 유효성 실패1, reference 수집11, index12 이후 준비36개다. host decision 경과 3.641초.

index1~11 yaw 특징 평균 기준은 원본 −0.127390·첫 유효영상 유지 대조 −0.152806, 마지막 값은 −0.124041·−0.144105, 잔차는 +0.003348·+0.008702다. 고정영상 대조 마지막 1초의 yaw 특징 범위 8.3e-9로 그 구간에서 거의 일정하나, 앞선 초기 11개 평균과 다르다. 이전 24묶음에서도 비슷한 현상이 있어 reference가 초기 상태 과도응답에 민감할 수 있다. 이는 2개 순차 진단 기록이며 실제 물리 회전/노출 시간은 미검증, 첫 영상 유지도 인공 대조다. 현재 11개 평균은 기존 합성학습 checkpoint의 입력 계약이므로 근거 없이 기준을 바꾸거나 결과를 새 actor에 승격하지 않는다. 시각 state 안정화 또는 초기 기준을 다시 학습할 때 감도·지연·회전 반응을 고정영상·양방향 움직임·다양한 밝기에서 동시에 검증해야 한다.

기존 불안정 회로 test의 난수 seed 0~11을 같은 별도 process에서 재현했을 때 모두 상태가 바뀌어 실패하지 않았다. 실패 원인은 여전히 미확정이며 통과 주장으로 덮지 않는다. optimizer/차량제어0. 비행 유래 단일 감각운동 회로·몸 기울기/회전/하중/타이어 피드백·조향/가감속/변속/셋업·빠른 유효 완주/라인/랩타임은 미완료다.


## 2026-11-01 정지 reference 창 사후 민감도

`scripts/audit_synthetic_reference_windows.py`와 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-reference-window-sensitivity/`는 원래 합성 구형 질감7개, 정지 및 ±0.23rad/s 각41결정의 보존된 FlyVis 결과/hash를 사용한다. 동일한 최초 11개 정지 구간에서 mean11과 마지막3 평균을 각각 빼고 마지막10개 결정의 yaw pool 평균을 비교했다. 정지 절대 잔차는 mean11 0.01023~0.01590, 마지막3 0.00241~0.00346; 정지 대비 좌우 상대 부호는 양 방법 7/7 서로 반대다. 같은 장면의 반사실 간 상대값은 공통 baseline이 거의 상쇄되므로 이것을 완전 독립적인 회전 일반화 검증으로 해석하지 않는다. 합성 일부는 기존 실험에서 사용된 사례다.

두 정지 원본의 마지막 특징 잔차도 기존 mean11 원본/고정 각각 0.002781/0.009521, 0.003348/0.008702에서 마지막3 사후 대조 −0.000193/0.000320, 0.000246/0.000384다. `.../20261101-recorded-reference-window-sensitivity/`에 원본 해시와 수치를 보존했다. 두 원본은 진단 기록이고 실제 물리 정지·노출 동기, 실시간 회전 및 여러 차/날씨의 생리/폐루프는 확인되지 않았다. 기존 checkpoint는 mean11 입력에 맞게 학습돼 있고 단지 이력에 마지막3을 사후 대조한 것이므로 기존 포트·배포 모델을 변경하지 않았다. 다음은 학습용 사례와 독립된 새 기준/조건에서 동일 graph·원래 가중치의 방향/정지 반응과 threshold 지연을 전부 측정하고 필요시 명시적 재학습한다. optimizer/차량제어0. 초파리 감각운동 회로·몸 되먹임·시각 선행·가감속/변속/셋업 및 빠른 유효 완주/라인/랩타임은 미완료다.


## 2026-11-01 마지막3 기준의 통합 graph 반응

`scripts/evaluate_unified_geometry_last3_known_textures.py`는 기존 고정 graph·시각 경계 checkpoint와 알려진 합성 질감 863102/105/106/107, 정지·±0.23rad/s 응답을 그대로 사용하되 기존 first11 mean 대신 last3 of first11 mean을 쓴다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-geometry-last3-known-texture-eval/`과 기존 `.../20261101-unified-geometry-yaw-known-texture-eval/`의 outputs를 비교했다. 동일 4×2×3×41×2 shape, zero projection 대조 bitwise 동일, 학습 projection 조건의 최대 절대차 0.00043432다.

4개 모두 양 기준에서 정지 대비 양 회전 조건의 turn 부호가 설정된 상대 목표와 맞았다. mean11→last3 상대 turn은 863102 [0.003673,−0.002812]→[0.003652,−0.002841], 863105 [0.004951,−0.004252]→[0.004909,−0.004287], 863106 [0.003853,−0.003494]→[0.003821,−0.003524], 863107 [0.003825,−0.002329]→[0.003807,−0.002353]다. 정지 raw first intent 마지막10 평균은 −0.04845~−0.04759에서 −0.04810~−0.04723 범위로 바뀌었으나 이는 보정된 조향 중립이 아니며 상대부호만으로 운전 기능을 인증하지 않는다. 외부 10m/s 명령·가상의 zero yaw 몸·누락 view body context를 사용한 기존 진단이므로 영상만의 자율 판단도 아니다. 이 질감들은 알려진 개발 사례이고 물리 노출/생리/실제 폐루프 미검증이다. optimizer/차량 제어0, 기존 mean11 runtime과 checkpoint는 수정하지 않았다. 다음은 독립 조건·정지/회전 반응의 기준 및 몸 감각 효과를 확인한 뒤 필요하면 학습 계약을 새로 정해 재학습한다.


## 2026-11-01 새 합성 질감의 기준 비교

기존 scripts/docs/config에서 사용하지 않은 texture seed863201의 합성 구면 정지·±0.13rad/s 41결정을 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-coherent-yaw-texture-863201-speed013/`에서 고정 FlyVis로 새로 만들었다. `.../20261101-unified-geometry-863201-speed013-paired-reference/`는 학습된 기존 checkpoint·통합 graph·동일 상태 초기화로 zero projection, mean11 projection, last3 projection을 비교했다. 새로운 합성 단일 질감이며 실제 환경 일반화 증거가 아니다.

정지 대비 좌우 raw turn 차이는 zero [+0.000219,−0.001101], mean11 [+0.002660,−0.003532], last3 [+0.002649,−0.003553]이다. 두 기준 모두 상대 부호는 가설적 회전 목표와 일치했고 회전속도는 합성값이다. 두 projection 조건의 모든 intent 최대 차 0.00035466이며 정지 raw first intent 마지막10 평균은 zero −0.047042, mean11 −0.047541, last3 −0.047187이다. 정지 raw 출력 자체를 조향 중립으로 검증한 것은 아니다. 외부 10m/s 명령·합성 몸 yaw0·view body context 결측은 기존 진단과 같고 자율 판단/물리 노출/생리/실제 폐루프가 아니다. optimizer·차량제어0.

원본 result의 status가 복사된 코드에서 `completed_known_texture_evaluation`으로 기록됐지만 이 실행의 role `new_synthetic_diagnostic`, source hash와 seed는 새 데이터다. 저장 결과는 보존하고 후속 스크립트의 status 문자열은 수정했다. 이 진단은 실제 회로와 변경 후보의 정지·회전 반응을 확인한 것일 뿐 기존 runtime을 변경하거나 최종 목표를 달성한 것이 아니다.


## 2026-11-01 새 합성 질감에서 시각·몸 yaw 상호작용

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-geometry-863201-speed013-body-matched/`와 `...-body-opposite/`는 기존 seed863201 고정 영상 ±0.13rad/s 이력과 같은 graph·학습 geometry 포트를 유지하고, 0.4초 이후 합성 몸 yaw를 각각 영상 회전과 같은 수치 또는 반대 수치 ±0.13rad/s로 설정했다. 기존 fixed 결과 `...-paired-reference/`는 몸 yaw0이다. 속도10m/s 외부 명령, 합성 나머지 몸, view history 결측 조건은 같다.

mean11 기준의 정지 대비 좌/우 raw turn은 body0 [+0.002660,−0.003532], matched [+0.015487,−0.016317], opposite [−0.010133,+0.009289]다. last3는 각각 [+0.002649,−0.003553], [+0.015476,−0.016338], [−0.010144,+0.009268]로 거의 비슷하다. matched에서도 기하 projection 0인 비교군은 [+0.013044,−0.013889], opposite에서는 [−0.012571,+0.011722]였다. 이 단일 합성 반사실에서 몸 yaw 경로의 출력 영향이 기하 보정보다 커 충돌 부호에서 가설적 시각 목표의 부호가 뒤집혔다. 그러나 opposite은 일부러 충돌시킨 가상 감각이지 실제 sensor 고장/물리 가능한 상황이나 오류 검출 자격이 아니다. 순수한 운동 반사인지, 하중·관성/실제 차량에서 유익한지, 생리 및 폐루프는 확인되지 않았다.

이전 fixed 실행의 코드에는 body-yaw-mode가 0이어서 실제 입력에 영향이 없던 ±0.23 상수가 남아 있었고 이번 paired 테스트 전 ±0.13으로 바로잡았다. 모든 checkpoint/source 증거와 이전 결과를 보존했다. 새로운 학습·차량제어/UI0. 다음은 실제 시각/몸 시간·부호를 검증하고 몸 기울기·가감속 및 타이어/셋업까지 같은 회로에서 대응을 시험한다.


## 2026-11-01 마지막3 정지 기준의 합성 yaw 시각 포트 학습

`scripts/train_unified_geometry_yaw_last3.py`는 동일 비행 graph와 나머지 checkpoint를 고정하고, 첫 11개의 정지 이력 중 마지막3 평균을 기준으로 빼는 공학적 기하 yaw 포트만 합성 텍스처863101/103/104에서 12회 Adam 갱신했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-synthetic-geometry-yaw-last3-fit/`에 프로그램/소스 hash·진행 상태·가중치·출력·결과를 보존했다. 포트 checkpoint SHA256 `16a28701427044134bea91ea5d4635edfdfa41112da45fb241ef97d1a0006f8a`은 기존 mean11과 호환되지 않는 last3 계약이며 actor_qualified=false다. 기존 배포 및 모델을 교체하지 않았다.

합성 학습 손실 0.000191747→0.000144768, 학습 3질감의 정지 대비 ±0.23rad/s yaw 상대 turn은 학습 전 2/3에서 원하는 양방향 부호, 이후 3/3이다. 앞서 별도 진단에 이미 사용됐던 알려진 개발 seed863201 ±0.13에서 정지 대비 상대 turn은 [+0.000219,−0.001101]→[+0.001240,−0.002144], 해당 진폭에 비례시킨 가상 목표 MSE 2.5116e-5→1.5887e-5다. 이 개발 질감은 이번 optimizer에 들어가지 않았으나 이전에 검토돼 독립 최종 시험이 아니다. raw 정지 turn은 약 −0.0471로 남는다. fixed body yaw0·외부 속도10m/s·view body context 결측·인위적 목표이며 생리/실제 도로 선행 예측·가감속·폐루프 성공을 뜻하지 않는다. 사용자 원본의 fit_allowed=false 제한은 유지했으며 실제 기록 학습/차량 제어0. 인과 포트·통합 회로 관련10검사 통과. 다음은 독립 합성 질감·몸 회전 반사실과 새로운 실제 촬영을 이용해 고정 checkpoint를 검증하고 실시간/조작 자격을 확인한다.


## 2026-11-01 새 합성 질감에서 고정 checkpoint 비교

기존 scripts/docs/config에 없던 seed879301로 정지·±0.13rad/s 이력41개를 새로 생성한 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-coherent-yaw-texture-879301-speed013/`을 사용했다. `scripts/evaluate_unified_geometry_last3_checkpoint_comparison.py`의 `.../20261101-unified-geometry-879301-checkpoint-comparison/`은 동일 graph·last3 reference·외부 10m/s 및 합성 body yaw0 조건에서 zero, 과거 mean11 학습 checkpoint, 새 last3 학습 checkpoint를 평가한다. old는 30회, new는 12회 학습해 갱신량과 baseline 계약이 모두 다르다.

정지 대비 상대 turn은 zero [+0.000614,−0.000180], old [+0.003832,−0.002666], new [+0.001975,−0.001229]이며 세 조건 모두 가설적 방향 부호와 일치한다. 진폭 비례 가상 목표 ±0.005652의 MSE는 각각 2.7663e-5, 6.1133e-6, 1.6544e-5로 이 한 사례에서 새 후보가 old보다 나빴다. 정지 raw turn은 각각 −0.046895/−0.047024/−0.046949다. 새 포트 학습은 더 짧았으므로 이 차이를 reference나 학습 방법 한 가지의 인과 효과로 돌릴 수 없다. 새 seed는 이 실행 전 해당 합성 포트 fit 및 development에 사용되지 않았지만 단일 합성 구면이고 실제 서킷, 생리·물리 시각·폐루프 검증이 아니다. optimizer/실제 제어0, checkpoint 배포0. 다음은 동일 갱신 예산의 사전등록 비교와 다양한 실제 움직임·몸 감각 계약을 검증한다.


## 2026-11-01 동일 30회 학습 예산의 정지 기준 비교

`scripts/train_unified_geometry_yaw_last3_30.py`는 기존과 동일 3개 합성 학습 질감·graph·초기값·Adam 0.1을 사용해 last3 reference 포트를 처음부터 30회 학습했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-synthetic-geometry-yaw-last3-fit-30/`, checkpoint SHA `093345b9d78b516a3c55804714e1a446fef057e7d4271d5e331aa5d67aae896c`, 학습 loss 0.000191747→0.0000875324, 실제 차량 제어0·actor_qualified=false. 이전 mean11 30회 fit 최종 loss 0.0000879223. 서로 다른 reference에서 계산한 학습 목표의 값이라는 점은 유지한다.

`.../20261101-unified-geometry-879301-equal-budget-comparison/`에서 이미 직전 실험에 사용된 개발 seed879301 ±0.13에 같은 last3 reference, graph, 영상·가상 몸 yaw0·외부 10m/s를 주었다. zero/old30/new30의 정지 대비 turn은 [0.000614,−0.000180]/[0.003832,−0.002666]/[0.003836,−0.002668], 가상 목표 MSE는 2.7663e-5/6.1133e-6/6.1024e-6이다. old/new 전체 intent 최대차 3.49e-6, 정지 raw 평균 약 −0.04702다. 12회 후보의 성능 차이는 이 30회 비교에서 거의 사라졌으나 단일 개발 사례로 일반화할 수 없고 reference의 인과 우열이나 실주행 성능은 입증되지 않는다. 원본 result의 `known_development_only=false` 및 status는 복사 코드의 라벨 오류이며 rows의 role `previously_examined_synthetic_development`가 정확하다. 원본 결과는 수정하지 않고 후속 코드 라벨을 바로잡았다.

기존 배포 모델은 변경하지 않았고 물리/생리/센서 동기·실제 통합 폐루프/유효 랩은 미검증이다. 다음은 동일 예산 기준을 동결한 뒤 아직 사용하지 않은 여러 조건과 실제 측정 근거를 확보해 방향·몸/가감속 반응을 평가한다.


## 2026-11-01 실제 기록에서 last3 시각 경계의 단일 graph 진단

`scripts/replay_unified_flight_candidate_last3_diagnostic.py`는 기존 actor 포트를 변경하지 않는 오프라인 전용 subclass로 같은 11개 정지 인정과 host camera/native freshness를 지키며 마지막3 평균을 쓴다. 체크포인트 `093345b9...`의 명시적 last3 reference 계약과 hash를 확인한다. 합성 11개 원본 가상 packet에서 마지막3 평균·episode reset을 확인했다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-train-last3-original-replay/`에서 기존 TRAIN 후보 원본728결정·네 시야·몸·동일 graph와 미자격 고정 preview를 재생하고 원래 `.../20261101-unified-train-camera-age-verification/`과 비교했다. `.../20261101-unified-train-last3-comparison/`에 원본 SHA와 통계를 보존했다. 결정시간, zero projection 대조 및 index441 준비 전 결과는 정확히 일치, 준비는 285결정/최초441로 같다. 준비 후 두 intent 평균 절대차 1.427e-5/4.691e-8, 전체 최대차 1.562e-5/1.788e-7, 첫 intent 부호 변경0이다. reference와 포트 weight가 동시에 바뀐 비교이므로 개별 원인 분리는 불가하다.

원본은 fit_allowed=false·optimizer/차량 제어0, preview는 미검증 공학 가설, 물리 exposure/시간 및 생리·실제 actor·유효 완주/레이싱라인/랩타임 미검증이다. 결과는 이 이력에서 후보가 큰 출력 변화를 주지 않았다는 관찰이며 운전 정확도의 향상/열화를 증명하지 않는다. 몸 기울기·하중·가감속/변속/셋업 적응과 실제 시간 기반 인과 검증을 계속한다.


## 2026-11-01 알려진 실제 개발 이력의 last3 후보

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-development-last3-original-replay/`는 이미 여러 번 검토한 개발 원본729결정의 네 시야·몸 native 및 고정 preview를 통합 graph에 last3 checkpoint로 재생한다. `.../20261101-unified-development-last3-comparison/`은 기존 `.../20261101-unified-flight-development-causal-geometry/`와 원본 SHA 및 clock을 대조했다. 두 run의 zero projection, index441 이전은 수치가 정확히 같고 readiness 최초441/286결정이다. 두 intent의 준비 후 평균 절대차는 5.572e-6/2.768e-8, 전체 최대차 6.374e-6/1.788e-7, 첫 intent 부호 변경0이다. TRAIN에 이어 DEV에서도 해당 두 변경의 합친 출력 차이는 작지만 단 두 연속 이력으로 일반화하거나 정확도 향상이라 부를 수 없다.

고정 preview는 시각 예측 gate 미통과, 두 이력의 목표 위치/경로는 actor에 주지 않았다. 이미 알려진 개발 이력·실제 노출/센서 동기·생리/물리 조작 미검증, fit_allowed=false·optimizer/차량 제어0, 새 checkpoint actor 미배포다. 다음은 실패한 시각 선행 경로를 개선하고 몸 회전/하중·가감속·변속/셋업을 같은 회로에 연결해 적합한 독립 데이터와 폐루프 검증으로 이어간다.


## 2026-11-01 실제 영상 이력의 T4d 채널 개입

기존 FlyVis 공개 가중치에서 T4d의 대비 선호 불일치가 남아 있다. `scripts/replay_unified_t4d_intervention.py`는 원본 TRAIN 네 시야 각 721열의 T4d 활동을 전부 0으로 한 반사실 이력을 만들고, 같은 망막·다른 운동채널·몸·미자격 고정 preview와 동일 graph를 유지하여 scene pool과 기하 시각 포트 양쪽에 전달한다. 원본 조건과 개입 조건은 독립 상태에서 전체728결정을 재생한다. T4d 전체 제거는 실제 생물학적 손상이나 contrast 수정이 아니고 분포 밖일 수 있으며 회로 유용성 및 운전 개선을 암시하지 않는다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-train-t4d-ablated/`의 원본 조건은 `.../20261101-unified-train-camera-age-verification/`의 원본과 clock 및 output bitwise 동일했다. 두 intent의 개입 평균 절대차 0.00014168/0.000001426, 최대 0.00059332/0.000004411이다. 조향 intent 부호 변경은 index291의 +0.0000492→−0.0000895 한 건으로 둘 다 0 근처다. 학습/차량제어0, 원본 fit_allowed=false, 물리 노출/센서 동기·생리·독립 방향 정확도·실제 주행은 미검증이다. 다음은 T4d 대비 불일치의 원본 생리 근거 및 공간/시간 반응을 구분하고 영상·몸 감각의 인과 정보를 같은 회로에서 개선한다.


## 2026-11-01 현재 조향 설정의 read-only 준비 검사

`scripts/preflight_current_steering_calibration.py`는 vJoy acquisition 없이 native 몸/AI, background AC, controller identity 및 profile SHA를 읽고 기존 정지 보정 guard를 확인한다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-steering-contract-read-only-audit/`: generation `19007968ULL:0.003981`, native packet739863, speed0.000617km/h·gas0·brake0·native AI false, one background engine·vJoy free·identity 일치다. 현재 profile SHA `8265c290dca2adb2a371bf6ec9b9a86f4d1b64a245045851cf311eaadd5b6f74`는 기본 `configs/ac-axis-probe.json`의 기존 `4e2c...`와 달라 기본 검사를 통과하지 못한다. 또한 현 brake0은 정지 full-brake guard 실패다. UI/장치 획득/actuator0.

현재 profile hash는 과거 `20261031-after-restart-steering-calibration`의 ±0.6 보정 profile과 같고 `20261031-after-restart-steering-independent-audit`는 당시32원본의 native 중위값 0/+0.84375/−0.84375/0, SDK 0/+270/−270/0, 정지 조건을 확인했다. 해당 결과는 다른 epoch/generation에서 얻은 정지 표본으로 현재 주행 중 전달 또는 전 범위 선형성/물리 각도·교사 표적을 증명하지 않는다. 현재 프로필 hash를 명시한 진단용 별도 `.../20261101-steering-contract-read-only-audit/diagnostic-profile-candidate.json`과 원본 config/과거 보정/독립감사 해시를 준비했으나 기존 config는 수정하지 않았다. 이후에는 새 generation의 제동 유지·권한 확인과 보호된 속도 범위의 움직이는 전달 계약을 명시적으로 검증해야 한다. 기준을 무시하여 조작하거나 과거 레이블을 fit으로 승격하지 않는다.


## 2026-11-01 새 generation의 정지 조향 보정

`scripts/calibrate_ac_steering_brake_bootstrap.py`는 기존 보정기의 full-brake 선행조건을 우회하지 않는다. 먼저 fresh offline·AI off·정지·무가속·background·프로필/장치 identity를 검사하고 vJoy open의 neutral/full-brake 요청을 보존한다. 원래 native packet에서 brake>0.98을 확인한 뒤에만 기존 정지 보정 guard와 0/±0.1/0을 실행한다. timeout/상태 불일치 시 중단하고 finally에서 full-brake 요청 후 장치를 해제한다. UI 자동화 없음.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-steering-brake-bootstrap-small-calibration/`: generation `19007968ULL:0.003981`, bootstrap 전 brake0에서 요청 후 brake 확인, 32표본, SDK 중위 0/+44.999996/−44.999996/0, actuator 요청34, foreground0. 첫 독립 audit `...-audit/`은 기존 스크립트가 ±0.6만 허용해 ±0.1을 거부했고 실패를 보존했다. `scripts/audit_stationary_steering_calibration_explicit.py`는 명시적 amplitude와 모든 단계 요청, 원본 308바이트/순서/AI-off/제동/배경을 검증한다. `...-audit-v2/`에서 native 중위 0/+0.140624985/−0.140624985/0, 최대속도0.018556km/h, 32표본 통과. SDK 수치는 저장값의 중위와 비교이며 물리 단위 검증이 아니다.

종료 직후 `.../20261101-steering-brake-bootstrap-postflight/`는 live/offline/AI off/camera off·brake1·gas0·속도0.000485km/h, 독립 vJoy 상태1(FREE)을 확인했다. actuator/mapping 상태는 후속 실행에서 재검사한다. 관련 vJoy/native9검사 통과. 정지에서 ±0.1만 확인했으며 움직이는 조향 전달, 전체 범위 선형성, 물리 yaw/노출 동기 및 교사 라벨 자격을 주장하지 않는다. 학습/actor 배포0. 다음은 출발과 재정지의 제어·안전 계약을 확보하고 실제 속도별 조향 반응을 독립 검증하며, 파리 시각·몸 되먹임/가감속·변속·셋업·유효 랩 목표를 계속한다.


## 2026-11-01 새 세션 정지 조향 세 크기

기존 ±0.1에 이어 동일 generation `19007968ULL:0.003981`에서 보호된 full-brake 상태로 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-steering-brake-bootstrap-mid-calibration/`의 ±0.3 및 `.../20261101-steering-brake-bootstrap-large-calibration/`의 ±0.6을 별도 실행했다. 각 0/양/음/0 요청·32표본·background/AI-off/native 제동 조건을 `...-mid-calibration-audit/`, `...-large-calibration-audit/`에서 원본 바이트 및 단계 요청으로 각각 확인했다. 각 실행은 장치를 해제했고 UI0이다.

요청 크기 0.1/0.3/0.6의 native 양·음 중앙값은 ±0.140624985/±0.421874970/±0.843749940, SDK는 ±44.999996/±134.999985/±269.999969다. native 원시값/요청은 각 1.40624985/1.40624990/1.40624990으로 세 측정점은 비례와 일치한다. 중간 모든 값, 히스테리시스, 움직임 중의 전달, 조향 물리 각도/회전 반응은 검증하지 않아 실제 조작 디코더나 교사 표적 자격으로 승격하지 않는다. 세 실행 최대속도는 각각 0.01856/0.01766/0.01575km/h다. `.../20261101-steering-range-postflight/`에서 offline·AI/camera off·brake1·gas0·속도0.000483km/h, 별도 vJoy status1(FREE) 확인. 새 학습/actor 제어0.

다음은 제한된 속도에서 조향의 동적 부호·지연과 제동/복귀 계약을 구체화해 독립 감사하고, 같은 초파리 시각·몸 feedback 회로의 조향/가감속 및 레이싱 라인·랩타임 학습으로 연결한다.


## 2026-11-01 저속 조향 펄스·제동 진단

`scripts/probe_low_speed_steering_response.py`는 새 독립 simulator 진단이다. fresh offline/background/AI-off·정지/brake·neutral·프로필/장치, 고유 vJoy 및 타이어 이탈0을 확인하고 기어 up 후 native raw gear2를 확인했다. 최대8km/h, 가속 최대4초, 조향 ±0.05 각각0.35초 및 중립을 실행하고 즉시 full brake·native 정지 확인 뒤 장치를 해제한다. 상태/권한/속도/기어/이탈 이상 시 중단 및 full brake 종료다. UI 자동화와 실제 하드웨어 제어는 없다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-steering-response-probe/`: completed, 68회 actuator 요청, 최대 native 속도3.216km/h, full-brake 정지 확인·vJoy FREE, actor/학습 자격0. `scripts/audit_low_speed_steering_response.py`와 `...-audit-v2/`는 native 원본72개 bytes/hash/필드/packet 순서, 별도 native AI bytes, request 범위, 탈선0 및 종료 정지를 검증했다. 요청 후 host 100ms 이상 구간에서 양/음 조향 각각7표본의 native steer 중앙 +0.0703125/−0.0703125, localAngularVel[2] 중앙 −0.000368884/+0.000357669, 속도 중앙3.169/2.962km/h다. 부호 반응 관찰이며 localAngularVel 축의 실제 좌표 부호/단위와 timestamp의 물리 발생 시각, 요청과 센서 간 정확한 지연은 미검증이다. 각 phase 한 번뿐이어서 방향별 전반적인 전달/반복성은 증명하지 않는다.

첫 `...-audit/`은 서로 다른 packet 두 개의 host monotonic receipt가 같은 값이어서 엄격한 시각 증가 검사에서 중단됐다. 원본은 보존하고 v2는 packet 증가·clock 비감소를 요구해 동시 표시를 유지했으며 요청/수신을 같은 host monotonic domain으로 비교한다. `...-postflight/`는 offline/AI off/camera off/brake1/gas0/속도0.000423km/h, 결과 파일은 vJoy status1(FREE)을 기록한다. 몸 감각 원본에서 지도/위치 진단을 actor에 주지 않았고 neural actor는 관여하지 않았다. 다음은 반복 양방향·다른 속도의 명시적 시각/몸 timestamp 및 조향 부호 계약을 검증하고 시각·비행 통합 회로의 실제 폐루프, 가감속·변속·셋업·유효 완주/라인/랩타임으로 이어간다.


## 2026-11-01 역순 저속 조향 반복

`scripts/probe_low_speed_steering_reverse_order.py`는 직전 종료 후 이미 1단인 상태를 요구하고, 이전과 같은 offline/background/AI-off·정지 제동·동일 profile/device·속도 최대8km/h·탈선0 조건에서 −0.05→중립→+0.05→중립 순서를 사용했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-steering-reverse-order-probe/`에서 최대 속도3.170km/h, 제동 후 정지·vJoy FREE, UI0·actor/학습 자격0이다.

`...-audit/`는 72개 native 원본 bytes·요청·AI off·packet 순서·탈선0과 정지를 재검사했다. 요청 시작 후 host 100ms 이상 양/음 구간 각7표본의 native 조향 중앙값은 +0.0703125/−0.0703125, localAngularVel[2]는 −0.000205593/+0.000292592, 중앙 속도2.912/3.124km/h다. 기존 양 먼저 실험의 −0.000368884/+0.000357669와 비교하면 두 순서에서 조향과 각속도 성분의 부호 관계는 같지만 크기는 달랐다. 한 회씩의 속도/장면·상태가 다른 진단이며 효과 크기나 반복 일반화를 확정하지 않는다. 물리 축/단위·노출/몸 발생 시각·정확한 지연 및 폐루프도 미검증이다.

`...-postflight/`에서 offline·AI/camera off·brake1·gas0·속도0.000368km/h, 결과의 vJoy FREE를 확인했다. 모델 학습/actor 제어0. 다음은 센서의 방향 계약과 시각 동기 검증 및 학습 자격을 확보해 초파리 감각운동 회로의 실제 조향·가감속, 변속·셋업·유효랩 검증으로 진행한다.


## 2026-11-01 저속 native heading 대 회전 성분 사후 감사

`scripts/audit_low_speed_native_angular_heading.py`는 두 저속 조향 이력의 308-byte 원본 hash를 확인하고, 해당 원본의 absolute `heading` 차분과 `localAngularVel` 세 성분의 사다리꼴 적분을 host receipt 간격으로 비교한다. 절대 heading은 오직 사후 진단이며 actor 입력이 아니다. 두 이력 각각72패킷·유효69 간격; 동시/5ms 미만 및 150ms 초과 간격은 별도 보존·제외한다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-native-heading-axis-audit/`.

셋째 angular 성분의 heading 차분에 대한 원점 통과 비례 회귀는 각각 0.3449/0.4302, 단위 배율을 가정했을 때의 residual RMS 0.000215/0.000194, 부호를 반전하면 0.000324/0.000319다. 처음/역순 각각 독립성이 없는 연속 시계열이며 heading 범위는 약0.00320/0.00316에 불과하다. timestamp는 물리 발생 시각이 아니라 host receipt이고 여러 원인(진동·노이즈·표본·좌표·배율)을 구분하지 못하므로 축의 물리 부호·단위나 부정확의 원인을 확정하지 않는다. 학습/차량 제어0. 다음은 더 충분한 방향별 실제 회전량과 물리 동기를 가진 독립 저속 진단을 통해 시각/몸의 좌표 계약을 검증하고 단일 비행 감각운동 회로의 조향·가감속·변속·셋업 및 레이싱라인/유효 랩 학습에 연결한다.


## 2026-11-01 더 큰 저속 자극과 native angular 축

`scripts/probe_low_speed_steering_higher_excitation.py`는 이전과 같은 offline·AI-off·background·정지 full-brake·first gear·profile/device·tyres-out0 확인을 유지하고 최고8km/h 한도 내 목표5km/h에 도달한 뒤 −0.1/중립/+0.1/중립을 각0.6초 적용, full-brake 후 정지/장치 해제를 확인했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-steering-higher-excitation/`: 실제 최대5.119km/h, native118패킷, 요청106, 정지 확인·vJoy FREE, actor/학습/UI0.

`...-all-axes-audit/`는 원본/native-AI bytes·요청·순서·탈선0·정지를 재검증했다. 각13표본에서 양/음 조향 요청의 native steer 중앙 +0.140625/−0.140625, localAngularVel[1] 중앙 −0.023192/+0.024686, [2] 중앙 +0.000289/−0.000546이다. `.../20261101-low-speed-higher-excitation-heading-audit/`의 host 간격 heading 차분 대 [1] 적분의 원점 통과 회귀는 −1.00476, residual RMS는 같은 부호 0.001134, 반대 부호 0.000157; heading 범위0.01639다. 이는 단일 실행의 후향적 상관·호스트 시간 근사이며 물리 노출/발생 시각, 실제 단위·왼쪽 방향 부호나 일반화의 인증은 아니다. absolute heading은 진단만 쓰고 actor 입력0이다.

기존 `configs/ac-virtual-body-v1.json`의 body yaw 채널은 이미 native index1을 사용한다. 이전 ±0.05 기록에서 [2]가 세 성분 중 더 일관돼 보였다는 관찰은 작은 회전에서의 제한된 결과였으며 이 새 자극에서 일반화되지 않는다. 코드를 뒤집거나 sensor contract를 새로 인증하지 않는다. `.../20261101-low-speed-higher-excitation-postflight/`: offline/AI·camera off, brake1/gas0, speed0.000496km/h, 결과 vJoy FREE. 다음은 같은 조건의 독립 반복과 시각/몸의 시간 정합 및 부호를 검증하고, 통합 회로의 주행 조작/가감속·변속·셋업·라인/유효 랩 목표로 이어간다.


## 2026-11-01 5km/h 자극의 역순 반복

`scripts/probe_low_speed_steering_higher_opposite_order.py`는 직전 5km/h 진단과 같은 제한·guard·±0.1/0.6초를 사용하되 양 조향을 먼저 적용했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-higher-opposite-order/`에서 최대5.187km/h, 114 native packet, full-brake 정지·vJoy FREE, UI0·actor/학습0. `...-audit/`가 원본 byte/request/AI-off/타이어 이탈0/정지를 재검증했다. 양/음 구간 각13개 native steer 중앙 +0.140625/−0.140625, angular[1] 중앙 −0.025309/+0.024163, angular[2] +0.001180/−0.000187이다.

`...-heading-audit/`: 111 유효 host 간격에서 heading 차분 대 angular[1] 적분의 원점 통과 회귀 −0.99298, 반대 부호 residual RMS 0.00015149; 직전 반대 순서 값은 −1.00476 및 0.00015686이었다. 원래 몸 yaw는 이미 angular[1]을 쓰며 config를 변경하지 않았다. 두 단발 저속 실험이 같은 경향을 보이지만 동일 세션·한 차/셋업·host timestamp이며 센서 실제 발생/노출 동기, 물리 단위·좌표 부호 및 생리/주행 자격을 인증하지 않는다. absolute heading은 사후 검증 외 actor 입력0이다. `...-postflight/`: offline/AI·camera off, brake1/gas0/속도0.000493km/h, 결과 vJoy FREE. 다음은 네 시야와 몸 신호의 물리 시간/방향 정합 및 통합 회로 폐루프의 조향·가감속/변속/셋업과 유효랩을 검증한다.


## 2026-11-01 저속 조향의 네 시야·몸 동시 기록과 정지 prefix

`scripts/probe_low_speed_steering_fourview.py`는 이전 제한 저속 ±0.1 조향 진단에 소유 CSP 네 시야 recorder를 추가하고 native receipt 전의 150ms 이내 후보를 선택한다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-steering-probe/`: 62그룹, native118개 중 선행 영상 연계116/결측2, 최대 view age139.4ms; `...-host-alignment-audit/`에서 bundle·62그룹 metadata/pixels·clock bound/host 순서 검증. 출발 전 native 표본은3개라 정지 reference 수집에 부족했다.

`scripts/probe_low_speed_fourview_stationary_prefix.py`는 가속 전 1.2초 동안 실제 native 정지·brake를 감시하며 네 시야를 유지한다. `.../20261101-low-speed-fourview-stationary-prefix-probe/`에서 총 영상78그룹, 출발 전 영상이 있는 몸30표본·서로 다른16그룹(초기 선택 host 50ms 이하16그룹)·시간 범위1.217초, 최대속도0.000468km/h다. 전체 native147개 중 영상 연계145·결측2, 관측된 서로 다른77그룹의 bytes/RGB hash 및 host 시각 재구성 `...-host-audit/` 통과; view age 중앙66.16ms/최대135.56ms, clock bound 불확실성3.79ms다. `...-body-audit/`: 원본 native/AI bytes·요청·탈선0·정지 통과, 최고5.185km/h, 양/음 angular[1] 중앙 −0.02535/+0.02362다. 이번 기록은 준비 조건을 수치 확인한 것이지 FlyVis/기하 reference가 실제로 성립했다는 검증은 아직 아니다.

`...-postflight/`: offline/AI·camera off/brake1/gas0/속도0.000447km/h, 결과의 vJoy FREE, UI0. 촬영 및 native 물리 발생 시각/노출의 정확한 동기·운동 생리·교사 라벨 및 actor/학습 자격은 미검증, neural 제어0이다. 다음은 보존된 16개 정지·이후 움직임 네 시야를 FlyVis와 같은 통합 회로에 연속 재생하여 결측/중복·기준과 시각/몸 반응을 감사하고, 가감속/변속/셋업 및 빠른 유효랩 목표로 이어간다.


## 2026-11-01 실제 정지→저속 네 시야 FlyVis/인과 포트 재생

`scripts/replay_low_speed_fourview_visual_port.py`는 `20261101-low-speed-fourview-stationary-prefix-probe/`의 host 및 몸 독립 감사가 해당 events SHA를 가리키는지 확인하고 묶음 원본 이미지 hash·native prefix를 검증한다. 호스트 perf-counter 몸 receipt 간격과 FlyVis 4시야 state를 연속 유지하고, 초기 카메라 결측은 회색 초기값, 중복/후속 결측은 관측 시 마지막 영상 hold로 취급한다. 포트는 실제 capture ID·원본 camera/native receipt·정지 조건·50ms 검사에 따라 중복을 새 영상으로 취급하지 않는다. 이는 시각 부품과 포트의 오프라인 replay이며 통합 flight graph 또는 실제 actor 제어는 아니다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-flyvis-port-replay/`: 몸147결정, 77개 서로 다른 영상, 시각 내부1396 tick, 정지 reference 최초 index20·첫 포트 준비 index21, ready66·repeated camera68·reference unavailable11·camera unavailable2. 첫 준비 시 native 속도0.000467km/h로 출발 전이다. `.../20261101-low-speed-fourview-geometry-yaw-audit/`는 port rows/events SHA를 보존하고 준비 상태에서 같은 호스트 표본의 |native angular[1]|>0.01인18개를 집계했다. 몸 부호 +8/−10, 기하 yaw 특징은18개 모두 양수 [0.003337,0.010646], 부호 일치8/18, Pearson r 0.753이다. 한 연속 저속 구간의 상관된 표본이며 부호 불일치는 reference 오프셋·시간차·광학/운동 변수 등 원인을 식별하지 못한다. 단순 상관만으로 생리 방향 선택성이나 정확한 차량 조향을 주장하지 않는다.

원본 물리 exposure/몸 발생 시각 및 yaw 물리부호는 미검증, diagnostic 한 이력·학습 불가·optimizer/차량 제어0다. 다음은 동일 원본의 조향/회전 사건과 고정영상 대조를 분리해 기하 특징의 정지 기준 편향·시간 응답을 검증하고, 검증된 시각/몸을 같은 flight graph의 실제 폐루프로 연결한다. 가감속·변속·셋업·빠른 유효 완주/레이싱라인/랩타임은 미완료다.


## 2026-11-01 실제 저속 네 시야·몸의 단일 graph 재생

`20261101-low-speed-fourview-flyvis-graph-input/`은 직전 원본 FlyVis 재생에 각 결정의 held retina를 추가 저장한 독립 재실행이다. 원본과 clocks 일치, motion 최대차 5.37e-7, capture/reference 결과는 동일하다. `scripts/replay_low_speed_unified_flight_graph.py`는 검증된 영상·native 및 이전 같은 세션 native static identity 증거를 사용한다(각 sample의 static identity를 별도로 획득한 것은 아님). 첫 두 영상 결측과 그 사이 0.529초 간격은 제외하고 첫 유효 영상 index2에서 새 graph episode를 명시적으로 시작하여 나머지145결정을 원래 간격으로 재생했다. 실제 순서가 front/left/right/rear임을 원본에서 확인했으며 후속 코드에도 명시 검사를 추가했다. 각 시야 render 하한 전의 original body receipt가 camera receipt 기준50ms 안에 없으면 해당 history는 None이며 580개 시야 시점 중253개가 결측이다. 원본 입력을 채워 넣지 않는다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-unified-graph-diagnostic-v2/`: 같은 단일 비행 graph, 동일 연속 입력 및 body adapter에 고정 yaw0·speed0 외부 명령을 주고 intact, 기하 projection0, 현재 몸 yaw 값만0(원래 mask·과거 view body·기하 입력 유지)을 별도 episode로 비교했다. geometry reference 준비66, 최초 원본 index21은 앞선 독립 시각 재생 readiness와 일치한다. 두 intent의 평균 절대차는 기하 제거 0.0001230/0.000000482, 현재 yaw0 0.0005174/0.000003098, 최대 절대차는 각각 0.0002300/0.000001103, 0.0026344/0.00001553이다. 두 차이는 개별 경로를 가상으로 제거했을 때의 반응이며 모든 몸/시각을 제거한 것이 아니고 0 대체는 분포 밖일 수 있다.

초기 결과 `...-diagnostic/`과 이전 세션 static evidence를 명시한 v2의 output SHA는 동일하다. source의 카메라는 dashboard-front geometry이고 합성 geometry checkpoint 학습은 exterior-front 등록을 썼으므로 전이는 미검증 가설이다. 시각 물리 노출/몸 발생 시각·생리·실제 actuator 명령과 폐루프는 미검증이다. 외부 명령을0으로 고정했으나 실제 움직인 차량의 당시 의도가 아니고 actor의 시각 예측도 구현되지 않았다. 원본 training_admissible=false·optimizer/차량 제어0, 새 graph/가중치 배포0. 다음은 정지/양방향 회전에서 포트의 기준 편향과 실제 시각/몸 시간 인과를 해결하고 actor가 영상·몸으로 조향/가감속을 결정하도록 발전시키며 변속·셋업·빠른 유효 완주/라인/랩타임을 평가한다.


## 2026-11-01 실제 저속 기록의 last3 기하 포트와 graph

`scripts/replay_low_speed_unified_flight_graph_last3.py`는 앞선145결정의 동일 source/graph/몸 감각·0 명령과 253개 view-body 결측을 유지하고 기존 mean11 포트 대신 명시적으로 last3 후보와 그 학습 checkpoint를 넣는다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-unified-graph-last3/`: reference 최초21·ready66, 기존 포트와 readiness 및 zero projection output bitwise 같다. 동일 이력 mean11과 last3 후보의 첫/둘째 intent 평균 절대차 0.00013550/0.000000521, 최대 0.00018298/0.000000834다. 이 비교는 기준과 포트 weight 둘 다 바꾸므로 각 원인을 분리하지 못한다.

이전에 동일 이력에서 보고한 기준(ready 및 |native angular[1]|>0.01)으로 `.../20261101-low-speed-last3-geometry-comparison/`에 재계산했다. 몸 yaw 양8/음10인18결정에서 기존 기하 yaw는 양18, 부호 일치8/18; last3는 양7/음11, 부호 일치15/18이고 범위 −0.004419~+0.002890, Pearson 0.753이다. 상수 기준 변화가 부호 경계를 옮긴 사후 분석이고 상관계수·분포가 생리 선택성/주행 정확도를 증명하지 않는다. 영상은 dashboard-front, synthetic 포트 훈련은 exterior-front 등록이므로 광학 전이 미검증이다. 이력/기준/임계값을 이미 관찰한 뒤의 단일 저속 실험, 신호 간 물리 발생/노출 시간과 yaw 물리 부호, 조향 actuator 및 실제 폐루프는 미검증이다. actor/학습/차량제어0, 기존 runtime 미변경. 다음은 고정한 last3 기준을 관찰에 쓰지 않은 새 조향 순서/속도·조명에서 사전 지정하여 평가하고, 몸/시각 통합이 가감속·변속·셋업과 유효 랩에 주는 기여를 확인한다.


### 2026-11-01 역순 조향 새 이력의 단일 회로 후보 비교
수집 전 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-reverse-plan/`에 역순 펄스·점수 기준·프로그램/가중치 해시를 기록했다. 기존 분석에 사용한 데이터와 다른 새 이력이지만 같은 AC 세션·차량·트랙이며 독립 환경 검증은 아니다. `.../20261101-low-speed-fourview-prefix-reverse-probe/`는 정지 prefix 후 −0.1/0/+0.1/0 조향, 최고 5.117km/h, 수집 후 정지 확인·vJoy FREE다. 정지 전 출발 전 native 32건 중 50ms 이내 서로 다른 영상 17개다. `...-body-audit/`와 `...-host-audit/`는 native 146건, 영상 연계144건·고유77그룹 해시를 확인했다. 물리 노출·몸 발생 시각의 동기는 입증하지 않는다.

동일 영상 이력을 `...-flyvis/`에서 FlyVis로 재생하고 `...-graph-mean11/`, `...-graph-last3/`에서 하나의 순환 비행 회로에 각각 고정 yaw/속도 0을 넣어 144결정을 재생했다. 두 대조의 zero 기준은 bitwise 동일하고 준비 판단도 같다. 사전 지정 조건인 준비된 표본 중 |native angular[1]|>0.01은 18건(양9/음9)이다. mean11 기하 yaw는 양18/음0, 몸 부호 일치9/18; last3는 양12/음6, 일치15/18이었다. 두 기하 값의 몸 yaw 상관은 각각 0.77271이며 연속 표본이므로 독립 시행으로 해석하지 않는다. `...-comparison/result.json`에 결과를 저장했다. 한 구간이며 기준과 학습 가중치가 함께 달라지는 비교다. last3가 일부 개선된 진단 근거이지 생리적 방향·시간 정합·예측·폐루프 운전의 검증이나 다른 조건 일반화는 아니다. 영상/몸 처리 지연 및 이전 몸 감각 275/576건 결측을 그대로 보존했다. 행위자 배포·학습·회로 출력의 차량 제어는 없었다.

목표는 별도 운전 두뇌가 아니라 초파리 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결하는 하나의 감각운동 시스템이다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 다음에는 물리 시각 정합과 다른 조건 검증을 강화하고, 고정 명령 대신 허용된 감각에서 판단하는 연속 회로 경로와 실제 가감속·변속·셋업 적응 및 유효 완주를 검증해야 한다.


### 2026-11-01 반복 영상의 호스트 신선도 한정 유지 후보
기존 `CausalGeometryVisualPort`는 새 영상 capture에서 계산한 보정을 한 결정에만 사용하며, 동일 capture를 다음 몸 결정에서 다시 받으면 아직 호스트 기준으로 신선해도 `repeated_camera_capture`로 보정을 0으로 돌린다. 이는 사용자 요구의 연속 감각운동 경로에서 관찰된 공학적 시간표본 문제다. 시각 신호가 계속 변해야 하는 생리학적 사실이나 실제 주행 오류로 단정하지 않는다.

`scripts/replay_low_speed_unified_flight_graph_fresh_hold.py`와 `src/flybrain_drive/flight_geometry_fresh_hold.py`는 기존 구현/결과를 변경하지 않는 실험 후보를 둔다. 마지막 3개 정지 기준과 기존 가중치를 유지하고, 최초 계산된 보정만 동일 capture의 원본 수신 시각에서 50ms 이하인 동안 반복 사용한다. 매 결정마다 원본 영상 시각·네 시야 유효성·원본 native 바이트와 packet·몸 수신 신선도를 검사하고, 결측/만료 시 보정을 끊는다. 재사용은 새 영상·새 FlyVis 반응이 아니며 실제 노출 시각이나 신경 시간응답의 검증이 아니다. 명시적 reset에서 기준과 보관값을 지운다. 기존 및 신규 회귀 6개 통과.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-prefix-reverse-graph-fresh-hold/`는 같은 역순 실제 영상/몸 이력과 기존 graph·last3 가중치·고정 yaw/속도0으로 144결정을 재생했다. 기존 준비 66결정에서 81결정으로 15결정 증가; 새 신호가 생긴 것이 아니라 호스트 50ms 안의 같은 값을 유지한 것이다. `.../20261101-low-speed-fourview-fresh-hold-comparison/result.json`에서 기하 보정0 대조는 bitwise 동일, 두 intent 평균 절대차 4.81e-6/2.77e-8, 최대 2.07e-5/1.79e-7이다. 단일 저속 기록이며 의도 출력의 차이를 운전 개선으로 해석하지 않는다. 학습·actor·차량 제어는 없었다. 다음은 반복 및 카메라/몸 실제 시간 정합, 신경동역학 재계산과 비교, 감각 기반 직접 조작 경로 및 가감속·변속·셋업과 폐루프 랩 검증이다.


### 2026-11-01 같은 영상의 진행 중인 운동 시각 상태
원본 `.../20261101-low-speed-fourview-prefix-reverse-flyvis/responses.npz`를 만드는 스크립트는 body decision 사이에 직전 영상을 유지하면서 FlyVis 상태를 원본 host 간격의 tick으로 연속 진행한다. 따라서 새 카메라 영상이 없다는 사실은 운동 시각 신경 상태도 고정됐다는 뜻이 아니다. 기존 fresh-hold 후보는 동일 capture의 최초 기하 feature를 보관하므로 이 상태 변화를 반영하지 않는다.

`FreshEvolvingGeometryPort`는 기존 후보의 원본 capture receipt·50ms·시야·native 바이트/시간 검증을 그대로 사용하면서, 통과한 반복 영상에 대해서는 호출자가 연속 진행한 현재 FlyVis 활동으로 기하 보정을 다시 계산한다. 원본 영상의 시각을 새로 만들거나 신선도를 갱신하지 않는다. 호출자가 실제로 FlyVis state를 진행해야 한다는 전제이며 실시간 연결은 아직 없다. 원본 및 후보 검사 7개 통과.

같은 144결정·동일 graph/가중치·고정 zero 명령을 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-fourview-prefix-reverse-graph-evolving/`로 재생하고 hold와 비교했다. 두 후보 모두 적용81회, 기존에는 끊겼던 15회의 현재 FlyVis 기하 yaw는 고정값과 15/15 다르며 평균 절대차 0.000560, 최대 0.002359다. 통합 graph intent 평균 절대차 1.10e-6/8.49e-9, 최대 1.40e-5/8.94e-8; 기하 제거 대조 bitwise 동일이다. 증거 `.../20261101-low-speed-fourview-evolving-vs-hold-comparison/result.json`. 이는 같은 입력 영상 아래 내부 상태 진화를 보존할 필요를 보여주는 한 오프라인 반사실 비교이며 생리 정확도·물리 노출 정합·운전 성능 개선을 입증하지 않는다.

진단용 실제 주행 자료는 학습 자격이 없고 기존 합성 yaw 학습을 반복하는 것은 새로운 주행 성능 근거가 아니다. 이번 optimizer·배포·차량제어0. 다음은 시각/몸의 발생 시간 및 다른 조건의 재현을 확인하며 통합 회로에서 허용된 영상·몸 감각이 조향·가감속 명령으로 이어지는 학습과 폐루프 검증을 진행한다. 변속·셋업 적응과 유효 완주·레이싱라인·랩타임 최적화는 미완료다.


### 2026-11-01 몸 수신 간격과 시야별 이전 몸 결측
기존 역순 원본은 native receipt 간격 중앙42.33ms, 영상 렌더 시작 하한→수신 중앙27.02ms였다. 각 시야 렌더 하한보다 엄격히 앞서고 영상 수신보다 50ms 이내인 이전 몸을 고를 때 576개 결정·시야 중 275개 결측이었다. 50ms를 느슨하게 하거나 누락 감각을 만들지 않고 수신 밀도를 늘리는 비교를 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fast-body-fourview-plan/`에 미리 기록했다.

`scripts/probe_low_speed_fourview_prefix_fast_body.py`는 기존 동일 offline·AI off·백그라운드 안전 조건, 정지 prefix, −/0/+/0 조향, 8km/h 상한을 유지하고 sleep을 40→10ms로 변경했다. 조작 요청 반복도 빨라지므로 순수 샘플링만 다르게 한 대조가 아니다. `.../20261101-fast-body-fourview-probe/`는 최대5.110km/h, 정지·vJoy FREE, 카메라73그룹 및 native464건, 독립 host/body 감사 통과다. 한 차·한 세션의 비동시 기록 두 개이므로 부하·장면·시간적 상관과 요청 주기 차이를 고려해야 한다.

사전 지정 기준으로 native 수신 간격 중앙42.33→12.35ms, 결정별 시야 결측275/576(47.7%)→54/1848(2.9%). 결정별 반복을 줄여 각 카메라 그룹 첫 관측만 세면 고유 시야 결측156/308(50.6%)→12/292(4.1%)다. `.../20261101-fast-body-fourview-comparison/unique-view-result.json`과 재계산 스크립트 `scripts/compare_per_view_body_receipts.py`에 원본 SHA를 보존한다. 영상 render 하한→수신 50ms 초과 사례도 있으므로 높은 샘플링으로 모든 결측을 없앨 수는 없다. host 시간 관계이며 센서 발생·노출 물리 동기 증거는 아니다.

`.../20261101-fast-body-fourview-flyvis/`에서 464결정·73영상의 원본 FlyVis 상태를 연속 재생했다. `.../20261101-fast-body-fourview-graph-evolving/`는 첫 유효 영상 이후462결정에 기존 하나의 graph와 evolving 후보를 적용해 이전 몸 결측54/1848을 재현했다. 시각 보정 준비246회, 기하 제거 시 intent 평균 절대차3.68e-5/1.46e-7, 현재 몸 yaw만 0이면 5.18e-4/3.10e-6이다. 고정 yaw/속도0 반사실; 실제 신체/영상 시간 정합, 생리, 다른 주행 조건, 학습이나 actor 제어/폐루프 성능은 입증하지 않는다. 다음은 독립 반복과 live 경로에서 이 수신 밀도의 부하·신선도 검증, 시각·몸 기반 직접 운전 판단과 가감속·변속·셋업 적응, 유효 완주·라인/랩타임 학습이다.


### 2026-11-01 외부 maneuver 명령을 중립화한 감각 경로
기존 단일 `UnifiedFlightSensorimotor`는 외부 desired yaw·desired speed를 필수로 받았다. 새 `sensory_decision`/`sensory_training_decision`은 별도 운전자 모델 없이 기존 graph·시각·몸·반복 state와 readout을 재사용한다. 기존 정확한 `ManeuverCurrents`에 yaw0, 원본 유효 몸 속도를 desired speed로 넣어 외부 turn 전류와 속도 오차를 중립화한다. 이는 생리 회로 측정이나 학습된 속도 판단이 아니라 원래 입력 경계의 실험적 중립화다. 목표 상한은 기존 계약대로 유지하며 실제 속도가 상한을 넘으면 기존 경계가 거부한다. 현재 몸 yaw와 body adapter·시각 입력은 유지한다. 결측/음수/비유한 속도와 수정된 command boundary는 거부하고 원래 그래프를 변경하지 않는다. 수치상의 중립 입력이 중립 출력이나 안전을 보장하지 않는다.

원본 동일 관측·시간에서 명시적 yaw0/몸속도 명령과 새 감각 경로의 state/output 및 training gradient를 비교하고 기존 maneuver 전류의 yaw0·tonic0.05를 검증했다. 통합 회로·시각 freshness 관련 총12검사 통과. 원본 `.../20261101-fast-body-fourview-probe/`의 네 시야·몸 이력과 같은 고정 graph/시각 가중치로 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fast-body-fourview-graph-sensory-neutral/`는 462결정을 끊김 없이 재생했다. 시각 기하 포트 제거 평균 intent 차 3.71e-5/1.45e-7, 현재 몸 yaw 제거 5.18e-4/3.25e-6. 의도는 첫 축 −0.06854~−0.02606, 둘째 −0.58031~−0.33832로 462건 모두 음수였다. `.../20261101-fast-body-fourview-sensory-vs-zero/result.json`에서 외부 desired speed0과의 평균 차는 각각 0.000632/0.03882다. 이 결과는 현재 회로가 이 조건에서 양방향 조향·가속을 배운 증거가 없고, 음의 종방향 의도에 치우쳤다는 진단이다. 직접 조작 명령이나 실제 제동 결과로 해석하지 않는다.

원본은 저속 펄스 단일 이력, host 시간·물리 발생/노출 동기 및 생리 미검증이며 학습 허용 자료가 아니다. optimizer·actor·차량 제어0. 다음은 본 경로가 가감속과 양방향 조향을 만들도록 적격 학습 자료/목표와 감각의 인과 기여를 확보하고, 변속·셋업 적응·유효 완주·라인/랩타임을 실제 폐루프로 검증한다.


### 2026-11-01 외부 maneuver 중립 경로의 합성 절대 조향 학습 — 실제 기록에서 편향 잔존
사전 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-sensory-absolute-yaw-plan/plan.json`에 기존 합성 세 질감(863101/103/104), 고정 비행 graph 및 30회 Adam, 정지/양방향 인공 목표, 알려진 개발 질감863201 ±0.13rad/s 평가와 미배포를 기록했다. `scripts/train_unified_sensory_absolute_yaw_synthetic.py`는 기존 graph·readout·몸 adapter를 고정하고 geometry projection과 시각 포트 bias만 학습한다. 외부 회전 명령은0, 속도 오차는 원본 몸 속도로 중립화한다. 합성 목표 정지0/−yaw+0.01/+yaw−0.01은 차량 조향 보정이나 초파리 생리 정답이 아니다. 실제 저속 기록은 학습에 쓰지 않았다.

`.../20261101-sensory-absolute-yaw-synthetic-fit/`은 실제 30회 업데이트를 완료했다. 기록된 각 업데이트 직전 합산 손실은 최초0.0069753, 18회0.00011745, 마지막30회0.00028082로 최저 뒤 상승했다. 사전 지정대로 마지막 checkpoint만 사용했다(SHA `89fffc314ddc6ffd2ad0d23329cc8c37952d953acdbd134ee410d81b3edb46a5`, schema `flybrain.experimental.sensory_neutral_visual_yaw.v1`, actor_qualified=false). 이전에 검토됐지만 optimizer에는 쓰지 않은 합성 개발 자극의 마지막10 평균 정지/−yaw/+yaw는 전 [−0.04704,−0.04682,−0.04814], 후 [+0.00787,+0.01087,+0.00391]이다. 가상의 절대 목표 [0,+0.00565,−0.00565] MSE는 0.002257→0.0000602, 정지 대비 상대방향은 양/음이지만 절대 출력은 여전히 세 조건 모두 양수다. 원본 `outputs.npz`에서 재계산했다. 기하 weight와 bias가 함께 바뀌고 bias 변화 norm11.11이므로 단일 인자의 인과 효과가 아니며 이 개발 자극은 독립 최종 시험이 아니다.

`scripts/replay_low_speed_sensory_synthetic_candidate.py`는 fit 결과 체크섬과 schema를 검증하고 같은 실제 저속 원본의 시각·몸·시간 및 기존 graph로 미배포 후보를 재생한다. `.../20261101-fast-body-fourview-sensory-synthetic-candidate/`와 `.../20261101-fast-body-sensory-synthetic-before-after/result.json`: 462결정 전/후 첫 intent 평균 −0.03014→+0.02225, 양수0→444, 음수462→18. 특히 속도>3km/h 256결정의 후 조향 의도는 모두 양수다. 전체 둘째 intent 평균 −0.34425→−0.34318이며 양수0으로 남았다. 양방향 회전 펄스에도 출력이 편향돼 실제 조향·가속 성공이 아니며 합성 편향 보정이 다른 한쪽 편향으로 옮겨졌다는 제한이 있다. 영상/몸 물리 시각·기울기 및 생리·장면 전이·변속/셋업·폐루프 유효랩은 미검증; 차량 제어/배포0. 학습 코드와 후보 재생 컴파일 통과, 저장 결과를 원본 배열로 재산출했다. 다음은 이 실패를 기준으로 단일 graph의 양방향 시각/몸 기여를 강화하는 적격 자료와 목표를 마련하고 가감속·변속·셋업 및 실제 라인/랩 검증으로 진행한다.


### 2026-11-01 합성 절대 조향 학습의 편향 원인 분리
직전 후보에서 기하 projection과 시각 포트 bias가 함께 학습돼 실제 기록 출력 변화의 원인을 알 수 없었다. `scripts/audit_sensory_training_component_effect.py`는 같은 원본 사건/host audit/body audit/FlyVis 배열·결정 clock·readiness와 고정 graph에 대한 원본 및 후보 replay의 geometry-zero 조건을 비교한다. 두 replay 프로그램의 기록된 SHA를 검사하고, 두 조건에서 geometry projection은 모두 0이므로 이 설정에서 두 그래프 간 입력 차이는 학습된 시각 bias다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-sensory-bias-versus-motion-component-audit/result.json`.

실제 저속 기록의 native 조향 펄스 음48·양49 관측에서 geometry-zero 후보−원본 첫 intent 평균은 각각 +0.054594/+0.054617이었다. 같은 후보에서 geometry intact−zero는 각각 −0.0000809/−0.0000126으로 훨씬 작다. 정지110개는 bias 차 +0.045270, geometry 효과 −0.0000116이다. 값은 해당 전체 반복 이력의 recurrent state까지 포함한 반사실 비교이며, 움직임 전체의 상대적 정확도나 생리 기능, 인과적 경로 최적화를 입증하지 않는다. 펄스 조향은 진단 개입이며 최적 steering 정답이 아니다. 실제 관찰된 편향 이동의 대부분이 입력 편향 변화와 함께 남아, 이를 운동 시각 처리 능력의 향상으로 볼 근거가 약하다.

합성 학습은 전방 차 외부 가상 mount를 명시한 구형 질감, 실제 diagnostic은 앞 차내 화면을 쓰며 두 설정의 시야 방향/투영은 같지만 실제 config에는 동일한 카메라 원점 계약이 없다. 서로 다른 mount/장면이 전이 실패의 가능한 원인이나, 이 비교만으로 그 원인 하나를 확정할 수 없다. 호스트 receipt는 물리 노출 시각이 아니며 actor·학습·실차/시뮬레이터 제어0, 결과는 미배포다. 다음은 실제 camera mount/가림과 원본 연속 시각·몸 감각 조건을 분리하여 정량화하고, 운동 시각의 방향 선택과 몸 되먹임이 회로의 양방향 조작에 인과적으로 기여하도록 학습 및 가감속·변속·셋업/유효 랩을 검증한다.


### 2026-11-01 동일 세션 차내/외부 네 시야 보존 프레임 비교 — 두 수집 종료 검사 실패
사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-camera-mount-pair-plan/`에 동일 세션의 paused/full-brake 네 시야를 순차 수집하고 원본 좌표와 RGB를 비교하기로 기록했다. CSP lifecycle API `.../20261101-stationary-camera-pair-pause/`는 pause 목표 상태를 관측했다(UI/차량 actuator0). 이후 dashboard `...-dashboard/`와 exterior `...-exterior/`에서 각 5그룹을 저장했으며 두 producer terminal은 completed·오류0·bundle SHA 확인이다. 그러나 두 부모 실행은 모두 촬영 직후 `Fresh paused stationary full-brake AI-off FREE state required` 종료 guard를 통과하지 못해 status=failed다. 어떤 단일 조건이 실패했는지는 당시 각 조건을 개별 기록하지 않아 확정할 수 없다. 나중에 fresh 세션이 paused·AI off·camera off·gas0/brake1임을 확인했지만 이것은 실패 순간을 증명하지 않고 실패를 성공으로 바꾸지 않는다.

`scripts/audit_stationary_camera_mount_pair.py`는 실패 원본을 명시적으로 보존한 채 각 frame metadata, revision, camera mount geometry, archive 파일 해시 및 동일 generation/car/track을 검증했다. `.../20261101-stationary-camera-pair-frame-audit/result.json`의 각 세 번째 그룹 비교에서 앞 시야 exterior 중심은 dashboard보다 전방 2.8446m·오른쪽 0.3611m·위 0.0953m, 거리 2.8690m이고 look 방향 내적 0.9999999다. 좌/우/뒤 카메라 중심 차는 기록된 float에서0m였다. 해당 LDR RGB의 채널당 평균 절대 차이는 앞 51.73/255, 좌 0.76/255, 우 0.93/255, 뒤 0.88/255다. 순차 촬영이라 렌더 잡음·장면/시각 변화 가능성을 배제하지 못한다. 실제 이미지에서 dashboard 전방에는 대시보드/차체 기둥이 보이고 exterior 전방은 열린 도로다. 이는 시야와 장착 차이가 명확하다는 프레임 증거이지 두 실패 run의 자격 승격이나 학습 전이 실패의 단독 원인 증명은 아니다. 물리 노출 동기·생리·차량 폐루프 및 시각 선행 예측은 미검증, optimizer/actor/차량 조작0이다. 마지막 관측 기준 세션은 같은 generation paused·오프라인·AI/카메라 off·brake1/gas0이며 live 학습은 없다. 다음은 실패 종료 guard의 개별 상태 원인을 새 원본 진단으로 파악하고, 명시된 카메라 조건별 연속 기록·운동 시각/몸 감각의 인과 검증과 가감속·변속·셋업/유효랩을 계속한다.


### 2026-11-01 카메라 종료 검사 원인과 수정
직전 두 실패는 유지했다. 별도 관측 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-dashboard-camera-postflight-observed/`에서 새로 발생한 종료 검사 실패 시, producer는 완료됐지만 session camera_active가 같은 요청 ID와 함께 true였다. 다른 정차·브레이크·장치 조건은 해당 관측에서 정상이다. 당시 개별 값이 없는 이전 두 실패의 원인은 확정할 수 없다.

`scripts/qualify_stationary_dashboard_mmf_owned_idle_v3.py`와 `src/flybrain_drive/owned_camera_postflight.py`는 완료된 같은 요청의 active를 관측한 뒤 더 새로운 idle 상태가 도착할 때까지 최대 2초 기다린다. 오프라인·일시정지·AI off·정차·제동·장치 해제 등 다른 조건이 달라지거나 다른 카메라 요청이 나타나면 거부한다. active를 관측하지 못하거나 새 idle 상태가 없으면 실패한다.

`.../20261101-dashboard-camera-owned-idle-v3/`는 5프레임을 기록하고 같은 요청 active 4회 후 새 idle을 확인해 완료됐다. 상태 확인 경과 212.11ms, producer 시계 5316.562941→5316.8309021, vJoy FREE였다. `.../20261101-dashboard-owned-idle-v3-observation-audit/`에 원본 관측 재생과 classifier 해시를 보존했다. 잘못된 요청·오래된 상태·감각 이상 등 16검사가 통과했다. 이는 정지 카메라 종료 처리의 진단이며 물리 노출 동기·주행 중 작동·자율 제어·학습 자료 자격을 입증하지 않는다. 기존 실패와 원본은 보존한다. 당시 시뮬레이터는 일시정지·오프라인·AI와 카메라 off·브레이크1·가속0이었다. 이후 카메라 장착 조건과 시각·몸 감각의 인과 기여, 조향·가감속·변속·셋업 및 완주·랩타임을 계속 검증한다.


### 2026-11-01 장착 조건과 정지 카메라 초기 밝기 변화
종료 검사 수정 후 기존 `.../20261101-dashboard-camera-owned-idle-v3/`와 새 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-exterior-camera-owned-idle-v3/`는 각각 정지/일시정지 조건의 5그룹, producer completed와 owned active→new idle, archive hash를 확인했다. `.../20261101-stationary-camera-mount-pair-owned-v3-audit/`에서 같은 generation/car/track의 세 번째 순차 프레임을 비교했다. 전방 중심은 2.869m 차이, 좌우뒤 중심 차0, 같은 겹눈 등록 방향이다. 721 수용점 평균 절대 차는 전방0.1547, 좌0.000280, 우0.000473, 뒤0.000547이다. 단일 정지 장면과 순차 프레임이며 환경 전이 실패 원인을 증명하지 않는다. `.../20261101-stationary-camera-mount-five-frame-retina/`는 다섯 프레임쌍의 전방 차0.0997~0.2257을 기록한다.

두 mount의 성공 및 이전 실패로 보존된 총 네 촬영에 대한 원본 해시 기반 `scripts/audit_stationary_front_brightness_transient.py` 감사에서, 렌더 시작 직후 네 시야 모두 밝기 변화가 반복 관측됐다. 예를 들어 완료된 dashboard 실행 5프레임의 수용점 평균은 전방0.6798→0.3128, 좌0.5784→0.2911, 우0.7619→0.4055, 뒤0.6660→0.3530이었다. 그 원인이 노출 처리·초기화·기타 렌더 방식 중 무엇인지는 확인하지 못했다. 실패 run들은 프레임 자료로만 남기며 성공으로 승격하지 않는다.

추가로 `.../20261101-dashboard-camera-stationary-30-brightness/`에서 동일 paused 조건의 30그룹·archive 검증 및 owned 종료를 완료하고 `.../20261101-dashboard-camera-stationary-30-retina-audit/`에 원본 각 수용점을 저장·감사했다. 생산자 clock 간격 총2.894초, 30번째 수용점과의 평균 절대차<1e-6이 마지막까지 유지되는 첫 frame은 전방/좌4(시작+0.170초), 우/뒤3(+0.134초)였다. 이는 이 한 정지 장면에서의 관측이며 다른 조명·트랙의 안정 시간이나 물리 exposure 동기를 보증하지 않는다.

`scripts/replay_stationary_camera_brightness_flyvis.py`는 30그룹을 원본 producer frame clock 간격·같은 FlyVis state로 연속 재생하며 첫 영상 고정 네 시야를 대조했다. `.../20261101-dashboard-stationary-brightness-flyvis/`에서 T4/T5 활동 최대 절대 차1.339, 공학적 6축 pool의 yaw 성분 최대 절대 차0.06449(+0.134초)였다. 최초 이미지를 영구 유지한 대조이므로 밝기 변화의 효과는 보이지만 이것을 실제 회전이나 생리적 오작동으로 단정하지 않는다. 기준 설정에 이 초기 영상/지속 신경 상태가 섞이면 방향 feature에 영향을 줄 가능성이 있으며, 새 기준은 원본 시간과 분포를 보존해 독립적으로 검증해야 한다. 학습/배포/차량제어0, 실제 가감속·변속·셋업 및 자율 완주·라인/랩타임은 여전히 미검증이다. 다음은 camera 초기화 원인과 안정 기준을 확인하고 시각·몸의 통합 회로가 다른 조건에서도 가감속과 양방향 조향을 학습하도록 진행한다.


### 2026-11-01 정지 영상 초기 변화와 긴 정지 기준 후보
`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-dashboard-stationary-finalframe-flyvis/`는 기존 정지 30프레임을 같은 FlyVis state로 재생하고, 비교군에 마지막 안정 화면을 시간0부터 유지한 반사실을 사용했다. 마지막 화면은 미래 정보이므로 이 대조는 actor 입력이나 실행 가능한 초기화가 아니다. 원본 활동은 이전 재생과 최대4.85e-7 차였다. 가상으로 첫 11표본 중 마지막3개(index8–10) 평균을 각 조건에서 빼면 이후 yaw pool 차의 최대 절대값은 0.007577, 마지막도 −0.007577이다. 원본 producer clock에는 긴 간격이 있고, paused native 패킷의 실제 참조 자격을 이 창에서 입증하지 않았으므로 이것을 실제 actor의 기정 결과로 해석하지 않는다. 초기 밝기 이력과 지속되는 FlyVis state가 기준 편차에 영향을 줄 수 있다는 반사실 근거다.

기존 카메라·몸 감각 검사와 last3 평균은 유지하되 적어도 1.0초에 걸쳐 정지로 관측된 신선한 표본·11개 다른 capture를 요구하는 실험 후보 `src/flybrain_drive/flight_long_stationary_reference.py`를 추가했다. 연속 관측 사이 최대 간격 및 움직임 감지 시 초기화는 기존 계약을 따른다. 표본 사이 실제 물리 정지를 증명하지 않으며 1초는 앞서 본 사례를 바탕으로 한 공학적 가설이다. 회로 및 가중치 변경0; 기존·신규 검사9개 통과.

사전 계획 `.../20261101-long-stationary-reference-comparison-plan/`에 비교 기준을 기록한 뒤 동일 session의 두 저속 이력과 원래 네 시야/몸 입력을 각각 재생했다. `.../20261101-long-stationary-reference-comparison/result.json`: 빠른 몸 수신 이력에서 첫 준비 원본 index66→93, 공통 준비 구간의 출발 전4결정(고유 영상1개)에서 |yaw feature| 평균0.001694→0.0000104; |native angular[1]|>0.01인 고유 영상16개 중 같은 부호12→13. 역순 이력에서 첫 준비22→29, 출발 전2결정(고유 영상2개) 평균0.001840→0.000120, 회전 고유 영상18개 부호15→15. 이 표본은 시간적 상관이 있고 두 이력은 같은 차량·세션이며 선행 기록을 본 뒤 정한 후보라 독립 일반화가 아니다. 긴 기준은 시작 전 관측을 더 요구해 기존 readiness를 각각246→230, 81→75로 줄였고 출발 전에 준비할 기회도 적었다. 빠른 이력의 geometry zero 대조는 완전 동일하다. 역순 기존 baseline은 외부 속도 명령이 달라 출력 intent는 직접 비교하지 않았다.

몸 회전 부호는 진단 관측이지 최적 조향 목표가 아니며 물리 발생/노출 시간·생리·실제 자율 운전/가속·제동·변속·셋업·유효랩은 미검증이다. 후보는 배포하지 않았고 optimizer/차량 조작0이다. 다음은 독립 환경에서 기준 준비 시간·지속성, 시각과 몸 회로의 인과적 기능 및 감각 기반 조향·가감속 학습을 검증한다.


### 2026-11-01 2초 정지 prefix의 새 저속 영상·몸 이력
기존 두 자료를 검토해 제안한 1초 정지 기준을 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-long-reference-new-history-plan/`에 사전 기록하고 새 이력을 수집했다. 같은 Monza/BMW/세션에서 정지 영상 2초 후 +0.1/0/−0.1/0 조향, 몸 감각 읽기 요청 간격10ms, 최대8km/h 제한을 사용했다. `.../20261101-long-reference-new-history-probe/` 결과 최고5.114km/h, 정차와 vJoy FREE, 카메라81그룹·body525건이다. `...-host-audit/`, `...-body-audit/`는 원본/해시를 확인했다. actor가 운전한 것이 아니며 physical camera exposure·body 발생 시각은 확인되지 않았다.

`...-flyvis/`에서 원본 전체를 FlyVis로 재생한 뒤 `...-baseline/`과 `...-candidate/`는 같은 단일 감각 회로/가중치/중립 외부 maneuver 입력으로 523결정을 실행했다. 기존 last3의 첫 준비 index67, 1초 후보95, 준비291→273; prior body 결측은 두 조건 모두390/2092 시야 결정이다. `...-comparison/result.json`의 사전 지정 기준: 두 방식 공통 준비 상태에서 출발 전 고유 영상11개의 |yaw 특징| 평균0.001111→0.0000961; |native angular[1]|>0.01인 고유 회전 영상16개에서 몸 방향 부호 일치12→11. 회전의 sign은 운전 최적 정답이 아니다. 원본/후보 geometry zero 대조는 bitwise 동일하다. intact 조향·전진 intent 평균 절대차 2.25e-5/7.50e-8, 두 방식 모두523결정에서 두 intent가 음수였다.

새 기록은 조향 순서와 시간이 다르지만 같은 car/track/session으로 독립 환경 검증이 아니다. 길게 기준을 잡으면 초기 특징 편차는 줄지만 방향 일치와 운전 출력이 개선됐다는 증거는 없고, 더 오래 정지해야 하는 제약도 생긴다. 신경 생리·진짜 시간 정합·시각 선행 예측·가감속/변속/셋업 적응·폐루프 및 유효 랩은 미검증. fit/optimizer0·actor 제어0이며 사전 정한 저속 조작만 수행했다. 이후 다른 환경과 camera/body 지연을 개선하고 시각·몸의 인과 기능을 우선하며 단일 회로의 실제 운전 학습으로 진행한다.


### 2026-11-01 통합 회로용 인과적 운동 시각 예측 실험
정답 경로/절대 위치나 미래 값을 운전 입력에 넣지 않고 네 시야 T4/T5 공간 요약의 현재값과 직전값만으로 다음 40ms 운동 채널을 추정하는 `src/flybrain_drive/flight_motion_forecast.py`를 추가했다. 8개 방향 채널별 bounded gain은 네 시야와 공간 위치에 공유한다. 수학적 예측 경계이며 초파리의 측정된 신경생리나 별도 운전자 모델이 아니다. `UnifiedFlightSensorimotor`의 선택형 입력 경계로 연결하고 단일 비행 graph·몸 되먹임·기존 readout/state는 유지했다. 예측기 미사용이 기본이며 gain0은 원래 회로와 동일하다. 원래 decision 간격이 36–44ms 밖이거나 시야 결측이면 거부하고, episode 종료 시 raw 이전 영상 상태를 지우며 실패한 결정은 상태를 갱신하지 않는다. 신규 및 기존 회귀 12개 통과.

사전 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-motion-forecast-plan/`에 합성 학습 3질감·200회 Adam, 현재 운동값 유지 baseline, 새 질감879303/±0.13rad/s 보류 평가를 지정했다. 공개 FlyVis·고정 spatial pool의 시간 이력에서 다음 운동 활동을 예측하도록 경계 8개 계수만 학습했다. `.../20261101-causal-motion-forecast-synthetic-fit/`은 optimizer200, 학습 loss 2.5803e-5→2.3517e-5, 미사용 새 합성 질감에서 평균 prediction MSE2.4821e-5 vs 현재값 유지2.5270e-5(1.78% 감소). 검증 질감은 optimizer에 들어가지 않았지만 하나의 합성 구형 장면/속도이고 실제 서킷이나 생리 일반화는 아니다. checkpoint SHA `6aa54e1f5f681ef236a8c687ec85c5d97a73cfe2d76026f50e1562a4274900f3`, actor_qualified=false.

`.../20261101-causal-motion-forecast-historical-eval/`은 기존 실제 8이력(원본 training_admissible=false)을 학습에 사용하지 않고 고정 후보를 평가했다. 원본 clock의 허용된 연속 5,916구간에서 predictor MSE3.4434e-5 vs 현재값 유지3.5781e-5(3.76% 감소), 각 이력 감소2.12–4.74%; 이전에 검토한 동일 트랙 이력이고 미래의 저장 운동 활동은 평가표적에만 썼다. 제외된 cadence window0. 이 값은 코너 예측/차량 성능이 아니라 바로 다음 신경 특징의 예측 오차다. 원본 허용 여부를 승격하지 않았다.

`.../20261101-causal-motion-forecast-unified-historical/`은 기존 known DEV02/07의 원래 네 시야·body21·원본 타임스탬프 및 각 native bytes를 확인하고 각 episode의 동일 단일 graph를 zero-gain/학습 gain으로 각각 재생했다. 카메라별 과거 몸은 모두 명시적으로 결측 처리했다. 각각742/741결정 전체 state를 유지했으며 조향 의도 평균 절대 차6.52e-6/1.03e-5, 전진 차5.91e-8/1.07e-7, 조향 부호 변경0이다. 두 원본과 후보 모두 조향·전진 intent가 모든 결정에서 음수였다. 예측이 회로 출력에 영향을 주지만 이 조건에서 양방향 조향·가속이나 주행 개선의 증거는 아니다.

물리 센서/노출 시각·실제 신경 생리·코너 선행 판단·가감속/변속·셋업·폐루프 및 유효 랩은 미검증이며 actor 배포/시뮬레이터 제어0이다. 다음은 다양한 실제 장면에서 예측 채널의 방향/시간 기여와 몸 피드백·가감속 경로를 검증하고, 허용된 데이터를 별도 확보해 통합 회로의 실제 운전 학습으로 이어간다.


### 2026-11-01 운동 시각 예측의 초기 구간 편향과 방향 전환 학습 실패
기존 8실제 이력의 전체 평균 예측 오차 3.76% 감소는 episode 시작의 높은 오차에 크게 좌우됐다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-motion-forecast-transient-audit/`는 원본 해시·시계·5,916 연속 window를 재계산했다. 이력당 첫 예측 8개에서는 현재값 유지 대비 MSE28.35% 감소했으나 그 다음5,908개 전체는 2.21% 증가했고, 첫10개 이후5,836개에서는 2.99% 감소했다. 다음 변화 방향을 사후 분류하면 일치3,912개는26.23% 감소, 반전629개는20.64% 증가, 약한 변화1,375개는6.44% 증가다. 미래 신호는 분류·평가에만 사용했고 predictor 입력에는 넣지 않았다. 이 상관된 관측들은 독립 시행이 아니다. 이전 전체 평균 자체는 맞지만 변화 시점의 실패를 가린다.

최근 두 변화의 방향 일치로만 예측을 켜는 사후 후보는 `.../20261101-motion-forecast-backward-gate-audit/`에서 동일한5,908개에 현재값 유지 대비 MSE9.25% 증가했다. 기존 예측기도 같은 구간에서2.21% 증가했다. 이 임계값은 이전 이력을 본 뒤 정한 분석용이며 통합 회로에 설치하지 않았다.

`.../20261101-reversing-yaw-forecast-plan/`은 합성 구면에서 정지0.4초→한 방향0.6초→반대 방향0.6초로 연속 자세를 유지하는 자극, fit seed879401/879402 ±0.23rad/s, 별도 seed879403 ±0.13rad/s, 전환 주변과 나머지 움직임에 각각50% 손실, 예측 계수8개 Adam200회를 사전 지정했다. `.../20261101-reversing-yaw-forecast-synthetic-fit/`에서 실제200회 학습, fit loss 1.2510e-5→4.9528e-6, 계수0.8245–0.9288이 됐다. 학습에 쓰지 않은 하나의 새 합성 자극에서 전환 주변 MSE는 현재값 유지8.435e-6, 직전 예측기6.778e-6, 새 후보3.705e-6; 나머지 움직임은 각각2.151e-6, 1.295e-6, 0.425e-6다. 원본 validation 배열에서 수치를 재계산했다. 갑작스러운 방향 전환은 동일한 이전 관측만으로 원천적으로 확정할 수 없으며 합성 자극 하나는 독립 서킷/생리 검증이 아니다.

동결된 새 checkpoint SHA `e95ada8de819a2d511be2243a64484bc9e894044176ae37596e83197980b1908`를 원래 fit 금지 실제 8이력에 학습 없이 적용한 `.../20261101-reversal-motion-forecast-transient-audit/`는 현재값 유지 대비 전체5,916구간 MSE61.03% 증가, 첫 예측 제외5,908구간79.95% 증가, 사후 반전629구간116.25% 증가를 확인했다. 합성상 개선이 실제 영상으로 옮겨가지 않았으며 새 후보는 배포하지 않는다. 기존 후보도 반전에서 나쁘므로 기본 예측기는 계속 off다. 두 분석과 합성 fit 관련 소스는 컴파일 및 저장 배열 재계산을 통과했다.

이 결과는 정답 코너·지도 입력 없이 시각/몸에 근거하는 하나의 비행 감각운동 회로를 향한 예측 모듈 진단이다. 실제 물리 시각 정합·생리·전후좌우 시야의 코너 인식, 가감속·변속·셋업 적응 및 자율 완주/라인/랩타임은 미검증이며 기존 원본을 학습에 쓰지 않았다. 다음은 합성의 이상적 일정 회전과 실제 영상의 밝기·프레임 유지·몸 운동 변화의 차이를 분해하고, 적격한 다양한 장면/조건을 마련해 시각 경로의 인과 기여 및 통합 조작을 우선 검증한다.

## 2026-11-01 — 기존 실제 이력의 시야·이력별 예측 오차

합성으로만 학습한 기존 예측기와 반전 자극 예측기의 동결 checkpoint 및 원래 dataset/episode SHA를 확인하고, 기존 8개 이력의 40ms 조건을 통과한 5,916개 창을 시야별로 사후 분석했다. 각 원천 receipt의 `training_admissible=false`를 확인했으며 실측 이력에 대한 학습이나 파라미터 선택, 차량 조작은 하지 않았다. 원래 선행 평가 데이터의 시야 순서는 전·좌·우·후다.

| 시야 | 기존 후보, 현재값 유지 대비 MSE 변화 | 반전 학습 후보, 현재값 유지 대비 MSE 변화 |
|---|---:|---:|
| 전방 | −0.56% | +69.77% |
| 좌 | −3.97% | +61.50% |
| 우 | −4.91% | +56.71% |
| 후방 | −5.87% | +55.59% |

반전 후보는 8개 이력 모두의 모든 시야에서 악화했다. 기존 후보의 전방은 8개 중 2개에서 악화했다. 네 시야 MSE의 동일 가중 평균을 독립 재계산하면 이전 전체 결과 −3.76%, +61.03%와 일치한다. 이 분석은 기존에 반복 검토한 단일 트랙의 시간상 상관된 창에 대한 사후 진단이며, 물리 노출 시각·동기화·카메라 보정, 초파리 생리, 회로의 운전 성능을 검증하지 않는다. 후보들은 배포하지 않았다. 다음에는 합성 자극의 시간·장면 분포와 실제 캡처의 시간/중복 특성을 분리 검증해야 한다.

- 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-forecast-view-episode-audit/result.json`
- 실행: `scripts/audit_forecast_view_episodes.py`, optimizer 업데이트 0, actor/차량 제어 0.

## 2026-11-01 — 캡처 반복에 따른 예측 오차의 사후 분해

원래 8개 이력의 dataset·episode·events 해시와 각 의사결정 시각의 일치, 원본 `training_admissible=false`를 확인했다. 직전·현재·다음 시점의 캡처 generation/ID로 40ms 의사결정 조건을 통과한 5,916개 창을 분해했다. 다음 캡처 ID는 평가 분류에만 쓰였으며 예측 입력이 아니다.

| 캡처 관계 | 창 수 | 기존 합성 후보의 MSE 변화 | 반전 합성 후보의 MSE 변화 |
|---|---:|---:|---:|
| 두 간격 모두 새 캡처 | 5,595 | −3.80% | +61.32% |
| 이전 캡처 반복만 | 160 | −1.80% | +53.54% |
| 다음 캡처 반복만 | 161 | −2.28% | +38.33% |
| 두 간격 모두 반복 | 0 | 평가 불가 | 평가 불가 |

변화율의 기준은 각 구간에서 현재 특징값을 유지하는 예측이다. 8개 이력 전체에서 바로 이전과 동일한 캡처 ID를 쓴 의사결정은 161개였고, 해당 반복을 제외해도 반전 후보의 오차 증가는 남았다. 새 캡처의 host 수신 간격은 이력별 중앙값 약 40.11–40.40ms, 최대 약 83.28–89.34ms였다. 이는 물리 노출 간격이 아니다. 반복된 픽셀에도 FlyVis 내부 상태가 계속 변할 수 있으므로 특징값이 동일하다고 가정하지 않았다.

기존에 검토한 시간 상관 이력의 사후 진단이며 독립적 일반화·생리·실시간 운전 성능 증거가 아니다. 학습·actor 배포·차량 제어는 하지 않았다. 다음에는 실제 노출/센서 동기화와 현실적인 시각 변화 분포를 검증해야 한다.

- 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-forecast-capture-repeat-audit/result.json`
- 재현: `scripts/audit_forecast_capture_repetition.py`.

## 2026-11-01 — 합성·실제 운동 특징 변화의 방향 비교

같은 512개 pooled 운동 특징에서 직전 변화와 다음 변화의 코사인을 사후 비교했다. 기존 실제 8이력의 40ms 조건 5,916창은 중앙값 0.206, 10–90백분위 −0.110–0.521, 양의 정렬(>0.1) 66.13%, 반대(<−0.1) 10.63%였다. 합성 회전 반전의 기존 검증 장면 1종, 두 방향의 움직임 60창은 중앙값 0.945, 정렬 100%였다. 특히 회전 명령 반전 주변 8창도 중앙값 0.766이며 정렬 100%였다. 반전 구간 정의는 이전 합성 평가와 동일하다.

회전 입력이 반전돼도 이 실험의 FlyVis/풀링 출력 변화가 바로 반전하는 것은 아니며, 이 자극은 실제 기록의 낮은 시간적 일관성을 충분히 재현하지 못했다. 이는 합성 학습 후보의 실제 실패를 설명할 수 있는 분포 차이지만 단일 원인이라는 증거는 아니다. 같은 실제 이력과 합성 검증 장면을 이미 살펴본 사후 분석이며, 서로 다른 영상 내용과 초기 조건, host 시각·물리 노출 미검증이 비교를 제한한다. 시간적으로 연결된 창은 독립 실험이 아니다. 실제 데이터 학습, 파라미터 선택, actor 배포 또는 차량 제어는 없었다.

다음 자극은 카메라·풍경·몸 움직임과 변화하는 조명 및 시간 샘플링을 다양화하고, 입력 회전 부호뿐 아니라 실제 운동 특징의 반전·약한 상관 자체가 발생하는지 사전에 검사해야 한다. 생리 기능·통합 회로 기여와 실제 폐루프를 별도로 검증한다.

- 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-increment-geometry-audit/result.json`
- 재현: `scripts/audit_motion_increment_geometry.py`.

## 2026-11-01 — 혼합 운동·조명 합성 자극 시험

이전 실패를 보고 설계한 진단용 자극 한 종류에서, 네 카메라가 동일한 합성 구형 환경과 연속적인 회전·이동을 공유하도록 했다. 각 축에 정현파 운동을 주고 공통 밝기 변화도 더했다. FlyVis 원래 모델의 시야별 독립 신경 상태로 재생했으며 실제 도로·차량 센서나 생리 재현이 아니다. 0.4초 정지 후 좌우 대칭 두 조건의 운동 구간 60창을 조사했다.

운동 특징 연속 변화 코사인 중앙값은 0.942, 정렬 93.33%, 반대 5.00%, 특징 다음 변화 RMS 0.01887이었다. 기존 실제 이력 중앙값 0.206, 반대 10.63%, RMS 0.005982와 차이가 남았다. 이전 합성 반전 운동 구간의 중앙값은 0.945였다. 두 동결 예측기의 MSE는 이 새 합성 자극에서 현재값 유지보다 각각 27.86%, 65.17% 낮았으나, 이는 이미 실패를 보인 실제 이력의 결과를 뒤집지 못한다.

진단 자극 1종·사후 분석이므로 독립 일반화, 현실적 시각 분포, 생리 기능, 폐루프 운전은 확인되지 않았다. optimizer 업데이트·actor 배포·차량 제어는 0이다. 단순 정현파 혼합만으로 실제 특징의 약한 일관성을 재현하지 못했으므로 다음에는 원인별 자극/센서 특성 분리와 별도 검증 장면이 필요하다.

- 합성 응답: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-varied-motion-879501/result.json`
- 평가: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-varied-motion-forecast-assessment/result.json`
- 재현: `scripts/replay_flyvis_varied_motion.py`, `scripts/evaluate_varied_motion_forecast.py`.

## 2026-11-01 — 합성 시각 자극의 시간 계약 불일치 및 인과 재생

실제 이력 `scripts/replay_flyvis_cameras.py`와 `FlightStream`은 시각 t의 운동 상태를 만들 때 직전 수용기 입력을 t까지 유지하며 적분한 다음 현재 수용기 영상을 결합한다. 기존 합성 스크립트 `replay_flyvis_coherent_yaw_reversal.py`, `replay_flyvis_varied_motion.py`는 각 영상 자체를 40ms 적분한 뒤의 운동 상태와 그 영상을 같은 시점으로 저장한다. 따라서 합성 운동 상태가 실제 계약 대비 한 샘플 앞서며, 합성 학습과 실제 평가의 입력 시간 의미가 달랐다. 과거 실험의 원본·스크립트·결과는 보존한다.

동일한 혼합 자극의 저장된 수용기 3조건을 원본 FlyVis와 고정 회색 정상상태 초기화로 다시 재생했다. 최초 시점은 초기 상태, 그 뒤 각 간격은 직전 영상으로 5ms씩 8회 적분했다. 인과 재생의 t 운동과 이전 합성 t−1 운동은 전체 평균 절대 차 4.29e−8, 최대 3.58e−7이었다. 같은 인덱스 간 평균 절대 차는 0.02739였다. 이는 해당 자극에서 한 프레임 선행을 확인하는 수치이며 물리 카메라 노출 동기화의 증명은 아니다.

인과 시간으로 고친 혼합 자극의 운동 구간 60창에서 변화 코사인 중앙값은 0.922로, 기존 0.942보다 낮지만 실제 이력 0.206과 여전히 차이가 크다. 이전 동결 예측기 둘은 인과 재생된 이 단일 합성 장면에서 기준 대비 −27.38%, −63.98% MSE였으나 사후 진단일 뿐이며 실제 실패를 해소하지 않는다. 기존 합성으로 학습된 두 후보는 시간 계약이 불일치하므로 actor에 사용하지 않는다. 앞으로 새 학습은 저장된 샘플을 인과 재생한 별도 데이터와 미사용 검증 장면에서 진행하고, 기존 실제 이력의 학습 금지 및 생리·차량 검증 구분을 유지한다.

- 인과 재생: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-varied-motion-causal-timing/result.json`
- 평가: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-varied-motion-causal-assessment/result.json`
- 재현: `scripts/replay_varied_motion_causal_timing.py`, `scripts/evaluate_varied_motion_causal_timing.py`. optimizer·actor 배포·차량 조작 0.

## 2026-11-01 — 인과 시각 합성 재학습과 실제 이력 대조

별도 계획을 기록한 뒤 이전 프레임을 유지하는 인과 시간 계약으로 구형 환경의 혼합 회전·이동·조명 자극을 생성했다. 합성 seed 879511/879512, 진폭 0.23으로 T4/T5 pooled 특징 예측기의 공유 gain 8개를 Adam 200회 학습했다. seed 879513, 진폭 0.13 장면은 업데이트 종료 후에만 열었다. 각 영상은 전후좌우가 같은 합성 몸과 환경을 공유하며 기존 FlyVis 상태는 각 시야에서 이어진다. 실제 영상·교사 신호는 학습에 쓰지 않았다.

학습 손실은 0.00033343에서 0.00011655로 감소했다(마지막 값은 최종 업데이트 직전). 별도 합성 장면의 운동 구간 60창에서 현재값 유지 MSE 0.000306745, 신규 후보 0.000108923으로 64.49% 감소했다. 기존 두 후보는 각각 0.000222644, 0.000110164였다. 저장된 검증 배열에서 MSE를 독립 재계산했다. 이 합성 결과는 같은 생성 방식의 한 질감일 뿐 독립적인 현실 일반화가 아니다.

기존 실제 8이력은 각 원본 receipt `training_admissible=false`를 확인하고 평가에만 썼다. 5,916창에서 현재값 유지 MSE 3.57809945e-05, 신규 후보 5.40963627e-05, 상대 변화 +51.19%였다. 모든 이력에서 오차가 증가했다. 이미 여러 번 검토한 시간 상관 이력이라 독립 시험은 아니다. 물리 노출/몸 동기화·생리·폐루프 운전 미검증이며 actor 배포와 차량 제어는 없다. 인과 정렬은 필수 수정이었지만 이 합성 학습의 현실 실패를 해결하지 못했다.

- 계획: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-varied-forecast-plan/plan.json`
- 학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-varied-motion-forecast-fit/result.json`
- 실제 이력 평가: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-varied-forecast-historical-eval/result.json`
- 재현: `scripts/replay_flyvis_varied_motion_causal.py`, `scripts/train_causal_varied_motion_forecast.py`, `scripts/evaluate_causal_varied_forecast_history.py`.

## 2026-11-01 — 통합 회로 몸 되먹임 투영의 반사실 대조

기존 development 이력 episode-02의 원본 이벤트·몸 샘플·시각 특징 및 출처 해시를 확인하고 742결정을 동일한 단일 비행 그래프에서 재생했다. 네 시야의 촬영 시점 몸 상태는 복구할 수 없어 명시적 결측으로 남겼다. 학습된 실험용 몸 되먹임 투영의 가중치만 제로, 전체, 유효 표시 부분만, 몸 수치 부분만 남기는 네 조건을 비교했다. 기존 기본 몸 감각 경로와 회로 상태 적분은 유지했고 예측기는 제로 gain이다.

제로 투영 대비 전체 투영의 평균 intent 변화는 조향 +0.01168, 전진 +0.01393이었다. 유효 표시 부분만 남긴 경우 +0.01044, +0.01059, 몸 수치 부분만 남긴 경우 +0.00100, +0.00483이었다. 비선형 경로이므로 이 수치들을 더해 전체 기여로 해석하지 않는다. 전체·부분·제로 네 조건의 742결정 모두 조향과 전진 intent는 음수였다. 첫 결정은 이전 상태가 없어 조건 간 동일했다.

원본은 `training_admissible=false`이며 학습에 쓰지 않았다. 이미 검토한 한 이력에 대한 모델 내부 개입이라 실제 감각 인과·생리·물리 시간·주행 성능을 증명하지 않는다. 기존 체크포인트의 학습 출처가 이 실험으로 정당화되지 않으며 배포하지 않는다. 센서 유효성으로 생기는 상수 효과를 분리한 채 몸 변화에 대한 반응과 부호·회복을 새로 검증해야 한다. optimizer 업데이트·actor/차량 제어 0.

- 전체 투영 제거: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-body-adapter-ablation/result.json`
- 수치/유효 표시 대조 및 재계산: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-body-adapter-components/comparison.json`
- 재현: `scripts/replay_unified_body_adapter_ablation.py`, `scripts/replay_unified_body_adapter_components.py`.

## 2026-11-01 — 몸 값 그룹별 고정 반사실

앞의 동일한 development 이력 742결정과 단일 회로를 재생했다. 기존 이벤트·원본 body의 일치와 출처 해시를 먼저 검증하고, 실험용 몸 되먹임 어댑터의 입력에서만 한 그룹을 이력 첫 유효 값으로 고정했다. 다른 센서 값, 모든 유효 마스크, 기본 몸 경로, 영상·목표·신경 상태 업데이트는 그대로 뒀다. 최초 시점만을 기준으로 한 고정값은 실제 물리적으로 가능한 차량 상태를 보장하지 않는다.

| 고정 그룹 | 원본 대비 평균 절대 조향 intent 변화 | 평균 절대 전진 intent 변화 |
|---|---:|---:|
| 차체 pitch/roll | 0.002505 | 0.0000462 |
| 네 바퀴 하중 비율 | 0.000631 | 0.0000151 |
| 네 타이어 온도 | 0.00000648 | 0.00000402 |
| 네 바퀴 각속도 | 0.001424 | 0.001608 |

각 조건에서 intent 부호 변화는 0건이었다. 새 원본 조건은 직전 실험의 전체 되먹임 조건과 최대 절대 차 0로 일치했다. 변화 크기는 이 기록의 값 범위와 초기 기준에만 해당하며, 그룹의 중요도나 물리적인 제어 방향을 일반화할 수 없다. 시야별 과거 몸 상태는 명시적으로 결측이다. 실제 이력은 평가 전용, optimizer 업데이트·actor/차량 제어 0이며 생리·완주·랩타임 검증은 없다.

- 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-body-group-counterfactual/result.json`
- 기준 재현: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-body-group-counterfactual/reference-parity.json`
- 재현 코드: `scripts/replay_unified_body_group_counterfactual.py`.

## 2026-11-01 — 단일 회로의 선택적 유효 표시 중심화 경계

`UnifiedFlightSensorimotor`에 기존 `CenteredVehicleBodyFeedbackAdapter`를 명시적으로 주입할 수 있게 했다. 완전 유효한 센서의 availability 입력은 0으로 중심화하지만, 결측은 −1로 보존한다. 차체 값·원래 몸 경로·단일 신경 상태는 유지한다. 기본 선택은 기존 어댑터 그대로이며 타입과 포트 수가 맞지 않는 경계는 거부한다. 단위 검사에서 완전 유효 시 상수 효과 제거, 결측의 구분, 포트 크기 거부, 기존 회로/예측 경로를 함께 확인했다. 관련 검사 17개 통과.

재학습 없이 과거 가중치를 중심화 경계에 적용해 기존 development 이력 742결정을 반사실 재생했다. 중심화 경계와 제로 가중치 대조의 평균 절대 조향/전진 변화는 0.001773/0.004827이었다. 선택된 센서가 모두 유효한 이 이력에서, 이전 원래 경계의 값 부분만 남긴 대조와 최대 차 0로 일치했다. 모든 intent는 여전히 음수였다. 즉 고정 유효 표시 효과는 제거했지만 양방향 운전이나 실제 몸 반응 개선은 확인하지 못했다.

재학습하지 않은 구 가중치의 입력 의미가 달라졌으므로 연구용 비교일 뿐이다. 단일 반복 이력·미검증 촬영/몸 동기화와 과거 학습 출처 제한이 유지되며, 생리·폐루프·주행 자격과 혼동하지 않는다. 원본은 `training_admissible=false`, optimizer 업데이트·actor 배포·차량 조작 0. 향후 허용된 실험 데이터와 결측/몸 변화 실험으로 재학습 및 독립 검증한다.

- 재생: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-centered-body-boundary/result.json`
- 기존 대조 동등성: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-centered-body-boundary/parity.json`
- 구현: `src/flybrain_drive/unified_flight_sensorimotor.py`, `scripts/replay_unified_centered_body_boundary.py`.

## 2026-11-01 — 중심화 몸 경계의 합성 좌우 기울기 학습

실험 전 계획을 기록했다. 동일한 단일 비행 그래프·고정 합성 시각에서 몸 roll −0.04/0/+0.04rad에 대한 최종 조향의 상대 차이를 +0.004/0/−0.004로 유도했다. 이는 제어 부호 보정을 주장하지 않는 임의의 공학 가설이다. centered 몸 어댑터만 Adam 12회 갱신하고 유효성 열의 가중치를 0으로 고정했다. 실제 이력은 학습하지 않았다. 기존에 다른 예측 실험에 사용한 합성 seed 879511을 학습, 879512와 작은 ±0.025rad를 별도 조건으로 썼다. 이 seed는 사전에 다른 연구에 사용돼 완전한 독립 시험은 아니다.

훈련 최초 상대 목표 MSE는 0.0000160, 마지막 업데이트 직전은 0.0003399로 진동했다. 최종 업데이트 후 저장 배열에서 다시 계산한 훈련 MSE는 7.51556166e-06이다. 별도 합성 조건의 zero 대비 조향 변화는 +0.004347/−0.004344였고 roll이 결측이면 세 입력의 차이는 0이었다. 절대 조향은 전부 음수였다. 학습 안정성, 방향 보정, 실제 신체 생리는 검증되지 않았다.

동결 후보와 제로 경계를 원래 development 이력 742결정에서 대조했다. 평균 절대 intent 변화는 조향 0.002646, 전진 0.00000924였고 조향 부호 변화는 0건, 양의 조향/전진 출력은 각각 [0, 0]건이었다. 제로 경계 기준은 이전 재생과 최대 차 0로 일치했다. 동일한 반복 이력의 반사실 평가이며 물리 시간/생리/차량 성능 증거가 아니다. 원본 이력의 fit 금지 유지, actor 배포·차량 제어 0.

- 계획: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-centered-roll-contrast-plan/plan.json`
- 학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-centered-roll-contrast-fit/result.json`
- 실제 이력 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-roll-historical-replay/result.json`
- 재현: `scripts/train_unified_centered_roll_contrast.py`, `scripts/replay_unified_synthetic_roll_historical.py`.

## 2026-11-01 — 통합 회로 중립과 속도 감각 경로

기존 합성 정지 영상/몸으로 동일 통합 그래프를 101결정 재생하고, 해당 신경 상태의 근육 출력을 진단상 중립 기준으로만 삼았다. 다른 합성 질감의 동일 정지 조건에서 마지막 대조 조향/전진은 +0.000147/+0.00000469였으며 ±0.025rad roll에서는 +0.00453/−0.00424의 조향 대조가 나왔다. 기존 절대 intent의 출력 치우침과 구분하는 출력 전용 반사실이며 실제 차량 중립을 인증하지 않는다. 별도 질감도 이전 실험에 사용됐다. 원본 readout·checkpoint를 바꾸지 않았다.

더 중요한 발견은 동일 영상·다른 몸 감각이 같은 합성 조건에서 0m/s와 5m/s의 모든 101개 신경 근육 출력이 bitwise 동일했다는 점이다. `sensory_decision`이 속도 명령을 현재 속도로 맞춰 속도 오차를 0으로 만들고, 기존 추가 몸 감각 경계는 속도 인덱스 16을 포함하지 않으며 기본 경로는 각속도만 받는다. 이 가상의 조건은 속도와 바퀴 각속도가 물리적으로 불일치할 수 있지만, 모델 경로에서 속도 자체가 이 조건에선 구별되지 않음을 보인다.

기존 기본 경로를 바꾸지 않고 `CenteredSpeedBodyFeedbackAdapter`를 선택적 실험 경계로 추가했다. unsigned m/s를 30m/s로 정규화하는 것은 공학적 가정이며 초기 가중치는 0이다. 값과 결측 표시를 구분하며 통합 단일 회로의 기존 포트로 연결한다. 관련 검사 18개 통과. 수동으로 speed 가중치 0.25를 모든 몸 포트에 넣은 별도 합성 탐침에서 0→5m/s, 101번째 intent 변화는 조향 0.000407312065, 전진 -0.000558823347였다. 신호 전달만 확인한 것이고 부호·크기·생리·차량 제어 적합성은 미검증이다. 목표 속도/가감속 학습을 대체하지 않는다.

속도 탐침의 원본 result.json에는 `model_mutated=false`가 잘못 기록됐다. 실제로 새 경계의 가중치를 메모리에서 수동 설정했다. 기존 결과와 당시 실행 소스는 보존하고 `correction.json`으로 정정했다. 원래 모델 checkpoint·기본 경로·차량은 변경하지 않았다. optimizer 업데이트·actor/차량 제어 0.

- 중립 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-synthetic-neutral-stability/result.json`
- 속도 탐침과 정정: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-synthetic-speed-path-probe/correction.json`
- 구현: `src/flybrain_drive/flight_body_feedback_adapter.py`, `src/flybrain_drive/unified_flight_sensorimotor.py`.

## 2026-11-01 — 속도·바퀴 되먹임의 합성 학습

사전 계획에 따라 같은 단일 회로에서 기존 합성 roll 경계 가중치를 feature별로 새 속도 포함 경계에 옮겨 고정했다. 속도·바퀴 회전의 값 열만 Adam 0.005로 20회 업데이트했고, 유효성 열과 roll 열은 bitwise 보존했다. 합성 입력은 바퀴 미끄럼이 없는 등속 관계 omega=v/r 가정으로, 학습 0/5/10m/s·r=0.33m, 검증 2/5/8m/s·r=0.29m다. 정지 영상은 이전에 사용한 별도 합성 질감이다. 속도 상승에 따라 상대 전진 intent를 줄이도록 정한 목표는 부호·크기가 임의인 공학 가설이며 목표 속도 정책이나 생리적 발견이 아니다.

훈련 MSE는 최초 0.0000160, 마지막 업데이트 직전 0.000000710, 최종 업데이트 후 저장값에서 7.78220386e-07였다. 검증 속도 2/8m/s에서 5m/s 대비 전진 intent 변화는 +0.003152/−0.002254였다. 바퀴 센서 결측 시에도 반응은 남았지만 작았다. 동일 검증의 2→8m/s는 조향도 -0.012556 바꾸고 전진은 -0.005406 바꿨다. 따라서 원하는 전진 반응 외에 큰 조향 교차 효과가 있어 분리·개선이 필요하다. 가중치 보존이 곧 roll 동작 보존을 보증하지 않는다.

동결 후보를 이전 evaluation-only 이력 742결정에서 zero 경계와 비교하면 평균 절대 조향/전진 차는 0.016292/0.007644이고 조향 부호 변화 0, 양의 조향/전진 [0, 0]건이었다. 제로 경계 출력은 이전 재생과 최대 차 0로 일치했다. 기존 실제 데이터는 학습하지 않았고, 시간 상관된 단일 기록·물리 노출/몸 동기 미검증이다. 배우지 않은 실제 토크·변속·슬립·셋업과 폐루프 완주에 대한 증거가 아니며 actor/차량 제어 0, 후보 미배포다.

- 계획·학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-wheel-contrast-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-wheel-contrast-fit/result.json`
- 실제 이력 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-speed-historical-replay/result.json`
- 재현 코드: `scripts/train_unified_speed_wheel_contrast.py`, `scripts/replay_unified_synthetic_speed_historical.py`.

## 2026-11-01 — 합성 속도 반응과 조향 교차 억제

이전 power-only 실험과 같은 roll 가중치·합성 학습 데이터·속도/바퀴 값 열로 시작하고, 전진 상대 목표 MSE에 속도 변화에 따른 상대 조향 변화 제곱을 동일 가중치로 더해 Adam 0.005, 40회 학습했다. 목표는 실험용 제어 가설이며 실제 운전 보정이 아니다. 유효성·roll 열과 원래 단일 그래프는 동결했다.

이미 다른 작업에 사용된 별도 합성 질감의 2→8m/s 및 반경 0.29m 조건에서 기존 후보의 조향 변화 -0.012556, 새 후보 0.000263; 전진 변화는 각각 -0.005406, -0.003652였다. 즉 이 조건에서 조향 교차 효과가 줄었지만 전진 반응도 약해졌다. 이 결과는 단일 정적 영상과 이상적인 바퀴 관계의 진단이며 실제 코너에서 속도에 따른 정상적인 조향 변화를 막아야 한다는 주장은 아니다. 목표·교차 항의 가중치는 실제 물리나 생리로 검증되지 않았다.

기존 fit-forbidden development 이력 742결정의 동일 zero 기준은 두 재생에서 bitwise 일치했다. 새 후보의 zero 대비 평균 절대 조향/전진 변화는 0.002583/0.005204이고 조향 부호 변화 0, 양의 두 intent는 각각 0건이다. 이전 후보 0.016292/0.007644와 비교할 수 있으나 실제 주행 개선은 입증되지 않는다. 합성 목표에 대한 학습 40회, 실제 이력 fit 0, actor/차량 조작 0. 생리·실제 시간·폐루프·랩타임 미검증으로 두 후보 모두 미배포다.

- 계획·학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-wheel-turn-control-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-wheel-turn-control-fit/result.json`
- 실제 이력 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-turn-control-historical-replay/comparison.json`
- 재현: `scripts/train_unified_speed_wheel_turn_control.py`, `scripts/replay_unified_speed_turn_control_historical.py`.

## 2026-11-01 — 시간 변화하는 시각·몸 조합의 통합 그래프 평가

기존에 다른 예측 실험에 쓴 합성 인과 시각 seed 879513에서 네 시야가 함께 변하는 두 반대 움직임 41프레임을 사용했다. 각 영상과 speed 2/8m/s, roll −/+0.025rad, r=0.29m의 이상적 바퀴 회전 조합을 하나의 그래프에 넣었다. 몸 값은 영상 운동과 독립적으로 고정한 controlled counterfactual이며 현실적으로 부합하지 않을 수 있다. roll-only·기존 speed-power·speed-power-turn 세 동결 경계와 모든 원본/모델 SHA를 확인하고 같은 2×2×2×41 결정을 재생했다. optimizer·실제 이력 학습·actor/차량 제어 0.

두 speed에 대한 매칭 조건별 평균 절대 조향 변화는 roll-only 0, power-only 0.011584, turn-control 0.000234; 전진의 고속−저속 평균 변화는 각각 0, −0.004850, -0.003281였다. turn-control의 roll 양−음 평균 조향 변화는 -0.007955, roll-only는 −0.007975였다. 정적 합성 조건에서 얻은 교차 억제와 기울기 반응이 이 움직이는 합성 입력에도 남았다.

turn-control에서 영상 두 방향만 바꾼 조향 평균 절대 변화는 0.000204였다. 이 실험에서의 입력 범위와 단위가 서로 달라 신호의 생물학적 중요도를 비교하거나 시각 기능이 없다고 단정할 수 없다. 다만 상대적으로 작은 회로 출력 변화를 확인했으며 실제 곡선 선행 인식·시각 기여와 연결되지 않았다. 모든 후보의 328결정에서 조향·전진 intent가 음수였다. 독립 텍스처·물리 시각/몸 동기·회로 생리·폐루프 운전은 검증되지 않았고, 후보는 미배포다.

- 계획·결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-roll-motion-grid-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-roll-motion-grid/result.json`
- 배열 재계산: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-speed-roll-motion-grid/independent-check.json`
- 재현: `scripts/evaluate_unified_speed_roll_motion_grid.py`.

## 2026-11-01 — 움직이는 시각 신호의 포트·회로 경로

기존 합성 seed 879513의 두 움직임과 동일한 body(5m/s·zero roll·가상 wheel)를 하나의 graph에 입력했다. 첫 11 결정 이후 두 조건 차이의 원소 RMS는 576 pooled scene 0.01336, 42 base visual logits 0.02274, 42 opponent projection logits 0.08520, 42 최종 시각 port drive 0.002820, 9,728 graph state 0.0001627, 두 intent 0.0002603이었다. 단계마다 차원·단위·비선형성이 달라 이 숫자의 비율을 전달 효율로 해석할 수 없다. 시각 port drive는 다음 간격 동안 적용되며 같은 결정의 state/intent는 이전 drive를 반영한다.

base visual sigmoid 출력의 5/50/95백분위는 0.0734/0.1011/0.1394, 기울기는 0.0680/0.0909/0.1199였으며 기울기 <0.01 비율은 0이었다. 이는 이 표본에서 base sigmoid가 극단 포화는 아니란 뜻이며 전체 시각/신경 경로의 비포화를 증명하지 않는다.

별도 세 조건에서 같은 두 영상을 재생했다. intact의 onset 이후 평균 절대 조향 차이는 0.0002783, 마지막 부호 차 +0.0004765였다. opponent temporal projection을 0으로 두면 평균 0.0002093, 마지막 −0.0003425였다. base visual 입력 가중치를 0으로 두되 bias를 유지하면 평균 0.0004862, 마지막 +0.0008499였다. intact 재생은 이전 포트 추적과 최대 차 0로 일치했다. 기본 가중치를 0으로 두면 평균 시각 구동 상태도 달라지고 회로는 비선형이므로, 두 경로가 단순 합산되거나 실제 시각 뉴런이 억제된다고 단정하지 않는다. 기존 시각/시간 경계가 출력에 상호작용함을 보여주는 모델 내부 반사실이다.

동일 합성 한 장면의 반복 진단이며 body는 영상과 물리적으로 연결되지 않았다. 기존 합성 생성·graph checkpoint의 출처 제한, 센서 물리 시간·생리·실제 코너 판단·가감속/완주·랩타임 미검증을 유지한다. 학습·actor/차량 제어 0, 후보 미배포. 다음에는 각 시야/시간 경로를 보정된 자극과 함께 생리/차량 대응의 부호·인과 기여를 별도 검증한다.

- 경로 추적·sigmoid: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-signal-path-audit/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-signal-path-audit/sigmoid-check.json`
- 경로 개입·기준 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-path-lesions/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-path-lesions/parity.json`
- 재현: `scripts/audit_unified_visual_signal_path.py`, `scripts/audit_unified_visual_path_lesions.py`.

## 2026-11-01 — 인과 정렬 합성 yaw의 운동 시각 경로 학습

사전 계획한 구형 합성 질감 879601/879602는 정지 0.4초 후 −/+0.23rad/s로 회전하고, 각 시각의 FlyVis 상태는 직전 수용기 영상을 유지해 적분했다. 동일한 단일 그래프와 동결 속도/몸 경계에서 운동 시각 opponent projection만 Adam 0.005로 8회 갱신했다. 좌우 시각 조건은 같은 몸 감각을 사용하고 출력 turn(−yaw)−turn(+yaw) = +0.008은 검증되지 않은 공학 가설이다. yaw 정답은 손실/평가에만 사용했으며 actor 입력이 아니다.

훈련 두 질감의 좌우 차이는 0.001079/0.001105에서 0.009435/0.012018로 변했다. 업데이트 후에만 연 별도 합성 질감 879603과 ±0.13rad/s에서는 0.000385→0.000456으로, 진폭을 비례 축소한 참고 목표 0.004522의 약 10.1%였다. 손실은 초기 4.77e−5에서 일시 1.26e−6까지 내려간 후 마지막 업데이트 직전 1.10e−5로 다시 커졌다. 같은 진폭의 새로운 질감 검증이 없어 질감과 속도 중 어느 차이가 원인인지 분리할 수 없다. 이 구형 자극은 도로나 생리 검증이 아니다.

기존 fit-forbidden development 이력 742결정에서 이전 projection과 새 projection을 같은 몸 경계로 대조했다. 새 projection의 평균 절대 intent 변화 조향 0.0004725·전진 0.00000414, 조향 부호 변화 0, 양의 intent는 [0, 0]건이었다. 기존 projection 기준은 직전 재생과 최대 차 0로 같았다. 반복 검토한 한 이력·물리 시각/몸 동기 미검증이므로 실제 일반화·코너 판단·운전 성능의 증거가 아니다. 실제 데이터 학습 0, actor/차량 제어 0이며 후보를 배포하지 않는다.

- 계획: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-yaw-visual-fit-plan/plan.json`
- 합성 학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-yaw-opponent-fit/result.json`
- 실제 이력 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-yaw-opponent-historical/result.json`
- 재현: `scripts/replay_flyvis_constant_yaw_causal.py`, `scripts/train_unified_causal_yaw_opponent.py`, `scripts/replay_unified_causal_yaw_opponent_historical.py`.

## 2026-11-01 — 합성 질감과 yaw 속도의 2×2 분리

이전 회전 검증 실패를 본 뒤 진단을 등록했다. 기존 학습 질감 879601과 앞서 검증에 사용한 질감 879603을 ±0.23 및 ±0.13rad/s에 교차했다. 빠졌던 두 자극은 동일한 인과 FlyVis 생성기로 새로 생성했다. body·회로·기존 및 학습 시각 경계는 고정했고 source/결과 SHA와 수용기/운동 상태 시계를 확인했다. 목표 yaw는 모델 입력이 아닌 합성 평가 식별자다.

| 질감 | yaw 속도 | 학습 전 좌우 turn 차 | 학습 후 좌우 turn 차 | 진폭 비례 참고 목표 |
|---|---:|---:|---:|---:|
| 학습 질감 | ±0.23rad/s | 0.001079 | 0.009435 | 0.008000 |
| 학습 질감 | ±0.13rad/s | 0.001162 | 0.002367 | 0.004522 |
| 이전 검증 질감 | ±0.23rad/s | −0.000187 | 0.003450 | 0.008000 |
| 이전 검증 질감 | ±0.13rad/s | 0.000385 | 0.000456 | 0.004522 |

정의는 마지막 다섯 결정의 turn(−yaw)−turn(+yaw)다. 저장된 출력에서 전 조건을 재계산해 일치시켰다. 이 두 질감 비교에서는 회전 속도만 낮추거나 질감만 바꿔도 반응이 줄었고 둘 다 바꾸면 가장 작았다. 두 질감·두 속도의 사후 합성 결과로 요인의 일반적 인과 효과나 실제 운전 성능을 주장할 수 없다. 선형 진폭 비례 목표도 임의의 평가 가설이다. optimizer/실제 데이터 학습/actor·차량 제어 0, 후보 미배포. 다음 학습에서는 다양한 속도와 질감을 학습·완전 미사용 검증으로 분리하고 몸 및 시각의 동시 반응을 다시 확인한다.

- 사전 등록 및 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-yaw-texture-speed-factorial-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-yaw-texture-speed-factorial/result.json`
- 재현: `scripts/replay_flyvis_constant_yaw_causal.py`, `scripts/evaluate_yaw_texture_speed_factorial.py`.

## 2026-11-01 — 두 속도·두 질감의 인과 시각 학습

사전 등록 후 합성 질감 879601/879602 각각에서 ±0.13/±0.23rad/s 네 조합을 사용했다. 서로 다른 샘플의 손실을 순차 역전파해 평균을 내고 동일 graph의 opponent 시각 투영만 Adam 0.002로 10회 갱신했다. 기존 몸 경계·비행 회로는 고정, 실제 이력 학습 0. 목표 좌우 turn 차는 yaw 진폭에 비례시키는 공학 가설이며 생리나 자동차 조향 근거는 아니다.

학습 전 지정한 미사용 합성 seed 879604(±0.18), 879605(±0.13)는 업데이트 후 처음 열었다. 원본/이전 단일 속도 후보/새 후보의 좌우 차는 각각 604에서 0.000522/0.004254/0.003270(가설 목표 0.006261), 605에서 0.000882/0.002684/0.002779(목표 0.004522)였다. 새 후보는 둘 다 원본보다 크지만 이전 후보보다 하나는 작고 다른 하나는 약간 크다. 두 질감만으로 일관된 개선이나 실제 일반화를 입증할 수 없다. 검증 배열에서 결과를 재계산해 일치시켰다.

기존 fit-forbidden development 이력 742결정에서 새 경계는 원본 대비 평균 절대 조향/전진 intent 0.0003931/0.00000329 차이, 조향 부호 변화 0, 양의 두 intent는 [0, 0]건이다. 원본 경계는 이전 이력 재생과 최대 차 0로 같았다. 반복 검토한 단일 이력·시각/몸 물리 동기 미검증이며 배우지 않은 실제 코너 예측·가감속·변속·셋업·완주 성능에 대한 증거가 아니다. actor/차량 제어 0, 후보 미배포.

- 계획·학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-yaw-multispeed-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-yaw-multispeed-fit/result.json`
- 이력 평가: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-yaw-historical/result.json`
- 재현: `scripts/train_unified_causal_yaw_multispeed.py`, `scripts/replay_unified_multispeed_yaw_historical.py`.

## 2026-11-01 — 기존 서킷 이력의 네 시야 고정 반사실

원래 development episode-02의 742결정, 원본 이벤트/신체/시각 특징과 허가 상태 `training_admissible=false`를 확인했다. 각 대조는 네 시야 중 하나의 9×16 pooled 입력을 최초 관측값으로 고정하고 다른 세 시야와 현재 몸 감각·시계·단일 재귀 회로는 그대로 유지했다. 네 시야 validity를 true로 남긴 인위적 특징 고정이며 카메라 결손 처리, 실제 입력 또는 생리적 뉴런 절제가 아니다. 첫 FlyVis 활동도 고정되므로 현실적으로 불가능할 수 있다.

| 고정 시야 | 평균 절대 조향 intent 변화 | 평균 절대 전진 intent 변화 |
|---|---:|---:|
| 전 | 0.0007431 | 0.00000678 |
| 좌 | 0.0017311 | 0.00000349 |
| 우 | 0.0004326 | 0.00000539 |
| 후 | 0.0005798 | 0.00000635 |

각 조건에서 조향 부호 변화는 0이었고 두 intent는 모두 음수였다. 원본 조건은 직전 다중속도 후보 재생과 최대 차 0로 일치했다. 좌측 효과가 이 기록에서 가장 컸으나 시야별 실제 중요도·물리적 독립성이나 코너/라인 판단 기여의 근거로 일반화할 수 없다. 영상 노출·몸 동기, 생리, 폐루프·빠른 유효 랩과 셋업 적응 미검증. optimizer 업데이트/actor·차량 제어 0, 학습 금지 유지, 후보 미배포.

- 계획·결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fourview-freeze-historical-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fourview-freeze-historical/result.json`
- 배열/기준 확인: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fourview-freeze-historical/verification.json`
- 재현: `scripts/audit_historical_fourview_frozen_input.py`.

## 2026-11-01 — 네 시야 반사실의 실제 이력 미래 경로 진단

기존 fit 금지 development episode-02에서 원본 이벤트·camera ZIP 해시와 frame별 바이트, 기존 네 시야 고정 결과와 clock을 확인하고 teacher API receipt의 driving·AI on·track 내 연속 위치에서 5/10/20m 미래 chord를 재구성했다. 원본 front 렌더 행렬과 시선은 오직 사후 평가에만 사용하며 actor에 위치·방향·미래 경로를 입력하지 않았다. 평가 기준은 speed >1m/s, 해당 chord 유효·전방, 절대 곡률 >=0.001/m이다. 이 교사 주행 경로의 chord는 즉시 조향 정답이나 최적 레이싱 라인이 아니다.

5m 유효 352결정 중 우측 348·좌측 4, 10m 169결정과 20m 59결정은 모두 우측이었다. 원래와 네 개 시야 고정 조건 모두 turn intent가 계속 음수이므로 raw 왼쪽 양수 대 chord 오른쪽 양수 부호를 반대로 비교하면 5m 348/352, 10m 169/169, 20m 59/59로 동일했다. 항상 우측이라고 말하는 상수 기준과 같으며 높은 숫자는 시각의 코너 판단을 증명하지 않는다. 네 시야의 고정은 불가능할 수 있는 pooled 특징 반사실이라 카메라 결손이나 실제 주행 기여를 뜻하지 않는다.

원본 한 이력은 시간적 상관·방향 편향이 심하고, 물리 노출/몸 동기와 즉시 조향 부호·제어 지연은 미검증이다. 학습에 쓰지 않았고 `training_admissible=false`와 fit 금지를 유지했다. 좌우 코너가 모두 포함된 별도 경로·속도·상태, 정지/직진 및 기준선과 사전 정의된 센서 인과 대조가 향후 필요하다. optimizer/actor/차량 조작 0.

- 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-view-future-path/result.json`
- 보존한 진단 배열: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-view-future-path/path-diagnostic.npz`
- 재현: `scripts/evaluate_historical_view_future_path.py`.

## 2026-11-01 — 좌우 경로 포함 이력 탐색 및 회로 대조

기존 8개 이력을 동일한 5/10/20m 미래 chord 기준으로 사후 조사했다. 각 원본 source의 `training_admissible=false`, event·camera ZIP/frame SHA를 확인했고 future position·render pose는 actor에 전달하지 않았다. 20m 조건에서 development episode-02는 좌 0/우 59, episode-07은 좌 0/우 551이었다. train으로 이름만 지정됐으나 원본 fit 금지인 episode-06은 좌 214/우 343, episode-03은 54/113, episode-04는 64/42였다. 연속 시계열 개별 결정이며 독립 코너 수가 아니다. 전체의 20m 분포는 좌 370/우 2,422였다. episode-02 재구성은 직전 독립 결과 배열과 정확히 같았다.

좌우 표본이 많은 episode-06을 결과를 본 뒤 진단용으로 선택했다. 이는 기존 train 역할·이전 연구 사용 이력이므로 독립 검증이 아니며 원래 `training_admissible=false`를 변경하거나 학습에 쓰지 않았다. 같은 graph/body에서 원래 시각 경계와 합성 다중속도 시각 후보를 원본 741결정으로 재생한 후 기존 보존 20m chord와 비교했다. 557개 유효 결정 모두 두 모델의 조향 intent는 음수여서 좌 214개 일치 0, 우 343개 일치 343이었다. 일치율 343/557=61.58%, 좌·우 균형 정확도 0.5는 언제나 우측으로 두는 상수 기준과 정확히 같다. 5m는 379/611, 10m는 365/590으로 역시 상수 우측과 같다.

chord 방향은 사후 평가 전용이며 즉시 조향 정답·최적 레이싱 라인이 아니다. 시각 노출 및 몸/조작 물리 시간, 신경 생리·폐루프 주행을 증명하지 않는다. 이 진단은 편향 이력에서의 100%와 달리 좌우 구간에서 현재 고정 부호 문제가 드러남을 보여준다. optimizer/actor/차량 제어 0, 후보 미배포, 출처 허가 변경 0. 다음에는 좌우·직진이 사전 분리된 신규 평가와 실제 학습 자격을 충족할 수 있는 수집·시간/조작 계약을 검증해야 한다.

- 8이력 조사: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-future-path-coverage/result.json`
- 회로 재생·비교: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-yaw-coverage-selected/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-selected-history-future-chord/result.json`
- 재현: `scripts/scan_historical_future_path_coverage.py`, `scripts/replay_unified_multispeed_yaw_selected_history.py`, `scripts/evaluate_selected_history_future_chord.py`.

## 2026-11-01 — 합성 중립 기준의 실제 이력 출력 반사실

이전 단일 graph 합성 정지 상태 101번째 decision의 원래 intent [−0.057086, −0.346652]를 별도의 고정 가설 기준으로 삼아, 동일 readout gain/mix를 가정하고 보존된 episode-06의 intent에 `tanh(atanh(intent) − atanh(reference))`를 사후 적용했다. 실제 경로·표적을 사용해 기준을 맞추지 않았다. 합성 기준은 다른 영상·몸 상태에서 얻었고 실제 차량 조향/가감속 중립으로 자격이 없으며 기존 checkpoint·회로 상태·actor는 변경하지 않았다. 이것은 단순 출력 좌표 변환이지 실시간 보정 구현이나 주행 시험이 아니다.

20m chord 평가 557결정에서 원래 출력은 모두 음수, 반사실은 모두 양수였다. 전자는 우측 343개만 맞고 후자는 좌측 214개만 맞아 균형 일치율은 둘 다 0.5였다. 중립 이동만으로 양방향 구분이 생기지 않았다. 미래 chord의 좌/우를 분류하는 연속 조향 순위 AUC는 0.000572였고 저장 배열에서 73402개 쌍을 직접 비교해 재확인했다. 단조 기준 이동은 순위를 바꾸지 않는다. 그러나 한 주행 이력의 좌우는 시간·위치·속도와 강하게 연관될 수 있고 frame들은 독립이 아니다. 또한 사용한 turn/chord 부호는 계약 가정이며 물리 시간·실제 조작 방향, chord와 즉시 조향의 관계는 미검증이라 AUC를 주행 정확도나 신경 인과 효과로 해석할 수 없다.

이전 검토 후 선택된 train-labelled 원본은 `training_admissible=false` 그대로 평가만 했다. 시각/몸 생리, 실제 코너 판단·가감속·셋업·완주 및 랩타임은 미검증. optimizer/actor/차량 조작 0.

- 결과와 원본 배열: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-neutral-selected-history/result.json`
- 순위 계산 재확인: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-neutral-selected-history/independent-pairwise-check.json`
- 재현: `scripts/evaluate_synthetic_neutral_selected_history.py`.

## 2026-11-01 — 미래 경로 방향과 시간 교란, 두 추가 이력

앞서 episode-06의 20m chord 557개 유효 표본은 좌 214가 먼저, 우 343이 나중에 나와 방향 전환이 한 번뿐이었다. 원본 시간과 조향 intent의 Spearman 상관은 0.811, 속도와의 상관은 0.598이었다. 7–9m/s 하위 표본에서도 두 방향을 찾았지만 방향과 시점 분리는 되지 않았다. 따라서 이전 순위 AUC 0.000572는 한 연속 구간의 시간/위치 경향을 반영할 수 있으며 독립 사례 수만큼의 증거가 아니다. 물리 카메라/몸 시각도 여전히 미검증이다.

기존 coverage를 보고 20m 방향이 두 번 바뀐 episode-03/04를 사후 선택해 같은 모델과 고정 합성 후보를 전체 원본 각 742결정으로 재생했다. 원본 둘 다 `training_admissible=false`였고 어떠한 학습에도 쓰지 않았다. 유효 20m 표본에서 episode-03은 좌54/우113, episode-04는 좌64/우42였다. 후보 조향은 각 167개·106개 모두 음수여서 각각 항상 우측 기준과 동일하고 균형 부호 일치율 0.5였다. 연속 점수 AUC는 각각 0.1847과 0.3981로 episode-06의 0.000572와 달랐다. 시간과 점수의 Spearman은 각각 0.388과 −0.052였다.

기록은 여전히 같은 서킷의 이전 연구 이력이고 사후 선택·시계열 상관이 있으며 교사 chord는 최적 라인이나 즉시 조향 정답이 아니다. 부호/물리 동기·생리·폐루프·가감속/완주·랩타임을 확인하지 못했고, 좌우 전환 횟수도 실제 독립 코너 수를 증명하지 않는다. optimizer/actor/차량 제어 0, fit 금지 유지. 다음에는 별도 좌우·직진 경로와 속도/시간을 사전 분리하고 물리적 sensor/action 타이밍을 검증해야 한다.

- 시간 교란: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-selected-history-temporal-confound/result.json`
- 두 이력 재생: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-yaw-reversal-histories/result.json`
- 방향 평가: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-reversal-histories-future-chord/result.json`
- 재현: `scripts/audit_selected_history_temporal_confound.py`, `scripts/replay_unified_multispeed_yaw_reversal_histories.py`, `scripts/evaluate_reversal_histories_future_chord.py`.

## 2026-11-01 — 네 시야 합성 도로 경계와 코너 자극

구형 무늬 회전 자극만으로는 앞 코너의 공간 형태를 검증할 수 없어 `SyntheticRoadVisual`을 추가했다. 공유된 평면 세계에서 12m 이후 직진/좌/우로 굽는 폭 8m 도로 경계, 절차적 표면 무늬와 기존 네 카메라 기하를 320×180 영상으로 렌더한다. 몸은 0.4초 정지 후 3m/s 직선 이동한다. 도로 형태는 실제 트랙·차량 동역학·초파리 광학 실측이 아니며 생성용 bend 값은 actor 입력이 아니다. 출력 재현, 네 화면/좌우 입력 차이, 범위 검사를 통과했고 전방 preview를 보관했다.

한 합성 질감의 세 도로를 동일 FlyVis에 시야별 독립 신경 상태로 통과시켰다. 직전 프레임을 유지하여 5ms×8회 적분하는 인과 시간 규칙이며 각 경우 41결정이다. 좌/우 도로 최종 영상의 망막 수용기 절대차 >1/255 개수는 전 50/721, 좌 2/721, 우 2/721, 후 0/721이었다. 앞쪽 일부가 다르지만 시야·자극 강도에 따라 다른 수용기 반응이 작을 수 있음을 뜻한다. 원본/합성 yaw 학습 후보를 같은 단일 회로·몸 입력으로 평가한 마지막 10결정의 좌 도로−우 도로 평균 조향 intent는 각각 −0.00000230/−0.00000375였고 세 조건 모두 전체 조향 intent가 음수였다. 입력이 완전히 동일해서 생긴 결과는 아니지만 작은 자극·단일 질감·직선 가상 궤적이므로 코너를 인식하지 못한다고 일반화할 수 없다.

이 도구와 결과는 도로 형태의 시각 선행 처리 실험을 위한 출발점이며, 레이싱라인·실제 AC·물리 노출/몸 동기·생리·폐루프 주행은 검증하지 않았다. 학습/actor/차량 제어 0이며 후보는 미배포다. 다음에는 곡률·거리·질감·자세를 사전에 분리한 자극과 비행 회로 반응 검증, 허용된 실제 기록과의 대조가 필요하다.

- 입력: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-synthetic-road-879701/result.json`
- 회로 평가·입력 확인: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-synthetic-road-eval/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-synthetic-road-eval/input-verification.json`
- 구현·검사: `src/flybrain_drive/synthetic_road_visual.py`, `tests/test_synthetic_road_visual.py`, `scripts/replay_flyvis_synthetic_road_causal.py`, `scripts/evaluate_unified_synthetic_road.py`.

## 2026-11-01 — 합성 도로 형태의 망막 경계 학습

사전 계획 뒤 도로 생성기에 곡률·굴곡 시작 거리 옵션을 더한 별도 스크립트를 만들었다. 학습은 합성 질감 879701(곡률 계수 .018, 시작12m)과 879702(.025, 10m), 검증은 학습에서 열지 않은 879703(.015, 11m)·879704(.020, 14m)로 지정했다. 네 시야 수용기와 FlyVis 운동 상태는 원래 causal hold로 생성했고 공통 body trajectory는 직선이다. 합성 bend의 좌·우는 손실에서만 쓰고 actor 입력이 아니다.

합성 다중 yaw 후보·몸 경계와 단일 비행 graph를 동결하고 base visual 입력 가중치 중 네 시야 망막 intensity 열만 Adam .003으로 10회 업데이트했다. 가중치 bias, T4/T5 열, 신경 graph와 몸 경계는 유지했다. 마지막 10결정에서 좌 도로가 우 도로보다 +0.004의 조향 차를 내도록 정한 값은 생리/차량 부호 보정이 아닌 임의 가설이다. 학습 손실은 8.02785e−6→마지막 업데이트 직전 8.02039e−6으로 거의 줄지 않았다.

미사용 도로 879703/879704의 좌−우 조향 차는 기존 −0.00000354/−0.00000741, 학습 후 −0.00000193/−0.00000583이었다. 두 도로 모두 가설 목표에 한참 못 미쳤고 절대 조향은 음수였다. 879703 직진 조향은 −0.058313→−0.058633으로 공통 편향도 바뀌었다. 학습 첫 도로의 기존 조건은 직전 합성 회로 재생과 최대 차 0로 일치하고, 검증 배열의 차이를 재계산했다. 좁은 전방 망막 자극·표면/회로 경계의 gradient·출력 기준 등 원인은 아직 분리되지 않았다.

기존 fit 금지 development episode-02 742결정의 원본 대비 신규 시각 경계 평균 절대 intent 변화는 조향 0.0002176, 전진 0.00000405, 조향 부호 변화 0, 양의 intent [0, 0]건이었다. 이전 다중 yaw 후보의 기준 재생은 최대 차 0. 실제 이력은 평가만 했고 반복 검토·물리 노출/몸 동기·생리/폐루프/레이싱라인/랩타임 미검증이다. actor/차량 제어 0, 후보 미배포. 다음에는 망막 공간 자극의 신호 크기와 회로 경계의 gradient·공통 편향을 분리해 효과적인 학습 조건을 검증해야 한다.

- 계획·학습: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-spatial-training-plan/plan.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-spatial-training-fit/result.json`
- 실제 이력 대조: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-spatial-historical-replay/result.json`
- 재현: `scripts/replay_flyvis_synthetic_road_varied.py`, `scripts/train_unified_synthetic_road_spatial.py`, `scripts/replay_unified_synthetic_road_historical.py`.


## 2026-11-01 합성 도로의 공간 기울기 진단

사전 등록 합성 학습 장면 879701의 직진·우·좌를 동일 통합 회로에서 재생하고 마지막 10결정 평균 조향의 망막 intensity 가중치 기울기를 측정했다. 저장된 원래 학습 전 출력과 최대 차 0이다. 직진/우/좌 기울기 L2는 각각 0.0193582/0.0193622/0.0193612, 좌−우 기울기는 0.00000926486, 좌우 평균 공통 기울기는 0.0193617이다. 둘의 크기 비는 약 2,090이고 코사인 유사도는 −0.1147이다.

좌−우 기울기 방향에 ±0.5 L2의 가중치 탐침을 적용하면 좌−우 차의 반차는 0.000004595, 공통 출력의 반차는 −0.001157이었다. 공통 기울기에 직교하는 방향에서는 각각 0.000004556과 −0.0000388이었다. 이 유한 크기 비선형 탐침은 국소 미분과 완전히 같지 않으며, 공통 변화 억제 가능성과 작은 좌우 신호를 보여줄 뿐 학습 성공이 아니다. 두 경우 모두 탐침 뒤 원래 가중치를 복원했으며 optimizer 갱신·후보 저장·차량 제어는 없다. 한 합성 도로에서의 국소 결과로 실제 서킷 또는 생리 기능을 일반화할 수 없다. 기록 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-spatial-gradient-audit/result.json`, 배열 `gradient-audit.npz`, 스크립트 `scripts/audit_synthetic_road_spatial_gradient.py`. 다음은 망막 수용기에서 pooled 입력까지 좌우 차이와 공간 손실을 측정해 병목을 구분한다.


## 2026-11-01 네 합성 도로의 공간 요약 진단

기존 `FlightScenePool`은 각 시야·채널에서 tanh 후 721 수용기를 동일한 16개 가우시안 공간 가중치로 요약한다. 네 기생성 합성 도로 879701~879704 각각의 마지막 10프레임에서 좌·우 도로의 tanh 후 망막/운동 반응 차이를 이 16행의 행공간에 직교 투영했다. 전체 공간 차이의 제곱 에너지 중 행공간 비율은 순서대로 1.195%, 2.206%, 0.919%, 1.058%다. 전방 망막 단독은 3.47%, 7.19%, 4.28%, 3.37%; 전방 운동 채널은 0.933%, 1.779%, 0.627%, 0.818%다. 각 시야·채널의 부호를 평균해 없애지 않고 프레임별 제곱을 합산했다. 실제 요약 출력과 독립 계산의 최대 차는 1.27e−6 이하였다. 두 장면은 이전 학습에 쓰였고 두 장면은 이전 평가에 이미 공개됐으므로 모두 사후 기술 분석이다.

이 값은 현재 가중치의 선형 행공간으로 재구성 가능한 자극 차이의 에너지다. 생리적 적합성, 코너 판별에 필요한 정보의 손실률 또는 출력의 인과적 개선을 나타내지 않으며, 압축 축의 개수가 적으면 낮을 수 있다. 원본과 요약의 단위·차원도 다르다. 다음에는 시야별 공간 위치를 더 보존하는 **공학적** 대안을 동일 통합 회로의 입력 경계에서 비교하고, 학습·검증 장면을 사전 분리해 시각 방향 반응과 실제 이력의 안전한 평가를 따로 확인해야 한다. 이 분석은 모델 갱신·배포·차량 제어 0이다. 원본 해시 및 수치: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-scene-pool-audit/result.json`, 배열 `pool-audit.npz`; 재현 스크립트 `scripts/audit_synthetic_road_scene_pool.py`.


## 2026-11-01 좁은 공간 요약의 합성 학습과 새 도로 평가

사전 등록 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-narrow-pool-plan/plan.json`. 기존 4×4 가우시안 위치는 유지하고 각 가중치를 네제곱 후 정규화해 폭을 절반으로 줄였다. 이는 공학적 시각 경계 변경이며 실제 겹눈 해상도의 측정값이 아니다. 동일 576차원 입력과 단일 비행 회로, 시간·몸 경로를 유지하고 망막 intensity 경계 가중치만 기존 합성 879701/879702로 Adam 10회 갱신했다. 미래 도로 방향은 손실 전용이다. 평가용 879705(곡률 .017·시작 13m)/879706(.023·9m)는 계획 등록 이후 별도로 생성했고 학습에는 사용하지 않았다. 한 시뮬레이터·트랙도 아닌 절차적 평면 도로다.

학습 손실은 8.01166e−6→7.98374e−6, 학습 장면 좌−우 차는 −0.000000313/−0.000004590→+0.000004958/+0.000005908이었다. 새 도로의 기존 요약·기존 가중치, 좁은 요약·기존 가중치, 좁은 요약·학습 가중치의 좌−우 차는 각각 879705에서 +0.000003770/−0.000006109/−0.000000816, 879706에서 −0.000008792/−0.000036370/−0.000023846이었다. 목표 +0.004에 크게 못 미치고 모든 조건의 절대 조향은 음수다. 기존 요약의 같은 새 데이터 재학습 대조는 수행하지 않았으므로 좁은 요약의 학습 효율이 우월하거나 열등하다고 단정하지 않는다. 학습 후보는 actor 자격이 없고 배포하지 않았다. 기존 실제 이력으로 fit하지 않았고 실패한 합성 평가 후 불필요한 실제 이력 재평가는 하지 않았다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-narrow-pool-fit/result.json`, 배열 `outputs.npz`, 연구용 checkpoint `narrow-retina-weight.pt`; 소스 `scripts/train_synthetic_road_narrow_pool.py`. 다음에는 공간 해상도뿐 아니라 작은 좌우 대비를 공통 영상과 구분해 회로에 전달하는 경계를 설계하고 새 장면에서 인과 비교해야 한다. 폐루프·생리·실제 완주 검증은 여전히 미완료다.


## 2026-11-01 작은 기울기에서 Adam epsilon 비교

같은 좁은 공간 요약·합성 학습 879701/879702·초기 가중치·Adam lr .003·10회 갱신을 유지하고 epsilon만 기본 1e−8에서 1e−12로 바꿨다. 첫 학습 단계에서 마스크된 전체 가중치의 기울기 최대 절댓값은 4.30e−8, 전체 RMS는 1.15e−9(마스크 밖 0 포함)였다. 손실은 이전 기본 설정 8.01166e−6→7.98374e−6, 새 설정 8.01166e−6→7.95169e−6이다. 이는 이 실험에서 optimizer 설정이 갱신에 영향을 준다는 증거이지 공간 시각 구조의 해결을 뜻하지 않는다.

새 검증 도로 879707(.016, 10m)와 879708(.022, 13m)은 계획 등록 후 생성해 학습에 넣지 않았다. 좁은 요약의 기본 epsilon 후보→작은 epsilon 후보 좌−우 조향 차는 각각 −0.00000395→+0.00000517, +0.00000758→+0.00001267이다. 사전 목표 +0.004의 약 0.13%와 0.32%로 여전히 매우 작고 모든 절대 조향은 음수다. 2개 절차적 도로 결과이며 유효 완주나 생리 검증은 아니다. 후보는 actor 미승인·미배포, 실제 데이터 fit·차량 제어 0이다.

사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-optimizer-epsilon-plan/plan.json`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-low-epsilon-fit/result.json`, 스크립트 `scripts/train_synthetic_road_low_epsilon.py`. 앞으로 반복적인 손실 감소만으로 성공을 판단하지 않고 시각 차이의 회로 유입과 조향의 실제 방향 변화를 별도 확인한다.


## 2026-11-01 시야별 망막 공간 대비 경계

기존 풀의 전후좌우 각 시야 망막 16개 공간값에서 해당 시야의 평균을 빼고 시각 포트에 투영하는 선택형 `RetinalSpatialContrastBoundary`를 추가했다. 이는 공학적 감각 경계 가설이며 실측 초파리 망막 회로나 별도 운전자 회로가 아니다. 결측 시야를 거부하고, 기본 모델은 경계를 사용하지 않으며, 새 경계의 0 초기 가중치는 기존 회로 출력을 보존한다. 경계 출력은 기존 단일 비행 회로 시각 포트에 들어가며 회로의 반복 상태가 다음 결정에 반영된다. 시야별 상수 밝기 변화 불변성·결측 거부·기본 등가·인과 상태 전달 등 관련 테스트 12개 통과 (`tests/test_retinal_spatial_contrast_boundary.py`, `tests/test_unified_flight_sensorimotor.py`).

사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-retinal-contrast-plan/plan.json`. 기존 합성 879701/879702만으로 새 경계 투영을 Adam 20회 학습하고 회로·기존 시각·시간·몸 경로를 고정했다. 학습 손실 8.02785e−6→8.00004e−6이다. 새 검증 장면 879709/879710의 경계 제거→활성화 좌−우 차는 각각 −0.00001339→−0.00000654, +0.00000429→+0.00000813이다. 목표 +0.004에 크게 못 미치며 모든 절대 조향은 음수다. 합성의 미래 굴곡은 손실 전용으로 actor에 들어가지 않는다. 생리 타당성·실제 코너 인지·차량 폐루프 미검증, 실제 이력 fit과 배포 및 차량 제어 0이다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-retinal-contrast-fit/result.json`, 출력 `outputs.npz`, 비승인 연구용 `spatial-contrast.pt`, 실행 소스 `scripts/train_synthetic_road_spatial_contrast.py`. 다음은 전체 공통 편향과 대비 경계의 가중치·신호 규모를 함께 분석하고, 코너 구분이 실제로 나오는 입력과 학습 조건을 분리해 검증한다.


## 2026-11-01 공간 대비 경계 증폭 진단

직전 학습 경계의 가중치에 0·1·10·100배를 적용해 이전 합성 검증 879709/879710을 사후 재생했다. 0·1배 재생은 직전 저장 출력과 최대 오차 0이다. 10배 시 좌−우 차는 각각 +0.00001989/+0.00002243, 100배에서는 −0.00000439/+0.00000848이다. 모든 평균 조향은 음수다. 경계 시각 포트 delta의 전체 RMS는 1배 약 0.050, 10배 약 0.334, 100배 약 0.494이며 출력 상한 ±0.5에 거의 포화된다. 인위적 가중치 증폭이고 실제 생리 강도나 안전한 제어 설정이 아니다. 경로에 신호를 크게 넣어도 이 두 자극에서 구분이 충분히 커지지 않는다는 제한된 반사실이며, 회로 전체의 능력 불가능성 또는 정확한 병목 위치를 증명하지 않는다. 학습·actor 배포·차량 제어는 0이다.

원본 해시와 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-retinal-contrast-scale-audit/result.json`, 배열 `scale-audit.npz`, 소스 `scripts/audit_synthetic_road_contrast_scale.py`. 다음에는 좌우 자극 자체의 차이와 시각 포트별 차이·하행 회로 응답을 짝지어 측정해 공통 변동과 방향 성분이 어디에서 섞이는지 확인한다.


## 2026-11-01 좌우 자극의 통합 회로 추적

직전 공간 대비 후보를 이전 합성 평가 879709/879710에 재생하며 각 41결정의 대비 경계 delta, 실제 시각 포트 drive, 9728개 회로 상태, 6개 날개 유래 출력군, 조향·전진 intent를 저장했다. 이전 후보 결과와 intent 최대 오차 0이다. 마지막 10결정에서 좌−우 차이 RMS는 두 장면 각각 대비 delta 0.0002985/0.0001147, 시각 포트 drive 0.0001057/0.00009741, 회로 상태 0.000005401/0.000004986, 6출력군 1.21e−8/1.07e−8, 2 intent 0.00001129/0.00001084다. 시각 포트의 공통 RMS는 약 0.1044, 출력군 공통 RMS는 약 0.001653이다. intent RMS는 조향과 전진 두 성분을 함께 포함하며 조향 차 그 자체가 아니다. 현재 시각 drive는 다음 간격에 유지되고 현재 출력은 그 이전 상태에서 읽는 기존 인과 규칙을 보존했다.

단계마다 차원·스케일·함수와 비선형 readout이 달라 위 RMS의 단순 비를 전달률로 해석할 수 없다. 결과는 사후 합성 자극 두 개에 대한 기술적 추적이며 특정 뉴런 연결의 원인이나 생리적 동작을 입증하지 않는다. 시각 포트별 개입과 날개 출력군의 민감도·부호를 통제된 시험으로 조사할 근거다. 옵티마이저·배포·차량 제어 0. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-paired-circuit-trace/result.json`, 원시 배열 `paired-traces.npz`, 스크립트 `scripts/audit_synthetic_road_paired_circuit_trace.py`.


## 2026-11-01 시각 포트의 국소 조향 민감도

합성 879709 직진의 기존 전체 입력 drive를 재생하고 42개 시각 포트에 모든 결정 간격에서 동일한 가법 교란을 넣었다. 각 결정 사이 원래 이전 drive 유지·세부 적분 및 단일 회로 상태를 보존했다. 고정 drive 재생과 원래 통합 모델의 intent 최대 차는 0이다. 마지막 10결정의 평균 조향에 대한 42포트 기울기 L2는 0.06810, 최대 절댓값 0.02388이며 42개 모두 1e−12을 초과했다. 기울기 방향에 L2 ±0.02 교란할 때 조향 평균은 −0.059590/−0.056874로 차 0.002716이다. 반차 0.001358은 1차 예측 0.001362와 유사하다. 모든 drive가 0~1 범위에 있음을 확인했다. 이는 적절한 시각 포트 패턴이 이 고정 이력의 회로 출력에 영향을 줄 수 있다는 반사실이지 실제 영상으로 그 패턴을 만들거나 좌우 코너 판단을 한 것은 아니다.

사후로 같은 장면의 마지막 10결정 좌−우 시각 포트 drive 평균 차와 이 직진 이력 기울기의 코사인은 +0.184였다 (`last10-alignment.json`). 기울기는 전 시간에 일정한 교란에 대한 것이고 이 벡터는 마지막 10결정 평균이므로 내적을 실제 출력의 인과 기여나 정확한 예측으로 해석하지 않는다. 다른 장면·속도·시간에의 일반화와 생리 타당성도 미검증이다. 학습·차량 제어·배포 0. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-port-sensitivity/result.json`, 포트별 기울기와 반사실 `port-gradient.npz`, 소스 `scripts/audit_synthetic_visual_port_sensitivity.py`. 다음에는 영상에 의해 생성된 시간별 포트 변화가 이 민감도와 어떻게 맞물리는지 실제 시간축을 보존해 분석하고, 그 뒤 별도 미사용 장면으로 입력 경계 학습을 평가한다.


## 2026-11-01 시간 정렬 시각 포트 교체

사후 합성 장면 879709/879710의 같은 몸 궤적에서 우회전 자극의 전체 drive를 기록하고, 42개 시각 포트만 각 시간의 좌회전 자극 값으로 교체해 고정 drive 비행 회로를 재생했다. 두 조건의 시각 포트 바깥 drive 최대 차는 0이고 원래 모델·고정 drive 기준·시각 포트 교체 출력은 각 대응 원본과 최대 차 0이다. 마지막 10결정 좌−우 조향 차는 각각 −0.000006538/+0.000008132였다. 우회전 기준 각 시간·포트의 국소 미분과 실제 시각 포트 차이를 같은 시간축에서 내적한 예측은 −0.000006415/+0.000008150이다. 부호가 다른 두 장면 모두 기존 경계가 일관된 목표 방향을 내지 못함을 보여준다. 이전의 시간 평균 코사인 진단을 정확한 기여 예측으로 해석하지 않는 이유가 해결됐지만, 이 결과는 합성 자극의 고정된 몸 궤적 및 비선형 회로의 근방에 한정된다. 지도 교사 신호를 actor에 전달하지 않았고 학습·차량 제어·배포는 없다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-time-aligned-port-swap/result.json`, 시간별 기울기와 포트 차이 원본 `time-aligned.npz`, 소스 `scripts/audit_synthetic_time_aligned_port_swap.py`. 다음은 이 국소 결과를 바탕으로 시각 입력이 포트 민감 방향과 일치하도록 학습하는 가설을 별도 사전 등록·미사용 장면에서 시험한다.


## 2026-11-01 회로 민감도 기반 공간 대비 학습

사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-jacobian-contrast-plan/plan.json`에 따라 기존 합성 879701/879702의 좌−우 차와 좌우 평균 대 직진 차를 전체 반복 회로를 통해 공간 대비 가중치로 미분했다. 정규화 Jacobian에 ridge 1e−4를 적용해 목표 좌−우 +0.004와 대칭 중심 0을 향하는 가중치를 계산하고 L2 반경 0/1/10/100 후보를 학습 손실만으로 골랐다. 가중치 해의 무제한 L2는 4221.46, 선택은 제한 100이다. 학습 손실은 경계 제거 1.60557e−5에서 선택 후보 1.59996e−5로 겨우 줄었다. 이 과정은 경계 가중치 **학습**이지만 Adam 갱신은 0이다.

계획 등록 후 만든 미사용 합성 879711/879712의 제거→후보 좌−우 차는 각각 −0.000007745→−0.000007331, −0.000002973→−0.000002861이다. 모든 조향은 음수이며 공통 출력이 약 −0.0025 이동했다. 두 장면에서 목표 방향 구분을 달성하지 못했다. 합성 목표는 손실에만 쓰였고 영상·허용 몸 감각 이외의 정답을 actor에 넣지 않았다. 실제 이력 fit·차량 제어·배포는 0이며, 미분 경계 자체가 생리적 회로라는 뜻도 아니다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-jacobian-contrast-fit/result.json`, Jacobian 및 출력 `outputs.npz`, 소스 `scripts/fit_synthetic_jacobian_spatial_contrast.py`. 계속 작은 변화만 추구하지 말고 영상에서 코너 차이가 구별 가능한지와 상수 속도 직선 합성 과제의 한계를 점검해 입력·과제 설계를 수정한다.


## 2026-11-01 합성 코너 영상 구분 가능성 진단

이전 합성 도로의 작은 출력 차이가 입력 정보 부재인지 구분하기 위해, 사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-signal-decodability-plan/plan.json`에 따라 각기 다른 질감 seed와 곡률·코너 시작 위치의 학습 12쌍과 별도 평가 12쌍을 생성했다. 각 쌍은 동일 배경·차량 위치에서 좌·우 굴곡만 다르다. 전후좌우 한 프레임의 망막 721개 표본과 기존 16개 가우시안 공간 요약에 각각 ridge 선형 진단기를 학습했다. 입력 중심은 학습 장면만으로 정하고 고정 정규화 강도를 썼으며 평가 영상을 fit·후보 선택에 사용하지 않았다. 모델의 기존 pool 가중치와 차이는 0이다.

평가 24개 이미지의 개별 분류 정확도는 원래 망막 표본 24/24, 공간 요약 21/24다. 좌·우를 비교한 쌍별 순서 점수는 둘 다 12/12다. 쌍은 공통 질감과 기하를 공유하므로 24개가 독립 실험이라는 의미는 아니다. 회색 도로와 인공 경계·고정 직선 몸 궤적에서 충분히 읽을 수 있는 방향 정보가 존재하고, 기존 풀에도 일부가 남는다는 진단이다. 일반 선형 분류기는 비교 도구일 뿐 초파리 회로나 actor가 아니며 영상에 정답 방향 값을 입력하지 않았다. 기존 회로가 그 정보를 읽어 운전한다거나 실제 AC 카메라·시각 생리에서 일반화한다는 증거는 아니다.

분류기의 합성 데이터 fit 외 회로 가중치 학습·배포·차량 제어 0. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-road-signal-decodability/result.json`, 가중치·점수 `diagnostic.npz`, 스크립트 `scripts/audit_synthetic_road_decodability.py`. 다음에는 이 진단기 예측을 배우는 일반 운전자 모델로 대체하지 않고, 미리 분리한 자극에서 기존 생물학적 시각·비행 회로 경계를 통해 방향 정보가 운전 출력에 인과적으로 전달되게 학습·검증한다.


## 2026-11-01 합성 공간 패턴의 시각 포트 전달

선택형 `RetinalSpatialContrastBoundary`에 네 시야의 공간 기준값을 명시적으로 저장하는 기능을 추가했다. 기준은 각 시야 평균을 제거해 공통 밝기 불변성을 유지하며, 모델 state dict에 포함된다. 기본 기준은 0이고 기존 동작을 보존한다. 관련 경계·통합 회로 검사 13개 통과.

사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pattern-port-plan/plan.json`. 이전 진단기 자체는 사용하지 않고 합성 12쌍의 공간 요약과 방향 라벨로 별도 ridge 패턴을 재학습했다. 이는 공학적 학습 어댑터다. 기존에 관측한 879709 직진 회로의 시각 포트 기울기 방향을 보정으로 써 패턴을 42개 시각 포트로 연결하고, 합성 학습 장면 879701/879702 손실로 gain 0/1/10/100/1000 중 1000을 선택했다. 전체 비행 그래프·기존 시각·몸·운동 시각 가중치는 고정하고 영상/허용 몸 감각 외 정답을 회로 입력에 넣지 않았다. 이전 879709는 이미 보정에 쓰여 독립 검증 자료가 아니다.

계획 후 생성한 새 합성 879713/879714의 경계 제거→활성 좌−우 상대 조향 차는 +0.000000376→+0.005506, +0.000001721→+0.005235다. 상대 목표 +0.004를 넘었고 경계 제거 대조도 있다. 하지만 두 장면의 절대 조향은 모두 음수(예 879713 −0.06139~−0.05589)로 실제 좌우 조향 성공이 아니다. 단 두 절차적 장면에서 선택된 강한 gain, 정적 학습 패턴, 합성 코너에 대한 과적합 가능성이 있고 생리적 타당성도 미검증이다. 학습 loss 1.60557e−5→8.13362e−6. 일반 분류기 출력을 차량 조작으로 직접 쓴 것이 아니라 학습한 시각 투영을 기존 단일 회로 포트에 연결했으나, 그 투영 자체는 생물학적 실측이 아니다.

기존 AC episode-02의 fit 금지 이력 742결정을 재생해 경계 제거와 후보를 비교했다. 제거 기준은 기존 저장 재생과 최대 차 0, 조향 평균 절대 변화 0.01393, 부호 변화 0이다. 기존 기록은 전 시야 과거 몸 문맥 결측·카메라 물리 노출 시각 미검증이며 실제 자율 운전 평가가 아니다. 기존 데이터 fit·배포·차량 제어 0이다. 합성 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pattern-port-fit/result.json`, 실제 이력 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pattern-historical-replay/result.json`, 소스 `scripts/fit_synthetic_pattern_visual_ports.py`, `scripts/replay_synthetic_pattern_historical.py`. 다음은 이 공학 경계가 다른 시점·차량 자세·서킷 영상에서도 의미 있는 시각 방향을 전달하는지와 중립 편향·가감속 영향을 분리해 확인하고, 생물학적 경로의 인과 기여를 검증한다.


## 2026-11-01 합성 공간 패턴의 좌우 포함 실제 이력 평가

사후 선택된 episode-06의 원본 source는 manifest role 이름이 train이더라도 `training_admissible=false`이므로 fit하지 않고 평가만 했다. 합성 패턴 경계 제거/활성으로 741결정을 같은 통합 회로·몸 감각에서 재생했다. 제거 기준은 이전 다중 속도 후보 재생과 최대 차 0, 후보 조향 평균 절대 차 0.01378이지만 부호 변화는 0이었다.

기록된 미래 주행 경로의 20m chord를 사후 평가에만 사용하면 유효 좌 214·우 343, 기준과 후보 모두 음수 557/557, 균형 일치율 0.5로 항상 우측을 내는 기준과 같다. 5m·10m에서도 균형 일치율 0.5다. 미래 chord는 직접 조향 정답이나 레이싱 라인이 아니고 부호 대응은 가정이며, 표본은 하나의 연속된 경로로 독립이 아니다. 기존 시각 노출의 물리 시간 정렬도 검증되지 않았다. 따라서 합성 장면의 상대 차가 실제 코너 구분으로 일반화됐다고 볼 수 없다. actor 입력에 미래 경로를 넣지 않았고 학습·차량 제어·배포 0.

원본 재생 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pattern-selected-history-replay/result.json`, 평가 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pattern-selected-future-chord/result.json`, 소스 `scripts/replay_synthetic_pattern_selected_history.py`, `scripts/evaluate_pattern_selected_history_future_chord.py`. 다음에는 영상 도메인 차이와 절대 출력의 중립 편향을 분리하고, 새로운 적격 데이터 수집과 물리 시간 계약을 갖춘 폐루프 평가가 필요하다.


## 2026-11-01 합성·실제 시각 경계의 입력 분포

합성 패턴 후보의 학습된 기준·가중치를 고정하고 미사용 합성 879713/879714의 각 3조건×41프레임과 fit 금지 실제 episode-06의 741결정을 비교했다. 합성은 원본 망막 강도에 기존 16개 pool 가중치를 적용했고, 실제 이력은 기록된 동일 형식의 pooled scene을 사용했다. 네 시야별 평균과 합성 학습 기준을 뺀 값의 전체 RMS는 합성 0.00475/0.00327, 실제 이력 0.12941이었다. 경계 projection의 tanh 전 입력 RMS는 합성 0.132/0.121, 실제 27.66이고 |입력|>2의 전체 포트·결정 비율은 합성 0%/0%, 실제 86.48%다. delta RMS는 합성 0.0633/0.0580, 실제 0.4792(상한 절댓값 0.5)였다. 이는 실제 영상 이력에서 이 후보 경계가 대부분 포화 구간에 있다는 직접적인 분포 관측이다.

합성 장면 내 세 굴곡 조건은 동일 질감·몸 경로를 공유하며, 실제는 연속된 한 이력이다. 카메라 물리 노출 시각·시야 기하가 완전히 검증되지 않았고, 포화가 부호 고정의 유일한 원인이라는 인과 증거는 아니다. 기록의 라벨이나 미래 경로를 이용해 분포를 보정하거나 fit하지 않았다. 검증용 실제 이력에 맞춰 사후 gain을 고르는 행위는 금지한다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-pattern-boundary-domain-shift/result.json`, 원본 배열 `boundary-distributions.npz`, 소스 `scripts/audit_pattern_boundary_domain_shift.py`. 다음은 실제 fit 금지 이력에 맞춘 튜닝 대신 합성 영상의 질감·조명·차체 자세를 다양화하고 정상 범위·포화 감지를 포함하는 경계를 설계해 새로운 적격 데이터로 검증한다.


## 2026-11-01 합성 기준 시각 대비 제한

`RetinalSpatialContrastBoundary`에 결정 시점의 네 시야 공간 대비 RMS가 지정 한도를 넘으면 축소하는 선택형 제한을 추가했다. 입력을 증폭하지 않고, 과거·미래 정보나 정답 지도를 쓰지 않으며, 기본값은 제한 해제다. 기준과 한도를 state dict에 보존하고 결측 시야는 이전처럼 거부한다. 합성 학습 영상 24개에서만 한도 0.00324179를 산출하고 학습 장면 879701/879702에서 기존과 동일한 gain 후보 중 1000을 골랐다. 한도 조절은 공학적 적응 가설이며 초파리 생리 측정이 아니다. 관련 테스트 14개 통과.

사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-adaptive-contrast-plan/plan.json` 후 생성된 합성 평가 879715/879716의 좌−우 차는 경계 제거 −0.00002499/+0.00000340, 이전 무제한 후보 +0.005897/+0.003471, 제한 후보 +0.005822/+0.002215다. 한 장면에서 상대 목표 +0.004를 넘고 다른 장면에서는 미달하며 모든 절대 조향은 음수다. 실제 fit 금지 episode-06의 같은 제한 후보는 제한 전 |logit|>2 비율 86.48%, 제한 후 1.50%, gain 중앙 0.02417이었다. 기준은 실제 이력으로 조정하지 않았다. 741결정 재생의 조향 부호 변화 0, 20m 미래 경로 좌214/우343의 균형 일치율 0.5로 상수 기준과 같다. 포화 완화는 관측됐으나 방향 판단 성공은 아니다. 미래 chord는 평가 전용이며 직접 조향 정답이 아니고 물리 노출 시간도 미검증이다. 기존 데이터 fit·배포·차량 제어 0이다.

합성 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-adaptive-pattern-fit/result.json`, 실제 경계 감사 `.../20261101-adaptive-contrast-historical-audit/result.json`, 실제 재생 `.../20261101-adaptive-selected-history-replay/result.json`, 평가 `.../20261101-adaptive-selected-future-chord/result.json`. 소스 `scripts/fit_synthetic_adaptive_pattern.py`, `scripts/audit_adaptive_contrast_historical.py`, `scripts/replay_synthetic_adaptive_selected_history.py`, `scripts/evaluate_adaptive_selected_history_future_chord.py`. 다음에는 중립 출력 편향과 시각 방향 부호를 분리하고, 합성 영상 외 적격한 실제 영상의 시각·행동 시간 계약을 검증해야 한다.


## 2026-11-01 적응 경계의 합성 중립 기준 반사실

합성 학습 장면 879701/879702의 직진·속도 3m/s 마지막 10결정에서 적응 후보의 조향 readout tanh 이전 값을 평균해 −0.05863237을 구하고, 실제 이력을 읽기 전 `synthetic-anchor.json`에 저장했다. 이 합성 기준만 기존 episode-06 출력에 `tanh(atanh(turn) − anchor)`로 사후 적용했다. 동일 readout의 상수 기준 변화에 대응하는 출력 전용 반사실로 회로 상태를 다시 구동하거나 actuator에 적용하지 않았고 실제 이력으로 기준을 fit하지 않았다. 속도·영상 조건이 다른 실제 차량에 이 기준이 유효하다는 뜻이 아니다.

20m 미래 경로 유효 좌214·우343에서 기존 후보는 음수 557개·균형 일치율 0.5, 합성 기준 적용 후 양수455·음수102, 좌 방향 재현율 0.523·우 0·균형 일치율 0.2617이었다. 5m/10m도 각각 0.280/0.273으로 상수 기준보다 낮다. 이력은 사후 선택된 단일 연속 주행이며 미래 chord는 조향 정답이 아니고 부호·물리 시간은 검증되지 않았다. 단순한 합성 중립 기준은 이 이력에서 좌우 구분을 해결하지 못했으며 주행 성능 일반화도 불가능하다. 실제 데이터 fit·배포·차량 제어 0이다.

첫 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-adaptive-synthetic-neutral-counterfactual/result.json`은 numpy float JSON 직렬화 오류로 실패했고 그대로 보존했다. 수정 재실행 `.../20261101-adaptive-synthetic-neutral-counterfactual-v2/result.json`, 배열 `outputs.npz`, 소스 `scripts/evaluate_adaptive_synthetic_neutral.py`. 다음에는 중립 편향만 조절하는 반복을 멈추고 방향별 입력 표현과 몸·속도 변화에 걸친 원인 대조 및 실제 적격 데이터의 시각·행동 시간 검증을 진행한다.


## 2026-11-01 통합 회로의 결정 시점 호스트 신선도

`HostFreshnessLimits`를 명시적으로 전달하면 `UnifiedFlightSensorimotor`가 결정 시 현재 몸 수신 나이, 이전 간격에 유지한 몸 수신 나이, 네 시야별 render start 호스트 하한의 나이를 검사하도록 했다. 허용 한도는 몸 최대 50ms, 영상 최대 150ms 이내로만 지정할 수 있으며 오래된 값은 에피소드 상태를 갱신하지 않고 거부한다. 첫 priming에는 이전 몸 수신값이 없음을 허용하며 제어 자격으로 보지 않는다. reset은 이전 몸 receipt도 지운다. 이 기능은 선택형이고 기본 연구 재생은 종전과 같으므로 호출자가 반드시 명시적으로 켜야 한다. 관련 경계·시간·통합 회로 테스트 23개 통과. 결정 완료 뒤 계산 지연에 따른 output-time 신선도는 별도 검사해야 한다.

fit 금지 episode-06 원본 이벤트 741개를 해시 확인해 호스트 시각만 사후 계산하면 첫 priming 제외 740/740결정이 현재 몸·이전 몸 ≤50ms와 네 시야 render 하한 ≤150ms를 통과했다. 현재 몸/유지 몸/가장 오래된 시야의 나이 95백분위는 각각 0.0000497/0.04104/0.06709초였다. 호스트 수신 순서와 상한만 보여주며 물리 프레임 노출 시각·물리 센서 발생 시각·출력 시점 freshness·학습 자격·차량 제어 자격을 증명하지 않는다. 이 이력의 `training_admissible=false`를 유지했고 학습/차량 제어 0이다.

감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-selected-host-freshness-audit/result.json`, 원본 시간 배열 `ages.npz`, 소스 `scripts/audit_selected_history_host_freshness.py`; 구현 `src/flybrain_drive/temporal_body_context.py`, `src/flybrain_drive/unified_flight_sensorimotor.py`, 시험 `tests/test_unified_host_freshness.py`. 다음에는 출력 완료 시점까지 같은 한도를 강제하는 live 통합 계약과 원천 물리 시간 미확인 상태의 명시적 거부·적격한 신규 수집을 다뤄야 한다.


## 2026-11-01 통합 회로의 출력 완료 시점 호스트 검사

`UnifiedHostGuard`는 명시적 `HostFreshnessLimits`를 장착한 단일 통합 회로에만 사용할 수 있는 연구용 추론 보호기다. 첫 결정의 출력은 priming으로 숨기고, 이후 계산이 끝난 호스트 시각에 현재·이전 유지 몸 수신 나이와 네 시야 render start 하한의 나이를 다시 검사한 뒤에만 `GuardedIntent`를 반환한다. CUDA 입력이면 완료 시각을 재기 전에 해당 장치 연산을 동기화한다. 계산 이후 오래된 것으로 판정되거나 시계·입력 오류가 나면 상태가 이미 진행됐을 수 있으므로 결과를 반환하지 않고 오류를 잠그며, 명시적 reset 없이는 재시도하지 않는다. fake host clock을 이용해 시작 시 신선했으나 완료 시 몸 또는 시야가 오래되는 상황, 첫 출력 억제, 오류 잠금·재시작 등을 포함해 기존 경계·통합 회로 검사 총 27개 통과했다.

이 보호기는 별도 opt-in 연구 인터페이스이며 기존 모델을 직접 호출하는 모든 경로에 자동 적용되지 않는다. 출력 객체는 `actor_qualified=false`, `physical_synchronization_verified=false`이며 실제 노출·물리 샘플 시각, 시뮬레이터 권한, actuator 계약이나 실제 차량 통합을 증명하지 않는다. GPU 동기화의 실제 live 지연·AC 동시 부하는 아직 측정하지 않았고, 시뮬레이터 조작·학습·배포 0이다. 소스 `src/flybrain_drive/unified_host_guard.py`, 시험 `tests/test_unified_host_guard.py`. 다음은 기존 background-only 센서 인터페이스와 별도 shadow 연결에서 실제 계산 완료시각 및 실패 순서를 검증하되 실제 제어로 격상하지 않는다.


## 2026-11-01 출력 보호기의 실제 계산 시간 결합 재생

fit 금지 episode-06 원본 네 시야 특징과 몸 입력을 기존 통합 회로·적응 경계 및 `UnifiedHostGuard`로 재생했다. 기록된 결정·수신·render 하한 간격은 원본 그대로 두고 각 결정 시작에서 실제 PC `perf_counter` 경과를 원본 결정 시각에 더하는 오프라인 시계를 썼다. GPU 동기화 포함 guard 호출 계산 시간은 측정하되 입력 준비·후속 CPU 전송·실제 카메라 동시 부하·실시간 pacing은 제외했다. 50ms 몸/150ms 시야 host 기준이고 실제 노출 시간 인증이 아니다.

별도 cold 실행 첫 준비 결정은 864.57ms가 걸려 현재 몸 수신 stale로 차단됐고 출력 0이다. 세 결정의 명시적 사전 준비 후 회로 상태를 reset한 별도 warm 실행은 첫 priming 5.78ms, 둘째 20.39ms에서 유지 몸 수신이 출력 완료 시 50ms를 넘어 차단됐고 공개 출력 0이다. 두 번째 결정 시작 시 원본 직전 몸 수신 나이는 40.43ms이므로 여유가 약 9.57ms였으며 이 단일 측정 계산 시간보다 짧다. 두 실패는 각자의 원본 `result.json`으로 보존했고 자동 재시작하지 않았다. 단 1~2결정의 비무작위 단일 실행이므로 처리율의 분포나 live 동작을 일반화할 수 없다. GPU 초기화와 시각 동시 부하·스케줄 지터를 갖춘 별도 반복 검증이 필요하다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-guard-offline-cold/result.json`, `.../20261101-guard-offline-warm/result.json`, 각 `timings.npz`; 소스 `scripts/replay_unified_guard_offline_timing.py`. 원본 데이터 fit·차량 제어·배포 0이다. 다음은 제한을 느슨하게 바꾸지 않고 계산 경로의 실제 병목을 분해하고, 더 빠른 회로 실행 또는 더 잦은 몸 수신의 시간 계약을 검증한다.


## 2026-11-01 통합 회로 backend 및 CPU 계측

같은 fit 금지 episode-06 처음 200결정을, 준비 3결정 후 명시 reset하고 동일 통합 회로·적응 경계·원본 몸 감각으로 GPU cached CSR과 CPU SciPy CSR에서 각 1회 순차 재생했다. GPU 중앙/95/최대 21.21/24.04/26.26ms, CPU 13.66/26.84/46.63ms. CPU 출력은 기존 GPU 전체 이력 기준과 최대 절대 오차 3.13e−7, GPU는 0이다. 기록된 결정 시각의 현재·유지 몸 및 시야 나이에 이 계산 시간만 더한 가상 판정에서는 첫 준비 제외 GPU 0/199, CPU 17/199가 기존 50ms/150ms 조건을 통과했다. guard 자체를 실행한 판정은 아니고 입력 준비·guard·추론 후 전송·FlyVis 동시 부하·live pacing은 제외했으므로 실제 지연이나 제어 가능률이 아니다. CPU와 GPU의 실행 순서도 무작위가 아니며 흔들림과 소수 표본 한계를 유지한다.

별도 CPU cProfile 계측(계측 오버헤드 존재, 위 benchmark 시간과 혼용 금지)에서 `_decision` 누적 3.152초 중 반복 `_advance_elapsed_tensor` 누적 2.329초, 그 안의 SciPy CSR 행렬-벡터 곱 자체 1.919초였다(200결정·1791 내부 tick). 이 실험 조건에서는 반복 비행 회로의 희소 연산이 큰 병목이다. 중첩된 누적 시간은 더하지 않는다. 별도 생물학적 회로를 만든 것이 아니라 동일 연결과 동역학을 다른 backend에서 계산한 비교다. 성능 개선 후보는 원본 결과·상태를 보존하면서 반복 희소 연산을 최적화하는 방향으로 검증한다.

GPU `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-gpu-offline-benchmark/result.json`, CPU `.../20261101-unified-cpu-offline-benchmark/result.json`, 계측 `.../20261101-unified-cpu-instrumented-profile/profile.txt` 및 `result.json`; 소스 `scripts/benchmark_unified_historical_backend.py`, `scripts/profile_unified_historical_cpu.py`. 원본 데이터 fit·차량 제어·배포 0; 물리 노출/샘플 시간 미검증.


## 2026-11-01 비행 그래프의 정확한 0 항목 생략

고정 비행 회로 9728뉴런의 coalesced 희소행렬 1,311,145개 저장 항목 중 57,083개(약 4.35%)가 정확히 0, 나머지 1,254,062개가 비영이다. 출력 67개를 기준으로 비영 연결을 역추적하면 9466뉴런과 1,204,084개 연결이 도달 가능하고 나머지 262뉴런은 그 닫힌 집합 밖이다. 이 그래프 도달 분석만으로 생물학적 상태를 지우거나 폐루프 동등성을 주장하지 않으며 해당 뉴런을 제거하지 않았다. 구조 감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-sparse-structure-audit/result.json`, `closure.npz`, 소스 `scripts/audit_flight_sparse_structure.py`.

선택형 `scipy_csr_compact_cpu` backend를 추가해 원본 torch 연결 행렬·뉴런·정규화·동역학은 그대로 유지하면서 계산용 SciPy CSR 사본에서 정확히 0인 항목만 생략한다. 기존 backend 기본값은 유지하며 CPU float32·no_grad 제한과 그래프 버전 및 backend 전환 시 캐시 재생성을 강제한다. synthetic 반복 상태 동등성·원본 행렬 불변·cache 교체 등을 포함한 관련 테스트 28개 통과. fit 금지 episode-06 200결정에서 기존 CPU backend와 조향/전진 출력이 비트 단위로 같았고 기존 GPU 기준 최대 차 3.13e−7이었다. 전체 내부 상태의 실제 이력 동등성은 별도로 저장하지 않았다.

단일 비무작위 오프라인 benchmark 중앙/95/최대는 기존 CPU 13.66/26.84/46.63ms, compact 13.81/23.53/46.47ms, 기록 나이에 계산시간만 더한 가상 통과는 둘 다 17/199였다. 측정 간 환경 변동이 있어 속도 개선을 주장하지 않는다. 입력 준비·guard·FlyVis 부하와 live pacing은 제외했다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-cpu-compact-benchmark/result.json`, 소스 `scripts/benchmark_unified_compact_cpu.py`; 실제 데이터 fit·제어·배포 0. 다음에는 원본 그래프와 전체 상태를 유지한 채 반복 CSR 실행과 데이터 이동을 최적화하고 다회 반복·동시 시각 부하를 검증한다.


## 2026-11-01 동일 비행 회로의 병렬 CPU CSR

원래 뉴런·연결·부호·정규화·상태·5ms 내부 시간 단계를 유지하는 선택형 `numba_csr8_cpu` inference backend를 추가했다. 정확히 0인 계산 캐시 항목만 생략하고 CSR의 각 행을 8스레드로 독립 계산한다. CPU float32·no_grad·고정 행렬만 허용하며 호출자의 Numba 스레드 설정은 복원한다. 기존 기본 backend는 바꾸지 않았고 Numba는 선택 경로에서만 필요하다. 첫 JIT 준비는 별도 비용이며 자동 실시간 승인이나 생리 구현 증거가 아니다. 배치 반복 상태·스레드 복원·원본 행렬 및 기존 회로/보호기 관련 테스트 29개 통과했다.

고정 회로의 고립 matvec 합성 벡터 4개·반복 100회에서 8스레드 중앙 0.235ms, SciPy 0.438ms였고 시험 벡터의 결과는 비트 단위 일치했다. 최초 JIT 준비는 해당 실행에서 0.788초였다. thread 순서는 무작위가 아니고 단일 연산 시험이며 시각 부하를 포함하지 않는다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-numba-flight-matvec/result.json`.

fit 금지 episode-06의 같은 200결정과 3회 준비·명시 reset 조건에서 Numba backend의 단일 오프라인 중앙/95/최대 11.65/13.67/15.37ms, 기록된 host 나이+계산시간 가상 통과 85/199였다. 이전 단일 SciPy 비교는 13.66/26.84/46.63ms·17/199이나 순차 비무작위 비교이며 실제 live throughput이나 제어 가능률이 아니다. 별도 두 프로세스의 같은 200결정에서 매 결정 **9728개 전체 회로 상태**와 조향·전진 출력은 SciPy와 Numba가 비트 단위 일치했다. GPU 기준 출력과 최대 차 3.13e−7. 결과 `.../20261101-unified-numba-cpu-benchmark/result.json`, 전체 상태 비교 `.../20261101-unified-numba-fullstate/comparison.json`, 두 backend의 `benchmark.npz`.

입력 준비, guard, 출력 후 전송, 동시 FlyVis, live pacing과 물리 발생·노출 시간은 위 시간에서 빠져 있다. 더 빨라졌더라도 가상 검사의 114/199가 실패했고 실제 제어 사용을 승인할 수 없다. 실제 데이터 fit·배포·차량 제어 0. 구현 `src/flybrain_drive/parallel_csr_cpu.py`, `src/flybrain_drive/vehicle_circuit.py`, 시험 `tests/test_vehicle_circuit.py`, 벤치마크 `scripts/benchmark_flight_numba_matvec.py`, `scripts/benchmark_unified_numba_cpu.py`, `scripts/compare_unified_backend_fullstate.py`. 다음은 동일 회로를 실제 시각 부하와 함께 반복 측정하고, 몸 감각 갱신 주기 및 출력시간 계약을 검증한다.


## 2026-11-01 병렬 단일 회로와 FlyVis 동시 부하

별도 소유 GPU 프로세스에서 기존 FlyVis 사전 학습 네트워크를 사용해 합성 879715의 보존된 네 시야 망막을 반복 입력하고 8×5ms 내부 tick으로 연속 상태를 진행했다. 동시에 별도 프로세스에서 동일 비행 회로의 8스레드 CPU backend로 fit 금지 episode-06 기존 pooled scene·몸 이력 200결정을 오프라인 재생했다. 두 프로세스의 host `perf_counter` 시작·종료 구간을 기록하고 원본 배열 hash를 검증했다. 영상 worker는 소유 프로세스만 종료 요청했고 정상 종료 0, 550개 chunk를 처리했다.

회로 결정 200개 중 영상 GPU host 구간과 실제 겹친 것은 124개였고, 겹친 계산 시간 중앙/95 11.63/13.07ms, 겹치지 않은 76개의 중앙 7.28ms였다. 전체 중앙/95/최대 11.38/13.01/13.90ms, 기존 GPU 기준 출력 최대 차 3.13e−7이다. 기록된 호스트 입력 나이와 계산 시간만 더한 가상 신선도 판정은 첫 준비 제외 86/199다. 겹침과 비겹침은 비무작위·동일 시간대가 아니며 원인이나 일반 성능 차이를 확정하지 않는다. prior 단독 Numba 85/199와의 단일 실행 비교 역시 환경 변동이 있다.

영상 부하는 반복 합성 망막이며 회로 입력은 이미 저장된 과거 pooled scene이다. GPU→CPU 특징 복사·두 경로의 실제 결합, 카메라 수집, guard overhead, live pacing·물리 노출 시각은 포함하지 않았다. 즉 하나의 비행 회로와 FlyVis의 **동시 부하** 측정이지 실제 실시간 통합 운전 검증이 아니다. 학습·차량 제어·배포 0. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-numba-flyvis-load/result.json`, 자식 `visual/visual-timing.npz`, `circuit/benchmark.npz` 및 로그; 소스 `scripts/owned_flyvis_load.py`, `scripts/benchmark_unified_numba_timestamped.py`, `scripts/benchmark_unified_with_flyvis_load.py`. 다음은 실제 GPU 시각 특징의 CPU 전달과 단일 회로 입력을 이어 계산 완료 시각을 검증한다.


## 2026-11-01 합성 FlyVis→CPU 단일 비행 회로 연결 재생

사전 생성 합성 도로 879715의 한 굴곡 조건·41프레임에 대해 실제 pretrained FlyVis를 GPU에서 네 독립 시야로 재생했다. 각 결정 전 이전 망막을 8×5ms 유지해 운동 시각 상태를 진행한 뒤 현재 망막+T4/T5를 GPU에서 기존 공간 요약하고 CPU로 복사해, 같은 단일 비행 그래프의 8스레드 CPU backend와 합성 몸 감각·적응 시각 경계에 전달했다. reference는 동일 저장 망막/운동 반응의 CPU 회로 재생이며 원래 GPU 합성 평가 출력도 비교했다. FlyVis 운동 반응 최대 차 3.58e−7, pooled 특징 1.07e−6, 동일 CPU 회로 출력 6.71e−7, CPU 기준과 이전 GPU 저장 출력 4.58e−7이다. 영상·회로 반복 상태는 에피소드 내 리셋하지 않았다.

회로 reference 재생 및 GPU steady-state 준비 후 첫 영상 프레임을 제외한 40프레임의 중앙/95/최대는 GPU 시각 15.12/16.41/16.73ms, GPU 요약+CPU 복사 1.21/1.76/1.99ms, CPU 몸 context 구성+회로 결정 8.13/8.87/9.09ms, 세 단계를 순차로 합친 시간 24.33/26.10/26.82ms다. 개별 요소의 백분위 합은 전체 백분위와 같다고 가정하지 않는다. 합성 망막은 GPU에 미리 적재했고 검증용 배열 대조 시간, 카메라 촬영·전달, 초기화/JIT, 독립 output-time guard·실시간 pacing·다른 부하를 제외했다. 1개 합성 시퀀스의 비무작위 측정이며 40ms 주기 실시간 보장이나 물리 노출/몸 시각 동기를 증명하지 않는다. 영상 특징이 실제 회로에 전달된 연결 검증이지 코너 판단·완주·생리 검증이 아니다. 학습·차량 제어·배포 0.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-unified-cpu-pipeline/result.json`, 원시 수치 `pipeline.npz`, 소스 `scripts/replay_flyvis_to_unified_cpu_pipeline.py`. 다음은 실제 background 센서 수신·특징 전달의 전체 시간과 output guard를 결합한 비제어 shadow 및 새로운 적격 데이터의 시간 근거를 검증한다.


## 2026-11-01 실제 영상·몸 기록의 현재 단일 회로 재생

기존 `20261031-process-body-owned-pause-live`의 진단 전용 원본 ZIP에서 실제 AC 네 시야 픽셀 253묶음 해시를 확인하고 최초 영상부터 원래 순서·이전 영상 유지로 pretrained FlyVis를 재생해 2135 내부 tick과 공개 105장면의 pooled 특징을 복원했다. 기록 당시 `vehicle_circuit.py`와 현재 소스가 달라 이 사실을 manifest에 명시했다. 당시 코드와 완전히 같다고 주장하지 않고, 기존 shadow의 원래 365출력과 독립 수치 대조해 최대 조향/전진 차 2.09e−7/2.38e−7, 2e−6 초과 0, 원본 몸 worker 정상 종료를 확인했다. 결과·복원 장면 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-real-pixels-flyvis-scene-reconstruction/result.json`, `published-scenes.npz`, 소스 `scripts/reconstruct_live_scenes_for_unified.py`. 이는 기존 shadow의 복원 동등성이지 현재 단일 회로와 옛 회로가 같다는 뜻이 아니다.

복원 장면과 원본 native 최초 수신 365결정을 현재의 하나의 통합 회로·합성 학습 적응 시각 경계에 전달했다. 영상 갱신이 없는 몸 결정은 원본의 마지막 공개 영상을 유지했고 각 시야의 원래 몸 context와 시간 하한은 보존했다. `cached_csr_gpu`와 `numba_csr8_cpu`의 별도 오프라인 실행에서 365×9728 전체 상태 최대 차 8.94e−8, 조향/전진 최대 차 3.05e−7/8.94e−8, 2e−6 초과 0이었다. 두 출력은 현 단일 회로 서로 간 비교이고 과거 shadow 출력을 목표로 맞춘 것이 아니다. 365개 조향은 전부 음수다. 원본은 fit 금지 진단 기록이며 미래 경로나 정답을 actor 입력으로 쓰지 않았다.

이 재생은 원본 이미지 해시→FlyVis→저장된 특징→단일 회로를 단계별로 잇지만, 실시간 카메라 전송·동시 계산·output guard·물리 노출/센서 발생 시간은 증명하지 않는다. 라이브 주행·생리·랩타임 검증도 아니고 학습·차량 제어·배포 0이다. 원본 당시 생체 신호 역할/차량 감각 대응은 공학적 가설이다. 결과 `.../20261101-unified-real-recording-gpu/result.json`, `.../20261101-unified-real-recording-cpu/result.json`, 독립 비교 `.../20261101-unified-real-recording-cpu-gpu-audit/result.json`, 소스 `scripts/replay_unified_real_recording.py`, `scripts/audit_unified_real_recording_cpu_gpu.py`. 다음은 이 연결을 실제 background 수신과 출력 시점 보호기에 잇는 비제어 shadow로 검증하며, 고정된 조향 편향과 코너 판단을 별도로 해결한다.


## 2026-11-01 단일 회로 영상 유지 shadow의 원본 재생

`UnifiedMultirateShadow`는 원본 시야 특징·발행 시각·몸 context가 함께 들어왔을 때만 장면을 복사해 보유하고, 그 사이 각 몸 결정은 기존 영상·원래 시야별 몸 context를 유지해 `UnifiedHostGuard`가 감싼 하나의 반복 비행 회로에 전달한다. 미래 발행, 불완전 장면, 최초 장면 부재를 거부하고 첫 priming 출력은 숨긴다. 출력 완료 freshness 실패가 생기면 guard의 실패 잠금을 유지하고 명시적 reset을 요구한다. 차량 인터페이스가 없는 연구용이며 camera source의 물리 시각을 보증하지 않는다. 가짜 시계·입력 복사·미래 발행 거부·오류 잠금 등 관련 통합 테스트 20개 통과 (`src/flybrain_drive/unified_multirate_shadow.py`, `tests/test_unified_multirate_shadow.py`).

원본 20261031 진단 기록의 실제 재구성 105장면·몸 365결정을 원래 발행 및 수신 순서대로 재생하고 guard 출력 완료 시간을 결정 시각과 같게 둔 이상적 시계 조건에서 첫 priming 1개가 억제되고 364개가 반환됐다. 기존 현재 단일 회로의 CPU 기준 대비 365×9728 상태 및 364출력 최대 차는 0이다. 이는 수치·시간 순서·상태 유지 검증이며 실제 계산 시간이 0이라는 뜻이 아니다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-multirate-ideal-replay/result.json`, 소스 `scripts/replay_unified_multirate_ideal_clock.py`.

별도 재생에서는 최초 세 결정을 준비 연산한 후 회로와 몸 인코더를 명시적으로 새 에피소드로 초기화하고 원본 결정 시각에 실제 warmed CPU guard 처리 시간을 더했다. 365결정 모두 통과·첫 1개 억제·364출력, 상태/출력 기준 최대 차 0, 처리 중앙/최대 2.95/7.30ms였다. 준비 전용 인코더를 재사용한 첫 시도는 시간 역행으로 실패했고 원본 `.../20261101-unified-multirate-timed-replay/result.json`을 보존했다. 수정된 결과 `.../20261101-unified-multirate-timed-replay-v2/result.json`, 소스 `scripts/replay_unified_multirate_timed.py`. 실제 카메라·FlyVis·GPU→CPU 영상 복사·입력 준비·live pacing과 동시 부하는 계산 시간에 포함하지 않으며, 결정마다 기록된 시각으로 기준을 잡은 단일 오프라인 실행이다. 따라서 실제 실시간 통과율이나 물리 노출/센서 시각은 미검증이다. 조향은 이전 분석에서 전부 음수였고 주행 자격이 없다. 실제 데이터 fit·차량 제어·배포 0.

2026-11-01 조회 시점 CSP lifecycle 상태는 live·AI/camera off·약 0km/h·브레이크 1이고 UI/lifecycle/actuator 요청은 0이었다(`.../20261101-unified-shadow-lifecycle-status/result.json`). 이 시점 이후 상태는 다시 확인해야 한다. 다음은 GPU FlyVis 결과를 이 보호 어댑터로 실제 전달하고 원본 수신 시간을 보존하는 background-only 비제어 live shadow를 구현·검증한다.


## 2026-11-01 정지 단일 회로 실시간 shadow 시작 실패 및 입장 검사 수정

백그라운드 정지 AC에서 단일 회로·FlyVis·원본 몸 감각을 비제어 frame-rate shadow로 연결하려고 시도했다. 당시 읽기 전용 lifecycle 확인은 unpaused·live·AI off·camera off·약 0km/h·브레이크1이었다. 준비된 CPU 회로와 FlyVis를 초기화한 뒤 MMF 카메라 요청 `mmf-stream-994ff2fcbe774db3930ea540434dfac2`을 제출했으나 4초 안에 유효 장면이 없어 timeout, camera.close는 소비자가 아직 살아 있다고 보고했다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-frame-shadow/result.json`; 그룹 0, 학습·lifecycle 명령·UI·actuator 0. 실패를 성공으로 바꾸지 않는다.

원인 조사에서 배포된 Lua는 MMF 요청 시작 때 **paused**, offline, AI off, 0.1km/h 미만, gas<.02, brake>.98을 명시적으로 요구함을 확인했다. 새 runner가 paused 조건을 반대로 검사한 사전 검증 오류가 있었다. 종료 확인 시 소유 요청 경로에 stop 파일은 있으나 그 요청의 `result.json` terminal은 없고, producer status의 request ID는 이전 `async-0e99...`다. fresh lifecycle heartbeat는 camera_active=false·unpaused·AI off이지만 이는 새 요청의 terminal을 증명하지 않는다. 요청 파일은 여전히 새 ID를 가리키므로 현재 세션을 단순히 일시정지하면 미처리 요청이 재평가될 가능성이 있다. 소유 요청이나 상태를 덮어쓰지 않았고 자동 재시작·재수집하지 않는다. 이 미확정 요청의 terminal 또는 명시적으로 검증된 안전한 취소가 먼저 필요하다.

`stationary_camera_admission.py`에 paused·offline·정지/브레이크·AI/카메라 off 검증과 이전 카메라 요청의 terminal/확인된 마지막 ID 검사를 추가하고 runner가 카메라 획득 전 호출하도록 수정했다. 카메라 request ID도 객체 생성 직후 기록하도록 바꿨다. paused가 아니거나 이전 요청 terminal이 불명일 때 거부하는 테스트 등 관련 9개 통과 (`tests/test_stationary_camera_admission.py`). live 영상·몸 결합·출력시간·물리 노출 동기 검증은 아직 수행되지 않았으며 모델 배포·차량 조작도 없다. 다음 작업은 미확정 카메라 요청의 안전한 상태 종료를 검증하고, 그 전에 새 요청이나 자동 pause/restart를 하지 않는 것이다.


## 2026-11-01 미확정 카메라 요청 종료와 정지 파일 카메라 shadow

앞선 MMF 요청 `mmf-stream-994ff2fcbe774db3930ea540434dfac2`의 원래 stop 표시와 같은 generation·정지·camera/AI off를 확인했다. 기존 Lua가 동일 요청을 paused에서 받아 다음 update의 stop 표시를 먼저 처리하는 경로를 확인한 뒤, **새 카메라 요청 없이** 허용된 `control_ac_session.py` API로 pause했다. 기존 요청은 소비자 프로세스가 이미 없어 공유 메모리를 열지 못해 0프레임 `failed` terminal을 기록했다. 성공/정상 중지로 재분류하지 않는다. `C:/Users/jeff4/AppData/Local/flybrain/csp-camera/mmf-stream-994ff2fcbe774db3930ea540434dfac2/result.json`, API `.../20261101-owned-camera-cancel-pause/result.json`. 카메라 inactive·정지·브레이크를 확인한 뒤 같은 API로 원래 unpaused 상태를 복원했다 (`.../20261101-owned-camera-cancel-restore/result.json`). 각 lifecycle 요청 1회, UI/actuator 0.

Lua 조건상 일반 yaw 구성의 PNG 파일 카메라는 MMF/exterior 전용 paused 시작 조건에 해당하지 않음을 확인했다. 별도 `validate_stationary_file_camera_admission`에 unpaused live·offline·정지·브레이크·AI/camera off, 같은 generation 및 기존 요청 terminal 확인을 요구하고 관련 검사 10개 통과했다. 비제어 `scripts/record_stationary_unified_file_shadow.py`는 실제 카메라/FlyVis/단일 회로를 연결했으나 첫 선택 영상에서 render 하한이 결정 시 150ms 초과여서 guard가 거부·출력0, 카메라는 18프레임 정상 stop·원본 ZIP 해시 검증했다 (`.../20261101-stationary-unified-file-shadow/result.json`).

별도 실행 `scripts/record_stationary_unified_file_shadow_warm.py`는 카메라 획득 전에 합성 gray 영상으로 망막 샘플·FlyVis·GPU 요약/전달·CPU 회로를 준비하고 FlyVis/회로 상태는 명시적으로 초기화했다. 실제 카메라 capture 12는 첫 상태 준비까지 도달해 출력은 의도대로 억제됐고, capture 13에서는 이전 몸 수신이 결정 전 최소 62.65ms 경과해 50ms 기준에서 거부·출력 0이다. 첫/둘째 시각 처리 host 구간은 7.69/30.77ms다. 한 번의 정지 관측이며 시야·몸의 물리 발생 시간 인증이나 운전 성공은 아니다. 카메라는 16프레임 `stopped`, 샘플러 오류 없음, 87파일 원본 ZIP의 hash 검증·원본 보존; UI/actuator/lifecycle 명령 0. 결과 `.../20261101-stationary-unified-file-shadow-warm/result.json`, 사건 `events.json`, 원본 native `native-copy.json`, ZIP `vision/camera.zip`.

결과·카메라/샘플러 정리가 끝난 뒤 해당 소유 Python 인터프리터가 종료되지 않아, 우선 `Stop-Process`는 PID 조회 불일치로 실패했고, 동일 PID·명령과 terminal 상태를 재확인한 후 소유 프로세스만 CIM Terminate해 return 0을 기록했다 (`process-cleanup.json`). 이후 해당 프로세스는 관측되지 않았고 fresh session은 unpaused·camera/AI off·정지/브레이크 상태였다. 이 강제 종료는 관측 실패를 성공으로 바꾸지 않는다. 실패 사건에서 guard 실제 결정/수신/출력 시각을 별도로 기록하도록 이후 source를 보강했으며 기존 실행 결과는 유지했다. 다음은 동기식 영상 결정에 몸 갱신을 결합하는 대신, 원본 수신을 독립 주기로 단일 회로에 공급하는 shadow를 검증해야 한다. 기존 모델/가중치 배포, 실제 주행, 학습은 없다.

### 2026-11-01 — 몸 ticker와 카메라 실패 상태 대조
단일 회로 상태를 소유하는 몸 ticker와 영상 1개 mailbox를 구현하고 합성 packet·합성 회로로 원본 수신시각, 복사, 덮어쓰기 거부, 준비 출력 억제와 종료를 확인했다. 실제 카메라와 결합한 시험은 아직 없다. CPU Numba의 `nogil` 변경 후 과거 원본 200결정 전체 9728 상태 및 intent 차이는 0 (`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-numba-nogil-fullstate/nogil-parity.json`), 단일 오프라인 중앙 13.09ms이며 live나 동시 부하가 아니다. 관련 검사 25개 통과.

직전 camera 요청의 stopped terminal과 같은 요청의 failed status (`Cannot commit camera metadata`)가 충돌함을 확인했고, 충돌 시 다음 요청을 차단하는 검사를 추가했다. 오류 발생 시점·원인은 미확정이다. 새 카메라 요청, 학습 및 차량 제어는 하지 않았다. 원본 상태를 보존하며 원인 조사 후 실제 통합을 재개한다.

### 2026-11-01 — 실제 몸 수신과 합성 영상의 회로 시험
실제 native 354 copy/297 packet, 기존 9728노드 통합 회로와 합성 장면·합성 render 시간으로 0.8초 동안 준비 1·shadow 출력 73, 오류 0. 모든 출력의 packet 최초 receipt 일치, 출력 기록 시 유지 몸 receipt 최대 21.41ms (`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-ticker-native-full-circuit/independent-audit.json`). 실제 영상·운전·생리 증거가 아니다. 실제 영상 연결 runner는 준비 및 구문 확인만 했고 camera terminal/status 불일치가 있어 실행하지 않았다. 자세한 원인과 다음 단계는 `docs/CONTEXT.md`.

### 2026-11-01 — 보관 영상·현재 몸 동시 처리
FlyVis GPU가 원본 PNG 16묶음을 처리하는 동안 현재 native 몸을 하나의 CPU 회로에 공급하여 priming1·출력59·거부0을 기록했다. 출력 40건이 영상 host 구간과 겹쳤다. 각 영상의 hash, body 최초 receipt, 게시 시각을 원본과 대조한 결과는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-archived-visual-live-body-unified/independent-audit.json`. 영상 host 경계는 합성이라 실제 카메라/물리 동기는 확인되지 않는다. 결과 저장 후 남은 소유 인터프리터는 별도 영수증을 남기고 종료했다. camera 실패 상태 불일치는 남아 있다.

### 2026-11-01 — 카메라 실패 ambiguity와 Python 종료
기존 배포 Lua에서 stopped terminal 뒤 status 쓰기 실패와 후속 clock-reply 쓰기 실패 둘 다 같은 failed status를 생성함을 각각 모의 주입해 확인했다. 원래 사고의 실패 위치는 미상이다. 대상 파일 경로를 오류에 추가하는 미배포 source-pinned 후보를 검증했고 관련 검사 236개 통과. 별도 idle clock 4회 성공은 실제 촬영 성공과 다르다. 반복 영상·몸 shadow 본문 반환과 atexit 진입 후에도 인터프리터가 남아 소유 자식 종료 영수증을 기록했다. 상세 및 경로는 `docs/CONTEXT.md`.

### 2026-11-01 — 실제 카메라 복구 진단 및 신선도 중단
정차 상태의 제한된 카메라 복구 시험에서 실제 네 시야·FlyVis GPU와 현재 native 몸을 단일 CPU 회로에 연결했다. 카메라 15묶음, priming1, shadow8 뒤 이전 시야 하한 나이150.13ms로 거부·중단했다. 다음 영상 처리 37.75ms, 처리 완료 뒤 2.46ms에 거부가 일어났다. 출력 기록 시 최대 시야 146.47ms, held 몸 21.61ms. 독립 원본 대조 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-multirate-recovery/independent-audit.json`. 새 terminal/status stopped이며 과거 metadata 실패 원인은 미확정이다. 물리 시각·실제 주행·생리·학습 검증이 아니다.

### 2026-11-01 — 시야 context 선행 재시험
과거 시야 context를 영상 연산 전에 고르는 변경 후 실제 정차 시험에서 카메라 30묶음, shadow49 출력 뒤 원래 시야 하한 150.15ms로 거부됐다. 마지막 두 최초 시야 하한 간격 69.16ms, 마지막 영상 host 처리 41.47ms였다. 이전 단일 8출력 시험과의 차이는 인과 효과로 볼 수 없다. 원본 hash, 최초 몸 receipt 및 시야 경계 감사는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-multirate-early-context/independent-audit.json`.

### 2026-11-01 — 유지 시각 상태 선행 계산
기존 유지 영상의 5ms FlyVis state를 미리 계산하고 실제 간격의 prefix만 채택했다. 단일 보관 입력 7간격의 state 최대차9.54e-7. 실제 정차 카메라 80묶음 중 55 선택, 준비1·shadow243·거부0, producer completed·process 정상종료. 선택하지 않은 25묶음 원본 보존. 독립 수신/시야 원본 대조 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-held-prefix/independent-audit.json`. 한 실험이며 물리 동기·장기 시각 재현·생리·주행 검증이 아니다.

### 2026-11-01 — 선택 이력 재생과 정차 반복
선택 RGB55묶음·54간격의 FlyVis 두 오프라인 누적 경로는 전체 state 최대차9.54e-7, motion3.58e-7이다. 별도 정차 반복은 카메라80 중59 선택, shadow247·거부0, 21개 미선택 원본 보존. 두 실제 시험 모두 원본 pixels·frame timestamp lower·최초 몸 receipt 독립 검증. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-held-prefix-selected-history-parity/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-held-prefix-repeat/independent-audit-v2.json`. 실제 주행·물리 동기 미검증.

### 2026-11-01 — 정차 shadow 원본 출력 재현
실제 선택 RGB55묶음에서 FlyVis 응답을 복원하고 원래 native/시야 context/244 결정시각으로 단일 회로를 재생했다. 준비1·출력243, 실제 기록 대비 조향/전진 최대차1.86e-7/5.96e-8. 계산 시간은 이상적 clock으로 처리해 실시간 증명이 아니다. 원본·비교 감사 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-held-prefix-recorded-output-replay/independent-audit.json`.

### 2026-11-01 — 정차·교사 주행 기록의 경로 제거 대조
정차 기록에서는 유효 과거 회전값이 모두0이라 시각×몸 회전 경로 제거 효과도0이었다. 별도 기존 AI 교사 주행365결정에서는 최대0.633rad/s의 유효 과거 회전이 있었고, 해당 projection 제거 시 조향 최대차0.0000609·전진0.000000685였다. 운동 시각·공간 대비·추가 몸 feedback도 각각 출력 변화가 있었다. 이는 하나의 모델 내부 기여이며 생리나 주행 정확도 검증이 아니다. 자료 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-moving-teacher-path-ablations/independent-audit.json`.

### 2026-11-01 — 네 과거 기록에서의 조향 부호 편향
양방향 20m 미래 chord가 있는 episode03/04/06의 830결정에서 현재 회로 및 운동 시각·공간 대비·추가 몸 경로 제거 조건 모두 조향 부호 음수, 각 기록 balanced accuracy0.5. episode07은 20m 좌측 표본이 없어 양방향 점수 불가다. 이미 조사한 상관 기록의 사후 평가이며 미래 chord는 정답 조향이나 레이싱 라인이 아니다. 네 시야 과거 몸 context 결측과 training_admissible=false를 유지했다. 원본 재계산 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-four-episode-path-ablations/independent-audit.json`.

### 2026-11-01 — 조향 readout 편향과 순위
기존 고정 b1/b2 readout은 정차244·교사 주행365상태에서 모두 음수 조향을 재현했다. 합성 중립 anchor의 사후 shift는 역사03/04/06의 20m balanced accuracy를 각각0.5/0.5/0.262로 만들었다. 원래 출력의 방향 순위 AUC는0.248/0.378/0.0이나 관련 시간 표본, 미래 chord와 부호 가정에 한정되며 조향 정답·생리 증거가 아니다. 자료 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-readout-bias-decomposition/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-turn-rank-diagnostic/result.json`.

### 2026-11-01 — 합성 영상·몸 경로 분리
이미 평가된 합성 도로 두 개에서 bend +1−−1 최종 조향 차는 +0.00405/+0.00128이지만 양쪽 절대 출력은 음수다. 직선 영상에서 몸 yaw +0.15−−0.15 차는 현재 몸만 -0.02934, 과거 시야 몸만 -0.00000986였다. 특정 synthetic 단위·조건이며 안정화 방향의 오류, 생리 또는 주행 성능을 입증하지 않는다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-controlled-visual-body-path-trace/`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-current-prior-yaw-intervention/`.

### 2026-11-01 — 새 합성 공동 학습 후보 평가 실패
새 fit3/validation2 도로에서 기존 단일 회로의 공간 시각 projection과 readout neutral을 12회 학습했다. Validation MSE는 0.003618→0.001734, 0.003586→0.001950이나 yaw0 양 bend 조향이 모두 음수였다. fit 금지 기존 실제 3개 양방향 기록의 20m 830결정도 후보 전부 음수, balanced accuracy0.5. 사후 future chord는 최적 조향이 아니고 과거 context 결측 및 관련된 기존 기록이다. 첫 baseline 비교 오류의 무효 결과와 수정 재실행을 구분 보존한다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-candidate-historical-path-evaluation-v2/independent-audit.json`.

### 2026-11-01 — 후보 요인과 합성 진단기의 실제 적용 한계
검증 합성 장면에서 공동 학습 후보의 MSE 감소는 공간 projection 단독보다 neutral reference 단독 효과가 더 컸고 bend 분리는 커지지 않았다. 별도의 기존 합성-only retinal probe는 두 synthetic 장면 마지막10프레임 쌍 순위0.9/1.0이나, 실제 기존 3 양방향 이력 20m score 전부 음수·순위 AUC0.586/0.237/0.403으로 일관되지 않았다. 가정된 부호와 상관된 과거 기록의 사후 진단일 뿐 실제 방향 정보 부재·생리·주행 증거는 아니다. 자료 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-candidate-factorization/independent-audit.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-frozen-synthetic-probe-real-history/independent-audit.json`.

### 2026-11-01 — 전방 시야 영역 차이 분해
고정 합성 시각 진단기의 실제 대비 차이는 전방 기여가 큰 편이었고 전방에서 균일 level보다 공간 패턴 항이 컸다. 정차 실제55/합성6 화상에 사후 전방 하단35% 회색 mask를 적용하자 평균 score 차이는 -5.32→-6.74로 작아지지 않았다. 진단기 대수와 비현실적 mask이며 운전석·조명 등 물리 원인을 특정하지 않는다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-frozen-probe-level-pattern-shift/result.json`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-exploratory-front-mask-probe/independent-audit.json`.

### 2026-11-01 — 합성 외형과 공간 대비 제한
두 기존 synthetic 도로의 비물리적 전방 cloud/roadside overlay는 고정 probe 방향 분리를 유지하면서 통합 회로의 좌우 상대 출력 차이를 크게 줄였다. 대비 limiter 제거는 cockpit/exposure 조건의 차이를 키웠으나 cloud/roadside에서는 더 낮췄다. 실제 이력에서 무제한 경계 포화가 있어 제거를 채택하지 않는다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-domain-no-limit-counterfactual/independent-comparison.json`. 실제 시야/생리·운전 검증 아님.

### 2026-11-01 — 시야별 대비 제한 후보
전방 contrast 변화가 다른 시야 gain에 직접 영향을 주지 않는 후보는 단위 검사를 통과했으나, 고정 기존 weights로 두 synthetic 도로의 exposure/cockpit 좌우 상대 출력이 대체로 작아지고 절대 조향은 한쪽에 머물렀다. 따라서 배우 미채택. 실제 fit 금지 장면의 |logit|>2는 경계에 따라 0–1.501% 범위이며 자세한 전역/시야별 비교 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-real-scene-per-view-limit-diagnostic/torch-parity-audit.json`. 생리나 주행 성공 증거 아님.


## 2026-11-01 — 합성 도로 시각 투영과 입력 포트 인과 추적

기존의 정차 후 직선 이동, yaw=0인 합성 도로 2종(881211/881212)과 다섯 전방 화면 개입을 재사용했다. 원본 FlyVis 출력·몸 감각·가중치를 고정하고 운동 시각 opponent 투영과 공학적 공간 대비 투영을 각각 0으로 하는 별도 재생을 비교했다. 기준의 양 굽이 최종 5프레임 평균 조향 차이는 각각 0.0037383/0.0067306, 운동 투영 제거 시 0.0037235/0.0067585, 공간 투영 제거 시 0.0000116/-0.0000321이었다. 이는 이 자극에서의 상대 차이일 뿐 전체 운동 시각 경로 제거가 아니다. 각 절대 조향은 여전히 음수였다. 이전의 다른 실제 교사 주행 기록에서는 운동 투영 제거에 따른 조향 최대 변화가 0.002926이었으므로 모든 조건에서 이 경로가 무의미하다는 결론은 성립하지 않는다.

FlyVis T4/T5 방향 대향 출력의 굽이 간 차이 RMS는 기준에서 0.00994/0.01757이었다. 새 추적에서 마지막 내부 opponent 투영 logit의 굽이 차이 RMS는 0.00153/0.00296이고 공간 투영 logit은 0.14329/0.36214였다. 서로 다른 변환과 스케일을 거치는 값이므로 이 크기만으로 기여도를 직접 비교하지 않는다. 하늘 화면 개입에서 최종 42개 시각 포트 입력의 굽이 차이 RMS가 0.00616→0.00156, 0.01488→0.00369로 줄었다. 포트 입력은 다음 시간 구간의 상태 적분에 쓰이므로 같은 결정의 출력에 즉시 작용한다고 해석하지 않는다.

- 추적: `scripts/trace_synthetic_visual_projection.py`; 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-visual-projection-trace/`. 전체 조건의 intent가 기존 저장 결과와 비트 단위로 일치했고 출력 NPZ SHA-256은 `7924d2143d76a18618f792a406342b6421794139d415dc9faf505040453ba24a`다. `trace-audit.json`에 재검산을 보존했다.
- 경로 제거 비교: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-domain-visual-spatial-lesion/independent-path-comparison.json`; 원시 운동 응답: 같은 폴더의 `raw-motion-difference.json`.
- 이 결과는 사후 선택된 합성 장면에서의 진단이다. 방향 정확도, 생리학적 기능, 실제 차선 판단, 폐루프 안전·유효 완주나 랩타임 최적화를 증명하지 않는다. 추가 학습·차량 조작은 하지 않았다. 다음에는 투영의 시간 지연과 운동 자극 방향성 및 독립 실제 영상에서의 인과 기여를 사전 정의해 검증한다.


## 2026-11-01 — 혼합 시각 자극과 운동 투영 제거 (정정)

기존에 살펴본 합성 FlyVis 네 시야 혼합 자극 두 종류(879511/879512, 자극 계수 -0.23/0/+0.23; 일정한 회전 속도 아님)을 동일한 통합 회로로 재생했다. 몸 감각은 속도 5m/s, 차체 yaw 0으로 유지했고, 원래 투영과 학습된 opponent 투영 가중치만 0으로 한 투영을 각각 독립 에피소드에서 비교했다. 이는 시각 자극과 몸 회전의 인위적 불일치 실험이며 실제 선회 또는 비행 조건이 아니다.

- 마지막 5프레임의 양 자극 계수 간 조향 차이(+0.23 빼기 -0.23)는 기준 -0.003997/-0.003858, 투영 제거 -0.000598/-0.000962였다. 각 자극에서 반대 회전 차이의 대부분이 투영 제거에 민감했다. 앞서 정차 후 직진 합성 도로의 굽이 차이에는 영향이 작았던 결과와 조건이 다르다.
- 기준 절대 조향은 모든 회전에서 약 -0.057~-0.064로 음수였다. 차이가 있다고 올바른 안정화 부호나 운전 정확도가 검증된 것은 아니다.
- 탐색적 1e-6 기준 첫 망막 차이는 0.44초, FlyVis 운동 차이는 0.48초, 투영 제거가 양 회전 조향 차이에 영향을 주기 시작한 시점은 0.56초였다. 입력·결정 간격은 40ms이고 수치 임계값은 사후 선택했으므로 물리적 지연 측정치가 아니다.
- 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-opposite-optic-rotation-lesion/`, 스크립트 `scripts/trace_synthetic_opposite_rotation_lesion.py`, 검산 `audit.json`. 응답 SHA-256 `be04abeb4c23673e04c3cfead5a3e46a3b86d7a5ca707fcdd79f6f01e1411977`; 원본 두 응답 해시와 고정 가중치 계보는 실행 결과에 기록했다. 학습 업데이트·차량 제어는 없었다.

생물학적 기능·실제 카메라·상대풍·시야와 몸 감각의 시간 정합·폐루프 주행은 별도 검증이 필요하다. 이 자료는 과거 확인된 합성 자극이므로 독립 검증 세트나 최종 성능 평가로 사용하지 않는다.


## 2026-11-01 — 혼합 시각 자극·몸 회전 교차 재생 (정정)

기존 합성 혼합 시각 자극 두 원본(879511/879512; 계수 0, -0.23, +0.23; 일정한 회전 속도 아님)을 가상 몸 yaw 0, -0.23, +0.23 rad/s와 교차했다. 각 몸 회전은 0.4초 이후 적용하고 현재와 과거 수신으로 구성해 원래 단일 회로에 입력했다. 시각과 몸 회전 부호의 실제 대응은 보정하지 않았고 이들은 독립 조합이다. 원래 회로와 학습된 운동 시각 opponent 투영만 제거한 회로를 각 조건마다 상태 초기화하여 재생했다.

최종 5프레임 평균 기준으로, 원래 회로에서 몸 yaw 양끝의 조향 차이는 각 시각 조건에서 약 -0.0441~-0.0445였다. 시각 회전 양끝의 차이는 몸 yaw에 따라 약 -0.0037~-0.0042였다. 투영 제거 후 시각 회전 차이는 약 -0.00042~-0.00115였고 몸 yaw 차이는 여전히 약 -0.0441~-0.0445였다. 시각/몸 회전 부호가 교정되지 않은 가상 조건에만 적용되는 차이이며 단위나 생물학적 감도를 비교하는 근거는 아니다. 모든 조합의 조향은 음수였다.

몸 yaw 0의 전체 조향 시계열은 직전 `20261101-opposite-optic-rotation-lesion` 결과와 비트 단위로 일치했다. 두 입력의 교호 차이는 마지막 5프레임 평균에서 약 0.00036~0.00039였으나, 이미 본 두 합성 장면만 사용한 탐색 결과이므로 일반화하지 않는다.

- 실행: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-body-rotation-factorial/`; `audit.json`에 전체 조합과 기존 결과 대조 기록. 응답 SHA-256 `35d344c20db3f3362bf7420ed49e8df517be1caca3f552cc37b8e1d1501a485f`. 코드 `scripts/trace_visual_body_rotation_factorial.py`.
- 학습 업데이트·차량 제어 없음. 실제 몸 센서 부호·상대풍·물리적 시각 정합, 독립 장면 방향 정확도, 실제 폐루프·빠른 유효 완주는 미검증이다. 절대 조향 편향 해소가 우선 과제다.


**정정 근거:** `scripts/replay_flyvis_varied_motion_causal.py`는 계수를 세 축 정현파 회전·병진 진폭에 사용하며, 비영 자극에는 시간 변동 조명도 추가한다. 따라서 위의 양 자극 차이를 순수 반대 yaw에 대한 반응이나 센서 부호 검증으로 해석할 수 없다. 원본 배열과 측정된 출력 차이에는 변경이 없다.


## 2026-11-01 — 일정 시각 회전·몸 회전 교차 재검증

이전에 사용한 혼합 자극의 해석을 정정하고, 다른 기존 FlyVis 원본 `20261101-flyvis-constant-yaw-causal-879601/879602`를 사용했다. 정지 후 구체 카메라의 순수 일정 yaw 0/-0.23/+0.23 rad/s를 가상의 몸 yaw 0/-0.23/+0.23 rad/s와 교차하고, 학습된 운동 opponent 투영 제거를 비교했다. 각 조건은 독립 상태로 시작한다. 두 텍스처는 이전에 관찰한 합성이며 독립 일반화 평가가 아니다. 가상 영상 좌표와 실제 몸 센서의 부호 및 물리 동기는 보정되지 않았다.

몸 yaw 0, 최종 5프레임에서 +시각 yaw 조향 평균 빼기 -시각 yaw 평균은 원래 회로에서 각각 +0.000302/-0.009526, opponent 투영 제거 시 +0.008693/-0.001108이었다. 따라서 투영의 차이에 대한 효과는 각각 약 -0.008390/-0.008418로 비슷했지만, 전체 반응은 두 무늬에서 부호가 반대였다. 몸 yaw 양끝 차이는 약 -0.0445로 나타났다. 모든 조건에서 절대 조향은 음수다. 생리·차량 안정화·유효 완주가 검증된 것이 아니다.

재생 시작부터 초기 정지 11프레임의 시각 양끝 출력 최대 차이는 3.36e-8/4.85e-8이다. 학습 업데이트와 차량 조작은 하지 않았다. 원본 유효성·해시를 확인한 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-constant-visual-body-rotation-factorial/`, 분석 `audit.json`, 코드 `scripts/trace_constant_visual_body_rotation_factorial.py`; 결과 SHA-256 `aa2b7a8278983f4c9839beee2ca0c910aefd9519c9e1d3f012e319fe027d7e14`.


## 2026-11-01 — 순수 회전에 대한 공간 투영 개입

동일한 기존 구체 영상 879601/879602, 시각 yaw 0/-0.23/+0.23 rad/s, 가상 몸 yaw 0과 고정된 통합 회로를 재생했다. 운동 opponent 투영은 유지하고 공학적 공간 대비 투영 가중치만 0으로 만든 결과를 기존 운동 투영 제거 결과와 별도로 비교했다. 원래 회로의 두 재생은 전체 조향 시계열이 비트 단위로 일치한다.

최종 5프레임의 +시각 회전 조향 평균 빼기 -시각 회전 평균은 원래 회로에서 +0.000302/-0.009526이었다. 공간 투영 제거 시 -0.009587/-0.009146, 운동 투영 제거 시 +0.008693/-0.001108이었다. 공간 투영 제거로 이 두 기존 영상 간 차이는 줄었으나, 이는 비선형 회로의 개별 개입으로 두 효과를 더해 분해할 수 없다. 이 경로는 이전 합성 굽이 도로의 좌우 상대 반응에도 크게 기여했으므로 공간 투영 삭제는 채택하지 않았다. 모든 절대 조향은 음수였다.

실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-constant-yaw-spatial-lesion/`, 비교·해시 검증 `audit.json`, 코드 `scripts/trace_constant_yaw_spatial_lesion.py`, 응답 SHA-256 `c4e07f7ba2505c051a9f846b11ff28f6d5e8776096b3ce75b839a1cf1650a434`. 이미 본 합성 영상에 대한 사후 분석이며 학습·차량 제어·생리·실제 주행 검증은 없다. 다음에는 도로 굽이 정보와 자기 회전 반응을 함께 보존하는 경로를 독립 영상과 실제 시간 정합에서 검증해야 한다.


## 2026-11-01 — 합성 도로 굽이와 몸 회전 동시 자극

기존 도로 생성기는 카메라가 회전하지 않아 두 변화를 함께 분리할 수 없었다. 새 `src/flybrain_drive/synthetic_yaw_road_visual.py`는 기존 도로 좌표계(X 전방, Y 좌, Z 상)에서 네 카메라 원점·광선을 함께 yaw 회전한다. 기존 원본 파일과 기존 봉인 SHA는 보존했다. 회전된 카메라 기하와 기존 생성기 비교, 영회전 일치 및 잘못된 각도 거부를 포함한 기존·신규 검사 3개를 통과했다. 이 렌더러는 평면 도로이지 실제 서킷이나 자동차 역학이 아니다.

한 새 텍스처 seed 881301에서 굽이 0/-1/+1과 시간당 yaw 0/-0.12/+0.12 rad/s의 9조건, 각 41프레임·네 시야를 causal FlyVis로 만들었다. 0.4초 정지 이후 월드 전방 3m/s 이동과 별도 처방 회전이 동시에 일어나므로 물리적으로 일관된 차량 주행은 아니다. 회로에는 해당 가상 yaw와 속도만 감각으로 공급했고 굽이 정답이나 위치를 actor 입력에 넣지 않았다. 원래 회로, 공간 투영 제거, 운동 투영 제거를 상태 초기화해 각각 재생했다.

최종 5프레임 굽이 +1과 -1의 평균 조향 차이는 yaw 0/-0.12/+0.12에서 원래 각각 0.002913/0.002260/0.002644, 공간 투영 제거 시 -0.0000014/0.0000186/-0.0000206, 운동 투영 제거 시 0.002913/0.002241/0.002663이었다. 모든 절대 조향은 음수다. 한 합성 도로의 수치적 개입이며 방향 정확도·실제 시각 처리·생리·폐루프 검증은 아니다.

입력 정지 11프레임의 서로 다른 yaw 조건 간 망막은 완전히 같았으나 독립 GPU FlyVis 재생은 최대 3.84e-7 차이가 났다. 첫 검산의 비트 단위 동일 가정은 실패했고 1e-6 허용오차로 재검사한 결과를 `audit.json`에 남겼다. 학습·차량 제어 없음.

- 원본 생성: `scripts/replay_flyvis_synthetic_road_with_yaw.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-road-yaw-881301/`, 응답 SHA-256 `93ad3bfcdb9f1ffba6248312735b00e831778d5c15b8d7f6ca617e70bfab52cf`.
- 회로 개입: `scripts/replay_unified_synthetic_road_yaw_lesions.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-road-yaw-881301/`, 응답 SHA-256 `9ce54a6cefe614667aff5e0098a8d751b64a5f3c53606f97d48aa2660870b72b`.


## 2026-11-01 — 다른 합성 도로의 굽이·회전 재생

첫 조건을 본 뒤 새로운 seed 881302, 곡률 0.025, 굽이 시작 9m를 사용해 같은 9조건과 41프레임을 생성하고 고정 회로를 재생했다. 앞 장면 881301은 곡률 0.018, 굽이 시작 12m다. 동일 렌더러와 공학적 가정의 서로 다른 장면 두 개이므로 실제 주행이나 폭넓은 일반화 검증이 아니다.

마지막 5프레임의 +굽이와 -굽이 조향 차이는 yaw 0/-0.12/+0.12에서 두 번째 장면의 원래 회로가 0.005274/0.006618/0.010551, 공간 투영 제거가 0.0000085/-0.0000127/-0.0000051, 운동 투영 제거가 0.005263/0.006625/0.010569였다. 원래와 두 개입의 모든 절대 조향은 음수였다. 공간 경로에 대한 의존과 편향이 이 두 조건에서 재현됐지만 생리 기능·차량 폐루프 성능을 증명하지 않는다.

정지 구간 yaw 조건의 망막은 일치했고 FlyVis 운동 출력 차이는 최대 3.60e-7로 기존 1e-6 허용오차 이내였다. 원본·응답 해시는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unified-road-yaw-881302/cross-scene-audit.json`, 생성 및 재생 원본은 `20261101-synthetic-road-yaw-881302/`, `20261101-unified-road-yaw-881302/`에 각각 보존했다. 학습 업데이트·차량 조작 없음.


## 2026-11-01 — 실제 기록의 시각 경로 개입과 몸 yaw 시계열

이전에 재구성한 학습 금지 AI 교사 주행의 365결정, 104개 실제 게시 시야 묶음과 동일 고정 회로의 원래/시각 경로 제거 결과를 사용했다. 원본 native packet의 첫 수신 시각을 재검증해 기존 body21 변환으로 yaw와 속도를 복원하고 원래 결정 및 시각 게시 시각에 연결했다. 영상·몸 감각은 학습에 사용하지 않았고, 원본 결과·NPZ·body ticker·native bytes·bridge 해시를 확인했다. 단일 과거 기록의 사후 분석이다.

속도 >1m/s, |body yaw| >0.02 rad/s의 양수 구간 95결정(27개 영상), 음수 구간 149결정(43개 영상)은 각각 실제로 **연속된 한 구간씩**이다. 평균 yaw는 +0.290/-0.127 rad/s였다. 원래 조향에서 공간 투영 제거 조향을 뺀 평균은 +0.012759/+0.012810으로 거의 일정한 양수였고, 운동 투영 제거에 대한 같은 차이는 -0.000784/+0.001176으로 두 구간 평균이 yaw와 반대였다. 시점별 부호 반대 비율은 운동 투영이 100%/77.2%였으나 연속 샘플은 독립 사례가 아니므로 통계적 일반화 근거로 쓰지 않는다. 공간 투영 제거 효과가 일정한 부호라는 관찰은 기존 한쪽 편향과 관련될 가능성이 있지만 이를 원인으로 입증하지 않는다.

결정 시점 기준 body 첫 수신 최대 연령 6.081ms, FlyVis 시야 게시 최대 연령 43.434ms였다. 이는 호스트 수신·게시 시각이며 카메라 물리 노출이나 몸 센서 물리 시각 동기는 검증하지 않는다. 시각 정보 이력과 비선형 상태가 영향을 주므로 같은 시점 비교는 국소 도함수나 운전 정확도 검증이 아니다. 회로 실행·학습·차량 제어는 새로 수행하지 않았다.

분석 `scripts/audit_archived_visual_body_path_alignment.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-archived-visual-body-path-alignment/`의 `result.json`, `aligned.npz` 및 `time-blocks.json`. 배열 SHA-256 `8eed4779f80c28bc4bbc5c431e0f0b7002a951b11209e9d48fffe7695d9d62a8`. 다음에는 서로 독립적인 주행·다양한 회전과 물리 노출시각을 확인해야 한다.


## 2026-11-01 — 과거 네 주행 이력의 시각 경로와 몸 yaw

기존에 살펴본 episode 03/04/06/07의 원본 학습 금지 기록에서 고정 통합 회로의 시각 경로 개입 결과와 원래 body21 이력을 시각별로 대조했다. dataset의 역할 표시에 train이 있는 기록도 원본 `training_admissible=false`이며 이 분석은 학습을 수행하지 않았다. 저장 경로 해석표로 고유 원본을 열고 원본 events·NPZ·결정 시각과 개입 결과 해시를 확인했다. 모델 재생 당시 네 시야의 과거 몸 감각은 모두 결측으로 처리됐다.

전체 결정의 원래 조향 minus 공간 투영 제거 조향 평균은 네 기록에서 각각 -0.007197/-0.008561/-0.007896/-0.005205였다. 앞서 별도 교사 주행의 양·음 yaw 구간에서 +0.012759/+0.012810이었던 것과 부호가 다르다. 영상·시점·주행 조건 및 시야별 과거 몸 맥락이 다르므로 어느 하나를 원인으로 확정할 수 없다. 2,966결정 전체에서 원래 조향 양수는 0개였다.

속도 >1m/s 및 |yaw| >0.02 rad/s로 나눈 공간 경로 효과는 4개 기록 모두 양·음 yaw 그룹에서 평균 음수였으나 그룹별 9~420결정, 1~5개 연속 구간으로 불균형하다. 운동 경로 효과는 기록·방향에 따라 달랐고 직전 단일 주행의 반대 yaw 경향을 일반화할 수 없었다. 전체 숫자와 연속 구간 수는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-visual-body-path-alignment/result.json`에 있다. 선택된 과거 기록·상관된 시간 이력의 사후 분석으로 독립 검증이나 생리적 증거가 아니다.

코드 `scripts/audit_historical_visual_body_path_alignment.py`, 정렬 배열 SHA-256 `4dc9b6efceaad8ccf4068bd48e5d6e204bb1d636cd451c5e9e6d707fe5f15f9d`. 학습 업데이트·차량 조작 없음. 고정 중립 보정으로 해결된다고 결론내리지 않으며, 시간 정합된 독립 영상과 감각에서 별도로 확인한다.


## 2026-11-01 — 합성 굽이·회전 공동 학습 가능성 시험

고정 단일 회로의 공간 대비 projection과 lateral readout reference만 학습 대상으로 하여, 사전 기록한 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-combined-road-yaw-fit-plan/plan.json`대로 합성 도로 881301의 3굽이×3yaw, 41프레임 이력을 사용했다. 몸·겹눈·기존 운동 투영과 나머지 회로는 고정하고, 원래 시각·재귀 상태를 유지하되 마지막 5프레임 전에서 학습 그래프를 detach했다. 목표 `0.02×굽이 - 0.1×몸 yaw`는 물리적 정답이 아닌 합성 공학 가설이다. Adam 2회, 제한된 projection/reference 변화, 실제 데이터 fit 0, actor 자격 false, 차량 제어 0이다.

학습 손실은 0.00366381→0.00358676였다. 학습이 끝난 뒤 다른 합성 도로 881302를 열었고, 이 장면도 이전에 이미 살펴본 것이므로 맹검·독립 검증으로 부르지 않는다. 가설 MSE는 0.00355203→0.00340095였지만 9개 조향 평균 모두 음수이고, +굽이 minus -굽이 차이는 yaw별 0.0052736/0.0066178/0.0105510에서 0.0052743/0.0066187/0.0105533으로 거의 변하지 않았다. 후보의 기준 대비 출력 변화는 0.0013124~0.0013181로 거의 공통 중립 이동이었다. 따라서 이전 중립 이동 실패를 해결하지 못해 후보를 채택하지 않았다.

평가 전 기준 회로는 별도 기존 881302 전체 재생의 마지막 평균과 최대 1.49e-8 차이로 재현됐다. `scripts/train_synthetic_combined_road_yaw_candidate.py`, 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-combined-road-yaw-candidate-fit/`, `audit.json`, candidate SHA-256 `fe4084d0e11ca31524421d5161d01acd978bba9700a75bcd30e37ec147df6eaa`. 합성 목표에 대한 제한적인 시험일 뿐 실제 운전·신경 생리·물리 시간 정합·유효랩은 미검증이다.


## 2026-11-01 — 고정 중립의 쌍별 굽이 차이 학습

직전 합성 후보가 주로 중립값만 이동시킨 문제를 피하기 위해 사전 기록한 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-paired-bend-fit-plan/plan.json`대로 reference를 고정하고 공간 대비 projection만 갱신했다. 학습 데이터는 합성 도로 881301, 각 yaw에서 +굽이 조향 minus -굽이 조향을 가설 목표 +0.04로 맞추는 제곱 오차다. 이는 합성 도로의 공학적 가설이며 실제 레이싱 정답이 아니다. 각 쌍의 차이를 먼저 평가한 후 동일 시퀀스를 다시 재생해 쌍별 미분을 누적했고 원래·학습 재생 값의 최대 차이는 매번 0이었다. 신경 수치 상태는 보존하고 학습 그래프는 마지막 5프레임 전에서 detach했다. 실제 기록 학습·차량 제어 없음.

3회 갱신에서 쌍별 손실 0.00139841270→0.00139839944. reference 변화 0, 공간 투영 가중치 변화 L2 0.03582·최대 0.00355였다. 학습 후 이전에 이미 본 다른 합성 도로 881302의 출력 변화는 최대 5.55e-7이었고 9조건 조향 모두 음수였다. 기준의 굽이 차이 0.0052736/0.0066178/0.0105510은 후보 0.0052733/0.0066177/0.0105510으로 사실상 변하지 않았다. 기존 881302 기준 재생과 최대 차이 1.49e-8. 이는 이 설정에서 유효한 개선이 없다는 결과이며, 정확한 원인을 포화·희소 전달·시간 절단 중 하나로 단정할 수 없다.

후보 미채택·actor_qualified=false. `scripts/train_synthetic_paired_bend_candidate.py`, 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-paired-bend-candidate-fit/`, 진단 `audit.json`, 후보 SHA-256 `7ad27a2ec092760165f48fb327435b99f220cb292fcfa2a593fd7a8751ae025a`. 두 장면은 이미 보았으므로 독립 실제 검증이 아니다. 다음은 상태·시간별 기울기 및 시각 경로의 인과 기여를 확인하고 절대 조향 편향을 해결해야 한다.


## 2026-11-01 — 재귀 학습 시간 범위 진단과 재학습

기존 합성 도로 881301, 몸 yaw 0의 +굽이 minus -굽이 마지막 5프레임 평균에 대해 공간 projection 가중치의 기울기를 측정했다. 상태 수치는 유지하면서 역전파 이력만 5/10/20/40프레임으로 자르고 각 시퀀스의 원래 출력 동일성을 확인했다. 쌍별 기울기 L2는 각각 1.0136e-6/4.2947e-6/8.6981e-6/8.7910e-6이고 40프레임 방향과 cosine은 0.9382/0.9905/0.999993/1.0이다. 따라서 해당 단일 자극에서 직전 5프레임 역전파는 더 오래된 기여를 크게 놓친다.

40프레임 기울기 방향의 중앙 유한차분은 epsilon 0.001에서 6.52e-5로 자동미분 8.79e-6과 불일치했다. float32 출력의 작은 차이가 포함될 수 있어 실패를 보존하고 epsilon 0.1로 별도 재실행한 값은 8.12e-6이었다. 두 점만의 확인이라 완전한 수치 검증은 아니며, 이 현상이 학습 실패의 유일 원인이라는 증거도 아니다. `20261101-synthetic-truncation-gradient-audit/`, `20261101-synthetic-truncation-gradient-fd01/` 각각 결과·gradient NPZ 보존.

그 근거로 같은 fit 장면·중립 고정·쌍별 가설 목표·3회 갱신을 20프레임 역전파로 반복했다. 계획 `20261101-paired-bend-long-gradient-plan/plan.json`, 코드 `scripts/train_synthetic_paired_bend_long_window.py`. 학습 손실 0.00139841270→0.00139837625, 재생 일치 최대 0, reference 변화 0. 이전에 본 881302의 최대 출력 변화는 3.78e-6(5프레임 학습 5.55e-7), 모든 조향은 여전히 음수였다. 굽이 차이는 기준 0.0052736/0.0066178/0.0105510에서 0.0052737/0.0066184/0.0105515로 사실상 그대로여서 후보 미채택. `20261101-paired-bend-long-window-fit/audit.json`, 후보 SHA-256 `2ac618667733a793d7a5b48251dc8523e5f2f42fbd0f619ea0bd3344af6c2df8`.

기존에 본 합성 장면의 감도 분석이며 실제 생리·독립 영상·주행 검증이 아니다. 학습은 합성 자료만 사용했고 실제 데이터 fit·차량 제어는 없었다. 다음에는 포트→재귀 상태→근육 출력 감도와 시각 자극 방향성 자체를 확인한다.


## 2026-11-01 — 학습 후보의 공간 경로부터 근육까지 추적

합성 도로 881301의 yaw 0·양 굽이 원본 이력을 원래 단일 회로, 20프레임 역전파 학습 후보, 공간 projection 제거 조건으로 독립 상태에서 재생했다. 후보는 미채택·actor_qualified=false이고 real-data fit과 제어는 없다. 각 결정의 공간 projection logit, 최종 시각 포트, 9,728개 회로 상태, 근육 및 intent를 기록했다. 원래 출력은 직전 독립 재생과 전체 시계열이 비트 단위로 일치했다.

마지막 5프레임에서 후보−원래 변화 RMS는 공간 logit 0.000436, 시각 포트 0.0000177, 전체 상태 0.00000124, 근육 1.20e-9였다. 같은 각 단계에서 공간 투영을 완전히 제거한 변화 RMS의 각각 약 0.24%, 0.23%, 0.33%, 0.13%다. 서로 다른 단위와 차원의 절대값을 직접 나눠 전달 이득으로 해석하지 않으며, 각 단계별 같은 단위의 두 개입만 비교한다. 해당 결정의 시각 포트는 다음 시간 구간에 적용되므로 즉시 영향으로 간주하지 않는다.

기준 공간 projection logit에서 |값|>2 또는 >4인 비율은 모두 0이었다. 이 한 합성 입력의 해당 tanh가 강하게 포화돼 학습이 막혔다고 설명할 근거는 부족하다. 후보의 공간 가중치 변화가 이 입력에서 만들어낸 logit 변화 자체가 전체 투영 제거보다 작고 그 작은 변화가 이후 상태에도 전달된다는 관찰이다. 신경 경로의 취약성·학습 목적·최적화 규모·상태 시간, 생물학적 정합 중 유일한 원인은 특정하지 못했다.

스크립트 `scripts/trace_synthetic_candidate_spatial_path.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-candidate-spatial-path-trace/{result.json,audit.json,traces.npz}`, 응답 SHA-256 `608f82d052919cdc0d262f771bccc1a1a8245be4b82fe1b1a0847a62c29bda17`. 이미 본 합성 한 장면이며 실제 입력·생리·자율 주행·가감속/변속·빠른 유효랩 검증이 아니다.


## 2026-11-01 — 공간 가중치 허용 범위의 국소 반사실

기존 합성 도로 881301의 40프레임 쌍별 조향 기울기 g와 직전 학습의 가중치당 ±0.02 제한을 사용했다. 현재 지점에서 1차 근사의 최대 변화는 0.02×sum(|g|)=4.361e-6이다. 기존 합성 가설의 목표 차이 0.04와 기준 0.00291345의 간격은 0.03709로 이 선형량의 약 8,504배이나, 이 목표는 공학적 가정이고 이 수치는 전역 비선형 상한이 아니다.

기울기 부호를 따라 모든 가중치에 ±0.02를 준 두 점을 원래 회로에 직접 재생한 결과 fit 장면의 차이 변화는 +4.329e-6/-4.462e-6, 이미 본 881302 평가 장면에서는 +0.609e-6/-0.818e-6이었다. 두 기준은 이전 저장 회로 출력과 최대 4.48e-9 이내로 일치했고 모든 절대 조향은 음수였다. 두 점만으로 상자 내부 전역 최적성이나 더 넓은 가중치·다른 회로의 가능성을 판정할 수 없다.

`scripts/audit_spatial_gradient_box_counterfactual.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-gradient-box-counterfactual/{result.json,audit.json}`. 결과 파일과 직접 계산은 확인됐으나 확인 시 원래 실행 핸들 83649, Python PID 45992/39796이 살아 있어 프로세스 종료 코드는 미확정이다. 사용자 데이터 학습·차량 제어 없음. 합성 입력의 경로 민감도일 뿐 생리·실제 주행 검증이 아니다.


## 2026-11-01 — 고정 상태·중립에서 근육 readout 혼합 범위

기존 `FlightMotorReadout`은 b1/b2 좌우 차이에서 각 중립 기준을 빼고 softmax 혼합한 뒤 양의 gain과 tanh로 조향을 만든다. 이 양의 혼합에서 출력은 두 근육 채널을 각각 100% 선택한 극단 사이에 있어, 고정된 상태·중립에서 두 채널 값이 모두 음수면 어떤 혼합도 양수 조향을 만들 수 없다. 이는 실측 비행 근육 기능이 아닌 기존 공학 대응 readout에 대한 수학적 성질이다.

합성 도로 기존 상태 82결정 중 74개, 기존 학습 금지 AI 교사 주행 기록의 재생 상태 365결정 중 351개는 두 중립 차이가 모두 음수였다. 나머지 8/14개는 부호가 다르거나 0이고, 두 값 모두 양수인 경우는 없었다. 샘플은 시간적으로 연관돼 독립 사례가 아니다. 고정된 실제 기록 상태를 원본 motor readout으로 다시 평가한 intent의 최대 차이는 0이었고 근육값에서 별도로 재계산한 조향의 최대 차이는 7.45e-9였다.

합성 양 굽이의 마지막 5프레임에서 b1/b2 각각 단독 선택의 평균 조향은 -0.07842/-0.03325와 -0.07519/-0.03081이었다. 극단별 좌우 굽이 차이는 0.003236/0.002436이다. 현재 gain·중립과 현재 두 상태에서 혼합만 조정해서는 양쪽 절대 조향을 얻을 수 없다는 결론이지, 중립 보정이나 회로 상태를 바꾸는 모든 방법이 불가능하다는 뜻은 아니다. 실제 영상 조건별 중립과 경로 방향을 함께 검증해야 하며, 고정 중립 이동만으로는 이전 역사 평가가 개선되지 않았다.

`scripts/audit_fixed_reference_muscle_mix_bounds.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fixed-reference-muscle-mix-bounds/`의 결과와 배열 SHA-256 `639c9f3cf8274aede617f9e692cfd34abfe7e9c4c601411191578ddd4a68422e`. 학습·차량 제어 없음. 생리·실제 운전·물리 노출시각 검증 아님.


## 2026-11-01 — 같은 에피소드 정지 기준의 코너 정보 손실

합성 도로 881301/881302의 각 굽이·yaw 이력은 첫 11프레임(0.4초)에서 화면·가상 몸이 정지한다. 이때 결정 8~10의 기존 조향을 atanh 좌표에서 평균하고, 그 에피소드의 나머지 출력에서 기준으로 빼는 출력 전용 사후 반사실을 비교했다. 11프레임 확보 전에는 기준을 적용할 수 없고 원본 회로 상태·checkpoint·차량 제어는 바꾸지 않았다. 이 두 장면은 이미 보았고 정지 센서는 합성 가정이다.

몸 yaw 0에서 +굽이 minus -굽이 조향 차이는 원래 0.002913/0.005274에서 기준 제거 후 0.000215/-0.001874가 됐다. 두 경우 모두 원래 좌우 굽이 절대 조향은 음수, 보정 후에는 둘 다 양수였다. 정지 중에도 보이는 도로 형태에 따른 회로 반응을 중립으로 빼 버리는 방식이므로 코너 선행 정보를 보존한다는 목표에 맞지 않아 채택하지 않았다. 이는 해당 회로·단일 합성 계열의 반사실이며 모든 생물학적 감각 적응이 불가능하다는 뜻이 아니다.

실제 episode03/04/06/07에서 기존 정지 기준에 필요한 유효 body angular 3개, speed, 네 바퀴 회전의 동시 가용 여부를 확인한 결과 가용 결정 자체가 각각 0개였다. 따라서 속도≤0.1km/h·바퀴≤0.05rad/s·각속도≤0.02rad/s, 11프레임·0.4초 조건의 실제 기준 창도 없었다. 이는 필수 감각이 결측이라는 뜻이지 차가 실제로 정지하지 않았다는 뜻이 아니다. 모두 원본 fit 금지 이력이다.

합성 스크립트 `scripts/audit_synthetic_stationary_reference_road.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-stationary-reference-road/`, 출력 SHA-256 `3b36da9feb538db89f8f8de41c49fdf4faa1bbde0f243d3c7d7493bb8ddcdbaa`. 실제 가용성 감사 `scripts/audit_historical_stationary_reference_windows.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-stationary-reference-windows/result.json`. 학습·차량 제어·생리 및 실제 중립 검증은 하지 않았다.


## 2026-11-01 — 정지 기준의 결측 원인과 원본 벡터 재검사

앞의 body21 정지 판정에서 네 이력 모두 필수 값 동시 유효 수가 0인 이유를 원본 패킷과 계약에서 확인했다. 각 이력에서 yaw·차속·네 바퀴 회전은 모든 결정에서 유효하지만 roll/pitch 각속도는 0개였다. CSP 원본 `localAngularVel`의 세 값은 존재한다. `configs/ac-virtual-body-v1.json`이 roll/pitch 축과 부호의 기하 검증 실패로 `angular_valid=[false,false,true]`를 명시해 `VirtualBodyEncoder`가 해당 축을 무효로 만든 것이다. 따라서 센서 원본이 기록되지 않았다고 한정해서는 안 된다. 실제 roll/pitch 값과 부호는 여전히 검증되지 않았고 actor 입력 마스크를 바꾸지 않았다.

축 이름·부호를 추측하지 않고 원본 세 값의 유클리드 크기 ≤0.02 rad/s, 차속 ≤0.1km/h, 각 바퀴 ≤0.05rad/s, 원본 첫 packet 수신 나이 ≤50ms를 이용해 정지 후보를 보수적으로 재검사했다. 크기 조건은 이전 max-성분 조건보다 엄격하다. 원본 패킷 해시·중복·시각을 확인한 host 기준 후보 결정은 episode03 1개, episode04/06/07 0개였고 고유 packet·시야 11개 이상·0.4초 이상의 창은 네 이력 모두 없었다. 결과는 원본 네 이력의 수신 정보와 이 기준에 한정되며 차가 실제로 정지하지 않았다는 증거는 아니다. 특히 몸 수신과 카메라 물리 노출의 동기는 여전히 인증되지 않았다.

감사 `scripts/audit_native_stationary_candidate_windows.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-native-stationary-candidate-windows/result.json`. 학습·차량 제어 없음. 다음에는 roll/pitch 축·부호를 독립적으로 검증하고 유효 실제 몸 감각 및 시야의 물리 시각 정합을 확보한다.


## 2026-11-01 — 별도 저속 기록의 renderer·몸 각속도 비교

기존 geometry review의 roll/pitch는 낮은 움직임과 불일치로 기준에 실패했다. 추가로 기존 학습 금지 `20261101-low-speed-fourview-stationary-prefix-probe`의 보존 camera ZIP·metadata 및 native packet SHA를 검증했다. 각 시야 생성 시작 하한보다 먼저 수신된 원본 packet 중 나이≤50ms를 선택하고, renderer의 3D 방향 변화에서 SO(3) 각속도를 계산해 native `[2,0,1]` 후보와 비교했다. 원본과 renderer는 같은 시뮬레이터이며 회전·속도 데이터의 물리 발생시각과 카메라 노출 동기는 검증되지 않았다.

기존 5프레임 중심 비교는 75개 적격 영상 중 원래 0.02~0.2초 간격 조건을 만족하는 구간이 0개여서 실패했다. `20261101-low-speed-renderer-body-rotation-audit/result.json`에 보존. 그 이후 3프레임 중심 차분을 별도 탐색으로 실행하여 73개 시간 연관 창을 얻었다. 기존 수치 기준(범위≥0.02, slope 0.9~1.1, r≥0.9, RMSE<0.005)을 기계적으로 적용하면 roll 범위 0.0385/slope 0.0359/r 0.090/RMSE 0.00688, pitch 0.0678/2.105/0.913/0.01188로 실패했고 yaw 0.0550/0.986/0.933/0.00418만 통과했다. 3프레임 방법은 데이터를 확인한 뒤 추가한 탐색이며 독립적인 생리·물리 검증이 아니다.

롤/피치 mask는 그대로 false이고 추정 축/부호를 actor에 개방하지 않았다. 코드 `scripts/audit_low_speed_renderer_body_rotation_threeframe.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-renderer-body-threeframe-audit/`, 배열 SHA-256 `9a269a10ffc818558a562df28ea288030fff48e54ff38096bef30fbf7ba9410d`. 실제 학습·차량 조작 없음. 더 높은 품질의 축별 자극과 독립된 시각·센서 시간 검증이 필요하다.


## 2026-11-01 — 단일 전방 뷰 기하를 이용한 각속도 재검산

직전 3프레임 비교는 front world look과 up에 다른 렌더 시점의 left world look을 합쳐 기준 행렬을 만들었다. 뷰 간 시차로 움직임이 왜곡될 가능성을 발견해, 동일 front 화면의 world look과 up만 사용하고 좌측 축은 `cross(up, look)`으로 구성했다. 시간 역시 묶음의 script clock 대신 front render 시작 시각을 사용했다. 엄격한 직전 native 수신 조건·0.02~0.2초·3프레임 비교와 원본 ZIP 해시 검증은 동일하다. 본 수정은 모델 입력이 아니라 원본 renderer 진단에만 적용한다.

73개 동일 창에서 native 값은 두 방법이 정확히 일치했고, renderer 각속도의 RMS 차이는 roll/pitch/yaw 0.00000551/0.0000143/0.000143 rad/s였다. 단일 뷰 비교의 roll slope 0.0359·r 0.0901·RMSE 0.00688, pitch 2.1057·0.9130·0.01189, yaw 0.9875·0.9319·0.00421이었다. 앞선 값과 거의 같아 이 저속 이력의 roll/pitch 불일치가 단순 뷰 혼합에서 왔다고 보기 어렵다. 다만 시뮬레이터 내부 수치끼리의 비교이고 source clock·실제 노출/몸 센서 시간 동기는 미검증이며 이 분석은 사후 탐색이다.

스크립트 `scripts/audit_low_speed_single_view_rotation.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-low-speed-single-view-rotation-audit/`의 원본 비교와 `comparison-audit.json`, 배열 SHA-256 `97b981e075e7357952c19f380853958aeb955b51c7428c51651e4547de1132f1`. 기존 roll/pitch validity는 변경하지 않았고 학습·차량 조작은 하지 않았다.


## 2026-11-01 — 시간 가중 각속도 비교와 인과적 적분 함수

과거 저속 이력의 native raw packet에 저장된 자세 변화와 angular 후보의 관계를 보았다. 동일 원본끼리이고 자세 Euler 미분은 작은 기울기·축 결합 조건의 근사다. 143개 겹치는 3패킷 창에서 중심 순간값을 쓸 때 roll/pitch slope 0.347/0.846, r 0.691/0.986이었고, host 수신 간격으로 사다리꼴 시간 가중한 값은 slope 1.105/1.073, r 0.995/0.998이었다. 독립 센서 검증은 아니다. 원본 `20261101-native-attitude-rate-exploration/`.

같은 저장 구간의 단일 전방 renderer 회전과도 두 endpoint 사이 **전체** native packet을 시간 가중해 비교했다. 73개 창에서 roll slope 1.035·r 0.989·RMSE 0.000395이나 범위 0.01872 rad/s로 기존 0.02 기준 미달; pitch 1.0646·0.9998·0.001212·범위 0.1795; yaw 1.0130·0.9985·0.000645·범위 0.0566이다. 결과를 본 뒤 사용한 사후 적분이고 renderer와 native는 동일 시뮬레이터이며 render host 하한은 물리 노출 시각이 아니다. pitch의 이 한 기록 내 수치 통과가 독립 물리 기준이나 생리 검증을 뜻하지 않는다. 기존 롤·피치 입력 마스크는 false로 유지했다.

`src/flybrain_drive/causal_native_angular_average.py`에 정렬된 원본 축 벡터를 지정 구간에서 사다리꼴 평균하는 독립 진단 함수를 추가했다. 결정까지 도착한 첫 수신만 사용하고 양 끝 packet 부재·50ms 초과 공백·역순/중복을 거부하며 축 등록이나 유효 판정을 하지 않는다. 불규칙 선형 적분·미래 bracket/외삽 거부·중복/긴 공백 거부 검사 3개 통과. 보존 기록의 각 구간 끝을 덮는 첫 packet이 도착한 시점을 결정으로 정한 재생 73개는 사후 적분 배열과 SHA-256까지 같았다. endpoint 이후 수신 대기는 중앙 20.34ms·최대 41.76ms이며 실제 live latency와 별개다. 회로·actor에는 연결하지 않았다.

스크립트 `scripts/audit_native_attitude_rate_exploration.py`, `scripts/audit_low_speed_integrated_native_rotation.py`, `scripts/audit_causal_available_integrated_rotation.py`; 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-available-integrated-rotation/result.json`, 배열 SHA-256 `4810d4a5fda7397e2c87754b3869b74147de4a9733e853c4d7acd5036aabd323`. 학습·차량 제어 없음. 충분한 롤 자극, 다른 세션, 독립 물리 시간 및 실제 폐루프를 확인해야 한다.


## 2026-11-01 — 이전 두 주행에 대한 구간 적분 재검사

이전에 geometry review에 쓰인 `ac-local-launch-trial-20261029-t1002`와 `ac-snapshot-launch-trial-20261029-t989`의 원본 events 및 camera ZIP SHA를 확인했다. 별도 저속 기록을 보고 고른 사후 방법으로, 같은 전방 화면 방향·위쪽에서 5프레임 회전 변화를 구하고 그 양 끝 host render 하한 사이의 native first-receipt 각속도를 원래 순서로 적분했다. 적분에 필요한 끝 bracket 수신 이후를 결정 시점으로 놓았고 packet 간격 >50ms와 bracket 부재는 거부했다. 이력은 이전에 본 기록이며 같은 시뮬레이터와 중첩 창을 사용한다. renderer·native 물리 발생 시각은 검증되지 않았다.

각 이력에서 991/999창이 계산됐다. roll은 범위 0.0596/0.0602 rad/s, slope 0.704/0.719, 상관 0.830/0.836, RMSE 0.00315/0.00300으로 기존 수치 기준에 실패했다. pitch는 범위 0.0231/0.0338, slope 1.013/1.023, 상관 0.9991/0.9970, RMSE 0.000126/0.000234로 이 사후 방법의 수치 기준에 들어왔다. yaw도 수치 기준에 들어왔다. 이 결과만으로 roll 원인을 단정하거나 pitch 축·부호를 물리적으로 인증하지 않는다. 기존 actor 마스크는 그대로다.

끝 bracket의 수신 대기 중앙은 16.29/16.59ms였으나 이는 저장된 host 하한에 대한 값이지 실제 live 지연 인증이 아니다. `scripts/audit_historical_integrated_rotation.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-integrated-rotation/`, 비교 배열 SHA-256 `dc1e9d244ef04aa3af52fb47ca14e92f010ff84df4e13a1fa989186c97e0f6c2`. 학습·차량 제어·생리 검증 없음.


## 2026-11-01 — 롤 불일치의 시간 구간 추적

이전 두 기존 주행의 구간 적분을 원본 해시를 다시 확인하고 재실행하면서 각 5프레임 구간의 capture 번호, 호스트 시간, 렌더 간격 및 native packet 최대 간격을 저장했다. 기존 비교의 native/renderer 배열 전체는 재생과 비트 단위로 일치했다.

각 주행의 롤 오차가 큰 상위 1%(10창)는 전체 롤 제곱오차의 97.31%/96.24%를 차지했고 capture 8~19/9~20의 겹치는 초반 한 구간에 모였다. 이 열 개 창은 동일 사건을 공유하므로 열 개 독립 실패가 아니다. 이 상위 창의 최대 packet 간격 중앙은 40.36/40.45ms로 전체 중앙 40.76/40.63ms와 비슷했다. 단순히 긴 host packet 간격 하나로 원인을 입증할 수 없으며 원본의 물리 발생·노출 시각은 아직 미검증이다.

사후에 capture 30 이후만 계산하면 roll slope 1.029/1.024, 상관 0.9979/0.9964, RMSE 0.000341/0.000414였다. 이는 오류를 본 뒤 제외한 탐색적 수치일 뿐 유효성 기준을 통과했다는 뜻이 아니다. 초반 이상이 실제 움직임·렌더/센서 시간차·상태 초기화 등의 무엇 때문인지는 미확정이고 모든 원본 창을 유지했다. 롤·피치 입력 마스크도 그대로다.

코드 `scripts/trace_historical_integrated_rotation_windows.py`, 결과·기록 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-rotation-window-trace/{result.json,comparisons.npz,residual-audit.json}`. 학습·차량 조작 없음. 독립 주행과 사전 정의한 조건으로 검증해야 한다.


## 2026-11-01 — 초반 롤 불일치의 native 자세·각속도 분해

직전 두 주행의 저장된 5프레임 창을 원본 packet SHA 및 first-receipt 시간에 대조했다. window host render 하한 양 끝에서 native roll 각도를 보간해 `-Δroll/Δt`를 만들고 구간 평균 native 각속도 및 renderer 회전과 비교했다. Euler 미분은 작은 기울기·축 결합 조건에서만 body 회전율의 근사이며 보간과 동일 시뮬레이터 원본은 독립 물리 기준이 아니다. 초기 capture≤25와 이후 capture≥30은 오차를 보고 고른 사후 구분이다.

초기 21개 중첩 창의 native gyro 대 native 자세 미분 RMSE는 0.02132/0.01974 rad/s, native 자세 미분 대 renderer RMSE는 0.000956/0.001624 rad/s였다. 이후 창에서는 각각 0.000411/0.000406 및 0.000370/0.000362였다. 따라서 이 기록에서 renderer 한쪽만 다르다고 설명하기보다 초반 native 각속도와 자세의 시간적 불일치도 확인해야 한다. 원인이나 해당 센서의 물리적 오류가 증명된 것은 아니다.

가정한 ZYX Euler와 기존 q/r 축 대응이 맞고 수신 샘플의 보간이 유효하다면, 롤 Euler 변화율과 body roll 사이의 교차항은 보수적으로 `tan(|pitch|) × (|q|+|r|)` 이하로 제한된다. 초기 창별 최대 피치·q·r로 구한 최대 교차항은 0.000525/0.000367 rad/s였고, gyro−자세 차이는 21/21 및 20/21창에서 이보다 컸다. 조건부 수치일 뿐 축 계약이나 시간 동기의 증거가 아니며, 샘플 사이의 미관측 값도 배제하지 못한다.

`scripts/audit_native_renderer_roll_chain.py`와 `scripts/audit_roll_cross_axis_bound.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-native-renderer-roll-chain/` 및 `20261101-roll-cross-axis-bound/`. 배열 SHA-256 각각 `f629a0833ef59974f2a71a47b71f3aa909ad0de0191b2414c17fc67368ad6eb4`, `22ef6ccb0b20f7811b9df77632d9d3c75136c7185ae966280355e3280698b831`. 기존 mask 유지·학습/제어 없음. 독립 물리 시각과 초기 상태 진단이 필요하다.


## 2026-11-01 — 롤 이상 구간의 출발·조작 맥락

직전의 사후 선택 capture≤25 구간과 원본 native packet·조작 요청 시각을 해시 검증해 대조했다. 각 이력의 약 0.96초 구간에서 native 속도는 0.0026→최대 2.02km/h, 0.0038→최대 2.56km/h 범위였고 가속 입력은 최대 0.280/0.276, native 조향값은 -0.467~+0.212/-0.476~+0.218 범위였다. 각각 학생 조향 요청 24건이 기록됐지만 요청만으로 실제 적용·원인 관계를 단정할 수 없다. native 롤 각속도 절댓값 최대는 0.0513/0.0482rad/s였고 해당 시점 속도는 1.19/1.45km/h였다.

별도 이전 저속 시험의 구간 적분 73창에서는 롤 오차 최대 0.00146rad/s, 앞 15창 최대 0.00000308rad/s였다. 이 시험은 촬영·프레임 창 조건과 정지/출발 순서가 다르고 롤 자극 범위도 0.0187rad/s여서 두 원본 주행과 직접적인 동등 비교는 아니다. 초반 불일치가 출발 및 조작과 시간상 겹치지만 초기화·센서 물리 시각·제어 등 어느 것이 원인인지 증명하지 못했다.

`scripts/audit_early_roll_launch_context.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-early-roll-launch-context/`, 원본 24개씩의 배열 SHA-256 `22dba4a058d757960dcafa0f683d1b1a416d64bc56a440779eff27115a273d91`. 학습·차량 조작 없음. 롤/피치 mask 변경 없이 독립된 정지-출발-조향 대조와 물리 시각을 확인한다.


## 2026-11-01 — 추가 세 저속 이력의 각속도 구간 적분

앞선 저속 이력의 분석 후 같은 차·세션에서 보존된 세 기록 `20261101-fast-body-fourview-probe`, `20261101-long-reference-new-history-probe`, `20261101-low-speed-fourview-prefix-reverse-probe`를 선택하고 동일한 전방 카메라 3프레임·원본 native packet 시간 가중 적분을 적용했다. 새 차종·독립 물리 센서가 아니고 이미 본 기록으로, 같은 시뮬레이터 및 중첩 시간 창에 대한 사후 탐색이다. 세 camera ZIP과 events, 분석 배열 해시 및 RMSE 재계산을 독립 스크립트로 대조했다.

- 빠른 몸 수신 기록 69창: roll 범위 0.02689, slope 1.0211, r 0.9844, RMSE 0.000647; pitch 범위 0.07044, slope 0.9904, r 0.9983, RMSE 0.000612. 각 수치가 기존 기준에 들어왔다.
- 장기 기준 기록 77창: roll 범위 0.02324, slope 0.8567, r 0.9336, RMSE 0.001126으로 slope 미달; pitch 범위 0.01082로 최소 0.02 미달.
- 역순 조향 기록 73창: roll 범위 0.02331, slope 1.0905, r 0.9804, RMSE 0.000719; pitch 범위 0.16661, slope 1.0269, r 0.9981, RMSE 0.001079. 각 수치가 기존 기준에 들어왔다.

구간 끝을 덮는 최초 packet 수신까지 host 하한 기준 대기는 중앙 5.76/6.44/20.07ms, 최대 12.58/12.14/41.08ms였다. 이는 저장된 순서의 오프라인 계산이며 실시간 기한이나 물리 노출 정합을 증명하지 않는다. 롤·피치 축 마스크는 계속 false이며 actor에 새 경로를 연결하거나 학습·차량 조작하지 않았다.

`scripts/audit_additional_low_speed_integrated_rotation.py`, `scripts/audit_additional_rotation_crosscheck.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-additional-rotation-audit-{1,2,3}/` 및 `20261101-additional-rotation-crosscheck/result.json`. 사전 고정된 새로운 세션·차종과 충분한 축별 자극, 독립 물리 시각이 필요하다.


## 2026-11-01 — 구간 끝 packet과 영상 게시의 겹치는 대기

앞서 적은 packet 대기 5.76~20.34ms 중앙값은 front render 시작의 host 하한부터 구간 끝 bracket packet까지의 차이였다. 영상 묶음이 그보다 나중에 게시되면 이 둘은 겹치므로 packet 대기 전부를 영상 게시 후 추가 지연으로 더하면 안 된다. 네 기존 저속 이력을 원본 bytes와 동일한 3프레임/적분 조건으로 다시 읽고 `max(0, packet_first_receipt - camera_received)`를 계산했다. native·renderer 배열은 직전 결과와 각각 비트 단위 동일했다.

빠른 몸 수신 두 기록은 각각 69/77창 모두 게시 시 필요한 끝 bracket이 이미 도착해 추가 대기 0ms였다. 일반 수신의 역순 조향 73창 중 14창은 추가 대기가 있었고 중앙 0·최대 7.056ms, 정지 prefix 73창 중 16창은 중앙 0·최대 8.858ms였다. 해당 네 영상 게시−render 하한 중앙은 각각 38.41/43.92/33.98/34.17ms다. 3프레임 renderer 검증의 끝 영상 게시를 기준으로 한 저장 이력 계산이며 actor에 필요한 전체 네 시야·FlyVis 연산 종료·가장 오래된 감각·물리 노출 및 실시간 스케줄을 검증한 것은 아니다. packet에 결정 이후 값을 사용한 것이 아니라, 끝 bracket 수신과 카메라 게시 중 나중 시점을 가능한 결정 시점으로 본다.

`scripts/audit_rotation_packet_camera_timing.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-rotation-packet-camera-timing-{1,2,3,4}/`, `...-1/crosscheck.json`에 arrays 해시 및 재검산 기록. 본 진단은 입력 축 유효화·학습·차량 제어를 하지 않았다.


## 2026-11-01 — 원본 native 각속도 이력과 순차 적분

`src/flybrain_drive/native_angular_history.py`에 bounded history를 추가했다. 원본 `NativePhysicsSample`의 바이트를 재해석해 ID·각속도를 대조하고, 중복 ID는 같은 바이트일 때 첫 수신을 유지한다. 수신/packet 역순, 바이트가 달라진 중복, 동일 수신시각의 서로 다른 packet은 거부한다. reset은 명시적이며, 저장 범위 초과 후 이른 구간·50ms 초과 공백·구간을 덮지 못하는 경우는 결측 오류로 남긴다. 평균은 결정 시각까지 수신된 원본 native 축 순서만 반환하며 양끝 packet ID·첫 수신·해시도 함께 기록한다. 실제 물리 발생시각이나 롤·피치 유효성을 주장하지 않고 actor에 연결하지 않는다.

단위 검사 7개는 중복 receipt 비갱신, 변경 중복·역순·reset, 미래 bracket 거부, 유실 구간 및 불규칙 선형 적분을 확인했다. 기존 빠른 패킷 기록과 역순 저속 기록을 시간순으로 각각 462/145 packet 공급하면서 카메라 게시와 필요한 packet 도착 중 나중 시각에 69/73개 창을 재생했다. streaming native 평균은 기존 일괄 계산과 각 창 최대 차 0, native·renderer 배열도 비트 단위 동일했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-native-angular-history-replay-{1,2}/`, 첫 폴더의 `crosscheck.json`.

이는 보존 이력의 호스트 수신 기준 오프라인 재생이다. 실제 노출·센서 물리 시각, 롤/피치 축, live 병목, 폐루프 주행 검증은 아니며 기존 센서 mask·조향·가감속/변속 경로는 바꾸지 않았다. 학습과 차량 제어 없음.


## 2026-11-01 — 등록된 yaw의 시야 구간 준비

`src/flybrain_drive/view_yaw_interval.py`는 기존 검증된 `VirtualBodyEncoder`의 yaw 등록이 유효할 때만 두 렌더 시작 host 하한 사이의 native yaw 평균과 각도 적분을 `NativeAngularHistory`에서 계산한다. 각 시야별 시각은 호출자가 따로 공급하며 시작<끝≤카메라 수신≤결정, 최대 0.2초, 결정까지 양끝 packet 수신을 요구한다. 양끝 packet ID와 최초 수신시각·계약 SHA를 결과에 넣는다. 축 미검증 roll/pitch를 반환하거나 `NavigationBodyObservation` 원본 값을 덮지 않으며, host 하한을 물리 노출시각으로 인증하지 않는다. diagnostic·actor_qualified=false.

불규칙 packet 적분, 누락·미래 packet, 순서·중복·변조, reset, yaw 등록 마스크를 포함한 관련 검사 10개 통과. 저장된 빠른 수신과 역순 조향 이력을 카메라 게시와 endpoint packet 수신의 늦은 시점에 순차 재생해 각각 69/73구간에서 원본 일괄 평균과 yaw 최대 오차 0이었다. 첫 기록 69행의 양끝 render 하한, 카메라·결정·packet receipt, 평균·적분 및 packet ID를 별도 저장하고 수신≤결정 및 적분=평균×시간을 다시 확인했다.

코드 `scripts/replay_view_yaw_interval_recorded.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-view-yaw-interval-recorded/`, 배열 SHA-256 `64e2a6557c6cd4c30eb3603a7dbc352c80eab2fc039df857bf953899e25ccb1d`. 같은 차·세션의 이미 본 기록, host 시계 기반 오프라인 준비이며 live·신경 생리·실제 폐루프는 미검증. 기존 단일 회로·센서 mask·학습 및 차량 제어는 변경하지 않았다.


## 2026-11-01 — 네 시야별 yaw 구간과 게시 시점 가용성

기존 빠른 몸 수신 및 역순 조향 저속 기록의 원본 camera metadata·ZIP·native bytes SHA를 확인하고 연속 capture마다 front/left/right/rear의 각 render 시작 host 하한을 따로 사용했다. 앞뒤 해당 시야 하한 사이 yaw 평균·각도를 `NativeAngularHistory`와 기존 검증 yaw 계약으로 준비하고, packet·시야 수신 순서를 보존했다. camera 게시 즉시 결정하는 가상 정책과 부족한 endpoint packet을 최대 50ms 기다리는 오프라인 이벤트 재생을 비교했다. 기다릴 정책은 보존된 미래 수신 시각으로 사후 시뮬레이션했으나 각 계산에는 해당 결정까지 실제로 도착한 packet만 공급했다.

빠른 수신 기록 72개 연속 영상쌍은 즉시 및 제한 대기 모두 각 시야 70/72, 네 시야 모두 준비 70/72였고 추가 대기 0ms였다. 일반 수신 기록 76쌍은 게시 즉시 front/left/right/rear 각각 60/51/44/33, 네 시야 모두 33/76이었으며 제한 대기에서는 각각 74/76, 네 시야 모두 74/76이었다. 대기 중앙 1.70ms·95백분위 19.92ms·최대 21.91ms, 41쌍에 대기가 발생했다. 두 기록의 나머지 2개 쌍은 packet 간격 >50ms라 네 시야 모두 명시적으로 결측이다. 이미 준비된 구간은 두 정책의 평균이 정확히 같았다.

`intervals.npz`는 원본 시야별 하한·결정·양끝 수신과 결측 마스크를 기록한다. 유효하지 않은 값이 NaN이고 양끝 receipt가 구간을 덮으며 결정 이전인 것을 별도 검산했고 관련 검사 10개가 통과했다. 코드 `scripts/replay_four_view_yaw_intervals.py`; 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-four-view-yaw-intervals-{1,2}/`, 첫 폴더의 `crosscheck.json`. 두 기록은 같은 차·세션의 이전에 본 데이터다. 이는 host 하한·게시 기준의 오프라인 결과이지 물리 노출·실제 live 대기·최종 FlyVis/회로 freshness 검증은 아니다. actor 연결, sensor mask 변경, 학습 및 차량 제어 없음.


## 2026-11-01 — 하나의 비행 회로에 선택형 시야별 yaw 구간 입력

`IntervalYawVisualBoundary`는 검증된 `ViewYawInterval`의 시야별 평균 yaw와 같은 시야의 운동 대향 흔적을 조합해 기존 단일 `UnifiedFlightSensorimotor` 시각 포트를 조절하는 0 초기화 공학 가설이다. 별도 운동 회로나 독립 조향 모델이 아니다. 기존 순간 몸 감각을 구간 평균으로 바꾸지 않으며, 기존 호출은 새 경계를 사용하지 않는다. 명시적 실험 메서드에서만 작동하고 물리 동기·actor 자격을 주장하지 않는다.

각 시야별 이전 render 하한과 interval 시작, 현재 render 하한과 interval 끝, 원본 body 계약 SHA, packet 양끝 receipt 및 현재 결정 순서를 확인한다. 첫 결정에는 이전 시야가 없으므로 유효 interval을 받을 수 없고, 시야 결측은 다른 시야에서 채우지 않는다. 시간·계약 불일치는 상태를 바꾸기 전에 거부한다. reset 시 이전 시야 시각도 지운다. 결측 입력은 단순 0 효과이지만 결측 자체를 유효 0 관측으로 바꾸지 않는다.

합성 fixture와 실제 규모의 고정 graph를 사용한 `20261101-interval-unified-neutral-synthetic-v2`에서 기존 경로 대비 0가중치와 전 시야 결측의 intent·9,728개 재귀 상태 최대 차이는 모두 0이었고 기존 보존 결과와도 동일했다. 임의 비학습 가중치와 합성 가상 yaw를 넣은 연결 시험은 상태 최대 차 0.02842, intent 최대 차 0.000340이었으나 이 자체는 방향 정확도·생리 기능·운전 개선이 아니다. 기존 경로·보호기·시간·이력 및 신규 시야 마스크를 포함한 관련 검사 40개 통과. 직전 시야 시간 일치 검사를 추가한 첫 시험에서 합성 테스트가 부동소수점으로 별도 계산한 시각을 사용해 실패했으며, 실제 보존된 이전 시각을 재사용하도록 고친 뒤 통과했다.

코드 `src/flybrain_drive/interval_yaw_visual_boundary.py`, `src/flybrain_drive/unified_flight_sensorimotor.py`, `scripts/audit_interval_unified_neutral_synthetic.py`; `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-interval-unified-neutral-synthetic-v2/`, 응답 SHA-256 `f2bee2cccefe07c8ce5b1911cbdb7658ea5e76fe2aefedc6dc946504cb9a7f25`. 새 경계 학습·실제 actor·차량 조작은 없었다. 네 시야 실제 이력의 포트 기여·물리 노출·live 지연과 조향/가감속/변속·빠른 유효랩은 미검증이다.


## 2026-11-01 실제 저속 기록의 yaw 구간 단일 회로 재생

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-real-interval-unified-replay/result.json`: 원본 저속 진단의 462결정과 네 시야를 고정된 비행 graph에서 재생했다. 기존 저장 intent와 재생 결과는 정확히 일치했고, 0 초기화 입력과 전체 결측 입력의 intent 및 전체 재귀 상태도 기존과 정확히 같았다. 시야별 구간 70개가 준비됐고 packet 간격 초과로 2개 영상쌍(시야별 총 8건)은 결측으로 처리했다. 결과의 `valid_by_view`는 조건별 준비 가능 수이며 `missing` 조건에 실제 전달한 입력은 전부 결측이다.

고정 난수·미학습 가중치의 변화는 intent 최대 1.3188e-6, 내부 상태 최대 0.0001251이었다. 이는 경로 반응 진단일 뿐 생리적 기능, 학습, 제어 향상 또는 완주의 증거가 아니다. 실제 물리 노출·센서 시각 정합과 live 검증은 하지 않았고 이 이력은 학습 금지다. 차량 제어와 optimizer 업데이트는 없었다. 원본·기존 가중치는 보존했다. 재생 스크립트는 `scripts/replay_real_interval_unified_graph.py`다.


## 2026-11-01 구간 yaw 경로의 학습 가능성

`experimental_sensory_interval_training_decision`을 추가해 진단 구간 어댑터와 기존 단일 재귀 비행 graph를 연결한 상태로 gradient를 유지한다. 기존의 시각 시간·시야별 구간·결측·전 회차 render bound 검사는 동일하게 적용하며, 호출자는 데이터 학습 허가를 별도로 검증해야 한다. 임의 합성 graph에서 출력까지 0이 아닌 유한 gradient와 detach 후 동일 수치 상태를 검사했다. `tests/test_unified_flight_sensorimotor.py`, `tests/test_interval_yaw_visual_boundary.py` 총 15개 통과. 첫 검사는 합성 fixture의 연결되지 않은 근육 출력을 선택해 gradient 0으로 실패했으며, 해당 fixture에서 연결된 intent를 검사하도록 정정했다. 실제 flight graph의 학습 효율이나 주행 성능을 보여 주는 결과가 아니다. 실제 기록을 학습에 사용하거나 차량 제어를 보내지 않았다.


## 2026-11-01 실측 비행 graph에서의 구간 입력 gradient

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-real-graph-interval-gradient-1/result.json` 및 `scripts/audit_interval_real_graph_gradient.py`: 고정된 실측 연결 기반 비행 graph와 기존 checkpoint를 사용하고 합성 도로 1개, 가상 yaw .12rad/s, 41개 시점을 적용했다. 최종 intent의 첫 성분에서 구간 어댑터로의 gradient 최대 절댓값은 7.3215e-6, L2는 9.1468e-5, 0이 아닌 원소 10752개였다. 모든 구간을 결측 처리한 대조에서는 gradient가 0이었다. detach 후 재귀 상태 수치가 일치했다. 옵티마이저 갱신과 차량 제어는 없었다. 합성 영상·가상 yaw는 실제 센서 정합이나 생리 기능을 입증하지 않으며, 이 검사는 학습 성능·완주·랩타임의 증거가 아니다.


## 2026-11-01 합성 yaw 안정성 학습 대조

`scripts/train_interval_synthetic_stability.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-interval-synthetic-stability-1/result.json`: 기존 실측 연결 기반 graph와 나머지 가중치를 고정하고 합성 도로의 두 굽이 및 yaw ±0.12rad/s에 대해 같은 굽이의 yaw 0 출력과 마지막 10시점 평균 조향을 비교했다. yaw 구간 어댑터만 Adam 2회 갱신하고 가중치를 ±.02에 제한했다. 학습 도로의 네 조건 평균제곱 차이는 3.52177e-7→2.98315e-7, 이미 이전에 검토한 다른 질감 도로는 1.67290e-6→1.57428e-6이었다. 체크포인트 SHA-256 `e1bcbce7965fe5203c22081a096aef74acf719749dc243a063455bb2a3861b1a`.

이는 yaw 변화의 출력 영향을 줄이는 공학 가설이다. 실제 상황에서는 회전에 대응하는 조향이 필요할 수 있으므로 억제 자체를 바람직한 운전이나 생리 기능으로 간주하지 않는다. 비교 도로는 독립 최종 평가가 아니고 실제 영상·몸 감각으로 학습하지 않았다. 물리 시각과 live 입력, 가감속·완주·랩타임 성능은 미검증이며 actor 배포·차량 제어는 없다.


## 2026-11-01 합성 학습 어댑터의 실제 기록 전이 진단

`scripts/replay_trained_interval_real_record.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-trained-interval-real-record-1/result.json`: 합성 학습 checkpoint SHA `e1bcbce7965fe5203c22081a096aef74acf719749dc243a063455bb2a3861b1a`를 기존 저속 기록 462결정에 적용했다. 학습 당시와 평가의 다른 body feedback·시각 context 설정에 유의한다. 이전 기본 출력, 0 가중치, 결측 조건의 출력과 상태는 정확히 일치했다. 학습 어댑터를 유효한 구간에 적용했을 때 intent 첫 성분 최대 변화 1.0431e-6, 둘째 5.9605e-8, 내부 상태 최대 변화 1.6376e-5였다. 462개 가운데 135개 intent에 수치 차이가 있었으며, 기준 첫 intent는 전부 음수였다. 시야별 70개 구간이 준비됐고 두 영상쌍은 packet 공백으로 결측이다.

이 진단은 실제 기록에 대한 미세한 반응만 보여 주며 필요한 조향 변화의 보존이나 개선을 판정할 수 없다. 학습과 평가 설정 불일치, 물리 시각 미검증, 단일 저속 기록, 비폐루프라는 한계가 있다. 실제 기록은 fitting에 쓰지 않았고 차량 제어와 optimizer 업데이트도 없었다.


## 2026-11-01 학습 설정 복원 후 실제 기록 진단

`scripts/replay_trained_interval_same_config.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-trained-interval-same-config-real-1/result.json`: 합성 학습 당시의 centered speed 몸 감각, opponent, spatial contrast와 원래 view context 가중치들을 출처·hash 확인 후 복원했다. 평가 입력은 기존의 저속 실제 기록으로, 합성 학습 입력과는 다르다. 462결정에서 0 및 결측 조건은 해당 설정의 기준 출력과 전체 상태에 정확히 같았다. 학습한 구간 어댑터의 첫 intent 최대 변화는 1.35042e-6, 둘째는 5.96046e-8, 내부 상태 최대 변화는 1.44690e-5였다. 156결정에 수치 차이가 있었고 기준 첫 intent는 462개 모두 음수였다. 시야당 구간 70개가 준비되고 2개 영상쌍은 packet 공백으로 결측이다.

이 결과는 앞선 다른 설정 재생의 설정 차이를 해소한 오프라인 진단이다. 양방향 제어 능력, 회전 대응의 적절성, 실제 물리 시간·실시간 자격이나 폐루프 주행 성능을 입증하지 않는다. 실제 자료 fitting, optimizer 업데이트와 차량 제어는 없다.


## 2026-11-01 실제 기록의 감각 경로 제거 대조

`scripts/audit_same_config_real_sensory_ablation.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-same-config-real-sensory-ablation-1/result.json`: 합성 학습 설정과 같은 고정 graph에서 저속 실제 기록 462결정을 재생하고, 한 번에 하나씩 몸 되먹임 어댑터 가중치를 0, 운동 시각 채널을 0, 공간 시각 어댑터 가중치를 0으로 바꾼 반사실을 비교했다. 기준은 직전 저장 baseline과 정확히 일치했고 첫 intent 범위는 -0.06854~-0.02575였다. 각 제거의 최대 intent 변화는 각각 0.0001957, 0.009951, 0.012093이며 모두 462결정에서 음수였다.

몸 어댑터 제거는 기존 몸 입력 전체 제거가 아니고 운동 채널을 인위적으로 0으로 만드는 것은 실제 센서의 결측을 모사하지 않는다. 이 단일 기록의 개별 제거 비교는 상호작용하는 경로의 단일 원인이나 운전 기여를 판정하지 않는다. 실제 기록은 학습 금지, 물리 시각 미검증이며 optimizer 업데이트·차량 제어는 없었다.


## 2026-11-01 동일 설정 실제 기록의 조향 readout 분해

`scripts/audit_same_config_readout_bias.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-same-config-real-readout-bias-1/result.json`: 합성 학습 설정을 복원한 실제 저속 재생의 저장 상태 462개를 고정 motor readout으로 독립 재계산해 intent 최대 오차 5.96e-8을 확인했다. 두 근육 좌우 차이에서 각각 고정 중립을 뺀 값은 449결정에서 모두 음수, 13결정에서 적어도 하나 양수, 둘 다 양수인 결정은 0개였다. 현재 양의 softmax 혼합과 gain 아래 첫 449개 고정 상태는 혼합 비율만 바꾸어 양의 조향으로 전환할 수 없다. 각 채널의 평균 가중 preactivation 기여는 -0.02441, -0.00524다.

이는 저장된 단일 저속 기록의 고정 상태와 현재 공학적 readout에 대한 결론이다. 회로 상태나 중립을 바꾼 모든 방법, 실제 생리 기능 또는 폐루프 운전을 판정하지 않는다. 정지 영상의 중립 제거가 코너 선행 정보를 지운 기존 실패도 고려해야 한다. fitting·차량 제어는 없었다.


## 2026-11-01 새 합성 도로의 yaw 어댑터 사전 고정 평가

평가 전 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unseen-interval-evaluation-plan/plan.json`에 seed 881399, 곡률 .022, 굽이 시작 11m, 고정 후보 SHA `e1bcbce7965fe5203c22081a096aef74acf719749dc243a063455bb2a3861b1a` 및 결과 지표를 기록했다. 새 장면은 같은 합성 생성기로 만들어 FlyVis 네 시야를 각 조건별로 독립 재생했다. `scripts/evaluate_unseen_interval_synthetic.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unseen-interval-evaluation-1/result.json`.

두 굽이에서 yaw ±.12와 yaw 0의 마지막 10결정 평균 조향 차이 MSE는 어댑터 0에서 1.66780e-6, 학습 어댑터에서 1.68690e-6으로 약 1.15% 증가했다. +굽이 minus -굽이 차이는 세 yaw에서 기준 [0.00584623, 0.00297603, 0.00364877], 후보 [0.00584623, 0.00297644, 0.00364825]였다. 양쪽 모두 9조건 전체 음수였다. 이 제한된 새 합성 조건에서 학습 목표 개선이 재현되지 않고 양방향 절대 조향도 없으므로 후보 미채택이다.

새 질감과 곡률이지만 생성기·물리 가정이 같아 독립 실제 주행 일반화가 아니다. 회전 억제를 실제 최적 조향으로 가정하지 않는다. 추가 학습·실제 데이터 fitting·차량 제어는 없었다.


## 2026-11-01 합성 yaw 시간 불일치 발견과 정정

`train_interval_synthetic_stability.py`와 최초 `evaluate_unseen_interval_synthetic.py`는 합성 영상이 첫 0.4초 정지하는 동안에도 회전 조건의 가상 yaw를 비영으로 공급했다. 따라서 해당 후보는 영상·몸 yaw가 정지 기간에 불일치한 데이터로 학습됐고, 최초 평가도 이 결함을 공유한다. 후보 미채택 결론을 유지하되 정합된 센서에서 학습한 결과로 해석하지 않는다.

정지 구간 yaw를 0으로 바꾼 사후 평가 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unseen-interval-corrected-timing-1/result.json`에서 기준 MSE 1.667799e-6, 후보 1.686870e-6, 9조건 모두 음수였다. 최초 평가를 본 뒤 수정했으므로 이 재평가를 사전 고정 맹검 결과라 부르지 않는다. 기존 자료·결과를 보존했다.

`synthetic_yaw_timing.synthetic_interval_yaw`는 정지 구간, 회전 구간과 경계가 겹치는 시간의 평균을 계산하고 잘못된 시간을 거부한다. `tests/test_synthetic_yaw_timing.py` 5개 통과. 정정된 평가 스크립트는 이후 이 함수를 사용한다. 이는 합성 자극 시간 계약일 뿐 실제 차량 센서나 노출 시간의 검증이 아니다. 추가 학습·차량 제어는 없었다.


## 2026-11-01 현재 몸 yaw 누락 정정 대조

앞선 `train_interval_synthetic_stability.py`와 평가 스크립트는 yaw 구간을 별도 입력하면서 기존 몸 관측의 yaw index 4를 항상 0으로 유지했다. 이는 감각 경로를 분리한 인공 조건일 수는 있으나 실제처럼 시각·몸 회전이 정합된 학습이나 평가가 아니다. 앞서 정지 0.4초 구간만 정정한 사후 평가에서도 이 제한은 남았다.

`scripts/evaluate_unseen_interval_body_yaw.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unseen-interval-body-yaw-1/result.json`: 기존 합성 장면에서 가상 현재·과거 몸 수신 시각에 따라 yaw를 0 또는 ±.12rad/s로 주고 구간 yaw도 정지·회전 경계에 맞췄다. 기준의 yaw 조건 간 조향 차이 MSE는 0.00014303, 기존 후보는 0.00014436, 각 9조건 모두 음수였다. 이전 현재 몸 yaw 0 대조의 약 1.67e-6과 비교하면 회로가 현재 몸 yaw에 훨씬 민감한 해당 조건의 반사실이다. 실제로는 몸 회전에 따라 조향을 바꾸어야 할 수 있어 차이가 크다는 것만으로 성능 실패를 결론내리지 않는다. 가상 센서 시간과 회전값, 같은 합성 생성기 및 사후 평가라는 한계가 있다. 기존 후보 학습·출력은 유지하며 신규 학습·실제 제어는 하지 않았다.

시간 평균 함수로 작성한 수정 평가 재실행 `20261101-unseen-interval-corrected-timing-2`는 최초 수정 결과와 배열 SHA가 동일하다.


## 2026-11-01 합성 코너의 시각·몸 회전 교차 대조

`scripts/audit_synthetic_road_visual_body_factorial.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-road-visual-body-factorial-1/result.json`: 이미 본 881399 합성 장면의 굽이 3개·시각 yaw 3개마다 가상 현재/과거 몸 yaw를 0, 시각 회전과 같은 부호, 반대 부호로 넣어 동일 회로를 독립 초기화하고 41결정씩 재생했다. 구간 yaw 어댑터는 사용하지 않았다. 몸 0·같은 부호 기준은 직전 각각의 0 어댑터 재생과 허용 오차 내 일치했다.

마지막 10결정의 시각 +yaw minus -yaw 차이는 세 굽이에서 몸 0일 때 [-0.000165, -0.000569, +0.000103], 같은 몸 yaw면 [-0.023713, -0.024110, -0.023451], 반대 몸 yaw면 [+0.023383, +0.022971, +0.023657]이었다. 같은 몸 조건의 +굽이 minus -굽이 차이는 yaw별 대략 0.00297–0.00585로 유지됐고, 27조건의 평균 조향은 모두 음수였다. 이 회로·범위에서 몸 yaw 조작의 영향이 시각 yaw만 바꾼 효과보다 훨씬 큰 반사실이나, 이를 몸 반응의 부적절성이나 시각 기능 실패로 단정하지 않는다. 합성·사후 선택된 같은 장면, 비현실적인 반대 감각 조합, 물리 시각·생리·폐루프 미검증이 한계다. fitting과 차량 제어는 없었다.


## 2026-11-01 합성 몸 감각의 수신 시각별 속도 정정

이전 `evaluate_unseen_interval_body_yaw.py`는 yaw는 각 수신 시각에 맞췄지만 속도·바퀴 회전은 현재 시각 값을 과거 관측에도 복사했다. 정지→출발 경계에서 과거 값이 미래 속도를 가질 수 있어 합성 관측 계약을 위반한다. `synthetic_yaw_timing.synthetic_body_at`은 현재와 과거의 별도 수신 시각을 기준으로 yaw·속도·네 바퀴를 만든다. 일정 속도와 속도/반경 바퀴 회전은 합성 가정이며 타이어 slip이나 차량 동역학을 나타내지 않는다. 시간 경계·순서 및 회전 adapter/회로 관련 검사 21개 통과.

`scripts/evaluate_unseen_interval_coherent_body.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-unseen-interval-coherent-body-1/result.json`: 사후 수정 재생의 9조건 전체 41결정 결과 배열 SHA는 이전 몸 yaw 정정 재생과 동일했고 9개 조향 평균은 음수다. 이미 본 합성 장면과 앞서 불일치한 입력으로 학습된 후보의 진단이며 새로운 성능 검증이 아니다. 새 fitting·차량 제어 없음.


## 2026-11-01 합성 폐루프를 위한 평면 차량 기반

기존 `SyntheticYawRoadVisual`은 yaw와 전방 위치가 변해도 세계 좌표 횡위치를 항상 0으로 두므로 회전이 다음 위치에 미치는 효과를 모델링하지 못한다. 기존 소스와 원본 결과는 보존하고, 별도 `synthetic_planar_road_visual.py`에 세계 좌표 횡이동을 추가했다. 횡이동 0이면 기존 렌더러와 네 이미지가 정확히 같고 횡이동은 장면을 변화시킨다.

`synthetic_planar_vehicle.py`는 SI 단위, 좌회전 양의 전륜각, 일정 가속·조향, 0속도 이하 정지, slip 없는 bicycle 가설로 위치·방향·속도를 적분한다. world pose는 합성 renderer/교사·평가에만 쓰고 actor 입력이 아니다. 선회 대칭과 제동 정지 거리 등 `test_synthetic_planar_vehicle.py` 및 시간 계약을 포함한 9개 검사 통과. 이는 실측 차·타이어·하중·차체 동역학·Assetto Corsa 트랙이 아니며 현재 비행 회로·행동 경계와 연결된 폐루프가 아니다. 학습·실제 차량 제어 0; 다음은 이 가상 몸의 시각·감각과 단일 회로 명령을 연결하여 상태·부호·시간·예외를 검증하는 것이다.


## 2026-11-01 합성 평면 차량·네 시야·단일 비행 회로 폐루프 첫 실행

`scripts/run_synthetic_planar_flight_loop.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-planar-flight-loop-1/result.json`: 합성 평면 도로의 전후좌우 영상→고정 pretrained FlyVis의 5ms 유지 입력 상태→기존 단일 비행 graph→두 intent→공학 codec→가상 전륜각·가속도→다음 위치/영상·몸 yaw/속도의 순서로 41결정(40ms 간격)을 연결했다. actor에는 영상과 가용 yaw·속도·가상 바퀴만 주고 위치·도로 굽이 정답은 넣지 않았다. 자세·하중·타이어 등 비모델 감각은 결측 처리했다. 양의 좌회전 intent는 합성 전륜각으로 최대 배율 .35rad, signed drive는 가속 3/감속 6m/s² 가설로 변환했다. 이는 실제 차량 보정이 아니다.

초기 3m/s에서 회로 출력 조향 각도는 -0.02399~-0.01077rad, 가속도는 -3.482~-2.104m/s²였고 1.64초 뒤 x=1.872m·y=-0.0109m·속도0으로 정지했다. 코너 시작 12m에 미도달하여 코너 추종을 검증하지 못했다. 같은 초기 상태에서 가상 조작을 0으로 고정한 대조는 x=4.92m·속도3m/s다. 두 조건의 첫 망막 해시는 동일하고 마지막 해시는 다르다. `trajectory-audit.json`은 저장 상태 연속성 및 같은 합성 적분기 재계산 오차0을 확인한 것이며 FlyVis/비행 회로를 독립 재실행한 증거는 아니다.

이 실행은 위치/영상/몸의 되먹임 연결과 실패 사례만 보여 준다. 단순 bicycle은 하중·타이어 slip·기어·셋업·실제 트랙과 생리/물리 동기·실시간을 모델링하지 않는다. 설정한 초기 속도와 단순 actuator 가설 때문에 실제 시뮬레이터 성능이나 완주를 주장할 수 없다. 기존 가중치 고정·optimizer0·실제 차량 제어0. 다음은 전진/제동 중립 편향과 양방향 조향을 실제 센서 계약과 분리해 학습하고 합성 폐루프 및 AC에서 검증한다.


## 2026-11-01 합성 폐루프 속도 피드백 학습

`scripts/train_synthetic_speed_feedback_closed_loop.py`, 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-speed-feedback-plan/plan.json`, 결과 `.../20261101-synthetic-speed-feedback-closed-loop-1/result.json`: 합성 폐루프의 원본 41결정 scene·몸 감각을 사용하고 단일 graph 및 나머지 가중치를 고정해 추가 몸 feedback projection만 full-history gradient로 Adam 4회 갱신했다. 교사는 합성 관측 속도의 3m/s 오차에서 만든 제한된 signed drive이며 목표값은 학습 표적에만 사용했다. 감각 시간·episode state를 보존했으며 실제 데이터와 지도는 쓰지 않았다.

같은 초기 조건 재실행에서 학습 전·후 마지막 속도는 모두 0, 이동거리 1.872→1.927m였다. 매 업데이트 전 손실은 0.52279, 0.51828, 0.51393, 0.50985로 감소했으나 사후 마지막 손실은 별도 측정하지 않았다. 학습 후 전륜각 범위는 -0.02399~+0.00097rad로 변해 조향 간섭도 있다. 원래 첫 실행과 학습 전 재생은 저장 전체 trajectory 최대차 9.32e-8 (`comparison-audit.json`; 독립 모델 재실행 아님). 가중치 후보 SHA `15cdbd26c1aae863e03d176f1bab7614d9bf290e0bcd39dcd1a5913c419960de`.

둘 다 12m 코너에 도달하지 못했으며 주행 문제를 해결하지 못했다. 속도 목표와 actuator/타이어는 합성 공학 가설, 검증 이력은 학습 이력과 같은 초기 조건이다. 새 실제 주행·물리 타이밍·비행 생리·변속/셋업·유효랩 근거가 아니며 actor 미배포·실제 차량 제어0이다. 다음은 조향 간섭을 포함한 가감속 원인과 분리된 적격 평가를 검증한다.


## 2026-11-01 합성 전진 readout 기준 학습과 새 속도 대조

`scripts/train_synthetic_power_reference_closed_loop.py`, 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-power-reference-plan/plan.json`, 결과 `.../20261101-synthetic-power-reference-closed-loop-1/result.json`: 이전 합성 자가 생성 41결정과 3m/s 가설 교사 오차를 사용해 graph·몸 및 시각 어댑터와 조향 변환을 고정하고 공학적 power_reference scalar만 Adam 20회 갱신했다. 매 업데이트 전 손실은 0.52279→0.13706이었다. 같은 초기3m/s·같은 도로 41결정에서 기준 차량은 1.872m 후 정지, 후보는 4.715m·2.944m/s였으며 12m 코너에는 둘 다 도달하지 못했다. 같은 고정 회로 상태의 조향 계산식은 전진 기준과 독립이지만, 폐루프에서는 주행 상태가 달라지므로 조향 보존을 확정할 수 없다.

학습하지 않은 초기 속도 0과 6m/s의 같은 도로 사후 대조 (`scripts/evaluate_synthetic_power_reference_transfer.py`, `.../20261101-synthetic-power-reference-speed0-1/result.json`, `.../20261101-synthetic-power-reference-speed6-1/result.json`)에서 후보는 각각 1.64초 뒤 0.142m/s와 5.933m/s였다. 고정 기준 이동은 이 조건에서 양방향 속도 목표 회복을 보여주지 못했다. 두 대조는 3m/s 교사 목표의 속도 유지 능력 진단이지 독립 도로·제동 능력 인증이 아니다.

후보 SHA `add93c34891874c6e79b4f5408366c2a89e62331f9d6683180c93b28a674cb09`; 합성 actuator·타이어 가정, 미검증 물리 시각·flight physiology·조향/가감속 폐루프와 실제 주행을 고려해 미채택·배포0이다. 실제 데이터 학습과 차량 제어도 없었다.


## 2026-11-01 합성 다중 초기 속도 몸 피드백 학습

`scripts/train_synthetic_multispeed_feedback.py`, 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-multispeed-plan/plan.json`, 결과 `.../20261101-synthetic-multispeed-feedback-1/result.json`: 기존 graph와 시각 경계를 고정하고 합성 자가 생성 0/3/6m/s 이력 각각 41결정으로 몸 feedback projection과 공학 power_reference만 Adam 15회 갱신했다. 목표는 속도 3m/s의 공학 가설, 손실은 signed-drive 오차와 기존 조향 출력 보존항이다. 세 이력은 같은 도로이며 매 episode state는 유지하고 원본 checkpoint를 보존했다. 학습 전 각 업데이트 시 손실은 0.44649→0.20205였고 마지막 업데이트 후 손실은 별도 측정하지 않았다.

같은 합성 도로 폐루프 41결정에서 후보는 초기0→최종0, 3→2.347, 6→5.349m/s였고, 학습에 쓰지 않은 중간 초기속도 1.5→0.846, 4.5→3.848m/s였다. 이는 완전 정지 출발과 저속 회복에 실패하고 일정한 감속에 가까운 결과다. 중간 속도는 별도 초기값이지만 도로·모델·생성기가 동일하므로 독립 일반화 평가는 아니다. 조향 보존항은 합성 fit 상태에만 적용되며 폐루프 조향 보존 보장은 없다. 후보 SHA `8b3425b78f3620f92cb27fe76bcf6db9667db473074cf3739cf64968d4d509a8`, 미채택·actor 미자격이다. 실제 데이터 fitting·실제 차량 제어0, 생리·실제 동기·주행·변속/셋업과 유효랩은 미검증이다.


## 2026-11-01 합성 다중 속도 후보 추가 학습

`scripts/continue_synthetic_multispeed_feedback.py`, 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-multispeed-continue-plan/plan.json`, 결과 `.../20261101-synthetic-multispeed-continuation-1/result.json`: 직전 15회 후보와 합성 fit 이력 0/3/6m/s를 해시 확인하고 Adam 상태를 새로 시작해 몸 feedback·공학 power reference만 30회 추가 갱신했다. 생물학적 연결·다른 경계는 고정했고 원본 후보와 이력을 보존했다. 업데이트 전 손실은 첫 0.18761, 마지막 0.11905였으나 중간 변동이 컸고 마지막 업데이트 후 손실은 별도 측정하지 않았다.

새 질감 881400·같은 합성 도로에서 41결정 후 초기속도 0/1.5/3/4.5/6은 각각 1.210/2.697/4.197/5.697/7.197m/s가 됐다. 저속 출발은 생겼지만 목표 3m/s보다 높은 속도에서도 가속해 목표 복귀에 실패했다. 모든 조건의 조향도 음수였고 가장 빠른 조건은 10.67m만 진행해 12m 코너에 미도달했다. 1.5/4.5는 fit에 없는 초기속도지만 장면 생성기·기하·모델은 같아 실제 주행 일반화가 아니다. 후보 SHA `a8111900ce79be27e4d9fbcc219ecb9399642f8dcebee5eb07ed63250dad3ca5`; 미채택·배포0이다. 실제 데이터 학습·차량 제어0, 물리 시간·비행 생리·제동/변속·실제 완주 미검증.


## 2026-11-01 합성 폐루프 명시 목표속도 대조

기존 `UnifiedFlightSensorimotor.sensory_decision`은 `_neutral_sensory_command`로 원하는 속도를 현재 몸 속도와 같게 설정한다. 이 경로는 의도적으로 speed error 입력을 중립화한 실험 설정이지, 시각 기반 목표속도 결정이 구현됐다는 뜻이 아니다. `scripts/evaluate_synthetic_explicit_speed_command.py`는 같은 고정 graph와 합성 차량에 목표 yaw 0·속도 3m/s 외부 명령을 주는 `decision`과 원래 sensory 중립을 별도 episode로 비교했다. 외부 목표는 진단 교사이며 배우가 예측한 값이 아니다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-explicit-speed-command-speed0-1/result.json`: 초기0m/s에서 중립과 명시 목표 모두 41결정 후 정지, 명시 목표는 가상 가속도 범위 -3.482~-0.749m/s²였다. `.../20261101-explicit-speed-command-speed6-1/result.json`: 초기6m/s에서 중립 2.316m/s, 명시 목표 1.935m/s였다. 목표가 전진 경로에 영향을 주지만 기존 readout/codec과 결합한 합성 차량의 정지 출발 및 목표속도 유지가 해결됐다는 증거는 아니다. 한 합성 도로·짧은 이력·공학 actuator/차량 가설로 물리/생리·실제 주행을 평가하지 않는다. optimizer·실제 차량 제어0.


## 2026-11-01 합성 전진 기준과 명시 목표속도 결합

`scripts/evaluate_synthetic_power_reference_explicit_goal.py`: 앞서 합성 fit된 공학 power_reference 후보 SHA `add93c34891874c6e79b4f5408366c2a89e62331f9d6683180c93b28a674cb09`를 고정하고 별도의 외부 목표 yaw0/속도3m/s를 같은 비행 graph에 주었다. 원래 power 기준과의 사후 대조다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-power-ref-explicit-goal-speed0-1/result.json`에서 원래 기준은 정지, 후보는 41결정 뒤 1.000m/s·0.765m 이동했다. `.../20261101-power-ref-explicit-goal-speed6-1/result.json`에서 원래 기준은 1.935m/s, 후보는 4.958m/s였다. 후보는 해당 짧은 시험에서 목표 아래 가속·위 감속 방향을 보였다.

목표는 외부 진단 명령으로, 회로가 시각이나 몸 감각으로 적정 속도를 결정한 것이 아니다. 후보가 합성 속도와 같은 도로에 맞춰졌고 대조는 사후이며 목표에 장기 수렴하는지, 조향·코너·제동·센서 실패에 적절히 반응하는지 확인하지 않았다. 단순 차량·actuator 가설의 결과로, flight physiology·물리 시간·실제 주행이나 유효랩을 증명하지 않고 후보는 actor 미자격·미배포다. optimizer와 실제 차량 제어0.


## 2026-11-01 단일 비행 상태 기반 속도 명령 경계

`flight_state_speed_command.FlightStateSpeedCommand`는 새로운 재귀 상태 없이 기존 graph의 직전 근육 출력과 현재 유효 몸 속도에서 속도 보정량을 계산해 기존 maneuver 입력에 전달하는 0 초기화 공학 adapter다. 허용된 외부 속도 ceiling을 넘지 않으며 직전 graph state가 없으면 근육 입력은 0이다. 현재 영상은 먼저 같은 graph에 반영된 다음 결정의 state에만 영향을 주므로 미래 정보는 사용하지 않는다. 새 명령은 선택형 `experimental_recurrent_speed_decision`/`experimental_recurrent_speed_training_decision`에서만 사용하며 기존 sensory, interval, guard 호출을 바꾸지 않는다. 이 projection은 실측 파리 synapse나 생리 모델이 아니고 학습 자료 자격은 호출자가 별도 확인해야 한다.

합성 graph의 중립 동일성·상태 유지/gradient·episode reset·ceiling·autograd 거부와 기존 guard/interval 회귀 포함 24개 검사 통과. `scripts/audit_recurrent_speed_real_graph.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-recurrent-speed-real-graph-1/result.json`: 합성 장면41결정과 기존 실측 연결 기반 graph에서 0 경계의 intent/전체 상태 최대 차 0; 최종 전진 출력에 대한 투영 gradient L2 0.05579, 7원소 비영, bias gradient0.55745다. 이 검사는 gradient 도달만 보여 주며 속도 정책 학습, 시각의 실제 인과 기여, 물리 시간·flight physiology·실시간 및 폐루프 성능을 입증하지 않는다. optimizer·실제 차량 제어0, actor 미자격.


## 2026-11-01 같은 비행 상태 기반 합성 속도 명령 학습

`scripts/train_synthetic_state_speed.py`, 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-state-speed-training-plan/plan.json`, 결과 `.../20261101-synthetic-state-speed-1/result.json`: 기존 synthetic 0/3/6m/s 자가 이력과 직전 실험의 합성 학습 power 기준을 고정하고, 이전 비행 근육 상태와 현재 몸 속도에서 속도 보정량을 내는 stateless projection만 40회 학습했다. 합성 3m/s 목표는 손실 표적에만 있고 평가 시 외부 명령으로 넘기지 않는다. 실제 그래프·시각·나머지 가중치 고정, full-history gradient, 0 초기화다. 업데이트 전 손실 0.57447→0.07157이며 마지막 업데이트 후 손실은 별도 측정하지 않았다.

새 질감 881401·같은 합성 도로 41결정 후 초기 0/1.5/3/4.5/6m/s는 0.840/1.928/3.140/4.375/5.531m/s였다. 1.5/4.5는 fit 이력에 없지만 동일 생성기·기하이므로 독립 실제 주행이 아니다. 마지막 10결정의 예측 목표속도 평균은 각각 2.674/3.138/3.642/4.149/4.630m/s로 교사 3에 정확히 수렴하지 않고 현재 속도에 따라 변했다. 3m/s 조건은 약간 가속했고 5조건×41결정 조향은 모두 음수다. 고정 합성 목표이므로 코너의 속도 판단이나 시각의 기여는 학습/검증되지 않았다. 이전 속도 대조는 배경이 달라 수치 차이를 인과 효과로 단정하지 않는다.

배열 SHA `ca1e12fe613cd80f7a99af2871a2fc46b00b14779099b9012f70ea93f1f0cba6`, 속도 경계 checkpoint SHA `007046c3a350955ef50b53cdcb9e70a0381c253d345cf8029a28ca05d72cb814`; `trajectory-audit.json`은 연속성과 저장 수치만 확인하고 모델을 독립 재실행하지 않았다. 단순 bicycle·공학 codec, 미검증 센서 동기/flight physiology, 실제 차량·변속·셋업·랩타임 한계를 유지한다. 실제 데이터 학습/차량 제어0, actor 미자격·미배포다.


## 2026-11-01 합성 속도 명령의 시각 경로 대조

`scripts/audit_synthetic_state_speed_visual.py`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-state-speed-visual-ablation-1/result.json`: 이미 본 881399 합성 도로의 직선·좌·우 굽이와 yaw0에서 몸 speed3m/s·yaw0을 같게 유지하고 학습한 speed 경계·합성 power 기준을 고정한 비행 graph를 각각 재생했다. 마지막 10결정 평균의 +굽이 minus -굽이는 목표속도 -1.287e-5m/s, 조향 intent +0.005856, 전진 intent -3.175e-5다. 공간 대비 projection을 0으로 하면 조향 굽이 차이는 -1.828e-5, 목표속도 차이는 2.384e-8m/s였다. 운동 채널을 인위적으로 0으로 하면 정상 대비 전 이력 최대 변화는 목표속도 2.36e-5m/s·조향 0.00672·전진 7.30e-5였다.

이 조건에서 속도 명령의 코너 의존성은 매우 작고, 일정 3m/s 합성 교사는 코너에 맞춘 속도를 가르치지 않았다. 공간 경로 개입은 조향의 합성 굽이 반응에 영향을 주지만 생리나 실제 코너 주행 기여를 증명하지 않는다. 0 대체는 분포 밖일 수 있고, 동일 과거 합성 장면의 사후 분석·고정 몸 이력이다. optimizer와 실제 차량 제어0, 물리 시간·실제 주행 미검증이다.


## 2026-11-01 합성 직선·코너 속도 readout 학습

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-visual-corner-speed-plan/plan.json`, 코드 `scripts/fit_synthetic_corner_speed_readout.py`: 사전 선언한 직선 3.5·좌우 코너 2.5m/s는 검증된 레이싱 표적이 아닌 합성 공학 가설이다. 기존 graph·시각 경계·몸 속도3을 고정하고 기존 속도 후보로 재생한 fit 질감 881301/881302의 직전 근육 상태 6개에서 마지막10프레임만 추출해 ridge로 speed projection을 한 번 계산했다. 굽이 표적은 학습 손실에만 사용했고 actor 입력은 영상과 몸이다. 고정 상태에서의 fit 예측 평균은 직선 2.83357, 코너 2.83303m/s로 차이 0.000535였다.

그 뒤에만 연 사전 지정 새 질감 881402의 전체 재귀 graph 재생 `.../20261101-visual-corner-speed-fit-1/result.json`에서 후보 목표 평균은 직선 2.995106, 좌 2.991934, 우 2.997325m/s다. 직선−코너 평균 0.000477m/s는 가설 표적 차이 1m/s에 한참 못 미친다. 목표 MSE는 기존 0.81735→0.24808로 감소했지만 공통 평균 이동이 주된 변화다. 공간 시각 경계를 제거하면 차이는 0.0000043m/s였으나 미세한 차이의 생리/주행 의미는 미검증이다. fitted 상태는 baseline state를 이용해 fit한 뒤 재귀 graph를 다시 실행했으며, candidate SHA `b559b96d822b1470cdbff8b530f2fdcfb481bb73c7bc8ce8c497dfe64e2e1ae6`; 미채택·actor 미자격이다.

새 질감도 같은 합성 생성기·비물리적 고정 몸 속도이며 학습된 실제 코너 속도나 폐루프 안전을 뜻하지 않는다. 실제 데이터 fitting·차량 제어0, 물리 시간·생리·실제 완주/변속·셋업 미검증.


## 2026-11-01 합성 코너 속도 고정 상태의 정규화·가중치 한계

`scripts/audit_corner_speed_fixed_state_bounds.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-corner-speed-fixed-state-bounds/result.json`: 기존 합성 fit의 두 질감·프레임별 직선과 좌/우 코너 총 40쌍에서 동일 몸 속도와 고정 여섯 근육 상태의 차이를 확인했다. speed adapter는 최대 보정 10m/s의 tanh이며 tanh와 출력 clamp의 Lipschitz 한계를 적용하면, 근육 가중치 각각의 절댓값이 5000 이하일 때 평균 쌍별 명령 차이의 상한은 0.4243m/s다. 가설상의 1m/s 차이를 위해서는 이 고정 상태에서 최대 가중치가 적어도 약 11784여야 한다는 필요조건이다. 이는 최적 해나 충분조건이 아니고, 재귀적으로 바뀐 상태·다른 경계의 전체 한계도 아니다.

저장 fit 특징에만 사후 ridge 강도 0.001/1e-5/1e-7/1e-9를 계산하면 fit 직선−코너 차이 0.000535/0.03798/0.22968/0.49449m/s, 최대 raw 가중치 164/13008/151610/1697725였다. 새 후보를 저장하거나 graph를 재학습한 실험이 아니며, 약한 정규화의 큰 gain은 실제 감각 잡음·분포 차이에 취약할 수 있어 성능·생리 타당성을 뜻하지 않는다. 합성 표본은 연속 시간이고 두 질감뿐이다. 실제 데이터 학습·차량 제어 0.


## 2026-11-01 합성 코너 정보의 비행 graph 상태별 진단

사전 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-state-corner-probe-plan/plan.json`; `scripts/probe_synthetic_flight_state_corner.py`; 결과 `.../20261101-flight-state-corner-probe-1/result.json`. 동일 고정 graph와 속도 경계에서 합성 fit 질감 881301/881302의 직선/좌/우 3장면×마지막10프레임, 몸 속도3·yaw0을 사용했다. 각 기존 graph의 HS8·VS34·DNg02 29·DNa04/05 각2·날개 운동 후보67 상태 및 기존 근육 집계6을 따로 표준화·ridge 진단했다. 분류 표적은 직선 +1/양 코너 -1이며 actor 입력이 아니다. 새 사전 지정 질감 881403에서 전체 graph를 재실행해 판단했다.

직선/코너 균형 정확도는 DNg02와 날개 후보 각각 1.0, VS 0.8, HS·DNa04·DNa05·근육 집계는 각각 0.5였다. 공간 시각 projection을 인위적으로 끄면 DNg02와 날개 후보 점수는 각각 0.5로 바뀌었다. DNg02 29개 중 22, 날개 후보 67개 중 49는 fit 표준편차가 1e-6 미만이라 분모 하한을 사용했다. 결과는 서로 상관된 30프레임·단 하나의 새 질감, 동일 합성 생성기·고정 속도·인위적인 0 개입에 한정된다. 높은 점수는 학습된 진단기에서 정보가 읽힌다는 관찰이지 실제 생리 선택성, 일반화된 주행 성능 또는 안전 속도를 뜻하지 않는다. 인공 진단기는 actor에 연결하지 않았고 비행 graph 가중치·실제 기록을 fit하지 않았으며 차량 제어0이다. 다음은 독립된 다양한 기하와 실제 시간 정합을 갖춘 이력에서 재현하고 기존 회로 내부의 인과 전달을 확인한다.


## 2026-11-01 — 고정 비행 상태 진단기의 새 합성 조건

이전 두 질감의 ridge 판독기와 표준화 통계·회로 가중치는 고정했다. 초기 결과를 본 뒤 별도로 정한 세 조건의 새 합성 영상에서 각각 직선·좌우 코너 3장면의 마지막 10프레임을 비교했다. 각각의 균형 정확도는 다음과 같다.

| 시드 | 곡률 | 코너 시작 거리(m) | DNg02 | 날개 출력 후보 | 여섯 근육 출력 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 881411 | 0.012 | 6 | 0.925 | 1.0 | 0.5 |
| 881412 | 0.032 | 15 | 0.95 | 1.0 | 0.5 |
| 881413 | 0.040 | 20 | 0.675 | 0.7 | 0.5 |

공간 대비 투영을 인위적으로 0으로 만든 대조에서 DNg02와 날개 출력 후보의 값은 모든 조건에서 0.5였다. 20m 조건에서 약화됐으나 거리·질감·곡률을 동시에 바꿔 원인을 특정할 수 없다. 샘플 프레임은 서로 상관돼 있고 생성기도 동일하다. 실측 연결 기반 그래프의 근사 상태를 공학적으로 판독한 결과이지 초파리 생리 기능, 배우의 판단, 실제 차량 주행 증거가 아니다. 기존 actor와 비행 상태는 하나이며 이 판독기는 학습·배치되지 않았다. 다음에는 하나씩 변수를 고정한 대조와 근육 출력까지의 인과 전달을 검증한다.

계획: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-state-probe-replication-plan/plan.json`; 원본 및 해시: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-state-frozen-replication-1/result.json`; 상태 배열: 같은 폴더의 `replication.npz`. 업데이트와 재학습은 모두 0회다.


## 2026-11-01 — 근육 동력 좌우 평균의 정보 상쇄 진단

기존 고정 판독기 재검증의 세 조건에 저장된 뉴런 값과 여섯 근육 값의 일치, 그래프 해시 및 각 출력군의 소속을 확인했다. 각 조건의 같은 시점에서 좌우 코너와 직선의 차이를 냈다. 동력 L·R의 절대 차이 평균에 대한 공통 평균의 절대 차이 비율은 각각 27.0%, 20.9%, 22.5%였다. 12개씩인 한쪽 내부 평균의 같은 비율은 약 80~88%로, 이 측정에서는 측면 내부보다 L/R을 합치는 단계에서 더 크게 상쇄됐다. b1/b2의 각 쪽은 단일 출력 뉴런이므로 그룹 내부 평균 상쇄는 적용되지 않는다.

이 수치는 절대 변화 크기의 비율이며 분류 정보량이나 실제 가감속 성능을 직접 측정하지 않는다. 코너에 따른 좌우 반대 부호 신호를 합치는 구조의 관찰이지, 생리학적 증거나 개입을 통한 단독 원인 검증이 아니다. 배우 수정이나 학습은 하지 않았다. 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-output-averaging-audit-2/result.json`.


## 2026-11-01 — 합성 기하·질감 단일 변수 대조

기존 새 조건 결과를 본 뒤 아래 조건을 정하고 새 영상 생성 전 계획을 고정했다. 기존 진단 판독기와 신경 회로는 고정했고 각 조건에서 직선·좌우 코너의 마지막 10개 연속 프레임을 평가했다.

| 기준 대비 변경 | 시드 | 곡률 | 시작 거리(m) | DNg02 | 날개 출력 후보 | 근육 6개 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 기준 | 881413 | 0.040 | 20 | 0.675 | 0.700 | 0.500 |
| 거리만 | 881413 | 0.040 | 6 | 0.750 | 1.000 | 0.500 |
| 곡률만 | 881413 | 0.012 | 20 | 0.525 | 0.500 | 0.425 |
| 질감만 | 881412 | 0.040 | 20 | 0.500 | 0.650 | 0.500 |

거리를 줄인 조건의 DNg02 평균 점수는 직선/좌/우 `0.484/-19.945/2.238`로 방향 간 일반화 차이가 크다. 같은 생성기의 소수 장면·연속 상관 프레임이며 이 조합에서만 관찰된 민감도다. 단일 변수 조건들로도 실제 도로, 생리 기능, 차량 폐루프나 배우의 사전 코너 판단은 확인되지 않는다. 특정 요인만이 실패 원인이라고 결론내릴 수 없다.

계획: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-factorial-flight-probe-plan/plan.json`; 결과와 해시: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-controlled-flight-probe-1/result.json`; 상태 배열은 같은 폴더 `replication.npz`. 학습·판독기 재적합·차량 제어는 모두 0회. 다음에는 새 조건 반복과 독립된 몸 감각 및 경계 개입 검증이 필요하다.


## 2026-11-01 — 좌우 근육 차이 크기를 이용한 사후 진단

기존 동결 합성 상태 60개(두 질감, 고정 속도)의 가설 교사 속도 직선 3.5/좌우 코너 2.5m/s에 두 ridge 진단기를 맞췄다. 하나는 근육 6개를 선형으로 사용하고 다른 하나는 학습 직선 평균에서 중심을 맞춘 세 좌우 차이의 절댓값을 추가했다. 1e-6 고정 특징 스케일과 ridge 1을 사용했고 결과 관찰 후 마련한 진단으로 두 모델 모두 배우가 아니다. 평가 상태는 기존 별도 합성 시드 881403/881411/881412/881413 및 공간 경계 0 대조다. 기존 학습 feature는 직전 상태, 비교 probe는 현재 상태여서 한 시점 차이가 있다.

절댓값 진단의 직선 예측에서 양 코너 평균을 뺀 값은 시드 순서대로 0.977/1.078/0.510/0.236m/s였고 선형 진단은 0.190/0.327/0.071/0.019m/s였다. 공간 경계를 끄면 절댓값 모델 차이는 약 -0.007/0.016/-0.0004/-0.0009m/s였다. 그러나 추가 기하 대조 중 6m 코너 한 방향의 평균 예측은 -0.315m/s였다. 근육 한 좌표에 ±1e-6을 일괄 더하는 결정론적 교란에 예측 하나가 최대 0.774m/s 바뀌었다. 이것은 실측 잡음 분포가 아니라 민감도 실험이다.

따라서 부호가 다른 양쪽 코너 신호를 같은 방향의 속도 지표로 읽을 가능성은 보였으나 정규화와 큰 gain, 포화·범위·몸 속도 변화, 시간 연속성과 폐루프 효과, 다른 질감/실제 영상에 대한 안정성은 미검증이다. 별도 운동 회로를 만든 것이 아니며 생리 기능이나 운전 성과로 해석하지 않는다. 기존 배우·회로는 변경하지 않았다. 계획: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-muscle-symmetry-diagnostic-plan/plan.json`; 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-muscle-symmetry-diagnostic-1/result.json`; 고정 기하와 교란 검사: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-muscle-symmetry-controlled-audit-1/result.json`.


## 2026-11-01 — 고정 영상 대비 몸 속도 단독 변화

기존 동결 비행 회로와 사후 비선형 진단기를 그대로 유지했다. 시드 881403/881413의 3m/s 합성 영상은 동일하게 두고 몸 속도와 바퀴 속도 감각을 0/1.5/3/4.5/6m/s로 변경했으며 각 장면마다 회로 상태를 초기화해 다시 전개했다. 각 시드 직선의 10프레임 평균 무제한 진단 예측은 각각 약 389.5/187.6/3.31/-162.4/-302.5 및 389.4/187.5/3.25/-162.5/-302.5m/s였다. 3m/s 학습 영역 이외의 몸 감각 변동이 진단을 지배했다. 공간 대비 경계를 0으로 한 대조도 저장했다.

이는 합성 영상과 몸 감각을 의도적으로 불일치시킨 인과적 입력 대조이며, 실제 차량의 속도나 배우 출력이 아니다. 기계적 동역학이나 시각이 속도에 맞춰 변하는 경우의 검증을 대신하지 않는다. 진단기는 3m/s 한 속도에서 학습됐고 속도 감각 자체가 회로 상태에 들어가므로 OOD 추론 실패로 해석한다. 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-frozen-symmetry-body-speed-plan/plan.json`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-frozen-symmetry-body-speed-1/result.json`.


## 2026-11-01 — 영상 이동 속도와 몸 속도를 맞춘 합성 대조

같은 질감 881403·곡률 0.022·시작 거리 11m에서 영상 전방 이동을 1.5/3/4.5m/s로 달리 생성하고 대응하는 몸 속도와 바퀴 속도를 넣었다. 초기 0.4초는 영상이 정지지만 몸 속도는 고정 입력이므로 완전한 일관성은 없으며, 이후 영상 이동과 몸 속도는 일치한다. 회로와 근육 진단기는 고정했고 각 장면을 따로 초기화했다. 직선/좌/우 코너의 진단 예측 평균은 각각 1.5m/s `[187.616, 187.321, 188.109]`, 3m/s `[3.306, 2.313, 2.345]`, 4.5m/s `[-162.393, -162.200, -163.369]`였다. 공간 경계 제거 대조도 결과에 기록했다.

속도 일치 합성 입력에서도 3m/s에만 맞춘 무제한 진단은 몸 상태 변화에 극히 취약했다. 단일 속도 분포의 합성 판독기를 배우로 사용하지 않는 근거이며 기존 배우가 해당 값을 출력했다는 뜻이 아니다. 실제 동역학, 실제 시각, 생리 및 폐루프 검증은 여전히 필요하다. 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-symmetry-matched-speed-plan/plan.json`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-frozen-symmetry-matched-speed-1/result.json`.


## 2026-11-01 — 세 몸 속도의 기준선 학습 시범

속도 변화로 기존 진단기가 폭주한 뒤, 이전에 평가에 사용했던 합성 질감 881403의 1.5/3/4.5m/s 자료를 이번 시범 학습에 재사용했다. 직선 장면의 평균 근육 값에 대해 속도별 선형 기준선을 추정하고 그 잔차와 세 쌍의 좌우 차이 절댓값을 고정 스케일 1e-6·ridge 1로 학습했다. 교사는 직선 3.5, 양 코너 2.5m/s인 임의 합성 가설이다. 출력을 0~5m/s로 제한한 진단이며 새로운 질감 881511은 기준선과 가중치 학습에 쓰지 않았다. 같은 기하·상관된 30프레임, 한 학습 질감과 한 평가 질감만 이용한 사후 시범이다.

| 몸/영상 속도(m/s) | 새 질감 직선/좌/우 예측 평균(m/s) | 직선−코너 평균(m/s) | 시각 경계 제거 후 차이(m/s) |
| ---: | --- | ---: | ---: |
| 1.5 | 3.095 / 2.877 / 3.075 | 0.119 | -0.002 |
| 3.0 | 3.323 / 2.726 / 2.771 | 0.575 | -0.011 |
| 4.5 | 3.187 / 2.507 / 2.588 | 0.640 | 0.009 |

평가 평균제곱오차는 0.216/0.060/0.043이며 고정 2.833m/s의 참조 오차 0.222와 비교한다. 낮은 속도에서 한 코너는 직선과 0.019m/s 차이에 그쳤다. 시각 경계를 인위적으로 끄면 차이가 거의 사라지지만 생리 기능이나 차량 제어의 인과적 성공을 증명하지는 않는다. 한 근육 값에 1e-6을 일괄 더하거나 빼면 출력은 최대 0.440m/s, 고정 근육 상태에서 속도 입력만 ±0.1m/s 바꾸면 최대 0.299~0.376m/s 변했다. 이 교란은 실측 잡음이 아니다. 영상의 첫 0.4초는 정지 장면이고 몸 속도는 고정돼 있다. 실제 영상·동역학·폐루프, 독립 기하·질감과 강건성은 미검증이며 학습한 공학 진단은 배우에 연결하지 않았다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-symmetry-pilot-plan/plan.json`; 결과·원본 해시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-symmetry-pilot-1/result.json`; 교란 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-symmetry-perturbation-1/result.json`.


## 2026-11-01 — 다중 속도 시각 신호와 네 시야 인위적 대조

시드 881403(시범 학습), 881511(평가)의 영상을 같은 프레임·yaw 0에서 직선과 코너 차이의 RMS로 비교했다. 평가 질감에서 1.5m/s 좌코너의 망막 차이 RMS는 0.005024, 운동 채널 차이는 0.005060, 근육 차이는 0.000000523이었다. 3m/s에서는 각각 0.005498, 0.006135, 0.000001634였다. 서로 단위와 차원이 다른 단계 사이 수치로 전달 효율을 계산할 수 없다. 이 조건에서 저속에도 시각 차이는 남았고 근육의 절대 차이는 더 작았다는 한정된 관찰이다. 초기 audit-1은 Windows 기본 인코딩 읽기 실패로 종료됐고 UTF-8 수정 후 audit-2를 실행했다.

네 시야 순서는 전/좌/우/후다. 각 코너의 마지막 프레임들의 망막 차이는 주로 전방에 있고 뒤쪽은 전 속도·좌우 코너 모두 정확히 0이었다. 이에 같은 평가 질감의 코너에서 전방 또는 양옆의 pooling 결과를 같은 시각 직선 장면의 해당 시야로 치환하고 회로를 다시 전개했다. 치환은 실험자만 진실 장면을 이용한 반사실 개입이고 배우의 입력 방법이 아니다. 고정 진단기의 직선−양 코너 평균은 다음과 같다.

| 속도(m/s) | 원래 | 전방 치환 | 양옆 치환 |
| ---: | ---: | ---: | ---: |
| 1.5 | 0.119 | -0.001 | 0.127 |
| 3.0 | 0.575 | -0.0004 | 0.577 |
| 4.5 | 0.640 | 0.0002 | 0.643 |

한 합성 생성기와 질감에 대한 현상이며 실제 전후좌우 겹눈의 일반적 중요도나 생리·자율주행 성능을 뜻하지 않는다. 이 코스에서는 전방이 예측을 주도하므로 사방 동시 시각의 가치를 검증할 수 없다. 측면·후방 장면과 운동이 실제로 변화하도록 설계하고 각 시야를 대조해야 한다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-visual-signal-audit-2/result.json`, 실험 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-view-swap-plan/plan.json`, 치환 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-multispeed-view-swap-1/result.json`.


## 2026-11-01 — 네 시야 합성 회전의 비행 회로 개입

앞선 코너 장면의 뒤쪽이 정적이어서 기존 직선 합성 장면의 회전 조건을 사용했다. 렌더 yaw 속도 0/-0.12/+0.12rad/s, 전진 3m/s이며 몸 yaw는 0으로 고정해 시각만 바꾼 인위적 조건이다. 영상의 첫 0.4초는 정지이고 나머지는 사전 지정한 회전이다. 원본 FlyVis 망막 및 T4/T5 채널로 만든 네 시야 장면을 동일 비행 회로에 넣고 마지막 10프레임을 분석했다.

무회전 대비 회전의 망막 차이 RMS는 전/좌/우/후 약 `[0.033, 0.057, 0.055, 0.025]`, 운동 채널은 `[0.029, 0.048, 0.047, 0.024]`였다. 각 시야 모두 변화했으나 이것은 모델 반응으로서 방향 선택성이나 생체 일치를 따로 검증한 것은 아니다.

같은 시각의 해당 시야만 무회전 장면으로 인위적으로 치환한 뒤 회로 상태를 초기화하고 다시 전개했다. 회전 -/+ 대비 무회전 조향 intent 차이는 원래 `[-0.003065, +0.002525]`, 전방 치환 `[-0.009829, +0.008941]`, 좌측 치환 `[+0.000791, -0.000050]`, 우측 치환 `[+0.001267, -0.000663]`, 후방 치환 `[+0.003730, -0.002129]`, 네 시야 치환 `[0, 0]`이었다. 개별 치환이 반응 방향도 바꿨으므로 이번 자극에서 전·측·후 입력의 기여와 상호작용이 있음을 보여준다. 부호가 실제 차량의 올바른 보정 조향과 같은지는 검증하지 않았다.

이 시야 치환은 실험자가 시각 정보만 바꾼 반사실 실험이며 실제 자연 카메라 손실이 아니다. yaw 감각과 영상이 의도적으로 불일치하여 실제 회전 운전·생리 기능·랩타임을 입증하지 않는다. 실측 연결 기반 회로의 근사, FlyVis 및 공학 입력/출력 경계를 구분한다. 모델 학습·배치·차량 제어는 없었다. 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fourview-yaw-causality-plan/plan.json`, 결과·배열 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fourview-yaw-causality-1/`.


## 2026-11-01 — 시각 회전과 몸 회전 입력의 합성 대조

원본 yaw 영상 881511과 881403에서 yaw 0/-0.12/+0.12rad/s를 비교했다. 차체 계약의 `angular_body.yaw_left_rad_s` 4번 채널을 사용하며 렌더의 양의 Z 회전도 좌회전인 합성 좌표를 사용했다. 초기 0.4초 영상과 몸 속도·각속도는 0, 이후 몸 속도 3m/s와 지정 각속도를 넣었다. 각 질감에서 시각·몸 회전 없음, 시각만, 몸 감각만, 두 입력 동시 네 가지를 비교하고 회로 상태는 장면마다 초기화했다. 몸 yaw 값을 유효 표시했으며 영상이나 몸에 서킷 정답을 주지 않았다.

| 질감 | 입력 | -0.12/+0.12에 대한 조향 intent 변화 |
| --- | --- | --- |
| 881511 | 시각만 | -0.003065 / +0.002525 |
| 881511 | 몸만 | +0.011928 / -0.011900 |
| 881511 | 결합 | +0.008862 / -0.009384 |
| 881403 | 시각만 | +0.002448 / -0.003137 |
| 881403 | 몸만 | +0.011929 / -0.011902 |
| 881403 | 결합 | +0.014383 / -0.015032 |

몸 단독 반응은 양 질감에서 같지만 시각 단독 반응의 부호는 서로 뒤집혔다. 결합에서 시각·몸 개별 차이의 단순 합과 실제 차이의 잔차는 절댓값 최대 약 0.000009였다. 이는 두 자극의 특정 출력 조건에서 거의 선형적으로 더해진 관찰일 뿐 생리적 감각 통합을 입증하지 않는다. 시각의 방향 일관성이 약한 문제가 드러나므로 단순히 사방 입력 반응을 기능 성공으로 판정하지 않는다. 두 질감·단일 기하·상관된 프레임의 한정된 합성 실험이다.

각속도와 시간은 맞췄으나 위치는 월드 전방 직선으로 계속 이동하므로 완전한 회전 운동·차량 타이어 동역학이 아니다. 실제 회전 방향, 조향 표지, 체감 균형, 생리와 실제 폐루프 검증은 별도다. 몸 yaw는 정답 경로가 아닌 허용된 몸 감각이다. 배우 배치·추가 학습·차량 제어는 없었다. 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-matched-body-yaw-plan/plan.json`, 결과와 상태 배열 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-matched-body-yaw-1/`.


## 2026-11-01 — 질감에 따른 회전 부호 반전의 단계 대조

질감 881403/881511에서 같은 픽셀·겹눈 열·마지막 10개 프레임의 `+0.12 yaw - (-0.12 yaw)` 벡터를 비교했다. 망막 전체 코사인 유사도는 0.994, FlyVis 8개 운동 채널은 0.996이었다. 각 시야 운동 채널 유사도도 전/좌/우/후 각각 0.990/0.997/0.999/0.995였다. 높은 전체 유사도는 지배적 패턴의 유사성이지 모든 요소의 부호 일치나 방향 판독·생리 기능의 보증이 아니다. 이전 회로 출력의 시각 단독 조향 반응은 두 질감에서 반대 방향이었다.

두 번째 질감에도 같은 전/좌/우/후 시야별 반사실 치환을 적용했다. -/+ yaw에 대한 조향 변화는 질감 881403 원본 `[+0.002448,-0.003137]`, 전방 치환 `[-0.008966,+0.010723]`; 질감 881511 원본 `[-0.003065,+0.002525]`, 전방 치환 `[-0.009829,+0.008941]`였다. 네 시야 치환은 둘 다 0이다. 전방과 나머지 경로의 상쇄 또는 상호작용에 대한 근거이나, 입력 차이가 매우 작고 개입은 pooling 후 인위적으로 장면을 섞으므로 특정 경계·뉴런이 원인이라고 결론낼 수 없다. 두 질감과 같은 생성기의 상관 프레임에 한정되며 물리 회전이나 주행 성능 검증은 아니다. 배우 변경·학습·차량 조작은 없었다.

입력 비교 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-yaw-texture-visual-correlation-1/result.json`; 두 번째 시야 개입 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-fourview-yaw-causality-881403-1/result.json`; 첫 번째 개입은 `20261101-fourview-yaw-causality-1/result.json`.


## 2026-11-01 — 합성 yaw 반응의 공간 대비·운동 경로 개입

이전 두 질감의 직선·시각 yaw -/+0.12rad/s 영상을 같은 비행 회로에 넣었다. 초기 0.4초 몸 속도·yaw는 0이고 이후 몸 속도 3m/s·몸 yaw 0으로 고정해 시각 입력만 바꾸었다. 저장한 학습 경계는 각 조건에서 복원하고 회로 상태는 리셋했다. 공간 대비 투영, 시간 방향 대항 투영을 각각 0으로 만들거나 pooling된 망막/운동 채널만 같은 시각의 무회전 영상으로 치환하는 반사실 대조다.

| 질감 | 조건 | -/+ yaw 조향 변화 |
| --- | --- | --- |
| 881511 | 원본 | -0.003065 / +0.002525 |
| 881511 | 공간 대비 0 | -0.000069 / -0.000100 |
| 881511 | 운동 방향 투영 0 | -0.002989 / +0.002638 |
| 881511 | 망막 무회전 치환 | -0.000073 / -0.000097 |
| 881511 | 운동 채널 무회전 치환 | -0.002994 / +0.002624 |
| 881403 | 원본 | +0.002448 / -0.003137 |
| 881403 | 공간 대비 0 | -0.000094 / -0.000056 |
| 881403 | 운동 방향 투영 0 | +0.002542 / -0.003065 |
| 881403 | 망막 무회전 치환 | -0.000093 / -0.000057 |
| 881403 | 운동 채널 무회전 치환 | +0.002544 / -0.003078 |

두 질감 모두 이 자극에서는 공간 대비/망막 경로가 조향 변화를 주도했고, 대항 운동 투영이나 운동 채널의 회전 정보를 없애도 변화가 대부분 유지됐다. 한 투영을 끄는 것은 모든 운동 시각을 끄는 것과 같지 않지만 별도의 전체 운동 채널 치환도 같은 경향을 보였다. 인위적 치환의 두 합성 질감·단일 yaw 속도·상관 프레임에 한정된다. 운동 시각의 전체 기능 부재를 증명하지는 않지만 현재 시각 yaw 조향을 운동 시각 기반 안정화로 인정할 수 없고, 원래 목표에 맞춰 T4/T5 방향·시간 → 같은 비행 회로 → 조향의 인과 검증과 개선이 우선이다. 배우 수정·학습·차량 제어는 없었다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-yaw-path-ablation-plan/plan.json`; 결과와 배열 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-yaw-path-ablation-1/`.


## 2026-11-01 — 합성 운동 시각 전용 yaw 경계 학습

앞선 회전 반응이 망막 공간 대비에 치우친 것을 확인한 뒤, 단일 비행 그래프와 몸 경계·기존 근육 출력은 동결하고 `OpponentTemporalMotionBoundary.opponent_projection.weight`만 Adam 0.002로 6회 갱신했다. 학습 질감 881301/881302의 좌우 yaw ±0.12rad/s에서 망막 풀링 채널을 같은 시각 무회전 값으로 맞추고 공간 대비 투영을 끄며 T4/T5 운동 정보는 남겼다. 목표는 `yaw - 조향 − yaw + 조향 = +0.004`인 공학적 가설이며 생리학적 표적이나 서킷 정답은 아니다. 마지막 10개 프레임을 사용했고 학습 손실은 1.57e-5에서 9.15e-6으로 감소했다.

학습에 쓰지 않은 새 질감 881601, 같은 합성 생성기의 결과는 다음과 같다.

| 조건 | -yaw 조향 − +yaw 조향 |
| --- | ---: |
| 기존 경계, 전체 시각 | -0.003510 |
| 학습 후보, 전체 시각 | -0.002387 |
| 기존 경계, 운동만 | +0.000096 |
| 학습 후보, 운동만 | +0.001227 |
| 후보에서 대항 운동 투영 제거, 운동만 | +0.000003 |
| 후보에서 운동 채널 무회전 치환, 운동만 | +0.000010 |

운동 단독 반응 증가와 투영·채널 제거 시 소실은 이 합성 자극에서 해당 공학 경계가 같은 회로 조향에 인과적으로 기여함을 지지한다. 목표 +0.004에는 못 미치고 원래 공간 경로를 켠 전체 출력은 반대 부호다. 샘플은 상관된 연속 프레임, 학습은 두 질감, 평가는 한 질감·단일 기하·단일 각속도다. 초기 정지 후 월드 직선 병진과 별개로 회전을 지정했고 몸 yaw 0이므로 실제 차량의 회전 안정화가 아니다. 후보는 배우 미적용·차량 제어 없음이며 생리학적 기능이나 실제 주행 성능이 입증되지 않았다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-motion-only-yaw-training-plan/plan.json`; 원본 결과·후보 해시·학습 손실 및 평가 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-motion-only-yaw-training-1/result.json`; 평가 영상 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-motion-only-yaw-validation-881601/`.


## 2026-11-01 — 전체 시각·운동 단독을 함께 이용한 합성 yaw 학습

직전 운동 전용 후보를 출발점으로 단일 비행 그래프, 기존 공간 경계와 몸 경계는 고정하고 T4/T5 대항 투영만 Adam 0.002로 6회 추가 갱신했다. 기존 합성 질감 881301/881302에서 전체 시각과 망막 회전 단서 제거·공간 대비 차단의 운동 단독 조건을 같은 가중치로 학습했다. -yaw와 +yaw의 조향 차이 +0.004는 운전 안정화를 가정한 공학 표적이다. 손실은 1.888e-5에서 1.210e-5로 줄었다.

별도로 생성한 질감 881602에서 직선 평가의 기존 후보 전체 차이는 +0.002737, 새 후보는 +0.003819였다. 운동 단독은 +0.001085→+0.002161이며, 새 후보에서 대항 투영을 끄면 -0.000010이고 운동 채널을 무회전으로 고정하면 +0.000006이었다. 전체 입력에서 운동 채널만 고정하면 +0.001652였다. 원본 학습 result의 `held texture and geometry` 문구는 직선 장면만 평가하므로 기하까지 독립 검증했다는 의미로 읽으면 안 된다.

그 뒤 재학습 없이 같은 새 질감의 좌우 코너 자극도 별도로 평가했다.

| 영상 | 기존 후보 전체 차이 | 새 후보 전체 차이 | 새 후보, 운동 채널 고정 | 새 후보, 운동만 |
| --- | ---: | ---: | ---: | ---: |
| 직선 | 0.002737 | 0.003819 | 0.001652 | 0.002161 |
| 좌 코너 | 0.002686 | 0.003761 | 0.001629 | 0.002156 |
| 우 코너 | 0.003023 | 0.004113 | 0.001936 | 0.002155 |

운동 단독의 대항 투영을 끄면 세 경우 모두 약 -0.00001이었다. 이 결과는 한 새 질감·한 곡률·한 회전 속도의 합성 경로 기여다. 몸 yaw는 0으로 고정했고 움직임은 미리 정해진 직선 병진과 회전의 조합이므로 실제 차체 동역학이나 안정화·생리 기능이 아니다. 정지 도입 이후 3m/s, 상관된 마지막 10프레임을 사용했으며 배우에 배치하지 않았다. 강한 공간 시각 영향, 입력 결손·속도·질감·각속도·실제 영상과 폐루프 일반화는 추가 검증해야 한다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-motion-yaw-plan/plan.json`; 학습·직선 결과 및 후보 해시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-motion-yaw-training-1/result.json`; 추가 기하 대조 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-motion-bend-eval-1/result.json`.


## 2026-11-01 — 통합 운동 후보의 기존 질감·몸 yaw 대조

직전 합성 학습의 새 후보를 동결해 기존에 살펴본 세 질감 881403/881511/881601에서 직선 영상 ±0.12rad/s 회전을 재생했다. 이들은 두 학습 질감에는 포함되지 않았으나 이전에 실험자가 결과를 본 자료로, 눈가림된 최종 시험은 아니다. 몸 yaw 0인 시각 전용과 영상과 몸 yaw를 같은 부호·시간으로 맞춘 조건을 비교했고, 후자에서 운동 채널이나 대항 투영만 제거했다. 조향 차이는 `-yaw − +yaw`다.

| 질감 | 이전 후보 시각만 | 새 후보 시각만 | 새 후보 몸 yaw 결합 | 결합 중 운동 채널 고정 |
| --- | ---: | ---: | ---: | ---: |
| 881403 | +0.006725 | +0.007872 | +0.031706 | +0.029463 |
| 881511 | -0.004497 | -0.003413 | +0.020419 | +0.018217 |
| 881601 | -0.002387 | -0.001260 | +0.022575 | +0.020226 |

몸 yaw 결합 중 대항 투영 제거는 각각 +0.029438/+0.018208/+0.020229로 채널 고정과 비슷했다. 새 운동 경계가 세 조건에서 차이에 기여하지만, 시각 단독 부호는 세 질감 중 두 개에서 목표와 반대로 남았다. 몸 감각에 의한 총 출력은 양수가 됐으나 시각 방향이 일관돼서 그런 것은 아니다. 질감과 공간·운동 경로 상쇄 문제를 해결하기 전 후보를 배우에 적용하지 않는다.

사전 지정 회전과 월드 직선 병진, 정지 접두 구간 후 3m/s의 개루프 합성이다. 실제 차량 동역학, 생리, 다른 속도·각속도나 독립 질감 일반화, 폐루프·랩타임은 미검증이다. 결과 및 해시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-motion-matched-body-audit-1/result.json`; 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-joint-motion-matched-body-audit-plan/plan.json`. 배우 배치·추가 학습·차량 제어 없음.


## 2026-11-01 — 질감 다양화 후 운동 시각 경계 재학습

두 조건에서 시각 단독 조향 부호가 반대로 남아, 기존 학습 881301/881302에 실패 질감 881511을 추가했다. 단일 비행 그래프·공간·몸 경계는 고정하고 기존 후보의 운동 대항 투영만 전체 시각/운동 단독 조건에서 Adam 0.002로 6회 갱신했다. 학습은 yaw ±0.12rad/s와 가설 조향 차이 +0.004만 사용했다. 손실은 1.702e-5→1.148e-5였다.

학습에 사용하지 않은 새 영상의 결과는 다음과 같다. 각속도와 질감을 동시에 바꾸었기 때문에 속도 단독 효과로 해석할 수 없다.

| 질감 / yaw 크기 | 기존 후보 전체 | 새 후보 전체 | 새 후보 운동만 | 새 후보 운동 투영 제거, 운동만 |
| --- | ---: | ---: | ---: | ---: |
| 881603 / 0.08rad/s | 0.011243 | 0.012039 | 0.002359 | -0.000020 |
| 881604 / 0.20rad/s | 0.003843 | 0.004709 | 0.002804 | -0.000009 |

동일한 새 질감의 각속도 비교가 아니며, 작은 yaw에서 전체 출력이 더 큰 현상은 질감·공간 경로 의존성 가능성을 남긴다. 이어서 이미 관찰했던 세 질감의 yaw ±0.12를 재검사한 시각 전용 결과는 다음과 같다. 881511은 이번 학습에 포함됐고 나머지는 포함되지 않았지만 이전 연구에서 관찰됐다.

| 질감 | 직전 후보 | 새 후보 | 자료 구분 |
| --- | ---: | ---: | --- |
| 881403 | +0.007872 | +0.008989 | 이전 관찰, 이번 학습 제외 |
| 881511 | -0.003413 | -0.002344 | 이번 학습 포함 |
| 881601 | -0.001260 | -0.000164 | 이전 관찰, 이번 학습 제외 |

몸 yaw를 결합한 총 차이는 각각 +0.032823/+0.021486/+0.023671로 양수이나 몸 감각이 크게 기여한다. 운동 경로 기여가 늘어도 시각 단독의 질감 방향 반전은 완전히 해결되지 않았다. 특히 학습한 실패 장면에서도 반대 부호다. 이 연구용 후보는 배우 미적용이며, 운동 투영만 더 조정하는 것으로 생리 기능·차량 제어가 해결됐다고 보지 않는다. 새 평가도 동일 생성기·직선 장면·상관된 10프레임·몸 yaw 0의 개루프 합성이고 실제 영상·차량 안정화·랩타임은 미검증이다.

학습 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-diverse-yaw-motion-plan/plan.json`; 학습·새 영상 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-diverse-yaw-motion-training-1/result.json`; 알려진 장면 재평가 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-diverse-yaw-known-texture-audit-1/result.json`.


## 2026-11-01 — 코너 반응 보존을 포함한 두 시각 경계 학습 실패

이전 운동 투영 후보에서 단일 비행 그래프와 몸 경계를 동결하고 운동 대항 투영과 망막 공간 대비 투영을 각각 Adam 0.001/0.0005로 4회 갱신했다. 학습 질감 881301/881302/881511에 대해 전체 시각 yaw 차이 +0.004, 망막 yaw 단서 치환·운동 단독 차이 +0.004, 운동 yaw 단서 치환·공간 단독 차이 0, 기존 모델의 동일 질감 좌/우 코너 조향 차이 유지의 네 제곱오차를 균등하게 사용했다. 마지막 조건은 기존 출력 보존이지 레이싱 라인 정답이 아니다. 전체 손실은 9.207e-6→8.641e-6이었다.

학습에서 제외한 새 합성 질감의 `-yaw 조향 − +yaw 조향` 및 `왼쪽 코너 조향 − 오른쪽 코너 조향` 차이는 다음과 같다.

| 질감 | 이전 전체 yaw | 새 전체 yaw | 새 운동 단독 | 새 공간 단독 | 기존→새 코너 차이 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 881605 | -0.006091 | -0.005778 | +0.003587 | -0.009360 | -0.006990→-0.006991 |
| 881606 | +0.004723 | +0.005047 | +0.003569 | +0.001474 | -0.007050→-0.007051 |

두 질감의 코너 차이는 유지됐지만 881605에서 공간 경로의 반대 방향 반응이 운동 반응보다 커 전체 방향이 여전히 반대였다. 공간 가중치는 변경됐지만(초기 대비 L2 0.0257) 이 네 회의 학습으로 해당 문제를 해결하지 못했다. 이는 기존 코너 반응 보존이 올바른 코너 주행을 증명한다는 뜻도 아니다. 새 질감 두 개, 같은 생성기·각속도·고정 몸 yaw·상관된 프레임의 제한된 개루프 합성 대조다. 생리 기능, 실제 영상·몸 운동, 완주·랩타임은 검증되지 않았다. 연구 후보는 배우에 적용하지 않았다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-corner-preserving-visual-plan/plan.json`; 학습 기록·검증·후보 해시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-corner-preserving-visual-training-1/result.json`.


## 2026-11-01 — 공간 시야별 회전 신호 상쇄와 소규모 개입

실패한 공동 시각 후보를 고정하고 두 새 질감 881605/881606의 yaw +0.12와 -0.12에서 마지막 10프레임을 비교했다. 4×16 망막 공간 pool의 시야 평균과 고정 기준을 뺀 다음 전체 RMS 제한과 공간 투영을 거쳤다. 두 질감의 정규화 입력 변화 벡터 코사인은 0.964였으나 42개 포트 투영 전 활성 변화 벡터의 코사인은 0.234였다. 각 시야를 개별 투영한 변화의 질감 간 코사인은 모두 0.998 이상이었다. 이는 시야별 큰 벡터가 합쳐질 때 작은 상대 크기 차이에 민감하다는 관찰이며 조향이나 생리의 직접 척도는 아니다.

네 시야 투영별 RMS의 합에 대한 전체 합산 RMS는 881605에서 0.158, 881606에서 0.176이었다. 전방 벡터와 나머지 세 시야 합의 코사인은 각각 -0.960/-0.973이었다. 해당 선형 투영 단계의 강한 상쇄이며, tanh 이후나 비행 회로 출력에서 같은 비율로 상쇄된다고 가정하지 않는다. 전체 RMS 상한은 30프레임 중 20프레임에서 작동했다.

같은 단일 회로에서 전방 공간 가중치 열만 ±5% 바꾸는 개입을 수행했다. `-yaw−+yaw 조향 / 좌−우 코너 조향` 차이는 881605 기준 `-0.005778/-0.006991`, 전방 0.95배 `-0.006612/-0.006688`, 1.05배 `-0.004917/-0.007284`였다. 881606 기준 `+0.005047/-0.007051`, 0.95배 `+0.003648/-0.006735`, 1.05배 `+0.006433/-0.007361`였다. 작은 전방 조정에도 회전과 코너 반응이 함께 바뀌지만 실패 질감의 부호는 바뀌지 않았다. 한 가중치 조정을 해결책으로 삼지 않았고 배우에 적용하지 않았다.

두 합성 질감·한 속도와 각속도, 정해진 반사실 매개변수 개입이며 실제 환경·생리·폐루프 검증이 아니다. 경계 진단 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-yaw-normalization-audit-1/result.json` 및 배열 `spatial-yaw-vectors.npz`; 개입 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-view-gain-counterfactual-plan/plan.json`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-view-gain-counterfactual-1/result.json`.


## 2026-11-01 — 공간 경계 평균 특징의 선형 제약 시범 실패

시야 간 상쇄를 보고 기존 공동 시각 후보를 출발점으로 3개 학습 질감의 마지막 10프레임에서 회전 `+yaw − -yaw`의 정규화 망막 pool 평균 방향과 무회전 `좌 코너 − 우 코너` 평균 방향을 뽑았다. 64차원에서 코너 3방향의 직교 여공간 안에서 최소 노름으로 공간 투영의 회전 방향을 제거하는 선형 제약해를 계산했다(SVD rcond 1e-4, 변화량 제한 기존 가중치 norm 50%). 이 회전 불변 조건은 공학적 가설이며 뇌의 회로 규칙이 아니다.

학습 특징의 공간 투영 회전 노름은 43.87에서 약 2.2e-13, 코너 투영 변화 노름은 약 1.2e-13이었다(64비트 선형 계산). 실제 저장 가중치 변화 norm은 기존의 29.5%였다. 이는 평균 특징의 투영 전 제약이지 비선형 `tanh`, 시간 전개, 개별 프레임의 완전한 보존이 아니다.

미학습 신질감의 실제 비행 회로 대조: 881607 전체 yaw 조향 차이 +0.005551→+0.004601, 코너 차이 -0.007095→-0.007054; 881608 전체 yaw +0.002916→+0.001866, 코너 -0.007086→-0.006497(약 8.3% 감소). 두 조건에서 운동만 차이는 +0.003626/+0.003651, 운동 투영 차단 시 거의 0이었다.

그 다음 이미 봤던 실패/성공 질감의 사후 점검: 881605 전체 yaw -0.005778→-0.005014로 여전히 목표와 반대, 코너 -0.006991→-0.006973; 881606 전체 yaw +0.005047→+0.003641, 코너 -0.007051→-0.004663(약 34% 감소). 따라서 학습 특징의 선형 제약이 다른 장면의 방향과 코너 반응을 보장하지 못했다. 이 후보는 기각·배우 미적용이다. 각속도·속도·기하가 제한된 개루프 합성이고 실제 차량, 사방 실제 영상 및 초파리 생리 검증이 아니다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-constraint-pilot-plan/plan.json`; 제약 계산·행렬 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-constraint-pilot-fit-1/`; 신질감 검증 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-constraint-pilot-eval-1/result.json`; 기존 질감 사후 점검 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-spatial-constraint-known-audit-1/result.json`.


## 2026-11-01 — 실제 AC 네 방향 정지 기록의 공간 경계 동작 범위

기존 `20261101-stationary-native-only-fourview-capture` 원본 기록과 이미 완료된 `20261101-stationary-native-flyvis-geometry-replay-v2`의 해시를 확인했다. 유효 네 시야 23/24프레임의 FlyVis 망막 샘플을 최근 공동 학습 공간 경계와 동일한 4×16 pool·고정 reference·RMS 상한에 넣어, 학습을 하지 않고 유효 범위만 분석했다. 합성 질감 881605/881606의 직선·회전 30프레임씩과 계산 방식은 같지만 실제는 정지 기록이고 합성은 정해진 회전이다.

| 조건 | 중심 대비 RMS 중앙값 | 상한 gain 중앙값 | 상한 적용 비율 | 공간 투영 전 활성 절댓값 95백분위 |
| --- | ---: | ---: | ---: | ---: |
| AC 기록 유효 23 | 0.07883 | 0.0411 | 100% | 2.832 |
| 합성 881605 | 0.00410 | 0.7905 | 66.7% | 0.592 |
| 합성 881606 | 0.00427 | 0.7589 | 66.7% | 0.720 |

고정 합성 기준 차감 후 실제 기록의 잔차 대비는 합성의 약 18~19배였고 해당 후보의 상한이 모든 프레임을 강하게 축소했다. 후속 검사에서 기준 차감 전 대비는 실제와 합성에서 비슷했다. tanh 이전 절댓값 3 초과 포트는 AC에서 약 2.6%, 두 합성에서는 0%였다. 이는 이 정지 장면에 대한 경계 동작 범위 차이이며 직접적인 주행 오류나 일반화 실패의 확정은 아니다. AC 망막은 `ac-csp-yaw-retina.json`, 합성은 `ac-csp-exterior-front-v3-retina.json`을 사용한다. 두 설정의 화면 표본 좌표와 시야 방향은 후속 동등성 감사에서 같았으며, 실제 카메라 원점은 미검증이다. 후속 동등성 감사에서 두 설정의 화면 표본 좌표·시야 방향은 동일함이 확인됐다. 실제 카메라 원점은 yaw 설정에 기록되지 않아 합성 원점과의 일치는 미검증이다. 두 조건의 시간 간격과 운동도 다르며 합성 배열에서 순서상 조건 경계를 포함하는 `mean_abs_step`은 시간 비교 지표로 사용하지 않는다.

AC 원본은 정지 native-only 기록으로 학습 부적격이며 물리 노출 시각과 몸 동기화는 확인되지 않았다. 실시간 파일 카메라 통합 시도는 별도로 render 신선도 가드에서 중단됐고 회피하지 않았다. 기록은 오프라인으로만 분석했고 배우 배치·학습·차량 명령은 없다. 동일 카메라 보정과 시간 계약에서의 동적 네 시야·몸 감각 검증이 필요하다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-actual-retina-spatial-range-audit-1/result.json`; 원본 replay `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-native-flyvis-geometry-replay-v2/`.


## 2026-11-01 — 실제 네 방향 이미지의 두 카메라 설정 표본 동등성

앞선 실제·합성 대비 차이를 카메라 설정 차이로 설명할 수 있는지 원본 이미지로 점검했다. 기존 native-only 기록의 24×4 PNG에 대해 archive와 이벤트 해시를 각각 검증하고, 기록의 721개 `u/v` 순서로 `ac-csp-yaw-retina.json`과 `ac-csp-exterior-front-v3-retina.json`의 `FlyvisCameraProjector`를 각각 적용했다. 원본 96장 모두 두 망막 표본의 최대 차이 0이었고 기존 FlyVis replay에 저장된 망막 값과도 차이 0이었다. UV, 수용 표본점, 네 시야 body 방향의 최대 차이도 0이다. 각 시야의 투영 계수 및 방향 행렬이 같다.

따라서 동일한 RGB 이미지에 대한 카메라 로컬 망막 표본 처리 차이는 앞서 관찰한 대비 범위 차이의 원인이 아니다. 두 설정 파일은 다르고 실제 yaw 설정에는 카메라 `origin_body_m`이 없어 합성 렌더의 가정 원점과 실제 카메라 시점의 일치는 확인되지 않았다. 실제와 합성의 장면 내용·동작·시점·노출 시간도 다르므로 차이의 다른 원인을 특정할 수는 없다. 생리 기능, 실제 동적 운전, 물리 시각 동기화 검증도 아니다. 기록만 읽었고 학습·배우 배치·차량 조작은 없었다.

결과와 원본·설정 해시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-camera-projector-equivalence-audit-1/result.json`.


## 2026-11-01 — 고정 합성 대비 기준과 실제 기록의 범위

앞서 비교한 RMS는 시야 평균을 제거한 다음 **고정 synthetic reference까지 차감한 잔차**다. 원래 보고에서 이를 raw contrast라고 표현한 부분은 잘못으로 위에서 수정했다. 동일 pool에서 reference 차감 전후를 재계산했다. 실제 정지 23프레임의 차감 전/후 RMS 중앙값은 0.08525/0.07883, 합성 881605는 0.09311/0.00410, 합성 881606은 0.09093/0.00427이었다. 고정 reference 자체의 RMS는 0.09283이다. 이 두 합성 장면은 기준과 유사해 잔차가 원래 대비의 약 4.4~4.7%인 반면 실제 정지는 약 92.5%였다. 개별 장면의 묘사이며 실제 영상 전체의 품질이나 주행 성능은 아니다.

이전에 이미 평가했고 학습 부적격으로 표시된 AI 교사 주행 캐시 episode03/04/06/07의 원본 해시와 자격을 확인하고, 고정 기준 및 같은 상한 0.003242를 피팅 없이 적용했다. 각각 742/742/741/741개 저장 장면의 기준 차감 후 RMS 중앙값은 0.0731/0.0917/0.1341/0.1391, gain 중앙값은 0.0444/0.0354/0.0242/0.0233이었고 전 샘플에서 상한이 작동했다. 기준 차감 후/전 RMS 비율 중앙값은 1.11/1.55/1.46/2.00이었다. 이는 합성 기준과 실제 장면의 구조적 차이가 큰 범위 문제를 보여주나, 캐시에서는 시야별 원래 시점 유효성을 재구성하지 않았고 물리 노출 동기·실제 배우 조작은 입증하지 않는다.

실제 기록으로 reference나 가중치를 재학습·선택하지 않았다. 기준을 단순히 지우거나 실제 정지 이미지로 맞추면 다른 시각 기능과 인과성에 영향을 주므로 별도 허용·독립 검증 없이는 배우에 반영하지 않는다. 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-reference-stationary-synthetic-audit-1/result.json`, 주행 캐시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-historical-reference-range-audit-1/result.json`.


## 2026-11-01 — 인과적 공간 기준의 오프라인 진단

고정 합성 기준의 실제 영상 분포 문제를 조사하기 위해, 기존 `MotionFilters`의 공학적 시간 적응 예에서 0.15초 시정수를 참고하고 2초를 사전 선언한 비교값으로 사용했다. 첫 관측의 4×16 중심 망막 공간을 기준 상태로 초기화해 첫 출력은 0, 이후 각 관측에서 이전 기준을 뺀 잔차를 계산한 뒤 기록 시각 간격에 따른 지수 갱신을 한다. 에피소드마다 초기화하며 미래 프레임을 사용하지 않는다. 고정 weight·상한은 그대로다. 이 계산은 실제 영상으로 기준 **상태를 초기화·갱신하는 입력 조건부 적응**이며 고정 가중치나 시간 상수를 실제 기록으로 fitting하지 않았다. 실험 계획의 `calibration_or_fit_on_real=false`는 가중치 fitting을 뜻하도록 한 표현이 부정확해, 실제 프레임 기반 초기화·적응이 있었음을 `reference-state-clarification.json`에 명시했다.

학습 부적격인 과거 주행 캐시 03/04/06/07의 시작 1초 이후 대비 gain 중앙값은 고정 기준 0.044/0.035/0.024/0.024, 0.15초 적응은 1.000/1.000/0.886/0.826, 2초 적응은 0.356/0.171/0.151/0.165였다. 기록에서 시야별 캡처 시간·유효성을 재구성하지 않았기 때문에 실제 실시간 사용 자격을 뜻하지 않는다.

다만 합성 881605/881606의 마지막 10프레임 공간 포트의 좌우 코너 차이 RMS는 고정 기준 약 0.136/0.136에서 0.15초 적응 시 0.073/0.074, 2초 적응 시 0.077/0.078로 줄었다. yaw 차이 RMS는 고정 0.245/0.269, 0.15초 0.248/0.275, 2초 0.296/0.320이었다. 이 값은 42개 공간 포트의 직접 대조이며 같은 비행 회로의 조향, 코너 인식, 인과적 운전 성공이나 생리 반응을 증명하지 않는다. 정적인 전방 선행 구조를 없앨 수 있어 단순 시간 적응을 해결책으로 채택하지 않았다.

두 합성 질감·4개의 이미 관찰한 실제 기록, 고정된 두 시간 상수의 진단이다. 실제 학습·배우 적용·차량 제어는 없고 물리 시간·시야별 신선도·폐루프 검증은 남는다. 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-retina-reference-diagnostic-plan/plan.json`; 결과·배열·상태 초기화 명세 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-causal-retina-reference-diagnostic-1/`.


## 2026-11-01 — 고정 공간 기준 제거의 실제·합성 대조

고정 synthetic reference 자체가 실제 영상 잔차를 크게 만든다는 관찰 후, 임의 0 reference를 사후 대조했다. 공간 투영과 global RMS 상한, 운동 경계, 단일 비행 그래프는 유지했고 기준을 0으로 하는 경우와 공간 경계 전체를 제거한 경우를 별도로 비교했다. 이는 기존 결과를 본 후 설계한 공학적 개입이지 초파리 적응이나 배포 후보가 아니다.

기존 fit 금지 실제 AI 기록 03/04/06/07의 공간 포트 계산에서 gain 중앙값은 기존 0.0444/0.0354/0.0242/0.0233, 0 기준 0.0502/0.0556/0.0362/0.0465였고 두 방법 모두 저장된 모든 장면에 상한이 작동했다. 포트 출력의 평균 절대 차이는 각각 0.0163/0.0559/0.0412/0.0729였다. 실제 영상에서 경계 입력을 크게 바꾸며, 별도 시야별 유효성/시간 검증이나 실제 차량 조작은 하지 않았다.

이전에 본 합성 질감 881605/606/607/608에서 시각 회전 `-yaw−+yaw` 차이는 고정 기준 -0.005778/+0.005047/+0.005551/+0.002916, 0 기준 +0.003147/+0.003646/+0.003715/+0.003621이었다. 그러나 좌우 코너 조향 차이는 고정 기준 -0.006991/-0.007051/-0.007095/-0.007086에서 0 기준 -0.000278/-0.000292/-0.000291/-0.000286으로 대략 96% 줄었다. 공간 경계 전체 차단 시 코너 차이는 약 -0.00002였다. 이 합성 조건에서는 기준 제거가 회전 부호는 바꾸지만 정적 코너 구분을 사실상 약화시킨다. 기존 고정 기준의 코너 신호 자체가 올바른 운전인지도 미검증이다.

실제 자료로 기준·가중치를 fitting하지 않았고 배우에 적용하지 않았다. 같은 생성기·한 속도·한 각속도, 미동기 실제 저장 장면 및 개루프 대조만으로 생리 기능·실제 차량 성능을 주장할 수 없다. 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-retinal-reference-plan/plan.json`; 실제 포트 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-reference-real-port-audit-1/result.json`; 합성 전체 회로 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-reference-synthetic-circuit-audit-1/result.json`.


## 2026-11-01 — 고정 기준 없는 공간 경계의 합성 재학습 실패

고정 합성 reference를 0으로 바꾸면 코너 반응이 사라졌기에, 기존 공간·운동 후보에서 단일 비행 그래프와 몸·운동 시각 경계는 고정하고 0 기준의 공간 투영만 재학습했다. 세 합성 질감 881301/881302/881511에서 전체 시각의 `-yaw−+yaw` +0.004, 운동 채널 무회전 치환 조건 0, 기존 고정 기준 후보의 동일 자료 `좌−우 코너 조향` 차이 유지라는 세 가설 손실을 균등하게 사용했다. 공간 가중치 Adam 0.1, 6회, 초기 norm 대비 변화 20% 상한을 사전 선언했다. 기존 출력 보존은 올바른 레이싱 라인이나 생리학적 정답이 아니다.

전체 학습 손실은 8.95516e-6→8.95337e-6으로 사실상 정체했고, 공간 기울기 norm은 약 5.51~5.52e-9였다. 공간 가중치 변화 norm은 0.323, 시작 norm은 1000.000이며 상한에는 걸리지 않았다. 기울기가 작은 이유를 포화·회로 전달·손실의 하나로 단정하지 않는다.

학습에 쓰지 않은 새 질감 881609/881610에서, 고정 기준의 기존 코너 차이는 -0.004856/-0.006630, 0 기준 학습 전 -0.000271/-0.000281, 0 기준 학습 후 -0.000271/-0.000281이었다. 새 후보의 전체 시각 yaw 차이는 +0.003128/+0.003371, 공간 차단 시 +0.003675/+0.003590으로 대부분 운동 경로에 남았다. 코너 신호를 복구하지 못해 후보는 기각했고 실제 기록으로 fitting하거나 배우에 적용하지 않았다. 두 질감·동일 생성기·단일 각속도 및 시간 상관 프레임의 개루프 합성 결과이며 생리·실제 주행 검증이 아니다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-reference-free-spatial-training-plan/plan.json`; 원본 결과·후보/데이터 해시 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-reference-free-spatial-training-1/result.json`.


## 2026-11-01 — 0 기준 공간 기울기와 대비 상한 개입

같은 고정 비행 회로·후보 가중치로 합성 881609/881610의 좌·우 코너를 각각 41단계 전개하고 마지막 10프레임 평균 조향 차이의 autograd를 추적했다. 고정 synthetic reference에서 공간 가중치의 좌우 차이 gradient norm은 1.275e-5/1.566e-5, 0 reference에서는 7.826e-7/1.029e-6이었다. 0 기준의 좌/우 개별 gradient norm은 각각 약 7.6e-5로 남았지만 두 gradient의 코사인은 0.999947/0.999909라 차이를 내면서 대부분 상쇄됐다. 이 값은 손실 자체의 기울기가 아닌 조향 차이의 Jacobian이다.

0 기준의 대비 상한 gain 중앙값은 0.0337/0.0344였고, 좌우 정규화 망막 특징 차이 RMS는 8.77e-6/8.93e-6이었다. 고정 기준에서는 각각 gain 0.710/0.958, 특징 차이 RMS 1.856e-4/2.395e-4였다. 이 두 조건에서 공간 `tanh` 미분의 전체 평균은 0 기준 약 0.500, 고정 기준 0.494/0.487이고 `|preactivation|>3`은 0%였다. 따라서 이 조건의 해당 공간 `tanh` 포화를 작은 기울기의 설명으로 삼기 어렵다. 포트 출력에서 조향까지의 gradient norm은 약 0.00214~0.00216으로 두 조건에서 비슷했다. 이는 단계별 국소 진단이지 모든 뉴런·상태의 전역 원인을 확정하지 않는다.

같은 회로에서 0 기준의 상한만 인위적으로 1/10/30배로 바꾼 사후 대조는 다음과 같다. 값은 `-yaw−+yaw 조향 / 좌−우 코너 조향`이다.

| 질감 | 상한 1배 | 상한 10배 | 상한 30배 |
| --- | --- | --- | --- |
| 881609 | +0.003126 / -0.000271 | -0.001673 / -0.002482 | -0.009608 / -0.007015 |
| 881610 | +0.003369 / -0.000281 | +0.001378 / -0.002542 | -0.002480 / -0.006997 |

상한을 키우면 코너 차이는 커지지만 yaw 조향 부호가 뒤집힌다. 이는 크기만 키워서는 공간·운동 경로의 방향 충돌을 해결하지 못한다는 이 합성 조건의 반사실이다. 두 질감은 이미 검토됐고 몸 yaw 0, 단일 속도·각속도, 시간 상관 프레임이며 실제 신선도·생리·폐루프 검증이 아니다. 어느 상한도 배우에 적용하지 않았다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-ref-gradient-trace-plan/plan.json`; gradient 결과·배열 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-ref-gradient-trace-1/`; 상한 개입 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-ref-limit-sensitivity-plan/plan.json`, 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-zero-ref-limit-sensitivity-1/result.json`.


## 2026-11-01 — 실제 주행 캐시의 0 기준 대비 상한 확대 대조

앞선 합성에서 0 기준의 상한을 키우면 코너 차이가 커졌지만 yaw 방향이 뒤집혔다. 같은 공간 weight를 유지하고 학습 부적격인 이전 AI 교사 기록 03/04/06/07의 저장 pooled scene에 상한 1/10/30배를 사후 계산했다. 실제 영상으로 reference·weight·상한을 fitting하거나 차량에 적용하지 않았다.

| 기록 | 상한 1배 `|pre|>3` | 10배 | 30배 | 30배에서 `tanh` 미분 <0.01 |
| --- | ---: | ---: | ---: | ---: |
| episode03 | 0.0% | 46.2% | 67.1% | 70.8% |
| episode04 | 0.1% | 64.8% | 82.0% | 84.6% |
| episode06 | 0.3% | 56.2% | 80.4% | 82.5% |
| episode07 | 1.1% | 50.4% | 70.9% | 73.9% |

여기서 미분은 공간 포트 함수 `0.5*tanh(pre)`의 pre에 대한 기울기다. 합성 코너 차이를 되살린 상한 30배는 이 과거 영상에서 다수 포트를 포화시키므로 그대로 일반화할 근거가 없다. 같은 상한에서 생리·주행 실패를 증명한 것은 아니며 시야별 원래 신선도와 물리 발생 시각은 재구성하지 않았다. 실제 차량 제어·학습·배우 적용은 없다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-real-zero-ref-limit-audit-1/result.json`.


## 2026-11-01 — 양측 대칭 공간 시각 경계와 합성 학습

고정 synthetic reference와 상한의 실제 범위 문제를 다루기 위해 `BilateralRetinalContrastBoundary`를 같은 단일 비행 그래프의 선택형 공간 시각 입력으로 구현했다. 4시야의 4×4 망막 pool에서 전·후방 좌우 반대 성분과 좌우 시야의 거울 차이를 형성하고, 중심 대비 크기+0.02로 정규화해 기존 시각 포트에 학습 가능한 bounded 투영을 보낸다. 네 시야 결측·비유한 값은 명시 거부하고 가중치 0 초기화다. 카메라의 좌우 반사 대응과 정규화는 공학적 가정이며 측정된 초파리 겹눈이나 연결이라고 주장하지 않는다. 별도 운동 회로·뇌는 만들지 않았다. `UnifiedFlightSensorimotor`는 명시적인 해당 타입을 선택할 수 있으며 기본 경로는 바뀌지 않았다. 반사 부호, 유효성, 0 초기화의 기존 출력과 단일 상태 보존 및 생성자 포트 검증을 포함한 관련 22개 검사가 통과했다.

기존 후보의 비행 그래프·몸·운동 시각은 고정하고 3개 합성 질감에서 새 투영만 Adam 0.01로 6회 학습했다. 전체 시각의 yaw 차이 +0.004, 운동 고정 시 공간 yaw 차이 0, 기존 고정 기준 후보의 합성 좌우 코너 조향 차이 보존을 동일 가중치로 사용했다. 목표는 공학 가설/기존 모델 증류이지 실제 라인이나 생리 측정치가 아니다. 손실은 9.657e-6→9.557e-6, 가중치 norm 0.528, 상한 1.0은 미도달이었다.

미학습 새 질감 881613/881614에서 기존 고정 기준의 코너 차이는 -0.006697/-0.006975, 새 경계 초기 0은 -0.000024/-0.000022, 새 경계 학습 후 -0.000060/-0.000059였다. 전체 yaw 차이는 기존 -0.000652/-0.004962, 새 학습 +0.003985/+0.003793였지만 공간을 끈 상태도 +0.003718/+0.003555로, 대부분 기존 운동 경로에 남았다. 코너 반응을 회복하지 못해 후보를 기각·배우 미적용이다.

기존 fit 금지 실제 이동 캐시 03/04/06/07의 pooled scene에서 오프라인으로 새 경계의 `|preactivation|` 95백분위는 0.162/0.268/0.226/0.283, `|pre|>3`은 0%였다. 실제 네 시야를 유효하다고 가정한 수치 범위 계산이므로 시간·시야 신선도나 주행 성능을 뜻하지 않는다. 실제 영상으로 학습하지 않았다. 정적 코너가 이 양측 대칭 특징에서 작아지는지 또는 비행 회로에 전달되지 않는지를 다음에 분리해야 한다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-spatial-pilot-plan/plan.json`; 학습·후보·미학습 평가 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-spatial-training-1/result.json`; 실제 cache 범위 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-spatial-real-range-1/result.json`.


## 2026-11-01 — 양측 시각 특징과 포트의 합성 코너 정보

앞서 훈련한 양측 공간 경계의 yaw 0 마지막 10프레임을 사용해, 학습 질감 881301/881302/881511에서 두 개의 사후 ridge 진단기를 따로 맞췄다. 하나는 정규화된 64개 양측 특징, 다른 하나는 학습된 42개 bounded 공간 포트 값을 사용한다. 학습 분포에서만 평균·표준편차(하한 1e-6)를 정했고 ridge 1을 적용했다. 라벨은 합성 `bend -1/0/+1`이며 진단기는 운전 배우나 새로운 회로에 연결하지 않았다.

미학습 질감 881613/881614에서 반대 코너 쌍의 점수 순서가 기대 순서인 프레임 비율은 64특징에서 yaw 0/-0.12/+0.12 모두 1.0/1.0/1.0, 42포트에서 각각 1.0/0.9/0.8 및 1.0/0.9/0.9였다. 프레임은 연속으로 상관되므로 독립 표본 정확도가 아니다. 무회전 포트의 양 코너 score 평균 차이는 0.671/0.678이었고 양측 특징에서는 1.445/1.467이었다. 단위가 다른 특징 RMS를 성능이나 전달 효율로 비교하지 않는다.

동시에 양측 특징 진단기의 ±yaw score 차이 RMS는 5.406/5.963, 무회전 코너 점수 차이는 1.445/1.467로 회전 교란이 크다(포트 진단기는 각각 1.454/1.844 대 0.671/0.678). 동일 시각의 좌우 코너 순위가 유지돼도 yaw 조건 간 절대 score가 안정됐다는 뜻이 아니다. 이 두 질감은 이전 후보 평가에서 이미 보았고 같은 합성 도로 생성기를 사용했다. 작은 공간 조향 출력과 함께 보면 코너 단서가 특징·포트에 전혀 없는 것은 아니지만, 특정 이후 뉴런의 손실을 이것만으로 확정할 수 없다. 독립 진단기는 회로의 행동이나 생리 기능을 입증하지 않는다. 실제 영상 fitting·배치·차량 제어는 0이다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-information-probe-plan/plan.json`; 진단 결과·가중치·점수 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-information-probe-1/`.


## 2026-11-01 — 양측 시각 투영의 코너 gradient 방향 대조

훈련된 양측 시각 후보와 기존 단일 비행 그래프를 고정한 채, 합성 학습 질감 881301/881302/881511의 무회전 `좌−우 코너` 조향 마지막 10프레임 평균에 대한 공간 weight gradient를 각각 계산했다. 세 gradient 평균 norm은 6.099e-5이며, 이를 정규화해 조향 차이를 음수로 만드는 단일 방향을 고정했다. 음수 목표는 기존 모델 반응에 맞춘 공학 가설로 최적 주행·초파리 생리의 정답이 아니다. 평가 질감은 방향 계산에 쓰지 않았다.

| 새 질감 | 개입 L2 크기 0 / 1 / 5 / 20의 코너 조향 차이 | 같은 순서의 yaw 조향 차이 | 기존 고정 기준 코너 차이 |
| --- | --- | --- | ---: |
| 881615 | -0.000065 / -0.000141 / -0.000446 / -0.001584 | +0.003850 / +0.004257 / +0.005876 / +0.011600 | -0.007087 |
| 881616 | -0.000070 / -0.000142 / -0.000439 / -0.001447 | +0.003934 / +0.004284 / +0.005674 / +0.010448 | -0.007117 |

개입 전 weight norm은 0.528, L2 20 개입 후 약 20.49였다. 큰 변화에서도 코너 차이는 기존 참조의 약 20~22%이고 yaw 반응은 함께 커졌다. 하나의 국소 gradient 방향에 대한 유한 변화이므로 이 경계나 모든 가능한 학습이 불가능하다는 증명은 아니다.

같은 합성 gradient 방향을 학습 부적격인 기존 실제 AI 주행 cache 03/04/06/07의 pooled scene에 사후 적용하면 L2 20에서 공간 투영 `|pre|>3` 포트 비율은 각각 32.6/37.8/30.3/24.2%였다. L2 5에서는 0/2.1/0.2/2.8%였다. 모든 시야 유효를 가정한 오프라인 범위 계산으로 실제 신선도·주행 성능이나 생리 기능의 증거는 아니다. 실제 영상은 방향 계산에 사용하지 않았다. 개입은 배우 미적용이다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-gradient-direction-plan/plan.json`; 합성 gradient·유한 대조 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-gradient-direction-1/`; 실제 입력 범위 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-gradient-real-range-1/result.json`.


## 2026-11-01 — 양측 공간 경계의 비행 상태 단계별 개입

기존에 평가한 합성 질감 881615/881616에서 새 양측 경계 weight 0, 학습 weight, 사전에 계산한 코너 gradient 방향 L2 5/20을 각각 적용하고 동일 회로를 장면마다 초기화했다. 몸 yaw 0, 무회전 좌·우 코너의 마지막 10프레임을 비교했다. 고정 운동 시각·몸·비행 그래프와 각 그룹의 노드 순서를 유지했다. 각 수치는 같은 시점의 `(좌−우 상태) − (공간 0에서 좌−우 상태)` RMS다.

| 개입 / 질감 | 시각 입력 뉴런 | HS | VS | DNg02 | 날개 출력 후보 | 근육 6 | 조향 좌−우 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 학습 / 881615 | 6.61e-5 | 5.35e-5 | 6.87e-5 | 1.8e-8 | 5.8e-8 | 2.4e-8 | -0.000065 |
| 학습 / 881616 | 6.39e-5 | 5.19e-5 | 6.64e-5 | 2.0e-8 | 5.6e-8 | 2.4e-8 | -0.000070 |
| L2 20 / 881615 | 2.586e-3 | 1.583e-3 | 2.770e-3 | 5.98e-7 | 2.274e-6 | 1.005e-6 | -0.001584 |
| L2 20 / 881616 | 2.339e-3 | 1.461e-3 | 2.501e-3 | 5.49e-7 | 2.048e-6 | 8.98e-7 | -0.001447 |

공간 경계를 켜는 개입의 효과가 named 시각 상태에서 날개 후보와 근육까지 수치적으로 나타났으나 코너별 조향은 매우 작다. 그룹별 크기와 연결·기능이 다르고 초기 상태·정규화·고정 다른 경로도 영향을 주므로 위 RMS 비율은 한 연결의 전달률이 아니다. HS/VS/DNg02 이름은 실측 연결 기반 graph의 표기이며 이 rate 근사가 각 뉴런의 생리 기능을 재현했다는 뜻은 아니다. 시간 상관 프레임·이미 본 두 합성 질감·하나의 가중치 방향에 대한 반사실로 실제 운전, 방향 선택성, 랩타임을 증명하지 않는다. 배우 배치·학습·차량 제어 0.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-flight-stage-causality-plan/plan.json`; 각 그룹 상태 배열·원본 해시와 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-flight-stage-causality-1/`.


## 2026-11-01 — 현재 비행 rate 그래프의 고정 국소 활성 경로

앞선 두 합성 질감 881615/881616의 좌·우 코너에서 양측 시각 weight 0 및 학습 weight를 적용해 마지막 결정 시점의 9728개 상태와 그다음에 사용할 `net = M × state + drive`를 저장했다. `M`은 현재 체크포인트의 부호 가설·정규화가 들어간 실제 sparse matrix이며 비영 항목 1,254,062개다. 시각 입력 42개에서 시작해 그 순간 `net>0`인 노드만 통과하는 고정 ReLU 게이트 도달성을 계산하고 게이트 없는 같은 비영 행렬의 구조 도달성과 대조했다. 양측 시각 개입·두 질감·양 코너 모두에서 결과는 다음과 같았다.

- HS 8/8, VS 34/34, DNg02 29/29는 그 시점 국소 활성이고 도달 가능했다.
- DNa04 2개와 DNa05 2개는 구조적으로 도달 가능했으나 `net<=0`으로 고정 국소 게이트 경로에서는 0개였다. 이 이름을 사용자 기준인 비행 운동 회로의 구현이라고 바꾸어 부르지 않는다.
- 날개 출력 후보 67개는 비영 연결만 보면 전부 도달 가능하고, 고정 활성 게이트에서는 56개가 도달 가능했다. 기존 여섯 근육 readout에 포함되는 고유 출력 뉴런 28개는 구조·고정 활성 경로 모두 28개가 도달 가능했고 최단 경로는 22개가 2홉, 6개가 3홉이었다.

따라서 이 단일 시점의 작은 근육 변화는 단순한 비영 연결 단절만으로 설명하기 어렵다. 단, 도달성은 부호 상쇄·가중치·반응 크기나 시각 기능을 보장하지 않는다. 실제 업데이트의 기억 항, 여러 시점의 게이트 변화·비선형 반응을 고정 그래프로 대체할 수 없고, 이는 특정 뉴런의 생리·비행·자율주행 성공을 입증하지 않는다. 합성 두 질감과 한 상태의 국소 분석이며 actor·차량 제어 0이다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-local-gated-path-plan/plan.json`; 상태·net·거리 배열과 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-local-gated-path-1/`.

## 2026-11-01 — 고정 입력에서 시각 포트별 유한 개입

직전 실험의 해시 검증된 마지막 상태 8개(합성 질감 2개 × 좌우 코너 × 공간 경계 0/학습 후보)를 사용했다. 각 상태에서 42개 시각 입력 포트에 각각 ±0.001과 ±0.01의 일정한 drive 섭동을 주고 나머지 drive를 고정했다. 동일한 상태로 시작한 양쪽 회로를 실제 비선형 rate 갱신식으로 10단계 진행하며 6개 근육과 2개 명령의 중앙차분을 계산했다. 각 단계는 0.01초이며 ReLU 활성 상태도 함께 변한다. 원 입력은 저장된 net에서 현재 행렬과 상태를 빼서 복원했으므로 부동소수점 반올림이 포함된다.

1~2단계의 출력 차이는 float32에서 0이었다. 3단계 근육 민감도 RMS는 ±0.001에서 1.16~3.50e-8, ±0.01에서 1.61~1.67e-8로 정밀도 영향이 크다. 10단계에서는 각각 1.334~1.344e-6, 1.334~1.336e-6이었다. 명령 민감도 RMS는 10단계에 각각 5.33~5.63e-4, 5.42~5.45e-4였다. 근육과 명령의 스케일은 다르며 직접 전달률로 비교하지 않는다. 이 값은 입력 단위당 차분이지 운전 성능이 아니다.

이는 단일 포트를 고정적으로 섭동하는 인공 대조다. 실제 영상의 시간 변화·입력 간 상쇄·몸 되먹임 폐루프·생리 기능·AC 주행은 검증하지 않는다. 학습이나 차량 제어는 없었다. 첫 시도는 저장 모드 이름 불일치로 실패했고 기록을 보존했다. 성공 결과: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flight-finite-intervention-2/result.json`; 실패: `.../20261101-flight-finite-intervention-1/result.json`; 코드: `scripts/audit_flight_finite_interventions.py`. 다음에는 실제 시간 변화가 있는 입력으로 개입을 반복하고, 후보의 시각 선행 예측과 몸 감각 통합을 폐루프 및 실제 AC에서 검증해야 한다.

## 2026-11-01 — 시간 변화 시각 yaw와 몸 yaw의 요인별 대조

기존에 확인한 합성 자료 881615/881616의 좌우 코너, 공간 경계 끔/학습 후보를 각각 사용했다. 장면별 네 시야 시계열 41프레임에 시각 yaw 0 또는 +0.12 rad/s, 몸 yaw 0 또는 +0.12 rad/s를 독립적으로 조합하여 매번 회로 상태를 에피소드 경계에서만 초기화했다. 몸 속도는 0.4초 이후 3 m/s이며 yaw 몸 감각도 그때부터 적용했다. 시각 회전과 몸 회전이 일치하지 않는 조합은 인과 분리용 반사실이다. 정답 지도는 배우에 넣지 않았다. 마지막 10프레임의 동일 조건 평균을 비교했다.

8개 조건에서 조향 intent의 시각 단독 효과는 -0.001811~-0.002028, 몸 단독 효과는 -0.011902~-0.011904, 두 효과의 합에서 벗어난 상호작용은 +0.0000043~+0.0000071, 두 입력을 함께 적용한 차이는 -0.013708~-0.013925였다. 공간 경계 후보의 효과는 이 자극에서 시각 회전 반응을 소폭 바꿨다. 음수 부호 자체는 좋은 제어 방향의 증거가 아니다. 서로 같은 크기와 단위의 yaw 조건만 비교했으며 다른 속도, 음의 yaw, 회전 차량 궤적 및 질감 일반화는 이 검사에 없다. 개루프 합성 자극이므로 시각 선행 예측, 폐루프 효과, 생리 기능이나 실제 주행을 증명하지 않는다.

결과와 41프레임별 intent/근육값: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-dynamic-visual-body-factorial-1/result.json` 및 `factorial-traces.npz`; 코드 `scripts/audit_dynamic_visual_body_factorial.py`. 결과의 `interaction_rms` 필드는 코드상 실제로 시간 표준편차이며 RMS가 아니다. 상호작용 평균은 `interaction`에 별도 기록돼 위 결과에 사용했다. manifest의 상속된 paired-bend plan 해시는 공통 초기화 출처이며 본 요인 실험의 사전 등록 계획이라는 뜻이 아니다. 최적화와 차량 제어는 없었다. 다음에는 부호·속도·장면을 확장하고 실제 카메라/몸 감각 동기 및 폐루프의 인과 기여를 검증해야 한다.

## 2026-11-01 — 정지 shadow에서 held body 신선도 실패의 원본 수신 감사

`20261101-stationary-unified-file-shadow-warm`의 result/events/native 원본 해시, 각 원본 packet prefix 해시 및 수신 순서를 검증했다. 첫 준비 결정에서 둘째 거부 결정까지 50.2016ms 동안 native copy 20개와 고유 packet ID 17개가 수신됐고 연속 copy 간 최대 간격은 2.774ms였다. 둘째 결정에서 직전 held body는 61.862ms, 새 body는 2.626ms 전 수신이었다. 즉 이 한 실패에서 새 패킷 자체는 있었으나, 영상마다 한 번만 회로를 갱신하는 동기식 경로가 이전 입력을 50ms 이상 유지했다.

별도 body ticker가 필요한 근거이지만, 수신 시간은 실제 물리 샘플 시각이 아니고 이 감사는 ticker나 영상·body 동시 계산을 실행하지 않았다. 기존 ticker의 10ms 실시간 시험도 stale가 있었으므로 성공을 보장하지 않는다. 정정: 현재 통합 모델의 독립 ticker 연결은 이미 구현됐고 아래의 선행 정지 시험 두 번이 완료됐다. 아래 영상 순서 보호 변경 후 새 코드의 동시 실행과 주행 조건 검증은 필요하다. 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-held-body-interval-audit-1/result.json`, 코드 `scripts/audit_stationary_held_body_interval.py`. 차량 제어·학습 0.

## 2026-11-01 — 기존 통합 ticker 정정 및 capture 순서 보호

직전 held body 감사에서 독립 ticker를 앞으로 연결해야 한다고 기록한 것은 기존 구현 및 성공 기록을 누락한 오류다. `UnifiedLiveBodyTicker`와 `scripts/record_stationary_unified_multirate_shadow.py --held-prefix`는 이미 단일 통합 회로를 사용한다. 정지 실시간 시험 `20261101-stationary-unified-held-prefix`와 `...-repeat`는 각각 243/247개 shadow 출력을 생성하고 stale 거부 없이 종료했다. 두 결과 모두 원본 독립 감사가 있으나 정지 상태이며 실제 주행, 물리 노출/센서 동기, 생리 및 actor 자격을 증명하지 않는다. 앞서 분석한 동기식 실패와는 다른 경로다.

기존 scene mailbox는 미소비 장면의 덮어쓰기는 거부했지만 소비 후 동일 또는 더 작은 capture ID의 재게시는 허용했다. 동시 처리 중 오래된 장면을 회로에 재투입하지 않도록 ID가 엄격하게 증가해야 하며 위반 시 이전 mailbox를 유지하도록 변경했다. 관련 mailbox·ticker·guard 테스트 11개 통과. 코드 `src/flybrain_drive/unified_live_ticker.py`, 검사 `tests/test_unified_live_ticker.py`. 기존 실행은 변경 전 source hash를 가진다. 변경 후 카메라/FlyVis/native 동시 실시간 재검증, 장면 누락 및 source 시각 확인, 이동 조건에서의 검증은 아직 없다. 학습·차량 제어 0.

## 2026-11-01 — 순서 보호 후 정지 shadow와 검정 카메라 입력

변경된 ticker의 실시간 정지 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-unified-ordered-scene-1/`은 camera 80묶음 completed, 그중 72묶음 선택·게시, priming 1·shadow 출력 296·거부 0, 프로세스 종료 0이다. 원본 해시와 receipt·render bound·게시 독립 감사 `independent-audit-v2.json`에서 출력 기록 시 최대 view 129.59ms, 이전/current body 27.74/13.41ms였다. 해당 감사는 픽셀 해시와 시간 일치만 검사하며 장면의 시각적 유효성은 인증하지 않는다.

이후 원본 ZIP 전체의 RGB를 확인하니 좌·우·후방 각각 80/80장이 완전 검정이다. 전방은 80장 모두 비영 픽셀이 있으나 180×320×3 중 1892~3027 채널만 비영으로 대부분 검정이다. 원인은 미확정이며 현 결과를 정상 영상·운전 또는 생리 검증으로 사용하지 않는다. 세션과 카메라 terminal은 정상, UI/lifecycle/제어/학습 0이다.

명백한 결측 시야가 다시 유효로 처리되지 않도록 `require_nonzero_camera_views`를 추가해 이 정지 multirate 실행기의 원본 four-view freeze 직후 전부 0인 RGB 시야를 거부한다. 이 보수적 차단은 정상적인 완전 암흑 장면도 거부할 수 있고, 비영 영상의 정상성이나 노출 동기를 증명하지 못하며 다른 실행기에 자동 적용되지 않는다. 관련 immutable camera·ticker·multirate 테스트 11개와 문법 검사를 통과했다. 이 검사는 위 실행 이후 추가됐으며 적용 후 실제 카메라 시험은 아직 없다. 다음은 UI 입력 없이 CSP/카메라 상태와 렌더링 원인을 확인하고 실제 네 시야 유효성을 입증하는 것이다.

## 2026-11-01 — 실시간 검정 시야 차단과 원인 조사

읽기 전용 Win32 상태에서 AC 창은 존재·visible·비최소화·비foreground였다. 직전 정상 보관과 검정 보관의 camera layout revision, native dashboard 설정 및 projection은 같고 원래 world 위치·방향은 가깝지만 정확히 같지는 않았다. 이것만으로 렌더링 원인을 확정할 수 없다.

신규 bounded stationary shadow `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-stationary-blank-rejection-check-1/`은 처음 선택된 시야에서 `Entirely black camera view: left`로 중단, shadow 출력 0·차량 제어 0·종료 코드 1이다. 6묶음의 원본 archive hash 검증과 camera terminal stopped·worker 종료를 기록했다. 기존 실패와 원본을 보존하며 자동 재시도하지 않았다. 이 실행에서는 검사 예외가 events 생성 전 발생해 events가 빈 배열이다. 이후 실행기에서 capture ID와 오류를 `invalid_camera_view` 이벤트로 남기도록 보완했으나 이 보완은 아직 실시간 재검증 전이다.

관련 테스트 중 합성 ticker 테스트가 내부 호출 기록과 이벤트 기록 사이의 thread 경합으로 한 차례 실패해, 실제 검증 대상인 두 번째 이벤트까지 대기하도록 수정했다. camera/ticker/multirate/guard 15검사 통과. 검정은 야간·렌더링·후처리·캡처 등 원인 불명이며 임의로 정상 처리하지 않는다. 다음은 UI 조작 없이 카메라 렌더와 환경 설정·producer 상태를 독립 확인한다. 학습·actor 제어·물리 노출 검증 0.

## 2026-11-01 — 동일 세션 보관 영상 밝기 비교

`generation 19007968ULL:0.003981`인 보관 camera ZIP 13개의 원본 ZIP 해시를 검증하고 각 첫 묶음 네 시야의 RGB 평균과 비영 채널 수를 기록했다. 왼쪽 영상은 첫 11개에서 RGB 평균 145.29~183.32였고 마지막 2개는 0이었다. 마지막 밝은 표본의 CSP script clock 19456.294와 첫 검정 표본 42153.408 사이 약 22697초(6시간 18분)의 관측 공백이 있다. 이전/최근에는 위치와 영상 조건도 달라 시각 경과와 어두워짐의 인과는 증명되지 않는다. 추세를 보간하거나 다른 archive·이후 프레임까지 포괄한다고 주장하지 않는다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-same-generation-brightness-audit-1/result.json`, 코드 `scripts/audit_same_generation_camera_brightness.py`. 첫 실행은 관련 없는 큰 archive 해시까지 계산하여 사용자가 기다리는 시간을 줄이기 위해 중단했고, generation 확인 후 해당 archive만 해시하도록 수정해 완료했다.

현재 읽은 사용자 race.ini에는 `SUN_ANGLE=-48`, `TIME_MULT=1`, weather `4_mid_clear`가 있으나 이 파일이 현재 동작 중인 엔진 설정과 계속 일치한다는 근거는 없다. 로컬 CSP SDK에는 `ac.getSunAngle()`이 있으나 이번에는 실행하지 않아 현재 태양 각도도 모른다. 야간 가능성은 가설일 뿐이다. 운영 Lua를 임의로 덮거나 UI 조작·actor 제어를 하지 않았다. 다음은 기존 producer의 무결성과 세션 소유를 지키는 읽기 전용 태양/환경 진단 또는 안전한 lifecycle 기반 환경 복원 검증이다.

## 2026-11-01 — API 세션 재시작 후 시야와 freshness 대조

카메라 terminal, 정차·브레이크·AI off 및 vJoy free 등을 기존 lifecycle 도구가 확인한 뒤 `20261101-camera-dark-session-restart-1`에서 CSP restart, `20261101-camera-dark-session-start-1`에서 start를 각 1회 실행했다. UI/actuator 0, 동일 generation epoch 1이다. 실제 태양 각도는 읽지 않았으므로 검정 원인이 밤이었다거나 재시작의 어떤 효과가 복구를 일으켰다고 확정하지 않는다.

`20261101-post-restart-vision-shadow-1`의 원본 첫 4시야 RGB 평균은 전/좌/우/후 159.23/163.31/201.95/171.30으로 모두 비영이며 카메라 8묶음을 보존했다. 시야 내용·물리 노출 동기·생리는 별도로 검증하지 않았다. 첫 시간 안전 조건을 통과한 shadow 출력 7회 뒤 ticker는 이전 capture 5의 첫 시야 render 하한 48504.0560905에서 decision 48504.2076145까지 151.524ms가 경과해 150ms 제한으로 중단했다. capture 6은 이후 ticker 종료로 게시 거부됐다. camera stopped terminal·원본 hash/archive 검증, 차량 제어/학습 0. 이 실행을 성공으로 표기하지 않는다. 관측 결과 파일 작성 뒤 Python 핸들 17214는 마지막 확인 시 아직 살아 있어 결과 출력만으로 프로세스 종료를 단정하거나 같은 요청을 재시작하지 않는다.

다음은 원본 카메라 그룹의 수신·선택·계산·게시 사이 지연과 다음 프레임 간격을 분리하고, 원본 영상·시각 및 150ms 기준을 보존하면서 개선한다.

## 2026-11-01 — 짧은 held visual prefix 대조

직전 `20261101-post-restart-vision-shadow-1`의 세 선택 영상에서 첫 시야 render 하한→camera receipt 42.2/48.0/45.7ms, receipt→visual snapshot 42.3/51.4/27.2ms, visual 처리 7.0/12.3/10.7ms였다. 앞선 두 게시 뒤 16단계 prefix는 각각 35.0/33.7ms였다. 시간 구간의 중첩·인과를 이 소수 기록만으로 확정하지 않는다. 원래 150ms 제한과 원본 시각은 바꾸지 않는다.

정지 multirate runner에 선택형 `--prefix-steps 8`을 추가했다. 기본 16은 유지하고, 8×5ms 이후의 실제 경과분은 기존 held input simulate로 이어간다. 이전 카메라 실제 네 시야 한 묶음·단일 초기 상태와 7개 경과 2~150ms에서 새 8단계 경로와 기존 계산의 모든 비교 대상 동적 상태 최대차는 9.54e-7이다. 독립 proof `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-held-prefix8-parity/result.json`, 코드 `scripts/verify_flyvis_held_visual_prefix_8.py`; runner는 각 설정에 맞는 source hash와 proof를 요구한다. 8/16 경계의 합성 테스트 포함 24개 통과. 단일 입력 오프라인 동등성이지 장기 이력·생리·실제 속도 향상 증거가 아니다.

직전 camera terminal은 stopped, run 결과·archive는 보존됐으나 Python 부모 PID14920·자식46632 및 tool handle 17214는 마지막 확인 시 계속 살아 있다. 단지 관측 시간이 만료됐다는 이유로 종료나 재시작하지 않았다. 8단계 실시간 동시 시험은 아직 없다. 다음은 소유 프로세스 종료 상태를 확인하고 별도 실행에서 수신→게시 지연, 처리 누락, freshness와 원본을 재검증한다. 학습·차량 제어 0.

## 2026-11-01 — 8단계 prefix 실시간 정지 시험

직전 16단계 run은 `stopped_on_error` result·camera stopped 영수증이 있고 Python 자식 PID46632가 종료된 것을 확인했다. 상위 래퍼 PID14920은 마지막 관측에서 생존했으므로 그 사실을 분리해 기록한다. 새 실험은 다른 설정과 고유 경로 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-prefix8-stationary-shadow-1/`를 사용했다. camera 13묶음 stopped, 전체 4×13 RGB 중 완전 검정 0, 8묶음 게시·priming1·shadow 출력31·거부1이다. 마지막 거부는 capture10의 첫 시야 하한→출력 완료 154.878ms로 기존 150ms 제한을 초과했다. 학습·UI·lifecycle·차량 제어 0.

원본 ZIP·픽셀·시각 경계·native 최초 receipt 및 게시 독립 감사 `independent-audit-v2.json`은 정상 출력에서 시야 최대143.73ms, 이전/current body 최대22.31/10.47ms, 시각 계산 중앙/최대14.23/24.02ms, 8단계 prefix 중앙/최대15.81/19.16ms를 기록했다. 이 실행에서는 전방 등 비영 확인 외 영상의 의미·노출/센서 물리 동기를 검증하지 않았다. 이전 16단계 실행의 선계산 시간 약34ms·7출력과 달리 시점·입력·선택 묶음·부하가 다르므로 인과적 개선이나 일반화는 입증되지 않는다. 새 안전 차단을 누락하거나 150ms 한계를 완화하지 않는다. 다음은 원본 반복 입력과 가능한 같은 부하에서 prefix 길이의 처리 지연 및 검증 가능한 scene 게시 경로를 비교하고 폐루프 요건을 확인한다.

## 2026-11-01 — 기한 임박 영상의 보류와 실시간 대조

이전 8단계 시험에서 capture10을 사용한 최종 ticker decision 48721.8591628은 capture11 게시 48721.859738보다 약0.575ms 빨랐고 출력 완료 시 오래된 capture10 시야 age 154.878ms였다. 시야의 기한이 지나기 직전 작업을 억제하기 위해 소유 ticker가 이미 소비한 화면의 가장 이른 render 하한 나이 140ms 이상이면 새 화면이 올 때까지 출력을 보류한다. 나이 150ms 초과는 실패로 중단하며 새 장면·몸 감각은 기존 원본과 시간 제한으로 guard 검증한다. 10ms는 출력 소요를 보장하는 기한이 아니라 보수적 예약값이며 짧은 지연에도 최종 guard는 유지한다. 보류가 길면 이전 몸의 50ms 제한도 여전히 적용된다.

`20261101-prefix8-freshness-defer-shadow-1` 실제 정지 실행에서 게시3·priming1·정상 출력12, 새 화면을 기다리다 150ms 초과로 ticker가 중단됐다. 원본 camera9묶음, terminal stopped·archive 및 native hash 보존, UI/lifecycle/차량 제어·학습0. 독립 원본 감사 `independent-audit-v2.json`은 정상 출력 시 최대 시야 age140.15ms, 이전/current body21.40/10.66ms였다. 감사의 rejected0은 ticker update 안에서 거부된 이벤트가 0이라는 뜻이며, update 바깥 대기 중 실제 ticker 오류는 result의 `ticker_error`에 남아 있다. guard의 stale 검사를 없애지 않았고 이 시험은 실패다. 최신 장면이 항상 제때 도착하지 않는 문제, 실제 움직임과 생리·노출 동기는 미해결이다. 관련 ticker·guard·prefix 검사25개 통과. 출력 파일 저장 뒤 핸들85044는 마지막 확인 시 살아 있다.

## 2026-11-01 — 통합 회로 MMF 시작 실패

파일 PNG 경로의 지연 대조를 위해 `scripts/record_stationary_unified_mmf_shadow.py`에서 기존 batched shared texture·immutable async persistence·MMF metadata/source timing을 통합 ticker에 연결하는 별도 시도를 했다. 이전 file 요청은 terminal stopped, 해당 Python 자식 종료를 확인했다. 그러나 `20261101-unified-mmf-stationary-shadow-1` 요청 `mmf-stream-d487677e57a14987aea819bb4f516523`은 unpaused에서 발행돼 CSP producer가 `MMF requires eight paused probe frames or bounded v3 stream starting paused`로 거부했다. producer status는 이전 request ID에 실패 사유를 표시하며 이번 요청의 terminal result는 확인되지 않았다. 최초 수신0·회로 출력0·UI/lifecycle/actuator/optimizer0. consumer close는 5초 이내 종료되지 않아 archive 불가로 기록됐고, 이후 Python tool handle 25666은 종료0이다. 이것이 producer 요청의 terminal을 뜻하지는 않는다. 기존 request/stop·status·실패 result를 보존하고 재요청하지 않는다.

사전 조건을 파악하지 못하고 시도한 오류다. 동일 실수를 막기 위해 해당 실험 실행기에 paused-start/resume 인계 미구현 상태에서는 camera 요청 전에 명시적으로 실패하는 차단을 추가했다. 기존 file runner를 변경하거나 안전 조건을 완화하지 않았다. 다음은 동일 요청의 명확한 terminal 상태와 producer 소유를 확인한 뒤 별도 검증된 paused-start·resume 및 카메라/몸 freshness 인계 계약을 구현하는 것이다. MMF 도입이나 지연 개선이 성공했다고 주장하지 않는다.

## 2026-11-01 — paused MMF 시작 인계의 순수 계약

이전 요청 `mmf-stream-d487677e57a14987aea819bb4f516523`의 자기 ID terminal은 재확인 시에도 없고 status는 이전 async ID 실패, 해당 폴더에는 stop만 있었다. 다른 요청이나 임의 재시작은 하지 않았다.

`src/flybrain_drive/paused_mmf_startup.py`는 이전 요청이 있으면 같은 ID의 terminal을 요구하고, 기존 paused·정차·브레이크·AI off admission 이후 동일 generation/epoch에서 일시정지 중 자기 MMF ID가 capturing인 것을 확인한다. 이후에만 같은 소유 카메라가 유지되는 live resume과 증가한 두 native packet을 허용한다. 선행 순서·세대/카메라 불일치·정지 패킷·미확인 terminal을 거부하는 순수 검사이며 API 호출·시간 신선도·실제 패킷 출처 검증은 별도 계층의 책임이다. 관련 단위12검사 통과. 실제 runner는 여전히 MMF 시작이 차단돼 있고 새 gate 연결·실제 paused→resume lifecycle·물리 노출 동기 및 주행은 미검증이다. 해당 과정에서 UI/lifecycle/카메라/차량 제어·학습0.

## 2026-11-01 — 미접수 stop 요청 자동 재검사 결함과 미배포 후보

운영 producer 원본에서 MMF paused 검사 실패는 `lastID=request.id`보다 앞서 발생하고 외부 pcall은 active가 없으면 camera terminal을 쓰지 않는다. request.json이 남아 있으므로 producer는 대략 250ms마다 동일 ID를 다시 begin에 넘긴다. 폴더의 stop 표식은 active가 생긴 뒤 처리하므로 현재 unpaused에서 거부된 요청을 둔 채 pause하면 재접수될 가능성이 있다. 관측된 현황 자체는 재확인한 다른 ID status failed·자기 terminal 없음이며 재접수 발생을 주장하지 않는다. 안전하게 해결하기 전 새 요청·pause를 자동 수행하지 않는다.

`integrations/ac_csp/candidates/FlyBrainCamera-pre-admission-cancel-v1.lua`는 같은 generation의 안전한 ID에 stop이 이미 있고 terminal이 없는 경우 begin 전에 0프레임·camera_started=false stopped terminal을 먼저 기록한 다음 ID를 소비한다. terminal write 실패 시 ID를 소비하지 않으며 기존 result를 덮어쓰지 않는다. `scripts/build_camera_pre_admission_cancel_candidate.py`로 정확한 운영 원본 한 구간만 치환한다. LuaJIT mock의 stop 처리·기록 실패 재시도·다른 generation·기존 terminal 보존과 기존 lifecycle 회귀를 포함한 237검사 통과. `lupa 2.8`을 기존 flyvis 가상환경에 설치해 실제 Lua mock 검사를 실행했다.

후보는 운영 producer에 배포하지 않았다. 따라서 미확정 현재 요청은 해결되지 않았고 후보 동작은 실제 CSP에서 검증되지 않았다. 실제 배포/복구에는 live producer 교체·재로드, 세대 및 무결성·단일 소유·재시작 후 원본 receipt 보존을 명시적으로 검증해야 한다. 이번 작업의 UI/lifecycle/카메라 요청·배우 제어·학습0.

## 2026-11-01 — 8단계 prefix의 원본 선택 이력 누적 동등성

기존 `20261101-stationary-unified-held-prefix` 원본 archive와 이벤트 해시를 확인하고, 당시 선택된 RGB 55묶음·원래 visual snapshot 54간격을 같은 순서로 두 독립 FlyVis 계산 경로에 넣었다. 8단계 prefix와 기존 held-input 계산의 누적 전체 nodes.activity 최대 차 9.536743e-7, T4/T5 motion 최대 차 4.186295e-7이다. episode 시작에서만 초기화했다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-held-prefix8-selected-history-parity/result.json`, `scripts/verify_flyvis_prefix8_selected_history.py`. 원래 선택되지 않은 25묶음은 포함하지 않았고 두 경로 모두 오프라인이므로 실제 live 출력 재현·움직임·물리 노출·생리·주행의 증거는 아니다. 미확인 MMF 요청은 건드리지 않았고 camera/lifecycle/actuator/optimizer0이다.

## 2026-11-01 — 실제 이동 이력 경로 제거의 평균과 시간 변화

기존 `20261101-unified-moving-teacher-path-ablations`의 fit 금지 AI 교사 이력 365결정·네 경로 제거 결과와 NPZ 해시를 확인했다. 기존 조향 최대/절대평균만 보면 공간 시각의 큰 효과가 차동 반응인지 고정 편향인지 구별되지 않는다. 사후 분석으로 전체와 앞 50결정 제외 315결정에서 제거 후 minus 원본 조향 차의 평균·시간 표준편차를 별도 계산했다. 앞 50 제외는 이 기록을 관찰한 뒤 선택한 탐색적 기준이며 엄밀한 적응 완료 증거가 아니다.

제외 후 공간 대비 제거의 평균 -0.01277476, 표준편차 0.00006405, 범위 0.00024148이다. 운동 시각 opponent 제거는 평균 -0.00025959, 표준편차 0.00128793, 범위 0.00456234다. 시각×몸 경로 제거는 평균 +0.00001614, 표준편차 0.00001714이고 추가 몸 경로 제거는 평균 -0.00114901, 표준편차 0.00048687이다. 해당 단일 이력에서 공간 경로 제거의 큰 조향 효과가 거의 일정한 치우침이고 운동 경로는 상대적으로 시간 변동이 크다는 관찰이다. 이 수치만으로 양 경로가 실제 코너·yaw를 올바르게 부호화한다거나 생리적으로 타당하고 운전을 개선한다고 판단하지 않는다. 동일 이력 프레임은 독립 표본이 아니고 다른 트랙·차량·조건 일반화도 미검증이다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-moving-ablation-temporal-decomposition-1/result.json`, 코드 `scripts/audit_moving_ablation_temporal_decomposition.py`. 원자료 변경·학습·제어 0. 다음은 서로 다른 독립 기록과 원본 시각 유효성·운동 자극 방향을 고정해 시간 변화가 실제 시각 운동과 일치하는지 검증하고, 공간 경계의 분포 의존 편향을 개선한다.

## 2026-11-01 — 기존 실제 이동 이력에서 양측 공간 경계 대조

학습 금지된 이전 AI 교사 이력 365결정과 원래 camera/FlyVis reconstruction·body receipt를 사용해 단일 비행 회로를 새로 재생했다. 무개입 baseline은 이전 전체 상태/intent와 차이0이다. 합성으로 학습한 비배치 양측 공간 경계만 기존 공간 경계 대신 적용하고 현재 재생의 운동·몸 경로를 고정했다. 첫 시도 `20261101-bilateral-moving-teacher-replay-1`은 양측 후보 학습 당시 opponent weight와 현재 이력의 weight가 달라 차단됐다. 둘째 `...-2`는 운동 경로 변경을 피하기 위해 현재 opponent를 그대로 유지했으며 `opponent_matches_bilateral_training=false`로 기록했다. 이 구성은 후보를 원래 학습 구성 그대로 평가한 것이 아니다.

사후적으로 첫50결정 제외 후, 같은 공간 제거 대조에 대한 기존 공간 경계의 조향 기여 평균 +0.01277476·시간 표준편차 0.00006405, 양측 후보는 +0.00278571·표준편차 0.00018383이다. 단일 이력에서 평균 편향 크기는 작아졌지만, 이 차이가 실제 조향 정확도를 향상한다는 증거는 없다. 후보는 앞서 합성 좌우 코너 반응 회복에 실패했고 이전에 본 실제 데이터로 선택한 탐색 비교이며 사용자 목표의 생리·실제 폐루프·빠른 완주·랩타임은 미검증이다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-moving-teacher-replay-2/result.json` 및 `ablations.npz`, 코드 `scripts/replay_bilateral_moving_teacher.py`; optimizer·차량 제어 0, candidate actor 미적용.

## 2026-11-01 — 합성 시각 포트 모방 학습과 실제 범위 실패

기존 단일 비행 회로·운동 경로·몸 회로를 고정하고 양측 공간 경계 projection만 학습했다. 사전 기록 `20261101-bilateral-port-distillation-plan/plan.json`: 합성 질감 881301/881302/881511의 yaw·코너·시간 전체에서 기존 고정 기준 공간 경계의 42포트 출력을 모방, MSE + 가중치 L2, Adam 300회, L2 norm 상한20. 이 목표는 기존 공학 경계를 교사로 쓰는 근사이며 초파리 생리 목표 그 자체가 아니다. 실제 데이터 fit0. 미학습 평가 두 질감 881613/881614는 본 fit에는 들어가지 않았으나 과거에 이미 관찰한 자료로 독립 최종 시험이 아니다.

학습 손실 0.007356→0.004956, 가중치 norm 20, 상한 도달 233/300회였다. 무회전 좌우 코너 조향 차는 고정 기준 -0.006697/-0.006975, 새 후보 -0.000808/-0.000812로 약12%에 그쳤다. yaw 반응도 고정 기준과 달라 기존 동작이 충분히 보존되지 않았다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-port-distillation-1/result.json`, 후보 및 계획 해시 포함.

추가로 이전에 살펴본 fit 금지 실제 주행 scene cache 4기록에 사후 적용했을 때 포트 preactivation 절대값>3 비율은 36.1~71.5%였다 (`.../20261101-bilateral-port-distillation-real-range-1/result.json`). 네 시야 유효성을 가정한 오프라인 범위 검사이지 실제 검증은 아니며, 높은 포화는 후보를 거부할 이유다. 현재 배우/실차 제어 미적용, actor_qualified=false. 무제한 gain이나 정답 라인을 배우에 주는 방향이 아니라 합성/실제 분포 차이와 대칭 입력의 정보 손실을 해결해야 한다.

## 2026-11-01 — 양측 공간 특징 선형 진단의 용량과 분포 밖 불안정

직전 합성 포트 모방 학습의 3개 학습 질감·총1107 frame/yaw/bend 입력에서 동일 양측 64특징과 기존 공학 spatial teacher 42포트를 구해 float64 최소제곱 `rcond=1e-8`으로 무제한 선형 진단을 만들었다. 관측 행렬 rank32/64, 계수 Frobenius norm 약9176, 학습 teacher RMS0.08577·잔차 RMS0.01573이다. fit에 넣지 않은 평가 질감(이미 이전에 본 자료) 두 개의 teacher/잔차 RMS는 0.08329/0.01380, 0.09358/0.02562다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-linear-capacity-1/result.json`, coefficient NPZ 및 `scripts/audit_bilateral_linear_capacity.py`. 이는 비선형 bounded 회로의 표현 한계에 대한 수학적 하한이나 배우가 아니다.

같은 진단 계수는 이전에 본 fit 금지 실제 주행 scene cache 4기록의 비제한 선형 예측에서 RMS186~332, ±0.5 초과97.2~97.5%였다 (`.../20261101-bilateral-linear-capacity-real-range-1/result.json`). 실제 bounded 배우 출력은 ±0.5를 넘지 않으므로 이 숫자를 실제 운전 명령이나 포화율로 오인해서는 안 된다. 4시야 유효를 가정한 사후 분포 외 대조이며 실시간 결측 검증은 아니다. 큰 계수와 분포 변화의 연관은 보이나 특정 원인의 인과 증거가 아니다. 실제 fit·배치·차량 제어0. 단순 선형 맞춤이나 무한 gain을 성공으로 처리하지 않고, 망막 특징의 장면 구조·시간 변화 및 독립 조건에서의 안정성을 다뤄야 한다.

## 2026-11-01 — 양측 선형 ridge 설정 범위 대조

합성 학습 3질감으로만 양측 특징→기존 공학 teacher 포트 ridge 해를 λ=1e-6,1e-4,0.01,1,100으로 계산했다. 같은 과거 관찰 평가 2질감과 fit 금지 실제 주행 cache4기록은 사후 점검에만 썼고 실자료로 가중치나 단일 설정을 고르지 않았다. 원본 해시·조건별 계수는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-bilateral-ridge-tradeoff-1/result.json`과 `ridge-coefficients.npz`, 코드 `scripts/audit_bilateral_ridge_tradeoff.py`.

λ=1e-6은 가중치 norm462.43·평가 teacher 포트 RMS 오차0.0135/0.0234이나 실제 cache의 무제한 예측 ±0.5 초과93.3~97.0%다. λ=1은 norm7.36·합성 오차0.0663/0.0747, 실제 초과44.0~68.3%다. λ=100은 norm0.10·실제 초과0이나 합성 오차0.0830/0.0933으로 teacher RMS0.0833/0.0936에 가깝다. 낮은 λ가 더 나은 포트 재현을 보이지만 실제 분포에서 불안정하고 높은 λ는 출력을 거의 지워 이 모방 목표를 만족하지 못한다. 제한 없는 선형 예측이지 실제 tanh bounded 포트나 운전 성능이 아니다. 기존 데이터·교사 분포에서의 사후 진단이며 다른 비선형 구조의 가능성을 배제하지 않는다. optimizer/실제 fit/actor 제어0.

## 2026-11-01 — 직교 방향 원저자 경계의 전체 열과 통합 회로

기존 수평 0°/180° 대조를 확장해 원저자 저장 자극의 밝음/어두움 90°/270°, 속도19를 같은 pretrained FlyVis·5ms로 재실행했다. 485시점·721수용기 위치의 T4/T5 8종 공간 응답을 보존했고 원래 중앙 뉴런 출력과 최대 차는 9.54e-7이었다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-flyvis-author-orthogonal-full-columns/result.json`, `scripts/replay_full_column_orthogonal_edges.py`. 이전 결과와 똑같은 검증 기준과 원본 stimulus를 사용했다.

이 반응을 한 번에 한 시야만 바꿔 고정 몸·명령의 단일 통합 비행 회로에 넣고 원래 40ms 표본 61개를 비교했다. 90° minus270°의 1초 이후 조향 차 평균은 dark/bright 각각 전방 -0.000188/+0.000384, 왼쪽 -0.002962/+0.002161, 오른쪽 +0.000626/-0.000597, 후방 -0.001187/+0.000656으로 네 시야 모두 극성에 따라 부호가 바뀌었다. 결과 `.../20261101-unified-orthogonal-edge-columns-1/result.json`·`outputs.npz`, 코드 `scripts/replay_unified_orthogonal_edge_columns.py`.

이는 밝고 어두운 인공 경계에 대한 한 모델의 관측 차이다. 방향의 물리적 의미나 대조 극성 독립이 정답인지, 자연 장면과 몸 회전의 부호가 맞는지, 생리 기능·차량 주행이 올바른지 입증하지 않는다. 나머지 세 시야는 초기 시각 반응으로 고정했으며 네 방향 동시 실환경 검증도 아니다. 학습·실제 제어0. 이 대조는 실제 차체 pitch/roll 및 수직 운동 시각의 부호·극성 기여를 분리할 다음 실험의 출발점이다.

## 2026-11-01 — 직교 경계의 망막 영상·운동 반응 반사실

기존 90°/270° 밝음·어두움 FlyVis 721열 원본에 대해 동일 극성 두 방향을 비교했다. 세 조건은 원본 full, 같은 극성의 90° 망막 이력을 양 방향에 공통 적용(shared_retina), 같은 90° T4/T5 이력을 공통 적용(shared_motion)이다. 다른 세 시야의 초기 상태도 같은 극성의 90° 기준으로 맞췄다. 나머지 몸·명령·회로를 고정하고 에피소드별로만 reset한다. 공통 적용은 실제 장면에서 동시 발생한 자극이 아니며 선택한 90° 이력에 의존하는 인공 반사실이다.

1초 이후 조향 차(90°−270°) dark/bright 평균: 전방 full -0.000188/+0.000384, shared_retina +0.000308/+0.000037, shared_motion -0.000485/+0.000352. 왼쪽은 full -0.002962/+0.002161, shared_retina -0.001995/+0.001514, shared_motion -0.000977/+0.000626; 오른쪽은 +0.000626/-0.000597, +0.000397/-0.000464, +0.000235/-0.000128; 후방은 -0.001187/+0.000656, -0.000905/+0.000475, -0.000275/+0.000190이다. 두 시각 경로 모두 이 자극에서 출력을 바꾸며, 특히 전방 dark에서는 망막 공유 대조의 부호가 full과 다르다. 비선형 상호작용·기준 선택 때문에 두 반사실의 차를 단순 가산 분해하거나 생리적 방향 판단으로 해석하지 않는다. 다른 극성·각도·자연 장면·차체 pitch/roll에 대한 올바른 부호 및 주행은 미검증이다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-orthogonal-retina-motion-counterfactual-1/result.json` 및 `outputs.npz`; 코드 `scripts/replay_orthogonal_retina_motion_counterfactual.py`. optimizer/actuator0.

## 2026-11-01 — 수직 시각과 합성 몸 롤 자세 요인 대조

기존 90°/270° 어두움·밝음 전체 721열 FlyVis 자극을 한 번에 네 시야 중 하나에 입력하고 다른 시야는 동일 극성의 시작 응답으로 유지했다. 몸 roll 자세 0, +0.05, -0.05 rad는 각 에피소드에서 일정하게 입력했으며 설정 계약의 롤 양수는 왼쪽이 위다. 미검증 롤/피치 각속도와 몸 가속도 mask는 false, yaw 등 나머지 몸 감각은 고정이다. 이 시각 자극과 롤 자세는 실제로 동기화된 물리 운동이 아니라 독립 합성 대조다.

1초 이후 +0.05 minus -0.05 롤 조건의 조향 intent 평균 차는 4시야×4자극에서 +0.0103728~+0.0103775였다. 동일 기간 90°−270° 방향 차가 +롤과 −롤 사이 달라진 양은 -2.85e-6~+2.33e-6이었다. 이 한 자극에서 roll 자세의 영향은 보이나 시각 방향 차에 대한 상호작용은 작다. 이 수치만으로 정상 보상 방향, pitch/roll 동역학, 수용기 생리 또는 실제 주행 안정성을 주장하지 않는다. 기존 roll 각속도·물리 시각 불일치도 해결하지 않았다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-orthogonal-visual-synthetic-roll-1/result.json` 및 `outputs.npz`; 코드 `scripts/replay_orthogonal_visual_synthetic_roll.py`. optimizer·실제 차량 제어0. 다음은 실제 시간 정합·축 검증 이후 시각과 자세가 함께 변하는 조건에서 인과 기여와 방향을 확인한다.

## 2026-11-01 — 합성 네 시야·몸 롤 연동 대조

`scripts/replay_flyvis_synthetic_road_with_roll.py`와 `src/flybrain_drive/synthetic_roll_road_visual.py`는 4개 카메라 위치와 광선을 공통 X-forward 축으로 회전시킨다. 양수는 Y-left가 올라가는 오른손 회전으로 body contract의 roll 양수와 맞춘다. 정지0.4초 뒤 평면 직선 도로를 world-forward 3m/s로 움직이며 시각 롤 속도 0/±0.05rad/s를 넣는다. 첫 0.4초는 동일 영상, 이후 각도는 최대 ±0.06rad다. 이는 바퀴·하중·동역학을 갖춘 주행 모델이 아니라 강제로 지정한 합성 경로다. zero roll 원본 생성기 동일·네 시야 변화·기초 축 부호 검사 2개 통과. FlyVis의 시야별 독립 상태를 episode 시작에 초기화하고 41프레임·40ms·5ms 내부 적분 응답을 보존한 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-roll-road-882001/result.json`.

같은 네 시야 이력으로 기존 단일 통합 회로를 재생하고 body roll을 카메라와 동일/0/반대로 입력했다. 실제 roll/pitch 각속도와 가속도는 기존 불확실성에 따라 마스킹했다. 나머지 미제공 하중·온도도 마스킹하고 몸 속도·바퀴 속도를 0.4초 이전 0, 이후 3m/s·반경0.29m로 지정했으며 command는 고정3m/s다. 맞춘 조건만 시각·몸 자세가 일관되며 나머지는 반사실이다. 롤 +0.05 minus -0.05 속도 자극의 0.4초 이후 조향 intent 평균 차는 matched +0.00525984, fixed +0.00021041, opposite -0.00483905; 자극 이전 최대 차는 모두0이다. 시각 단독 효과와 몸 감각에 따른 출력 변화는 관찰되나 부호가 실제 안정화에 적합한지·생리·AC 주행은 미검증이다. 결과 각각 `.../20261101-coherent-roll-matched-aligned-1/`, `...fixed-aligned-1/`, `...opposite-aligned-1/`; 코드 `scripts/replay_unified_coherent_roll.py`. 기존 인자명 `--coherent-yaw`/`--body-yaw-mode`는 재사용한 CLI 이름이며 이 실험의 실제 자극은 roll이다.

처음 생성한 matched/fixed/opposite 결과는 영상3m/s 대비 몸10m/s라 불일치해 증거에서 제외했다. 후속 speed3 단일 결과도 첫0.4초 정지 불일치로 제외하고 보존했다. 위 aligned만 영상과 몸 속도를 일치시켰다. 합성 하나의 질감·직선, 물리 노출 및 차체 동역학 미검증, 최적화·실제 제어0.

## 2026-11-01 — 두 번째 질감의 합성 네 시야 롤 반복

같은 직선·전진3m/s·시각 롤 0/±0.05rad/s, 같은 네 카메라 배치와 고정 회로에서 texture seed 882002의 새 자극을 만들고 matched/fixed/opposite 몸 롤 반사실을 별도 재생했다. 이전과 같이 시작0.4초 몸 속도0 이후3m/s·바퀴3/0.29rad/s, 하중·온도·검증되지 않은 롤/피치 속도/가속도 마스크 false다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-synthetic-roll-road-882002/result.json` 및 `.../20261101-coherent-roll-882002-{matched,fixed,opposite}/result.json`. 모든 재생 optimizer/제어0.

+롤−−롤의 0.4초 이후 조향 intent 평균 차는 matched +0.00526867, fixed +0.00021931, opposite -0.00483028이다. 첫 질감의 +0.00525984/+0.00021041/-0.00483905와 부호가 같으며 수치도 비슷하다. 자극 전 최대 차는 float 오차 수준 8.94e-8이다. 고정/반대 몸 자세는 원인 분리용 인공 대조이며 fixed의 작은 시각 효과와 matched의 큰 몸 효과는 이 조건에 한정된다. 둘 다 같은 생성기의 직선·단일 속도·강제 롤이므로 독립 실제 트랙, 물리 롤/영상 발생 시각, 차체 동역학·생리·폐루프 일반화의 증거가 아니다.

## 2026-11-01 — 네 시야 합성 롤에서 망막·운동 채널 반사실

같은 두 직선 합성 질감 882001/882002, 0/±0.05rad/s 롤, 몸 롤 고정0과 속도·마스크를 기존 aligned 대조 그대로 사용했다. 각 시점 네 시야에서 망막 영상만 롤0 이력으로 교체하거나 T4/T5 motion만 롤0 이력으로 교체했다. 둘은 가능한 물리 장면이라고 가정하지 않는 시간별 반사실이며 나머지 회로와 몸을 고정했다.

반대 롤(+−−)의 0.4초 이후 조향 차 평균은 원본 +0.00021041/+0.00021931, 망막 롤0 고정(운동 반응 보존) +0.00014143/+0.00014714, 운동 롤0 고정(망막 영상 보존) +0.00006901/+0.00007228이었다. 이 조건에서 두 입력이 모두 작은 반응에 기여하며 운동 보존 대조가 더 컸다. 비선형 회로이므로 반사실 차이를 항상 가산 가능한 독립 회로로 해석하지 않는다. 원본 움직임은 같은 생성기의 강제 롤·직선·단일 속도이며 roll 보상 정답·실제 물리 시간·생리·주행은 미검증이다.

결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-coherent-roll-{882001,882002}-{retina-zero-roll,motion-zero-roll}/result.json` 및 각 outputs, 코드 `scripts/replay_coherent_roll_visual_intervention.py`. 각 실행 optimizer0·차량 제어0·배우 미적용.

## 2026-11-02 — 외부 maneuver 명령 없는 감각 경로의 합성 롤

이전 합성 롤 비교는 `model.decision`에 고정 yaw0·목표속도3m/s를 넣은 부품 진단이었다. 새 `scripts/replay_sensory_coherent_roll.py`는 동일 4시야 FlyVis 입력, 몸/속도 마스크, recurrent 단일 회로와 episode reset을 유지하되 `sensory_decision`을 사용한다. 이 경로는 원래 몸 속도로 속도 오차를 중립화하는 공학적 명령 경계이며 외부 방향·곡선 목표나 지도는 전달하지 않는다. 목표속도 판단이 학습됐다는 증거는 아니다.

texture882001/882002에서 +롤−−롤 조향 intent 차의 자극 이후 평균은 matched +0.00526012/+0.00526888, 몸 롤0 fixed +0.00021051/+0.00021933, 부호 반대 opposite -0.00483915/-0.00483041이다. 기존 명령 주입 대조의 롤 차와 비슷하지만 두 경로의 원래 절대 intent는 시간·조건 전체에서 조향 최대 약0.00356, 전진 최대 약0.225 차이가 난다. 전진 intent는 페달 실제값이나 주행 성능이 아니다. fixed/opposite는 감각 원인 분리용 반사실이다. 동일 synthetic generator의 두 직선 장면·0/3m/s·강제 롤뿐이며 물리적으로 올바른 보상 부호, 속도 계획, 신경 생리, AC 폐루프·완주/랩타임은 검증되지 않았다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261101-sensory-roll-{882001,882002}-{matched,fixed,opposite}/result.json` 및 outputs; 각 source/checkpoint hash와 optimizer0·actor 미적용·차량 제어0. MMF 미확인 요청과 운영 상태는 건드리지 않았다.

## 2026-11-02 — 감각 기반 합성 롤의 비행 그래프 집단 추적

두 직선 질감 882001/882002의 matched body roll과 body roll 0 대조를 기존 `sensory_decision`으로 재생했다. 원본 graph hash와 뉴런 순서를 검증하고 HS8, VS34, DNg02 29, graph output 후보67 및 6개 근육 readout 상태를 매 시점 기록했다. 각 matched intent 배열은 직전 기록과 정확히 일치했다. 마지막10시점의 +롤−−롤 RMS: matched HS 0.001716/0.001755, VS 0.001788/0.001849, DNg02 약4.74e-7/4.82e-7, wing 후보 약3.67e-6/3.67e-6, 근육 약6.68e-6/6.69e-6이다. 몸 롤0 대조 근육 RMS는 2.64e-7/2.75e-7이다. matched와 fixed의 롤 차를 다시 뺀 근육 RMS는 양쪽 약6.47e-6이었다.

각 집단의 수·상태·readout 단위는 달라 RMS 비율을 시냅스 전달률이나 생리 기능으로 해석하지 않는다. 역할·연결은 측정과 모델 가설이 혼합된 현재 그래프 기준이며, 단일 합성 생성기·고정 속도/직선·강제 롤에서 생긴 변화이지 올바른 방향 안정화나 실제 주행의 증거가 아니다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-roll-populations-{882001,882002}-{matched,fixed}/result.json`·outputs, 코드 `scripts/trace_sensory_coherent_roll_populations.py`. optimizer·actuator0, 생리/물리 노출 검증 미완.

## 2026-11-02 — 제한된 좌우 대칭 비행 상태 속도 경계

기존 `FlightMotorReadout`의 [b1 L/R, b2 L/R, power L/R] 6값과 현재 몸 속도만 받아 세 좌우 차이의 절댓값을 tanh로 제한하고 bounded projection·속도 상한으로 명령을 내는 `SymmetricFlightStateSpeedCommand`를 추가했다. 단일 비행 회로의 상태를 읽는 공학 어댑터이며 별도 뇌, 생리적 수용기 또는 검증된 운전자가 아니다. 초기 가중치0은 외부 상한 이내 현재 속도를 유지하고, 좌우 교환 불변이며 쌍 scale 하한0.001·가중치 gain 상한2를 둔다. 기존 `FlightStateSpeedCommand`와 명시적 타입만 선택 가능하고 기존 체크포인트는 바꾸지 않았다. 대칭·초기값·범위·입력·기존 통합 회로 17검사 통과; 처음 테스트는 속도 상한/float 정밀도 기대 오류로 두 차례 실패 후 수정했다.

이전 `20261101-multispeed-symmetry-pilot-1`의 합성 학습 질감 881403 고정 상태 90개를 원본 해시와 기존 저장 정규화/중심에서 복원했다. 가상 teacher 직선3.5·좌우 코너2.5m/s, 속도1.5/3/4.5의 공학 목표로 Adam400회, 손실1.75→0.61075. 이전에 이미 본 평가 질감881511에서 직선−평균 코너 속도 차는 0.000995/0.002945/0.003269m/s, 목표 1m/s에 크게 못 미쳤다. 근육 하나 +1e-6 교란 시 출력 변화 최대 약0.0051~0.0053m/s였다. 고정 상태에서만 평가했으며 재귀 회로 재실행·물리·속도 예측·실제 차량 일반화는 하지 않았다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-symmetric-speed-state-fit-1/result.json`, 후보 actor_qualified=false, `scripts/train_symmetric_speed_on_archived_states.py`. 합성의 실패를 근거로 배우에 미적용하며 속도 경로를 성공이나 폐루프 운전으로 표기하지 않는다.

## 2026-11-02 — 다중 속도 단일 그래프 집단의 속도 목표 진단

기존 합성 source 881403(학습)·881511(이미 과거에 본 평가), 같은 곡률0.022·코너 시작11m와 1.5/3/4.5m/s를 기존 단일 비행 회로에 재생해 VS34, DNg02 29, 날개 output 후보67, 여섯 근육의 각 조건 마지막10프레임을 기록했다. 가상 target은 직선3.5·양 코너2.5m/s다. 각 집단에 별도의 사후 ridge 진단을 훈련 질감으로만 맞췄고 feature 표준편차 하한1e-3, ridge1, 출력0~5m/s clip을 사용했다. 프레임은 서로 상관되며 시간의 독립 샘플이 아니다. 기존 extractor의 몸 속도는 첫0.4초에도 일정하지만 원본 영상은 그때 정지했으므로 몸/시각의 전 구간 물리 정합은 보장되지 않는다.

평가 질감에서 VS 직선−코너 평균 차는 속도별 0.1659/0.1712/0.3935m/s, 각각 최악 코너 margin0.1620/0.1676/0.3674m/s였지만 목표 1m/s 대비 작고 전체 target RMS 오차0.7736m/s였다. DNg02·날개 후보·6근육은 같은 floor/정규화에서 차이가 거의 0이고 RMS 약0.471m/s였다. 예컨대 target의 상수 3m/s 대비 오차도 약0.471m/s이므로 작은 RMS가 성공을 뜻하지 않는다. VS의 근사적 포트 하나 +1e-6 교란당 비클립 선형 변화 최대 약0.000303m/s로 계산했으나 다른 feature/시간 교란과 경로 폐루프 민감도는 미검증이다. VS는 움직임 시각 집단이지 직접 근육이나 인증된 속도 회로가 아니며 임시 진단기를 새로운 뇌 또는 배우에 연결하지 않았다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-multispeed-graph-populations-1/result.json`·`population-states.npz`; 코드 `scripts/probe_multispeed_graph_populations.py`. 한 학습·한 평가 질감, 단일 기하·가상 속도 목표의 사후 진단이며 생리·다른 트랙·안전 속도·실제 폐루프는 미검증이다. 기존 단일 회로와 원본을 보존했고 real fit/배치/차량 제어0.

## 2026-11-02 — 합성 다중속도 시작 정합과 공간 제거 대조

직전 단일 그래프 집단 진단의 첫0.4초 몸 속도 불일치를 수정해 영상이 정지한 동안 몸과 바퀴 속도를0, 이후 원본 영상의 1.5/3/4.5m/s와 일치시켜 episode 전체를 재생했다. 정지→등속은 강제로 지정한 불연속 합성 이력이며 차량 동역학은 아니다. 동일 기존 학습/이미 본 평가 질감과 목표, VS/DNg02/날개 후보/근육의 사후 ridge 조건을 유지하고, 평가에서 공간 투영만 0으로 해 매 속도·코너의 동일 이력 대조를 추가했다.

VS의 평가 직선−평균 코너 차는 1.5/3/4.5m/s에서 원본 +0.16586/+0.17116/+0.39349m/s, 공간 제거 +0.04169/+0.04674/+0.05019m/s였다. 평가 속도 목표 RMS 오차는 원본 0.77364m/s로 높았다. 다른 집단의 같은 설정 대조는 거의 0이었다. 따라서 이 한 합성 경로에서 VS 진단의 일부 구분이 공간 투영에 의존하지만, 0이 아닌 제거 대조 및 초기 표본·비선형 재귀의 영향 때문에 시각 코너 예측을 구현했다거나 이 경로가 최적이라고 주장할 수 없다. 기존 시작 불일치 수정 전 마지막 시점 결과와 유사했으나 전체 상태 동등성은 별도 검증하지 않았다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-aligned-multispeed-populations-1/result.json`, `population-states.npz`, 코드 `scripts/probe_aligned_multispeed_graph_populations.py`. 예전 평가 질감에 대한 사후 탐색, 단일 기하·가상 target·상관 프레임이며 VS readout은 운전자에 연결하지 않았다. 실제 fit/생리/폐루프/vehicle control0.

## 2026-11-02 — 단일 비행 그래프 HS/VS 집단 상태 절제

기존 네 시야 합성 롤 이력 882001/882002, 원래 감각 경로·고정 몸 롤0·시각 롤0/±0.05rad/s를 재생했다. 현재 정확한 graph hash의 HS8 또는 VS34 노드에만 상태 mask0을 적용하고 매 결정에서 해당 상태가 0임을 검사했다. 나머지 그래프·입력·시야·몸·episode reset을 유지했다. 이는 입력 연결뿐 아니라 해당 집단의 재귀 상태 전체를 제거하는 인공 개입으로, 원래 상태의 국소 기능이나 실제 뉴런 억제를 동일하게 재현한다고 가정하지 않는다.

0.4초 이후 +롤 minus −롤 조향 intent 평균은 질감 882001/882002의 원본 +0.00021051/+0.00021933, HS 절제 +0.00017437/+0.00018173(각 원본의82.8%), VS 절제 +0.00003162/+0.00003283(각15.0%)였다. 집단 절제로 기준 활동도 바뀌므로 이를 독립 경로의 정확한 기여율로 합산하지 않는다. 동일 합성 생성기·직선·단일 속도·작은 롤 조건에서 현재 모델의 대조이며 실제 초파리의 생리 기능, 올바른 조향 부호, 차체 안정화나 AC 폐루프를 입증하지 않는다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-roll-{882001,882002}-lesion-{HS,VS}/result.json` 및 outputs; 코드 `scripts/lesion_sensory_roll_flight_population.py`. actor 미적용·optimizer/차량 제어0. 추가로 서로 다른 시간 자극·다른 축 및 실제 물리 정합 조건을 검증해야 한다.

## 2026-11-02 — 동일 합성 도로에서 yaw와 roll의 HS/VS 대조

기존 롤과 같은 planar 직선 도로·질감882001/882002·정지0.4초 후 world-forward 3m/s와 각속도0/±0.05rad/s를 사용하되, 이번에는 네 카메라의 공통 회전을 Z-up yaw로 생성했다. 동일 FlyVis와 에피소드 초기화로 T4/T5 응답을 만들고 동일 단일 감각 회로에서 몸 회전 감각을0으로 고정해 원본·HS8 상태0·VS34 상태0을 비교했다. `scripts/replay_flyvis_synthetic_straight_005_yaw.py`, `scripts/lesion_sensory_yaw_flight_population.py`. yaw 방향의 외부 목표나 교사값은 회로에 넣지 않았다.

0.4초 이후 +yaw−−yaw의 평균 조향 intent는 질감별 원본 +5.2208e-6/+2.5628e-6, HS 절제 +4.2488e-6/+2.1822e-6, VS 절제 +4.9078e-7/−1.2834e-7이었다. 같은 조건의 roll 원본은 +0.00021051/+0.00021933이었다. yaw 수치가 작고 VS 제거에서 한 질감은 부호가 바뀌므로 비율이나 방향성 정확도를 과장하지 않는다. 두 축이 동일 각속도라고 영상의 변화량·FOV·원점/질감 효과가 동일한 것은 아니며, yaw 상태에서도 world-forward 이동은 회전된 차량의 실제 진행 궤적이 아니다. VS/HS 생리적 축 선택성, 실제 조향/균형·폐루프는 이 실험만으로 입증할 수 없다. 몸 yaw는0의 반사실이다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-synthetic-yaw005-road-{882001,882002}/result.json` 및 `.../20261102-yaw005-{882001,882002}-lesion-{none,HS,VS}/result.json`; 결과의 상속 필드 `opposite_roll_*`와 `synthetic_body_roll_mode`는 코드 재사용으로 붙은 이름이며 이 실행의 실제 자극은 yaw·몸 회전0이다. 원본 보존, optimizer·차량 제어0.

## 2026-11-02 — 합성 yaw/roll 입력과 상태의 마지막 구간 감사

같은 두 질감의 yaw 및 roll scene source hash와 몸 회전0·무절제 flight-state hash를 확인해 마지막10프레임의 반대 방향 차를 계산했다. yaw T4/T5 motion RMS는 시야별 약0.027~0.039, roll은 약0.055~0.087이었다. 시각 HS/VS 그룹 상태 RMS는 yaw HS 약0.000222~0.000247·VS 0.000169~0.000184, roll HS 0.001716~0.001755·VS 0.001788~0.001849였다. 값은 서로 다른 집단·차원·단위이며 원인 경로나 전달률을 뜻하지 않는다.

자극 후 전체 평균은 앞서 yaw 두 질감 모두 양수였으나 마지막10프레임의 조향 차 평균은 첫 질감 +5.35e-6, 두 번째 −1.44e-6이었다(시간 RMS 각각 1.54e-5/1.29e-5). 같은 마지막 구간 roll 평균은 +0.000414/+0.000431이었다. 이는 작은 yaw 출력의 시점·질감 민감성에 대한 추가 근거다. 동일 각속도라도 카메라 이동과 시각 기하·FOV 자극량이 다르고, 합성 yaw는 world-forward 이동이므로 이 관찰은 회로의 생리적 축 선택성이나 실제 제어 부호를 확정하지 않는다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-synthetic-roll-yaw-signal-audit-1/result.json`; 코드 `scripts/audit_synthetic_roll_yaw_signal.py`. fit/제어0. 다른 시간 궤적, 방향 반전·변화하는 실제 몸 감각과 고정/절제 대조가 필요하다.

## 2026-11-02 — 진행 방향과 몸 yaw를 맞춘 합성 원호

이전 회전 영상은 차체가 yaw해도 world-forward로 직진했다. 새 `SyntheticPathYawRoadVisual`은 기존 투영을 보존하면서 body yaw ω·속도 v에 대해 카메라 중심을 x=v sin(ωt)/ω, y=v(1−cos(ωt))/ω로 이동한다(ω0은 x=vt, y0). 네 카메라 위치·광선은 같은 자세로 회전한다. 롤·yaw0 원본 영상 동일, 측방 변화 및 위치 미분이 차체 방향 속도와 일치하는 기초 검사2개 통과. 이는 평면 도로와 무슬립 원호를 강제한 운동학 가설이지 타이어·접지·가속·조향 동역학이나 AC 서킷이 아니다. 정지0.4초 이후 3m/s·yaw 속도0/±0.05rad/s, 질감882001/882002. 3m/s로 순간 전환한다.

두 합성 네 시야 FlyVis 이력을 기존 단일 비행 회로의 `sensory_decision`으로 재생했다. body yaw rate를 시각과 같게 주거나 0으로 고정했으며, 롤·피치 각속도 및 가속도·하중/온도는 원본과 동일하게 미제공 mask다. 몸 yaw0은 원인 분리용 인공 반사실이다. +yaw−−yaw의 0.4초 이후 조향 intent 평균은 질감별 matched -0.00867586/-0.00867779, fixed +0.00001614/+0.00001426이었다. 마지막10프레임은 matched -0.00996372/-0.00996859, fixed +0.00003596/+0.00003116이다. 자극 전 차0. 이 조건에서는 몸 회전 감각의 영향이 더 크지만 수치만으로 보상 부호가 옳다거나 시각 선행 예측을 했다거나 생리·실제 AC 안정화/완주가 검증됐다는 뜻은 아니다.

원본 영상·시각 응답 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-synthetic-path-yaw005-{882001,882002}/result.json`, 회로 `.../20261102-path-yaw-{882001,882002}-{matched,fixed}/result.json`·outputs; 코드 `scripts/replay_flyvis_synthetic_path_005_yaw.py`, `scripts/replay_sensory_kinematic_yaw.py`. source/checkpoint hash 기록, 배우 미적용·최적화/차량 제어0.

## 2026-11-02 — 네 시야·몸 yaw 동기 합성 S자 반전

이전 일정 yaw의 원호를 확장해 0.4초부터 3m/s·±0.05rad/s로 움직이다 1.0초에 yaw rate를 즉시 반전하고 위치·heading은 연속인 무슬립 합성 S자 경로를 만들었다. 네 카메라 위치/광선은 공통 자세를 따른다. yaw rate 전환과 출발의 순간 변화, 평면 가상 도로·고정 속도는 실제 조향·타이어 동역학이 아니다. 같은 FlyVis·단일 회로·검증되지 않은 몸 축 마스크, 질감882001/882002에서 몸 yaw matched와 0 fixed를 재생했다. 외부 steering target은 입력하지 않았다.

+방향 시작 minus −방향 시작의 조향 intent 차는 0.80≤t<1.00초 평균 matched -0.00996260/-0.00996274, 1.40≤t≤1.60 평균 +0.01001050/+0.01001104였다. matched 차가 반전 전 평균 부호와 처음 달라진 표본은 두 질감 모두 반전 후0.12초였다(40ms 표본·1e-7 제외 기준). fixed 반사실은 같은 구간 +0.00001758/+0.00001763, +0.00002528/+0.00002594로 해당 기준 반전이 관찰되지 않았다. 시각과 몸 상태를 분리한 모델 내 반응이며 0.12초를 실제 차량 제어 지연, 올바른 보상 부호 또는 신경 생리로 해석하지 않는다. last/mean 창은 이 실행 결과 관찰 후 구분한 탐색적 요약이며 프레임들은 독립 표본이 아니다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-yaw-reversal-path-{882001,882002}/result.json`, `.../20261102-sensory-yaw-reversal-{882001,882002}-{matched,fixed}/result.json` 및 상태/intent outputs. 코드 `scripts/replay_flyvis_synthetic_yaw_reversal_path.py`, `scripts/replay_sensory_yaw_reversal.py`. fit·차량 제어0, 실제 폐루프·노출 정합·생리 및 트랙 완주 미검증.

## 2026-11-02 — 합성 S자 시야별 원본 무회전 치환

기존 yaw reversal 경로 882001/882002에서 몸 yaw를0으로 고정한 반사실에 대해 전/좌/우/후 각 시야의 retina와 T4/T5 전체를 동일 시점 yaw0 원본 이력으로 치환했다. 각 조건은 에피소드 상태를 새로 시작하고 나머지 세 시야·몸·회로는 동일하게 유지했다. 네 시야를 모두 치환한 대조도 수행했다. 이는 동시 물리 영상이 아니며 시야 상호작용이 있으므로 각 차이를 독립적인 뇌나 선형 기여도로 합산하지 않는다.

+방향 시작−−방향 시작 조향 차의 0.80≤t<1.00초 평균은 질감별 원본 +1.758e-5/+1.763e-5, 전방 치환 +2.749e-5/+2.699e-5, 좌측 +1.398e-5/+1.341e-5, 우측 -3.30e-6/-3.65e-6, 후방 +1.418e-5/+1.531e-5, 전부 치환0/0이었다. 1.40~1.60초에는 원본 +2.528e-5/+2.594e-5, 전부 치환0/0이다. 우측 치환에서 이 합성 구간의 작은 부호가 달라졌지만 실제 방향 판단, 생리, 맞춘 몸 yaw에서의 조향이나 폐루프 이득은 검증하지 않았다. 같은 생성기·두 질감·한 속도/궤적, 40ms 표본·사후 선택 구간이다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-yaw-reversal-{882001,882002}-view-{front,left,right,rear,all}/result.json` 및 outputs. 당시 실행 코드 `scripts/replay_yaw_reversal_view_interventions_v1.py`; 결과의 `all_views_move_together=true`는 상속된 잘못된 메타데이터다. 각 `replaced_view`의 실제 배열 치환이 우선이며 전체 치환 외에는 시야를 물리적으로 함께 움직인 조건이 아니다. 이후 `scripts/replay_yaw_reversal_view_interventions.py`의 해당 플래그를 false로 수정했으며 이 수정 버전은 재실행하지 않았다. optimizer·실제 제어0, actor 미적용.


## 2026-11-02 — 단일 회로의 합성 폐루프 속도 경계 비교

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 이 실험은 기존 단일 회로의 속도 출력 경계만 비교하며 별도의 운동 회로를 추가하지 않는다.

고정 상태 합성 학습(400 Adam 단계)은 손실 1.75→0.611이었으나 직선과 코너 출력 차이가 목표 1m/s에 비해 약 0.001–0.0033m/s에 그쳤다. 후보는 미인증이며 배포하지 않았다. `20261102-symmetric-speed-state-fit-1/result.json`에 원본이 있다.

이후 `scripts/evaluate_symmetric_speed_closed_loop.py`로 기존 및 대칭 속도 경계를 각각 합성 좌·우 굽이에서 실행했다. 시작 x=8m, 속도 3m/s, 40ms 간격 81단계, texture seed 882201, 굽이 시작 x=12m, 곡률 계수 0.018이다. 이전 회로 상태를 이어가며 조향과 가감속 결과로 차량 위치를 갱신하고 다음 네 시야와 속도·바퀴·yaw 감각을 생성했다. 중심선과 위치는 렌더링·종료·평가에만 썼고 actor의 경로 정보로 직접 주지 않았다. 최적화 단계는 0회다. 초기화 과정에 이전 학습 계획을 읽는 코드는 있으나 그 계획을 이 평가의 사전등록으로 간주하지 않는다. seed의 다른 실험에서의 사용 여부는 독립 확인하지 않았다.

| 경계 | 굽이 | 종료 x (m) | 종료 속도 (m/s) | 중심선 오차 (m) |
|---|---:|---:|---:|---:|
| 기존 | -1 | 18.136 | 3.437 | +0.343 |
| 기존 | +1 | 18.141 | 3.438 | -0.885 |
| 대칭 | -1 | 17.055 | 2.876 | +0.195 |
| 대칭 | +1 | 16.717 | 2.637 | -0.551 |

모두 정해진 단계 수로 종료했다. 종점은 서로 다르고 굽이를 지난 거리도 다르므로 오차만 비교해 성능 향상이나 완주를 주장할 수 없다. 합성 무슬립 자전거 차량과 `steer=-axis*0.35`, `accel=throttle*3-brake*6`은 실측되지 않은 공학 가정이다. 하중 이동·타이어·셋업과 실제 AC 제어, 생리 반응, 빠른 랩은 검증되지 않았다. 새 경계는 생물학적 회로가 아니라 학습된 공학적 출력 어댑터다.

원본: `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-symmetric-speed-closed-loop-1/`의 `result.json`, `trajectory.npz`, `manifest.json`, `independent-audit.json`. 궤적 SHA-256은 `c669268bb3c1fdf5195b96cda67d49fa90628390bfdd89b1edb7a6d48c824919`이다. 별도 산술 검산은 유한값, 초기 조건·40ms 연결, 차량 갱신, yaw 되먹임, 중심선 및 범위를 통과했으나 신경망 결정을 재생하거나 물리/생리를 검증한 것은 아니다. 기록의 이전 미사용 질감이라는 표현은 확인되지 않았으며 위와 같이 정정한다.

다음에는 같은 진행 거리에서 좌우 오차와 속도를 비교하고, 시각·몸 감각 차단 대조와 여러 곡률/질감으로 방향성 및 인과 기여를 검증해야 한다. 기존 MMF 카메라 요청의 독립 종료 결과는 미확인 상태로 남아 있으며 이번 평가에서 실제 카메라 요청이나 AC 제어는 보내지 않았다.

같은 진행 위치 x=16.5m에서 기록을 선형 보간한 추가 비교: 기존 경계의 중심 오차는 굽이 -1/+1에서 +0.1285/-0.5129m, 대칭 경계는 +0.1314/-0.5082m였다. 속도는 각각 기존 3.3541/3.3544m/s, 대칭 2.8656/2.6384m/s였다. 단일 질감·짧은 구간의 관측이지만 속도 감소에 비해 중심 오차 차이는 작아 새 경계의 주행 개선 근거는 없다. 보간은 기록된 40ms 표본 사이에서 했다.


## 2026-11-02 — 합성 폐루프 시각·몸 yaw 반사실

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 기존 단일 회로와 기존 속도 경계를 고정하고 기존과 같은 합성 폐루프·질감 882201·양쪽 코너·81단계에서 정상, 네 시야의 도로만 직선으로 렌더링, 몸 yaw 입력을 유효한 0으로 고정하는 각각의 조건을 비교했다. 각각 2회, 총 6회이며 최적화는 0회다. 시각 치환은 다른 몸 위치에 맞춰 직선 영상이 생성되고, yaw 0도 값 치환이므로 감각 결측·센서 고장 대조가 아니다.

| 조건 | 종료 yaw: 굽이 -1 / +1 (rad) | 두 종료 yaw 차 (rad) | x=16.5m 중심 오차: -1 / +1 (m) |
|---|---|---:|---|
| 정상 | -0.06572 / -0.03823 | +0.02750 | +0.1285 / -0.5129 |
| 직선 영상 | -0.04621 / -0.04621 | +0.000000008 | +0.1833 / -0.5457 |
| 몸 yaw 0 | -0.06848 / -0.03985 | +0.02863 | +0.1196 / -0.5185 |

각 조건의 합성 조향이 이후 궤적과 영상까지 바꾸므로 값 차이는 폐루프 전체 영향이고 특정 뉴런 또는 독립적 즉시 반응을 증명하지 않는다. 정상에서 좌우 굽이의 출력 차이는 있으나 두 종료 yaw가 모두 음수인 비대칭이 남고, 이로써 안정적인 코너 주행을 주장할 수 없다. 제한된 하나의 질감, 단일 속도·곡률과 공학적 무슬립 차량·조작 부호 가정이며 생리, 실제 AC, 하중·타이어, 변속·셋업, 완주·랩타임은 검증하지 않았다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-sensory-interventions-1/`: `result.json`, `manifest.json`, `trajectory.npz`, `audit.json`. 궤적 SHA-256 `b012c53d5a9581c81789852c8f9ba2b481306cfce772f204d8af628dd37f75ff`. 감사는 해시, 유한값, 초기 이후 연속성·간격, 평가 중심선과 yaw 개입을 점검했으며 신경망과 렌더링을 독립 재현하지 않았다. `scripts/evaluate_closed_loop_sensory_interventions.py`가 실행 스크립트다. 임의의 교사 중심선·절대 좌표는 렌더링·종료·평가에만 사용한다. 이전 실험 초기화 계획은 이 실험의 사전등록이 아니다.

다음에는 초기 조향 편향과 시각 좌우 대칭 실패를 같은 상태에서 진단하고, 여러 곡률·질감의 동등한 진행 위치에서 감각 개입을 반복해야 한다. 실제 카메라 MMF의 독립 종료 결과와 실제 폐루프는 여전히 확인되지 않았다. 이 실험에서 실제 차량 제어는 보내지 않았다.


## 2026-11-02 — 합성 고정 이력에서 공통 조향 편향 분리

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 앞의 정상 합성 폐루프 양쪽 이력 81단계를 입력으로 고정하여 단일 회로를 매 조건 새 episode로 재생했다. 공간 시각 경계, 새 몸 피드백 어댑터, opponent 시간 경계, 기본 시각 선형층 가중치를 각각 0으로 만들고 나머지는 유지했다. 기본 시각 조건의 bias는 유지한다. 개입은 전체 시간에 적용하며 계산된 조향은 기록된 다음 상태를 바꾸지 않는다.

| 조건 | 굽이 -1 평균 조향 intent | 굽이 +1 평균 조향 intent |
|---|---:|---:|
| 정상 | -0.04628 | -0.02707 |
| 공간 시각 경계 제거 | -0.03711 | -0.03802 |
| 몸 어댑터 제거 | -0.04652 | -0.02731 |
| opponent 시간 경계 제거 | -0.04617 | -0.02686 |
| 기본 시각 가중치 제거, bias 유지 | -0.04267 | -0.02263 |

공간 경계는 이 이력의 좌우 차이에 크게 기여하지만 제거 후에도 음의 공통 출력이 남는다. 기본 시각 가중치 제거는 두 평균을 양의 방향으로 약 0.0036/0.0044 변화시켜 공통 편향의 일부에 영향을 주지만 원인을 완전히 분리하지는 못한다. 몸 어댑터 제거는 전체 몸 감각 제거가 아니며 본 모델의 다른 몸 경로는 남는다. 비선형 회로의 개별 경계 제거 효과를 더해 원인을 분해할 수 없다. 단일 질감·짧은 두 합성 이력으로 실제 생리나 일반적 원인, 주행 개선을 증명하지 않는다.

`C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-fixed-history-boundary-bias-1/`에 원본 `result.json`, `trajectory.npz`, `analysis.json`, manifest를 보존했다. trajectory SHA-256 `d94b1e6c5c7c7b6dd0fd0c597acd9cd2fd3afd1da9a82cf9fa74c9c313c9a8a3`; 스크립트 `scripts/audit_fixed_history_boundary_bias.py`. 정상 intent는 원본 폐루프와 최대 9.4e-8 이내로 재현됐고 각 개입의 위치·속도·yaw 입력은 원본과 일치했다. 결과 파일의 final_position/final_yaw는 고정 이력 마지막 상태에서 계산한 **한 단계 예측**이며 각 개입의 폐루프 결과가 아니다. 독립 신경망 재현, 실제 물리·시간 검증, 배우 배포, 실제 차량 조작 및 학습은 하지 않았다.

다음은 초기 신경 상태와 고정된 시각 bias·몸 경로의 영향 및 직선 입력의 조향 대칭을 별도 검증한 후, 좌우 균형을 보존하는 단일 회로 경계 수정 후보를 여러 독립 합성 조건과 실제 센서 계약에서 평가한다. 실제 카메라 요청의 독립 종료는 미확인이다.


## 2026-11-02 — 기존·과거 중립 기준의 합성 폐루프 비교

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 기존 단일 회로, 시각 및 속도 경계를 고정하고 조향 readout의 lateral_reference만 기존 값과 과거 등록 값을 비교했다. 두 값의 대응 근거와 gain/mix 일치를 실행 시 확인했다. 과거 값은 다른 비시각 물리 quiet 상태에서 얻은 것이므로 현재 영상·몸 입력에 적합한 중립이라고 가정하지 않는다. 새로운 회로가 아니며 모델 외부의 별도 조향 정책을 추가하지 않았다.

사후 탐색 실험으로 합성 texture seed 882201/882331, 굽이 -1/0/+1, 각 81단계 40ms, 시작 x=8m·3m/s다. 12조건 모두 단계 제한으로 종료했다. x=16.5m의 중심 오차를 선형 보간하면 두 질감에서 기존 기준은 굽이 -1/+1에 약 +0.128/−0.513m 및 +0.131/−0.516m, 과거 기준은 +0.292/−0.347m 및 +0.294/−0.349m다. 직선 오차는 기존 −0.181/−0.187m, 과거 −0.026/−0.027m다. 종료 yaw도 기존 양쪽 코너 모두 음수에서 과거 기준의 양쪽 부호가 달라졌다. 직선 편향과 코너 간 불균형은 줄었지만 한쪽 코너 오차는 악화됐고 양쪽 코너 모두 중심을 잘 추종했다고 보기는 어렵다.

간단한 무슬립 bicycle, 추정 조향/가속 매핑, 낮은 속도와 짧은 구간만 시험했다. 두 seed의 전역 미사용 여부는 확인하지 않았고 사전등록된 일반화 평가가 아니다. 실제 영상·하중·타이어·상대풍·기어·셋업·생리·AC 폐루프·유효랩 및 lap time은 검증하지 않았으며 과거 기준을 actor에 배포하지 않았다. 훈련/optimizer 0, 실제 차량 제어 0.

첫 실행 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-registered-neutral-loop-1/result.json`은 CUDA/CPU tensor 비교 오류로 초기화에서 실패했다. 비교만 수정한 두 번째 실행 `.../20261102-registered-neutral-loop-2/`의 result/manifest/trajectory/analysis에 결과를 보존했다. 궤적 SHA-256 `e650d6754afcd8b3b951ad9d137f760576d71a1c329d0a5dafa0f0dc01cd4d59`; `scripts/evaluate_registered_reference_closed_loop.py`. 감사는 12조건의 유한값, 40ms 상태 연결, yaw 되먹임, 중심선 계산과 기존 882201 코너 재현을 점검했으나 신경망/물리를 독립 재현하지 않았다. 초기화가 이전 학습 계획을 읽는 것은 이번 평가의 사전등록이 아니다.

다음은 조향 편향 보정과 코너 응답을 따로 취급해 더 긴 거리·속도·곡률·미사용 질감에서 양쪽 코너의 동일 진행 위치 오차 및 센서 개입을 평가하고, 적합한 기준은 독립적인 감각 조건에서 정해야 한다. 실제 카메라 MMF 요청의 독립 종료는 아직 확인되지 않았다.


## 2026-11-02 — 공간 시각 경계 배율의 합성 폐루프

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 앞서 시험한 과거 등록 중립 기준을 동결하고, 같은 단일 회로의 공간 시각 projection 가중치에 1·2·4배를 적용한 공학 탐색이다. tanh와 후속 회로가 비선형이므로 가중치 배율은 실제 포트 출력의 배율이 아니다. 몸/시간 시각/속도 경계 및 그래프는 고정했다. 추가 학습이나 실제 차량 조작은 없다.

실행 전 계획은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-spatial-gain-loop-plan-1/plan.json`: 각 배율 × texture 882401/882503 × 굽이 -1/0/+1, 시작 x=8m·3m/s, 81단계·40ms다. 18조건 모두 단계 제한으로 종료했다. seed의 과거 사용 여부는 독립 확인하지 않았으며, 배율 선택 자체가 이전 결과를 바탕으로 한 탐색이다.

| 배율 | 굽이 -1 중심 오차 (m) | 직선 오차 (m) | 굽이 +1 중심 오차 (m) |
|---|---|---|---|
| 1 | +0.2910 / +0.2937 | -0.0254 / -0.0273 | -0.3471 / -0.3495 |
| 2 | +0.2828 / +0.2841 | -0.0219 / -0.0241 | -0.3395 / -0.3408 |
| 4 | +0.2787 / +0.2792 | -0.0183 / -0.0199 | -0.3359 / -0.3366 |

각 칸은 두 질감 순서이며 모두 x=16.5m에서 선형 보간한 값이다. 속도는 모든 조건 약 3.35m/s다. 4배에서도 코너 절대 오차는 0.28~0.34m, 2→4배의 추가 개선은 작다. 포화가 관여할 가능성은 있으나 여기서는 내부 신호 포화를 측정하지 않았으므로 원인으로 확정하지 않는다. 이전 과거 중립 기준은 비시각 조건에서 취득되어 현재 actor의 적격한 중립 기준이 아니다. 짧고 단순한 합성 무슬립 차량·한 곡률만의 결과이며 생리, 실제 물리·영상·차량 제어, 코너 안정성·완주·랩타임은 검증하지 않았다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-spatial-gain-loop-1/`의 result/manifest/trajectory/analysis, 코드 `scripts/evaluate_spatial_gain_closed_loop.py`. trajectory SHA-256 `7c2cbf952da0f9daf10a6b4c34cf12b9c790d7e93e8deecadb91114835506376`. 기록 감사는 계획과 소스 해시, 상태 연결·유한값·시간 간격·중심선·yaw 피드백을 확인했으며 신경망·물리를 독립 재현하지 않았다. 실행 초기화가 읽는 과거 학습 계획은 이번 평가의 사전등록이 아니다.

다음은 신호 증폭이 약해지는 원인을 포트 포화와 근육·조향 readout 단계에서 분리하고, 더 긴 구간과 여러 곡률·속도에서 실제 감각 기여와 선행 코너 반응을 평가해야 한다. 실제 카메라 MMF 요청의 독립 종료와 실제 AC 폐루프는 미확인이다.


## 2026-11-02 — MMF 재개 안전 검사 누락 수정

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 실제 영상 통합을 점검할 때 기존 요청 `mmf-stream-d487677e57a14987aea819bb4f516523`의 stop만 있고 자기 ID terminal이 없으며, request.json은 그대로이고 producer status는 이전 ID의 paused 시작 조건 오류를 계속 기록 중임을 다시 관찰했다. AC 프로세스는 존재한다. 새 카메라/lifecycle/차량 조작은 보내지 않았다.

순수 `PausedMMFStartupGate`는 최초 인가 시 정지·브레이크·가스·보조선 등 조건을 확인했으나, 이후 카메라 준비 확인과 재개 관찰에서는 그 조건의 변경을 재검사하지 않았다. `src/flybrain_drive/paused_mmf_startup.py`에서 기존 정지 인가의 상태 검사를 복사한 heartbeat에 재사용하고, 카메라 소유와 paused/live는 기존 단계별 조건으로 따로 검사하도록 수정했다. 속도·가스·브레이크, expert probe·보조선, 오류·미완료 영수증 변경 시 두 전이 모두 실패하고 phase를 유지한다. 원본 상태는 수정하지 않는다. `tests/test_paused_mmf_startup.py`에 각 두 단계의 경계·NaN·오류 조건을 추가했고 관련 두 테스트 파일 총 23검사가 통과했다.

이 순수 게이트는 heartbeat 신선도를 검증하거나 lifecycle/camera 명령을 보내지 않는다. 실제 MMF shadow runner의 paused-start/resume 연결은 미구현이며 fail-closed 상태다. 이 수정만으로 소유 요청이 종료되거나 안전하게 재시작되었다고 주장하지 않는다. 다음에는 기존 요청의 동일 ID terminal 또는 안전한 취소를 먼저 확보하고, 별도로 명시적 소유·최신 heartbeat·API receipt·예외 cleanup이 검증된 핸드오프를 runner에 연결해야 한다.


## 2026-11-02 — 실제 미접수 MMF 형식의 Lua 모의 재현

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 현재 카메라 request는 `mmf-stream-d487677e57a14987aea819bb4f516523`, immutable_ram_v3·80프레임·40ms이며 stop은 있으나 자기 ID result가 없다. 최신 producer는 같은 generation에서 이전 ID의 paused 시작 실패를 계속 출력하고 session은 unpaused·정지·브레이크다. 관측만 했고 원본 파일과 프로세스를 조작하지 않았다.

`tests/test_camera_pre_admission_cancel.py`에 관측된 요청과 동일한 주요 프로토콜 필드·stop·동일 generation을 넣은 독립 LuaJIT 테스트를 추가했다. 모의 운영 원본은 unpaused에서 두 update 후에도 결과가 없고 paused 시작 오류를 보였으며 MMF open은 0회였다. 미배포 후보는 같은 입력으로 같은 ID의 0프레임 `stopped`, `camera_started=false`, `actor_stream_created=false`를 기록하고 이후 update에서도 MMF/lifecycle 호출은 0회였다. 앞선 취소·재개 gate 테스트를 포함한 24검사 통과. 이는 후보의 로컬 모의 근거이고 운영 앱의 갱신이나 취소 완료가 아니다.

실제 원본은 stop을 active 분기에서만 확인하며 begin 내부의 MMF open과 카메라 초기화는 그보다 먼저다. 따라서 현재 세션을 pause해 미결 요청을 접수시키는 것은 안전한 사전 취소를 증명하지 않는다. 운영 Lua 후보의 적용·재시작 시점, 기존 요청의 동일 ID terminal, 기존 세션/원본 보호, 이후 paused-start/resume 핸드오프 검증이 여전히 필요하다. 원본 학습·실제 차량 제어·새 카메라 요청 0. 영상 처리 지연의 새로운 live 측정이나 개선은 이 작업에서 하지 않았다.


## 2026-11-02 — 재시작 시 이전 generation 요청 관찰 후보

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 현재 미종료 MMF 요청은 unpaused에서 재시도 중이다. 기존 엔진 restart API는 paused 조건이지만 기존 producer는 pause 시 stop 확인 전에 MMF/카메라를 열 수 있다. 재시작으로 producer generation이 달라지면 기존 v1 취소 후보는 같은 generation 조건을 충족하지 못한다. 이를 근거로 기존 요청·실제 세션은 변경하지 않았다.

비배포 v2 후보 `integrations/ac_csp/candidates/FlyBrainCamera-pre-admission-cancel-v2.lua`, 생성 `scripts/build_camera_orphan_observation_candidate.py`: 다른 generation의 안전한 ID에 stop이 있고 original result가 없으면 별도 `orphan-observation.json`에 이전 generation 결과·시작 여부·프레임은 미상, 옛 프로세스 종료 미검증, actor 자격 false로만 기록한다. 원래 result.json을 만들거나 덮어쓰지 않으며 이 기록 자체는 새로운 요청의 자격을 주지 않는다. 기록 실패 시 ID를 소비하지 않고 재시도한다. 이미 별도 관찰이 있으면 덮어쓰지 않고 검토를 요구한다. 같은 generation v1 취소는 유지한다.

LuaJIT 모의에서 다른 generation 관찰, 쓰기 실패 재시도, 기존 terminal 보존, 동일 generation 취소를 검증하고 기존 gate/취소 시험 포함 28개 통과. 후보 SHA-256 `e5f5c189fa5fed5f0dda4be8a48c0c906599b2ef9b429aae02ae5921e041a2b2`. 실제 앱 동적 재적재, 엔진 재시작, 기존 원 프로세스의 종료 및 terminal, camera 재수집은 검증하지 않았다. 미배포 후보는 생리·주행 성능과 무관하며 새 camera/lifecycle/차량 명령이나 추가 학습을 보내지 않았다.

다음에는 기존 producer를 안전하게 fence할 수 있는 절차와 같은 프로세스 핸들의 종료 증거를 확보하고, 미확인 요청을 terminal로 오인하지 않는 host 측 소유 인계 및 실제 runner의 paused-start/resume 연결을 검증한다. 이들 조건을 충족하기 전에는 재요청하거나 재시작의 성공을 추정하지 않는다.


## 2026-11-02 — 공간 시각 배율의 포트·근육 추적

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 합성 폐루프 gain1의 한 질감(882401) 좌우 81단계 이력을 고정해 단일 회로의 공간 projection을 1/2/4배로 재생했다. 시각 경계 선형 출력(tanh 전), 6개 근육 활성 및 조향 intent를 기록했다. 조작이 고정 이력에 되먹임되지는 않으므로 폐루프의 후속 변화를 재현한 것은 아니다. 기준 중립은 앞선 비시각 조건의 미인증 과거 기준이다.

| 배율 | tanh 입력 절댓값 >2 비율, 굽이 -1/+1 | 절대 포트 변조 평균, -1/+1 | 평균 조향 intent, -1/+1 |
|---|---|---|---|
| 1 | 8.38% / 7.44% | 0.2519 / 0.2487 | -0.01399 / +0.00557 |
| 2 | 31.39% / 30.36% | 0.3362 / 0.3334 | -0.01589 / +0.00735 |
| 4 | 49.00% / 48.74% | 0.4065 / 0.4044 | -0.01681 / +0.00822 |

절댓값 >2는 tanh 도함수가 약 0.071보다 작은 구간을 나타내는 사후 분석 기준이며 전체 포트·프레임을 합친 비율이다. 원래 증폭 시 포트 비선형 포화가 증가한 관측 근거이지만, 회로·readout과 상호작용하므로 이전 폐루프 오차 개선 둔화의 유일한 원인으로 단정하지 않는다. 실제 생리나 회로 신경 기능, 실제 차량 제어·타이어·셋업·유효랩은 검증하지 않았다. 1배 intent는 원래 폐루프 기록과 최대 1.84e-7 이내로 재현됐고 이력 상태·파일 해시를 확인했다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-spatial-gain-internal-trace-1/`의 result/trajectory/port-muscle-traces/analysis, 스크립트 `scripts/audit_spatial_gain_internal_trace.py`; trace SHA-256 `493a8d83d944d64fa798ed1d6aeab638681b89901c6470b44734335ac6fffc61`. 생성된 trajectory의 각 next state는 해당 고정 입력에서의 한 단계 예측이지 개입 조건의 폐루프 궤적이 아니다. 학습·실제 조작 0. 다음에는 포트 포화를 회피하도록 신호 정규화와 가중치 학습을 기존 단일 회로에서 분리 실험하고, 독립 코너·속도 조건의 폐루프/생리 인과 대조를 해야 한다. 미종료 MMF 문제와 실제 카메라 재개는 미해결이다.


## 2026-11-02 — 포화 페널티를 둔 단일 회로 시각 경계 합성 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 기존 단일 회로와 조향 중립 기준을 고정하고 공간 시각 projection만 Adam 8회·lr .001·원래 값 대비 최대 .015로 학습했다. 합성 굽이×몸 yaw 9조건의 마지막 5프레임 평균 조향에 공학 가설 `0.02*bend - 0.1*yaw`를 적용하고, projection의 `mean(relu(abs(pre_tanh)-1)^2)`에 0.00005를 곱해 손실에 더했다. 초반 36프레임은 수치 재귀 상태를 유지하면서 gradient만 detach했다. 합성 교사 가설은 측정된 조향 정답이나 생리학 법칙이 아니다.

학습 손실은 첫 업데이트 전 0.003663806에서 8번째 업데이트 전 0.003662951로 소폭 감소했다. 이전에 이미 검토한 별도 합성 평가 881302에서 포화 페널티 0.00130347→0.00130557, 가설 MSE 0.003552026→0.003552214로 개선되지 않았다. 9조건 전부 조향 부호가 음수였다. 이 평가 자료와 학습 자료는 이전 실험에서 사용됐고 독립적 일반화나 폐루프 성능을 의미하지 않는다. 학습 자료의 고정 yaw 자극과 이전 폐루프 포화 추적 이력은 분포가 다르며 둘의 포화 수치를 직접 비교하지 않는다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-saturation-aware-fit-plan-1/plan.json`; 원본 `.../20261102-saturation-aware-fit-1/`의 result·candidate·audit 및 소스 `scripts/train_saturation_aware_spatial.py`. candidate SHA-256 `44379c9e4f69a04431a81f66df1ab45bfcc2b973884d5377d95432562735e23a`, 가중치 최대 변화 0.00694로 제한 내, 조향 기준 불변을 확인했다. 해시·가중치 감사는 신경망의 독립 재생이나 실제 생리/차량 검증이 아니다. 후보 미인증·미배포, 실제 데이터 fit·차량 제어 0. MMF 요청 미종료 상태도 그대로다.

다음에는 포화가 실제 관찰된 폐루프 이력에 맞는 합성 학습 분포와 분리된 새 평가를 마련하고, 중립 편향과 좌우 반응을 분리해 원인 경로의 인과적 기여를 확인해야 한다.


## 2026-11-02 — 실제 폐루프 분포를 반영한 합성 이력 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 이전 gain1 합성 폐루프의 네 시야·FlyVis·몸 상태를 texture 882401/882503 각각 직선·좌우 81결정으로 재구성했다. 원본 폐루프 trajectory와 최대 차 2.39e-7, 배열 해시 확인. 이력 재구성 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-closed-loop-scene-reconstruction-1/`, `scripts/record_synthetic_closed_loop_scenes.py`.

단일 회로·몸·운동 시각·속도 경계 및 과거 비시각 중립 기준을 고정하고 공간 projection만 Adam 6회, TBPTT 9결정, lr .002, 변화 최대 .02, 포화 페널티 가중치 .00005로 학습했다. 882401 학습, 882503 고정 이력 평가. 둘 다 이전에 본 질감이므로 미사용 독립 평가가 아니다. 합성 교사는 별도 ground truth 위치·yaw로 3m 선행 곡률, 횡편차·방향오차를 계산해 가상 bicycle 조향 목표를 주는 공학 가설이다. 위치와 정답 중심선은 actor 입력이 아닌 목표/평가에만 사용했다. 실제 측정 레이싱 라인이나 생리 정답이 아니다.

첫 실행 `20261102-closed-loop-history-fit-1`은 모듈이 아닌 Parameter에 hook을 달아 업데이트 전 실패했고 보존했다. 수정 후 `.../20261102-closed-loop-history-fit-2/`는 6회 완료, 학습 MSE 0.1700413→마지막 업데이트 전 0.1700389로 미미하게 감소했다. 동일 교사 목표를 유지한 882503 평가의 좌/직선/우 MSE는 0.241493/0.000781/0.270522에서 0.241489/0.000781/0.270518로 사실상 불변이다. 좌우 목표 평균 -0.4433/+0.4709, 후보 출력 평균 -0.01349/+0.00530이었다. 포화 페널티는 좌 0.111154→0.111202, 우 0.148386→0.148710으로 소폭 증가했다. baseline은 기록된 각 조향을 정확히 재현했다.

계획 `.../20261102-closed-loop-history-fit-plan-2/plan.json`, 코드 `scripts/train_synthetic_closed_loop_histories.py`, 결과 `result.json`, 원본 평가 `evaluation.json`, 감사 `audit.json`, 후보 `candidate.pt` SHA-256 `aba09ecfeb36689afd261356aed48ca798c486cf283a39a89bb713161bb1a124`. 감사는 입력·계획·결과 해시와 가중치 최대 변화 0.01179, 평가 목표 및 baseline 재현을 확인했고 학습이나 신경망을 독립 재실행하지 않았다. 새 후보는 미인증·미배포, 실제 데이터 fit·차량 제어 0이다. 고정 이력에서의 미약한 학습을 실제 폐루프·생리·빠른 완주로 해석하지 않는다. 미종료 MMF와 실제 영상 검증도 여전히 미해결이다.

다음은 gradient가 시각 경계에서 조향 근육까지 얼마나 전달되는지 및 고정 회로의 출력 범위를 정량화하고, 공학 교사의 부호·크기와 물리 대응을 따로 검증한다. 편향 보정만으로 목표 달성을 선언하지 않는다.


## 2026-11-02 — 단일 회로 시각 경계·조향 readout 이득 합성 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 앞선 고정 합성 이력 882401의 동일 공학 교사·3조건을 사용하고, 단일 회로의 공간 시각 projection과 기존 근육 차이 조향 readout의 양의 gain만 학습했다. 회로·몸·운동 시각·속도 경계와 과거 중립 기준은 고정했다. 이는 별도 운동 회로가 아닌 공학 readout의 민감도 조정이며 측정 생리가 아니다.

12회 Adam, projection lr .002·최대 변화 .02, log gain lr .15·최대 변화 ±2, TBPTT 9결정. 학습 이력 MSE는 첫 업데이트 전 0.17004에서 12번째 전 0.14683으로 감소했고 최종 gain은 원래의 약 6.284배였다. 이미 본 다른 질감 882503의 고정 이력에서 좌/직선/우 MSE는 0.24149/0.000781/0.27052→0.18714/0.002493/0.24254: 좌우 교사 추종은 개선되고 직선은 악화됐다. 교사는 합성 위치·방향으로 계산한 공학적 값이며 actor에 지도·위치를 입력하지 않았다.

이어 같은 두 질감·3굽이에서 기존/후보 12개 합성 폐루프를 실행했다(시작 x8m·3m/s, 81결정×40ms). 모두 정해진 단계로 종료했고 x=16.5m의 선형 보간 오차는 다음과 같다.

| 조건 | 굽이 -1 (m) | 직선 (m) | 굽이 +1 (m) |
|---|---|---|---|
| 기존 882401 / 882503 | +0.291 / +0.294 | -0.025 / -0.027 | -0.347 / -0.349 |
| 후보 882401 / 882503 | +0.006 / +0.017 | -0.111 / -0.130 | -0.270 / -0.284 |

속도는 모두 해당 위치 약 3.35m/s, 최대 조향 각도 가정은 기존 약 .0121rad, 후보 약 .0746rad였다. 한쪽 코너는 큰 개선이지만 반대 코너 오차가 남고 직선은 나빠졌다. 과거 기준은 다른 비시각 조건에서 얻었으며 두 질감 모두 이전에 검토한 합성 환경이다. 공학적 무슬립 차/액추에이터, 저속·짧은 구간, 단일 곡률에서의 진단으로 실제 생리·타이어/하중·셋업·변속·실제 AC·유효랩·랩타임을 검증하지 않는다. 후보 미인증·미배포, 실제 차량 조작 0.

학습 계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-joint-turn-gain-fit-plan-1/plan.json`, 결과 `.../20261102-joint-turn-gain-fit-1/`, 코드 `scripts/train_joint_turn_gain_histories.py`, 후보 SHA-256 `d11e1c50abae531ba2bd3c676f1b962510b1254de0aef04ad742cabde9e01171`. 합성 폐루프 계획 `.../20261102-joint-turn-gain-loop-plan-1/plan.json`, 결과·원본 궤적·감사 `.../20261102-joint-turn-gain-loop-1/`, 코드 `scripts/evaluate_joint_turn_gain_closed_loop.py`. 감사는 파일·후보 해시, 이전 baseline 재현, 유한값·상태 연결·yaw 되먹임을 확인했으며 신경망이나 물리를 독립 재현하지 않았다. 이전 학습 계획을 초기화에 읽는 코드가 있어도 새 폐루프 계획과 구별한다.

다음에는 직선 중립과 좌우 반응을 동시에 만족하도록 학습 목표·중립 검증을 개선하고 새로운 곡률/속도/질감에서 감각 인과 대조 및 길어진 폐루프를 평가한다. MMF 요청의 terminal과 실제 센서/주행은 여전히 미확인이다.


## 2026-11-02 — 직선 중립 비중을 높인 단일 회로 학습과 곡률 대조

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 이전 시각 projection·조향 gain 후보를 출발점으로 기존 단일 회로의 이 두 값과 lateral reference만 8회 학습했다. 직선 교사 손실 가중치 5, reference 변화 최대 3e-5, log gain 변화 최대 .25, projection 최대 변화 .02, TBPTT 9다. 직선 교사는 고정 이력에서 위치·heading 편차를 복구하는 합성 공학 목표이며 배우는 위치·정답 중심선을 받지 않는다.

앞서 본 평가 질감 882503의 고정 이력에서 기존→새 후보 MSE는 굽이 -1 0.1871→0.2203, 직선 0.00249→0.00116, 굽이 +1 0.2425→0.1922였다. gain은 이전 후보의 약 1.159배다. 같은 이전 질감 둘의 합성 폐루프 x=16.5m 중심 오차는 기존 후보의 굽이 -1 +0.006/+0.017, 직선 -0.111/-0.130, 굽이 +1 -0.270/-0.284m에서 새 후보 +0.186/+0.204, +0.027/+0.023, -0.035/-0.050m로 바뀌었다. 12조건 모두 81단계 제한으로 끝났고 baseline은 직전 폐루프를 재현했다.

추가 texture 882617, 곡률 계수 0.012/0.024, 직선·양쪽 굽이 12조건에서도 같은 위치 오차를 비교했다. 직선은 기존 -0.105m, 새 +0.027m였다. 곡률 0.012에서 굽이 -1/+1은 기존 -0.009/-0.174m, 새 +0.121/+0.051m; 곡률 0.024에서는 기존 +0.114/-0.380m, 새 +0.298/-0.144m였다. 직선과 +1의 개선에 반대쪽 악화가 동반되며 안정적인 양방향 코너 추종을 증명하지 못한다. 새 texture 번호는 코드·문서 텍스트에서 이전 사용을 찾지 못했지만 모든 과거 데이터의 전역 미사용은 입증하지 않았다.

학습 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-straight-balanced-fit-1/`, 계획 `.../20261102-straight-balanced-fit-plan-1/plan.json`, 코드 `scripts/train_straight_balanced_histories.py`, candidate SHA-256 `ecf48a3d160f1b18dcb6e9e06b5ff1b239926e1c5e837c3d13573c1a52fea5c3`. 최초 폐루프 `.../20261102-straight-balanced-loop-1/` 및 `scripts/evaluate_straight_balanced_closed_loop.py`; 추가 곡률 계획 `.../20261102-straight-balanced-varied-plan-1/plan.json`, 결과·궤적·감사 `.../20261102-straight-balanced-varied-1/`, 코드 `scripts/evaluate_straight_balanced_varied_curvature.py`. 해시·유한값·상태 시간 연결·곡률별 평가 중심·yaw 되먹임을 확인했으나 신경망/물리 독립 재현은 아니다.

이는 낮은 속도와 짧은 무슬립 합성 경로의 공학적 시험으로, 실측 생리·실제 AC 제어, 바람·하중·타이어/셋업, 변속·완주·랩타임은 미검증이다. 실제 데이터 학습과 차량 조작 0, 후보 미인증·미배포, 기존 MMF 요청 terminal 미확인. 다음은 한쪽에 공통 상수를 더하는 중립 이동과 좌우 시각 차동 반응을 분리해 학습하고, 다른 속도와 긴 구간에서 감각 인과·양방향 응답을 확인한다.


## 2026-11-02 — 좌우 차동 목표로 단일 회로 경계 학습

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 직전 직선 비중 후보에서 시작해 동일 합성 이력의 좌우 교사 출력 차이를 별도 목표로 포함했다. 각 업데이트 전 좌우 출력을 구하고 차동 잔차를 고정해 양쪽 재생에 부호가 반대인 기울기를 전달한다. 별도 작은 tensor 산술 검사에서 이 방식과 직접 차동 MSE의 출력 기울기는 일치했으나 전체 재귀 학습의 독립 재현은 아니다.

기존 단일 회로의 projection·조향 gain·lateral reference만 8회, TBPTT 9결정, 직선 손실 2배·차동 손실 2배로 학습했다. 원본 이력은 이전에 사용한 합성 texture 882401, 평가도 이전에 본 882503이며 교사는 합성 도로 기하 가설이다. 개별 상태 기준 학습 MSE는 0.136235→마지막 업데이트 전 0.127312, 평가의 굽이 -1/직선/+1 MSE는 기존 후보 0.22028/0.001157/0.19222에서 새 후보 0.19075/0.001845/0.19185다. 출력 gain은 직전 후보의 약 1.293배다.

이전 합성 질감 882617, 곡률 .012/.024에서 기존과 후보를 각각 3가지 굽이로 폐루프 비교했다. 같은 x=16.5m 기준 곡률 .012의 굽이 -1/직선/+1 오차는 기존 +0.121/+0.027/+0.051m, 후보 +0.044/+0.007/+0.025m였다. 곡률 .024는 기존 +0.298/+0.027/-0.144m, 후보 +0.158/+0.007/-0.170m였다. 굽이 +1의 급한 코너는 악화됐다. 모두 81단계 제한 종료이며 낮은 속도·짧은 무슬립 합성 코스다.

**감사 제한:** 12개 궤적의 유한값·40ms 상태 연결·중심선 계산·yaw 피드백·해시는 확인했으나, 이전 폐루프의 동일 baseline 재현을 2e-5 이하로 요구한 엄격 검사는 직선 2조건에서 실패했다. 직선 조향 intent 최대 차 약 0.0002703, 횡위치 최대 차 약 0.0000249m, 원인은 미확인이다. 코너 baseline은 최대 약 2.23e-6 차이다. 사후에 기준을 완화해 통과 처리하지 않았고 `audit.json`에 partial_baseline_replay_mismatch로 기록했다. 이 결과를 독립적 성능 증명이나 실제 주행으로 승격하지 않는다.

학습 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-paired-direction-fit-1/`, 계획 `.../20261102-paired-direction-fit-plan-1/plan.json`, 코드 `scripts/train_paired_direction_histories.py`, 후보 SHA-256 `a3fa71813b803b5371900708fff9423f9b08fe0c3c107127b8e6a8bfe9e13a48`. 합성 폐루프 `.../20261102-paired-direction-loop-1/`, 계획 `.../20261102-paired-direction-loop-plan-1/plan.json`, 코드 `scripts/evaluate_paired_direction_varied_curvature.py`. 실측 생리, 실제 AC, 바람·하중·타이어·셋업·변속, 유효랩과 랩타임은 미검증이며 실제 데이터 fit·차량 조작 0, 후보 미인증·미배포다. MMF 미종료 요청은 그대로다. 다음은 baseline 차이 원인 및 새로운 조건에서의 재현성, 코너 +1 악화와 직선 안정성의 동시 개선이다.


## 2026-11-02 — 직선 재현 차이의 단일 영상 반사실

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 직전의 동일 baseline 합성 직선 이력 둘에서 조향 intent 재현 차이가 2e-5 기준을 넘은 원인을 추적했다. 초기 미세 차가 난 후 64번 결정에서 두 합성 자세의 좌표/방향 수치 최대 차는 약 9.13e-8이었으나 도로 경계 RGB 한 픽셀이 최대 채널 142만큼 달라졌다. 원래 겹눈 투영으로 재계산하면 수용기 하나가 약 0.09248 차이 났다. 원본 포즈·렌더러·camera 설정에서 `scripts/audit_raster_receptor_difference.py`가 재현하며 `raster-audit.json`에 각 시점과 소스 해시를 저장했다.

동일 기존 몸 이력·정상 영상의 재귀 재생을 기준으로, 64번 영상 한 프레임만 다른 이력의 포즈에서 만든 영상으로 치환했다. 같은 단일 회로·FlyVis 상태를 이어간 후 65~69번 intent 차는 약 0.0000114/0.0000835/0.0001797/0.0002584/0.0002951이었고, 원래 두 폐루프에서 같은 구간 차는 약 0.0000117/0.0000835/0.0001797/0.0002528/0.0002703이었다. baseline 고정 이력 재생의 최대 intent 차는 6.66e-7, 치환 전 두 재생의 최대 차도 6.66e-7이다. 단일 영상 치환의 인과 효과가 해당 합성 재현 불일치의 주요 가능성을 지지하지만, 두 폐루프의 모든 차이나 초기 미세 차의 원인을 증명하지 않는다.

원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-single-frame-raster-audit-1/`의 `result.json`, `trajectory.npz`, `analysis.json`, `raster-audit.json`, 코드 `scripts/audit_single_frame_raster_divergence.py`, `scripts/audit_raster_receptor_difference.py`. 고정 이력의 next state는 폐루프 결과가 아니다. 엄격 baseline 재현 실패를 유지하며 사후 허용 오차 확대는 하지 않았다. 합성 래스터 경계와 공학 겹눈 투영의 반응이며 실제 노출, 생리, AC 주행, 속도·변속·셋업·유효랩은 검증하지 않았다. 실제 조작과 학습 0, 미종료 MMF 요청 미해결. 다음에는 합성 도로의 aliasing을 줄인 별도 렌더러와 원본 렌더러를 고정 장면·물리 조건에서 비교하고 양방향 코너 후보를 재평가한다.


## 2026-11-02 — 합성 도로 부분 픽셀 평균과 후보 폐루프

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 과거 renderer는 보존하고 별도 `src/flybrain_drive/synthetic_antialiased_planar_road_visual.py`에 화면 픽셀당 1×1·2×2·4×4 광선 평균을 추가했다. 1×1은 과거 renderer와 동일하고 2×2는 180×320 uint8 네 시야를 출력함을 관련 테스트 4개로 확인했다. 픽셀 평균은 합성 영상의 공학 처리이며 초파리 겹눈의 생리 모델이 아니다.

이전 baseline 차이의 고정 포즈 60~71번에서 1/2/4배 표본을 비교한 결과, 문제의 64번에서 과거 수용기 최대 차 0.09248은 2×2와 4×4에서 0이었다. 60~71의 최대 수용기 차는 각각 0.09248/0.02626/0.00496이고 합계 L1은 0.11817/0.06301/0.04600이었다. 다른 프레임에도 픽셀 차이는 있어 완전 연속성을 보장하지 않는다. `.../20261102-supersampling-pose-audit-1/`, `scripts/audit_supersampling_pose_sensitivity.py`.

같은 고정 몸 이력에서 64번 한 프레임만 치환할 때 과거 renderer의 이후 조향 intent 최대 차는 0.000295였고 2×2 renderer에서는 0이었다. 치환 전 두 계산은 최대 1.33e-6 차였으며 전체 영상도 새로운 renderer로 바뀌었다. `.../20261102-antialiased-frame-swap-1/`, `scripts/audit_antialiased_single_frame_swap.py`. 특정 포즈 쌍의 원인 대조이지 일반적 무잡음 또는 이전 엄격 재현 실패 해결 증거가 아니다.

2×2 renderer, 기존 직선 학습 후보와 좌우 차동 후보, 이전에 본 질감 882617, 곡률 .024의 직선·양쪽 굽이를 합성 폐루프로 실행했다. x=16.5m 선형 보간 중심 오차는 기존 +0.283/+0.048/-0.140m, 차동 후보 +0.127/-0.045/-0.168m(굽이 -1/직선/+1)였다. 양쪽 81단계 제한, 속도 약 3.35m/s였다. 새로운 렌더링에서도 굽이 -1은 개선됐으나 +1은 악화, 직선은 절대 오차가 비슷하고 부호가 달랐다. 반복 실행 재현성은 아직 검증하지 않았다.

계획 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-antialiased-candidate-plan-1/plan.json`, 원본 결과·궤적·감사 `.../20261102-antialiased-candidate-loop-1/`, 코드 `scripts/evaluate_antialiased_candidate_closed_loop.py`. 감사는 렌더러·모델·결과 해시, 6개 유한 궤적·상태 시간 연결·중심선·yaw 감각을 확인했으나 신경망/물리 재현 시험은 아니다. 시각이 달라지는 분포 변화, 합성 저속·짧은 무슬립 구간, 기존에 본 질감의 한계가 있다. 실제 생리·AC·상대풍·하중·타이어·변속·셋업·완주/랩타임은 미검증이며 실제 데이터 학습·차량 조작 0, 후보 미배포다. MMF 미종료 요청도 그대로다. 다음은 동일 2×2 폐루프의 반복 재현과 새로운 조건에서 좌우 코너·시각 인과를 검증한다.


## 2026-11-02 — 부분 픽셀 평균 폐루프 반복과 4m/s 조건

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 동일 2×2 렌더러·texture 882617·곡률 .024·초기 3m/s의 기존/차동 후보 각 직선·좌우 굽이 6조건을 별도 프로세스에서 반복했다. 전체 trajectory 열의 최대 차는 조건별 2.28e-7~1.99e-6, 사전의 엄격 2e-5 기준을 모두 통과했다. 한 번 반복된 합성 조건에 한정하며 기존 1×1 재현 실패를 고치거나 모든 환경의 결정성을 증명하지 않는다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-antialiased-candidate-repeat-1/`, `repeat-audit.json`, 코드 `scripts/evaluate_antialiased_candidate_closed_loop.py`.

다른 synthetic texture 882723, 시작 x=6m·4m/s, 곡률 계수 .024, 2×2 renderer에서 같은 두 후보의 6조건을 평가했다. 저장소 코드·문서 텍스트에서 이 texture 번호의 이전 사용은 찾지 못했으나 과거 전체 데이터에서 미사용이라고 보증하지 않는다. 모든 조건 81단계×40ms로 종료했다. x=16.5m 중심 오차는 기존 직전 후보 굽이 -1/직선/+1 각각 +0.320/+0.081/-0.103m, 차동 후보 +0.135/-0.044/-0.165m다. 해당 위치 속도 약 4.06m/s, 종료 속도 약 4.11m/s다. 차동 후보는 직선과 -1은 개선, +1은 악화됐다. 새 속도에서 양방향 안정적 주행이나 랩타임 개선은 증명되지 않는다.

계획 `.../20261102-antialiased-speed4-plan-1/plan.json`, 결과·궤적·감사 `.../20261102-antialiased-speed4-loop-1/`, 코드 `scripts/evaluate_antialiased_speed4_closed_loop.py`. 감사는 모델·renderer·출력 해시, 초기 조건, 40ms 상태 연속성, 중심선 및 yaw 피드백을 확인했으나 신경망/물리를 독립 재실행하지 않았다. 경로·차량은 공학 합성 무슬립 모델이며 실제 AC, 상대풍·하중·타이어·셋업·변속, 생리 및 유효랩은 미검증이다. 실제 데이터 fit·차량 제어 0, 후보 미인증·미배포, MMF 요청 terminal 미확인이다. 다음은 양쪽 코너에서 다른 감각 신호와 조향 부족을 분리하고 다양한 속도·곡률·긴 구간 및 실제 센서 이력에서 인과 검증한다.


## 2026-11-02 — 취소 후보 설치와 실제 Lua 자동 재적재 관찰

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 동일 요청 `mmf-stream-d487677e57a14987aea819bb4f516523`, stop, 자기 terminal 없음, offline·정지·브레이크·AI/camera off인 unpaused 세션을 확인했다. repo 원본과 운영 Lua hash가 일치함을 확인하고 원본을 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-camera-cancel-candidate-install-1/installed-original.lua`에 보존했다. LuaJIT 후보·기존 lifecycle·paused gate 257검사 통과 후 `scripts/deploy_camera_cancellation_candidate.py`로 설치 직전 request/generation/정지 상태와 hash를 재확인하고 v2 후보를 원자 교체했다. 설치 hash `e5f5c189fa5fed5f0dda4be8a48c0c906599b2ef9b429aae02ae5921e041a2b2`; 원본 backup `475e085d974595aa503fa687d58913cf1551b39fb4d230721763f5fd68a8d119`. UI/lifecycle/새 camera/actuator 0.

이후 AC가 같은 엔진 PID 34992에서 Lua 앱을 자동 재적재했고 generation이 `19007968ULL:0.003981`→`19007968ULL:0.0016764`로 바뀌었다. 새 producer는 이전 요청 폴더에 `orphan-observation.json`을 생성해 이전 generation, stop, 기존 terminal 부재, 이전 시작/프레임 미상, 이전 프로세스 종료 미검증, actor 불허를 기록했고 camera status도 같은 ID의 `prior generation unresolved; host review required`다. 새 session은 unpaused·정지·브레이크·camera/AI off로 관찰됐다. 이 실제 기록은 후보가 로드됐다는 증거이나 이전 요청의 terminal, 이전 producer의 안전한 종료 또는 새 카메라 요청 자격을 증명하지 않는다. 원래 request.json과 stop은 그대로이며 result.json은 없다. 설치 스크립트의 `result.json`은 설치 직후의 reload 미확인 시점이고, 이후 `reload-observation.json`이 새 증거다.

운영 Lua는 현재 repo 원본이 아니라 v2 후보이므로 기존 `control_ac_session.py`의 기본 해시 검사는 실패한다. 향후 검증된 lifecycle 진단에 한해 실제 후보 해시를 명시하고, 동일 요청 소유 인계와 별도 시간/프로세스 증거 없이 camera 재요청·자동 UI 대체를 하지 않는다. 이전 요청은 terminal 미확인으로 유지한다. 자동 재적재라는 새 외부 상태를 확인했으므로 다음에는 구 producer의 release/fencing을 입증하는 독립 증거와 적법한 미해결 요청 인계 계약을 구현·검증한다. 실제 학습·차량 주행·생리·완주/랩타임은 여전히 미검증이다.


## 2026-11-02 — 실제 Lua release callback 관찰

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 기존 v2를 바탕으로 자기 generation의 onRelease callback에서 camera 활성 전후 상태를 기록하고, 기존 orphan 기록은 일치할 때만 보존해 재적재를 허용하는 v3 후보를 만들었다. callback 관찰은 이전 generation이나 엔진 종료를 주장하지 않고 actor 승인도 주지 않는다. 후보 `integrations/ac_csp/candidates/FlyBrainCamera-release-witness-v3.lua`, 생성기 `scripts/build_camera_release_witness_candidate.py`, `tests/test_camera_release_witness.py` 포함 Lua·lifecycle·gate 261검사 통과.

기존 v2 파일을 보존하고 동일 미해결 요청·generation, 세션 정지·브레이크·AI/camera off 및 설치 파일 hash를 재검사한 뒤 v3를 설치했다. 설치 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-camera-release-install-1/installed-v2.lua`, 설치 결과 `result.json`, 코드 `scripts/deploy_camera_release_witness.py`, 현재 v3 SHA-256 `196469d68e2849bceba1489dee336ea2839677057c38c0d66934bd466ae926f2`. 실제 자동 재적재 뒤 현재 generation의 camera status는 여전히 이전 요청 미확인으로 나타났다.

그 다음 `scripts/reload_camera_witness_once.py`가 동일 v3 바이트만 원자 교체해 한 번 재적재하고 원래 엔진 process handle을 유지하며 확인했다. `.../20261102-camera-release-reload-1/result.json`, `after.json`: generation `19007968ULL:0.0016956`→`19007968ULL:0.0016075`, 직전 generation의 `C:/Users/jeff4/AppData/Local/flybrain/csp-camera/release-19007968ULL_0_0016956.json`에 active_before=false·active_after=false·prior_generation_resolved=false·engine_exit_verified=false가 실제 기록됐다. 동일 엔진 PID 34992 handle은 미종료다. 기존 orphan 기록과 stop·원래 request를 보존했고 원래 result.json은 여전히 없다. UI·lifecycle·새 camera·actuator 요청 0.

이는 새로 계측된 v3 인스턴스의 callback이 실제로 실행된 증거일 뿐, 이전 최초 producer의 release나 미종료 요청이 완료됐다는 소급 증거가 아니다. 또한 새 camera 요청을 허용하는 host 소유 인계는 미구현이다. 현재 운영 Lua는 v3이고 이전 문서의 v2 설치 상태는 당시 기록이다. 다음은 구 generation의 활동 가능성과 원본 보존을 다루는 명시적 host 인계 계약을 검증해야 하며, 새로운 카메라 요청 전 기존 결과 미확인 상태를 명시적으로 유지한다. 실제 영상 학습·생리·차량 제어·유효 완주/랩타임은 미검증이다.


## 2026-11-02 — 미종료 camera 요청 보존·격리와 엔진 프로세스 인계

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 과거 file shadow 3개는 각각 오류 결과, 자기 camera terminal stopped·worker 종료·보관본 hash를 보존했으나 child PID 37108/46632/25836의 Windows handle은 종료 요청 상태로 미신호였다. `scripts/fence_finished_shadow_processes.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-finished-shadow-fence-1/`에서 각 정확한 프로세스 신원, 결과·archive hash 및 종료 요청을 재확인한 뒤 종료 요청을 완료시켜 세 handle의 신호를 확인했다. 관련 부모도 사라졌고 어떤 원본 파일도 지우지 않았다. 이들은 미종료 MMF 요청의 terminal을 의미하지 않는다.

기존 MMF 원본 request ID `mmf-stream-d487677e57a14987aea819bb4f516523`의 stop·orphan 관찰·원래 terminal 부재, 현재 v3 producer idle 및 엔진 handle·다른 shadow 소비자 부재를 확인했다. `scripts/quarantine_unresolved_camera_request.py`, `.../20261102-camera-request-quarantine-1/`에서 `request.json`을 같은 요청 폴더의 `quarantined-request.json`으로 원자 rename, 원본 SHA-256 `c3459cc7201d77eaf70bc29462becee3321a7fd1837c88798c3333771ef37af5` 보존. 같은 generation heartbeat가 진행하는 동안 request 게시 슬롯 비어 있음과 camera 비활성을 확인했다. 격리는 재접수 방지이며 원 요청 결과는 여전히 미상, terminal을 위조하거나 새 camera 자격을 부여하지 않는다.

`control_ac_session.py`가 v3 hash를 명시해 pause 1회 성공(`.../20261102-quarantine-pause-1/`). 첫 엔진 restart 사전 검사는 기본 조작 프로필 hash 불일치로 요청 0회·실패(`.../20261102-quarantine-engine-restart-1/`); 현재 profile은 기존 진단 프로필 `.../20261101-steering-contract-read-only-audit/diagnostic-profile-candidate.json`의 hash와 일치해, 설정을 변경하지 않고 해당 진단 profile을 명시했다. `scripts/restart_ac_after_camera_quarantine.py`, `.../20261102-quarantine-engine-restart-2/`에서 CSP restart-engine 1회, 원래 PID 34992의 동일 OS process handle 종료 신호 및 새 PID 38464·새 file publisher session과 generation `12717214ULL:0.0033783`을 확인했다. publisher 파일은 교체하지 않았다.

새 엔진 메뉴에서 CSP start 1회 성공(`.../20261102-quarantine-new-engine-start-1/`). start 후 brake0이라 기존 `scripts/restore_stationary_brake_hold.py`와 진단 profile로 정차·AI off·중립·가속0에서 full brake를 복원하고 native sensor로 확인한 후 vJoy 해제(`.../20261102-new-engine-brake-restore-1/`, actuator submit 2회·UI0). 현재 새 트랙 세션은 unpaused·offline·camera/AI off·brake1, camera status idle, camera request 게시 슬롯 비어 있다. 과거 요청의 result.json은 여전히 없고 orphan/stop/quarantined request를 보존했다.

다음에는 이전 OS process 종료 증거·격리된 원본·새 generation/소비자 부재를 결합한 명시적 host 인계 검증을 구현하고, 미종료 결과를 terminal로 위장하지 않은 상태에서 제한된 새 MMF 카메라 인가와 paused-start/resume을 검증해야 한다. 현재 새 카메라 요청·실제 자율주행 학습/제어는 아직 수행하지 않았다. 원래 생리·바람·하중·타이어·변속·셋업·완주/랩타임 검증은 미완이다.


## 2026-11-02 — 이전 producer 종료 인계 후 새 MMF 정지 진단

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. `src/flybrain_drive/camera_orphan_fence.py`의 순수 검사는 격리된 이전 request·orphan 일치, 이전 엔진의 실제 종료 신호, 직전 producer의 inactive release, 새로운 engine/generation·camera idle을 요구한다. 이전 request terminal은 unknown으로 유지하고 actor 자격이나 camera 호출을 부여하지 않는다. `tests/test_camera_orphan_fence.py`와 paused gate 총 31검사 통과.

`scripts/audit_camera_orphan_fence.py`, `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261102-orphan-fence-audit-1/`에서 원래 request hash, 이전 engine handle 종료 기록, 현재 PID 38464·generation `12717214ULL:0.0033783`, 신선한 camera idle·native AI off/정지/brake1을 대조했다. 옛 요청 `mmf-stream-d487677e57a14987aea819bb4f516523`의 결과는 여전히 미상이다. 이어 CSP pause 1회(`.../20261102-orphan-fence-probe-pause-1/`) 후 `scripts/probe_mmf_after_orphan_fence.py`가 원본 fence evidence·격리 hash와 기존 terminal 부재, 새 generation·paused·정지/brake 및 camera 빈 슬롯을 다시 검사한 다음 새 MMF 요청 `mmf-probe-073992947a7e4c9db7456f7f5e461180`을 1회 보냈다.

`.../20261102-orphan-fenced-mmf-probe-1/result.json`: 정지 8묶음 × 전후좌우 32장, shared texture RGB와 producer의 같은 순간 PNG 최대 차 0, producer의 해당 새 ID terminal completed·8프레임, camera inactive, actor_stream_created=false·training_admissible=false다. `independent-pixel-audit.json`은 저장된 NPY/PNG 32쌍을 독립 재계산해 최대 차 0을 확인했고 옛 terminal 부재도 확인했다. 이 결과는 정지 캘리브레이션으로, 물리 노출 시각 동기·실시간 stream·생리·움직임·주행 학습에 적합함을 증명하지 않는다. UI·조향/가감속 제어·optimizer 0.

CSP resume 1회(`.../20261102-orphan-fenced-mmf-probe-resume-1/`) 후 동일 새 generation unpaused·offline·정지·brake1·camera/AI off를 확인했다. camera 게시 슬롯은 새 probe ID를 가리키고 자기 terminal completed다. 이전 요청의 격리 파일·stop·orphan은 보존하며 terminal을 위조하지 않았다. 다음은 새 probe terminal을 이전 요청으로 검증하고 기존 paused-start/resume gate를 실제 MMF shadow에 연결해 실시간 시각·몸 감각의 신선도와 독립 원본을 확인하는 것이다. 실제 AC 자율주행과 가감속·변속·셋업·완주/랩타임은 여전히 미완이다.


## 2026-11-02 — 소유 MMF paused-start/resume 실제 진단과 복사 실패

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 정지 MMF probe 성공 후 확인한 구조적 공백: 기존 Lua는 카메라 active 중 일반 resume을 거부했다. `scripts/build_owned_mmf_resume_candidate.py`의 v4 후보는 명시적인 owned_mmf_resume·정확한 활성 MMF ID·첫 수신 프레임·신선도·잔여 시간/프레임·stop 부재를 모두 요구할 때만 CSP resume을 허용한다. 다른 명령·ID·AI·속도·brake·온라인 등은 계속 거부한다. `tests/test_owned_mmf_resume_candidate.py` 및 기존 lifecycle 포함 282검사 통과. 실제 v3를 보존하고 `scripts/deploy_owned_mmf_resume_candidate.py`, `.../20261102-owned-mmf-resume-install-1/`에서 v4 설치, AC가 자동 재적재해 generation `12717214ULL:0.0017241`에서 owned_mmf_resume capability와 v3 release witness를 확인했다. 현재 운영 Lua SHA-256 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`.

Python recorder에 선택적인 정확한 소문자 SHA-256 pin을 추가하고 기본 원본 hash 검사는 유지했다. paused gate가 camera status뿐 아니라 session의 camera_request_id까지 같아야 함을 보강했다. 이 관련 84검사 통과. `scripts/probe_owned_mmf_resume.py`는 기존 probe의 terminal, paused·정지/brake·AI off, 카메라 소유, 원래 몸 packet을 확인하고 소유 ID로 lifecycle resume을 1회만 보낸다. 두 시도 모두 CSP api_returned_true와 동일 active camera·재개 후 몸 packet 증가를 확인했다(1차 139919→139922, 2차 164476→164480).

첫 `.../20261102-owned-mmf-resume-probe-1/`: 대기 timeout을 허용 상한 40ms보다 큰 80ms로 잘못 지정해 영상 수집 전에 실패. camera `mmf-stream-0fbae3a952d4417ab498a532709d9572`는 자기 ID stopped·7프레임; 당시 unpaused에서 source timing flush 조건이 성립하지 않아 camera close timing 오류도 기록됐다. 이후 별도 pause에서 원본 ledger가 기록됐고 validator상 complete·1083 records·SHA-256 `abf3419873657eda472e45f0b4dfab47fc1a45ab9f103c6c6d2d04f7bfe6a054`임을 추가 확인했지만 원본 실패를 성공으로 변경하지 않았다.

둘째 `.../20261102-owned-mmf-resume-probe-2/`: 대기를 40ms로 고치고 source timing 선택을 끈 별도 시도. 재개 후 camera 그룹 capture 5/6/7과 원래 몸 packet을 관찰했지만 공유 네 시야 복사 40ms deadline을 넘겨 실패; 새 요청 `mmf-stream-418ef26ac6224653a53feb4d27458c2e`의 producer terminal stopped·9프레임, consumer는 8프레임·worker 종료, 부분 그룹은 ACK하지 않았다. 원본 `.../csp-camera/client-recordings/mmf-stream-418ef26ac6224653a53feb4d27458c2e/shared-copy-failure.json`과 `timing-audit.json`: 완료 그룹 8회의 host 구간 1.74~35.52ms, 실패 그룹 46.01ms, 그중 첫 view map 성공→unmap 30.85ms. 이는 host 경과 시간이지 GPU·복사·OS 스케줄러 중 어느 원인인지 증명하지 않는다.

두 진단 모두 실패로 보존하고 타임아웃이나 freshness 기준을 완화하지 않았다. UI·조향/가감속·optimizer 0, actor/실제 주행 미적용, 실제 물리 노출 시각·생리·stream 지속성 미검증. 현재 session은 동일 generation unpaused·정지·brake1·camera/AI off, 둘째 request의 자기 stopped terminal 확인. 원래 옛 MMF request는 격리 상태로 terminal 미상 유지. 다음에는 map→unmap 내부 작업과 스케줄 대기를 분리 계측해 제한 안에 복사를 안정화한 뒤, paused-start/resume을 통합 단일 회로 shadow에 연결하고 실제 영상·몸 감각 지속성과 인과 기여를 검증한다.

## 2026-11-02 — 실시간 네 시야 지연과 저장 버퍼 진단

- `20261102-owned-mmf-resume-probe-3`: 첫 map 이전 복사 준비 구간에서 40ms 초과. 기록만으로 드라이버와 OS 스케줄링을 구분할 수 없다.
- `20261102-mmf-short-switch-probe-1`: Python 전환 간격 1ms, 40ms 제한 유지. 28묶음 수신 뒤 8개 저장 큐 포화로 실패, 28개 모두 저장 완료. 저장 지연 중앙 5.98ms, p95 151.97ms. 요청 `mmf-stream-d79337689d234402af5fa8631e90dd94` terminal stopped 29.
- `20261102-mmf-buffered-probe-1`: 기존 등록 프로파일 최대 32묶음/32MiB, 같은 1ms 전환과 40ms 제한. 3묶음 수신 뒤 GPU 그룹 복사 제한 초과로 실패. 첫 시야 RGB 변환 구간 경과 31.74ms, flush까지 약 14.98ms. 운영체제·드라이버·변환 연산의 기여는 구분할 수 없다. 저장 완료 3, 실패 0, 최대 큐 2. `mmf-stream-db6c620847bb44b79044fb7444320635` terminal stopped 4, 소비 스레드 종료. 본 시험에서 완전 archive 검증 성공을 주장하지 않는다.
- 원본: `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-mmf-{short-switch,buffered}-probe-1/` 및 `%LOCALAPPDATA%/flybrain/csp-camera/client-recordings/<요청 ID>/`. 계측 변경 관련 회귀 95개는 통과했다. 이후 기본 Python 환경에서 pytest 모듈을 찾지 못해 버퍼 진단 추가 회귀는 실행하지 못했고 새 스크립트 구문 및 import는 확인했다.
- 현재 확인된 세션은 generation `12717214ULL:0.0017241`, pause 해제, 정지·brake 1·카메라/AI off. 배포 v4 SHA256 `78863113df3f28f59fafdb0a4103c09c4b3deb7302873b7b5f956475404b78a6`. 격리된 과거 요청 `mmf-stream-d487677e57a14987aea819bb4f516523`의 terminal은 여전히 미상이다.
- 다음: 시간 계측을 유지한 채 픽셀 변환 및 저장 스케줄링 기여를 격리하고 40ms 안에서 연속 네 시야를 확보한 뒤 소스 시각과 몸 시각을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 검증 대상이다.

## 2026-11-02 — 재개 관찰 보완과 CPU 시계

- `20261102-mmf-cpu-trace-probe-1`: 소유 resume 성공 후 바로 읽은 관찰이 gate의 모든 조건을 충족하지 않아 실패. 원인 필드별 기록이 없어 어떤 조건인지는 미상. 카메라 `mmf-stream-e2e69f3666f64c02a109e2f28bd6d98f` 소비 3, 같은 ID terminal stopped 3, 작업자 종료. 진단 스크립트는 재개 명령을 재발행하지 않고 동일 세션·카메라·전진 패킷 관찰만 최대 1초 재확인하며 각 관찰을 보존하도록 변경했다.
- `20261102-mmf-cpu-trace-probe-2`: 위 관찰 통과, 첫 몸 packet 316215에서 316217 확인. 소비 10, 다음 네 시야 그룹 40ms 초과; 카메라 `mmf-stream-b030a841879a4b4198977bf65e7a589f` terminal stopped 11, 작업자 종료. `shared-copy-failure.json`에서 enqueue→flush 완료 약 16.0ms, 첫 map 호출 구간 약 16.1ms, RGB 변환 벽시계 약 8.55ms. 변환 전후 `thread_time`은 모두 0.15625초로 실제 시계가 15.625ms 단위로 보이므로 연산과 미스케줄링을 구분할 근거가 되지 않는다. 재현성/실시간 사용 미인정.
- 원본은 `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-mmf-cpu-trace-probe-{1,2}/`와 `%LOCALAPPDATA%/flybrain/csp-camera/client-recordings/<요청 ID>/`. 관련 검사 95개 통과. 추가 버퍼 검사 전체에서는 108 통과, 1 실패: 기존 `scripts/record_ac_engine_ai_probe.py`가 `scripts/prepare_buffered_frame_profile.py`의 과거 고정 SHA와 달라 빌드가 거부됐다. 원본 변경 또는 해시 교체로 실패를 숨기지 않았다.
- 발견된 `20261102-closed-loop-history-fit-1/result.json`은 `Parameter`에 `register_forward_hook`을 요청해 학습 업데이트 0으로 실패한 과거 결과다. 해당 결과를 실행 중이거나 학습 성공으로 취급하지 않는다.
- 현재 CSP generation `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off; 과거 격리 요청 terminal 여전히 미상. 다음에는 더 높은 해상도의 스레드 시간 또는 별도 프로세스에서 복사/스케줄링을 대조하고 엄격한 실패·원본 보존 조건으로 실제 영상 연속성을 재검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 원시 스레드 사이클 대조

- `QueryThreadCycleTime` 원시 값으로 GPU flush·map·RGB 변환 구간의 실행 사이클을 추가 기록했다. 주파수 변동·CPU 이동 때문에 밀리초로 변환하지 않는다. 시스템 사용 불가 시 오류로 종료하며 deadline과 성공 요건은 변경하지 않았다. 관련 검사 95개 통과.
- 같은 호스트의 별도 간단 대조에서 50.39ms sleep 약 156,287 cycles, 50.00ms busy loop 약 184,142,607 cycles였다. 이는 대기와 활동의 정성적 대조이고 고정 환산율이나 동일 부하의 엄격 실험이 아니다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-mmf-cycle-probe-1/`: 소유 resume 뒤 소비 7묶음, 다음 그룹에서 40ms 초과. 원본 `%LOCALAPPDATA%/flybrain/csp-camera/client-recordings/mmf-stream-dbbd61f99c724d6091fea23ab6ce8665/shared-copy-failure.json`: 첫 RGB 변환 벽시계 31.0721ms에 1,012,543 cycles; 이후 unmap 완료부터 다음 map 타임스탬프까지 약 31.86ms이고 그 사이 측정된 사이클 차는 작다. 스레드 실행이 중단된 기여를 시사하나 GIL 대기/OS 스케줄링 또는 측정 간격 경계를 확정하지 못한다. 센서 시간·운전·학습 자격은 미검증이다.
- 카메라 `mmf-stream-dbbd61f99c724d6091fea23ab6ce8665` terminal stopped 8, 소비 스레드 종료. 현재 세션은 generation `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off. 격리된 이전 orphan terminal은 여전히 미상.
- 다음: 시각 소비자와 저장·관찰 작업의 실행 경쟁을 독립 프로세스/대조 조건으로 분리해 40ms 내 다방향 입력을 확인하고 물리 노출 시각 및 몸 감각 정합을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 동일/별도 프로세스 경쟁의 합성 대조

- `scripts/diagnose_synthetic_frame_contention.py`는 랜덤 고정 320×180 RGBA 원본을 같은 `unpack_rgba_rows`로 150회 변환하고 픽셀을 매회 검사한다. 화면/차량에 접근하지 않으며 각 샘플 벽시간·스레드 사이클 및 설정·소스 해시를 보존한다. 전환 간격 1ms. 결과 `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-synthetic-frame-contention-2/measurements.json`.
- 단독 median/p95/max 0.181/0.265/0.308ms, 같은 프로세스 Python CPU 스레드 30.850/32.073/32.159ms (150 중 139회 5ms 초과), 별도 프로세스 CPU 작업 0.192/0.249/0.322ms, 디스크 쓰기·읽기·해시 스레드 0.224/0.310/0.437ms. 별도 프로세스는 진단이 소유한 유한 실행용 작업으로 종료·회수했다. 앞선 대조 1도 보존한다.
- 이 통제 조건에서 같은 Python 프로세스의 CPU 경쟁이 과거 실제 변환의 약 31ms 지연과 비슷한 크기를 재현하며, 단일 외부 CPU 작업이나 해당 합성 디스크 작업만으로는 재현하지 않았다. 네 조건의 동일 시점 실행·CPU 점유 동등성·OS 스케줄링은 통제하지 않았고, 실제 카메라의 경쟁 스레드나 GIL을 특정한 증거가 아니다. 실제 카메라 복사·타이밍·폐루프는 여전히 실패/미검증이다.
- 이 결과에 따라 다음은 원본 카메라/GPU 소유와 저장 프로세스를 분리하는 제한 진단 설계 및 종료·원본 해시·누락 검증이며, 동일 실제 카메라와 40ms 기준의 대조가 필요하다. 현재 session `12717214ULL:0.0017241` unpaused, stationary, brake 1, camera/AI off; 이전 요청 terminal stopped.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 별도 카메라 소유 프로세스 정지 실험

- `scripts/isolated_camera_diagnostic.py`는 기존 camera recorder/GPU 소유를 spawn 자식 프로세스에 두고 부모가 IPC로 상태·선택 메타데이터만 받는 진단이다. 픽셀/배열은 IPC로 전달하지 않아 actor 경로가 아니다. 모르는 IPC 응답은 연결을 봉쇄하며 카메라 요청 자동 재발행을 금지한다. 저장 writer는 여전히 카메라 프로세스 안의 스레드다. 단일 관찰의 결과이므로 모든 지연 원인이 해결됐다고 볼 수 없다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-camera-probe-1/`: paused 상태에서 1회 소유 resume, native packet 404636→404638. 이후 관찰 이벤트 76개 전부 다른 capture ID, 첫→마지막 선택 2.973초, 선택 시점의 수신 후 최대 경과 30.15ms (물리 노출 나이가 아님), 몸 packet 404639→405630. 정지·무조작, 40ms 그룹 복사 제한 유지.
- 카메라 `mmf-stream-1a12e134675f44c0be9f54aec2e97ae5`는 소비 80과 동일 ID terminal stopped 80, consumer worker 종료. 원본 570 파일 bundle SHA `cdeec88aac874a414c6722c0e067447f154c549960ededb826f38ce429692b50`, 보존/해시 확인, 80 그룹 전체 MMF·비동기 저장 회계 검증 및 별도 프로세스에서 검증 재실행. IPC 실패 봉쇄를 포함한 관련 검사 97개 통과.
- 한 번의 약 3초 정지 진단 성공이지 지속 신뢰성, 물리 노출-차체 동기, 학습 또는 자율주행 증명이 아니다. 다음: 픽셀까지 원본 ID·해시·시각을 보존한 제한 IPC를 설계하고 통합 시각/몸 회로에서 시간·누락·인과 기여를 검증한다. 장시간 반복 및 실제 주행은 별도 검증한다.
- 현재 generation `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off. 기존 격리 orphan terminal은 미상으로 보존한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 별도 프로세스 네 시야 픽셀 전달

- `src/flybrain_drive/isolated_pixel_packet.py`: 각각 320×180 RGB, 네 시야 원본 bytes 크기·순서·ID·원본 해시와 양쪽 host monotonic 시각을 확인한다. 수신 완료 시각으로 보수적 나이 제한을 다시 검사하며 결과는 변경 불가 배열이다. 픽셀 변조·누락·오래된 패킷을 거부한다. 물리 노출 시각이나 actor 입증은 아니다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-pixels-probe-1/`: actual AC stationary, 소유 resume 1회, 76개 다른 capture ID의 네 시야를 원본 해시 검사하며 IPC 수신, 첫→마지막 수신선택 2.986초. 선택 시점 최대 보수적 render-start lower-bound 나이 24.43ms이며 물리 exposure 시간이 아니다. native packet 437271→437276 후 관찰 437277→438273. 조작 없음.
- 카메라 `mmf-stream-72812efa48d640bba4b9dfec18c182bf` terminal stopped 79, consumer stopped 79; 원본 bundle SHA `fbe8486bed81137ff374e1ae607e7489ed2c51e1329fd1887198600729076540`, 563개 파일 및 79개 그룹 저장 회계 검증. 별도 검증에서 76개 이벤트의 네 시야 해시가 archive와 모두 일치 (`independent-pixel-match.json`). 관련 검사 100개 통과. 40ms 그룹 복사 제한 유지, 실험 1회, 학습에 사용하지 않았다.
- session generation `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off; 이전 격리 orphan terminal 미상. 다음은 기존 겹눈/운동 시각 및 단일 비행 유래 회로의 명시적 입력 인터페이스와 실제 픽셀 IPC를 연결하는 정지 shadow, 몸 감각 인과 순서 및 현실 노출 시각 검증이다. 실제 조향·가감속·변속·셋업 적응, 완주·라인·랩타임은 미검증이다.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 2026-11-02 — 분리 프로세스 실제 픽셀·몸 수용기 정지 shadow

- `scripts/probe_isolated_retina_body_shadow.py`는 원본 네 시야 IPC와 native physics에서 허용된 `required_native_fields`만 기존 `CSPCameraRetina`에 넣으며, 절대 heading은 이 입력에 넣지 않는다. 기록한 native prefix 전체는 독립 재생/진단용이다. 이 수용기는 공학 근사이며 FlyVis 생물학적 운동 시각망이나 전체 단일 회로를 실행하지 않는다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-retina-probe-1/`: 정지·무조작 소유 resume, 76개 영상·몸 감각 입력(첫→끝 decision 2.9865초), 총 82개 카메라 그룹 저장. 네 시야 각각 1709 수용기 모든 입력이 유효, 첫 프레임에는 시간차 이력 없음, 마지막에는 네 시야 전부 시간차 유효. 몸 감각 organ 132개 중 유효 수는 100~128개이며 누락을 보정하지 않았다. 값의 변화가 실제 운동 시각 생리나 차량 제어의 인과적 기여를 증명하지 않는다.
- 해당 `mmf-stream-25ce31d594814b6d93f01daa6682fe18` terminal stopped 82, consumer stopped, 보관 584개 파일 bundle SHA `b112a3cd01ec403f0a865d48b5ae02785c2b264ad7e9950591d7f478fab4dfd2` 및 82그룹 검사 완료. 수용기 결과 `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-retina-probe-1/sensory.npz`, 원본 몸 prefix/결정시각과 해시 `events.json`. 40ms 복사 제한 유지.
- `scripts/audit_isolated_retina_body_shadow.py`가 위 원본 bundle의 이미지 해시와 native prefix 해시를 검증해 같은 설정으로 76프레임을 재계산했다. `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-retina-replay-1/result.json`에서 RGB·시간차·유효성·나이·organ 모든 배열 최대 절대 차 0. 관련 기존 검사 61개 통과.
- 물리 노출 시각과 몸 감각 시간 정합, FlyVis 운동 시각, 단일 회로 작동과 인과 기여, 실제 조향·가감속·변속·셋업 적응 및 완주·랩타임은 미검증이다. 다음은 FlyVis 운동 시각 입력을 동일 네 시야 원본에 연결하고 반복 상태·타이밍 및 회로 진입을 검증한다. 현재 generation `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off. 오래된 격리 orphan terminal 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 2026-11-02 — 실제 정지 네 시야 FlyVis 오프라인 재생

- `scripts/replay_isolated_stationary_flyvis.py`: `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-retina-probe-1/`의 검증된 archive 이미지 76묶음과 원본 host 결정시각을 입력했다. RGB/BMP 원본 SHA, 프레임 보관 회계, FlyVis 체크포인트 및 설정 해시를 확인했다. 시야별 721 수용기는 카메라별 공학 투영이며 실측 겹눈 광학이 아니다. 원본 결정 간격을 최대 5ms 내부 tick으로 분할하고 프레임 간 이전 입력과 회로 상태를 유지했다. T4a–d/T5a–d 8계열을 기록했다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-stationary-flyvis-1/`: 76프레임, 약 2.987초의 host 시각 구간, 내부 632 tick, 네 시야의 실제 프레임과 첫 프레임 유지 대조를 동일 초기 상태·시간에 함께 실행. 실행 계산 5.07초(모델 초기화·원본 입력 처리 제외), 실시간 성공이 아니다. 응답 `responses.npz` SHA `c9b5c631377e8394111f3d01185226ad0de59a03f09bc8e1a02b6b96b64e3e00`.
- 실제 영상과 첫 프레임 대조의 망막 입력 최대 차 0.4151, T4/T5 출력 최대 절대 차 0.5548 (전체 평균 절대 차 0.0419, 첫 프레임 차 3.32e-7). 정지한 장면이라도 렌더/노출/애니메이션 등 영상 변화가 있으므로 변화 원인과 동작 방향 선택성은 이 결과만으로 알 수 없다. 이는 FlyVis 부품 실행 증거일 뿐 생리 기능 검증·차량 통합·실제 완주나 학습이 아니다.
- 다음: 별도 프로세스 실제 영상에서 FlyVis 상태 보존 실시간 처리와 몸 센서의 원본 시간 검증을 통합하고, 실제 정지 잡음 및 통제된 움직임 대조·시야/몸 감각 제거 실험으로 단일 회로의 인과적 기여를 확인한다. 현재 generation `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off. 과거 격리 orphan terminal은 여전히 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 실제 정지 영상·몸 감각 단일 회로 오프라인 재생

- `scripts/replay_isolated_unified_stationary.py`: `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-stationary-flyvis-1/responses.npz`의 정지 실제 네 시야 FlyVis 수용기/T4·T5 반응과 `20261102-isolated-retina-probe-1/events.json`의 원본 native physics prefix를 해시·packet·원본 host decision 시각 검증 후 기존 `UnifiedFlightSensorimotor` 하나에 함께 넣었다. body encoder는 절대 world heading을 actor 감각에 넣지 않고 유효 마스크를 지킨다. 선택 회로와 경계는 기존 실험 체크포인트와 공학 어댑터이며 생물학적 검증을 의미하지 않는다. 과거 view별 body context는 입증하지 못하여 전부 None; 보간하거나 복구하지 않았다. native static identity는 캡처 후 같은 session generation에서 읽었으므로 원본 시점 독립 증거가 아니다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-unified-replay-1/`: 원본과 첫 영상 고정 FlyVis 대조 각각 76 단계, 같은 원본 몸 감각 및 host 시각, 대조마다 명시적 episode reset 후 상태 유지. 본래 몸 감각 valid 21개 중 매회 16개, 두 조건 mask 일치. 원본 대 고정 입력의 최대 출력 차 0.00290145와 내부 상태 차 0.0366145, 첫 출력 차 0, 출력 채널별 최대 차 0.00290145/0.00001428. `responses.npz` SHA `5f377e92a6eaa0320b4d2ba371ec2828aaf2f5c96ee81a5db0e998a945a2065e`. 현재 차는 정지 영상 변동을 포함하며 주행 개선이나 생리적 방향성의 증거가 아니다.
- FlyVis는 저장 후 별도 오프라인 계산한 결과이며 물리 노출 동기·실시간 IPC-to-actor·통합 반복 재현·몸 감각/시야 제거 인과 실험·실주행은 미검증. 학습/차량 명령 0. 계산 시간 단순 합산도 라이브 성능 보증이 아니다. 관련 실제 원본과 실패는 보존한다. 다음은 같은 입력 재생 반복과 시야·몸 감각 대조, FlyVis 및 회로의 전체 라이브 지연 측정 후 실제 실행 요건 판단이다.
- 현재 generation `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off; 과거 격리 orphan terminal 미상. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 단일 회로 재생 반복과 몸 감각 제거 대조

- `scripts/ablate_isolated_unified_body.py`의 첫 실행 `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-unified-body-ablation-1/result.json`은 속도만 남기고 yaw를 제거하여 기존 회로가 요구하는 실제 yaw·속도 입력을 충족하지 못해 실패했다. 실패를 보존한다.
- 수정 후 `20261102-isolated-unified-body-ablation-2/`: 동일 실제 원본·FlyVis 출력·몸 감각·시간의 원래 및 첫 영상 고정 대조가 이전 `20261102-isolated-unified-replay-1`의 intent/state와 최대 차 0으로 재현됨 (허용 기준 2e-5). 76단계, 명시적 episode reset 및 상태 보존.
- 같은 실제 영상·원본 시각에서 필수 yaw(인덱스 4)와 속도(16)만 유지하고 기울기·하중·타이어 온도·바퀴 등 나머지 감각 값과 유효 마스크를 제거한 대조: 원래 대비 출력 최대 차 0.0005522072, 내부 상태 최대 차 0.0000296906. 출력 변화는 값 제거와 availability mask 변경을 동시에 포함하므로 개별 감각·물리 자극에 대한 선택성으로 볼 수 없다. 출력은 차량에 적용하지 않았다. 응답 SHA `44082b84207e4b6e8acac5844dfae47b7c8fa705a2986c69c3b761d98afa05ee`; actor 자격/물리 시각 미검증.
- 다음: 각 몸 감각의 값을 유효 마스크와 분리해 통제한 대조, 실제 운동 시각 기능·출력 기여, live 영상/몸 감각과 회로의 물리적 시간/지연 및 실제 주행 평가가 남았다. 반복된 정지 오프라인 결과로 학습·주행 성능을 주장하지 않는다. 현재 generation `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off, 이전 격리 orphan terminal 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 몸 입력값과 유효 표시 분리 및 어댑터 열 감사

- `scripts/ablate_isolated_body_components.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-isolated-body-components-1/`: 이전 실제 정지 입력 76단계, 동일 FlyVis/몸/시간 및 2e-5 반복 기준에서 원래 및 첫 영상 고정 상태·출력 차 0 재현. 기울기(0,1), 하중(8–11), 타이어 온도(12–15), 바퀴 회전(17–20)을 그룹별로 값만 0 또는 유효 표시만 false인 조건으로 분리했다. 필수 yaw·속도 및 다른 입력은 유지했다.
- 각 조건별 실제 출력 최대 차는 tilt 0.000552341, load 0, tyre 0, wheel 0.0000002496. 동일 그룹의 값 변경과 유효 표시 변경이 같은 결과였다. 하나의 정지 자료에서만 측정했으며 부호·물리 단위 반응이나 주행 성능을 증명하지 않는다. 결과 배열 SHA `793be95bd9499177be930b8a41c08722c735822ba76ee9bc107492fee7f9a04e`.
- `scripts/audit_body_adapter_columns.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-body-adapter-columns-1/result.json`: 위 회로에서 사용한 기존 `speed-body-adapter.pt`의 201×30 projection 열을 확인. pitch·4개 하중·4개 타이어 온도 입력 가중치와 15개 전체 availability 열은 정확히 0, roll 최대 절댓값 0.02208, 속도 0.14776, 바퀴 각 열 0.14617. 이 특정 어댑터 경로는 pitch·하중·타이어 및 센서 유효성에 반응하지 못한다. 이는 해당 체크포인트/경로에 대한 결론이며 다른 회로나 다른 후보 전체에 대한 결론은 아니다. 실제 무효 센서 안전 처리와 차체 하중·타이어/셋업 적응은 목표상 누락으로 남는다.
- 다음: 검증된 비행 유래 감각 되먹임 구조와 차량 대응 가설을 명시하고 각 유효 몸 감각 및 누락에 대한 통제 자극·반응을 먼저 검증한다. 값만 임의로 학습해 생리 기능으로 선언하지 않는다. 실시간 시간 정합·가감속·변속·셋업·실제 완주/랩타임은 미검증. 학습·차량 명령 없음. session `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off; 격리 orphan terminal 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 2026-11-02 — 과거 하중 경계의 임시 재사용 대조

- 이전 `20261030-body-feedback-brake-steering-anchored` 후보는 공학적 학습 어댑터이며 실제 제동 원본에 하중 반응을 보였으나 기존 학습 데이터의 제동 MSE는 0.19334→0.19537로 악화했고, 시각 없는 별도 제동 반사실에서 제동 출력 기여는 매우 작았다. 기존 근거 [몸 감각 개입](body-feedback-input-intervention-20261030.md)과 원본 체크포인트를 보존하며 후보를 채택하지 않았다.
- `scripts/probe_legacy_load_port_reuse.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-legacy-load-port-reuse-1/`: 기존 단일 회로의 201개 몸 포트와 현재/과거 명시적 feature 순서·공통 페달 source hash를 확인하고, 과거 어댑터의 하중 값 열 네 개만 현재 0인 열에 임시 복사했다. availability 열·다른 값 열·회로는 유지했다. 수용기/포트 숫자 및 source hash만으로 포트별 생리적 동등성을 입증하지 않는다. 가상 반사실은 실제 정지 영상·몸 입력 위에서 매 프레임 앞바퀴 각각 +0.025, 뒷바퀴 각각 -0.025로 총 하중 비율을 보존한다. 이는 실제 측정 하중이나 검증된 센서 자극이 아니다.
- 76단계 같은 시간·기존 원본 및 첫 영상 고정 대조가 직전 기준과 오차 0. 기존 회로의 가상 하중 이동은 출력·상태 차 0; 임시 과거 열 적용 후 같은 이동은 출력 최대 차 0.000157945, 상태 0.000018626. 하지만 임시 적용 자체가 원래 하중에서도 출력 최대 0.000848085, 상태 0.00191919 변화시켜 기존 동작 보존에 실패한다. 응답 SHA `13a053784d33e15298d0f9203b331dab5fa3e509de4e79e6b7a94a2488a98637`; 과거 후보 SHA `4030ac7e53c9175a517adb3d588d9c5be4746f818ddc09762c6ee524771e2315`. 업데이트/배포/차량 제어 0.
- 생물학 근거: [초파리 날개 고유감각 뉴런의 말초 해부와 중앙 연결](https://pmc.ncbi.nlm.nih.gov/articles/PMC12154767/)과 [평균곤봉 감각 입력에서 날개 조종 회로까지 연결 지도](https://pmc.ncbi.nlm.nih.gov/articles/PMC12872070/)는 감각-비행 회로 연결의 근거다. 자동차 타이어 하중을 이 특정 뉴런이나 가중치에 대응시키는 증거는 아니다. 임시 이식은 공학 가설로만 남긴다.
- 다음: 입출력 기준점 보존 및 구체적 비행 유래 포트 정체성을 확인한 뒤 실제 하중 이동의 연속 기록과 움직임·시야 변화가 함께 있는 통제/반사실에서 누락·하중·온도·셋업의 기능적 인과와 부작용을 검증한다. 신경 생리, 시간 정합, 실제 가감속·변속·완주·랩타임은 미검증. 현재 세션 generation `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off, 오래된 격리 요청 terminal 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 2026-11-02 — 기준 하중 보존 경로의 오프라인 대조

- `src/flybrain_drive/reference_load_feedback.py`는 현재 `CenteredSpeedBodyFeedbackAdapter`의 출력 logit에 과거 공학 후보의 네 하중 값 열을 기준 하중과의 tanh 차이에만 적용한다. 기존 회로·기존 가중치·출력 readout은 그대로이며 별도 운동 신경망이 아니다. 기준은 최초 실제 정지 observation 하나 `[0.258385,0.257070,0.241825,0.242720]`로 미래를 사용하지 않는다. 하중 네 개 중 하나라도 무효거나 합계·부호가 불일치하면 실패하며 결측을 대체하지 않는다. 차량-비행 감각 대응은 가설이지 실측 생리 연결이 아니다.
- `scripts/probe_reference_load_feedback.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-reference-load-feedback-1/`: 정지 원본 76단계와 기존 FlyVis/몸 input, 동일 시각·별도 reset. 과거 결과 반복 최대 차 0. 네 하중을 기준으로 고정할 때 base와 참조 경로 출력·상태 최대 차 0; 실제 정지 하중 이력에서는 출력 2.7567e-7, 상태 4.4703e-8 차이. 기준 경로에서 가상 전후 5%p 이동은 출력 최대 차 0.000158325, 상태 0.0000186712. 가상 자극은 실제 하중 기록이 아니고 단위·방향 적절성 또는 실제 제동 효과를 입증하지 않는다. 결과 SHA `b24d0a86c642d2e6947f762152738a4ae5dd7dccfdc3b5d50b8a726653f08e5c`, 과거 하중 계수 `4030ac7e53c9175a517adb3d588d9c5be4746f818ddc09762c6ee524771e2315`.
- 기준 출력 일치·변화·하중 누락/불량 거부 및 기존 몸 adapter 검사 7개 통과. 초기 기준에서만 수학적 동일을 보장하며 주행 중 정상 변화, 입력 식별·셋업 변화, 참조가 부적합한 경우나 다른 조건의 기존 운전 성능은 미검증이다. 실험 후보 미배포·학습/조작 0.
- 다음: 원본 이동·제동 기록의 실제 하중 변화와 좌우/종방향 부호, 시작 기준 변화·결측, 동일 영상과 몸의 시간 관계 및 회로 제거 대조를 평가한다. 실시간, 몸·시각의 물리 시각, 타이어/셋업 적응·가감속·변속 및 자율 유효랩/랩타임은 미검증. 현재 session `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off, 이전 격리 orphan terminal 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 2026-11-02 — 실제 제동 몸 이력의 기준 하중 경로 오프라인 재생

- `scripts/replay_reference_load_brake_body.py`: 독립 감사된 개발 진단 `20261029-t1467-independent-native-brake-response`의 원본 physics prefix SHA 및 원래 host 수신간격을 보존하여 제동 전 0.5초부터 1.5초까지 63단계를 재생했다. 과거 실제 기록의 제동 전 앞하중 평균 0.5116, 제동시 0.6280으로 관찰됐으나 정확한 물리 샘플 시각은 미확정이며 진단은 학습 사용 금지다.
- 시각 경로를 명시적 고정으로 유지하기 위해 다른 날의 정지 영상으로 계산한 하나의 FlyVis 상태를 사용했다. 카메라 bound는 몸 host 수신 이후에 인위적으로 정한 가상 시각으로, 그 영상이 제동 상황과 동시에 취득되거나 실제 exposure에 정합됐다는 뜻이 아니다. 기존 단일 회로의 상태는 각 조건별로 시작 한 번만 초기화하고 몸 원래 시간 간격에 따라 유지했다. 첫 제동 전 원본 네 하중을 기준으로 고정했다.
- `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-reference-load-brake-body-1/`: 원본 하중을 기준으로 고정한 경우 base와 참조 후보 출력·상태 차 0. 실제 하중 이력 대 기준값으로 고정한 반사실의 후보 출력 최대 차 0.000555616, 상태 최대 차 0.0000717491. 응답 SHA `ab132b18d9c8922ebc9060a8746095736aee0d0e6008004aa1f7d0b976063ed2`. 몸만 이용한 제어된 재생에서 하중 경로의 수치적 기여이며 실제 자동차 제동 개선, 초파리 생리, 원래 장면의 시각 판단을 보여주지 않는다. 학습/배포/제어 0.
- 다음: 동시 원본 네 시야·이동 몸 감각과 원본 body receipt/exposure 시간 근거를 사용해 속도·조향·가감속 구간에서 가림/대조 및 중립 보존을 평가하고 하중 기준·타이어/셋업 조건에 대한 외부 검증을 준비한다. 사용자 목표에 대한 현재 단계는 정지·기록 오프라인 부품 및 일부 회로 인과 검증이며 생리 기능 전반, 실시간 통합, 실제 자율 완주 및 라인/랩타임 최적화는 미완료다. 현재 generation `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off, 기존 격리 orphan terminal 미상.
- 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 실제 교사 주행 시각·몸과 기준 하중 경로

- `scripts/replay_moving_teacher_reference_load.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-moving-teacher-reference-load-1/`: 이전 진단 AI 교사 주행 원본 `20261031-process-body-owned-pause-live`의 네 시야 영상에서 재구성된 FlyVis scene 105개, 동일 기록의 native body receipt와 365결정·시야별 prior body context를 원본 순서로 재생했다. 기존 해시·출처 감사와 FlyVis scene 재구성·원본 시각을 확인했다. 원래 자료 자체는 `training_admissible=false`, 물리 노출/몸 발생 시간 정합 미검증이다. 회로는 학생 제어에 사용되지 않았다.
- 첫 원본 body의 네 하중을 reference로 삼고, 원본 하중·reference 고정 조건마다 base와 후보를 비교했다. 원본 회로 상태/출력은 기존 CPU 기준 대비 최대 차 0. reference 고정의 기존 회로와 후보도 상태/출력 차 0. 실제 하중과 reference 고정 간 후보 출력 조향 최대/평균 절대 차 0.000486096/0.000183189, 전진 최대/평균 0.0000188649/0.0000037101, 상태 최대 차 0.0000813678. 앞 하중 비율은 0.21225~0.78052였다. 이 값은 동일 영상·몸·mask에서 하중 값만 고정한 경로 영향이지 하중 생리, 조향 정답 또는 실차 주행 개선의 증거가 아니다.
- 출력 `ablations.npz` SHA `8e102c4e6dd98a43192f3406faa27994b726168abf421f20cfe35b15e5cef47b` 및 `independent-numeric-audit.json`에서 저장 배열로 차이를 재계산했다(회로 재실행은 아님). 비행 유래 몸 신경 연결·기능, 실제 조향·가감속 적절성, 온도/셋업, 센서 오류와 live 시간·전체 유효랩은 여전히 미검증이다. 학습·새 차량 명령·배포 0.
- 다음: 여러 다른 실제 코너와 하중 이동에서 방향별 효과 및 teacher-only 평가와 분리된 actor 반응을 비교하고, 카메라/몸 물리 시간과 전체 live latency를 확인한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다. 현재 session `12717214ULL:0.0017241` unpaused·정지·brake 1·camera/AI off, 과거 격리 orphan terminal 미상.

## 2026-11-02 — 네 과거 주행 기록의 기준 하중 경로 평가

- `scripts/evaluate_historical_reference_load.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-historical-reference-load-1/`: 이미 살펴보고 경로 범위에 따라 선택한 episode03/04/06/07의 원본 FlyVis 네 시야 scene·native body·host 결정 시각을 별도 원본 hash 및 body 재구성 확인 후 재생. 각각 742/742/741/741결정, episode마다 최초 원본 몸 하중을 기준으로 정하고 조건마다 reset 후 연속 상태 유지. 과거 시야별 몸 이력은 명시적 결측. 원본 `training_admissible=false`; 독립 holdout·생리 증거가 아니다.
- 기준 하중 고정 base와 후보 출력 차 전부 0. 원본 하중에서 후보 대 base의 조향 출력 최대 차 각각 0.00099993/0.00099505/0.00089249/0.00101075, 전진 최대 차 약 0.0000264~0.0000279. episode06에서 기존 별도 baseline과 차 0. 네 조건의 전체 저장 배열 hash 및 수치 재계산은 `independent-numeric-audit.json` (모델 재실행 아님).
- 모델 추론 이후에만 열린 원본 20m 미래 주행 chord에 대해 방향 평가: 좌/우 표본 episode03 54/113, 04 64/42, 06 214/343, 07 0/551. 원본과 후보 모두 세 양방향 기록에서 balanced accuracy 0.5, 평가 결정의 조향 부호 변화 0; 07에는 한 방향 표본만 있어 balanced accuracy 없음. 이 상관된 프레임들의 미래 chord는 정답 조향이나 최적 라인이 아니며 부호 대응은 가정이다. 기존 회로의 이 방향 편향이 하중 경로로 해결됐다는 증거는 없다.
- 실제 차량 제어·optimizer·배포 0, 원본 물리 exposure/몸 시간 미검증. 다음은 학습 가능한 독립 자료에서 좌우 및 시각 선행 반응과 몸 회로 구조의 인과 기능을 먼저 다루고, 실제 통합 시간·가감속·변속·셋업 적응 및 유효 완주·랩타임을 검증한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 현재 session `12717214ULL:0.0017241`, unpaused·정지·brake 1·camera/AI off, 과거 격리 orphan terminal 미상.

## 2026-11-02 — 합성 좌우 차동 후보의 실제 기록 전이 대조

- 이전 `20261102-closed-loop-history-fit-2`의 6 업데이트 완료 결과는 당시 이미 존재했다. 이번에 같은 폴더로 다시 실행한 프로세스는 `FileExistsError`로 업데이트 이전 종료됐으며, 새 학습으로 계산하지 않는다. 원본 실패 로그 `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-closed-loop-history-fit-2-err.log`; 기존 후보 및 결과는 변경하지 않았다. 이전 고정 이력 학습의 아주 작은 개선과 이후 합성 차동 후보의 이력은 위 기존 기록에 있다.
- `scripts/evaluate_paired_candidate_historical.py`, `%LOCALAPPDATA%/flybrain/runtime-runs/20261102-paired-historical-diagnostic-1/`: 이미 본 합성 학습 `20261102-paired-direction-fit-1` 후보 SHA `a3fa71813b803b5371900708fff9423f9b08fe0c3c107127b8e6a8bfe9e13a48`, 동일 실험의 synthetic 속도 경계·power reference를 적용해 base와 후보를 동일한 나머지 회로 조건에서 비교했다. 현재 base는 과거 historical 평가와 속도/출력 계약이 달라 과거 baseline 수치 재현은 주장하지 않는다. 네 실제 episode03/04/06/07의 카메라/몸 원본 해시 및 인과적 host 순서 확인, 과거 시야별 몸은 결측. 최초 상태 reset 후 기록 내 유지. 실제 자료는 fit 금지·기존에 검사된 development 자료, 독립 holdout이 아니다.
- 20m 미래 주행 chord는 모든 예측이 끝난 뒤에만 평가용으로 열었다. 좌/우 표본 03 54/113, 04 64/42, 06 214/343, 07 0/551. base→후보 balanced accuracy는 03 0.5→0.32186, 04 0.5→0.29762, 06 0.5→0.5. 07은 한 방향 표본이라 산출 불가. 해당 평가 집합의 후보 조향 부호 변경 65/17/0/14건. 이 지표의 미래 chord는 최적 조향 정답이 아니고 부호 대응도 가정이며 연속 프레임은 독립 표본이 아니다. 다만 이 고정 조건에서 합성 학습이 실제 기록의 방향 문제를 해결했다고 볼 수 없고 두 기록에선 해당 지표가 악화된다.
- `independent-numeric-audit.json`은 결과 배열·미래 chord hash 및 20m 지표를 재계산한다(회로 재실행은 아님). physical exposure/몸 정합·실시간·생리·차량 제어·유효 랩은 미검증. 현재 후보를 미배포로 유지한다. 실제 자료 fit·optimizer·차량 제어0.
- 다음: 단일 회로에서 시각 방향 정보가 사라지거나 부호가 반전되는 지점을 실제 기록의 좌우/시야별 반사실로 분해하고, 원본 시간/허용 입력을 유지하며 독립 합성·실제 평가를 설계한다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응도 포함한다.

## 2026-11-02 — 사용자 요청으로 반복 분석 중단 및 실행 계획 수립

사용자는 토큰 소비만 반복되는 것 같다며 명확한 계획에 따라 진행하라고 지시했다. 진행 중인 `20261102-paired-historical-factorial-1`은 같은 두 기록의 8조건 반복으로 판단하여, 소유 부모 PID 40312와 실제 계산 자식 PID 23668의 명령·부모 관계를 확인한 뒤 중단했다. `interrupted.json` 및 부분 진행을 보존하며 완료나 검증 성공으로 취급하지 않는다. optimizer·차량 제어 0. 현재 세션은 이전 확인에서 정지·brake1·camera/AI off였다. [실행 계획](EXECUTION_PLAN.md)에 최종 목표, 단계별 할 일·통과 조건, 중복 실험 제한과 바로 실행할 작업을 명시한다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 및 상대 바람도 목표에 포함한다.

## 2026-11-02 — 실행 계획 1단계: 분리 카메라와 실제 단일 회로 정지 shadow

`scripts/record_isolated_unified_stationary_shadow.py`, `D:/CodexResearchRuns/flybrain-isolated-unified-shadow-20261102-01/`: 기존 FlyVis·단일 비행 회로 shadow에 해시 검증된 별도 MMF 네 시야 픽셀 owner를 연결하고 CSP API로 pause·owned resume을 각 1회 수행했다. 네 시야 129장면과 회로 출력 454개를 냈으나 마지막 출력은 body receipt→완료 약 162.6ms로 원래 50ms 한도를 초과해 거부됐다. 시각 처리 간격 150ms 초과도 기록됐다. `stopped_on_error`이며 전체가 안정적/실시간이라고 판정하지 않는다. `mmf-stream-77c04fcd6def457c91acddade8684df8` terminal stopped 145, consumer 종료, 145그룹 보관본 hash/저장 회계 검증(archive SHA `9765fc4fe20238f8ddd8b5b07ca6b95d3bec8e83de94f6455fd9978963edd587`). actuator·optimizer 0, 물리 노출 시간 미검증. 테스트 50개 통과. 마지막 세션은 unpaused·정지·brake1·camera/AI off. 다음은 원본 한 번의 지연 분해와 그 결과에 따른 통합 경로 수정이다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 및 상대 바람도 포함한다.

## 2026-11-02 — 실시간 shadow의 held-prefix 대조

`D:/CodexResearchRuns/flybrain-isolated-unified-shadow-20261102-01/latency-audit.json`: 원본 hash 확인 후 몸 업데이트 p50 5.20ms/p95 8.48ms/최대 157.81ms, prefix p50 16.33ms/p95 17.95ms/최대 165.22ms. 마지막 몸 업데이트와 prefix는 약 153ms 겹쳤다. host 경과 시간이지 실행 중 CPU 시간이나 GPU 단독 원인은 아니다.

동일 프로그램에서 held-prefix만 생략한 `D:/CodexResearchRuns/flybrain-isolated-unified-shadow-no-prefix-20261102-01/`은 135개 시각 장면·479개 출력 후 몸 receipt→출력 50ms 한도 초과로 `stopped_on_error`; 마지막 몸 계산 154.43ms, 중앙 5.81ms/p95 9.20ms. 인접 FlyVis 처리와 겹쳤으나 스케줄 원인 미확정. 해당 카메라 `mmf-stream-4caaac1b4a9f4ce3b6c826ef5f3365c8` terminal stopped 149, 전체 archive 149그룹 검증, optimizer·차량 제어0. 각각 단일 실험이므로 확률·일반 성능으로 확장하지 않는다.

다음은 기존 단일 회로와 상태를 소유하는 CPU 프로세스 분리 경로의 원본 replay 동등성을 확인하고 기존 한도를 유지한 라이브 대조다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업·바람도 포함한다.

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
