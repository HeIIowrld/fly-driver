# 최신 비행 actor의 실제 운동 시각·몸 감각 연결 — 2026-11-05

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

최신 input_adjoint 후보를 실제 GPU 운동 시각→CPU 비행 회로에 연결했다. 최종 shadow-04는104개 영상 묶음·455출력·5.66655초를 원래 body50ms/view150ms/source-render25ms 및 요청50ms 제한으로 처리했다. 별도 CPU 전체455출력 재생 최대차0, 원본104영상 묶음의 독립 시각 특징 재구성 최대차1.78814e-7이다. 학생 제어0·추가학습0·미승격이다. 저속 교사 관측의 연결 성공이며 실제 학생 조향 개선/한계 주행/온도 적응을 검증한 것은 아니다.

## 모델과 실행 경계

체크포인트는 runs/fixture-adjoint-learning-20261104/input_adjoint-fit-01/ports.pt, SHA256 723286d6f79401e266d24a7c384680f95cf3edbabf7807139bfb39177dea6812다. 현재 SpatialMotionDNOverlay와 SymmetricNormalizedFlightSpeedCommand를 모두 실행하는 PowerBoundedSpeedDecisionRuntime을 CPU 자식에서 복원했다. 기존 측정 화학 그래프·body/visual port·근육 판독과 누적 bounds는 그대로다. 실제 근육 힘/생리 전체 재현과 공학적 입력·학습·차량 변환을 구분한다. 지도·world pose·진행률·교사 명령은 학생 입력에 들어가지 않는다. 몸 가속도·온도 등은 원래 native bytes와 감각 계약을 사용하며 값을 교사로 보충하지 않는다.

새 moving entrypoint와 owned launcher는 원래 source-qualified camera/coordinator/teacher를 재사용한다. 카메라/교사/CPU 소유권은 분리하고 학생에게 액추에이터를 주지 않는다. 같은 에피소드의 회로/각 view 수치 상태를 계속 유지하며, GPU 준비용 합성 상태는 실제 author steady prior로 시작하는 영상 이력에 주입하지 않는다. 계산 detach를 상태 초기화로 바꾸지 않는다.

## 실제 결과

| 항목 | shadow-04 |
|---|---:|
| 원본 영상 묶음 / 유효 출력 | 104 / 455 |
| 첫 결정부터 마지막 출력까지 | 5.666549초 |
| 최대 몸 감각 나이 | 21.451ms |
| 가장 오래된 view 최대 나이 | 137.366ms |
| 최대 추론 시간 | 17.399ms |
| 최대 일반 RPC 시간 | 5.820ms |
| heartbeat 최대 속도 | 19.843km/h |
| 전체 native 사본의 실제 타이어 온도 | 78.234–79.862℃ |
| CPU 원본 출력 연속 재생 | 455개 최대차0 |
| 픽셀 재구성 최대차 scene/static/motion/coarse | 1.788e-7 / 0 / 5.960e-8 / 5.960e-8 |
| 학생 제어 / optimizer / desktop 자동화 | 0 / 0 / 0 |

교사만10/20km/h의2초씩 유한 단계를 주행했다. 교사103개 조작 명령·유효랩0이다. 최대 속도는 기존 실제 학습 지원13m/s보다 낮아 이 조건의 운전 오차를 고속 모델 능력으로 해석하지 않는다. 시작 온도는 별도 에피소드 초기 조건이며 달리는 동안 목표값을 센서 대신 넣거나 thermostat으로 유지하지 않았다. 각 source prefix/hash/receipt·camera generation·epoch·원래 view clock을 기록했다.

최종 GPU7852/launcher42644 원래 핸들의exit0, CPU5576/native4612 exit0, teacher exit0, 카메라/archive worker 종료·API pause 성공을 확인했다. 독립 GPU 픽셀 감사와 CPU 출력 감사도 원래 실행 세션exit0이다. 최종 heartbeat는 generation1117858ULL:0.0016212·epoch9·paused=true다. 관련32검사가 통과했다.

## 보존한 실패와 진단

- shadow-01은 구 Lua 소스의 타이어 초기화 증거를 넘겨 교사 시작 전 거부됐다. 교사/학생 조작0이다. 이 실패 뒤 thermal 증거를 비싼 GPU 초기화 전에 검사하도록 옮겼다. mode=None에서는 native source가 시작되지 않았음을 명시해 후처리한다.
- prepare80-01/gear80-01은 현재 소스로 새 epoch8·80℃·1단·정지/full brake를 원본 native로 확인했다. shadow-02는83묶음·유효254출력 뒤 요청252 context에서 시간 초과했다. 25새 사본의 native drain 단계 wall59.006ms, 해당 요청 관측57.541ms다. 동시에 마지막 추론은83.615ms/몸 output age85.550ms로 거부됐다. 이 실패를 재시도하거나 receipt를 갱신하지 않았다. 유효254출력만 별도 재생 최대차0을 확인했고 마지막 rejection도 보존했다. whole shadow=false다.
- owned-03은 앞선 주행 뒤 실제 타이어 온도가 시작 허용±5℃ 밖이어서 새 사전 검사에서 거부됐다. GPU 카메라/학생 에피소드는 시작하지 않았다. 실패 로그와exit1을 보존했다.
- prepare80-02/gear80-02로 명시적 새 epoch9·80℃와1단을 재검증한 뒤 shadow-04를 실행했다. 두 gear 준비의 조작은각5개이며 학생 조작이 아니다. 모든 초기화·resume/pause/restart는 기존 CSP/API·배경 인터페이스만 사용했다.

CPU 자동 cyclic GC를 유한 owner scope 안에서 미루는 CurrentFlightGCOwnerProcess를 추가했다. 원래 owner의 actor/프로토콜/신선도 구현을 상속·재사용하며 scope 종료 시 원래 GC 설정을 복원한다. Python 참조 카운팅과 명시적 gc.collect는 유지한다. GC callback은 phase/time/thread/generation을 남긴다. 독립128결정의 예측/근육 최대차0 및 owner exit0을 먼저 확인했다. shadow-04의 GC는 활성 추론이 끝난 뒤 scope 복원 시 약9.96ms 발생했고, 활성 추론 안의 GC 사건은없다. 이는 실행 지연을 줄이는 공학 조치다. shadow-02에는 GC callback이 없었고 두 실제 궤적은 완전히 같지 않으므로 앞선59ms 지연의 원인이 GC 하나였다고 증명하지 않는다. 한 성공 사례를 반복 운전 안정성으로 일반화하지 않는다.

shadow-01의 GPU32480/launcher50656은 별도로 남은 종료 문제다. terminal failed 파일은 저장됐고 CPU·교사·카메라는 닫혔지만 원래 launcher 핸들은 아직 반환되지 않았다. Win32 query의 child exit code1에도 WaitForSingleObject(0)=258이며 종료 신호는 미확인이다. Stop-Process 시도는 PID 없음으로 실패했고 실제 강제종료를 수행하지 못했다. attempted/failed receipt와 원래 관측 핸들을 보존했다. 이 PID들을 학습·추론 진행 또는 정상 종료로 세지 않는다. 이전 moving-shadow-02/03의 잔여 PID도 건드리지 않았다.

## 원본과 남은 과제

원본은 runs/current-flight-moving-20261105의 summary-01.json, shadow-01/02/04, owned-01/02/03/04 receipts/logs, 두 prepare80/gear80, gc-owner-validation-01, timing-02/04, owner-audit-02/04, pixel-audit-04와 source-receipt-01.json이다. source를 수정한 시도는 각 manifest/hash와 executed-entrypoint.py로 구분하며 과거 실행 소스를 현재형으로 덮어 쓰지 않는다. 기존 실험/체크포인트는 hash로 보존 확인했다.

실시간 연결은 이제 이 조건에서 검증됐으므로 같은 실행 경계의 반복 구현보다 조향 기능 개선으로 돌아간다. 최신 후보의 같은 몸/회로 상태 영상교환 방향25/30·목표13/30, coarse_static 평균 signed margin 음수·fine_motion보다 full 약화, 전체 조향 label 오차 악화가 남아 있다. 다음은 기존 fixed bounds와 몸 감쇠·조향 보호를 유지하면서 coarse/fine 시각이 하행 회로에서 충돌하는 작동점을 분리하고, 새 조건 영상/코너의 선행 판단과 실제 제한된 폐루프 기여를 평가하는 것이다. 학생 제어는 이 관측 성공만으로 승인된 성능이라고 간주하지 않는다. 온도·자세·회전·가속도·하중·타이어/접지·셋업·차종·상대풍/변속 적응과 빠른 유효 완주·레이싱 라인/랩타임·한계 주행은 미완료이며 목표는 active다.
