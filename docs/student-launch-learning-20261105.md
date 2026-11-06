# 출발 실패 분리 진단·저속 페달 학습·부모/후보 실제 비교

2026-11-05, Asia/Seoul. 원본: `runs/student-launch-learning-20261105/`. [이전 연결 시험](student-closed-loop-20261105.md)의 부모/후보·초기40/80°C 출발 실패를 이어서 검증했다.

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

## 원인별 증거

| 구분 | 원본 확인 | 판단/처리 |
|---|---|---|
| 페달 동시 출력 | 이전 네 학생 시험의 두 페달>0.02 비율100%. 요청 gas 약0.05~0.08/brake 약0.09~0.13이 native 값에도 나타남 | 전송값 뒤바뀜이나 SDK 요청만의 현상이 아니다. 독립 sigmoid 페달과 저속 감독 누락을 진단하고 실제 교사 label로 근육→페달 대응을 다시 학습 |
| gear | 같은420바이트 authority의 active native gear는 모두2(1단), autoShifterOn1, pitLimiterOn0, RPM 약2100~2381 | 중립에 방치된 실패가 아니다. gear-up 준비와 forward-gear guard 유지 |
| clutch | 같은 저장 원본의 byte364에 clutch가 이미 존재. 이전 active 값은0~약0.09, 실제 assists.ini AUTO_CLUTCH1·수동 clutch 축은 미할당 | gear만으로 clutch 체결을 가정하지 않는다. 원본 범위와 이후 실제 가속 반응을 비교하며 제어 축이나 actor 입력을 새로 추가하지 않음 |
| 저속 학습 | 현재 cold TRAIN full-lap 최저속도13.421m/s, hot TRAIN16.483m/s. 13m/s 미만 교사 label은 각160개가 있지만 기존 fit은0개 | 실제 출발은 감독 범위 밖이었다. 현재 코호트/원본 무수정 요청을 재검증해 TRAIN131개(66/65)의 저속 label을 새로 사용 |
| 회로의 속도 명령 | 실제 이전 부모80의 원본 시각/픽셀/몸 감각 재생: target0~1.415m/s, power current 약0.050~0.065 | 속도 명령이 항상0이거나 회로 전체가 실행되지 않았던 것으로 설명하지 않음. 페달 readout과 신경/근육 반응을 분리 |

`diagnosis-01`은 기존의 더 짧은 몸 prefix와420바이트 authority를 혼동해 clutch 부재라고 적었고 옛 코호트를 읽었다. 실행 소스를 보존하고 `diagnosis-02`에서 현재 코호트와 원본420바이트로 정정했다. byte364/AI416 offset은 로컬 pinned `sim_info.py`의 AST와 독립 검사했다. 물리적 clutch 방향/토크 단위를 새로 교정했다고 주장하지 않는다.

자동 클러치/변속 설정과 프로파일·차종을 바꾸지 않은 채 학습된 페달만 달라진 시험에서 차량이 출발하고 clutch 값도 변했다. 따라서 영구적인 기어/클러치의 고정적인 연결 불량만으로 결과를 설명하기는 어렵다. 이는 동일 설정의 출발 반응에 따른 추론이다. 페달과 자동 클러치의 결합 반응을 완전한 구동계 시스템 식별로 확대하지 않는다.

## 학습과 actor 경계

기존6개 modeled flight muscle proxy와 기존 정규화만 입력받는 `LearnedFlightPedalReadout`을 추가했다. 기존 디코더의 두 페달 행14계수만 실제 지도 학습했다. 원래 조향·감각 포트·비행 회로·재귀 상태·속도 정책·그래프는 고정했다. 이는 자동차 가상 몸을 위한 공학적 디코더이며 실측 날개 힘/연결이라고 부르지 않는다. 새 CNN/RNN이나 속도별 가속 규칙으로 비행 회로를 대신하지 않았다.

라벨은 독립 자격 확인된 기존 코호트에서 원본 teacher 요청과 canonical3 변환을 재대조했다. `unlimited==limited`이고 자동 변속 coast가 아닌 실제 요청만 사용했다. 수동 제한/교사 warmup 보정이 들어간 저속 요청은 제외했다. 기존 고속 TRAIN5659개와 TRAIN 저속131개를 사용하고, 소비한 같은 Monza development 저속64개는 fit/중간 선택에 사용하지 않았다. 지도 truth/지도/world pose/절대 heading/teacher observation/gear/clutch를 actor에 주입하지 않았다.

각 모델은 원래 고정 checkpoint에서 시작해 새로 학습 가능한14계수에 새 Adam(lr0.003)을3000회 적용했다. 과거 회로/port Adam의 재개로 바꾸어 부르지 않으며 원래 optimizer 기록을 보존했다. seed783105, 고정 횟수, development/실제 주행 결과로 조기 종료·최적 checkpoint 선택 없음이다. 타력/조향·throttle·brake 계약과 독립 긴급 제동은 그대로다. 출력의 clipping·최소 throttle·brake 삭제·페달 상호 배제·교사 혼합·조건별 boost는 없다. 실제 teacher label에 brake가 있으면 학습 목표에도 남는다.

1. **저속 보완형(`*-pedal-fit-02`)**: 저속 오차를 줄였지만 기존 고속 페달 오차가 증가했다. 고속 성능 실패를 보존하고 유한 출발 연결 실험으로만 검증한다.
2. **고속 오차 제한형(`*-constrained-fit-01`)**: 같은 학습에 각 TRAIN 에피소드·각 페달의 고속 MSE≤원래×1.02+1e-8(float32) 조건을 추가했다. 매 계수 갱신을 backtracking하고 위반 시 그 계수 갱신을 수용하지 않는다. 차량 명령을 사후 보정하는 규칙이 아니다. 고정3000 Adam step을 모두 남긴다. 이 조건도 개별 코너/접지/제동이나 미사용 조건의 안전을 보장하지 않는다.

첫 `parent/dual_guard-pedal-fit-01`은 frozen 원본 projection을 복제하며 requires_grad=False를 유지한 구현 오류로 backward에서 실패했다. 실제 optimizer 갱신0·제어0이며 소스/실패 result를 보존했다. 학습 가능 행을 명시한 뒤 gradient·고정 steering·페달 동시 활성 유지·signed power·ABI 검사11개를 통과했다.

| 학습형/모델 | cold TRAIN 저속 throttle/brake RMSE | hot TRAIN 저속 throttle/brake RMSE | development 저속 throttle/brake RMSE | TRAIN 고속 최악 MSE 비율(원래=1) |
|---|---|---|---|---:|
| 저속 보완 부모 | 0.314826/0.076578 | 0.321326/0.045670 | 0.317696/0.054870 | 2.354500 |
| 저속 보완 후보 | 0.313293/0.074778 | 0.320162/0.044403 | 0.316741/0.056028 | 2.411512 |
| 고속 오차 제한 부모 | 0.454191/0.273124 | 0.462037/0.241980 | 0.460798/0.268331 | 1.020001 |
| 고속 오차 제한 후보 | 0.455789/0.267579 | 0.465278/0.238865 | 0.463227/0.266326 | 1.020001 |

원래 저속 throttle/brake RMSE는 부모 cold .462991/.290217·hot .473447/.253590·development .472083/.283434, 후보 cold .464777/.285077·hot .477040/.251022·development .474812/.282082였다. 조향은 모든48866개 기존 재생 사건에서 각 원본과 차이0이다. 네 새 fit 합계 Adam12000회이며, 고속 오차 제한형은 조건 경계에 도달했다. 부모의 비영 수용 scale은33회, 후보의 비영 scale은3000회였지만 float32에서 실제 계수가 달라진 횟수는 두 모델 모두33회였다. `parameter-step-counts-01.json`과 전체 trajectory가 실제 계수 변화/서로 다른 벡터 수를 기록한다. MSE 비율은 RMSE 비율과 다르다.

| 모델/학습형 | 새 checkpoint SHA256 |
|---|---|
| 저속 보완 부모 | `71b432d6d47a947e0a67dadfb69d38f001e7eb153bc07376eb364d61e9c69a6e` |
| 저속 보완 후보 | `f875e19c7019f2e56a879d9baa11f1507615792825dfaa52f240bd4caaced3c4` |
| 고속 오차 제한 부모 | `002e53f0b2f5ae2cafd399bcf110eee240217d080ff69e9b45d142896c5050cb` |
| 고속 오차 제한 후보 | `d53ea3f5c33ebe343ed7dbcf84c3783c0ef364e5dd674c4836fa8ac3fb89ee65` |

## 실제 폐루프 프로토콜과 실패

검증된 CSP restart/start/pause/resume, background MMF/vJoy만 사용했다. AC 데스크톱 입력0, 설치 Lua·assists·조향 배율 변경0이다. 준비0.5초/3영상 묶음/실제 안정 출력, body50ms/view150ms/source-render25ms/RPC50ms와 source-bound 결정·단일 재귀 owner를 유지했다. 새 native gear/RPM/clutch 필드는 진단 로그에만 있다.

저속 보완 부모의 첫0.8초 예정 시험 `linear-parent80-student-01`은 실제54명령·최고10.1456km/h·변위0.708187m 뒤 **10km/h guard 정지**였다. tyres out0·gear2였고 native 중립 제동·SDK FREE·pause·소유 owner/actuator exit0을 확인했다. 다음 apply RPC는 unknown timeout으로 보존하며 재시도하지 않았다. GPU interpreter는 실패 exit1이고 전체 시험도 실패다. 실제 출발 반응과 시험 성공을 구분했다. 과거 pilot이 실패 경로에서 최대속도/변위를 result에 적지 않던 누락도 수정했으며 이 첫 원본은 덮어쓰지 않고 별도 감사로 보완했다.

이 실패 뒤 비교 관측 구간을 **모든 모델 공통 예정0.35초**로 줄였다. 강제 상한1초/100결정·10km/h·travel/변위1m·tyres out0·damage guard는 그대로다. 원래0.8초 실패를 짧은 성공 기록으로 교체하지 않는다. 출발 확인 기준 최고속도≥0.5km/h 및 변위≥0.02m도 유지했다. 이는 짧은 출발 연결 기준이며 운전/유효랩의 성공 기준이 아니다.

| 조건(기본 예정0.35초) | SDK accepted | 실측 첫~마지막 요청 초 | 최대 km/h | armed 진단 변위 m | 판정/종료 |
|---|---:|---:|---:|---:|---|
| [보완 부모80 예정0.8초](../runs/student-launch-learning-20261105/linear-parent80-student-audit-01/result.json) | 54 | 0.564731 | 10.145631 | 0.708187 | 실패·ValueError: Speed/tyres/forward-gear trial limit |
| [보완 부모40](../runs/student-launch-learning-20261105/linear-parent40-short-audit-01/result.json) | 29 | 0.354608 | 5.645204 | 0.238186 | 출발·짧은 연결 통과 |
| [보완 후보40](../runs/student-launch-learning-20261105/linear-candidate40-short-audit-01/result.json) | 26 | 0.400460 | 6.671803 | 0.319730 | 출발·짧은 연결 통과 |
| [보완 부모80](../runs/student-launch-learning-20261105/linear-parent80-short-audit-01/result.json) | 25 | 0.356672 | 5.570057 | 0.238361 | 출발·짧은 연결 통과 |
| [보완 후보80](../runs/student-launch-learning-20261105/linear-candidate80-short-audit-01/result.json) | 18 | 0.225132 | 3.378963 | 0.088907 | 실패·source_lease_expired |
| [제한 부모80 준비 실패](../runs/student-launch-learning-20261105/constrained-parent80-short-01-audit-01/result.json) | 0 | — | 0.000457 | 미ARM | 준비 실패 |
| [제한 부모80 반복1](../runs/student-launch-learning-20261105/constrained-parent80-short-02-audit-01/result.json) | 23 | 0.396846 | 0.122243 | 0.001325 | 실패·pause CLI timeout |
| [제한 부모80 반복2](../runs/student-launch-learning-20261105/constrained-parent80-short-03-audit-01/result.json) | 9 | 0.142913 | 0.112037 | 0.000208 | 실패·source_lease_expired; pause 반환1 |
| [제한 후보80 준비 실패](../runs/student-launch-learning-20261105/constrained-candidate80-short-01-audit-01/result.json) | 0 | — | 0.000452 | 미ARM | 준비 실패 |
| [제한 후보80 반복1](../runs/student-launch-learning-20261105/constrained-candidate80-short-02-audit-01/result.json) | 5 | 0.062138 | 0.059104 | 0.000200 | 실패·source_lease_expired |
| [제한 후보80 반복2](../runs/student-launch-learning-20261105/constrained-candidate80-short-03-audit-01/result.json) | 32 | 0.386573 | 0.097359 | 0.001210 | 출발 미달·짧은 연결 종료 |

이번 단계 학생 SDK accepted는 **221회**다. [모든11개 원본 요약](../runs/student-launch-learning-20261105/comparison-summary-01.json)을 남겼다. 예정0.35초는 실측 고정 구간이 아니며 실제 요청 지속시간은 표와 같다. GPU 처리·종료 확인 지연에 따른 초과도 숨기지 않으며 독립1초 상한을 바꾸지 않았다. 중단 길이·반복 수가 달라 단순 속도/변위로 후보 우월성을 판단하지 않는다. 고속 오차 제한형의 실제 출발 기준 통과는0회다.

영상 전달 실패도 분리했다. `linear-candidate80-short-01`은 실제18명령·3.379km/h·0.088907m 뒤 source lease가 만료됐다. 새 영상의 결정이 이전 view deadline 약0.16ms 전에 pipe로 전달돼 다음 actuator poll에서 만료를 관측했다. 아직 계산 중인 새 source가 있어도 오래된 source 추론을 한 번 더 수행하던 경계였다. 당시 두 추론은 약17/16ms였다. 비교 네 조건을 모두 끝낸 뒤 원본 ticker를 보존하고 pending source 전환에 두25ms 추론 슬롯을 예약하도록 수정했다. 실제 대기는25ms 이내, 원래 body50ms/view150ms·source-render25ms·RPC50ms와 stale 거부는 그대로다. 옛 코드에서 실패하는 회귀 검사를 먼저 저장했고 새 코드에서 전체 검사를 통과했다. 재귀 상태나 원본 시각은 다시 찍지 않는다.

## 독립 감사·다른 부품 검토

최종 관련 전체 검사 `tests-04`: 187개 중186통과/1 Windows POSIX resource-tracker 해당 없음 skip, 오류/실패0, 원래 child handle exit0이다. 원본 시각·순서·세대·unknown·caller stall·독립 stop/release와 기존 vJoy 검사를 포함한다. 소유권과 실패시 정지 사유는 speed/tyres/gear를 별도로 기록하도록 고쳤다. 실패 결과에도 실제 motion 지표를 남기되 실패 상태를 성공으로 바꾸지 않는다.

학습 optimizer의 저장 gradient/Adam 제안/투영/연속 step/마지막 checkpoint를 별도로 감사했다. 저속 보완형 각3000step의 독립 Adam 최대차는 약4.88e-7, 투영차0이며 steering gradient0이다. CPU owner와 실제 원본 몸/영상의 각128결정 비교에서 근육 최대차0, 새 head 출력은 batch 수치 차이≤2.39e-7이다. 기존 원본 회로의 state를 teacher trajectory로 주입하거나 arm 때 reset하지 않았다.

실측 연결·생리 근거·기능 근사·차량 대응 가설·학습 어댑터를 구분한다. 이번 학습이 시각의 코너 선행 판단이나 비행 균형의 생리 기능을 새로 입증하지는 않는다. 원래 시각 개입의 방향/크기 부족과 저속 조향 오차도 사라진 것으로 처리하지 않는다. 기존 실제 시각·몸 경로와 하중/회전·기울기 되먹임은 유지했다. 원본 strict loader/학습 manifest를 바꾸지 않고 새 readout wrapper와 명시적 checkpoint lineage로 연결했다. 이전 성공 제어 소스도 `prior-control-source-recovery-01`에서 당시 SHA와 정확히 같은 bytes로 보존했다.

실제 전체 기록의 **794개 회로 출력**을 원본 clock·몸 bytes·선택 영상으로 독립 재생해 최대차0을 확인했다. SDK 호출 종료 시 최악 body age26.688ms/view age137.673ms로 원래50/150ms 안이다. 실제 출발한 부모80의16개 선택 영상 묶음을 원본 RGB·네 view clock부터 재구성했고 scene/static/fine motion/coarse 최대차는7.45e-8/0/5.96e-8/3.17e-8이었다. GPU 감사 원래 handle exit0이다. 이는 실행 재구성 증거이며 경로 판단·생리 기능·유효랩 자격을 대신하지 않는다.

첫 고속 오차 제한형 폐루프(`constrained-*-short-01`)는 두 모델 모두 준비 중 실패·학생0이었다. 부모는 느린 준비 조회가 이미 만료한 cached 결정을 읽었고, 후보는 pending 대기 뒤 held body age가50ms를 약간 넘었다. 기존 빠른 읽기 전용 observer를 준비에 재사용하고 arm 직전에도 준비 조건을 다시 검사했다. pending 대기는 held body age≤25ms일 때만 허용해 실제 회로 진행 여유를 남겼다. 원래 stale 거부·body50/view150·재귀 state는 유지한다. 앞선 실패와 두 단계 수정 전 소스는 각각 보존했다.

수정 뒤 제한형은 고정된0.35초 시험을 부모/후보 각2회로 사전에 한정해 반복했다. 첫 부모는 학생23회·출발 미달이었고 종료 pause CLI가30초에 끝나지 않아 전체 실패다. 이 CLI의 final-pause 디렉터리에는 manifest가 없고, 소스 순서상 API 발행 전 준비에서 멈춘 것으로 추정한다. 다음 단계 전 독립 post-state는 native brake1·속도0.000464km/h·SDK FREE지만 paused=false였다. pause 성공으로 처리하지 않았다. 이후의 명시적 새 epoch 준비는 별도로 검증했다. 두 번째 부모의 pause 반환1과 반복 source lease 만료도 기록한다. 성공만 고르거나 관측 구간을 추가로 줄이지 않았다.

변위 지표는 armed 상태에서 진단 위치를 샘플링한 최대값이다. 전체 제동 후 위치를 모두 샘플링한 총 이동 거리 또는1m의 물리적 hard bound로 해석하지 않는다. 실제 정지는 native 속도·pedal·교정된 중립의 서로 다른3packet과 SDK FREE로 별도 검증한다.

고속 오차 제한형의 모든3000step에서 실제 저장 계수의 float32 고속 MSE 제한(×1.02+1e-8)을 독립 재검사했다. 두 모델의 최악 float32 비율은1.020000577, NumPy와 예측 최대차1.79e-7이다. 위 epsilon을 비율에서 숨기지 않으며 수학적 정확한1.02 또는 각 코너의 보존을 주장하지 않는다.

모든 TRAIN 계수 벡터를 확인하는 첫 NumPy 감사에서는 부모의 경계값 비교가 실패했다(`parent-every-step-constraint-audit-01`). 이 실패도 보존한다. 저장한 계수의 실제 float32 연산과 독립 NumPy 연산을 함께 확인하는 v2 감사에서 원래 학습 제한 자체를 재검사하며, 모델 가중치나 제한을 바꾸지 않았다.

## 현재 판정과 다음 연구

출발 실패에 실제 저속 감독 누락과 근육→페달 대응 오차가 기여했다는 증거를 얻었다. 저속만 크게 맞춘 디코더가 고속 성능까지 개선됐다고 처리하지 않는다. 온도별 짧은 출발 차이는 실제 tyre grip 적응이나 통계적 우열이 아니다. 동일 Monza에서 소비한 development 평가와 새로운 트랙 일반화도 구분한다. 후보와 새 디코더 모두 연구용이며 승격하지 않는다.

실시간 source lease 만료도 여전히 남았다. 새 전환/준비 수정이 모든 Windows 지연을 해소했다고 주장하지 않는다. 다음 검토에서는 GPU 작업/공유 RPC와 결정 소비 경로를 분리하는 owner→actuator 독립 mailbox를 계획으로 검토하고 원래 신선도 한도를 유지한다.

다음은 기존 고속 제동/가속 성능을 보존하며 저속 출발을 함께 구현할 감각·하행·국소 운동/몸 감각 경로와 근육 디코더의 역할 분리다. 추가 증류에 앞서 누락된 시각/비행 제어 부품의 반응·인과 기여를 확인하고, 실제 코너 선행 조향·가속/감속·하중 이동·접지/타이어·셋업/차종/상대풍/변속 적응, 반복 유효 완주·빠른 레이싱 라인/랩타임으로 이어간다. 몸 기울기 되먹임과 actor 감각 경계를 유지하며 연구 목표는 active다.

## 최종 외부 상태·인계

`final-preflight-01/result.json`은 paused=true·camera_active=false·AI=false·pending 없음·vJoy FREE·native brake1·속도0.000451230km/h다. 이번 소유 제어/추론/학습/감사 실행은 종료했다. 실제 모든 학생 run의 CPU owner/actuator는 원래 handle exit0이다. 실패한 GPU launcher exit1·pause CLI 실패도 원본 그대로이며 모든 학생 run에서 native 중립 정지와 SDK FREE를 확인했다. 누적 heartbeat_write_failures12를 지우지 않았고 과거 소유권 불명확 orphan은 건드리지 않았다.

`final-source-receipt-01.json`에 변경 소스29개·원래/새 checkpoint6개의 SHA와 HEAD/branch를 기록했다. 원래 `.723`/dual_guard 가중치는 그대로다. `final-review-receipt-01.json`이 최종 문서·검사·감사·실제 결과 hash를 묶는다. 기존 dirty workspace·원자료를 유지했고 branch/worktree 전환·commit/push·배포를 수행하지 않았다. 전체 목표는 active이며 유효랩과 빠른 레이싱 라인/랩타임은 여전히 미검증이다.
# 후속 증거

이번 문서는 이전 실험 당시 결과다. 이후 source lease·API 종료 보완과 저속/고속 결합 학습은 [후속 보고](lease-balanced-learning-20261105.md)에 기록했다.
