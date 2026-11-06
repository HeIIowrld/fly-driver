# native 후보의 시각 기여와 실제 움직임 추론 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 변속·차종/타이어/셋업 적응·검증된 상대풍, actor의 지도/world pose/절대 heading/교사 명령 배제를 유지한다. 측정 연결·기능 근사·차량 대응·학습한 경계를 구분한다.

**전체 목표는 진행 중이다.** 이번에는 이전에 개선한 native19 감각·조향 후보에서 시각 기여를 확인하고, 그 후보의 실제 움직임 중 추론을 처음 실행했다. 한계 주행이나 학생 차량 제어를 검증한 것은 아니다. 네 움직임 시험은 모두 전체 실패로 보존했다. 마지막 시험의 유효93출력과 시험 전체의 성공 여부를 구분한다.

## 같은 몸에서 새 native 후보의 시각을 바꾸었다

루트 `runs/native-visual-precorners-20261104/`, 계획 `plan-01.json`, 실행 `replay-01`, 독립 감사 `visual-audit-01`이다. 이전 native 없는 head의 결과를 새 후보에 옮겨 쓰지 않고 **새 native 후보 자체**를 세 기록48,956사건에서 연속 재생했다.

보호 head는 이전 `runs/precorner-visual-transfer-20261104/protected-refit-01/native-steering.pt`, SHA `6f1a569224ee8fe6d1988086508568a260fbe0a29a82d7e01e1cda469d247430`다. native 경계는 `runs/thermal-driving-20261103/native-fit-02/body-extension.pt`, SHA `d41677fcf67ad0ae4412e53788a91d0d7f83ba061653126f1e2708631e03d687`이며, 원래 비행/시각 경로와 가속·제동 head를 보존한다. `audited_native_flight_candidate` 로더가 두 감사의 result 바인딩, 체크포인트 계보/해시, native 경계·근육 정규화·페달 행의 값/형태/dtype 동일성을 검사한다. 로더는 제어 자격을 부여하지 않는다.

분기 시작 신경 상태·이후 몸/과거 몸·native19 값과 mask·원래 packet/첫 receipt·clock을 고정하고, 기록된 scene576/static320 특징만 정상/반대 회전 진입/0.6초 지연/분기 시작 고정/공간 경계 중립으로 바꿨다. donor의 native 감각을 receiver에 섞지 않는다. 정상 재생은 저장된 native 궤적과2e-6 이내, 별도 근육→head 수식과 raw native sidecar join도 독립 확인했다.

| 기록 | 정상 진입 전 조향 RMSE | 공간 시각 경계 중립 | 중립 시 오차 증가 |
|---|---:|---:|---:|
| 냉간 시작 기록 | .061707 | .064501 | 4.53% |
| 고온 시작 기록 | .046453 | .049817 | 7.24% |
| 명목 시작 기록 | .062980 | .065400 | 3.84% |

237감독 라벨·15구간의 예상 회전 방향 순서는 **6/15**로 여전히 불안정했다. 같은 Monza 코너들을 반복한 자료이며 이미 소비된 TRAIN/개발 기록이다. 이 수치는 새 코너 분류 정확도가 아니다. donor의 도로 상대 위치·진입 준비까지 맞추지 않았고 반사실 조작 정답도 없다. 원본 픽셀/FlyVis 상태 전체 재시뮬레이션이 아닌 특징 경계 개입이다.

공간 경계가 해당 기록의 조향에 기여한다는 증거는 있지만, 위 오차 증가율을 전체 운전에서의 시각 비중으로 해석하면 안 된다. 익숙한 장면·몸 반응·누적 신경 상태의 기억과 시각 반응은 양립한다. 이번 optimizer 업데이트는0이다. 이전 head의 개발 전체.68%/진입 전1.19% 개선을 이번에 새로 학습한 성과로 세지 않는다. 새 코너·생리/R2 선행 판단은 미완료다.

## 실제 움직임을 위한 실행 경계

`shadow_native_flight_teacher_motion.py`는 기존 특권 SAC 교사만 vJoy를 소유하게 하고, 학생은 영상·몸 감각으로 추론만 한다. 교사의 지도/곡률/조작은 학생 입력에 들어가지 않는다. 고정10/20km/h 각2초·상한40km/h, 기존 source/50ms 교사 actuation guard·track/gear 조건을 유지했다. AC 창 활성화·클릭·키 입력은0이며 재시작/재개/일시정지는 CSP API로만 수행했다.

실행은 기존 네 시점 촬영 순서의 FlyVis 경로와 단일 CPU 감각 소유자를 사용한다. 현재 legacy21/native19는 같은 native packet/첫 receipt를, 시야별 과거 몸은 원래 수신값을 사용한다. 기존 **몸 출력50ms·시각150ms·source→render25ms** 기준을 유지했다.

준비 문제를 다음과 같이 수정했다.

- `teacher_shadow_camera`와 `OwnedSourceCamera`에 명시적 native-ready barrier를 추가했다. 교사의 모델/물리 상태 사전 검사가 끝난 뒤 native 프로세스를 준비하고, 완료 영수증이 있어야 촬영과 owned resume를 시작한다. 오래 쌓인 일시정지 native 이력을 첫 context RPC에서 한꺼번에 처리하지 않는다.
- native encoder/selector를 준비한 뒤 sampler를 시작한다. 마지막 실행은 native 시작 요청부터 준비 완료까지 약2.947초였다. 선택적인 시작 대기는30초, 해당 준비 구간의 교사 카메라 대기는45초다. **이는 촬영·운전 전 프로세스 준비 대기이며, 감각/출력의50ms·150ms 기준을 늘린 것이 아니다.** 원래 기본 시작 대기는 유지한다. 준비 전에 멈춘 소유자도 생산자 미시작과 빈 원본을 명시적으로 기록한다.
- 온도 검증 뒤 중립이 된 차는 검증된 gear-only vJoy 버튼으로 첫 기어를 확인한 직후, 제어기를 해제하고 API로 일시정지한다. 가속 요청은0이다. 실제 gear2·자동 변속·풀브레이크가 유지되는지 확인하며, 신경 상태를 중간에 재설정하지 않는다.

## 온도 준비의 실패도 원본으로 남겼다

`motion-preparation-01`은 epoch8, `motion-preparation-02`는 새 epoch9다. 각 epoch의80℃ setter는 한 번만 호출했다. 초기 readback 실패를 재설정으로 덮지 않고, 원래 성공한 SDK 호출 영수증에 바인딩한 **읽기 검증**으로 서로 다른 세 native 패킷의 실제 네 타이어80℃를 확인했다.

epoch8 첫 거절 표본은 당시 helper가 저장하지 않아 결손이다. 이를 현재값으로 복원했다고 주장하지 않는다. 이후 helper는 거절한 same-copy 원본도 저장하도록 수정했다. epoch9의 `initial80/rejected-native-preflight.json`에서는 속도 채널 **1.4485km/h**, gas0/brake1/AI-off/gear1/타이어80℃가 관측됐다. 초기 조건을 정지로 간주하지 않고 거절한 것이다. 이 초기 속도 관측의 원인은 별도로 분리하지 않았다. 설정 직후 표본은 학습/주행 성적으로 사용하지 않는다.

`audit_initial_thermal_evidence`는 readback-only의 내장 사본과 원래 요청/terminal/publication·해시를 대조하고 새 raw native의 AI·속도·브레이크·온도·packet/receipt 순서를 재계산한다. 원래 실패 run은 그대로 실패다. `prepare_native_motion_condition.py`는 새 epoch의 단일 설정→독립 읽기 검증→gear-only freeze를 재현한다. 준비 lap은 평가에서 제외한다.

## 네 움직임 시험 — 전체 성공으로 판정하지 않음

| 실행 | 교사/원인 | 학생 출력 | 판정 |
|---|---|---:|---|
| `moving-shadow-01` | 온도 검증 후 중립. 교사 시작 전 거절, 제어0·촬영0 | 0 | 전체 실패; `moving-pretrigger-audit-01` |
| `moving-shadow-02` | 긴 준비 중 쌓인 native 이력 처리에서 queue full/응답 불명. 교사는 풀브레이크 open/close2회, 정책 요청0 | 0 | 전체 실패·sampler 종료 실패 원문 보존; `moving-failure-audit-02` |
| `moving-shadow-03` | 교사101정책 요청으로10→20km/h 완료, feedback 최고19.847km/h. 별도 native 원본 최고14.223km/h·tyre-out 최대1; 학생 소유자 시작 응답5초 초과 | 0 | 전체 실패; `moving-failure-audit-03` 완료·whole=false |
| `moving-shadow-04` | 새 epoch9·80℃·첫 기어·준비 barrier 통과. 이후 시각 대기 기한 초과 | **93** | **전체 실패**; `moving-audit-04` |

마지막 실행의 독립 재계산 결과:

| 항목 | 확인값 |
|---|---:|
| 출력 수/연속 구간 | 93개 / 1.0025초 |
| 같은 영상에서 추가 몸 갱신 출력 | 72개 |
| 출력 시점 몸 감각 나이 최대 | 14.922ms |
| 가장 오래된 view 나이 최대 | 130.677ms |
| 추론 p95 / 최대 | 10.794ms / 11.900ms |
| 출력과 같은 packet의 차속≥2km/h | **8출력**, 그 출력들의 최고 차속3.813km/h |
| 전체 native 관측 최고 차속 | 7.192km/h — 출력 구간 밖 관측도 포함 |
| 학생 차량 제어 | **0** |

학생93출력을 모두 움직임 출력이라고 부르지 않는다. 교사는 정책17회, open/close 포함 제어19회를 보냈고 중단 요청 후 풀브레이크로 해제했다. 원본 native/이미지·source witness·첫 receipt·연속 상태·publication·50/150ms 나이와 native/교사/카메라 종료는 독립 확인했다. 해당 하위 자원의 종료 확인과 GPU 부모 프로세스 전체 종료 영수증은 구분한다.

## 마지막 지연과 수집 영수증 문제

마지막 게시 영상은 capture32였다. 그다음 capture33은 `visual_seconds=0.1238514`로 처리되며 게시되지 못했다. 게시에 성공한 영상만 보면 처리 p95는18.60ms라 이 실패가 숨겨진다. 정상 prefix의 p95는29.13ms였다. 실패 프레임도 포함한 `moving-phase-analysis-04.json`을 따로 남겼다. frame34의 원래 render 하한55500.6381에 대한 host receipt가55500.7469로 늦어졌다.

그동안 기존 시각이150ms를 넘겨 `View render bound stale while waiting for new scene`으로 종료했다. 유효 출력의 기준을 넓히거나, 실패 뒤 같은 회로를 재설정해 계속하지 않았다. **123.85ms wall time에 Python/GC·GPU 작업·동기화가 각각 얼마나 들어갔는지는 아직 분리하지 않았다.** 다음 개선 대상은 이 tail 지연이며 몸 회로 계산만의 지연으로 단정하지 않는다.

외부 launcher에서도 별도 결함을 찾았다. run01의 venv redirector는 실제 자식과 제어 자원의 종료를 확인한 뒤 명시적으로 정리했으며 원래 실패를 재분류하지 않았다. run04 watcher는 NAS status 파일의 WinError5 atomic rename에서 종료해 원래 자식 핸들의 종료 영수증을 수집하지 못했다. 재시도 가능한 공통 `write_json`을 사용하고, 상태 기록 오류가 나도 자식 핸들을 버리지 않도록 수정했다. 이 수정은 회귀 검사까지이며 새 실제 GPU 실행에서 재검증하지 않았다.

run03 GPU 부모36336의 Win32 종료 코드는1로 확인했다. 원래 watcher21676은 여전히 대기하며 원래 핸들의 exit 파일은 없다. **GPU 부모 전체 종료 영수증까지 모두 정상 수집했다고 주장하지 않는다.** `final-session-01.json`/`process-identity-followup-01.json`에 상태를 남겼다. PID44420은 현재36336의 `conhost.exe`로 재사용됐으므로, 과거 run02 native sampler라고 보고 종료해서는 안 된다.

## 검증·현재 상태·다음 작업

- `tests-final-02.txt`: 관련 **86검사 통과**. default/새 준비 계약, 감사/해시·dtype 보존, 중간 reset 거절, metadata 오류 시 원래 자식 대기 유지, thermal/gear/50ms 감각 경계를 검사했다.
- `summary-01.json`, 각 manifest/result·실행 소스, `source-receipt-final-01.json`/`source-snapshot-final-01/`에 원본·해시·실패·소스 버전을 보존한다. run03의 옛 worker도 원래 manifest SHA와 일치하는 코드로 보존했다. source alias/첫 receipt와 새로운 native 감각을 donor 것으로 바꾸지 않는 조건도 확인했다.
- 최종 fresh API/드라이버 확인: epoch9, 일시정지·카메라 비활성·AI-off·gas0/brake1·vJoy FREE. 실제 학생 자율 완주/한계 주행·온도 적응·새 트랙/차종 일반화는 미검증이며 **actor_qualified=false**다.

다음은 실패 프레임을 포함해 영상 CPU 전처리/GC/GPU/동기화 시간을 나누고, 보존한 움직임 픽셀과 원래 시각으로 parity를 확인하며 tail 지연을 줄이는 작업이다. whole moving shadow와 종료 영수증까지 통과한 뒤, 같은 연속 회로 상태를 유지하는 제한된 학생 폐루프 조향·가감속/하중·타이어 되먹임으로 연결한다. 도로 상대 위치/자세를 맞춘 영상 대조와 별도로 봉인한 새 코너/트랙, 다른 실제 온도·자세/가속도 조건, 변속·셋업/차종/상대풍 적응·유효 완주·빠른 라인/랩타임 검증을 남긴다. 정지/부분 추론·소부품 실행을 그 목표의 완료로 바꾸지 않는다. 정확 GPU 노출/physics atomicity·생리 전체 검증의 한계도 유지한다.
