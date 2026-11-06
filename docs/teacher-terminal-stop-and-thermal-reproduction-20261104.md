# 교사 종료 제동 수정과 온도 초기화 재현 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 변속·차종/타이어/셋업 적응·실제 검증된 상대풍, actor의 지도/world pose/절대 heading/교사 명령 배제를 유지한다. 실측 연결·생리 근거·기능 근사·차량 대응·학습한 경계와 공학 교사를 구분한다.

전체 목표는 **active**다. 이전 턴은 CPU 수치 지연·종료 경계의 수정과 실제 실패 증거를 확보했으므로progress다. 이번에는 교사의 저속 횡방향 추종과 정지를 수정해 **80℃의 전체 움직임 shadow 한 번**을 독립 감사까지 통과했다. 학생 제어와 optimizer 업데이트는0이다. 40℃ 초기화 재현은 실패했으며, 이를 온도 적응이나 다른 조건 주행 성공으로 바꾸지 않는다.

## 원래 실패와 수정 범위

증거 루트 `runs/teacher-braking-20261104/`. 이전 `runs/visual-tail-20261104/moving-shadow-04`는 종료 직후 이탈1이384행/321고유 packet으로 지속돼 whole=false였다. 이번에 앞 카메라 capture109/121/134를 원본 archive에서 추출해 확인했다(`front-00/01/02.bmp`). 제동 전부터 왼쪽 경계에 접근했고 마지막 source feedback gap은-0.832m였다. 마지막 SAC 조향은+0.00352였고 close는0이었다. 종료 조향 중립 하나가 원인이라고 확정하지 않는다.

`old-lateral-replay-01.json`은 원래101개 요청의 SDK pose/native heading을 동일한 geometry로 재구성했다. 마지막 상태의 경로 보정 조향은+0.21이었다. 이 비교는 개입을 실제 적용한 반사실 궤적이나 회복 성공을 뜻하지 않는다.

새 선택 모드 `--teacher-lateral-calibration`/`--low-speed-lateral-calibration`은 다음과 같다.

- 주행 단계의 SAC 가감속과 기존 속도 governor를 유지하며 교사 조향만 경로 기하로 계산한다. world pose/heading/route는 **교사 전용**이다. 학생 native19/legacy21/시각 입력에 넣지 않는다. actor 회로·체크포인트는 바꾸지 않았다.
- 기존 `GeometricRecoveryTeacher`와 역사적 실측 curvature gain -0.0986716을 사용한다(`ac-csp-teacher-kinematic-fit-20261028-t180`). 당시/현재 profile SHA는 다르며, 동일 BMW·축 binding/DLL·SDK 조향 응답 gain449.9999의 호환성을 확인했다. **현재 움직임 curvature를 다시 실측 적합한 것은 아니다.** 이 전이는 교사 공학 가설이며 새로운 생리 회로가 아니다. 요청 범위는 실제 정규화 조향±0.21이다.
- 10/20km/h 각2초 뒤 별도 정지 단계를 둔다. 경로 조향을 유지하면서 가속0/제동1, 같은40ms 요청 주기와 source50ms·원래 입력50ms·트랙/기어/AI/프로필/배경 상태 guard를 적용한다. 최대3초 안에 서로 다른 native packet3개에서 속도≤0.5km/h·실측 brake≥0.98을 확인한 뒤 원래 중립 close/제어권 반환을 한다.
- 정지 확인을 포함해 선택 모드의 read-only 관측 예산은8초다(기본은6초 유지). **감각 기준 몸50ms·영상150ms·source→render25ms를 늘린 것이 아니다.** 회로 정지 후 큰 기록을 저장하며, 상태를 중간 reset하거나 거절 결정을 재시도하지 않는다.

## 실제80℃ 움직임 시험과 독립 감사

`motion-preparation-01` epoch14, `moving-shadow-01`, `moving-audit-01`, `moving-analysis-01.json`이다. 실제80℃ 초기조건은 원래3 native packet 감사로 확인했다. 교사만 vJoy를 소유하고 학생은 시각/몸 감각 추론만 수행했다. AC 창 활성화·클릭·키 입력0, lifecycle은 CSP API만 사용했다.

| 항목 | 독립 확인값 |
|---|---:|
| 유효 출력/연속 구간 | 569 / 6.1748초 |
| 차속≥2km/h의 출력 | 375 |
| 출력 packet 최고 차속 | 19.834km/h |
| 몸/시각 나이 최대 | 17.410/123.627ms |
| 계산 p95/최대 | 11.216/13.876ms |
| native 타이어 이탈 최대 | 0 |
| 정지 요청/소요 시간 | 16 / 0.6413초 |
| 서로 다른 정지 확인 packet | 3 |
| 학생 차량 제어/optimizer | 0 / 0 |

독립 감사는116개 주행/정지 조향을 원본 SDK wire·native heading·고정 geometry로 다시 계산해 최대차0을 확인했다. 정지 feedback의 packet·speed/brake/tyres-out를 원래 raw pair와 대조하고 마지막3개가 close보다 앞서는지 검사했다. 전체 source/native의 이탈0, 원래 첫 receipt, 연속 상태, 몸/영상 시간, CPU owner/native/camera/교사 종료와 GPU 원래 자식 핸들의 exit0를 확인했다. whole_shadow_passed=true다. 그 이전 실패 원본은 보존한다.

**이것은 같은 BMW/Monza의 저속 교사 주행 중 학생 추론 한 번의 통과다.** 학생의 폐루프 조향·가감속, 높은 횡가속/한계 주행, 다른 트랙/차종·타이어/셋업 적응이나 반복 안정성을 입증하지 않는다. 기존 조향 head의 개발 .68%/1.19% 개선·동시2% 기준 미달과 시각 기대 방향6/15도 그대로다. 미승격/actor_qualified=false를 유지한다.

## 40℃ 재현 실패 — 실제80℃와 요청40℃를 구분

기존 역사적40/80/110℃ 초기화의 native3패킷 증거(`runs/thermal-initial-20261103/`)는 보존한다. 이번 재시작 흐름에서는40℃가 재현되지 않았다. API가 반환됐다는 사실을 실제 적용으로 부르지 않는다.

| 시도 | 실제 관측/결과 |
|---|---|
| `motion-preparation-40c-01`, epoch15 | SDK40℃ 요청1회. 첫 resume의 정지 guard 거절을 보존. 같은 SDK 영수증의 읽기 검증은89서로 다른 packet 모두80℃라 실패 |
| `motion-preparation-40c-02`, epoch16 | setter 전 native 준비 진단은5초·237표본이 같은 packet1개라 실패; 그 단계의 setter0 |
| `post-live-40c-initial-01`, epoch16 | 별도 명시적 pause 뒤 최초40℃ setter1회. 첫 resume native 정지 guard 거절; setter를 재호출하지 않음 |
| `post-live-40c-readonly-01`, 같은epoch16 | 추가 resume/설정0인 읽기 검증.92서로 다른 packet 모두80℃로 실패 |
| `motion-preparation-40c-03`, epoch17 | 검증된 기어 준비를 온도 설정 앞으로 옮긴 선택 진단. 새 native packet timeout으로 중단, 온도 setter0. vJoy 요청2개는 neutral open/close·모델 요청0 |

따라서 ‘native가 갱신되지 않아서만40℃가 안 보였다’는 설명은 충분하지 않다. 다른 읽기 검증에서는89/92개 고유 packet이 실제80℃였다. 재개 직후 native 정지 거절과 패킷 정체는 원본으로 남기며, 이를 데이터 손상·수면·초기화 순서 중 하나로 확정하지 않는다. 설치 SDK의 setter 인자/℃/destination15와 설치 Lua 호출은 일치했다. 배포 Lua SHA `b2227739...`는 변경하지 않았다. SDK 호출이 실제로 언제 적용되는지와 재개/첫 물리 갱신의 관계가 미검증이다.

`--temperature-c`는40/80/110 선택을 명시하고 epoch당 한 번만 설정한다. 실패한 새 준비 진단은 `--verify-native-initialization`/`--gear-before-thermal`로 분리했으며 기존 기본80℃ 준비를 바꾸어 실패한 진단을 강제하지 않는다. 이 옵션들의 현재 실패도 숨기지 않는다. ±2℃ 초기3패킷, 시작±5℃와 모든 감각 유효성 기준을 완화하지 않았고 온도를 다른 숫자로 보정하거나 재라벨하지 않았다. **이번40℃ 움직임 및110℃ 시험은 실행하지 않았다.**

## 구현·검사·현재 상태·다음 작업

- `teacher_terminal_stop`, `teacher_lateral_stop_evidence`: 명시적 교사 전용 저속/정지 경계 및 원본 조향/정지 재계산. 원래 SAC 기본 경로와 VJoySession.close는 유지한다.
- `verify_native_initialization`/`native_initialization_readiness`: 설정 전 읽기 전용 준비 진단. 실제 native packet 중복으로 실패한 경우도 보존한다. actor 입력·추론으로 사용하지 않는다.
- native ticker의 `maximum_native_body_delta` 기록을 추가했다.80℃ 통과 실행 **뒤** 추가한 항목이므로 unit 검사와 실제 실행 검증을 구분하며 새 필드가 live 검증됐다고 주장하지 않는다.
- `tests-02.txt`의1실패는3초 경계 오류 문구의 assertion 불일치였다. 실제 deadline 거절은 유지하고 assertion을3s 경계로 맞췄다. 최종 `tests-final-01.txt` **89검사 통과**(28.36초). callback/신선도/이탈/중복/기어/시각·몸/GC/FPU 및 기존 기본 경로를 검사한다. 이전 반복 합산이 아니다.
- 원본·실패·실행 소스는 각 run과 `source-receipt-final-01.json`/`source-snapshot-final-01/`에 남긴다. 네 보호 체크포인트와 시각 스트림의 원래 parity pin을 재확인한다.
- 최종 `final-session-01.json`: epoch17·paused·live=false·카메라/AIoff·gas0/brake1·vJoy FREE·실제 네 타이어80℃. epoch14는80℃ 설정1회, epoch15/16은40℃ 설정 각각1회 소모했고 epoch17은온도 설정0회다. 현재 실제 시험은 종료했다.

다음은 SDK 온도 호출 전후의 SDK/native 값과 첫 물리 갱신을 별도로 기록해 초기화 재현 문제를 해결하는 것이다. 성공한80℃ 실행 경계를 바탕으로 제한된 학생 폐루프의 조작 단위/부호·신선도·독립 정지 계약도 구현·검증해야 한다. 도로 상대 자세/위치가 맞는 시각 대조, 새 코너/트랙의 봉인 검증과 실제 다른 온도·자세/가속도에서의 기능/인과 기여를 이어간다. 조향과 가감속·하중/타이어 되먹임·변속·셋업/차종/검증된 상대풍·유효 완주/빠른 라인·랩타임, 정확 GPU 노출/physics atomicity·전체 생리 검증은 미완료다. 이 준비/추론 통과로 원래 목표를 완료 처리하지 않는다.
