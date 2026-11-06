# 준비 이후 조작 불일치와 실제 고온 적격 시연 — 2026-11-04

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준, 가감속·변속·타이어/셋업/차종·검증된 상대풍 적응을 유지한다. 지도/world pose/절대 heading/교사 명령은 actor 입력에 넣지 않는다. 실측 구조·생리 근거·기능 근사·학습한 경계를 구별한다.

앞선 턴은 progress다. 실제 출력 재구성·감각 개입·초기 상태 readout 문제와 연속 준비 경계를 구현/검증했다. 이번에는 새 조작 비교와 적격 시연/학습 입력을 만들었으므로 progress이며 전체 목표는 active다. 실제 학생 한계 주행은 여전히 미완료다.

## 준비 이후 무엇이 달랐는가

`moving_teacher_command_comparison`과 `compare_prepared_teacher_commands`는 원래 냉간/고온 shadow의 학생 출력을 과거 수락된 교사 요청과 비교한다. 기존 학습과 같은 조향 `requested steering/.6`, 분리된 throttle/brake를 사용한다. 학생 decision time 이전에 driver call이 끝난 가장 최근 정상 curriculum 요청만 고르고 최대60ms를 넘으면 결측으로 남긴다. 0.5초/다중 촬영 준비 전, 미래 요청, 종료 stop 요청을 정상 목표로 채우지 않는다. 반복 몸 tick의 과대표현을 줄이는 command-balanced RMSE도 별도로 계산한다.

| 준비 후 비교 | 냉간 | 고온 |
|---|---:|---:|
| 비교 사건 / 고유 교사 요청 | 370 / 100 | 373 / 100 |
| 조향 RMSE / command-balanced | .353282 / .353365 | .310650 / .311219 |
| 가속/제동 RMSE | .284707 / .198935 | .330079 / .178494 |
| 유의미한 조향에서 반대 부호 | 233/300 | 231/303 |
| 첫10km/h 단계의 반대 조향 부호 | 185/185 | 186/186 |
| 가속과 제동 모두>.1 | 278/370 | 305/373 |

이 결과는 **과거 요청과의 불일치**다. driver accepted와 게임에 실제 적용된 조작은 다르며 game-applied 증거는 false다. 두 관측은 완전히 동일한 몸 snapshot도 아니다. 교사는 저속10/20km/h curriculum 및 privileged geometric recovery이고 학생은 외부 레이싱 ceiling300km/h로 실행했다. 따라서 페달 차이를 같은 주행 목표의 정답 오차로 해석하거나 해당 준비 자료를 바로 레이싱 actor에 학습하지 않는다. 큰 조향 부호 차이는 초기 readout만 제거한다고 선행/복구 판단이 검증되는 것이 아님을 보여 주지만, 최적 racing 조향의 반사실 정답을 증명하지 않는다.

원본은 `runs/thermal-visual-motion-20261104/{cold40,hot110}-command-comparison-01`이다. 두 `*-command-audit-02`가 raw 요청을 별도로 탐색해 단위·미래 배제·준비/결측 mask와 단계별 RMSE/MAE/부호/반복 명령 가중을 재계산했다. 관련13검사 통과. 첫 cold 감사는 float32 저장 target과 Python float64 target의 exact 비교 때문에 실패했다. dtype에 맞는 독립 양자화 비교로 고쳤고 최초 소스/실패 로그를 보존했다. 지표나 허용 시각을 완화하지 않았다.

## 새 실제 고온 시연 수집과 적격성

새 루트 `runs/thermal-control-improvement-20261104/`의 `hot110-preparation-01`은 epoch6에서 SDK110℃ 설정1회·사전/사후 실제 정지 packet·최종 pause와 독립 초기 온도 감사를 통과했다. `hot110-gear-01`은 host를 paused에서 준비한 뒤 CSP API로 재개하고 기어만 준비해 다시 pause했다. AC desktop 입력은0이다. 기존 설치 Lua SHA `0c20d720a7a2bf20a7e30fdc7370ee5e9932babf71fb360a78a5f07719360b7e`를 유지했다.

`hot110-demonstration-01`은 기존7M 교사만 운전했다. 학습용 actor 진로 입력으로 지도나 교사 관측을 추가하지 않았다. 원래 수집 launcher handle66695 exit0·teacher/camera 모두exit0다. `hot110-qualification-01`은 별도 lap proof와 영상/native 정합 검사를 통과했다.

- 독립 유효 교사 랩1회: **112.320초**, post-stop 검증true. 학생 랩 성적이 아니다.
- 최종 영상 **1,565그룹**, 적격 **1,561**. 첫4그룹의 부적격 근거와 원본을 보존했다. 진행 중 health의1,564는 최종 archive 수를 대신하지 않는다.
- 전체 랩 라벨 coverage **1.0**, native 원본25,697개. source-to-render 최신 bound 최대9.1815ms, source-to-driver 최대26.6442ms.
- 정합 기준은 이전 계획에서 source SHA만 현재 설치본으로 바인딩했다. timing/pixel threshold는 바꾸지 않았으며 `alignment-plan-lineage-01.json`에 기록했다. 정확 GPU 노출/physics atomicity가 입증된 것은 아니다.
- 시연 수집 자체는 unqualified 원본이고, 독립 certificate가 `training_admissible=true`를 부여했다. 초기 설정/준비 구간의 기존 제외 규칙은 유지한다.

## 현재 실행 식별과 실제 학습 입력

처음 사용한 과거 static witness는 창531348을 가리켜 현재 창1117858과 달랐다. `hot110-actor-data-01`은 실패 결과를 남겼으며 학습에 쓰지 않는다. 현재 epoch6/동일 generation의 실제 static mapping을 두 번 읽고 엔진 window/PID를 읽기 전후 확인한 `native-static-identity-01.json`을 새로 확보했다. 원래 witness는 수정하지 않았다.

`hot110-actor-data-02`는 현재 식별로 전처리를 완료했고 원래 실행 handle61974 exit0다. **16,172 연속 사건·전체 랩 감독2,808개·native-only 무라벨12,692개**를 보존한다. 준비 구간672라벨은 기존 규칙대로 burn-in에만 사용한다. unqualified pixel input0, 시작 영상0–3 제외 기록을 남겼다. 원래 시각·몸 마스크와 episode state를 유지하며 renderer/world truth는 actor 배열에서 제외한다.

식별 검사를 GPU 초기화 앞으로 옮겨 같은 잘못된 witness가 큰 모델을 올리기 전에 거절되게 했다. 실제 stale-witness CLI 검사 handle79004가 GPU 초기화 로그 없이 기대한 exit1/failed를 반환했다. 원래 전처리 소스는 `source-before-identity-preflight-01`에 보존했다.

최초 실패 전처리는 failed 결과 뒤에도 teardown에서 남아 있었다. `Stop-Process` PID lookup은 실패했고, CIM에서는 동일 PID/명령을 계속 관측했다. 두 API의 관측을 종료 성공으로 해석하지 않았다. 이후 동일 CIM 객체의 실패 Python PID29776만 Terminate(return0)했고 원래 handle70244 exit1을 확인했다. 실패 결과를 성공으로 바꾸지 않았으며 성공한 새 전처리는 건드리지 않았다. 두 정리 영수증에 각 API의 실패/효과를 구분한다.

## 온도 범위가 새로 넓어졌는가

`hot110-coverage-01`은 적격 full-lap fit mask의 정확한 몸 packet2,808개로 원본 native bytes를 다시 골랐다. 설정값110℃를 실제 입력으로 대체하지 않는다.

| native 바퀴 index | 감독 온도 최소℃ | 최대℃ |
|---|---:|---:|
| 0 | 81.3584 | 99.0141 |
| 1 | 73.5119 | 92.1990 |
| 2 | 75.5507 | 97.7882 |
| 3 | 70.1363 | 94.9092 |

**2,808개 모두 이전 바퀴별 학습 온도 min/max 안이었다.** 이번 자료는 검증된 새 실제 시연이지만 온도 지원 범위 확대의 증거는 아니다. 기존 min/max는 바퀴별 약42.72–100.98℃를 포함한다. 출발 직후40/110℃ shadow 전체의 OOD 표시와 완주 감독 시점의 지원 범위를 구별한다. min/max 안이라는 이유로 조건 조합·빈도·자세/하중/슬립 적응이 입증된 것도 아니다.

새 시연은 수집 전 `condition-extension-plan-01.json`에서 후속 TRAIN 후보로 지정했다. 봉인된 일반화 시험으로 다시 이름 붙이지 않는다. 이번 optimizer 업데이트와 학생 actuator는0이다.

## 다음 작업

새 적격 full-lap actor 자료에 공간 시각/native sidecar를 원본과 정합해 고정된 현재 모델의 새 시연 조향·페달 오차와 시각 기여를 평가한다. 새 냉간 조건과 필요한 출발/복구 감독을 추가할 때는 외부 ceiling/레이싱 목표와 교사 curriculum 차이를 명시하고, 원래 제외 mask를 덮어쓰지 않는 별도 적격성 기준이 필요하다. 원본 body/scene과 비행 회로를 보존하면서 실제 필요한 감각 경계 학습으로 연결한다.

도로 상대 자세/위치를 통제한 선행 시각 방향6/15 문제, 실제 학생 폐루프·차체 자세/가속도/하중/타이어·변속/셋업/차종/검증된 상대풍 적응, 미학습 코너/트랙의 유효 완주·빠른 레이싱 라인과 랩타임, 생리/R2 검증은 남아 있다. 초기 준비와 입력 반응·부품 실행을 위 목표 완료로 대체하지 않는다. 최종 epoch6은 CSP pause 완료·카메라/AIoff·gas0/brake1이며 실제 시연/전처리는 종료했다.
