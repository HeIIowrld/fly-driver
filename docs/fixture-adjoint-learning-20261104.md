# fixture 시각 입력 미분을 보존한 속도 경계 연속 학습 — 2026-11-04

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

현재 bounded speed-policy와 Adam4그룹1580/1324/331/331을 그대로 이어받아 stop_all/input_adjoint를 각395회 학습했다. 수정 조건의 진입 전 조향 RMSE 감소율(냉간/고온/개발)은 같은 출발 대조군 대비 -0.520% / -1.124% / -0.519%이며 직전 부모 대비 +0.907% / +2.917% / -0.725%다. 실제 영상 방향25/30·목표13/30이다. 음수는 악화다. 수정 경로는 이제 실제 학습됐지만 두 후보 미승격·실제 한계 주행/타이어 적응 미검증·목표 active.

기존 fixture 경계는 speed 계수를 고정하려고 출력 전체를 detach해 앞단 시각 입력의 gradient도 끊었다. 이번 input_adjoint는 같은 forward 수치를 유지하면서 speed weight/bias만 detach한다. 합성·retimed 영상에서는 speed/native coefficient gradient=None과 Adam 이동0을 유지한다. 시각 회로를 지나는 미분만 frozen speed 함수의 입력을 통과한다. 이는 공학적 학습 경로 수정이며 측정 시냅스나 생리학 기능을 새로 검증했다는 뜻이 아니다.

두 조건의 실제 출발 파라미터 차이0, 저장된4그룹 Adam moment/step을 그대로 이어갔다. 둘 다 motion8192/native9246/speed8 총17446 파라미터를 동일 데이터·목적함수·seed·순서로 학습한다. 신규 속도 계수0 초기화가 아니다. 기존 TRAIN5659점 정규화·clip5·D10m/s·power 직접 민감도2000·speed 직접 계수≤0·고정 화학 그래프와 signed head를 유지했다. native 총 변화.05/entry.02는 fixed-full 부모를 계속 기준으로 삼으며 출발점을 리셋하지 않았다. motion 전체 norm2와 speed norm.5/bias.5도 원래 전체 기준이다. actor에는 영상과 허용된 몸 감각만 들어가며 지도·world pose·교사 명령·미래 곡률은 들어가지 않는다.

각32597 실제 결정을 자체 연속 수치 상태로 처리했다. 실제 감독301·원하는 명령30·TRAIN 영상교환10·기존 합성54로395갱신했고 부모 prefix 주입0이다. gradient detach는 numerical reset이 아니다. 각1020 window hash 체인과 모든 Adam 제안·power weighted-L1 투영을 독립 감사했다. 최종 step은1975/1655/662/662다. 실제 gradient FD와 primed optimizer의 fixture gradient/momentum 차단·14개 strict-loader 계약을 확인했다. 정해진 마지막 가중치를 모두 보존하고 개발/새 영상으로 checkpoint를 선택하지 않았다.

각 최종 가중치로48866결정의 세 기록을 재생했다. 아래 수치는 조향 RMSE 감소율이며 음수는 악화다. source의 original은 이전 fine 기준이며, 그 기준의 고정목표 몸 감쇠0/30·반대30/30이라는 과거 국소 검사도 함께 해석해야 한다. 낮은 label replay 오차만으로 실제 제어 품질을 판단하지 않는다.

| 조건 | 진입 전 원래 fine 대비 | 진입 전 직전 부모 대비 | 영상 방향/목표.002 | 새 합성 방향 |
|---|---|---|---|---|
| stop_all | +2.566% / +5.996% / +3.021% | +1.420% / +3.996% / -0.205% | 25/30 / 13/30 | 18/18 |
| input_adjoint | +2.060% / +4.939% / +2.517% | +0.907% / +2.917% / -0.725% | 25/30 / 13/30 | 18/18 |

| 조건 / 기록 | 전체 조향 감소율 | 전체 가속 감소율 | 전체 제동 감소율 |
|---|---|---|---|
| stop_all / 0 | -25.519% | -4.397% | +14.914% |
| stop_all / 1 | -31.407% | -8.433% | +15.704% |
| stop_all / 2 | -29.616% | -8.292% | +14.534% |
| input_adjoint / 0 | -25.056% | -4.207% | +14.887% |
| input_adjoint / 1 | -30.771% | -8.394% | +15.662% |
| input_adjoint / 2 | -29.045% | -8.226% | +14.520% |

동일 출발 대조군 대비 진입 전 조향 감소율은 -0.520% / -1.124% / -0.519%다. 이는 fixture 입력 미분 규칙 변경을 포함한 공동 학습 전체 효과다. gradient가 바뀌면 이후 상태와 실제 데이터 gradient도 달라진다. 한 시점의 speed 계수만의 독립 효과로 해석하지 않는다. 같은 Monza/Z4 GT3의 이미 소비한 개발 기록이며 새 트랙 일반화가 아니다. 새 texture915320267은 이번 학습 후 평가로 소비했고915320293은 미생성 계획이다. 가설 도로/몸/온도를 실제 타이어 적응으로 세지 않는다.

첫 차이는 합성 fixture의 시각 gradient에서 발생했고 그전 파라미터/gradient/상태 출발과 schedule이 일치했다. 첫 fixture의 수치 목적함수 차이0이며 native/speed gradient와 parameter 차이도0이었다. source matched-training-audit에 초기와첫분기 수치를 남겼다.

영상 개입은 수용자의 몸/native/clock/학생 prefix를 유지한30 donor 조합이다. donor 특징은 원래 FlyVis 이력을 포함한다. visual_change_off/native_change_off는 이번 증분을 부모로 돌리고, motion_off는 원래 motion residual만 끈다. speed_policy_off는 부모부터 이어온 전용 속도 경계 전체를 중립화하므로 이번 gradient 수정만 제거한 조건이 아니다. 두 조건을 zero-speed negative control로 읽지 않는다. 실제 RGB의 동일 FlyVis 수치 상태/held retina를 복제한 이전 feature bundle은 hash 확인 후 재사용해 각 후보의 자체 prefix를 맞췄다. 한 TRAIN 사례이고 retimed RGB라 실제 물리 반사실/새 코너 정답은 아니다.

| RGB 조건 | input_adjoint margin | stop_all margin |
|---|---|---|
| fine | 0.001954729 | 0.001908920 |
| visual_change_off | 0.001313158 | 0.001284863 |
| native_change_off | 0.001955212 | 0.001908881 |
| speed_policy_off | 0.002386668 | 0.002377819 |

| input_adjoint 채널 변경 | 방향 | 목표.002 | 평균 margin |
|---|---|---|---|
| full | 25/30 | 13/30 | 0.001896460 |
| coarse_static | 18/30 | 1/30 | -0.000211182 |
| fine_motion | 25/30 | 15/30 | 0.002222811 |
| identity | 0/30 | 0/30 | 0.000000000 |

coarse_static에도 운동 정보가 있고 fine_motion은 원래/new motion→DN 양쪽을 바꾼다. 비선형 상호작용 때문에 독립 기여율로 합산하지 않는다.

| 몸 감각 검사 | yaw 감쇠 | speed 감쇠 | yaw 변화>1e-5 | speed 변화>1e-5 |
|---|---|---|---|---|
| feedback / fixed_reference | 45/45 | 45/45 | 45/45 | 43/45 |
| feedback / actor_policy | 45/45 | 45/45 | 45/45 | 17/45 |
| stop_feedback / fixed_reference | 45/45 | 45/45 | 45/45 | 43/45 |
| stop_feedback / actor_policy | 45/45 | 45/45 | 45/45 | 17/45 |

위 검사는 현재/모든 view history의 감지 yaw/speed만±.01 바꾸며 영상/native/clock/학생 상태를 동일하게 시작했다. fixed_reference는 후보 자체 factual 원하는 명령 trace를 고정하고 actor_policy는 자연스럽게 재계산한다. 국소 모델 부호와 변화 크기이며 차량 폐루프 안정성 증거가 아니다.

정규화된 대칭 특징과 이전 MN 군 평균에서 입력을 독립 재구성했고 speed command/power 전류를 NumPy로 감사했다. b1/b2/power L/R의 모델 운동뉴런 군 평균이며 실측 근육 힘이 아니다. 고정 합성 직선 특징/몸을3속도×2온도×2후보×속도경계on/off의24독립0상태 에피소드에서40ms×301결정 처리했다. 마지막2초 action peak-to-peak 최대는 0.0다. 유한 합성 조건의 수치 수렴과 전체 회로/차량 안정성 증명을 구분한다. native 가속도/온도의 최신 학습분 reference-clamp도 실제 타이어 물리 변화가 아니다. 원본 native bytes와camera/body clock128결정×2 runtime parity는 live 시간 보장/완주 증거가 아니다.

입력 미분을 바로잡은 것을 운전 성능 성공으로 세지 않는다. 이번 matched 비교의 진입 전 RMSE는 수정 조건이 대조군보다 약.5–1.1% 컸고 영상 방향은둘다25/30이다. 원본 RGB는 수정 margin약.001955·대조군.001909로 소폭 차이나 둘다.002 미달이다. coarse_static 변경의 평균 signed margin은 음수이고 fine_motion 단독보다 full 평균 margin이 낮았다. 이는 이번 consumed retimed 조합의 경로 간섭 증거이며 순수 정지 시각/일반 생리 실패로 일반화하지 않는다. 전체 목적은 아직 미완료다. 실제 시각·몸 감각을 쓰는 빠른 유효 완주·레이싱 라인/랩타임, 다른 코너/트랙·타이어 온도/셋업/차종 적응, 차체 기울기·관성·하중·접지, 검증된 상대풍·변속과 실제 폐루프 한계 주행을 이어가야 한다. 숫자상 오차나 한 합성 부품 성공으로 목표를 대체하지 않는다. 과거 live owner는 unknown-response timeout/EOF와 출력0으로 종료한 기록이며 남은 PID를 성공한 실주행으로 세지 않는다. 같은 코스의 동일 학습 반복만을 이어가는 대신 현재 actor를 read-only CPU owner에 정확히 연결하고 request/response 단계 시간과 원래 packet/receipt/epoch를 기록해 timeout 원인을 재현한다. 원래 body50ms·camera150ms freshness와 no-retry/poison/명시적 재시작 및 제어 권한 경계는 유지한다. 실제 시각/native 정합을 검증한 뒤 조건을 나눈 자료·실제 폐루프로 이어간다. coarse/fine 합성과 premotor operating point도 현재 반응·부분 제거 증거에 따라 비교한다. 이번 소유 프로세스는 종료했고 학생 제어·simulator 변경API·desktop 자동화0회다.

원본은 runs/fixture-adjoint-learning-20261104의 plan·normalization·validation·두 fit/evaluation·각 audit·summary·source-receipt-final-01.json이다.
