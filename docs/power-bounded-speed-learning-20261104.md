# power 되먹임 민감도 제한 속도 경계의 연속 학습 — 2026-11-04

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

TRAIN5659결정으로 고정한 대칭 근육 특징 정규화를 연결하고 power 입력 직접 민감도를 제한하고 같은 부모/Adam에서 두 조건을 각각395회 이어 학습했다. speed_policy의 진입 전 조향 RMSE 감소율(냉간/고온/개발)은 직전 부모 대비 +18.828% / +26.522% / +18.442%이고 같은 학습의 중립 대조군 대비 +16.936% / +24.345% / +16.993%다. 원래 fine 대비는 +1.163% / +2.083% / +3.219%다. 실제 영상 방향27/30·목표14/30이다. 음수는 악화다. 고정 입력24조건은 마지막2초 변동0으로 수렴했다. 영상 fixture에서 앞단 gradient가 끊기던 문제도 확인해 수정 helper를 검증했지만 이번 checkpoint의 학습에는 아직 적용하지 않았다. 두 후보 미승격·실제 AC 한계 주행/타이어 적응 미검증·목표 active.

이전 대칭 정규화 속도 경계는40ms 고정 입력에서 목표 속도와 조향이 지속 진동했다. power 특징 제거 및 출력 이득 감소 개입으로 기존 power 운동뉴런 활동이 같은 power 입력으로 되돌아가는 고리가 문제에 기여함을 확인했다. 이번에는 기존 TRAIN5659결정의 정규화·clip±5·속도 보정 최대10m/s·8계수를 유지하면서 power 쌍의 두 특징 계수만 weighted L1 ball에 투영한다.

`B_power = D * (abs(w[2])/scale[2] + abs(w[5])/(2*scale[5])) <= 2000`

D=10m/s이며 bound 단위는 model activity 한 단위당 m/s다. 다른 입력을 고정했을 때 각 power-MN 군 평균 입력의 직접 편미분을 제한한다. 실제 근육 힘·새 측정 시냅스·생리적 시간상수로 부르지 않는다. abs/tanh/clamp의 Lipschitz 성질을 사용한 경계지만 전체 회로 Jacobian·실제 차량 폐루프 안정성을 증명하지 않는다. 이전 두 합성 사례의 이득 감소 성공보다 여유를 둔 공학적 가설이며 실제 학습 후 별도 검사한다. 정규화는 다시 계산하지 않았고 actor에 지도/world pose/정답 라인은 들어가지 않는다.

두 조건 모두 같은 motion2048→4DN/native19 부모와 복구된 Adam step1185/993에서 시작했다. 중립 대조군은 새8계수0 고정이고 speed_policy만8계수를 학습한다. 두 조건 모두 motion8192/native9246을 연속 학습한다. 부모/고정 화학 회로·광학·기존 motion·정규화·signed decoder는 보존했다. native norm.05/entry.02는 이전 fixed-full 출발점 대비 누적으로 계산하며 이번 부모에서 예산을 초기화하지 않았다. motion 전체 norm2, speed weight norm.5/bias.5, 직접 속도 계수≤0을 투영한다. 이 계수 제약만으로 전체 actor의 몸 감쇠나 물리 안정성을 주장하지 않는다.

각32597 실제 결정을 자체 연속 수치 상태로 처리했고301실제 감독+30명령+10TRAIN 영상교환+54기존 합성으로395갱신했다. detach는 numerical reset이 아니며 부모 prefix 주입0이다. 두 조건 기존 Adam은1580/1324로 이어지고 speed 새 두 그룹은331/331이다. 합성/retimed fixture에서 native와 speed gradient=None, Adam 이동0을 확인했다. 추가8계수의 실제 gradient 중앙차분·초기1024결정 parent parity0·19계약 검사를 통과했고 전체 Adam 제안과1020 window 체인을 독립 감사했다. 이전 raw 속도 경계에서 발견한 모의 fixture tensor alias 실패와 원본은 이전 run에 보존했다. 투영6검사와 로더13검사를 통과했다. 첫 로더 검사 중 sensitivity fixture가 frozen scale 변경에 먼저 걸릴 수 있는 coverage gap을 발견해 원본/첫 통과 로그를 보존했다. 정규화는 유지하고 더 작은 선언 bound를 적용한 뒤 bound-specific 오류를 확인하는 fixture로 수정해 재검사했다.

각 최종 가중치로 세 실제 기록48866결정을 재생했다. 아래는 진입 전 조향 RMSE 감소율이며 음수는 악화다. 원래 fine의 낮은 오차는 몸 고정목표 local anti-damping30/30이라는 이전 검사와 함께 해석한다. 낮은 teacher replay 오차만으로 제어 품질을 판정하지 않는다. 반대로 원래보다 큰 조향 오차를 영상 반응이 있다는 이유로 성공 처리하지 않는다.

| 조건 | 원래 fine 대비 냉간/고온/개발 | 직전 부모 대비 | 실제 영상 방향/목표 .002 | 새 합성 방향 |
|---|---|---|---|---|
| neutral_control | -18.988% / -29.427% / -16.594% | +2.278% / +2.877% / +1.745% | 30/30 / 25/30 | 12/18 |
| speed_policy | +1.163% / +2.083% / +3.219% | +18.828% / +26.522% / +18.442% | 27/30 / 14/30 | 12/18 |

조향만으로 가감속 품질을 대신하지 않는다. 아래는 원래 fine 대비 전체 fit 결정의 각 축 RMSE 감소율이다. 음수는 악화다.

| 조건 / 기록 | 조향 | 가속 | 제동 |
|---|---|---|---|
| neutral_control / 0 | -20.170% | -8.684% | +14.730% |
| neutral_control / 1 | -25.460% | -11.126% | +15.222% |
| neutral_control / 2 | -24.254% | -10.890% | +14.107% |
| speed_policy / 0 | -25.879% | -6.306% | +14.367% |
| speed_policy / 1 | -32.157% | -9.882% | +15.191% |
| speed_policy / 2 | -30.048% | -9.118% | +14.207% |

중립 대조군과의 차이는 새 경계를 포함한 공동 학습 전체 효과다. 기존 motion/native도 각 조건에서 이어 학습하므로 최종8계수만의 독립 효과로 읽지 않는다. 최종 한 후보의 speed_policy_off가 동일 prefix에서 경계 자체를 끄는 개입이다.

speed_policy의 중립 대조군 대비 진입 전 조향 RMSE 감소율은 +16.936% / +24.345% / +16.993%다. 각 축 전체/진입 전 RMSE, 명령45개, 모든 이미지 arm은 summary/evaluation-audit에 남겼다. 개발 자료는 이미 소비한 같은 Monza/Z4 GT3 기록이다. 이번 새 texture seed915320241는 학습 후 평가로 소비했고915320267는 계획만 예약했다. 합성 도로/온도는 실제 새 코스나 물리 타이어 실험이 아니다.

몸과 학생 prefix/시간을 고정한30 donor 영상 조합 및 coarse_static/fine_motion/identity 채널 분리를 검사했다. donor 특징은 원래 FlyVis 이력을 포함하므로 완전 동일 상류 RGB 검사와 구분한다. visual_change_off/native_change_off는 이번 연속 학습 증분만 부모로 되돌린다. motion_off는 원래 motion residual 포트만 끈다. speed_policy_off는 이번 desired-speed 경계를 중립화한다. 모든 시각을 제거한 조건으로 읽지 않는다. 비선형 상호작용 때문에 기여를 독립 백분율로 더하지 않는다.

완전 동일 FlyVis 수치 상태와 held retina에서 원본 RGB를 재분기한 이전 feature bundle은 hash 검증 후 재사용했다. 이번 후보의 자체 prefix를 맞추고 동일 몸/시계/학생 상태에서32결정을 비교한 한 TRAIN 사례다. 실제 물리 반사실 정답·새 코너 일반화가 아니다.

| 원본 RGB 조건 | signed margin |
|---|---|
| speed_policy / fine | 0.001167703 |
| speed_policy / visual_change_off | 0.001427084 |
| speed_policy / native_change_off | 0.001168000 |
| speed_policy / speed_policy_off | 0.001681076 |
| neutral_control / fine | 0.001378507 |
| neutral_control / visual_change_off | 0.001858924 |
| neutral_control / native_change_off | 0.001378580 |
| neutral_control / speed_policy_off | 0.001378507 |

아래 채널 분리는 speed_policy 조건이다. 조향 오차가 더 낮은 neutral_control의 동일 상류 RGB 결과는 위에서 별도로 비교했다.

| 영상 채널 변경 | 방향 | 목표 .002 | 평균 signed margin |
|---|---|---|---|
| full | 27/30 | 14/30 | 0.001936738 |
| coarse_static | 19/30 | 1/30 | -0.000096977 |
| fine_motion | 27/30 | 15/30 | 0.002090584 |
| identity | 0/30 | 0/30 | 0.000000000 |

현재와 모든 view의 yaw/speed 몸 감각을±.01 바꾸는 검사는 후보의 factual 목표 trace를 고정한 조건과 자연스럽게 재계산하는 actor 조건을 분리했다. 1e-5 초과 변화 수와 방향을 구분하며 물리 폐루프 안정성 증거로 부르지 않는다.

| 몸 감각 검사 | yaw 감쇠 | speed 감쇠 | yaw 변화>1e-5 | speed 변화>1e-5 |
|---|---|---|---|---|
| feedback / fixed_reference | 45/45 | 45/45 | 45/45 | 43/45 |
| feedback / actor_policy | 45/45 | 45/45 | 45/45 | 17/45 |
| control_feedback / fixed_reference | 45/45 | 45/45 | 45/45 | 43/45 |
| control_feedback / actor_policy | 45/45 | 45/45 | 45/45 | 16/45 |

정규화된 대칭 특징6+속도1 입력과 저장 power를 NumPy로 독립 재계산했다. 아래 전용 경로 범위는 각 실제 전체 궤적이다. brain-logit과 speed-logit의 범위는 상관된 성분이므로 독립 인과 기여율로 읽지 않는다.

| 후보 / 기록 | 속도 오차 m/s | power current | 정규화 근육 logit | 속도 logit |
|---|---|---|---|---|
| neutral_control / 0 | [0.0, 0.0] | [0.05000000074505806, 0.05000000074505806] | [0.0, 0.0] | [0.0, 0.0] |
| neutral_control / 1 | [0.0, 0.0] | [0.05000000074505806, 0.05000000074505806] | [0.0, 0.0] | [0.0, 0.0] |
| neutral_control / 2 | [0.0, 0.0] | [0.05000000074505806, 0.05000000074505806] | [0.0, 0.0] | [0.0, 0.0] |
| speed_policy / 0 | [-0.3032188415527344, 1.4633431434631348] | [0.0471976213157177, 0.06575263291597366] | [-0.06810509362492712, 0.1301866318284577] | [-0.006895149905880582, -9.403591012100999e-08] |
| speed_policy / 1 | [-0.3079261779785156, 1.4633439779281616] | [0.04715532809495926, 0.06575263291597366] | [-0.060354759043515546, 0.1301866318284577] | [-0.006949715368598786, -1.116490542592973e-08] |
| speed_policy / 2 | [-0.3076744079589844, 1.4633439779281616] | [0.047157589346170425, 0.06575263291597366] | [-0.06443004317907877, 0.1301866318284577] | [-0.006946818474287131, -8.474807000962113e-09] |

같은 합성 직선-prefix 특징과 몸/native를12초간 고정한 추가 수치 스트레스 검사도 수행했다. 실제 픽셀 재렌더링이나 물리 타이어/폐루프 안정성 검증이 아니다. 3속도×2온도×2후보×speed 경계on/off의24조건을 각각0상태에서301결정 처리했고 마지막2초 action peak-to-peak 최대는 0.0다. 마지막 순간의 변화량과 근육/목표 속도 변동은 held-input-probe-01/audit에 보존했다. 이전 단계의50ms fixture 오류와 원인 정정은 이전 run에 보존했고, 이번에는 첫 실행부터 원래40ms 합성 주기와 시점별 고유 ID를 사용했다.


같은 최종 speed_policy에서 power 특징 계수2/5만 배율0/1/2/4/8로 바꾸는 음성 대조를30m/s60C 한 합성 사례에 적용했다. 원래 배율1은 마지막2초 변동0, 배율2는 작은잔여변동, 배율4/8은 조향 변동약.573/.988과 목표 속도 변동약6.0/11.0m/s를 보였다. 다른 계수·시각·몸·clock·초기상태는 고정했다. 이는 power 직접 민감도 제한의 수치 기여를 보여주지만 실제 차량 안정성이나 더 큰 이득의 운전 후보를 승인하는 근거가 아니다. 변경된 계수는 checkpoint에 저장하지 않았다. 첫 음성 대조01은 옛 held-input 경로명으로 시작 전 종료돼 source/log/manifest를 보존했고02에서 경로만 수정했다.


현재 학습의 fixture 경계는 speed 계수의 gradient를 막으려고 출력 전체를 detach했다. 이는 speed 값에 들어오는 이전 근육/시각 상태의 gradient도 끊는다. 새 fixture_speed_adjoint helper는 계수만 detach하고 입력 gradient를 보존한다. 한 TRAIN retimed 영상 쌍에서 두 방식의 수치 출력 차이0, speed/native gradient=None을 확인했다. signed margin 진단01과 전체 영상 목적함수 진단02/03을 구분했다. 02의 finite difference는 neutral target도 다시 계산했으므로 한 optimizer update의 frozen target과 달랐고,03에서 base-weight neutral target을 고정해 정확히 비교했다. 실제 등록 image loss의 검사 방향에서 step1e-4일 때 기존 analytic -0.212582, finite difference -0.134120, 수정 analytic -0.133802로 상대 차이는58.50% 대0.237%다. 이는 모든 gradient나 영상 실패 전체의 원인을 검증한 수치가 아니다. 독립 loss 재구성 차이 약1.47e-10은 F32 reduction 차이이며 첫 audit의1e-12 assertion을 보존하고1e-8 허용/실제 차이를 기록한 audit02로 완료했다.

수정 helper3검사는 입력 gradient 동일·계수 gradient 차단·출력 동일을 확인했다. 이어 학습용 FixtureAdjointSpeedExperiment와128결정×2 parity/Adam 복원 검사를 추가했다. 두 조건 모두 current speed_policy 가중치와4그룹 Adam step1580/1324/331/331을 정확히 이어받았고 출력 차이0이었다. 이 단계 optimizer update는0이며 현재 학습 결과를 수정된 gradient로 학습한 결과라고 부르지 않는다.


native 가속도/온도 reference-clamp와 원본 byte camera/body join128결정×2 runtime parity도 검사했다. 이 replay는 live 시간 보장·타이어 물리 적응이 아니다.

코너를 전부 외웠다는 사실은 입증되지 않았다. actor에 코너ID/지도/world pose를 넣지 않아도 반복 코스의 몸 속도·yaw·과거 근육/회로 상태가 다음 조향과 강하게 맞물리면 영상 기여가 작게 나올 수 있다. 동일 상태 영상 교환은 영상 사용을 보이지만 암기를 배제하지 못한다. 별도 코너/트랙과 바뀐 진입 속도·타이어/셋업에서 선행 조향·가감속과 실제 유효 완주/랩타임을 검증해야 한다.

다음은 현재 bounded speed_policy와4그룹 Adam1580/1324/331/331에서 stop-all 대 input-adjoint를 같은 출발점으로 이어 학습하는 것이다. fixture에서는 speed/native 계수와 moment는 그대로 고정하되 시각 gradient는 frozen speed 함수의 입력을 통과시킨다. 속도 경계의2000 제한·native 누적 origin·정규화·화학 그래프·고정 signed head를 유지하고, 영상 인과 반응·전체/진입 전 조향/가감속·자연 몸 감쇠·고정 입력 수렴을 함께 검사한다. 민감도 제한만으로 한계 주행을 구현했다고 판정하지 않는다. 수치 게이트를 통과한 뒤에도 기존 live owner 실패를 먼저 해결하고 read-only 실시간 시각/native 정합을 검증한 다음 실제 폐루프로 이어가야 한다. 이미 보존한 Adam과 최신 checkpoint를 사용하고 native 누적 경계를 리셋하지 않는다. 상대풍·변속·하중 이동·타이어/셋업/차종 적응·빠른 유효 랩·레이싱 라인/랩타임 및 한계 주행은 계속 미완료다. 이번 소유 학습/평가 프로세스는 종료했고 학생 제어·simulator 변경API·desktop 자동화0회다. 별도 기존 shadow PID는 남아 있지만 moving-shadow-03의 원본 result는02:04 당시 TimeoutError/owner EOFError로 종료됐고 outputs0이다. 이를 실행 중인 새 학습/실주행으로 세지 않으며 해당 프로세스는 건드리지 않았다.

원본은 runs/power-bounded-speed-learning-20261104/plan-01.json, 각 fit/evaluation, 독립 audit, summary-01.json, source-receipt-final-01.json이다.
