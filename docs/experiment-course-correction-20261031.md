# 운전 목표 중심 후속 실험 — 2026-10-31

## 고정 목표와 정보 경계

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 실제 상대풍의 출처·단위·동기와 결측도 확인한다. 경로 정답·지도·world pose는 교사/보상/평가에만 사용한다.

## 순서와 판정

1. 기존 시각·몸·비행 회로를 보존하며 방향·시간 성분이 좌우 코너 판단과 출력에 미치는 인과 기여를 확인한다. 지속 경로만 TRAIN 6개 저장 이력의 마스크된 조향/전진 교사 표적으로 학습하고, 기존 순간 경로와 그래프는 고정한다. 중간 그래디언트 절단은 상태 초기화가 아니다. 먼저 그래디언트/해시 적격성을 검증한다.
2. 기존 known DEV 02/07은 이미 반복 관찰되었으므로 블라인드 테스트가 아니다. 원본·기존 순간 경로·새 경로와 새 경로 제거를 동일 이력에서 비교하고, 양/음 코너별 부호, 조향·전진 오차, 무조향 기준과 시각 경로 제거 효과를 함께 보고한다. 단순 평균 오차 감소나 임의 fixture 반응만으로 성공이라 판정하지 않는다. 이 자료는 후보 기각·진단에 쓰며 최종 모델 선택/일반화 주장은 새 독립 수집 데이터가 필요하다.
3. 현재 이력의 코너 제동 표본은 부족하고 바람은 검증되지 않았다. 기록 원본·센서 시각·제어 권한과 카메라 계약을 확인한 뒤 별도 서킷 구간과 코너 진입·제동·탈출을 수집한다. 상대풍은 실제 출처·단위·축·시각이 확인되기 전에는 결측으로 유지한다. 재설정·수집은 CSP API와 백그라운드 인터페이스를 사용한다.
4. 유효한 제동·차체 피드백과 시각 선행 판단을 통합한 후보에 대해 폐루프 주행, 유효 완주·반복 안정성, 타이어/셋업 변화, 변속, 레이싱 라인·랩타임을 평가한다. 결과가 나오기 전에는 빠른 랩이나 생리적 타당성을 주장하지 않는다.

## 첫 학습의 한계

이는 기존 30km/h 저속 교사 기록에 대한 예비 지도학습이다. 빠른 랩 최적화·신규 제동 학습·생리 검증은 아니고, 기본 모델과 같은 TRAIN 기록을 사용하므로 독립 일반화의 근거가 아니다. 새 지속 경로는 0으로 초기화하고 기존 학습된 순간 경로만 불러온다. 실패나 불일치도 그대로 보존한다.

## 첫 실행 결과

`scripts/train_sustained_opponent_30.py`는 사전 그래디언트 적격성 검사를 통과한 뒤 TRAIN 6개 이력에서 새 지속 투영만 6회 갱신했다. 기존 본체는 해시 검증으로 불변을 확인했고, 학습 child는 종료 코드 0이었다. checkpoint SHA-256은 `e444682a32423f8a0b28b247a61dce2f67f968d442203d3e69fd700a1df0afbd`; run은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-sustained-driving-trained/`다.

`scripts/evaluate_sustained_opponent_30.py`로 known DEV 02/07에서 원본·기존 순간만(지속 경로 제거)·학습된 지속 경로·지속 경로 반전을 비교했다. 조향 RMSE는 순간만→학습 후 0.068179→0.066813, 0.093321→0.093839이고 무조향은 0.052572/0.093186이다. 양 조향 부호 일치는 각각 0/13과 7/15로 변하지 않았다. 양 조향 관측은 적고 연속적이므로 독립 표본이 아니다. 전진 RMSE도 각각 0.133740→0.133745, 0.174544→0.174537로 거의 변하지 않았다. 해당 경로 제거 대비 평균 절대 조향 변화는 0.00241/0.00299다. 한 이력 악화, 양방향 판단 미개선, 두 이력 모두 무조향보다 높은 오차로 인해 후보는 주행 투입 불가다. 평가 결과 및 배열은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-sustained-driving-known-dev/`에 있다. 이미 관찰한 개발 이력이므로 독립 검증이나 일반화 성과가 아니다.

다음 핵심은 이 두 평가 기록에 맞춰 수치를 반복 튜닝하는 것이 아니라, 시각/차체 센서의 실제 시간 정합과 좌우 코너·제동·탈출 구간의 데이터 계약을 검증하고 새 독립 기록을 확보하는 일이다. 기존 수집 자격에 소스·카메라 차이가 확인되어 현재 상태를 먼저 다시 적격화해야 한다.

## 2026-10-31 후속 설계: 방향 불균형 진단

직전 지속 경로 학습은 양 조향에서 0/13·7/15에 그쳤다. 양/음/중립 손실의 가중치를 오직 원래 TRAIN 6이력의 표적 빈도에서 계산해 지속 투영만 다시 학습하는 비교를 추가한다. 기존 시각 순간 경계·비행 그래프·몸 경계·직접 운동 경로·전진 손실은 그대로 둔다. 에피소드 상태는 연속 유지하고 매 에피소드 1회 업데이트한다. 기존 DEV 02/07은 이미 알려진 진단자료이므로 성공 증명이나 반복 선택 기준이 아니며, 동일한 경로 제거·반전과 양/음 조향 결과를 확인한다. 방향 가중이 시각 의존성이나 폐루프 주행을 보장하지 않으며, 과거 순간 경로의 균형 학습도 실패했다는 점에서 이번은 원인 분리용 예비 실험이다. 별도 코너/제동·물리 동기 자료 확보와 실제 상대풍 검증은 여전히 필수다.

실행 결과: TRAIN 6이력으로 6회 갱신, 기존 본체 불변, 학습 프로세스 종료 코드 0. checkpoint SHA-256 `368694c4861704305c5a46fadee8036ab1d3f5ab94ed4c988f3f57144eeb2a55`. known DEV에서 순간 경계만→균형 지속 경계의 조향 RMSE는 0.068179→0.067184, 0.093321→0.093859; 무조향 0.052572/0.093186보다 모두 높다. 양 조향 부호 일치는 0/13과 7/15로 변하지 않았다. `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-balanced-sustained-trained/`와 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-balanced-sustained-known-dev/`에 보존했다. 기존 개발 기록 재사용에 따른 한계가 있으며 후보는 미배포다.

최종 목표와 대조한 판단: 이 비교는 시각 경로의 제한된 진단에 해당하지만 동기화된 시각·몸 감각의 폐루프 제어, 실제 제동·변속·바람, 빠른 유효랩을 학습하는 실험은 아니다. 양방향 실패가 계속되어 동일 저속 표적의 가중치 조정 반복을 주 경로로 삼지 않는다. 다음 우선순위는 기존 수집 환경의 소스/카메라 자격 및 생명주기 상태 재확인, 독립 코너 진입·제동·탈출 데이터와 감각/명령 시각 정합 확보, 이후 시야별/몸 감각별 제거를 포함한 폐루프 검증이다. 초파리 비행 유래 회로의 생리적 근거와 공학 어댑터를 구분한다.

## 2026-10-31 새 수집 준비: 오프라인 재점검

감사 시 Assetto Corsa의 `acs`/`AssettoCorsa` 프로세스는 보이지 않았으며 CSP heartbeat 파일은 약 48,907초 오래되었다. 기존 paused 등의 값은 현재 상태의 증거가 아니다. `scripts/audit_pedal_collection_offline.py`는 이전 수집 선언의 105개 소스와 현재 소스·설치 카메라의 SHA를 비교하고 heartbeat 결측/노후도도 결과에 남긴다. 여전히 카메라 Lua, `ac_csp_retina.py`, `ac_csp_vision.py` 3개가 기존 봉인과 다르며, 설치 카메라는 별도 B2 수집 후보와 다르다. 결과와 실행 시 소스는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-pedal-offline-lineage-audit/`에 보존했다. 테스트 40개 및 Ruff 통과는 단위·호환성 검증이며 실시간 수집 자격을 뜻하지 않는다.

새로운 코너·제동 데이터 수집에는 실행 중인 시뮬레이터에서 새 heartbeat와 소유권·시작 위치를 확인하고, 실제 설치 카메라와 현재 수집 소스에 대한 자격 및 영상/차량 물리 시각 정합을 검증해야 한다. 시뮬레이터 시작·카메라 교체·차량 조작·수집은 이 감사에서 수행하지 않았다.

## 2026-10-31 사용자 재실행 후 현장 상태

사용자가 재실행한 AC engine PID 32912, 새로운 generation `593846ULL:0.0036658`, BMW Z4 GT3/Monza 및 offline CSP heartbeat를 확인했다. `control_ac_session.py --command start --execute`는 결과 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-live-session-api-start/`에서 API 요청 1회 및 목표 상태 도달을 기록했다. 게임 창 입력 및 actuator 메시지는 없다.

이후 pause API 사전검사에서는 native 속도 0.000623km/h, 가속 0, 브레이크 0으로 요구되는 정지 브레이크 유지 조건을 충족하지 못해 API 요청 없이 실패했다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-live-session-api-pause/`에 실패 이유와 native 샘플을 보존했다. 화면 자동화나 조건 우회, 차량 제어는 하지 않았다.

재실행 후 controls.ini SHA는 기존 `4e2cbe…`에서 `8265c2…`로 바뀌었다. 이전 원본 보관 파일과 현재 파일의 44개 INI 섹션 및 항목은 같고 텍스트 차이는 마지막 빈 줄 삭제뿐임을 `scripts/audit_ac_control_profile_equivalence.py`와 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-live-controls-equivalence/`에 기록했다. 이것은 실제 장치 연결·축 반응 검증이나 기존 수집 봉인의 재사용 자격이 아니다. 설치 카메라와 B2 후보의 해시 차이 및 망막·시각 소스 차이도 남아 있다. 새 수집·학습·학생 배포는 없다.

## 2026-10-31 브레이크 복구 및 시작 위치 관측

사용자의 재시작·복구 지시 후 동일 INI 항목과 새 파일 해시를 연결한 별도 설정으로 `scripts/probe_ac_axes.py`의 읽기 전용 preflight를 통과하고 기존 한정 정지 축 검사를 수행했다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-axis-current-stationary-test/`에는 66개 드라이버 요청과 native 물리 표본을 기록했다. 정지 기준 full brake 1, half 0.5, release 0, 조향 양/음 약 ±0.140625 및 brake 유지 중 가속 0.1이 각각 관측됐고 샘플 최대 속도는 약 0.0321km/h였다. 이 간단한 확인은 전체 동적 축 보정이나 독립 동기 검증을 대체하지 않는다. 장치 해제 후 CSP pause API 요청은 성공했고 UI 입력 0이다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-live-session-api-pause-after-axis/`.

새 `scripts/observe_live_pedal_start.py`는 기존 GET-only 관측기를 호출해 paused·full brake·vJoy FREE·background, 진행하는 graphics 두 표본과 고정 physics를 확인했다. lap0 현재 spline 0.8564369는 B2 시작 허용 0.49–0.54 밖이다. 시간은 호스트 수신 시각이며 정확한 물리 표본 발생 시각은 미검증이다. 원본 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-live-pedal-start-observation/`. 위치는 actor 입력이 아니다. 카메라 후보와 현재 소스/controls의 봉인이 다르며, 새로운 수집·학습·학생 배포는 아직 없다.

## 2026-10-31 현재 카메라 정지 재검사

현재 설치된 475e… 카메라의 새 소유 요청으로 전·좌·우·후 2그룹을 수집했다. 원본과 복사본 해시, 320×180 이미지의 비상수값, 그룹 간 렌더 프레임 증가를 확인했고 프로세스 종료 코드 0이었다. 결과 및 원본 메타데이터는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-current-fourview-stationary-check/`. 같은 그룹 내 시야별 `render_started_clock` 최초~최종 차는 51.49ms 및 13.41ms였으며 전체 render-call 종료까지의 범위는 52.06ms 및 13.70ms였다. 이 숫자는 정지 중 두 그룹의 렌더 호출 시각이며 실제 카메라 노출 시각이나 주행 중 시각/물리 정합의 증거가 아니다. clock은 `csp_os_preciseClock_unaligned`, 동시 노출·독립 광학 보정·공유 텍스처 소비 검증은 false다. 학습 가능 데이터가 아니다.

현재 B2 수집기와 native 이동 계획은 과거 특정 엔진 PID/generation·시작 위치·카메라·원본 해시에 묶여 있다. 해당 식별자만 바꿔 검증을 우회하지 않는다. 새 세션에서 수집에 필요한 위치/역할, 카메라와 원본 소스의 재적격화, 실제 물리와 영상 시각의 대응이 남아 있다.

## 2026-10-31 정지 시각의 producer/host clock 교환

`scripts/audit_stationary_fourview_clock.py`는 사용자 재확인 목표에 따라 현재 paused·브레이크 유지·오프라인·background 세션에서 CSP camera clock ping을 수집하고, 독립된 정지 네 시야 2그룹 전후 동일 generation 교환 구간을 교차했다. 호스트 `perf_counter` 기준 offset 교집합 폭은 촬영 전 3.4391ms, 후 3.2899ms였고 각 원본 시야의 render-start 호스트 하한/상한을 결과에 기록했다. camera child 종료 코드 0, 원본 프레임·요청·시각·교환과 실행 소스는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-fourview-current-clock-audit/`. 첫 시도는 앞선 카메라의 정상 terminal 상태를 idle만 허용해 촬영 전 거절했고, terminal 소유 ID/완료 파일을 확인한 뒤 다시 실행했다. Ruff 통과.

이는 짧은 관측 구간에서 단위 시계 속도 가정과 ping 인과구간이 양립한다는 증거다. render 호출은 물리적 노출 시각이 아니고 native 물리 샘플은 수신 시각만 있으므로 시야 간 동시 노출·영상과 물리의 동시점·차량 운동 상태의 정합은 증명하지 않는다. 학습 데이터·폐루프 주행·실제 바람·제동 수집 자격은 얻지 않았다.

## 2026-10-31 정지 상태 live camera/native 수신 기록

CSP lifecycle API로 정지·full-brake 세션을 resume하고 `scripts/audit_live_camera_physics_receipts.py`가 두 네 시야 그룹과 native 물리 1,363개 수신을 같은 host `time.perf_counter`로 기록했다. 촬영 전후 clock ping 교집합 폭은 5.5766ms, 8개 시야 render-start 호스트 범위를 각각 감싼 앞뒤 native 수신 간격은 10.618–11.481ms였다. 관측 중 native 최대 속도 0.000486km/h, 최소 브레이크 1, 최대 가속 0. camera child 종료0, 이후 CSP pause API 목표 상태 관측 및 UI 입력 0. 결과와 원본은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-live-fourview-physics-receipts/`; resume 및 pause receipt는 각각 `20261031-live-clock-receipt-resume/`, `20261031-live-clock-receipt-pause/`다. Ruff 통과.

이 구간은 **호스트 수신 시각**이 카메라 렌더 호출 시각의 추정 범위를 앞뒤로 감싼다는 진단일 뿐이다. native 원본에는 물리 표본 발생 시각이 없고 repeated-copy는 원자적 seqlock을 보장하지 않으며, 카메라 호출은 물리적 노출이 아니다. 따라서 실질적 동시성·주행 시 정합·레이블 자격은 여전히 입증되지 않았고 이 정지 기록으로 학습하지 않는다.

## 2026-10-31 움직이는 차량의 네 시야·몸 수신 진단

첫 시도 `20261031-moving-sensor-producer-clock-probe/`는 exterior-front 구성이 MMF v3를 요구하는 반면 producer-time retina는 해당 조합을 허용하지 않아 촬영/차량 조작 전 실패했다. 두 번째 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261031-moving-sensor-mmf-clock-probe/`는 기존 MMF·source-timing 및 same-copy native authority 진단 경로를 사용했다. 3초 엔진 AI probe의 실제 최고 속도는 16.587km/h, 변위는 11.418m였다. 네 시야가 유효한 77결정, camera 287그룹, 원본 시간 ledger 4,507건(overflow 0), native 동일 복사 AI 표식 조건 77/77을 보존했다. 이는 교사 조작과 영상의 시간 동기나 독점 제어 검증이 아니다. UI/vJoy 요청은 0이고 종료 후 실제 session paused·AI off·full brake·camera off다.

`scripts/audit_ac_engine_ai_probe.py`가 별도 `20261031-moving-sensor-mmf-clock-audit/`에서 원본 2,021파일·MMF 원시 슬롯·프레임 저장·aid off·77결정 actor 재생을 검증했다. actor 재생 최대 차는 0, 최대 결정 간격은 43.946ms, 80ms 초과 0이다. producer-time retina 재생은 사용하지 않았으며 물리 노출 동기·teacher action codec 및 학습 자격은 false다. `scripts/audit_moving_camera_body_receipt_age.py`의 `20261031-moving-sensor-receipt-age-audit/`는 각 시야 render-start의 host 하한과 native **수신** 시각의 차를 저장했다. 시야별 중앙값은 28.15–29.30ms, 95백분위는 58.14–59.35ms, 최대는 73.92–75.04ms다. 이는 render 호출에서 body 수신까지의 가능한 시간차 상한이며 실제 영상 노출과 물리 표본이 그만큼 어긋났다는 측정값이 아니다. 같은 3초 저속 구간이며 연속 표본은 독립적이지 않다. 새 라벨·학습·학생 배포는 없다.
