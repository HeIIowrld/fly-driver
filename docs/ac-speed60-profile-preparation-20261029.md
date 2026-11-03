# 60km/h 교사 시험 경로 준비

2026-10-29, Asia/Seoul. 고정30km/h를 레이싱 기준으로 사용하지 않는다는 사용자 정정을 따른다.
**아래는 배포 전 준비 당시 기록이다. 후속 t1478 배포/61.14km/h partial 시험은 [최종 기록](flight-steering-terminal-and-speed60-pilot-20261029.md)을 따른다.**

후속으로 `racing_teacher_contract.py`를 추가했다. 교사 지시60과 사전 선언한 actor 상한80을 구분하고,
actor task는 새 racing goal의 `[0, .8]`로 명시한다. 이는 현재 구동 중인 학생 입력이 아니라 새 자료의
prospective 계약이다. source 역할·시간 정렬·native codec·실제 속도 검증이 끝났다는 뜻은 아니다.
calibration/개발/평가 역할은 fitting 불가이며, 기존30/감속/강제 제동 요청은 이 계약으로 재해석하지 않는다.
목표/새 계약의 검사20개와 Ruff를 통과했다. 실제 요청 전에
`configs/flight-speed60-calibration-pilot-20261029.json`에 calibration-only/fit 금지와 두 단계를 고정했다.
이 작업은 실제 운동 회로 조향 학습 t1473이 같은 PID40084/session31336에서 실행되는 동안 CPU로 진행했다.
GPU 교사 수집을 동시에 시작하거나 그 학습을 재시작하지 않았다.

- 새 명령은 `expert-motion-speed60`(3초)과 `expert-collect30-speed60`(30초)다.
  recorder의 명시적 `--speed-hint-kmh 60`으로만 선택한다. 60→15 감속/제동 개입과 미검증 flight shadow는 거부한다.
- AI 속도 지시는60km/h이며 독립 관측 상한80km/h, 이동 제한은 각각80/720m다.
  이 숫자는 새 시험의 공학적 envelope이며 차량의 물리적 한계·최적 속도·검증된 운전 성능이 아니다.
- 종료 정지 대기는 새60 프로파일만8초로 정했다. 호스트 deadline과 초기 카메라 잔여 lease도 이에 맞췄다.
  기존30 프로파일의40km/h/40·360m/4초 종료 기준은 그대로다.
- 30초 수집은 실제 최고속도가50km/h에 미달하면60 단계 성공으로 인정하지 않는다.
  단순 최고속도 기준 역시 유효 랩·코너 제동·레이싱 최적화의 검증은 아니다.
- 기존15/30kmh 자연 AI 라벨 계약과 기존0..30kmh 모델 입력은60 자료를 자동 수용하지 않는다.
  새 목표 명세/정답 버전과 실제 센서/시간 검증 없이 과거 체크포인트 의미를 바꾸지 않는다.

관련224개 검사는 Lua의 실제 상태 기계에서60 설정/별도 제한/종료를 확인하고,
잘못된30 요청,80km/h·720m 초과 receipt, 기존30 경로에60 receipt가 들어가는 경우를 거부하는지 검사했다.
검사 통과는 실제60km/h 주행 증거가 아니다. Ruff도 통과했다.

준비 당시 작업 폴더 Lua와 배포된 Lua는 **의도적으로 서로 달랐다**. t1478 배포 후 일치한다. 실행 중인 조향 학습을 먼저 마친 뒤,
paused/AI-off/camera-off/probe-off와 소유권을 새로 읽고 백업·원자 배치·실제 generation/capability 확인을 진행한다.
그 전에는 소스 일치 검사가 있는 lifecycle/recorder를 강제로 우회하지 않는다. UI 조작 fallback도 없다.
새 수집의 TRAIN/개발/평가 경로 역할은 수집 전에 정한다. 기존 개발 구간을 TRAIN으로 바꾸지 않는다.

조향 학습의 최종 결과와 함께 단계적으로 연결한다. 초파리 시각/비행 회로, 기울기·하중·타이어/셋업 적응,
조향·가감속과 빠른 유효 레이싱 라인/랩타임이 최종 범위다. 아직 학생 제어와 고속 주행은 미검증이다.

소스/검사: `integrations/ac_csp/FlyBrainCamera/FlyBrainCamera.lua`,
`scripts/control_ac_session.py`, `scripts/record_ac_engine_ai_probe.py`, 관련 `tests/test_ac_*`.
보존: `runs/ac-speed60-profile-preparation-20261029-t1475`.
