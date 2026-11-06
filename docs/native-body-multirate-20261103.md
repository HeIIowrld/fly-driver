# 몸 속도 갱신·CPU worker·촬영 순서 prefix의 실제 검증 — 2026-11-03

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·변속·타이어/셋업/차종·검증된 상대풍·교사/평가 전용 지도/pose 경계를 유지한다.

## 구현한 경로와 실제 범위

`NativeFlightBodyTicker`는 새 영상이 없어도 같은 영상·원래 촬영 당시 몸 이력을 유지하고 최신 native 몸21/추가19로 회로를 갱신한다. 회로 상태는 결정 사이에 유지하며 초기화로 지연을 숨기지 않는다. packet/원래 첫 receipt·pending publication·episode·기존50ms 몸/150ms 화면 기준을 검사한다. 원래 영상576/공간320·여섯 날개근 readout·측정 비행 topology와 학습 가중치는 고정했고 새 optimizer0이다.

`NativeFlightProcessRuntime`은 회로를 순서가 있는 독립 worker에 맡긴다. worker는 simulator/vJoy 핸들 없이 CPU 모델만 소유한다. 응답 timeout/순서 불일치 뒤 retry·reset을 금지한다. 이번 실패 worker들은 정상 종료했고 강제 종료는 사용하지 않았다. 일반 startup/timeout 경계 전체가 폭넓게 검증됐다는 뜻은 아니다.

별도 worker를 GPU용 Torch 환경에서 실행한 첫 시험도 느렸다. 부모의 GPU/FlyVis는 그대로 두고 worker만 기존 Python312 실제 실행 파일+CPU 전용 site-packages로 시작하도록 분리했다. 자식의 PID·실제 Python 경로·Torch2.9.1+cpu를 ready receipt로 확인했다. 고정 CSR 준비 이후 연산은 GPU용 환경 약12ms에서 CPU 전용 환경 약2ms였다. 이는 관측된 한 고정 연산/환경 비교이며 모든 지연 원인을 설명하지 않는다.

`PerViewFlyvisStream.prepare_held_prefix`는 이전 시야를 계속 유지한다는 가정의 미래 상태16개를 미리 계산하되 **원래 다음 촬영 사건까지의 경과만 commit**한다. 다음 화면을 미리 넣거나 타임스탬프를 바꾸지 않는다. `source-prefix-parity-01`: 원본60그룹/기존 actor 대응55개에서 scene 최대차5.96e-8, 정적 공간 잔여 차이0. 현재 화면 처리 p95 약25.39ms와 별도 prefix p95 약28.66ms를 모두 기록했다. prefix 비용을 제외한 수치로 전체 실시간 성능을 주장하지 않는다.

## 다섯 실제 정지 실험 — 전부 전체 실패

원본은 `runs/native-body-multirate-20261103/`이다. 카메라/원본 native/실행별 entrypoint·manifest·실패·독립 감사/종료를 보존한다. 출력을 실제 차량에 보내지 않았다.

| 실행 | 정상 출력 | 이후 실패/제한 |
|---|---:|---|
| stationary-shadow-01: 같은 프로세스의 몸 ticker | 1 | 회로 계산57.47ms·입력 포함60.74ms로 거절 |
| process-shadow-01: GPU용 Torch의 독립 worker | 1 | 계산51.61ms·입력 포함55.12ms로 거절 |
| process-shadow-02: CPU 전용 worker | 6 | 다음 영상 대기 중 기존 화면150ms 만료 |
| process-prefix-shadow-01: CPU worker+원래 촬영 순서 prefix | 130 | 다음 출력의 몸/화면 deadline 초과; 종료 중 native queue full도 기록 |
| process-prefix-shadow-02: 만료 전25ms 여유·native 먼저 종료 | 2 | 이전 몸 입력53.90ms로 거절; 현재 몸은3.04ms로 신선했음 |

가장 긴 구간의 **130출력은2.475초** 이어졌고, 그중89개는 같은 카메라를 유지한 추가 몸 갱신이다. 정상 출력의 최대 몸 나이28.24ms·최고 화면148.88ms, worker 계산 p95 21.15ms·최대21.50ms였다. 이후131번째 worker 계산은 출력 기준에서 거절됐고, run 전체는 실패다. 한 구간의 성공을 지속 실행 해결로 바꾸어 부르지 않는다.

두 번째 prefix 시험은 만료에 가까운 입력에서 새 추론을 시작하지 않도록 여유를10→25ms로 늘렸다. 실제 출력의50/150ms 제한은 그대로다. 결과는 오히려 이전 몸 age 거절이었다. 이 여유 조정은 해결책이 아님을 보존한다. native 수집을 오래 걸리는 pause subprocess 전에 종료하도록 순서를 바꾼 뒤, 마지막 시험은 queue 오류 없이 정리됐다. 첫 prefix 시험의 cleanup_verified=false와 queue failure는 그대로 남긴다. 해당 sampler의 stop은 process join 이후 원래 오류를 전달한 것이며 정상 종료 성공으로 재분류하지 않는다.

## 감사·종료와 미완료 목표

`audit-*` 다섯 개는 원본 RGB/source render-body bracket·25ms host bound·현재/이전 native 첫 receipt·publication 순서·몸50ms/화면150ms·worker/카메라 종료를 재계산했다. **모두 whole_shadow_passed=false**다. cleanup 실패도 별도로 남겼다. `final-tests-01.txt`는83검사 통과이며 새 몸 갱신·중복 packet/첫 receipt·source loss·future packet·prefix 미확정 시간·기존 회로/카메라 계약을 포함한다. 전체 일반적인 worker 장애/실시간 성능 보증은 아니다.

`source-receipt-final-01.json`/snapshot에는 현재 소스와 실패 당시 ROCm worker·10ms 여유 ticker를 연결했다. `final-state-01.json`에 모든 실행/감사의 hash·관측한 소유 프로세스/worker 없음·fresh paused·AI/cameraoff·gas0/brake1 상태를 남겼다. 실제 차량/GUI/학습 명령0, actor_qualified=false, 목표 active다. 정지 타이어 온도 OOD 표시는 유지했고 이를 적응/한계 주행 성능으로 세지 않는다.

## 즉시 다음 구현

회로 계산만 독립시킨 현재 구성에서는 **몸 패킷 선택과 ticker 스케줄링이 GPU 부모의 Python 스레드에 남아 있다.** 이 루프까지 별도 CPU owner로 옮겨 수집/첫 receipt와 연속 회로 상태를 함께 소유하게 해야 한다. 영상 publication·기존 촬영 당시 몸 이력·현재 몸 source를 명시하고, 미래/누락/큐 포화/응답 불명·종료를 계속 거부한다. source/view/held-body 제한을 늘리거나 지연 구간을 삭제하지 않는다.

그 뒤 실제 정지 지속 shadow→움직이는 시각/몸 정합과 같은 몸 영상 기여·새 코너→학생 폐루프 조향/가감속·하중/타이어/셋업·차종/변속/상대풍·빠른 유효 랩으로 진행한다. native-fit의 조향 악화와 실제110℃ 감독 부족도 남아 있다. 몸 속도 갱신 부품 구현·짧은 정상 출력과 한계 주행 목표 달성을 구별한다.


후속 2026-11-04: 몸 선택/갱신과 과거 감각까지 단일 CPU 소유자로 옮겨3초 정지 시험266출력·종료 독립 감사까지 통과했다.5초 반복426출력은 종료 실패이며 종료 수정의 실전 재검증/카메라 metadata 충돌 처리가 남았다. 이 문서의 다음 구현은 당시 계획이다. [현재 근거·실패·다음 작업](isolated-body-owner-20261104.md).
