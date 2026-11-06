# 원래 촬영 순서의 시각 처리와 실제 정지 shadow — 2026-11-03

**초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.** 비행 기준·조향/가감속·하중/타이어·변속·차종/셋업·검증된 상대풍·빠른 라인/랩 및 teacher/평가 전용 지도/pose를 유지한다.

## 학습과 과거 실시간 입력 사이의 차이

학습 전처리 `prepare_synchronized_sac_student.py`는 원본 네 시점의 render lower를 순서대로 정렬하고, 각 시점 직전까지 기존 held 시야로 author FlyVis를 적분한 뒤 해당 시점의 화소만 갱신한다. 과거 `record_native_process_unified_shadow_v2.py`는 호스트 처리 시각의 간격으로 적분하고 네 시야를 한 번에 갱신했다. 따라서 과거 shadow 경로를 이번 학습 모델에 그대로 연결하면 시간 처리 방식이 다르다. 이 차이가 이전 낮은 시각 기여의 유일 원인이라고 확정하지 않는다.

새 `PerViewFlyvisStream`은 학습과 같은 촬영 사건 순서·author weights·5ms 이하 적분·held 시야·steady .5 prior를 보존한다. 현재 시점보다 나중인 화면을 미리 회로에 넣지 않는다. 같은 원본 화소로 scene576과 소실된 정적 공간320을 만들어 몸/비행 회로 경계에 전달한다. 새 CNN이나 별도 운전 모델로 바꾸지 않았으며 이 시각 모델의 author-trained 동역학과 차량에 학습한 어댑터를 구분한다.

`source-parity-01`: 원본64그룹 중 처음 부적격4그룹을 회로에 넣지 않고60그룹을 실행했다. 기존 actor와 대응하는55그룹 scene 최대차7.45e-8, 정적 잔여256은 차이0이다. 원래 source clocks/episode 상태를 유지한 좁은 원본 재생이며 전체 주행/실시간 판정이 아니다. 처리 p95 약52.9ms, 최대202.3ms도 보존했다.

## 실제 정지 shadow와 보존한 두 실패

`shadow_native_flight_per_view.py`는 실제 background CSP 카메라·별도 native 수집 프로세스·고정 author GPU 시각 처리·기존 CPU 비행 회로/native19 감각과 여섯 근육 readout을 연결한다. 영상 출처 증거를 online candidate에도 보존하도록 collector에 원래 source render/body metadata 복사만 추가했다. 이 진단 metadata는 actor 신경 입력에 들어가지 않는다. 시작 부적격 화소는 제외하고, 정상 실행 뒤 정합을 잃으면 중단한다. source25ms·몸50ms·화면150ms와 기존 held body50ms 기준을 유지했다.

- `stationary-shadow-01`: 첫 회로 추론에서 장치 계약 거절. 설치 FlyVis `__init__.py`가 `torch.set_default_device(device)`를 호출하며 GPU 기본 장치가 CPU 입력 생성에도 영향을 줬다. CPU runtime 입력/goal/mask/cache의 device를 명시했다. GPU 기본 장치가 있어도 CPU 경계를 보존하는 meta-device 회귀 검사 추가. 원본13그룹/몸 기록·미추론 결과를 보존했다.
- `stationary-shadow-02`: 실제 source-qualified 영상·현재 native 몸 감각으로1개 shadow 출력을 생성했다. 계산16.875ms, 출력 몸 나이21.162ms, 가장 오래된 화면96.277ms다. 다음 결정은 `Held body receipt stale at decision`으로 중단했다. 최신 현재 몸은2.165ms 전 자료였지만, 이전 몸 입력은76.117ms 됐다. 시각 처리 후 최신 몸을 고르는 것만으로 이전 회로 입력의 age가 해결되지는 않는다. live 전체 실험은 **실패**다.

교사 조작·지도·world pose는 추론 입력에 없고 simulator 조향/가속/제동/vJoy 명령0, GUI 입력0, optimizer0이다. CSP의 owned camera resume와 종료 후 pause만 사용했다. 카메라/수집 프로세스 종료·원본 archive·실패·일시정지를 보존했다. 실제 정지 타이어는 약27℃로 새 학습 지원 범위 밖이며 OOD 표시를 유지했다. 이 출력을 냉간 적응이나 한계 주행 성과로 세지 않는다.

## 독립 감사와 현재 상태

`live-audit-01/02`는 원본 RGB hash·per-view source timestamp/render bracket·25ms host bound·원래 native bytes/첫 receipt·몸/화면 출력 나이·정상 종료와 pause를 독립 확인했다. 둘 다 `whole_shadow_passed=false`다. 두 번째 감사는 현재 몸은 신선하지만 이전 몸이50ms 이상 유지됐음을 원본에서 재계산했다. 실패를 통과로 바꾼 것이 아니다. 정확한 GPU 노출/physics atomicity는 여전히 미검증이다.

`final-tests-01.txt`:74검사 통과. `source-receipt-final-01.json`과 snapshot에 최종 소스/첫 실패 당시 CPU 경계·실행별 manifest를 연결했다. 두 GPU shadow·원본 parity·두 audit 핸들 모두 종료했다. `final-state-01.json`에는 신선한 paused·AI/cameraoff·gas0/brake1 heartbeat와 owned 프로세스 없음·0차량/GUI/optimizer·전체 목표 active를 기록했다.

## 다음 필수 구현

현재 단일 루프는 시각 처리 약20–53ms와 카메라 대기 동안 몸 입력을 유지한다. 기존50ms 기준을 늘리거나 사건을 삭제/시간 재작성/회로 reset으로 숨기지 않는다. **시각 갱신과 독립된 몸 감각 속도의 연속 추론 루프**가 필요하다. 기존 body ticker/독립 회로 worker 패턴을 native19/공간320·같은 첫 receipt 경계에 확장하고, 새 영상이 없을 때 원래 영상과 촬영 당시 몸 이력을 보존하면서 최신 몸으로 회로를 갱신한다. publication·stale·worker 실패·종료도 명시적으로 검증한다.

그 뒤 실제 정지 shadow에서 전체 source/body age와 지속 실행을 통과하고 움직이는 시각 정합/시각 선행 기여·새 코너를 검증한다. native-fit의 조향 악화·고온110℃ 감독 부족도 남아 있다. 실제 학생 폐루프 조향/가감속·하중/타이어/셋업/차종/변속/상대풍·빠른 유효 랩이 검증될 때까지 목표를 완료로 표시하지 않는다.


후속 몸 속도 loop/독립 CPU worker·원래 촬영 순서 prefix를 실제 시험했다. 최대130출력2.475초까지 이어졌으나5실험 전체는 실패다. 같은 프로세스의 body 선택/스케줄링 분리가 다음 과제다. [원본·실패·다음 구현](native-body-multirate-20261103.md).
