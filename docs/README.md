# 프로젝트 문서 안내

- [공동 연구 작업 방식과 첫 작업 카드](COLLABORATION.md) (2026-11-03, 제안)

- [최종 목표 로드맵 — 시각 선행 판단·범용 주행·빠른 랩](ROADMAP.md) (2026-11-03, 계획)

## 현재 기준 — 이 순서로 읽기

1. [프로젝트 대전제](PROJECT_CHARTER.md):사용자가 확정한 목적과 바꾸지 말아야 할 경계.
2. [현재 문맥/인계](CONTEXT.md):현재 체크포인트·자료·실행 상태·바로 다음 작업.
3. [연구 계획](research-plan.md):필수 부품 검증과 AC완주/최적화 순서.
4. [설계](assetto-fly-design.md):가상 몸·비행 감각운동 구조와 정보 계약.
5. [구현 현황](implementation-status.md):현재 구현과 미구현의 구분.

## 현재 판단을 뒷받침하는 기록

- [진행 t1769–t1794: same-copy 교사 원본·경계 감사·heartbeat 실제 정차 handoff](flight-teacher-authority-and-heartbeat-runtime-20261030.md)

- [최신 t1733–t1762: MN anchor v2의 CPU·GPU/실차 연결과 source timing](flight-embodiment-v2-and-source-timing-20261030.md)

- [최신: native몸시계·신경운동·작은페달 실제응답](flight-native-neural-actuator-20261030.md)
- [최신:8간격자격·fold전용보정·실제저자시각망1,116회비교학습과미통과](flight-visual-temporal-fold-learning-20261030.md)

- [최신 t1676–t1684: 페달 cutoff·명시적 codec 실제 정차 검증·원본 독립 감사·TRAIN 자료 부족](flight-actuator-codec-and-data-audit-20261030.md)

- [최신: 빠른 haltere–b1 부품 후보·운동뉴런 기준·수치 세분화·전체 원래 이력](flight-fast-haltere-prototype-20261029.md)

- [최신: 몸·시각 명령 분해·회로 대역폭·직접 감각운동 경로와 생리 근거](flight-yaw-bandwidth-and-fast-path-20261029.md)

- [최신: 12회/3블록 전체 비교·독립 감사·안정화 기준 미통과·원본 응답](../runs/flight-physical-yaw-study-report-20261029-t1245/summary.md)
- [수명주기 상태 파일 공백 보완·실패 보존·비교 재개](ac-lifecycle-heartbeat-gap-20261029.md)

- [앞선 GPU 작업 종료 판정 수정·자원 회수·첫 비교 블록 감사·다음 블록 pause 실패](gpu-worker-teardown-and-realtime-20261029.md)

- [앞선 API 엔진 프로세스 재시작·새 게시자 적용·full 감사·yaw0 시간 초과](ac-api-engine-restart-and-publisher-deployment-20261029.md)

- [앞선 진단 게시자 파일 교체 충돌·준비한 수정·미배포와 재개](flight-diagnostic-publisher-collision-20261029.md)

- [앞선 GPU 그래프 시각 계산·실제 full 주행 감사·yaw0 진단 중단](flight-graphed-vision-20261029.md)

- [앞선 고정 시간 조향·계산 경로 대조·실패와 재개](flight-timed-control-handoff-20261029.md)

- [앞선 실제 full/yaw0 비교·입력 시간 불일치·운영 실패와 지연](flight-physical-yaw-comparison-20261029.md)
- [입력 지속시간 일치 조건의 추가 시점](flight-physical-yaw-timing-amendment-20261029.md)
- [최신: 실제 저속 이동·회전 감각 피드백과 전체 원본 재생](flight-moving-body-feedback-20261029.md)
- [최신: 비행 회로 출력→실제 AC 정차 좌우 조향·원본 재생](flight-stationary-steering-20261029.md)
- [최신: 실제 영상→시각/비행 회로의 연속 추론·3초/30초 계측](flight-live-stream-integration-20261029.md)
- [회로 조향 범위·특징 적합 진단·균형 학습과 전이 한계](flight-steering-capacity-and-balanced-learning-20261029.md)
- [주행선 OFF 새 주행·고정 평가·좌우/중립 추종 실패](flight-aid-free-recording-and-tracking-20261029.md)
- [학습 안정성 개선·고정 개발 평가 실패·소비자 입력 검사](flight-visual-optimizer-diagnosis-20261029.md)
- [최신: v6 시각 학습 실패·보조 주행선 입력 감사](flight-visual-learning-and-aid-audit-20261029.md)
- [v5 조향/추진 명령 분리·몸 보호·시각 경로 문턱](flight-separated-commands-20261029.md)
- [몸 yaw 경로 분리·최소 경계 보정·모형 내16조건](flight-yaw-boundary-calibration-20261029.md)
- [다중 속도 학습·시각 기여 문제·회전 억제 부호 실패](flight-speed-learning-feedback-20261029.md)
- [기존 기록 재사용 검사와 새30km/h 백그라운드 자료](flight-diverse-data-20261029.md)
- [비행 후보의 실제 기록 초기 학습과 평균 편향](flight-initial-training-20261029.md)
- [실행 가능한 비행 후보 제어기](flight-circuit-prototype-20261029.md)
- [여러 시점의 시각 운동 기능 어댑터](visual-self-motion-adapter-20261029.md)
- [실제 네 시점 영상의 연속 시각 회로 재생](ac-flyvis-camera-replay-20261029.md)
- [가상 몸 관측 v1·기울기/회전 대조](ac-virtual-body-v1-20261029.md)
- [비행 실측 경로 후보·독립 감사](fly-flight-route-candidate-20261029.md)
- [운동 시각 부품 반응과 불일치](flyvis-component-response-20261029.md)
- [비행 기반 상세 계획·근거](fly-flight-mechanism-plan-20261029.md)
- [현재 뇌 출력과 차량 조작의 연결](current-fly-brain-control-map-20261029.md)
- [실제 회복 자료 학습과 운전 실패](ac-recovery-training-validation-20261029.md)
- [이동 중 교사 수집의 시간 간격](ac-moving-collection-timing-20261029.md)
- [공유 영상과 기존 GPU학습 자료](ac-shared-engine-ai-learning-20261029.md)

## 역사 기록

- [실시간 연결 전 README 보존본](readme-before-stream-20261029.md)
이전 문서의 ‘최신’과 ‘다음 작업’은 당시의 문맥이다. 현재 우선순위를 대신하지 않는다.

- [이전 README 전체 기록](readme-history-20261029.md)
- [비행 기준 재정리 전 구현 현황](implementation-status-before-flight-20261029.md)
- [비행 기준 재정리 전 연구 계획](research-plan-before-flight-20261029.md)
- [비행 기준 재정리 전 설계](assetto-fly-design-before-flight-20261029.md)

그 밖의 날짜별 문서·`runs/`는 실험 근거로 보존한다. 문서에 적힌 성공은 원본 실행의
모델·데이터·정보 조건·거리·반복 범위를 넘어 확대 해석하지 않는다.
