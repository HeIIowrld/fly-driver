# 시각 회로의 GPU 호출 비용 개선

기록일: 2026-10-29, Asia/Seoul. 목표는 [프로젝트 대전제](PROJECT_CHARTER.md)를 그대로 따른다.
이 작업은 실제 주행의 지연 제한을 지키기 위한 실행 개선이다. 시각 경로 추종·회로의 생리학적 타당성·자율 완주를 증명하는 실험은 아니다.

## 바꾼 것과 검증 조건

고정 평가 상태의 저자 시각망에서 원래 `_next_state`를 GPU 그래프로 준비한다.
1/2/4/8단계의 별도 그래프를 조합해 각 원본 관측 간의 5ms 적분 단계를 모두 실행한다.
남은 소수 시간은 원래 함수로 계산한다. 가중치·정밀도·시냅스 합산·dt·입력 픽셀·원래 시간과
회로 상태를 유지하며, 반환 상태를 복사해 다음 그래프 호출이 이전 상태를 덮어쓰지 않게 한다.

저자 함수가 매 단계 만드는 `torch.tensor(.005)`의 CPU→GPU 전송은 캡처할 수 없어,
해당 리터럴만 동일한 float32 GPU 상수로 미리 준비한다. 다른 dt나 dtype 요청은 거부한다.
이는 계산식의 변경이 아닌 상수 할당 경로의 변경이다. 준비 단계는 실제 에피소드와 별도다.
GPU 그래프의 고정 주소 입력·출력 제약은 [PyTorch 공식 문서](https://docs.pytorch.org/docs/main/notes/cuda.html)를 참고했다.
현재 장치의 PyTorch 인터페이스는 `cuda`이며 실제 backend는 ROCm/HIP, GPU는 RX 9070 XT다.

구현: `src/flybrain_drive/flyvis_graphed_tail.py`.
출력은 차량 축이 아닌 원래 시각 상태다. 제어와 생리학적 주장을 추가하지 않는다.

## 원본 전체 대조와 실패 기록

| 기록 | 결과 |
|---|---|
| [t1193](../runs/flyvis-graphed-tail-stage-20261029-t1193/result.json) | 원래 스칼라 텐서 생성이 캡처 중 허용되지 않아 실패. 차량 입력 0 |
| [t1194](../runs/flyvis-graphed-tail-stage-20261029-t1194/result.json) | 준비 중 계산된 RefTensor 입력 gather 캐시가 재사용되어 큰 수치 불일치. 실패를 보존하고 fresh 참조 생성으로 수정 |
| [t1195](../runs/flyvis-graphed-tail-stage-20261029-t1195/result.json) | 단일 단계 그래프의 전체 대조는 통과했지만 연산 합계가 원래 경로보다 느렸음. 속도 개선으로 채택하지 않음 |
| [t1196](../runs/flyvis-graphed-batch-stage-20261029-t1196/result.json) | 묶음 그래프로 원래 225관측·448적분 묶음·모든 카메라/뉴런 대조 통과. 최대 오차 1.43e-6, 원본 5ms 단계 1,746회와 소수 단계 224회 유지 |
| [t1197](../runs/flight-graphed-batch-coupled-verification-20261029-t1197/result.json) | 별도 fresh process에서 full/yaw0 각각 225관측의 몸 값·유효성·출력 대조 통과. 의도 최대 오차 9.69e-8 / 2.16e-7 |

t1196의 같은 기록 대조에서 시각 연산 합계는 원래 5.336초, 새 경로 4.059초로 약24% 감소했다.
두 경로를 번갈아 실행한 오프라인 계측이다. 실제 게임의 최악 지연이나 모든 부하에서의 성능을 보장하지 않는다.
최대 할당 메모리도 두 상태 궤적을 보유한 대조 프로세스의 수치이며 실제 운전 프로세스의 메모리로 해석하지 않는다.

합성 프로토콜 검사는 원본 단계 수, 상수 경계, 반환 상태 불변성과 중간 에피소드 교체 거부를 다룬다.
측정 뉴런의 검증은 위 전체 원본 재생이 담당한다. 관련 16검사와 Ruff는 통과했다.

## 실제 적용 경계

새 계획은 `configs/flight-physical-yaw-graphed-tail-v2-20261029.json`이다.
원래 12회/3블록 비교, 속도6km/h, 257관측, 입력150ms/명령80ms 제한과 교란280ms를 유지한다.
기존 계획의 성공 시험을 이 계획의 실적으로 가져오지 않는다.
실행과 독립 감사 모두 같은 backend를 설치하며 t1196/t1197과 현재 실행 모듈 해시의 일치를 요구한다.

실제 실행 상태는 [현재 문맥](CONTEXT.md)과 해당 실행 원본을 따른다.
전체 목표·운전 후보·생리학적 동등성과 자율 완주 검증은 미완료다.

## 실제 적용과 후속 확인

- [t1198](../runs/flight-graphed-tail-preflight-20261029-t1198/result.json) 읽기 전용 점검은 통과했다.
  [t1199](../runs/flight-graphed-tail-block0-20261029-t1199/result.json)는 정차 2관측을 적용한 뒤
  세 번째 추론74.331ms 중 명령80ms 유효기간 만료로 제동·일시정지했다.
- [t1200](../runs/flight-graphed-tail-failed-prefix-profile-20261029-t1200/result.json)은 원래3관측을 모두 재생해
  출력을 대조했다. 두 번째/세 번째 계측 재생은22.892/28.745ms였다.
- 게임을 제동한 채 API로만 재개한 [t1201](../runs/flight-graphed-running-engine-resume-20261029-t1201/result.json)
  뒤, 같은 원본을 [t1202](../runs/flight-graphed-running-engine-profile-20261029-t1202/result.json)에서 재생했다.
  두 번째/세 번째는25.783/30.019ms였다. [t1203](../runs/flight-graphed-running-engine-pause-20261029-t1203/result.json)으로
  pause를 확인했다. 엔진의 지속 실행만으로 앞선74ms 지연을 설명할 수 없으며 단일 원인으로 확정하지 않는다.
- [t1204 자원 표본](../runs/flight-graphed-running-resource-20261029-t1204/result.json)의 수집 의도는 동시 계측이었지만,
  QPC를 대조하면 실제 본계산 종료 약2.65초 뒤 시작했다. 실패 순간이나 t1202 계산 중의 부하 증거로 쓰지 않는다.

재개 직후 준비의 영향을 구분하기 위해 별도 계획
`configs/flight-physical-yaw-graphed-settle-v2-20261029.json`을 만들었다.
첫 actor 관측 전에 실제 native 센서를2초간 보존하며 정차/제동을 확인한다.
이 준비 중 actor 관측을 만들거나 상태를 초기화하지 않는다. 원래257관측·시간·가드·12회 비교 조건은 유지한다.

| 실제 기록 | 판정 |
|---|---|
| [t1205/trial-00 full](../runs/flight-graphed-settle-block0-20261029-t1205/trial-00/result.json) → [t1206 독립 감사](../runs/flight-graphed-settle-full-audit-20261029-t1206/result.json) | 257관측 전체 완료·재생, 의도 최대 오차1.23e-7. native716기록, 최고6.159km/h, 위치 변화9.383m, 교란280.802ms. 최대 관측 간격42.431ms, 추론 중앙값27.119ms/p95 30.240ms/최대33.172ms |
| [t1205/trial-01 yaw0](../runs/flight-graphed-settle-block0-20261029-t1205/trial-01/result.json) | 195관측·회복 단계37까지 진행 후 진단 기록50ms 신선도 오류로 중단. 원본과 실패를 보존하고 제동·pause 확인. 완전한 비교 시험으로 세지 않음 |

2초 준비 후 full1회 성공은 준비 지연의 인과 효과나 반복 안정성의 증명이 아니다.
완료한 full의 페달은 여전히 시험용6km/h 제어기다. yaw0의 불완전한 회복 구간으로 안정화 개선을 판정하지 않는다.

## 진단 신선도 보완과 다음 작업

`ac_file.py`의 snapshot은 읽기 충돌과 오래된 동일 순번에 대해 합계 최대10ms 안에서
새 파일 순번을 기다릴 수 있다. 기존 순번의 최초 수신 시각은 보존하며,50ms를 넘은 기록은 반환하지 않는다.
새 기록이 오지 않으면 실패한다. actor 영상·명령의150/80ms 제한은 바꾸지 않았다.
관련27검사 통과; [t1207 읽기 전용50표본](../runs/flight-diagnostic-fresh-wait-readonly-20261029-t1207/result.json)에서
최대 원래 기록 나이41.710ms, 최대 snapshot 호출0.443ms를 확인했다.
실제 이 표본에서는 재시도0회였으므로 새 순번 대기 분기는 합성 게시자 시험으로만 검증했다.
보완한 transport의 실제 이동 중 동작은 아직 재검증하지 않았다.

다음은 transport 변경을 명시한 계획/출처로 실제 반복 비교를 재개하고,
진단 중단이 재발하면 마지막 원본 순번·시각·wire bytes와 게시자 정체를 기록하는 것이다.
이전 full1회의 재사용은 원본 경로·해시와 변경 전후 동등성 검토를 명시한다.
실패한 neural episode를 중간부터 이어 붙이지 않는다.

이 단계의 실행·감사 프로세스는 모두 종료했다. 마지막 실제 주행 종료 기록은 제동·중립·pause 확인이다.
재개 시 실제 상태를 다시 확인한다. UI 입력·다른 작업 종료·새 optimizer 학습은 없었다.
