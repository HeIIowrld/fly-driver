# 독립 서킷 영상 저장본 재확인 · 2026-11-11

프로젝트의 `runs`, `assets`, 로컬 연구 저장소 `D:/CodexResearchRuns`에서 카메라 스트림과 ZIP 번들 322개를 찾았다. 이미지 파일은 열지 않았다. 매니페스트에서 스트림 해시를 확인하고, 네 방향 영상과 주행선 꺼짐이 기록된 캡처의 세대별 처음·중간·마지막 메타데이터에서 서킷 ID를 읽었다. 확인된 269개는 모두 기존 실험에 사용된 일곱 서킷이었다. [검색 코드](../runs/vision-corner-independent-source-rescreen-20261111-01/rescreen.py) · [아카이브 목록과 메타데이터 해시](../runs/vision-corner-independent-source-rescreen-20261111-01/result.json)

| 저장 위치 | 발견 | 서킷과 네 방향·주행선 조건 확인 |
| --- | ---: | ---: |
| 프로젝트 runs | 179 | 136 |
| 프로젝트 assets | 0 | 0 |
| 로컬 연구 저장소 | 143 | 133 |

나머지 53개는 매니페스트 없음 4개, 검증할 수 있는 스트림 없음 10개, 네 방향·주행선 꺼짐 조건에 해당하는 프레임 없음 39개였다. 이 재확인은 아카이브 후보 검색이며 원본 전체의 촬영 조건을 입고 심사한 결과와 구분한다.

Silverstone 1967의 고정 평가 파일 해시, 기존 합격 기준, 90ms 요청 간격을 평가 실행기로 재검증했다. 같은 프로토콜로 읽기 전용 촬영 사전 검사도 실행했다. 실행 중인 엔진은 Spa BMW Z4 GT3 내장 AI 주행이었고 카메라는 비활성이었다. Silverstone과 정차·브레이크·일시정지 조건이 충족되지 않아 결과는 `not_ready`였다. [고정 설정 검사](../runs/vision-corner-independent-source-rescreen-20261111-01/silverstone-lock-check.log) · [사전 검사 영수증](../runs/vision-corner-independent-source-rescreen-20261111-01/silverstone-preflight/result.json)

촬영을 재개하려면 Silverstone 1967 오프라인 핫랩에서 BMW Z4 GT3가 주행선 표시 꺼짐, AI 꺼짐, 정차·브레이크·일시정지 상태여야 한다. 첫 카메라 묶음 저장을 확인한 뒤 주행을 시작하고, 1,200묶음을 입고 검사와 고정 채점기에 넣는다. [고정 촬영 절차](corner-silverstone-independent-preflight-20261111.md)
