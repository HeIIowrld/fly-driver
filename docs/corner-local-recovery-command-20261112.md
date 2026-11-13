# 원본 영상 재생에서 경계 복구 실행 · 2026-11-12

기존 네 카메라 재생 명령의 `--boundary-recovery local-image` 옵션으로 폭·과거 관측과 현재 영상 전환을 이용한 경계 복구를 실행할 수 있다. 개발 영상에서 동결한 폭 점수 문턱 0.65와 영상 전환 문턱 80을 모든 시험 영상에 적용했다. 원본 영상과 차량 국소 감각으로 계산하며, 코너·물리 경계 라벨은 재생 후 채점에 사용했다. [재생 명령](../scripts/replay_corner_camera_archive.py) · [경계 복구 코드](../scripts/corner_archive_recovery.py)

```powershell
python scripts/replay_corner_camera_archive.py --archives runs/vision-silverstone1967-independent-capture-20261112-05/archive-map.json --expected-track ks_silverstone1967 --boundary-recovery local-image --output runs/silverstone-local-image-replay
```

기존 FlyVis 환경에서 실행한다. 결과 경로는 새 폴더여야 한다. 같은 영상의 원본·반사 T4/T5 추출 기록이 있으면 `--motion-extraction`에 해당 폴더를 지정할 수 있다. 아카이브 목록과 추출 기록의 해시가 맞아야 한다.

- `sensors.npz`는 영상에서 직접 관측한 좌우 경계와 센서 특징을 보존한다.
- `recovered-sensors.npz`는 같은 센서 특징에 누락 경계를 보완한 좌우 좌표를 담는다.
- `predictions/forecasts.jsonl`은 각 프레임의 공통 후보와 경계 기반·T4/T5 판독을 기록한다. 상태는 직선·좌회전·우회전·보류이며 회전에는 4–20m 시작 거리가 붙는다.
- `result.json`에는 영상 묶음, 두 입력, 코드와 모델 및 판독 결과의 해시가 기록된다.

Silverstone 1,200장, Vallelunga 1,200장, Monza 3,152장을 이 명령으로 실행했다. 세 영상의 복구 좌표와 평가 프레임 3,389장의 출력이 기존 검증과 일치했다. Monza 채점은 보존된 989장에 한정했다. Silverstone은 현재 코드로 경계를 다시 계산해 일치 여부를 확인했고, 관측과 복구 후보가 전혀 없는 입력에서 빈 경계가 유지되는 것도 확인했다. [세 영상의 명령 검증](../runs/vision-corner-local-recovery-command-20261112-01/verification.json)

| 같은 영상·후보에서 경계 → T4/T5 | 우회전 방향·거리 적중 | 직선·범위 밖 경보 | 코너 내부 경보 |
| --- | ---: | ---: | ---: |
| Silverstone | 2 → 2 / 13 | 14 → 11 | 5 → 7 |
| Vallelunga | 2 → 5 / 38 | 10 → 6 | 5 → 10 |
| Monza | 2 → 4 / 42 | 12 → 26 | — |

Silverstone 우회전 접근에서 도로 오른쪽 경계 누락이 10→9점으로 줄었다. 620번 전방 10m에서 추가한 좌표의 횡오차는 0.43m였다. 복구 전보다 두 판독의 직선 오경보는 각각 한 건 늘었고 우회전 적중은 유지됐다. 같은 복구 후보에서 T4/T5는 경계 기반의 직선 경보 두 건과 범위 밖 경보 한 건을 없앴다. 코너 내부 경보는 한 건을 없애고 세 건을 추가했다. 적중 기준은 방향과 시작 거리 ±5m의 동시 일치다. 이 재생은 이전에 분석한 저장 영상을 사용했다. 기본 명령은 기존 설정을 사용하며 복구는 옵션으로 실행한다. [경계 라벨 검증](../runs/vision-corner-local-pixel-recovery-20261112-01/equal-cost/verification.json) · [복구 전후 원자료](../runs/vision-corner-local-pixel-recovery-20261112-01/equal-cost/result.json)

기존 재생 명령의 수정 전 바이트는 이전 증거의 SHA-256과 일치하는 사본으로 남겼다. [수정 전 재생 코드](../runs/vision-corner-local-recovery-command-20261112-01/previous-replay_corner_camera_archive.py)
