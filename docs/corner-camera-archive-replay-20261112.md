# 원본 네 카메라 녹화에서 코너 판독 재생 · 2026-11-12

원본 카메라 아카이브에서 앞·뒤·좌·우 영상과 차량의 국소 속도·요율을 읽어 경계와 코너 판독까지 실행하는 명령을 추가했다. 기존 FlyVis 추출기를 별도 프로세스에서 실행해 원본·반사 영상의 T4/T5 특징을 만든다. 같은 프레임 순서로 전방 영상의 경계 모델과 네 카메라의 표면 경계를 적용한다. 이후 이전 센서 재생 명령으로 경계 복원, 코너 후보 확인 및 네 상태·시작 거리 출력을 실행한다. 방향·거리의 정답, 절대 위치 및 지도는 별도 채점에 사용한다. [원본 재생 코드](../scripts/replay_corner_camera_archive.py) · [센서 입출력](corner-sensor-only-replay-20261112.md)

프로젝트 루트에서 기존 FlyVis GPU 환경으로 실행한다. 아카이브 목록은 녹화 이름을 아카이브 폴더에 대응시키는 JSON이다.

```powershell
python scripts/replay_corner_camera_archive.py --archives runs/vision-silverstone1967-independent-capture-20261112-05/archive-map.json --expected-track ks_silverstone1967 --output runs/vision-corner-archive-observer-20261112-01/silverstone-replay
```

`predictions/forecasts.jsonl`에 프레임별 두 판독의 상태와 거리, `predictions/boundaries.npz`에 관측·복원 경계가 나온다. 아카이브, 센서 특징, 체크포인트, 출력의 해시를 기록한다. `--motion-extraction`을 쓰면 해당 아카이브 목록과 원본·반사 추출 설정 및 파일 해시가 일치하는 기존 추출을 재사용할 수 있다. 지정 서킷과 실제 아카이브의 서킷이 다르면 출력 전에 종료하는 것을 확인했다.


## 원본 재추출 검증

Silverstone 1967과 Vallelunga Extended의 기존 녹화 각 1,200장을 원본부터 다시 처리했다. 각 녹화의 좌우 경계, 복원 여부, 최종 네 상태와 코너 시작 거리는 이전 재생과 일치했다. T4/T5 특징의 최대 절대 차이는 Silverstone 6.29×10⁻⁷, Vallelunga 6.36×10⁻⁷였다. 두 녹화에서 각각 1,198장의 T4/T5 특징을 얻었으며, 처음 두 장은 세 프레임을 사용하는 특징에 필요한 앞선 관측이 없었다. Silverstone의 우회전 동시 적중은 두 방법 모두 2/13이었다. Vallelunga에서는 같은 후보에서 T4/T5를 추가한 조건의 우 동시 적중이 3→5/38이었다. [Silverstone 검증](../runs/vision-corner-archive-observer-20261112-01/silverstone/verification.json) · [Vallelunga 검증](../runs/vision-corner-archive-observer-20261112-01/vallelunga/verification.json)

기존 Monza 공식 랩 두 편에서 네 카메라와 차량 감각이 담긴 3,152프레임을 확인했다. 현재 판독기의 경계·코너·직선 학습에 Monza는 포함되지 않았다. 과거 연구에서 사용한 녹화이며, 별도로 보존된 989개 라벨 중 4–20m 접근은 좌 24장과 우 42장이다. 이번에는 이 대상을 고정하고 판독을 시작했다. 기존 Monza 라벨은 코너 내부 프레임을 제외하므로 이 결과로 코너 내부 오경보나 물리 경계 오차를 평가하지 않는다. [Monza 대상과 출처](../runs/vision-corner-archive-observer-20261112-01/monza-transfer-inputs/result.json) · [실행 시작 기록](../runs/vision-corner-archive-observer-20261112-01/monza-transfer-inputs/launch.json)


Monza 재생 및 채점이 완료됐다. [Monza 전이 평가와 측면 경계 검증](corner-outward-camera-monza-20261112.md).
