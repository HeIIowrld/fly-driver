# FlyBrain ZIP 추가 이동 결과 — 2026-11-05

검증 완료: 2026-11-05T11:10:13.165458+09:00.

**이번 추가 이동: 177개, 64,873,310,630 bytes (64.87GB). 이전 작업 포함: 212개, 67,613,823,552 bytes (67.61GB).**

| 구분 | 파일 수 | 용량 | 선별 근거 |
|---|---:|---:|---|
| 앞선 1차 작업 | 35 | 2.74GB | 현재 코드·설정 및 최근 기록에서 실험 이름 참조가 없는 과거 카메라 ZIP |
| 이번 10월 자료 이동 | 141 | 48.24GB | 종료된 10월 카메라 원본·주행 시험·이전 corpus의 보존용 묶음. 현재 actor 로더는 PT/그래프/포트 상태를 읽고 현재 입력은 live 영상임. 현재 학습 경로는 11월 actor-only/특징·상태 NPZ를 읽음 |
| 이번 11월 실패 기록 이동 | 36 | 16.63GB | demonstration 결과가 incomplete_teacher_demonstration 32개 / failed_camera_demonstration 3개 / failed 1개. 현재 chronological train/heldout 세 원본은 제외했고, 실패 결과 해시를 다시 확인함 |

## 보존한 자료

현재 `student-split-01.json`의 train/heldout 세 원본 영상은 그대로 NAS에 있다.

- TRAIN: `teacher-synchronized-20261102/qualified-series-04/episode-02/demonstration`
- TRAIN: `teacher-synchronized-20261102/qualified-series-06/episode-02/demonstration`
- HELDOUT: `teacher-synchronized-20261102/qualified-series-11-memory-receive/episode-01/demonstration`

온도 조건 학습 원본, 최신 shadow/학생 제어·카메라 reconciliation 기록, 모든 PT 체크포인트·optimizer·NPZ 학습/특징 배열, 원래 metadata와 원본 소스 스냅샷 ZIP도 보존했다. 초기 ZIP 목록에서 이동하지 않은 항목은 82개, 12.35GB이며 모두 원래 경로에 남아 있음을 확인했다. 이후 다른 작업에서 새로 생성한 ZIP은 이 초기 목록의 수치에 포함되지 않는다. 보존한 ZIP 모두가 현재 optimizer에 필수라는 의미는 아니며, 현재·최근 검증과 원본 재현 경로를 유지한 것이다.

## 검증과 이동 방식

각 파일의 원본 bundle.json/manifest 해시와 크기를 먼저 확인했다. 원본을 잠금 상태로 열어 다른 프로세스가 쓰거나 바꾸지 못하게 하고, 전송하면서 전체 원본 SHA-256을 계산했다. D 복사본 전체 SHA-256, ZIP 전체 CRC, archive.json member 목록을 확인한 뒤에만 잠근 원본 파일 핸들로 NAS 파일을 제거했다. 사용 중인 파일은 건너뛰도록 했으며 실제 건너뜀·오류는 0개다. CRC가 정상이라는 것은 원래 기록이 학습 적격이라는 뜻이 아니며 실패/데이터 역할은 그대로 보존한다.

완료 후 212개 모두 D 파일의 존재·크기와 원본 경로 부재, 이동 포인터와 SHA-256 영수증의 일치를 대조했다. 전송 시 수행한 전체 해시/CRC 검증을 최종 영수증과 연결했다. 보존 대상으로 따로 기록한 모델·계획·코드 19개를 다시 해시해 변경 0개를 확인했다. 무거운 학습/시뮬레이터 재실행이나 코드 수정은 하지 않았다.

## 위치와 복원

D에는 각 배치의 `payload/runs/...` 아래 원래 경로 구조가 보존된다.

- 1차: `D:/CodexStorage/flybrain-zip-move-20261105/`
- 2차: `D:/CodexStorage/flybrain-zip-move-20261105-batch2/`
- 3차: `D:/CodexStorage/flybrain-zip-move-20261105-batch3/`

각 폴더의 `result.json`에 원래 경로·D 경로·바이트 수·SHA-256이 있고, `events.jsonl`에 검증 후 제거 기록이 있다. NAS의 각 원래 폴더에는 `camera.zip.moved-to-D.json`을 남겼다. 원래 bundle.json/archive.json의 bytes는 수정하지 않았고 D에도 복사본을 보존했다.

**옛 ZIP을 직접 읽는 과거 스크립트를 재실행하려면 해당 ZIP을 먼저 복원해야 한다.** 현재 소스의 pin/hash를 유지하기 위해 자동 리디렉션은 추가하지 않았다. 예를 들어 프로젝트 루트를 작업 폴더로 두고 다음을 실행한다.

```powershell
python D:/CodexStorage/flybrain-zip-move-20261105-batch2/restore_selected_zips.py --path "runs/ac-shared-fullbatch-student480-trial-20261029-t929/vision/camera.zip"
```

배치에 맞는 restore_selected_zips.py와 result.json의 정확한 path를 사용한다. `--path`를 반복할 수 있다. 복원은 D 사본과 NAS 복사본의 전체 해시를 검증하며 D 사본을 제거하지 않는다.

## 영수증과 OneDrive 반영 범위

- 전체 최종 영수증: `D:/CodexStorage/flybrain-zip-move-20261105-batch2/combined-final-verification.json`
- NAS 사본: `runs/storage-relocation-20261105-batch2/combined-final-verification.json`
- 선별·의존 경로 근거: 2차 폴더의 `dependency-evidence/`, `protected-inputs-before.json`; 3차 폴더의 `failure-evidence/`, `plan.json`
- PC OneDrive 경로 검사: 2차 폴더의 `pc-onedrive-path-check.json`

NAS에서 D로 분리한 논리 파일 용량이다. OneDrive 서버의 실제 quota 감소량은 확인하지 않았다. NAS 휴지통/스냅샷과 OneDrive 휴지통은 비우지 않았다.
