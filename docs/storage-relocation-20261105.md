# FlyBrain 과거 카메라 ZIP의 D 드라이브 이동

- 작업일: 2026-11-05 (Asia/Seoul)
- 사용자 요청: ZIP 용도를 확인하고 현재 학습에 영향 없는 파일부터 D 드라이브로 이동.
- 완료: **35개, 2,740,512,922 bytes (약 2.74GB)**. NAS 원본 ZIP을 제거하고 D 복사본을 보존했다.
- 대상 용도: 2026-10-28~29의 종료된 카메라 전송·조향·학생 시험 영상/메타데이터 재현 묶음. 모델 체크포인트나 NPZ 학습 데이터가 아니다.
- 이동 위치: `D:/CodexStorage/flybrain-zip-move-20261105/payload/` 아래 원래 `runs/...` 구조.

## 선별과 검증

전체 ZIP 294개(79,960,089,973 bytes)를 목록화했다. 현재 src/scripts/configs의 Python/JSON에서 해당 실험 폴더 이름을 참조하지 않는 10월 camera.zip을 선택했다. 11월 1~5일 실험 폴더의 Python/JSON/JSONL(개별 8MB 이하)에서도 후보 실험 이름이 검색되지 않았다. 대용량 텔레메트리 파일까지 모든 간접 참조가 없다고 증명한 것은 아니다. 실행 중인 native-visual-precorners-20261104와 student-closed-loop-20261105 및 최근 교사/온도 학습 데이터는 이동하지 않았다.

원래 bundle.json의 크기/SHA256 및 archive.json 해시를 확인했다. 복사하며 원본 해시를 계산하고 D 복사본의 전체 SHA256, ZIP 전체 CRC와 member 목록을 검증한 뒤 원본의 크기/수정시각/파일 식별자가 그대로인지 확인하고 해당 ZIP만 제거했다. 완료 후 D 파일 전체 SHA256을 다시 검증했다. 원래 bundle.json/archive.json은 수정하지 않았고, D에도 사본을 보존했다. 파일별 원래 경로·D 경로·해시는 아래 영수증에 있다.

## 과거 실험 재생 시 복원

**이동한 35개 과거 실험의 재생에는 먼저 ZIP 복원이 필요하다.** 자동 리디렉션이나 현재 로더 변경은 하지 않았다. 각 원래 폴더의 `camera.zip.moved-to-D.json`이 D 위치와 해시를 가리킨다. 아래 명령은 선택한 ZIP을 원래 NAS 경로로 해시 검증하여 복원하고 D 사본은 보존한다. 프로젝트 루트를 작업 폴더로 사용한다.

```powershell
python D:/CodexStorage/flybrain-zip-move-20261105/restore_selected_zips.py --path "runs/ac-ai-fourview-phase-probe-20261028-t603/vision/camera.zip"
```

`--path`는 result.json의 정확한 path 필드로 지정하며 여러 번 사용할 수 있다. 필요하지 않은 ZIP을 일괄 복원하지 않는다.

## 영수증

- [이동 계획](../runs/storage-relocation-20261105/plan.json)
- [이동 결과 및 전체 파일 매핑](../runs/storage-relocation-20261105/result.json)
- [복사 검증/원본 제거 이벤트](../runs/storage-relocation-20261105/events.jsonl)
- [완료 후 전체 해시 재검증](../runs/storage-relocation-20261105/final-verification.json)
- D 감사 폴더: `D:/CodexStorage/flybrain-zip-move-20261105/` (전체 초기 목록·스크립트 포함)

NAS의 논리 파일 용량이 감소한 수치이며 OneDrive 서버 사용량 감소를 직접 확인한 값은 아니다. NAS 휴지통·스냅샷 및 OneDrive 휴지통은 건드리지 않았다. 코드·설정·체크포인트·NPZ 데이터·원래 실험 메타데이터는 변경하지 않았다.
