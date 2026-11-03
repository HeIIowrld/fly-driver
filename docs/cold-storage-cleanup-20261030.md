# 종료 자료 이동과 중복 이미지 정리 — 2026-10-30

종료 자료306묶음·966,818파일·167,061,543,849byte의 D 이동과 검증된 중복 이미지 정리를 완료했다. 최종 원본 session38865/chunk `52c8ad`는 `eedf96` 종료0, 상태 `completed_compaction_after_verified_recovery`다. 80개 case 중79개가 ZIP·member·원래 raw SHA 및 이미지 decode 검증을 통과했고, 알려진 불일치1개는 전체 보존했다. 검증된 이미지254,676개, 논리44,021,765,304byte를 삭제했다. 디렉터리·metadata 삭제와 자료 역할 변경은0이며 고유·미대응 파일과 canonical ZIP은 보존했다. 학습 허용은false다.

원본 이동36125는284묶음 뒤 종료1로 실패했고 별도 복구85116이 남은22묶음을 종료0으로 마쳤다. 이후 이미지 정리95096은 실측한 매 이미지 fsync 비용 때문에 명시 중단(종료1)했다. 원래 로그를 보존하고 복원78267을 `eed548` 종료0으로 마쳐5개 case의20,052이미지를 복원했다. 복원 검증은80개 원래 파일 이름·크기와 누락0이며, 그 단계에서 다른 내용 전체를 재해시한 것은 아니다. 초기 최적화 `4f8cb9`의 PID 가드 실패도 output 생성·DB·payload 접근 전 실패로 보존했다. 별도 identity 수정판의 최종 성공으로 앞선 실패를 바꾸지 않는다.

root의 최종 `1fa9da` 종료0은 launcher71640·worker37604 부재, SQLite exclusive 열기, 같은 generation의 새 idle 상태를 확인했다. native의 metadata 교차검토 `94eddf`도 종료0으로 최종 개수와 영수증의 일치를 확인했다. MAIN·설치807은 유지했고 새 배포·수집은 실행하지 않았다.

| 보존 증거 | SHA256 |
|---|---|
| [이동 aggregate](D:/CodexResearchRuns/flybrain-cold-20261030/result.json) | `d59b5e337ebb1ed5c9da0342edbc57e164ae1eeb0efed9e6d78a19bf86641e4f` |
| [80개 명시 매핑](D:/CodexResearchRuns/flybrain-cold-20261030/zip-raw-candidates80.json) | `411dd34de6724124b95f77525c9542c3a077feb4cbbe190ca009236907e2effb` |
| [95096 중단 영수증](D:/CodexResearchRuns/flybrain-cold-20261030/pixel-compaction-actual-v1/root-stop-terminal-receipt.json) | `2c0577c07f34f75670ea776a8f161c8c535ff1ee8a8d7e24620fae3b6ea9c328` |
| [복원 소스](D:/CodexResearchRuns/flybrain-cold-20261030/restore_compacted_pixels_v1.py) | `1beb5510dc7ffbe39392848bb9727399091a882d0c1e9a01a28da3165982dbba` |
| [복원 결과](D:/CodexResearchRuns/flybrain-cold-20261030/pixel-restoration-actual-v1/result.json) | `b25a3965b0a57371f0e460ea144562474dadfdc79a380722c172c28749104cd3` |
| [복원 종료 영수증](D:/CodexResearchRuns/flybrain-cold-20261030/pixel-restoration-actual-v1/terminal-receipt.json) | `0b6ea2add7c986ad3502e6fbfcdb75a684cfb852fde52e8bda8662311095e2fa` |
| [최종 정리 소스](D:/CodexResearchRuns/flybrain-cold-20261030/verify_and_compact_archived_pixels_buffered_identity_v2.py) | `35642bc1812fa8291c352b960f58c4441def37e7900bac5b0c88bd09606c987a` |
| [최종 process 영수증](D:/CodexResearchRuns/flybrain-cold-20261030/pixel-compaction-buffered-identity-v2-actual/execution-process-receipt.json) | `99bd3918e529224eb78c1da7ccdae3cafe043e8496bfb09dae89a9a8916ddbd8` |
| [최종 정리 결과](D:/CodexResearchRuns/flybrain-cold-20261030/pixel-compaction-buffered-identity-v2-actual/result.json) | `40b508882e0f465a0094d72dad616bbaabe702d3257a2187ce22c60d46a1e2cd` |
| [최종 종료 영수증](D:/CodexResearchRuns/flybrain-cold-20261030/pixel-compaction-buffered-identity-v2-actual/terminal-receipt.json) | `b09c28f8e3b53dafcf8b8c8c97caea543f8d7f6414ffe10618f36384b90fd665` |

불일치 case `async-d6edaa574cf3486fa818151d6e6582ad`는 t1335 episode09의 producer131/수용130 기록을 그대로 보존한다. 직접 old raw 경로를 `Path.open`하는 코드는 자동으로 ZIP으로 전환되지 않는다. 삭제된 cache가 필요하면 보존 ZIP과 **해당 정리의** `recovery.jsonl`에 있는 원래 경로·member·SHA·D 시각을 묶어 별도로 검증·복원해야 한다. 95096 전용 복원 소스를 최종 정리 로그에 자동 재사용할 수 있다는 뜻은 아니다. CameraArchive 기반 재생은 보존 ZIP을 사용한다.

최종 관측 여유 공간은 C351,161,659,392byte/D1,736,809,971,712byte다. 별도 정리와 hardlink가 섞여 있으므로 이 값 전체를 이번 작업의 성과로 귀속하거나, 삭제 논리 용량을 정확한 물리 확보량으로 해석하지 않는다. 실제 새 운전fit·학생배포·유효랩0, 전체 연구 목표active·미완료는 그대로다.
