# 같은 native 원본의 교사 상태 기록과 heartbeat 실행 경로

## 최신: 시각 방향 회로 재생 및 잔차 후보 평가 (2026-10-30)

1917 오프라인 내부 holdout에서 실제 시각 yaw 근사 RMSE 0.16593, 몸 전용 0.07221 rad/s였다. 1919의 알려진 개발 기록에서도 시각 잔차 0.08361, 몸 전용 0.03626으로 후보가 비교를 통과하지 못했다. 새 1909 영상의 저자 시각망 재생은 조직 코드 무결성 정책 차단으로 실패했다. 학생 배포·유효 자율 랩은 0이며 레코드라인 추종은 미검증이다. 기존 페달 fit 123회는 유지되고 별도 몸 baseline 200회는 저자 시각망 학습이 아니다. [실험·제약·원본 증거](visual-yaw-causal-replay-20261030.md). 아래 상태와 다음 작업 표현은 해당 시점의 이력이다.


## 현재: 새 TRAIN 첫 페달 fit 완료, 학생 라인 추종은 미검증

갱신 2026-10-30T16:47:19.284517+09:00. 새 1909 TRAIN986상태를 원래 시간·마스크와 함께 전부 유지해 TBPTT8/한 번의 순회/Adam1e-3로 **실제123회 업데이트**했다. 두 DNg 페달 계수만 바뀌었고 고정 backbone·readout·DNa fingerprint는 동일하다. 학습18.32초/전체59.81초, 같은 TRAIN 평균 페달 MSE0.12065442→0.11707838(2.9639% 감소)다. fullbrake1 오차·정지 hold는 미해결이며 시각 조향·학생 트랙/레코드라인 추종·유효랩 검증은0이다. 전체 비행·겹눈·몸 되먹임·변속·하중/타이어·셋업·빠른 레이싱 라인 목표는 active·미완료다.

[원본 실행·범위·다음 시각 조향 연결](first-new-train-pedal-fit-20261030.md). 원본955d16/session36427→d528e1 종료0, source drift0, 새 작은 두 계수 checkpoint를 보존했다. 현재 fit는 조건부 zero-yaw/속도 과제의 페달 학습이며 레코드라인 학습 성공이 아니다. 아래의 GPU/fit0·미봉인·다음 gradient 표현은 이 첫 fit 이전 진행 시점의 기록이며, 프로젝트 대전제와 원본 이력은 유지한다.

## 현재 상태: 새 TRAIN986상태 원시 감사·전체 로딩 통과, 첫 GPU 전

갱신 t1897–t1913 · 2026-10-30T15:56:26.509004+09:00

초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응까지 포함한 전체 목표는 **active·미완료**다. Actor는 영상·허용된 몸 감각·선언한 외부 과제로 판단하며, world pose·정답 라인·교사 상태는 관찰/교사/평가용으로 유지한다.

- **실제 배포를 완료했다.** Root 원본 `2576c2`는 종료0, 7.08초다. MAIN과 설치 소스는 `80605ed6aa1cf64851aad648b5107c9ea6a98a684e3d583ce0638beba4e7ab5f`이며 이전807 소스는 역사적 기록이다. [1897 실행 결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1897-camera-lifecycle-deployment-registration/execution-result.json) SHA `bb1208931dae40d2be0a6b69ff514a309a98546cbb6d7068630eafa02d032b06`, [1898 배포 결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1898-camera-lifecycle-deployment/result.json) SHA `2647498002d889a79f4037062a511a508c7a2e9f73a2f0dd7d0273489ff7ef5e`를 보존한다. UI·actuator 메시지·AI API 호출은0이며 학생 배포가 아니다.
- **배포 당시 보존 관측:** PID24200, generation `733364ULL:0.0019078`, epoch0, paused·CSP AIoff·gas0/brake1·FREE·비전면·cameraoff·expert inactive·no owner다. [새 generation 상태](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1898-camera-lifecycle-deployment/after.json)를 근거로 하며, paused의 frozen native packet을 새 학생 실행 권한으로 해석하지 않는다.

### 실제 3초·30초 원본 검사

| 진단 | 원본 종료 | 관측 / aidOFF 영상 | 평균 / 최대 decision 간격 | 80ms 초과 / camera wait miss | 7개 배우 배열 재생 |
|---|---|---:|---:|---:|---|
| 1899 camera3 | f92e2b/session79844 → 04e5ae 종료0 | 100 / 264 | 30.1857 / 33.5161ms | 0 / 0 | 모두 차0, native packet100개 |
| 1903 pace30 | 291c6e/session94793 → 5cdad9 종료0 | 986 / 1,073 | 30.5507 / 44.0383ms | 0 / 0 | 모두 차0, native packet986개 |

두 원본 모두 first MMF·종료 producer echo와3profile이 일치했고 정상 반납·closed·error없음·threads[]·leasefalse로 끝났다. 1903의 교사 최고 속도는100.701km/h, 최대 Cartesian 변위는366.725m다. 원래 상태·시간·held-camera 이력을 보존했고 실제13 same-copy 소스를 감사했다. 교사 flag는 privileged metadata에만 있으며 배우 배열에 넣지 않았다. 두 진단은 영구 diagnostic/fitfalse다.

- 1899 [부모 결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1899-lifecycle-heartbeat-evidence-camera3/result.json) `79d6c0e0…4e2b`; [1900 raw heartbeat](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1900-lifecycle-camera3-heartbeat-audit/result.json) `99ef27e6…bf3f`; [1901 권한](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1901-lifecycle-camera3-authority-audit/result.json) `dd588a53…2003`; [1902 배열·trace](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1902-lifecycle-camera3-array-trace-audit/result.json) `5821c612…5a84`. 1900의 원래 map 관측 빈 배열은 바꾸지 않고 [별도4개 map 부재 증거](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1900-lifecycle-camera3-heartbeat-audit/retired-map-absence.json) `146f72e0…d7d7`을 추가 바인딩했다.
- 1903 [부모 결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1903-lifecycle-heartbeat-evidence-pace30/result.json) `258e125a…9d6b`; [1904 raw heartbeat](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1904-lifecycle-pace30-heartbeat-audit/result.json) `d0a0fcd5…5483`는42,583검사·4개 map 부재; [1905 권한](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1905-lifecycle-pace30-authority-audit/result.json) `0b69fd3f…b17b`는27,661검사; [1906 배열·trace](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1906-lifecycle-pace30-array-trace-audit/result.json) `0f956179…5168`는 원본3bee02/session9888 → 23bbe7 종료0이다.

새 정책의 실제 자격 범위는 이 두 horizon이다. 모든 OS I/O·스케줄러 공백 해결이나 독립적인 학생 주행 성과는 아직 판정하지 않았다. 카메라40ms·wait40ms·cache110ms·camera guard150ms·전체 이력80ms·heartbeat2초를 유지한다. t1875의 관측0 실패와 t1856의82.979ms 간격/전체 fit 불가는 보존한다.

- **1909 수집 전1907 GET 관측:** 원본 `f4c3ba` 종료0, [결과](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1907-post-lifecycle-fresh-native-observer/result.json) SHA `4c6df721af6dcfc5a31ff96ec8dde4c534a13f9a10605d1bb17c2e3ed92d39bd`. 두 실제 진단의 종료 barrier 뒤 같은 PID24200/gen.0019078에서 spline `.5097106695175171`/lap0, graphics8807207→8807221, paused·CSP AIoff·gas0/brake1·FREE·비전면·no owner를 확인했다. physics187028의 native AI=true는 frozen baseline이며 학생 권한이 아니다. 이 GET 자체는 새 페달 보정이나 역할·nonce·fit 허용을 발급한 것이 아니다. 후속1908의 별도 사전 선언을 아래에 보존한다.

### 사전 봉인한 새 TRAIN1909: 전체 원래 이력 확보

1908에서 원래 역할을train으로 선언하고 별도 nonce `cb0c1f6723724f16ae6a6fa5047121d9`를 수집 전에 봉인했다. [원래 선언](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1908-b2-TRAIN-role-registration/sealed-before-capture-train-declaration.json) SHA `1b9e7a062b6c4b982c5576e0a0458d1a638a595ab5ee19af9b48f14b4f293f42`, 소스105개다. 이전 진단의 역할 승격이 아니다. 실제 원본 `04d6a1/session82172`는 `248bd4` 종료0으로 끝났다.

| 항목 | 실제 결과 |
|---|---|
| 원래 상태·시간 | 986상태, body21·원래 mask·held-input clock 보존, drop0 |
| 원래 decision 간격 | 평균30.4928ms·최대47.795ms·80ms 초과0·camera wait miss0 |
| 원시 감각 검사 | 7배열 차0·aidOFF1,052영상·native packet986개·교사 최고101.015km/h |
| 페달 loss cell | gas974·brake974, fullbrake1 원래266개/eligible257개 보존 |
| 지리 범위 | 원래1274개 표본 pose가 `.5097104311`–`.5919309258`로B 안에 있음; `continuous_geometry_verified=false`, heldout 독립성 주장없음 |
| 준비된 TBPTT8 예산 | 124개 전체 window·123개 supervised update budget; 실제 optimizer update0 |

[1909 부모](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1909-new-pedal-b2-TRAIN/result.json) `eed5759c…ddb6` / [recording](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1909-new-pedal-b2-TRAIN/recording/result.json) `e34fb53c…7f2b`. [1910 raw heartbeat](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1910-new-pedal-b2-TRAIN-heartbeat-audit/result.json) `9ddb1feb…df4a`는41,366검사·4개 map 부재·선언 포함106bindings를 확인했다. [1911 same-copy](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1911-b2-TRAIN-authority-audit/result.json) `edd2daf2…9906`는27,661검사·실제13 helper와105개 봉인 소스 전후 일치를 확인했다. B2 부모 프로그램4a00034c와 하위 recorder16ad12a8을 각각 보존했다. [1912 원시 배열·trace](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1912-b2-TRAIN-array-trace-audit/result.json) SHA `4d80551c916cc1037aa48afbc13a8ed0978a0e5d7419f9d765b46057b74ae397`, [1913 전체 이력 로딩](C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/20261030-t1913-b2-whole-history-load/result.json) SHA `8f0eb56d213accd1a8a34571fdd69674981e8ac1a874ef01a34c3233b1814067`을 보존한다.

root는 **원래 관측된 전체 이력**에 대한 첫 offline fit을 이 표본 기반 범위로 허용했다. 관측 사이의 연속 기하를 인증한 것은 아니다. 새 이력의 모델/GPU 초기화·optimizer·학생 실행·유효 랩은 현재0이며, 손실 마스크 밖 상태도 삭제하거나 시간을 다시 맞추지 않는다. 정차 fullbrake1의 zero-task/quiet hold 한계도 유지한다.

현재 첫 실행 전 입력 계약을 정리 중이다. 기존1908 원래 pedal review `89d9b19d…704b2`는 status 없는 원본 bytes/SHA로 loader에 전달하고, nonce를 묶은 typed `continuity_review_only` 증거는 별도로 받는 최소 신규 trainer/plan 파생본을 준비·검토한다. 원래 review와 예전 trainer를 고치지 않고 OriginalGraphAdapter·학습 engine·backbone·optimizer는 유지한다. 실제 새 이력 gradient gate와 실행 결과는 다음 원본 증거로 판단한다.
- **다음 실제 단계:**관측된 원래986상태 전체를 유지한 첫 offline fit의 원본 review 입력/typed 증거 분리와 새 이력 gradient 자격을 확인한다. 124window/123supervised step은 예산이며 실제 update는0이다. B2 새 TRAIN 역할과 자료 확보는 완료했지만 fit·학생 배포·유효 랩은 아직0이고 전체 목표는active·미완료다.
- **이전 저장공간 turn도 실제 진전으로 보존한다.**306묶음·167,061,543,849byte 이동과79개 case 검증/중복 이미지254,676개 정리를 완료했다. 원본 실패·별도 복구·불일치1개와 ZIP·고유 파일을 보존하며 논리44,021,765,304byte를 물리 여유 공간 증가량으로 환산하지 않는다. [완료 증거와 복원 제약](cold-storage-cleanup-20261030.md)을 따른다.

아래는 이전 체크포인트다. 당시의 배포0·MAIN807·이동 중·다음 작업 표현은 위 현재 상태를 대체하지 않는다.

## 이전 체크포인트: 저장공간 정리 완료·배포 실행 전

- D 이동은306묶음·966,818파일·167,061,543,849byte 모두 완료했다. 원본36125는284묶음 뒤 종료1을 보존했고 별도 복구85116이 나머지22묶음을 종료0으로 완료했다. aggregate `D:/CodexResearchRuns/flybrain-cold-20261030/result.json` SHA는 `d59b5e337ebb1ed5c9da0342edbc57e164ae1eeb0efed9e6d78a19bf86641e4f`다. 기존 C 경로는 D junction으로 유지한다.
- 실제 중복 이미지 정리95096은 매 이미지 fsync의 실측 비용 때문에 의도적으로 중단해 종료1로 보존했다. 실제 실행 소스는 `verify_and_compact_archived_pixels_recovery_v1.py` SHA `4f072c1d11e189d3793354603783c2dc722bca4677fbdf97429ea5cb42c3b672`다. 원래 로그와 중단 영수증 SHA `2c0577c07f34f75670ea776a8f161c8c535ff1ee8a8d7e24620fae3b6ea9c328`을 보존한다.
- 복원78267은 원본 chunk `eed548` 종료0으로5개 case의20,052이미지를 복원했다. 소스는 `restore_compacted_pixels_v1.py` SHA `1beb5510dc7ffbe39392848bb9727399091a882d0c1e9a01a28da3165982dbba`다. 전체80개 원래 namespace의 이름·크기가 일치하고 누락0이며, 다른 내용 전체를 재해시했다는 뜻은 아니다. `D:/CodexResearchRuns/flybrain-cold-20261030/pixel-restoration-actual-v1/result.json` SHA `b25a3965b0a57371f0e460ea144562474dadfdc79a380722c172c28749104cd3`, root 종료 영수증 SHA `0b6ea2add7c986ad3502e6fbfcdb75a684cfb852fde52e8bda8662311095e2fa`가4개 PID 부재와 SQLite exclusive 확인을 묶는다.
- 이후 최적화 실제 실행 `4f8cb9`는 옛 PID 부재 가드에서 종료1로 실패했다. output 디렉터리는 생성되지 않았고 DB·payload 접근 전이다. 당시 CIM의 옛 PID들은 부재했고 Windows PID 재사용 가능성을 점검했다. 이 실패 실행은 output 디렉터리 없이 보존한다.
- 복원 완료와 별도 identity 가드 수정 뒤 최종 원본 session38865/chunk `52c8ad`는 `eedf96` 종료0, 상태 `completed_compaction_after_verified_recovery`로 완료했다. 소스 `verify_and_compact_archived_pixels_buffered_identity_v2.py` SHA `35642bc1812fa8291c352b960f58c4441def37e7900bac5b0c88bd09606c987a`다. 80개 case 중79개가 ZIP/member/원래raw SHA·이미지decode 검증을 통과했고 알려진 불일치1개는 보존했다. 이미지254,676개·논리44,021,765,304byte 삭제, 디렉터리/metadata 삭제0·자료역할 변경0·고유파일 보존·fitfalse다. result SHA `40b508882e0f465a0094d72dad616bbaabe702d3257a2187ce22c60d46a1e2cd`, 종료 영수증 SHA `b09c28f8e3b53dafcf8b8c8c97caea543f8d7f6414ffe10618f36384b90fd665`. root 원본 `1fa9da` 종료0이71640/37604 부재·SQLite exclusive·같은generation의 새idle을 확인했다. [경로·해시·복원 제약](cold-storage-cleanup-20261030.md)을 따른다.
- canonical ZIP과 원래 로그, 실패·불일치1개 case `async-d6edaa574cf3486fa818151d6e6582ad`는 보존한다. 직접 raw 경로 읽기의 자동 ZIP fallback은 없으며 고유·metadata·하위 폴더는 삭제 대상이 아니다. hardlink 해제로 얻은 실제 물리 여유 공간은 추론하지 않는다. 최종 관측 C 여유351,161,659,392byte/D 여유1,736,809,971,712byte에는 별도 정리가 섞여 있어 이 작업의 성과로 전부 귀속하지 않는다.
- t1895/96 카메라 정책은 소스 준비·검토 상태이고 새 배포·3초/30초 자격은 미실행이다. 새 운전fit·학생배포·유효랩0, 전체 목표active·미완료를 유지한다.

아래의 live·이동 중·정리0·다음 후보 표현은 해당 시점의 과거 체크포인트이며 위 현재 상태를 대체하지 않는다.

## 이전 체크포인트: 저장공간 정리와 다음 실행

사용자 최신 지시·저장공간 정리 진행 · 2026-10-30T12:52:56.331594+09:00

- 실제 shell 작업 디렉터리는 계속 `\\hyunjin\homes\hjpark\02_Personal\01_학교\20.활동\개인프로젝트\flybrain`다. 앱에서는 flybrain이 primary이고02_Personal은보조 폴더지만, 이 기존 chat의 cwd metadata는02_Personal로남아있다. 현재chat-only 변경 도구/UI는없어임의DB·새chat/worktree를만들지않고모든exec에정확한flybrain workdir를명시한다.
- 사용자가 C의 당장 불필요한 자료를D로옮기고중복파일을정리하도록명시승인했다. C의본작업4개폴더는논리186.15GB였고, 종료확인된33개큰old runtime+273개camera/client폴더306개·167,061,543,849byte를대상으로D이동중이다. 환경·현재소스/최근run·csp-session/globalcamera status·신규수집경로는C에유지한다.
- **원본 이동 job은 실행 중**: original906c64/session36125, Python redirector PID14176→실제worker35296(같은한job), 시작2026-10-30T12:34:46+09. source/D/삭제경계검사+SHA256검증복사→C원본짧은backup→C경로junctionD→검증된backup만삭제한다. BMP/PNG/NPZ/PT/ZIP의정확size+SHA동일payload만Dhardlink로공유한다. 원래mtime은Dledger에보존한다. 모든다른소유capture/model/큰datajob은보류하며mtime/sourceinventory/retire직전idle·프로세스검사를유지한다.
- 상태는 `D:/CodexResearchRuns/flybrain-cold-20261030/{plan.json,progress.jsonl,execution-process-receipt.json,manifests/}`에서확인한다. 이번관측 moved_groups=38, moved_logical_bytes=70473481683, currentevent=copy_progress; 이것을최종완료로해석하지않는다. 관측timeout시같은36125를계속poll하고원본이살아있으면재시작하지않는다.
- 사용자가별도로80종료세션원본+ZIP41.7GB중복추천을문의했다. 현재이동과겹쳐삭제하지않으며, metadata로ZIP↔raw매핑을찾고모든memberSHA/실제재생/현재직접reader미사용을확인한뒤검증된중복만정리한다. bundle은flat일부확장자만포함하므로고유파일·하위폴더는계속보존한다. CameraArchive의bundle경로는ZIP만읽지만raw경로자동fallback은없다는제약도기록한다.
- 주행측은1895새MMF lifecycle-only status후보/1896독립검토까지준비완료: CPU3그룹+Lua compile+Ruff,독립88검사, candidateLua80605ed6aa1cf64851aad648b5107c9ea6a98a684e3d583ce0638beba4e7ab5f. source91, old99변경0, client13SAME_COPY_SOURCE_FILES다. 1895 result319de7c7a32b903c94591ef8a5c652761b5555884bfdc758aa4bfc2f4e2776c9 /1896 b2f0bbfe14fa4d3fb8db6e19fa2c3aa7ea7349d913d541917d20f99ac42484e3. 현재MAIN/설치807·gen.0013066은그대로이고신규배포/3초/30초/학생0이다.
- NEW strict installer `scripts/deploy_ac_camera_lifecycle_v1.py` SHA063c3ca746e58081858ebcbe6b82ceb8ef397ac7cd835a723aafd3ea99712e4c 및C `runtime-scripts/deploy-camera-lifecycle-1897.py`는prepromotionFREE/native·실제설치직전expectedPID/gen/epoch/background를보완해읽기전용peer를통과했다. 아직등록plan/실행0. D이동이끝난뒤1897등록→1898배포→1899camera3→각오프라인감사→1903pace30 순서를준비한다. 숫자는prospective이며해당실행은없다.
-1883 B1 source9개/35CPU/Ruff/closure85는준비완료(실제bindingnull·pipelinefalse). 새13-source producer/host는B1에자동승계할수없어B2와새역할선언이필요하다. 실제새운전fit0/학생배포0/유효랩0, 목표active·미완료이며정지hold·넓은조향·하중/타이어·시각선행·빠른유효완주전체목표를유지한다.

아래는 이전 체크포인트다.


## 최신 병목과 실행 상태

갱신 t1875–t1878 · 2026-10-30T11:40:09.333228+09:00

**현재 병목은 새 학습 계산이 아니라 실시간 자료 수집이다.** 새 운전 fit·학생 배포·학생 유효 랩은 모두 0이며 전체 목표는 active·미완료다. 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응 및 actor의 영상/몸 감각 정보 경계를 유지한다.

- **t1875 첫 실제 R2 camera3는 실패했다.** 원본 session34556/final43c91b terminal1, parent SHA `760a9fb24e3c710e1ed9c1b710d8b0c99e9d709ad6088172d317e5ef5494d21b`. 교사 제어 시작 요청이 거부돼 관측은 0개다. 126 camera groups는 보존했지만 observations.npz는 생성되지 않았고, 실제 30ms decision 분기는 한 번도 수행되지 않았다. 제어 주기 개선은 미판정이다. 1879 pace30은 준비/실행하지 않았고 자동 재실행도 없다.
- 원본 source ledger에서 `csp-camera/status.json`의 동기 `io.save` 430.0603ms를 확인했다. 해당 쓰기 뒤 다음 제어 요청 검사에서 오래된 last-capture가 150ms 소유 카메라 조건을 넘는다는 좁은 원인 분석을 진행했다. 요청 expiry는 남은 2.4964629초로 유효했고 PID/generation/epoch/camera ID·정지 조건도 일치했다. OS I/O·스케줄링 중 어느 것이 실제 430ms를 유발했는지는 미확정이다. 이전 t1856의 83ms actor 간격 원인을 이 결과로 소급 확정하지 않는다.
- **자료 실패 뒤 정상 transport 반납 분기를 처음 실제 검증했다.** cleanup complete/control child39004 terminal1/camera·persistence·precompute·shared reader 종료를 확인하고, 같은 heartbeat writer의 SYNC/ACK·join·lease 제거·closed·threads[]·leasefalse로 끝났다. t1876은 115개 원본 blob/379,078byte·26회 발행·5,786검사·등록99 SHA 일치·정확한 이전4 map 부재를 확인했다. 기존 normal auditor의 failed-parent 거부를 보존하고, 별도 감사에서 정확히 두 status 비교만 failed로 바꾸었다. 원본 자료는 FAILED/fitfalse로 유지한다. 예외적 실패-owner 복구는 수행하지 않았다.
- t1878은 actor trace 10,788검사와 HOST V4 metadata 4,897검사를 통과했다. 431 iterations는 상태 대기 430회와 종료 poll 1회이고 trace 3,875행/93,000byte/decision0이다. status read 최대60.9695ms·중앙값1.97975ms, idle sleep10.0276–11.3722ms를 보존했다. 카메라126그룹 모두 persisted, peak6작업/4,611,677byte/maxpersist210.7965ms다. 이는 원래8개를 넘는 stress가 아니며, 7배열 주행 재생 또는 새 수집 합격을 뜻하지 않는다.
- t1877 GET-only 새 관측은 원본001a9e terminal0/result `e68789de8836648b4fe0ed4aa2e42cabe83b66d4e77d5a2d368f7266d7c5a0d5`다. PID24200/설치와MAIN807/generation `733364ULL:0.0013066`/epoch0, spline `.4202456474304199`/lap0/native174638 그대로, advancing graphics5848271/paused·CSP AIoff·gas0/brake1·FREE·비전면·cameraoff·expertinactive·no lease다. frozen nativeAItrue는 학생 실행 허용이 아니다.
- t1856의 원래 TRAIN697/1,000영상은 여전히 82.979ms 한 간격으로 전체 fit 불가다. 상태·시간·fullbrake1=189를 보존하고 손실1,394비트 전체false를 유지한다. t1867-r2 보관967사본+4메타데이터/14,389,691byte 및 t1874 실제목적지7,749검사는 완료 상태다.
- **다음 좁은 수정:** 현재 소스는 그대로 두고, 새 명시적 MMF 정책에서 초기 capturing·종료·idle 상태는 유지하면서 활성 주행 중 반복 카메라 상태 파일 쓰기를 생략할 수 있는지 source-only 검토한다. HOST V4는 global camera/status를 생성자에서만 읽고, 활성 수집은 MMF와 원래 per-request terminal 파일을 읽는다. 다른 consumer·동시 시작·option 무시·uncertain cleanup 검토 후 새 소스 자격을 받기 전에는 변경/배포하지 않는다. 기존 150ms/80ms/2초 기준을 완화하지 않는다. B 역할은 미봉인·수집0이고 native-only30 예산은 계속 소진 상태다.

학습 처리량은 새 자료의 실제 optimizer를 아직 실행하지 않아 미측정이다. 준비/실행별 소스·감사 비용 역시 진행 지연에 기여했으며, 새 목록 기반 탐색은 준비 비용 개선이지 GPU 학습 속도 향상으로 보고하지 않는다.

아래는 이전 체크포인트이며 당시 현재·미실행·다음 작업은 위 상태를 대체하지 않는다.


작성 시각: 2026-10-30T05:34:18.408118+09:00. 이 판본은 진행 중인 작업의 체크포인트이며 목표 완료 보고가 아니다.

> 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다.

actor는 영상·허용된 몸 감각·외부 과제에서 판단한다. world pose/정답 라인/교사 상태는 진단·교사·평가용이며, 교사 AI flag를 actor 배열에 추가하지 않는다. 가변속·변속·하중/타이어/셋업 인과 적응·유효 완주·라인/랩타임 전체 목표를 유지한다.


## 최신 상태: R2 소스 검토와 실제 보관 완료

갱신 t1871 R2–t1874 · 2026-10-30T11:26:46.425542+09:00

**자율 완주와 새 운전 학습은 아직 0이다.** 초파리의 겹눈/운동 시각과 비행 유래 균형·운동 회로를 자동차라는 가상 몸에 연결한다. 차체 기울기·회전·하중·타이어 상태를 몸 감각으로 되먹임하며 조향·가감속과 빠른 레이싱 라인/랩타임을 학습한다. 변속·셋업 적응까지 포함한 전체 목표는 active·미완료다. actor의 입력은 영상·허용된 몸 감각·사전 선언한 외부 과제이며, pose·지도·정답 라인·교사 flag는 입력하지 않는다.

- t1856의 새 TRAIN 기록은 697관측·1,000영상의 수집과 정상 반납을 완료했지만, 원래 간격 하나가 82.979ms로 80ms 기준을 넘었다. 후보 전체를 fit 불가로 유지한다. 697상태·fullbrake1 표적 189개·원래 시간과 마스크를 보존했고, 진단 로더의 1,394개 페달 손실 비트는 전부 false다. 실제 새 이력 gradient·optimizer·학생 배포·유효 랩은 모두 0이다.
- t1871 **R2** 제어 주기/추적 코드가 CPU 56검사와 Ruff를 통과했다. t1873의 독립 소스 검사 246개와 CPU 총 66사례(저자 56+독립 10)도 통과했다. 승인된 trace 소스 SHA는 `37e1448e2d4f54e57ba991f110e3095c03e6f961ffb721d9b40e9824c817595a`다. 초기 판본에서 발견된 진단 오류의 원래 예외 대체 및 외부 스레드의 조기 필드 변경은 반례와 함께 보존하고 R2에서 고쳤다.
- 새 nominal 30ms 예산은 child poll·상태 읽기 앞에서 시작한다. poll·상태·영상 대기·native 읽기·패킹·인코딩·쓰기·flush·sleep을 고정 RAM 배열 4,882,432byte에 기록하고, 소유 수집기와 기존 heartbeat 작업자가 모두 종료된 뒤에만 저장한다. 원래 센서/결정 시각·이벤트 byte/순서·호출 수를 유지한다. camera 40ms, wait 40ms, cache 110ms, camera freshness 150ms, 전체 이력 80ms, heartbeat 2초 기준도 유지한다. **이는 소스 자격이며 실제 지연 개선의 증거는 아니다.** 새 3초/30초 실제 진단은 아직 미실행이다.
- t1872의 새 소스 목록 기반 import 탐색기는 기존 77/새 83개 의존성 및 73개 Python 파일 SHA와 동등함을 확인했다. CPU 19사례·실제 150검사 통과다. 준비 과정에서 반복 파일 metadata 호출을 줄이는 도구이며, 주행 지연 해결이나 기존 수집기의 변경으로 취급하지 않는다.
- t1867-r2 보관은 실제 완료됐다. `runs/flight-teacher-heartbeat-pedal-20261030-t1867-r2`에 967개 선택 사본과 생성 메타데이터 4개, 총 971파일/14,389,691byte를 저장했다. t1874는 실제 목적지의 SHA·크기·내용 7,749검사를 통과했다. 원시 이벤트·영상·MMF·NPZ/PT는 이 보관에 복사하지 않았다. 첫 t1867의 경로 길이 실패와 22개 부분 사본은 그대로 보존했다.
- 마지막 실제 차량 관측은 t1869다: PID24200, 설치/MAIN807, generation `733364ULL:0.0013066`, epoch0, spline `.4202456474304199`, lap0, frozen native174638. paused·CSP AIoff·gas0/brake1·vJoy FREE·비전면·cameraoff·expertinactive·no lease다. 이 뒤 실제 제어/GPU 작업은 없었다. frozen native AI=true는 새 학생 실행 허용이 아니다.
- 다음은 R2의 별도 3초·30초 실제 진단과 원본 heartbeat/권한/7배열/전체 trace 감사다. 통과 후 새 native 위치·nonce·소스/페달 설정 연속성을 다시 확인해 새 지리 B의 TRAIN 역할을 수집 전에 봉인한다. B 시작 [.49,.54]/전체 [.49,.64]는 아직 미봉인 초안이며, 기존 TRAIN 일부와 겹쳐 새 heldout으로 주장하지 않는다. 보호 구간 [.08,.125], [.2,.25], [.67,.84], [.94,1]은 유지한다. native-only 30회 예산은 소진됐고 추가하지 않는다.

증거 결과 SHA: R2 `7fbc1ba3…04e5b`, 독립 R2 `d9c12bed…c4310`, 목록 탐색 `344cc0fb…e610`, 실제 보관 감사 `9267c9d0…b06`. 원본 결과는 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`의 해당 t1871–t1874 폴더에 있다.

아래는 이전 체크포인트다. 당시 현재 상태·미검증/다음 작업은 위 기록을 대신하지 않는다.

## 이전 실행 상태와 미분 검증

갱신 t1834–t1869 · 2026-10-30T10:13:09.456456+09:00

전체 목표는 초파리 겹눈·운동 시각과 비행 유래 균형·운동 회로를 자동차 몸에 연결하고, 기울기·회전·하중·타이어 감각으로 조향·가감속·변속·셋업 적응을 학습해 빠른 유효 완주·레이싱 라인·랩타임까지 검증하는 것이다. actor에 pose·지도·정답 라인·교사 flag를 넣지 않는다. 이번 운전 학습 업데이트0, 학생 배포0, 유효 랩0. 목표 active·미완료다.

- **현재 MAIN+설치는807e708e098654cfaf8cf58ebd91efc15f850d3da31795b02d28bc063eeb024b**, samePID24200/generation `733364ULL:0.0013066`/epoch0.1856신규TRAIN transport는정상반납했으나82.979ms한간격으로전체fit gate실패/업데이트0이다.1869새native위치 `.4202456474304199`/lap0/frozen174638을확인했다. paused·CSP AIoff·gas0/brake1·FREE·비전면·cameraoff·expertinactive·no lease. Frozen nativeAI를학생허용으로쓰지않는다. 실제신경/GPU/제어작업모두terminal. b6/.0014992는1832복구·1835이동때이력이다.
- 1829 corrected observer의 읽기 전용 preflight는 통과했다. 실제 이동은 첫 자식 명령에서 카메라 신선도 제한으로 실패(원본46750/chunk5db604 terminal1): 15관측·141camera, 최고18.6842km/h·최대위치변위3.1110m. 두 번째 활성화는 없었다. 종료 위치 spline `.09641832113266`/lap0은 보호 EVAL 구간이며 TRAIN 시작이 아니다.
- 1831 독립7479검사: metadata 검증→GPU read 진입에 533.8991ms 공백, 실제 D3D copy+ACK는2.3535ms. CSP의 원래150ms freshness guard가 camera age157ms에서 중단했다. 그 공백을 덮는 기록된 persistence WRITE/queue saturation은 없었다. 실제 어느 OS/Python 호출에서 지연됐는지는 미확정이다. 1832는 예외적 실패-owner 복구이며 정상 handoff나 자료 자격 성공으로 세지 않는다.
- 1828 actual-v2 GPU 원본28433/chunka93d4b terminal0, 18.3978초. 고정4조건·원래CPU quiet·원래 graph·CSR transpose autograd에서 손실미분/페달Jacobian rank2와 CPU parity 통과. state 최대4.47e-8, output/encoded1.0431e-7, gradient2.2352e-8, Jacobian5.9605e-8. 1834 독립403검사/160parity cells; optimizer0·source41변경0. 전체 조건 합의 rank이며 한 상태에서 두 페달 독립 또는 정차 hold 해결을 뜻하지 않는다. fullbrake1 표적과 goal0/body0 미분 한계는 유지한다.
- GPU peak206,028,288B, cleanup 측정 allocated/reserved113,246,208B로 **0이 아니다**. owned_allocator_cleanup_passed=false를 보존하고 원본 프로세스 terminal0과 구분한다. 지금 실행 중인 모델/GPU 작업은 없다. full neural state가 저장되지 않아 state-max 독립 재생은 source/summary 기반이라는 한계도 유지한다.
- 1823r2 TRAIN parent/전체이력 loader와 1825 독립275검사는 준비 완료이나 실제 역할 seal·새 수집·fit은 없다. 시작[.275,.44]/전체[.275,.54]와 DEV/EVAL 보호, 원래 held command·모든 몸21·모든 상태와 dt를 유지한다. 1824는 조건부 입력설정 연속성 점검일 뿐 새 nonce의 물리 응답/조향 자격을 대신하지 않는다.
- **1837 native-only 단일 검증은 실제 완료했다.** 원본session2103/chunk6899af terminal0, result `f8fc37f803eaf7f0fa8d20a1d63c96d32e6b10a7462f881d366f47ec6789df57`. 기존3초60/.9 명령1회, 최고59.04365km/h·최대위치변위40.63998m, 전 `.09641832113265991`→후 `0.1034642830491066`/lap0/정지. 카메라 요청0·heartbeat context0·GPU0·vJoy 획득0. 설치 b6/현재generation 유지, API가 teacher를 끄고 전제동·pause한 뒤 fresh native 관측까지 확인했다. 모든 자료는 transit/fit=false이며 TRAIN/영상 자격이 아니다. 이후1835는 이 실제1회를 포함해 전체30회, 남은29회를 강제해야 한다. 이 결과의 remaining29 필드 자체는 선언값이며 여러 실행의 누적예산 강제를 대신하지 않는다.
- 1838 독립 실제 감사는759검사로 원래167native308 copy와207CSP telemetry를 모두 확인했다(result61ec87d0151a1263b044607609e1138c2da21cfa47c90ce43036ae09468fd9a5). Driving119/stopping88, native 최고58.99625km/h는 CSP 최고와 별도 원본 값이다. paused420 AI=true는 frozen flag이며 새 학생 소유권 자격이 아니다.
- 1836 HOST V4는 소스 준비를 완료했다:26CPU/Ruff/79transitive input drift0(result96834be12edcf6c5712f682f119d71dcf86d13ceb5a9466b839a9ac0e5be2c64). 1840 독립 소스·전용합성47검사(result81aff6335ca9f30e1b2f4d8fe1d3ab1efff66937e348d124e5b77e281dd4dcb6)는 최대32future를 두 capacity drain으로 처리해도 trace140/142events가160 한도 안임을 확인했다. 실제 복사·타이밍·실패 자료 후 정상반납 자격은 아직 없다. 원래150ms/80ms/2초 제한 유지.
- **1835 유한 native-only 이동은 원본13943/chunkcbd077 terminal0으로 종료했다.** 추가29회 모두 정상이며1837을 합친30/30 예산을 소진했다. 최종 spline `.25553950667381287`로 TRAIN 시작 `.275`에 도달하지 못했다. 결과 `6cc168b71fd9a8df4af523ceddfa6234d121cbdfcd63ab33cc5874691f2659ad`의 start_reached=false를 유지하고 native 예산을 연장하지 않는다. 독립5315검사(resultbe34b44acbd7fa618c16410841096bf76ba877f90763a1937a9d1a790a08ffd3)는4477원래native308 HOST copy·5567CSP telemetry·60native264/420 observer pairs와55sourcepins를 검증했다. 모든29 task는 카메라/HB/GPU 없는3초60/.9 진단, 정상 종료·전제동·paused·온트랙·역행/불명확자식0이었다.
- 1844는 검토된 wire-v2 807을 실제 재배포해 원본7602ba terminal0/resultb8f5a665901d081ed9537efd1589472df306278a51f67d4e83329e284b74a912/newgeneration `733364ULL:0.0013066`를 얻었다. 1832 b6 복구를 지우지 않고 현재 설치를 구분한다. source replacement만 수행했고 UI/vJoy/엔진 재시작은 없다.
- **1845 새 HOST V4 camera3는 원본94034/chunk8324af terminal0**, parentresult68b8203d86a97a616f48564536c50d45dfd0ce9cd15e690d18ab946a769ce026. 72decisions/238camera/0wait, 최고56.07278km/h·최대위치변위31.03669m, closed/threads[]/leasefalse/full-byte evidence complete/정상handoff. 1846 원본heartbeat core9454검사 resultd54d49c55081ff7ebbaac0d08847d3c6a1f37c00bc0d0ad64fdbb63eff52165c와원래87sourcepins·옛4map부재,1847 samecopy2063검사/all72AItrue·narroweligible,1848 원래7배열차0/max45.1127ms/>80=0/238aidOFF/72unique packet을 확인했다. 카메라3 이후 저장된CSP receipt 위치 `.26091450452805`/lap0은 아직 TRAIN 시작이 아니다.
- **1849 새 HOST V4 pace30도 원본25131/chunk3a74de terminal0으로 완료했다.** parentresult96243ca520ab75f614515e2508c48be4685ef709f7d54765cd076f6ebeeac368,703decisions/990camera/0wait, 최고100.885895km/h·최대위치변위446.289387m,60→100→정지→60 과제통과/closed/threads[]/leasefalse/full-byte complete/정상handoff.1845·1849는 영구diagnostic/fitfalse이며 native30예산 추가나 TRAIN 역할 승격이 아니다. 종료CSP 위치 `.34021300077438`/lap0에 도달했고1853의 새 native graphics가 `.34021300077438354`로 확인했다.
- 1850 독립76794검사(resultdc6f591c78957a6cad7c8b58e5ec235ab5f425e8900424f4c6239f9f97607936)는 heartbeat원본39080+V4profile37697+binding/timing17을 구분했다.990모두 produced=accepted=persisted, peak4작업/3,074,463B/maxpersist132.2437ms,15860traceevents/최대28events pergroup, namespace·FIFO·RAMtargets·3960view SHA 참조·원래4maps부재를 확인했다. peak4는 old8초과 stress가 아니며889ms 흡수/근본 원인 해결을 주장하지 않는다. metadata→copy 최대92.5µs,copyhelper wall7.9ms(하드웨어GPU시간 아님),threadCPU는15.625ms 양자화·다수0이므로 scheduler 원인 추론을 하지 않는다.
- 1851 samecopy19731검사/all703AItrue·narroweligible.1852 원래7배열재생차0/max47.8166ms/>80=0/all990aidOFF/703unique packets. SourceHB159max2.2008ms/nestedio.save0. 별도camera/status.io.save198.5593ms는teacherterminal452.4215 뒤454.779–454.978에 발생했다. 마지막script.update326.0178ms는captureclose456.0371을 가로지르며capture_indexFalse다. 이 종료행을 보존하고 전체326ms를 활성teacher/heartbeat 지연으로 부르지 않는다. 실제 healthy-failed-data 정상반납 분기는 아직 실행하지 않았다.
- **1853 새 읽기 전용 guard는 원본0009e5 terminal0/819검사**, summary329eca79b2a4d5fdc16c0de810d5e790c1fe1f048827aa6a4b2898f54a469f29.1854 독립929검사(resultc5cca98ee190f95da62499ee46b48e744993b4e463f9187e0c8abbdd752932e9)는 이전1824에서 generation만 바뀐 정확한 소스·14rawfiles/metadata·819checks·788옛페달정수요청·PID/creation/controls/DLL/GET축범위/SDK/native/CSP fields를 검증했다. 현재generation.0013066/PID24200/SDK693369→693373/frozen native163913AItrue/graphics4151453/.34021300077438354/lap0/FREE·정지·전제동·no lease. 지리적 시작 조건은 맞지만 새nonce/source review·TRAIN/fit/student 허용은 아직 없다. 현재 응답·제동동역학·hold·넓은 조향 재측정은 하지 않았다.
- 1839 offline trainer는16CPU mock/Ruff,1841 독립36source+20CPU mock을 통과했다. 전체이력1회/TBPTT8/이전행helddt/emptywindow 상태보존/최종짧은window/Adam2기존DNg/frozen state/실제rank2certificate 전optimizer거부를 확인했다. actual graph·새자료·GPU·fit은 아직0. 1842 신규TRAIN V4코드는28CPU/Ruff·85pins·77mandatory closure·10childsource fields·정확한6parent-edit inverse·실제decisions schema fixture를 통과했다. 원래1823 source는 보존하며 새V4 loader만 실제키 오류를 고쳤다.1865 독립147소스검사/1866 역할준비31검사/1868 실제seal packet134검사를통과해1855선언을수집전에고정했다.후속1856수집과clock실패는아래에기록한다.
- **1855 역할을 수집 전에 seal한1856 첫 신규TRAIN 후보는 정상transport 종료 뒤 전체이력 fit gate에서 실패했다.** nonce58ba2076710a46448d057d1f22207a5d/선언3ea279eb9ad842fb3003d480e8ea84b829ad0502be9f7e10b030fcf20b437341,manifestabce74d998518925b1005ff368f2802643401a4fbeaa4523beebba460c876fe0. 원본95986/chunk4636f1 terminal0/parentdf09ab55e8715233253ec2b370a5f9b7be10b9b3e308276eb2489896188d2bd5.697decisions/1000camera/0wait/max101.07492km/h·변위446.23123m/정상handoff·closed·threads[]·leasefalse. 실제pre-capture TRAIN 역할은 보존하며, 이를진단으로바꾸거나좋은자료로승격하지않는다.
- **1859 원래7배열 차0·697unique·1000aidOFF이나621→622의 원래간격82.97899999888614ms가80ms기준을1회 초과했다.** 원본audit session74131/chunkbcca4a terminal0/result9581510b9b2d8f9982180850231dcdeb3bd0c1dcd4bd0c4519d82cd2adcf6204.따라서후보전체는fit불가/actualfit0. 원래actor/source clock을줄이거나간격을쪼개거나상태를삭제하지않는다.1858 권한raw감사와1857 원본heartbeat·profile통과는이clock gate실패를대신하지않는다.
- 1860은기존fit용계획을미실행으로남기고별도DIAGNOSTIC실행b14608 terminal0으로frozen loader를한번불렀다. 결과60a352668b672003a0a89dc18a33162a8c1d94987c502e4f66f9a1a6b4a08e02.모든697상태/body21·원래mask·696이전행heldindices·원래시간/명령/카메라참조/페달이보존됐다. source_scheduleFalse여서모든1394pedal-loss bits가False다. 원래fullbrake1=189/qualified0 보존,phase116/232/186/163. 몸validcounts는yaw4/speed16·바퀴17:21각697이며2/3/5:8의원래누락mask도복원하지않았다. 모델/GPU/픽셀/NPZ열기0·optimizer0이다.
- 1857별도targeted-gap-forensics43검사(result0febdfa0b25fb0ea2dbc90ff972169084c70ae95dbe9a5e326ba4a8d34035512)는native172855→172882/receipt82.9891ms와같은cachedcamera766을확인했다.767render HOSTlower가622결정뒤.7811ms,source766→767간격101.3985ms. Sourceupdate65.1008ms중pollExpertProbe64.8801ms/sessionState22µs가겹쳤고그update에는io.save/HB/capture가없었다. HOSTpersist764167.1444ms는전체배우간격을덮었지만capacity/copy2.869·2.3911ms는그간격을설명하지않는다. HB125는622뒤commit돼원래held60/source1854.4207897을보존한다. Actorpack/log/flush/sleep/status와sourcecancel/per-file/persistCPU substage는미계측이므로IO/GIL/CPU/scheduler중하나로원인을단정하지않는다.
- 1869 새읽기전용native관측72abc7 terminal0/resultdd5b38b0a0713c2c31a36b124b8416d141d8d01f23521fe7dc44d63eea759ee5:현재spline `.4202456474304199`/lap0/graphics4691429/frozenphysics174638/FREE·비전면·paused·AIoff·전제동·no lease.추가제어/GPU/모델은0이다.
- 다음은별도미봉인source-only actor cadence/trace prototype이다. 기존40ms source/default를보존하고새nominal30ms budget을status read앞부터측정하며, poll/status/candidate/native/pack/encode/write/flush/sleep을고정RAM에계측한다. camera40ms/wait40ms/cached110ms/camera150ms/wholehistory80ms/HB2초는유지한다. 실제원인해결·새자격/수집은아직주장하지않는다. 별도3/30초자격후새TRAIN지리B start[.49,.54]/whole[.49,.64]는보호DEV/EVAL을그대로둔prospective DRAFT이며아직seal/수집0이다.1856실패와native30예산소진을지우지않는다.


- 실제 recorder 결과의 관측 수 키는 `decisions`다. 원래 미실행1823 loader의 `result['observations']` 참조 오류를 발견했다. 기존 자료/source를 고치거나 진단을TRAIN으로 승격하지 않고1842 신규V4 loader에서 명시적으로 수정·실제schema검사를 추가한다. 원래1845 등록에 묶인 준비 코드는 정확한 기존SHA로 복원했고, 향후pace30 수정 준비 코드는 별도v2 파일이다. 실제 수집·원본 데이터 변경0, 등록87inputs drift0 확인을 남겼다.

- 새 학습의 비실행 초안 configs/flight-pedal-first-train-fit-draft-20261030.json은 전체이력1회/TBPTT8/Adam1e-3/DNg2개만/최대150update를 선언한다. 실제 새TRAIN/source/V4-loader/admission seal 전 model0, 같은 이력의 rank2 미분 자격과 원본 종료 증거를 추가 seal하기 전 optimizer0으로 두 단계를 구분한다. 현재 모든 data/loader/admission binding은 null이다.

아래는 이전 체크포인트다. 당시 현재 설치·위치·다음 작업은 위 상태를 대신하지 않는다.

## 이전 실제 자격 t1811·t1815: bounded32 영상·교사·heartbeat 결합

갱신: 2026-10-30T07:55:20.556490+09:00. 전체 목표는 active·미완료이며 이번 운전fit/학생배포/학생 유효랩은0이다.

t1810은 검토된807 source를 같은engine24200에 재배포했고 새generation `733364ULL:0.0013302`를 확인했다. t1811의3초와 t1815의30초는 새로 등록한 명세로 각각 한 번 실행해 정상 종료됐다. 앞선 t1799의 queue-full 실패와 t1806 예외 복구는 그대로 보존한다. 새 프로필은 원래 default8/writer를 유지한 별도32pending/32MiB immutable-payload 상한이며, 같은 단일 저장 작업자·복사→ACK→enqueue 순서·원래 센서 시각을 보존한다. 45 CPU 검사와 t1821의41개 독립 source 검토를 통과했다. 초기 verifier가 허위로 큰 대기 수/바이트를 받을 수 있던 문제와 수정본의 정확한 FIFO suffix 검사는 별도 판본으로 남겼다.

| 실제 새 프로필 | 원래 관측 | camera groups | 최대 원래 간격 | 기존7배열 재생 차이 | 원래 heartbeat 발행 |
|---|---:|---:|---:|---:|---:|
| t1811,3초60 |71|254|44.8398ms|모두0|47|
| t1815,30초60→100→0→60 |702|1005|46.7025ms|모두0|182|

두 실행 모두 camera wait miss0/80ms초과0이다. t1813 authority2,031검사와 t1817의19,699검사는 같은420byte body/AIflag·packet·receipt를 확인했고71/702개 모두 좁은 nativeAI 조건이 참이었다. t1814/t1818은1,790/7,047개 원본 archive 파일과 모든 선택 관측을 검증했다. 이는 최적 교사·동기 물리 source 시각·TRAIN 허가의 증거가 아니다.

t1812의10,258검사와 t1816의39,402검사는 원래 MMF/control bytes·calibration·lease·arm·고정된 원래 SYNC bytes/ACK·immutable source proofs와 정상 해제를 검증했다. 두 소유권의 원래4개 maps 부재도 독립 확인했다. 보존한 heartbeat raw는 각각199blobs/666,495byte와739blobs/2,539,494byte이며 누락·overflow flags는0이다. 원본을 재직렬화하거나 뒤의 현재 status로 대체하지 않았다.

별도 프로필 독립 감사3,319/13,789검사는254/1005개 produced=accepted=persisted와 실제 FIFO payload 합을 확인했다. 관측 최대는 각각2개 작업/1,537,203byte 및4개 작업/3,074,424byte, 가장 긴 저장 작업은21.3983/134.7968ms였다. 두 실행 모두 peak가 old8 이하였으므로889ms 정체를 실제 흡수한 대조나 근본 원인 해결로 부르지 않는다. heartbeat 최대1.6647/4.4679ms이고 그 내부 io.save는0회였다. 다른 source io.save 최대20.1079/22.1001ms, update 최대35.1694/29.7736ms를 구분한다.

t1815의 네 위상 관측 수는117/234/186/165다. native 최고값은첫60의60.7768/100위상101.0238/재출발60.8802kmh이고 stop 최소는.00074949kmh다. CSP 전체 요약 최고101.0339127kmh와는 다른 source/clock 관측이다. 위치 최대 변위471.7646m는 path length나 lap이 아니다. stop0에는 fullbrake185행(42행은speed>20kmh)이 있으며 coactivation4개를 보존했다. 행350은 held100인데 HOST receipt bound가 stop 호출 뒤, camera는 그 앞이다. 마지막699–701의 HOST receipt bounds는 disable 뒤이고700/701 camera도 그 뒤다. 원래 모든702상태를 유지하고, 이런 경계는 미래 loss-only 불확실성으로 처리한다.

현재 main+설치 source는 `807e708e098654cfaf8cf58ebd91efc15f850d3da31795b02d28bc063eeb024b`, engine24200/gen733364ULL:0.0013302/epoch0/no-lease SYNC/pausedAIoff/fullbrake/FREE/background/cameraoff/expertinactive다. nativeAI=true frozen baseline을 학생 소유권 허가로 쓰지 않는다. GPU/neural/control 실제 작업은 모두 종료됐다. 아래의 b6/current 또는 oldgen 문장은 각 과거 체크포인트 당시 상태다.

v2 페달 손실54CPU와 t1819의 독립 검토, t1820 실제 원래 chemical graph의 고정4조건/50ms CPU rank2 기울기 및 t1822의116검사는 별도 진전이다. 학습 업데이트는0이며 정확한 CPU quiet/private offsets와 fixed anchor를 유지했다. 정지0/0의 fullbrake1 손실·두 DNg 계수의 한계는 보존한다. 다음은 실제 원본 역할이 사전 등록된 새 TRAIN-candidate 선언/전체 이력 loader(t1823), source·plant/페달 calibration applicability(t1824), 독립 검토(t1825)다. 과거 진단·실패·DEV/EVAL을 TRAIN으로 승격하지 않는다. 시각 선행/조향/가변속·변속/하중·타이어·셋업 인과 적응/유효 완주/빠른 라인·랩타임의 전체 목표가 남아 있다.

## t1795–1806 당시 영상 결합과 저장 큐 실패

t1795의 실제 3초 영상+same-copy native+heartbeat worker 결합은 정상 종료됐다. 71개 관측/236개 camera groups에서 기존7배열 재생 차이는 모두0, 최대 원래 decision 간격45.4616ms/80ms초과0이었다. 좁은 authority 검사는2,029개를 통과했고71개 모두 nativeAI=true였다. 이는 동기·최적 교사 표적이나 TRAIN 허가와 다르다.

t1796 독립9,553검사는183개 원본 raw blobs(612,154byte),709/709 data read pairs와711 Lua-control pairs,43개 원래 상태 발행, 실제 calibration/lease/arm, 원래 frozen SYNC 및 raw ACK를 검증했다. SYNC SHA `4b87e441e9aaf949968f2225f0ee2ea748723440368096e2f152edb5ed946281`/CRC3422087072/1225byte를 직접 재계산했고 같은 writer 종료·lease 해제·옛4개 maps 부재를 확인했다. sequence34의 rename2회도 같은 원래 bytes/source/proof였으며 당시 오류코드는 기록되지 않았다. 이 새 원본은 t1793의 빠진 bytes를 소급 보충하는 자료가 아니다.

t1799의 별도30초 과제는 **실패**했다. 60→100 이후302관측에서 이미지 persistence queue가 가득 차 중단됐고 stop0/restart60에 도달하지 않았다. accepted432개 묶음은 모두 저장됐으며, 다음 index432의 메타데이터는 보존됐지만 RGB copy/ACK 전 거부됐다. t1805 독립5,211검사는3,024개 accepted 원본 파일과 거부 프레임·당시8개 미완료 작업을 확인했다. persistence 작업424의 실제 기간889.075ms(평균이 아닌 단일 관측) 동안425–431이 대기했다. 이 기록만으로 disk/antivirus/GIL/heartbeat 중 어느 것이 지연의 원인인지 정하지 않는다.

t1800의17,275검사는 실패한 수집의390개 blobs/1,339,833byte/96개 상태 발행/각1591 read pairs/97 host sends를 원본으로 검증했다. 원래 인덱스는 partial이며 SYNC witness·정상handoff·lease release가 없다. 작업자들은 종료됐고 표식은 의도대로 남았다. t1801의8,497검사는302개 같은-copy 권한 원본을 통과했으나 source_status=failed/fit=false를 유지했다. t1802의 정상 전체 raw 감사는433개 메타데이터와432개 accepted groups 차이를 거부했다. 별도 t1802/partial-replay-v1은302개 원래 관측/301간격/7배열 차이0/최대47.5072ms/>80ms0을 확인했다. 모든433aidOFF 메타데이터와3,040archive 파일 SHA도 검증했으며, 정상 전체 감사와 원래 수집의 실패 판정은 바꾸지 않는다.

| source timing 원본 | heartbeat 횟수/최대 | heartbeat 안 io.save | 다른 io.save 최대 | script.update 최대 |
|---|---:|---:|---:|---:|
| t1795 정상3초 |38 /1.9495ms|0|42.6760ms|43.2227ms|
| t1799 중단30초 |70 /3.3106ms|0|402.9168ms|403.1727ms|

큰402.9168ms camera/status 저장은 queue-full 수신 시각보다232ms 이상 뒤에 시작했으므로 앞선 포화의 원인으로 설명할 수 없다. 서로 다른 주행에서 얻은 최대값을 인과 대조로 부르지 않으며 모든 stall 해결을 주장하지 않는다. 새 evidence runner는 기존MMF 읽기/보낸 buffer를 그대로 캡처하고 두 번의 추가 local metadata 읽기를 명시했다.93 CPU 검사와 독립 source 검토를 통과했으며 측정 overhead를0으로 가정하지 않았다.

t1804는 같은 실패 owner의 실제 raw maps를 보존했다. host30624의 원래 실행은 root session12601/chunk8db062에서 terminal1이었고 owned_threads=[]였다. 이후 LuaACK105는 `Host commit/source freshness timeout`을 기록했다. 이는 host 종료 뒤의 source fence 결과이며 수집 중 원래 실패가 CRC나 clock echo였다는 증거가 아니다.

t1806은 별도 검토·read-only preflight와 그 원본 보존 후 정확한 실패 소유권만 정리했다. 같은PID24200의 진행하는 paused graphics/frozen native90032/gas0/brake1/FREE/background, exact lease/fileID/failedACK/설치source를 재확인하고 검증된 b6 source를 복원했다. **정상 handoff가 아닌 예외 복구**이며 UI/차량API/vJoy/engine restart0이다. 현재 main+설치 Lua는 `b6d87d2708a13b37e858bf6cfe87f362540bf645514223e18e2c870fbd66e1f7`, generation `733364ULL:0.0011438`, no-lease SYNC/pausedAIoff/fullbrake다.807 candidate는 역사적 검증 소스로 보존돼 있다. 옛 nativeAI=true frozen flag는 학생 소유권 허가가 아니다.


### v2 페달 손실과 원래 그래프의 실제 CPU 기울기

새 `flight_pedal_learning_v2.py`는 고정된 원래 v2 decoder를 직접 호출하며 등록 CPU reference SHA97fc57...30175와 private readout SHA1647d1...dd032를 고정한다. canonical은 원래float32, hard codec은 원래 numpy 경로와 같은float64로 계산한다. 기준/anchor/gain/mix를 학습하거나 STE·axis residual을 추가하지 않았다.54 CPU 검사와 t1819 독립52검사+새2사례를 통과했다. offset installer는 CPU-only이며 CPU quiet 해시를 실제 GPU 상태 해시로 대신 쓰지 않는다.

t1820은 사전 고정한 목표/몸 속도(kmh) `(0,100),(60,100),(100,60),(60,20)`에서 각100×10ms quiet 이후 연속50ms를 원래 `PhysicalManeuverCircuit.drive/advance`와 chemical graph로 계산했다. 원래 실행63747은terminal0/8.5906894초였다. 네 quiet state는 모두 t1734의 원래 CPU 해시와 정확히 일치했고3neutral buffers만 설치했다. 보호된 parameter/buffer 지문과31 source pins는 유지됐다. 모든 조건에서 `[gas1,brake0]`와 `[gas0,brake1]`의 두 진단 basis를 사용했으며, 교사·최적 표적을 새로 만들지 않았다.

실제 original-graph loss-gradient 8×2의 singular values는`.4094272335/.03115459221`/threshold`3.90460e-7`, pedal-output Jacobian은`.4718855992/.03583336625`/threshold`4.50025e-7`로 모두rank2였다. `(0,100)`의 feedforward gradient0과 모든 clamp0도 그대로 포함했다. 각 canonical `[throttle,brake]`는`[0,.15000807],[0,.13864411],[.14249058,0],[.14249058,0]`였다. 이는 고정 합성 입력에서 원래 신경 경로의 국소 학습 가능성 검사이며 corpus fit/차량 식별/운전 성공/GPU 기울기 자격은 아니다. optimizer0/GPU초기화0/신규신경trajectory·PT복사0이다.

두 계수만으로 정지 유지 정책이 해결되지는 않는다. 목표속도0/몸속도0의 DNg current는 두 계수에 대한 기울기가0이고, 정확한 quiet MN에서 v2brake0이므로 active StopCounter의 brake1 표적에 손실.5가 남는다. fullbrake1 셀과 이 손실을 유지한다. 이 명제는 이력이 다른 모든 정지 상태를 quiet로 간주하지 않는다. 다음 신규 TRAIN은 명시적으로 사전 등록한 역할·외부 과제·전체 상태 이력과 loss-only masks를 갖춰야 하며, 과거 진단 자료를 승격하지 않는다.

다음은 old default8을 보존하는 명시적32pending/32MiB persistence profile을 구현·검증하고 새로 등록한 수집으로 자격을 얻는 것이다. 실패1799는 재분류하거나 자동 재시도하지 않는다. 새 TRAIN은 진단 자료와 별도로 사전 등록하고 전체 route의 DEV/EVAL 보호·실제 API start/finish·원래 시간 이력·coactivation loss-only mask를 검증해야 한다. DNg 두 계수의 pedal-only 학습을 준비하되 기존 loss가 v1 decoder를 쓰는 문제를 먼저 별도 differentiable v2 loss로 해결한다. 조향/yaw reference와 calibration을 꾸며 넣지 않고 fullbrake1 표적을 유지한다.

운전fit0/학생배포0/유효랩0/목표active·미완료다. 시각 선행 예측·몸 기울기/하중/타이어/셋업 인과 적응·가변속/변속·유효 완주·빠른 라인/랩타임의 전체 목표를 유지한다. 아래는 각 당시 상태를 기록한 이전 체크포인트다.

## 실제 같은-copy 기록

`--same-copy-native-authority`를 선택하면 같은420byte repeated copy에서 원래 body308byte와 AI flag@416을 나눈다. body와 authority의 packet/수신 시각/부분 원본 bytes가 일치한다. 기본 recorder/actor 배열·clock·caps·driver/cleanup 경로는 유지하며,104CPU 검사와 Ruff를 통과했다. 당시 실제 실행 recorder는 `baf4cf8bd4cb4ff7284e94cc424a3094536dfb07027cbcddfd149b7df4a8ed04`이고 각 수집의 `executed-program.py`와 source-proof를 보존했다.

| 실제 진단 | body/네 시점 관측 | native AI true | 최대 관측 간격 | 기존7배열 재생 오차 | aid OFF 원천 camera groups |
|---|---:|---:|---:|---:|---:|
| t1773,3초60과제 |75|75|41.2335ms|모두0|253|
| t1778,30초60→100→0→60 |737|737|42.7387ms|모두0|1002|

두 실행은 camera wait miss0/80ms초과 간격0/초기 frozen baseline 이후 packet 증가를 확인했다. t1773의 CSP 최고속도58.720997kmh, 최대 위치 변위40.3277m, t1778은100.832588kmh/247.001854m였다. 위치 변위는 path length나 랩타임이 아니며 actor 입력이 아니다. t1781 독립19,027검사가 네 위상의 실제 속도와 정지·재출발을 확인했다. source별 native/CSP 최고값과 각자의 관측 시각은 구분한다.

원본은 `C:/Users/jeff4/AppData/Local/flybrain/runtime-runs/`의 t1773-same-copy-camera3 및 t1778-same-copy-pace30이다. t1774의2,141검사와 t1779의20,677검사는 같은420/308 raw·SHA·ABI·packet·receipt·기존 body age·좁은 eligibility와 원래 관측시각을 검증했다. t1775/t1780은 네 시점 pixel·contrast·valid·age·organs/organ_valid의 기존7배열을 원본으로 재생했다.

## 좁은 flag 증거와 학습 표적의 구분

`teacher_authority_eligible`의 의미는 같은 body packet의 native AI flag와 loop-start CSP 상태·기존 age 조건뿐이다. camera/물리 source 시각 동기, engine atomicity, 배타적 교사 제어, 최적 행동, 학습 허가를 뜻하지 않는다. 모든 원본 관측과 간격·mask를 보존하며, 진단 자료를 TRAIN으로 승격하지 않는다.

독립 t1776의2,063검사는3초의75개 관측과 baseline flag0→활성 관측 true를 확인했다. 다만 event0의 camera는 enable_ai 이전에 render됐고, event74의 HOST body receipt bound는 disable_ai 이후였다. 후자는 gas0/brake1인데 native flag는 여전히true이며, camera는 disable 이전이었다. 이 camera/body 경계가 바로 flag 하나만으로 모든 표적을 활성 교사의 동기 표적으로 부를 수 없는 이유다. HOST receipt bound는 정확한 물리 sample time이 아니다.

원래75상태/74간격을 지우거나 다시 시각을 붙이지 않는다. 새 TRAIN에서 과제·teacher-active 경계와 원래 camera/receipt clock bounds를 사전에 정한 loss-only mask로 처리해야 한다. positive full-brake 표적을 모두 제외해 제동 학습을 해결한 것으로 세지 않는다. t1781은16,801개 기본 검사와2,226개 phase/body 검사(총19,027개)를 봉인했다. 관측735·736은 HOST receipt bound가 disable 이후이고 gas0/brake1인데 native flag는true였다.736의 camera도 disable 이후였다. 원래737상태/736간격을 그대로 유지한다.

연속 위상 관측수는123/245/196/173이다. 첫60은 median55.43534/max60.93679kmh,100위상의 native peak100.83407은 CSP 요약100.83259와 별도 관측이다. stop0의196관측은 brake1이며46개는 speed>20kmh다. 최종60위상의 마지막2개 cleanup brake1과 별개다. 실제 gas .8/.8/.8/.192 + brake1의 coactivation4개를 보존했다. 작은±.1 조향 보정 범위를 벗어난145관측에도 큰 입력 calibration을 외삽하지 않는다. body21은737개를 RAM에서 수동/encoder 재계산했고 실제 저장 actor organs132와 구분한다. gear2→Neutral1→3→Neutral1→2도 보존했다.

stop0의 외부 과제 목표속도는0이고 AI top-speed cap100은 유지된다. 선언된8초 stop 요청과0 해제 간격은7.9961674초였으며, engine 실제 counter를 읽었다고 하지 않는다. cleanup에 StopCounter8을 다시 설정하는 호출은 없었다. 첫60은 이번에는 실제 달성했으므로 과거 이력의 첫 위상 저속 실패를 반복 일반화하지 않는다.

## t1773/t1778 당시 heartbeat 준비 상태

당시 배포 Lua는 `b6d87d2708a13b37e858bf6cfe87f362540bf645514223e18e2c870fbd66e1f7`이며 이번 두 수집에 새 heartbeat worker는 없었다. 과거 직접 측정한 동기 heartbeat io.save 최대73.4062ms를 줄이기 위해 별도 MMF/host worker/ACK/retire 경로를 구현 중이다.

실제 Windows Python3.12의 time.time_ns/monotonic 해상도15.625ms와 QPC100ns를 확인했다. 기존 후보의1ms wall/QPC guard와 맞지 않는 반례, 저장 뒤 wall-clock 역행 시 mtime-only reader가 과거 상태를 새것처럼 받는 반례를 t1772에 저장했다. GetSystemTimePreciseAsFileTime+명시적 QPC calibration, immutable source JSON의 hash와 source-QPC에 묶인 proof, 원래 source-age, writer join/fence와 synchronous return 후 lease 해제를 구현·검토 중이다. 정밀 mtime이 기존 coarse time.time보다 미래가 되는 문제도 실제 OS qualifier에서 검사한다.

새 `session_heartbeat_provenance.py`/`session_heartbeat_runtime.py`와 일부 reader guard는 CPU 검사/검토 단계다. 수정된 현재 recorder(C0...)는 수집 후의 QPC reader migration 버전이며 Baf 실수집의 자격을 그대로 물려받지 않는다. 기본 no-lease 동작과 전체 reader inventory를 검사하고, 이후 실제 Windows IPC/파일시각과 생산 Lua arm/cleanup/retire를 각각 자격 검증해야 한다. 배포·실제 arm은 아직0이다.

## 실제 Windows IPC 격리 자격

t1783은 실제 Windows의 전용 nonce/별도 namespace에서 유한 Python producer4개 원본 snapshot을 모두 관측했다. 실제 공유 메모리의 odd slot16개를 거부했고 source JSON/SHA/QPC 시각이 일치했다. 두 번째 파일 lease와 이미 존재하는 map 이름을 거부했으며, child terminal0 및 모든 views/handles 해제 뒤 같은 이름의 OpenFileMapping이 not-found임을 확인했다.

실제 C: NTFS의 os.utime/stat 증분은100ns였다. 정밀 벽시계 기준에15,625,000ns를 추가로 과거 방향 padding하여 기존 coarse time.time의 `0<=age<=2`도 네 번 모두 통과했다. source QPC age9.27–10.44ms, precise wall age27.29–28.64ms, coarse legacy age26.60–27.70ms를 관측했다. 미보정 probe도 이번 네 번은 양수였으나 이것이 시계 해상도 위험을 부정하지는 않는다. 실제 OS 시계를 바꾸지 않았다.

이는 host-QPC producer fixture이며 실제 CSP clock/Lua FFI/production arm·ACK·retire를 자격 검증한 것이 아니다. t1784 독립 artifact 감사는 진행 중이다. production 연결에서는 forced task heartbeat가 중복 period gate에 막히지 않도록 수정하고, 실패 뒤 늦은 ACK 및 sync-return bytes 재기록 경쟁도 검토·회귀 검사한다. 기본 sink와 일부 reader 경로는 바뀌는 중이므로 새 코드에 대한 실제 자격을 따로 얻어야 한다.

## t1782 당시 관측과 남은 일

읽기 전용 t1782는 PID24200/generation733364ULL:0.0027459, paused/fullbrake/gas0/vJoyFREE/background, speed.00778731kmh/spline.94559014를 확인했다. nativeAI=true는 마지막 교사 packet의 frozen baseline이며 학생 소유권의 신선한 증거가 아니다. 학생 제어 전에는 재개 뒤 진행 packet에서 별도 검증한다.

이번 운전 fit/학생 배포/유효 랩은0이다. 과거 시각망6개 각186회·총1,116업데이트와 별개다. 다음은 heartbeat 실제 OS/producer/consumer 자격, 교사/과제/몸21/clock/단위/loss-mask를 갖춘 새 TRAIN, 등록된 CPU private quiet offsets와 고정 v2 anchor를 일치시킨 회로 내부 물리단위 학습이다. 현재 합성·GPU·저속 v2 자격이 새 훈련 graph나 시각 자율주행을 자동 인증하지 않는다.

이전 t1767 소형 보존은 t1768 독립3,049검사에서 payload0을 확인했다. t1763 실패/원본 JSON 복사본4개만 제거한 교정은 이전 보고서에 남아 있으며, 원본 데이터는 유지했다. 이번 raw 데이터도 새 소형 보존에 중복 복사하지 않는다. UI0/HTML0/다른 개인 프로젝트 변경0/외부17212·2328 종료0을 유지한다.


## 첫 실제 handoff 실패와 명시적 복구

소스209CPU 검사와 독립421 test-instance 검토 후 c159 후보를 t1787에 배포했다. 실제 정차 t1788은2개 상태 발행에는 성공했지만 SYNC 반환을 완료하지 못했다. t1789의 같은 소유권 raw control/header/CRC와 독립83검사는 Lua의 `Host ACK must retain original source clock` 실패를 확인했다. 내부 raw double은 당시 기록되지 않아 수치 차이와 정확히 어느 host sequence를 처리하던 중 실패했는지는 소급 측정하지 않는다. CPU 검토가 실제 JSON clock 왕복의 이 문제를 놓친 사실을 보존한다.

같은 소유권을 재시작하거나 timeout을 성공으로 바꾸지 않았다. t1790은 원래 host의 terminal1와 threads[]/현재 PID handle 부재, 정확한 lease·failed ACK·소스 SHA, 두 번의 진행하는 paused graphics와 fullbrake/FREE/background를 검증했다. 그 실패 소유권만 정리하고 검증된 b6 소스를 복원했다. 새로운 generation과 no-lease SYNC, 옛4개 map 부재를 실제 확인했다. 이는 실패 후 예외 복구이며 정상 handoff 성공이 아니다. 차량 API/vJoy/engine restart는0이었다.

## wire-v2 실제 정차 시험

별도807e708e 후보는 실제 보낸 JSON을 다시 파싱해 source_clock을 nested snapshot과 맞추고 **wire 값**을 host echo와 정확히 비교한다. 내부 raw clock은2초 source-age·단조성 검사에 따로 유지한다. 임의 tolerance를 추가하거나 관측 시각을 다시 붙이지 않는다. raw↔wire 차이는 기존 clock rounding margin 안인지 검사하고17자리 text로 남긴다.84 CPU 회귀와 추가2개 독립 clock-boundary 검사를 통과했다.

t1792 배포 후 t1793 실제 정차 시험은 정상 종료됐다.2개 발행/final sequence2/SYNC 반환 ACK4/같은 worker join/정확한 소유권 unlink1회/threads[]/lease false를 확인했다. 실제 raw text `31.123779900000002`, wire text `31.123779899999999`, delta `-3.5527136788005009e-15`가 있는 상태에서도 strict wire echo가 통과했다. 위치나 액추에이터를 조작하는 시험은 아니었다.

t1794 독립64검사는 남은 proof/mtime·소스·실제 map4개 부재와 위 기록을 확인했다. 다만 당시 runner는 원래 snapshot JSON·정확한 SYNC bytes·원래 calibration/lease bytes를 모두 저장하지 않았다. 실제 CRC/hash 비교는 실행 코드와 결과에 근거하며, 원래 SYNC bytes의 독립 CRC 재생은 불가능하다. 뒤의 현재 status를 과거 원본으로 대체하지 않는다.

다음 camera3/pace30용 별도 evidence runner는 기존 data/control 읽기에서 raw bytes를 유한 RAM ledger로 보존하고, 원래 calibration/lease/arm 및 frozen SYNC bytes/ACK를 release 이전에 캡처한다. 저장은 worker join/handoff 뒤에 하며 기존 프로토콜807과 source/runtime·actor 배열·제어 시간/caps를 바꾸지 않는다. 현재 이 새 runner는 구현·검토 중이며 실제3/30 worker 결합 수집은 아직 미실행이다.

현재 engine24200/generation733364ULL:0.0014195, main/설치 source807e708e098654cfaf8cf58ebd91efc15f850d3da31795b02d28bc063eeb024b다. source status는 다시 no-lease SYNC/paused/AIoff/fullbrake/cameraoff/expertinactive이며 actor 및 학생 자율주행 자격은 여전히 별도다. 과거 b6를 현재 설치 소스로 해석하지 않는다.
