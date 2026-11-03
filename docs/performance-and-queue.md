# 대기 중 최적화와 작업 분할 — 2026-10-05

## 노드와 권한 확인

09:58 KST 조회에서 `hpc`는 GPU 2개가 모두 미할당, `brl0`는 GPU 8개가 미할당이었다.
반면 현재 사용할 수 있는 `partition1/hpc-stat1`은 GPU 5개 모두 할당돼 있었다.
`sbatch --test-only`로 실제 계정/partition 조합을 확인했으며 `jobs/hpc`와 `brl/brl0`는
`Invalid account or account/partition combination specified`로 거절됐다.
이는 여유 GPU의 존재와 계정의 사용 권한이 별개임을 뜻한다. 계정 association은
`hpc`, `partition1`이며 실제 hpc 노드의 partition 이름은 `jobs`다. 이 이름 불일치는
관리자가 확인할 사항이다. 사용자/관리자 계정이나 다른 작업의 GPU를 우회 사용하지 않았다.

`partition1`의 GPU array와 GPU 없는 CPU 집계 작업은 test-only 권한 검사를 통과했다.
QoS `normal`, `ug_qos`의 priority는 둘 다 0이어서 QoS 변경으로 얻는 우선순위 이득은
확인하지 못했다. `normal`은 사용자 GPU 상한 2개, `ug_qos`는 1개다.
다른 노드로의 즉시 재배치는 현재 권한으로 불가능하다.

## 최적화와 검증

V 카메라 경로의 cProfile에서 `_render_cameras`가 300 tick 중 1.933/2.268초를
사용했다. 정적인 도로 표식을 카메라마다 다시 계산하면서 road frame 7800회를 호출했다.
노면 표식의 위치/색/크기/회전을 환경 초기화 때 계산하고 렌더링마다 재사용하도록 바꿨다.
3D 투영의 고정 segment와 길이 제곱도 재사용하고, O 관측은 같은 tick의 진단 투영을
다시 계산하지 않는다. 물리 주기·영상 크기·노면·reward·센서/정답 구분은 유지했다.

로컬 Windows에서 고정 action [-0.75, 0]으로 O/V 각 300 tick을 3회 반복했다.
첫 반복은 cProfile용이며 아래 처리량은 프로파일러 없는 두 반복의 중앙값이다.
시작/renderer warmup은 측정 구간에서 제외했다.

| 단일 환경 | 이전 step/s | 이후 step/s | 배율 |
|---|---:|---:|---:|
| O 진단 | 934.42 | 1179.17 | 1.26 |
| V 카메라 | 226.51 | 625.03 | 2.76 |

두 모드 모두 모든 반복의 observation(영상/몸 센서/validity/age), qpos/qvel, reward,
종료 이유를 누적한 SHA-256이 전후 일치했다. 이 고정 300 tick의 정확한 replay 검증이며
모든 지형/시드의 일치나 GPU 학습 전체의 2.76배 가속을 주장하지 않는다.

증거는 `runs/20261005-performance-before/`, `performance-after/`,
`performance-analysis/`의 manifest·profile·trace·hash·처리량·placement 기록에 있다.
pytest **28 passed in 12.90s**, ruff와 Bash 구문 검사를 통과했다. seed/조건의 array
분할과 범위, GPU 동시 사용 1개, GPU 없는 집계, 실제 shard 학습/평가와 누락 작업의
실패 보존을 추가 검증했다.

## 작업 재구성

후속 캠페인의 4개 조건 × 2개 시드, 조건당 65,536 transition과 동일한 평가는 유지한다.
3시간짜리 한 작업을 **30분 상한 array task 8개**로 바꾼다. `--array=0-7%1`로 GPU는
동시에 하나만 요청한다. 특정 node pin은 제거하고 허용된 partition 안에서 배정받는다.
이 cluster에서는 현재 partition1에 hpc-stat1만 있으므로 pin 제거가 즉시 다른 노드의
권한을 만드는 것은 아니다.

각 task에 별도 GPU/EGL gate, runtime/log, checkpoint, initial/final 평가와 종료 기록을
남긴다. array 전체가 종료되면 GPU 없이 CPU 1개로 시드/조건별 결과를 합친다.
누락·timeout·실패 task를 성공 결과에서 숨기지 않고 명시적으로 실패로 기록한다.
watcher는 array와 집계 작업 모두 종료할 때까지 원격 tree를 보존한다.

후속 array는 기존 smoke의 출력에 의존하지 않으므로 `afterany:262880`을 제거한다.
집계는 새 array의 종료만 기다린다. 기존 smoke는 높은 대기 우선순위를 보존하면서
요청 상한을 3시간에서 20분으로 줄인다. 새 array가 접수된 뒤 기존 미실행 단일
후속 캠페인 262946을 대체하고, 그 입력/소스와 취소 증거를 기존 watcher로 회수한다.
실행 중인 무관한 작업을 중지하지 않는다.

짧은 요청은 scheduler가 예약 사이 빈 시간에 배정하는 backfill 기회를 늘릴 수 있다.
다만 현재 GPU가 모두 사용 중이므로 즉시 시작이나 대기 시간 단축을 보장하지 않는다.
참고: [Slurm scheduling](https://slurm.schedmd.com/sched_config.html),
[job array](https://slurm.schedmd.com/job_array.html),
[sbatch](https://slurm.schedmd.com/sbatch.html).

## 실행 영수증

새 array **264771**, CPU 집계 **264772**, source revision `c85d903`을 제출했다.
`runs/remote-followup-array-20261005/`의 submission·stage-manifest·replacement·
collection plan에 기록했다. ZIP SHA-256은
`dc536ac4ff2bd93240d5bbf4257e1212ad14bec2ecf930ca641b7b5eb1ef48c2`다.
watcher PID 24292가 8개 array task와 집계 작업을 모두 추적하는 heartbeat를 확인했다.
array task들은 PENDING(Priority), 집계는 PENDING(Dependency)다.

기존 smoke **262880**은 PENDING 상태와 소유 경로를 확인한 뒤 `TimeLimit=00:20:00`으로
수정했다. 기존 후속 **262946**은 같은 검증을 거쳐 hold/cancel했고,
기존 watcher가 **10:14:41 KST**에 회수·해시 검증·원격 소유 tree 삭제를 완료했다.
원래 입력/소스는 `runs/remote-followup-ablation-20261005/collection/` archive에 남았다.
각 변경 사유와 전/후 시간 상한은 scheduler-adjustment/replacement JSON에 보존했다.

10:15 KST의 scheduler 예측은 smoke **10월 6일 01:56**, 새 array 첫 배정
**10월 6일 02:16**이다. 예측은 변동할 수 있다. 작업 분할과 처리량 개선은 완료했으나
현재 GPU 확보 자체가 빨라졌거나 GPU 학습이 시작됐다고 표시하지 않는다.
