# AC 기록을 사용한 MuJoCo 조향·제동 응답 적합

기존 Monza 교사 회차의 실제 SDK 요청과 native 속도·요레이트에서 8스텝 응답 523개를 추출했다. 이전에 이미 발행된 조작만 적용하고 타이어 이탈이나 시간 간격이 큰 표본을 제외했다. 가속·제동·항력·조향 이득·언더스티어·요 응답률을 적합했다. 8~42m/s의 별도 회차 524개 창에서 같은 입력을 MuJoCo 평면 강체에 넣어 검사했다. 미적합 모델 대비 조향 창 요레이트 RMSE는 .152→.143rad/s, 제동 창 속도 RMSE는 1.272→.738m/s였다. 조향과 제동이 겹친 창 35개도 따로 기록했다. 원래 관측 창이 겹칠 수 있고 비교 회차의 트랙과 차종은 같았다.

초기와 학습한 6개 제어 계수를 같은 54개 미사용 그립·반경·회전 방향·초기 상태 조합에서 평가했다. 워커 4개에서 교차 엔트로피 최적화로 1,800회 학습했다. 완주는 48/54→54/54, 횡오차 RMS의 에피소드 평균은 .592→.168m였다. 실패는 낮은 그립 .72에서 6건→0건이었다. 학습 중 이탈도 모두 집계했다.

MuJoCo는 AC 입력에 맞춘 평면 응답을 적분했다. 실제 타이어의 슬립·서스펜션·하중 이동 모델은 포함하지 않았다. 그립 변화는 정해진 힘 한도의 합성 교란이다. 제어기에 정답 경로를 준 실험이며 학습 모델의 평가 속도 범위는 약 45.6~68.8km/h였다. AC의 접지 한계나 시각 기반 실제 주행은 별도 검증이 필요하다.

```powershell
python scripts/fit_ac_mujoco_corner_control.py --train runs/braking-corner-learning-20261105/monza-user-started-teacher-01/demonstration/teacher --test runs/braking-corner-learning-20261105/fresh-monza-develop-01/demonstration/teacher --output runs/mujoco-ac-corner-control-reproduction --workers 4
```

원본 결과는 `runs/mujoco-ac-corner-control-engine-20261107/calibration.json`과 `result.json`에 있다. 앞선 구현 확인은 `runs/mujoco-ac-corner-control-20261107`에 보존했다. 최종 비교에서는 MuJoCo 적분으로 응답을 재검증하고 그립·반경·회전 방향을 독립적으로 조합했다.
