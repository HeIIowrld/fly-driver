# 저장 영상의 미사용 코너 곡률 예측

Monza의 서로 다른 저장 회차를 학습과 평가에 사용했다. 지도에서 고른 두 코너 중심의 앞뒤 90m를 학습에서 제외했고, 별도 회차에서 각 중심 앞뒤 32m를 평가했다. 차량 앞 5·10·20m에서 지도 중심선의 현 곡률을 정답으로 만들었다. 지도와 카메라의 위치·방향은 표본 등록과 정답 생성에 사용했다. 예측기에 들어간 값은 네 방향 영상과 약 0.18초 간격의 프레임 차분이다.

690개 학습 프레임과 20/14개 평가 프레임을 사용했다. 20m 곡률 RMSE는 평균 예측 .02945/.01368, 정지 영상 .02866/.01233, 영상과 차분 .02926/.01277 m⁻¹이었다. 유의 곡률의 방향 정확도는 정지 영상 50%/86%, 영상과 차분 40%/71%였다. 둘째 코너의 정지 영상 방향 정확도는 평균 예측과 같았다. 차분을 추가해도 정지 영상보다 나아지지 않았다. 두 코너는 같은 트랙에 있으며 네 방향 픽셀의 선형 판독 실험이다. 겹눈·운동 시각 회로의 곡률 인식이나 미사용 트랙 일반화는 별도 검증이 필요하다.

```powershell
python scripts/verify_ac_heldout_curvature.py --train runs/braking-corner-learning-20261105/monza-user-started-teacher-01/demonstration/camera/vision --test runs/braking-corner-learning-20261105/fresh-monza-develop-01/demonstration/camera/vision --track assets/external/ac-teacher-geometry-20261027/monza.csv --output runs/vision-heldout-curvature-reproduction
```

원본 결과는 `runs/vision-heldout-curvature-20261107/result.json`에 있다. 지도, 카메라 번들 설명자, 예측과 학습 가중치 해시를 기록했다.
