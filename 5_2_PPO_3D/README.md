# 5_2_PPO_3D: PPO 기반 3D 연속 제어 UAV 경로 최적화

## 개요

`5_2_PPO_3D`는 `4_2_DDPG_3D`와 같은 확장 3D 환경을 사용하되, 정책 업데이트는 PPO의 clipped surrogate objective와 GAE로 수행합니다. UAV는 `100 x 100 x 10` 공간에서 3D 건물 장애물을 피하면서 3개의 waypoint를 순서대로 방문해야 합니다.

## 환경 설정

| 항목 | 값 |
| --- | --- |
| 공간 크기 | `100 x 100 x 10` |
| 시작 위치 | `(0.5, 0.5, 0.5)` |
| Waypoints | `(33.5, 33.5, 5.5)`, `(66.5, 66.5, 5.5)`, `(99.5, 99.5, 9.5)` |
| 장애물 | `3 x 3` footprint 건물 `50`개 |
| 건물 높이 | `1`부터 `grid_size_z - 1`까지 |
| 최대 이동량 | XY `2.2`, Z `1.0` |
| Waypoint 도달 반경 | `2.0` |
| Z축 이동 비용 가중치 | `1.7` |
| 에너지 예산 | waypoint 기준 경로 비용 x `3.6` |
| 최대 step | `1500` |

장애물은 `env/config.py`에서 footprint-aware 방식으로 생성됩니다. 각 건물은 `x0`, `y0`, `size_x`, `size_y`, `height` 메타데이터를 가지며, 실제 충돌 판정은 해당 footprint와 높이에 포함되는 모든 3D cell을 사용합니다.

## 상태와 행동

상태 벡터는 총 `24`차원입니다.

| 구간 | 의미 |
| --- | --- |
| `0-2` | 정규화된 UAV 위치 `(x, y, z)` |
| `3` | 정규화된 잔여 에너지 |
| `4-6` | waypoint 방문 여부 |
| `7-20` | 14방향 인접 장애물/경계 감지 |
| `21-23` | 다음 waypoint까지의 정규화된 3D 방향 |

행동 공간은 연속 3차원 `Box(-1, 1, shape=(3,))`입니다.

```text
dx = action[0] * max_step_size
dy = action[1] * max_step_size
dz = action[2] * max_step_size_z
```

## 보상

| 이벤트 | 값 |
| --- | --- |
| 매 step | `-0.5` |
| waypoint 도달 | `+100.0` |
| 전체 mission 완료 | `+300.0` |
| 벽 충돌 | `-5.0` |
| 장애물 충돌 | `-10.0` |
| Z축 방향 반전 | `-1.0` |
| 에너지 소진 | `-50.0` |
| potential shaping | `gamma * Phi(s') - Phi(s)`, `gamma=0.99` |

Z축 방향 반전 페널티는 큰 3D 공간에서 불필요한 고도 지그재그를 줄이기 위한 PPO 전용 shaping 항목입니다.

## PPO 튜닝값

환경이 `4_2_DDPG_3D` 기준의 큰 공간으로 확장되어, 기본 PPO 설정도 작은 3D 환경보다 긴 탐색과 큰 rollout을 견디도록 조정했습니다.

| 파라미터 | 값 |
| --- | --- |
| 학습 episode | `6000` |
| Learning rate | `1.5e-4` |
| Gamma | `0.99` |
| GAE lambda | `0.93` |
| Clip epsilon | `0.12` |
| Entropy coeff | `0.004` |
| Value coeff | `0.5` |
| Max grad norm | `0.35` |
| PPO epochs/update | `4` |
| Mini-batch size | `256` |
| Hidden dim | `256` |
| LR decay | enabled, floor `1e-5` |
| Target KL | `0.025` |
| Actor log_std clamp | `[-3.0, -0.8]` |

`training_log.csv`에서 1000 episode 전후 success가 90%대까지 올라간 뒤 entropy가 2.0 이상으로 커지면서 policy가 무너지는 패턴이 확인되어, 새 기본값은 탐색 보너스와 PPO update 폭을 낮추는 쪽으로 조정했습니다. 학습 중 가장 좋은 100-episode rolling success checkpoint는 `ppo_3d_best_model.pt`로 별도 저장됩니다.

## 실행

PowerShell에서 repo root 기준:

```powershell
cd .\5_2_PPO_3D
..\venv\Scripts\python.exe .\train.py
```

저장된 모델과 로그를 다시 평가/시각화할 때:

```powershell
cd .\5_2_PPO_3D
..\venv\Scripts\python.exe .\test.py
```

## 결과 파일

학습과 평가 결과는 `results/` 아래에 저장됩니다.

| 파일 | 내용 |
| --- | --- |
| `ppo_3d_model.pt` | PPO 3D 모델 |
| `ppo_3d_best_model.pt` | 100-episode rolling success 기준 best checkpoint |
| `training_log.csv` | episode별 reward, success, loss, entropy, learning rate |
| `training_curve.png` | reward 곡선 |
| `success_curve.png` | success rate 곡선 |
| `policy_loss.png` | policy loss |
| `value_loss.png` | value loss |
| `entropy.png` | entropy |
| `best_path_3d.png` | 최종/최고 경로 3D plot |
| `flight_3d.gif` | 비행 경로 애니메이션 |
