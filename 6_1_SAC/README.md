# 6_1_SAC: SAC 기반 2D 연속 제어 UAV 경로 최적화

## 개요

`6_1_SAC`는 `5_1_PPO`와 같은 확장 2D 환경을 사용하되, 학습은 Soft Actor-Critic 방식으로 수행합니다. UAV는 `100 x 100` 공간에서 `3 x 3` footprint 장애물을 피하면서 3개의 waypoint를 순서대로 방문해야 합니다.

SAC는 replay buffer 기반 off-policy 알고리즘이므로, 큰 map에 맞춰 buffer, warm-up, update 주기, discount, entropy 관련 학습률을 함께 조정했습니다.

## 환경 설정

| 항목 | 값 |
| --- | --- |
| 공간 크기 | `100 x 100` |
| 시작 위치 | `(0.5, 0.5)` |
| Waypoints | `(33.5, 33.5)`, `(66.5, 66.5)`, `(99.5, 99.5)` |
| 장애물 | `3 x 3` footprint block, ground coverage 약 `10%` |
| 기본 장애물 block 수 | `111`개 |
| 실제 obstacle cell 수 | 약 `999`개 |
| 최대 이동량 | `2.2` |
| Waypoint 도달 반경 | `1.8` |
| 에너지 예산 | waypoint 기준 경로 길이 x `3.0` |
| 최대 step | `1200` |

장애물은 `env/config.py`에서 footprint-aware 방식으로 생성됩니다. 각 장애물 block은 `x0`, `y0`, `size_x`, `size_y` 메타데이터를 가지며, 충돌 판정은 footprint에 포함되는 모든 cell을 사용합니다.

## 상태와 행동

상태 벡터는 총 `16`차원입니다.

| 구간 | 의미 |
| --- | --- |
| `0-1` | 정규화된 UAV 위치 `(x, y)` |
| `2` | 정규화된 잔여 에너지 |
| `3-5` | waypoint 방문 여부 |
| `6-13` | 8방향 인접 장애물/경계 감지 |
| `14-15` | 다음 waypoint까지의 정규화된 방향 |

행동 공간은 연속 2차원 `Box(-1, 1, shape=(2,))`입니다.

```text
dx = action[0] * max_step_size
dy = action[1] * max_step_size
```

## 보상

| 이벤트 | 값 |
| --- | --- |
| 매 step | `-0.5` |
| waypoint 도달 | `+100.0` |
| 전체 mission 완료 | `+300.0` |
| 벽 충돌 | `-5.0` |
| 장애물 충돌 | `-10.0` |
| 에너지 소진 | `-50.0` |
| potential shaping | `gamma * Phi(s') - Phi(s)`, `gamma=0.99` |

## SAC 튜닝값

| 파라미터 | 값 |
| --- | --- |
| 학습 episode | `1000` |
| Actor learning rate | `2e-4` |
| Critic learning rate | `3e-4` |
| Alpha learning rate | `1e-4` |
| Gamma | `0.995` |
| Tau | `0.005` |
| Batch size | `256` |
| Replay buffer | `500000` |
| Random exploration warm-up | `5000` steps |
| Update interval | every `1` step |
| Initial alpha | `0.2` |
| Gradient clip | `1.0` |
| Hidden dim | `256` |

큰 환경에서는 episode가 길고 replay 다양성이 중요하므로 buffer와 warm-up을 늘렸습니다. Actor와 alpha 학습률은 낮춰서 탐색 정책과 entropy 계수가 급격히 흔들리지 않게 했고, `gamma=0.995`로 먼 waypoint 보상이 더 오래 전달되도록 했습니다.

학습 중 가장 좋은 100-episode rolling success checkpoint는 `sac_best_model.pt`로 저장되고, 마지막 모델은 `sac_model.pt`로 저장됩니다.

## 실행

PowerShell에서 repo root 기준:

```powershell
cd .\6_1_SAC
..\venv\Scripts\python.exe .\train.py
```

저장된 모델과 로그를 다시 평가/시각화할 때:

```powershell
cd .\6_1_SAC
..\venv\Scripts\python.exe .\test.py
```

## 결과 파일

| 파일 | 내용 |
| --- | --- |
| `sac_model.pt` | 마지막 SAC 모델 |
| `sac_best_model.pt` | 100-episode rolling success 기준 best checkpoint |
| `training_log.csv` | episode별 reward, success, alpha, loss, buffer size |
| `training_curve.png` | reward 곡선 |
| `success_curve.png` | success rate 곡선 |
| `actor_loss.png` | actor loss |
| `critic_loss.png` | critic loss |
| `alpha_curve.png` | entropy coefficient alpha |
| `best_path.png` | 최종/최고 경로 plot |
| `flight.gif` | 비행 경로 애니메이션 |
