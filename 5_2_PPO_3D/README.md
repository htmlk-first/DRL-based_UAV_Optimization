# 5_2_PPO_3D: PPO 기반 3D 연속 제어 UAV 경로 최적화

## 개요

- **PPO** 를 3차원 연속 제어 환경에 적용하여, **UAV가 3D 공간에서 건물형 장애물을 회피하며 3개의 웨이포인트를 순서대로 방문**하도록 학습합니다.
- `4_2_DDPG_3D` 와 동일한 3D 연속 환경을 사용하지만, 정책 업데이트는 **PPO의 clipped surrogate objective** 로 수행합니다.
- 3D 환경 특성상 **z축 이동 비용**, **건물형 장애물**, **14방향 인접 정보**, **z축 방향 반전 패널티**가 함께 반영됩니다.

## 환경 (Environment)

### 상태 공간 (State)

| 인덱스 | 요소 | 범위 | 설명 |
| :---: | --- | :---: | --- |
| 0 | `x / sx` | [0, 1] | 정규화된 x 좌표 |
| 1 | `y / sy` | [0, 1] | 정규화된 y 좌표 |
| 2 | `z / sz` | [0, 1] | 정규화된 z 좌표 |
| 3 | `energy / budget` | [0, 1] | 정규화된 잔여 에너지 |
| 4-6 | `wp_visited` | {0, 1} | 3개 웨이포인트 방문 여부 |
| 7-20 | `adj_14dir` | {0, 1} | 6개 면 방향 + 8개 꼭짓점 대각 방향 차단 여부 |
| 21 | `dx_wp / sx` | [-1, 1] | 다음 목표까지 x 방향 정규화 거리 |
| 22 | `dy_wp / sy` | [-1, 1] | 다음 목표까지 y 방향 정규화 거리 |
| 23 | `dz_wp / sz` | [-1, 1] | 다음 목표까지 z 방향 정규화 거리 |

> 총 **24차원** 상태 벡터를 사용합니다.

### 행동 공간 (Action)

| 요소 | 범위 | 설명 |
| --- | --- | --- |
| `action[0]` | [-1, 1] | x 방향 이동 비율 |
| `action[1]` | [-1, 1] | y 방향 이동 비율 |
| `action[2]` | [-1, 1] | z 방향 이동 비율 |

실제 이동량:

```text
dx = action[0] * 1.5
dy = action[1] * 1.5
dz = action[2] * 1.0
```

### 환경 설정

| 항목 | 설정값 |
| --- | --- |
| 공간 크기 | 30 x 30 x 5 |
| 행동 공간 | 연속 3차원 `Box(-1, 1, shape=(3,))` |
| 시작 위치 | `(0.5, 0.5, 0.5)` |
| 웨이포인트 | 3개 (`(10.5,10.5,2.5)`, `(20.5,20.5,2.5)`, `(29.5,29.5,4.5)`), **순서대로 방문** |
| 장애물 | 건물형 장애물 `40`개 |
| 최대 이동량 | XY `1.5`, Z `1.0` |
| 웨이포인트 도달 반경 | `1.0` |
| z축 비용 가중치 | `1.5` |
| 에너지 예산 | 3D 경로 비용 x `4.0` |
| 최대 스텝 | `300` |

### 보상 체계 (Reward)

| 이벤트 | 보상 | 설명 |
| --- | --- | --- |
| 이동 (매 스텝) | **-0.5** | 짧은 경로 유도 |
| 웨이포인트 도달 | +100.0 | 목표 방문 유도 |
| 전체 임무 완료 | +300.0 | 모든 목표 방문 완료 |
| 벽 충돌 | -5.0 | 공간 이탈 방지 |
| 장애물 충돌 | -10.0 | 건물 회피 유도 |
| z축 방향 반전 | -1.0 | 고도 지그재그 억제 |
| 에너지 소진 | -50.0 | 에너지 효율 학습 |
| Reward Shaping | `γ·Φ(s') - Φ(s)` | 목표 접근 시 추가 보상 |

> 포텐셜 함수는 다음 목표까지의 **3D 유클리드 거리의 음수**입니다.

## 알고리즘

### PPO (Proximal Policy Optimization)

- **On-policy rollout 학습**을 수행합니다.
- **Gaussian policy** 로 연속 행동을 샘플링합니다.
- **GAE** 로 advantage 를 추정하고, **clipping** 으로 정책 업데이트 폭을 제한합니다.
- **Entropy bonus** 와 **learning rate decay** 를 사용해 탐색과 안정성을 함께 확보합니다.

### 네트워크 구조

```text
Actor : Input (24) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Gaussian policy
Critic: Input (24) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> V(s)
```

### 학습 하이퍼파라미터

| 파라미터 | 값 |
| --- | --- |
| 학습률 | `3e-4` |
| 할인율 `γ` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_epsilon` | `0.2` |
| `entropy_coeff` | `0.01` |
| `value_coeff` | `0.25` |
| `max_grad_norm` | `0.5` |
| `k_epochs` | `10` |
| mini-batch 크기 | `64` |
| Hidden 차원 | `256` |
| 학습 에피소드 수 | `10000` |

## 실행 방법

```bash
cd 5_2_PPO_3D
python train.py
```

## 실험 결과

### 학습 곡선

![Training Curve](results/training_curve.png)

### 성공률

![Success Rate](results/success_curve.png)

### Policy 손실

![Policy Loss](results/policy_loss.png)

### Value 손실

![Value Loss](results/value_loss.png)

### 엔트로피 추이

![Entropy](results/entropy.png)

### 최적 경로

![Best Path 3D](results/best_path_3d.png)

### 비행 경로 애니메이션

![Flight GIF](results/flight_3d.gif)
