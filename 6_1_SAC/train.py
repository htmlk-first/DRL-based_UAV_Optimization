"""
SAC Training Script
- Off-policy 학습: 매 스텝마다 버퍼에 저장 → 배치 업데이트
- 주기적으로 평가(deterministic) 수행 후 best 모델 저장
"""
import os
import sys
import csv
import copy
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

sys.path.insert(0, os.path.dirname(__file__))
from common.experiment import ExperimentPaths, success_from_info
from env import UAVEnv, EnvConfig
from sac_agent import SACAgent


def train(agent, env, num_episodes, update_after=5000, update_every=1,
          print_every=500, eval_interval=500, eval_episodes=10, log_path=None,
          best_model_path=None, save_best_window=100):
    """
    Off-policy SAC 학습 루프

    Args:
        update_after: 학습 시작 전 최소 버퍼 크기 (random exploration)
        update_every: 몇 스텝마다 네트워크 업데이트
        log_path: CSV 로그 저장 경로 (None이면 저장 안 함)
    """
    reward_history = []
    success_history = []
    total_steps = 0
    best_window_success = -1.0
    best_window_reward = -float("inf")

    csv_file = None
    csv_writer = None
    if log_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        csv_file = open(log_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "reward", "success", "alpha",
                              "ep_avg_actor_loss", "ep_avg_critic_loss",
                              "buffer_size", "total_steps"])

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        actor_log_start = len(agent.actor_loss_log)
        critic_log_start = len(agent.critic_loss_log)

        while not done:
            total_steps += 1

            # 초기에는 랜덤 탐색
            if total_steps <= update_after:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(terminated))
            state = next_state
            ep_reward += reward

            # 네트워크 업데이트
            if total_steps > update_after and total_steps % update_every == 0:
                agent.update()

        reward_history.append(ep_reward)
        success = success_from_info(info)
        success_history.append(success)

        if best_model_path is not None and len(success_history) >= save_best_window:
            window_success = np.mean(success_history[-save_best_window:])
            window_reward = np.mean(reward_history[-save_best_window:])
            is_better = (
                window_success > best_window_success
                or (window_success == best_window_success
                    and window_reward > best_window_reward)
            )
            if is_better:
                os.makedirs(os.path.dirname(os.path.abspath(best_model_path)),
                            exist_ok=True)
                agent.save(best_model_path)
                best_window_success = window_success
                best_window_reward = window_reward

        if csv_writer is not None:
            ep_actor_losses = agent.actor_loss_log[actor_log_start:]
            ep_critic_losses = agent.critic_loss_log[critic_log_start:]
            ep_avg_al = float(np.mean(ep_actor_losses)) if ep_actor_losses else float("nan")
            ep_avg_cl = float(np.mean(ep_critic_losses)) if ep_critic_losses else float("nan")
            csv_writer.writerow([ep, ep_reward, success, agent.alpha,
                                  ep_avg_al, ep_avg_cl,
                                  len(agent.buffer), total_steps])

        if ep % print_every == 0:
            avg_r = np.mean(reward_history[-print_every:])
            avg_s = np.mean(success_history[-print_every:]) * 100
            alpha = agent.alpha
            print(f"[Ep {ep:5d}] AvgReward={avg_r:8.1f} | "
                  f"Success={avg_s:5.1f}% | Alpha={alpha:.4f} | "
                  f"Buffer={len(agent.buffer)} | Steps={total_steps}")

    if csv_file is not None:
        csv_file.close()

    return reward_history, success_history


def evaluate(agent, base_config, num_eval=10):
    """
    Deterministic 평가 → 최고 경로(best_path) 반환
    """
    best_reward = -float("inf")
    best_path = None
    best_env = None
    successes = 0

    for _ in range(num_eval):
        env = UAVEnv(config=base_config)
        state, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        successes += success_from_info(info)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = list(env.path)
            best_env = copy.deepcopy(env)

    # deterministic이 실패하면 stochastic fallback
    if best_path is None or successes == 0:
        for _ in range(num_eval):
            env = UAVEnv(config=base_config)
            state, _ = env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = agent.select_action(state, deterministic=False)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward

            if ep_reward > best_reward:
                best_reward = ep_reward
                best_path = list(env.path)
                best_env = copy.deepcopy(env)

    return best_path, best_env, best_reward, successes


# ──────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 하이퍼파라미터 ──
    NUM_EPISODES = 1000
    LR_ACTOR = 2e-4
    LR_CRITIC = 3e-4
    LR_ALPHA = 1e-4
    GAMMA = 0.995
    TAU = 0.005
    HIDDEN_DIM = 256
    BATCH_SIZE = 256
    BUFFER_CAPACITY = 500_000
    UPDATE_AFTER = 5000        # 랜덤 탐색 스텝 수
    UPDATE_EVERY = 1           # 매 스텝마다 업데이트
    INITIAL_ALPHA = 0.2
    GRAD_CLIP = 1.0
    PRINT_EVERY = 100
    EVAL_EPISODES = 20

    # ── 환경 ──
    config = EnvConfig(
        grid_size=100,
        obstacle_mode="fixed",
        obstacle_footprint_size=3,
        energy_budget_multiplier=3.0,
        max_step_size=2.2,
        wp_reach_radius=1.8,
    )
    env = UAVEnv(config=config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # ── 에이전트 ──
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        lr_alpha=LR_ALPHA,
        gamma=GAMMA,
        tau=TAU,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        initial_alpha=INITIAL_ALPHA,
        grad_clip=GRAD_CLIP,
    )

    # ── 저장 경로 ──
    paths = ExperimentPaths.from_file(__file__)
    save_dir = str(paths.results_dir)
    final_model_path = os.path.join(save_dir, "sac_model.pt")
    best_model_path = os.path.join(save_dir, "sac_best_model.pt")

    print("=" * 60)
    print("SAC Training Start")
    print(f"  Grid: {config.grid_size}x{config.grid_size}  |  Obstacles: {config.num_random_obstacles} "
          f"({config.obstacle_footprint_x}x{config.obstacle_footprint_y} footprint)")
    print(f"  Energy budget: {config.compute_energy_budget():.0f}  |  Max steps: {config.max_steps}")
    print(f"  State dim: {state_dim}  |  Action dim: {action_dim}")
    print(f"  Episodes: {NUM_EPISODES}  |  Hidden: {HIDDEN_DIM}")
    print(f"  LR(actor/critic/alpha): {LR_ACTOR}/{LR_CRITIC}/{LR_ALPHA}")
    print(f"  Gamma: {GAMMA}  |  Tau: {TAU}  |  Batch: {BATCH_SIZE}")
    print(f"  Buffer: {BUFFER_CAPACITY}  |  Update after: {UPDATE_AFTER}  |  Grad clip: {GRAD_CLIP}")
    print("=" * 60)

    # ── 학습 ──
    reward_history, success_history = train(
        agent, env, NUM_EPISODES,
        update_after=UPDATE_AFTER,
        update_every=UPDATE_EVERY,
        print_every=PRINT_EVERY,
        log_path=os.path.join(save_dir, "training_log.csv"),
        best_model_path=best_model_path,
        save_best_window=100,
    )

    agent.save(final_model_path)
    if os.path.exists(best_model_path):
        print(f"Loading best rolling-success checkpoint for evaluation: {best_model_path}")
        agent.load(best_model_path)

    # ── 평가 ──
    print("\n── Evaluation ──")
    best_path, best_env, best_reward, successes = evaluate(
        agent, config, num_eval=EVAL_EPISODES,
    )
    print(f"  Best reward: {best_reward:.1f}")
    print(f"  Successes:   {successes}/{EVAL_EPISODES}")
    if best_path:
        print(f"  Path length: {len(best_path)} steps")

    # ── 모델 저장 ──
    print(f"  Models saved -> {save_dir}")

    # ── 시각화 ──
    from visualize import (
        plot_reward_curve, plot_success_curve,
        plot_actor_loss, plot_critic_loss, plot_alpha_curve,
        plot_path, make_flight_gif,
    )

    plot_reward_curve(reward_history, window=50, title="SAC Training Curve",
                      save_path=os.path.join(save_dir, "training_curve.png"))

    plot_success_curve(success_history, window=50, title="SAC Success Rate",
                       save_path=os.path.join(save_dir, "success_curve.png"))

    if agent.actor_loss_log:
        plot_actor_loss(agent.actor_loss_log, window=50, title="SAC Actor Loss",
                        save_path=os.path.join(save_dir, "actor_loss.png"))

    if agent.critic_loss_log:
        plot_critic_loss(agent.critic_loss_log, window=50, title="SAC Critic Loss",
                         save_path=os.path.join(save_dir, "critic_loss.png"))

    if agent.alpha_log:
        plot_alpha_curve(agent.alpha_log, window=50, title="SAC Entropy Coefficient (α)",
                         save_path=os.path.join(save_dir, "alpha_curve.png"))

    if best_env is not None and best_path is not None:
        plot_path(best_env, best_path, title="SAC - Best Path",
                  save_path=os.path.join(save_dir, "best_path.png"))

        make_flight_gif(best_env, best_path, title="SAC Flight",
                        save_path=os.path.join(save_dir, "flight.gif"), fps=5)

    print(f"\n=== Results saved to {save_dir} ===")
