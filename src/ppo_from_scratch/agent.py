# train.py
# from scratch

# Minimal, from-scratch PPO for CartPole-v1 (Gymnasium) using PyTorch.
# - Vectorized envs (n_envs) for stable batches
# - GAE advantages, clipped policy loss, value + entropy terms
# - Works on CPU/GPU (device auto-detect)
# - Saves model to ppo_cartpole.pt and prints eval reward

import math
import time
import utils
from utils import make_env, ActorCritic
from ppoconfig import Config
from ppoeval import evaluate

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from torch.distributions import Categorical

# -------------------------
# PPO training
# -------------------------
def ppo_train(cfg: Config):
    # Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    # Seeding
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Envs
    envs = SyncVectorEnv([make_env(cfg.env_id, seed=cfg.seed + i) for i in range(cfg.n_envs)])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    assert isinstance(act_space, gym.spaces.Discrete), "This example assumes a discrete action space."

    obs_dim = obs_space.shape[0]
    n_actions = act_space.n

    # Model & optimizers
    model = ActorCritic(obs_dim, n_actions).to(device)
    policy_opt = optim.Adam(model.parameters(), lr=cfg.policy_lr)  # simple single opt for both pi+v
    # (Optionally, separate params for value with different lr; here we keep it simple.)

    # Storage
    batch_size = cfg.n_envs * cfg.rollout_steps
    obs_buf      = torch.zeros(cfg.rollout_steps, cfg.n_envs, obs_dim, dtype=torch.float32, device=device)
    actions_buf  = torch.zeros(cfg.rollout_steps, cfg.n_envs, dtype=torch.long, device=device)
    logp_buf     = torch.zeros(cfg.rollout_steps, cfg.n_envs, dtype=torch.float32, device=device)
    rewards_buf  = torch.zeros(cfg.rollout_steps, cfg.n_envs, dtype=torch.float32, device=device)
    dones_buf    = torch.zeros(cfg.rollout_steps, cfg.n_envs, dtype=torch.float32, device=device)
    values_buf   = torch.zeros(cfg.rollout_steps, cfg.n_envs, dtype=torch.float32, device=device)

    obs, _ = envs.reset(seed=cfg.seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    num_updates = math.ceil(cfg.total_timesteps / batch_size)

    for update in range(1, num_updates + 1):
        # --------- Collect rollout ---------
        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            with torch.no_grad():
                action, logp, entropy, value = model.action_value(obs)
            actions_buf[t] = action
            logp_buf[t] = logp
            values_buf[t] = value

            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated).astype(np.float32)

            rewards_buf[t] = torch.tensor(reward, dtype=torch.float32, device=device)
            dones_buf[t]   = torch.tensor(done, dtype=torch.float32, device=device)

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            # bootstrap value for last state
            _, _, _, last_value = model.action_value(obs)

        # --------- Compute GAE advantages / returns ---------
        advantages = torch.zeros_like(rewards_buf, device=device)
        last_adv = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)
        for t in reversed(range(cfg.rollout_steps)):
            next_nonterminal = 1.0 - dones_buf[t]
            next_value = last_value if t == cfg.rollout_steps - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + cfg.gamma * next_value * next_nonterminal - values_buf[t]
            last_adv = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_adv
            advantages[t] = last_adv
        returns = advantages + values_buf

        # Flatten rollout [T, N] -> [T*N]
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # --------- PPO update (multiple epochs, minibatches) ---------
        inds = np.arange(batch_size)
        for epoch in range(cfg.epochs):
            rng.shuffle(inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = inds[start:end]
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_old_logp = b_logp[mb_inds]
                mb_adv = b_adv[mb_inds]
                mb_returns = b_returns[mb_inds]
                mb_old_values = b_values[mb_inds]

                new_logp, entropy, new_values = model.evaluate_actions(mb_obs, mb_actions)
                ratio = torch.exp(new_logp - mb_old_logp)  # pi(a|s)/pi_old(a|s)

                # Policy loss (clipped surrogate)
                unclipped = ratio * mb_adv
                clipped   = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                # Value loss (optionally clip value)
                value_loss = torch.mean((new_values - mb_returns) ** 2)

                # Entropy bonus
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * entropy_loss

                policy_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                policy_opt.step()

        # --------- Logging ---------
        with torch.no_grad():
            approx_kl = (b_logp - new_logp).mean().item()
            clipfrac = (torch.gt(torch.abs(ratio - 1.0), cfg.clip_eps)).float().mean().item()
            v_loss = value_loss.item()
            p_loss = policy_loss.item()
        if update % 10 == 0 or update == 1:
            print(f"Update {update}/{num_updates} | "
                  f"loss_pi {p_loss:.3f} | loss_v {v_loss:.3f} | "
                  f"kl {approx_kl:.4f} | clipfrac {clipfrac:.2f}")

    envs.close()

    # Save model
    torch.save(model.state_dict(), "ppo_cartpole.pt")
    print("Saved model to ppo_cartpole.pt")

    # Evaluate
    avg_ret = evaluate(model, cfg)
    print(f"[final] Eval return over {cfg.eval_episodes} eps: {avg_ret:.1f}")


