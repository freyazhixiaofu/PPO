import utils
from utils import ActorCritic
from ppoconfig import Config

import numpy as np
import torch
import gymnasium as gym
from torch.distributions import Categorical
def evaluate(model: ActorCritic, cfg: Config):
    env = gym.make(cfg.env_id)
    device = next(model.parameters()).device
    returns = []
    for _ in range(cfg.eval_episodes):
        obs, _ = env.reset(seed=cfg.seed)
        done = False
        trunc = False
        ep_ret = 0.0
        while not (done or trunc):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1)  # greedy for eval
            obs, r, done, trunc, _ = env.step(action.item())
            ep_ret += r
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns))
