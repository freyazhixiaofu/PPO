import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return thunk

def layer_init(layer, std=1.0, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        # Shared torso
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.pi = layer_init(nn.Linear(64, n_actions), std=0.01)
        self.v  = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(self, x):
        z = self.net(x)
        return self.pi(z), self.v(z)  # logits, value

    def action_value(self, x):
        logits, v = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, dist.entropy(), v.squeeze(-1)

    def evaluate_actions(self, x, actions):
        logits, v = self.forward(x)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, v.squeeze(-1)