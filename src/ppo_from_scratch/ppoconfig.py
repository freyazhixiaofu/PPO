from dataclasses import dataclass

@dataclass
class Config:
    env_id: str = "CartPole-v1"
    total_timesteps: int = 200_000
    n_envs: int = 8
    rollout_steps: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 3e-4
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 10
    minibatch_size: int = 2048
    seed: int = 42
    device: str = "auto"
    eval_episodes: int = 10
