import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

def make_train_env(env_id: str = "CartPole-v1", n_envs: int = 8):
    # Vectorized env helps PPO collect batches efficiently
    return make_vec_env(env_id, n_envs=n_envs)

def make_eval_env(env_id: str = "CartPole-v1"):
    # Single non-vector env for evaluation / rendering
    return gym.make(env_id)
