import time
import gymnasium as gym
from stable_baselines3 import PPO

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load("ppo_cartpole", device="cpu")

    obs, _ = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # slow down for visibility (optional)
        time.sleep(1/60)
    env.close()

if __name__ == "__main__":
    main()
