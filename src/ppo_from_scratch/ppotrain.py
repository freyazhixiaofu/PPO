import argparse
from ppoconfig import Config
from agent import ppo_train

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")  # "auto"/"cpu"/"cuda"
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = Config(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        device=args.device,
        seed=args.seed,
    )
    ppo_train(cfg)

if __name__ == "__main__":
    main()