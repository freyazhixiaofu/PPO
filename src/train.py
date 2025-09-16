# for update in range(num_updates):
#     # 1) Collect rollout
#     traj = collect(envs, policy, n_steps)           # states, actions, rewards, dones, logp, values
#     # 2) Compute returns & advantages
#     adv, ret = compute_gae(traj.rewards, traj.values, traj.dones, gamma=0.99, lam=0.95, last_value)
#     adv = (adv - adv.mean()) / (adv.std() + 1e-8)

#     # 3) PPO updates (K epochs over shuffled minibatches)
#     for epoch in range(ppo_epochs):
#         for mb in minibatches(traj, adv, ret, batch_size):
#             ratio = exp(new_logp - old_logp)
#             L_clip = mean(min(ratio*adv, clip(ratio,1-ε,1+ε)*adv))
#             v_loss = mse(value_pred, ret)
#             ent = mean(policy_entropy)
#             loss = -(L_clip + c_ent*ent - c_v*v_loss)
#             optimizer.zero_grad(); loss.backward()
#             clip_grad_norm_(params, 0.5)
#             optimizer.step()

#     # 4) Evaluate periodically
#     if update % eval_every == 0:
#         avg_return = evaluate(policy, "CartPole-v1", episodes=10, deterministic=True)
#         log(avg_return, steps_so_far)
#         save_best_model(...)
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from make_env import make_train_env, make_eval_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--logdir", type=str, default="./tb")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_random_seed(args.seed)

    # ===== 1) Env =====
    train_env = make_train_env(n_envs=args.n_envs)
    eval_env  = make_eval_env()

    # ===== 2) PPO =====
    # Key knobs you’ll likely study:
    # n_steps: rollout length per env  | batch_size: SGD batch size
    # gamma: discount                  | gae_lambda: GAE parameter
    # clip_range: policy clip epsilon  | ent_coef: entropy bonus
    # vf_coef: value loss weight       | n_epochs: SGD passes per update
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=256,            # per-env steps ⇒ total batch = n_steps * n_envs
        batch_size=2048,        # larger than 256*n_envs is fine; SB3 will handle it
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        tensorboard_log=args.logdir,
        verbose=1,
        device=args.device,
        seed=args.seed,
    )

    # Optional: periodic eval + best model saving
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints",
        log_path="./checkpoints",
        eval_freq=10_000 // args.n_envs,  # called every X calls to env.step (per env)
        n_eval_episodes=10,
        deterministic=True,
    )

    # ===== 3) Learn =====
    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save("ppo_cartpole")

    # ===== 4) Quick evaluation =====
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"[final] Eval return: {mean_r:.1f} ± {std_r:.1f}")

if __name__ == "__main__":
    main()
