from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gym_env import DroneSwarmEnv


def train(total_timesteps=100_000, num_effectors=3):
    env      = make_vec_env(lambda: DroneSwarmEnv(num_effectors), n_envs=4)
    eval_env = make_vec_env(lambda: DroneSwarmEnv(num_effectors), n_envs=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy", env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    print(f"Training PPO | effectors={num_effectors} | timesteps={total_timesteps:,}")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save("ppo_drone_swarm")
    print("Done. Saved: ppo_drone_swarm.zip")

    # Auto-evaluate after training
    from evaluate_rl import evaluate
    evaluate(episodes=20)

    return model


if __name__ == "__main__":
    train()