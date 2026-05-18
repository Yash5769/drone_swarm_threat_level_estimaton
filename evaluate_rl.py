"""
evaluate_rl.py
--------------
Evaluates a trained PPO model against all 4 rule-based baselines
and computes comprehensive accuracy metrics per swarm type.

Run AFTER train_rl.py has saved ppo_drone_swarm.zip
"""
import numpy as np
from stable_baselines3 import PPO
from gym_env import DroneSwarmEnv
from environment import (Environment,
                         STRATEGY_CLOSEST, STRATEGY_FASTEST,
                         STRATEGY_THREAT,  STRATEGY_PREDICTED)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def run_rl_episode(model, num_effectors=3):
    env    = DroneSwarmEnv(num_effectors=num_effectors)
    obs, _ = env.reset()
    done   = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.env          # return underlying Environment for metrics


def run_heuristic_episode(strategy, num_effectors=3):
    env = Environment(num_effectors=num_effectors, strategy=strategy)
    while not env.is_done():
        env.step()
    return env


def compute_metrics(env_instance):
    acc = env_instance.get_accuracy()

    # Per-swarm interception breakdown
    swarm_stats = {}
    for sid, label in env_instance.swarm_labels.items():
        swarm_drones    = [d for d in env_instance.drones if d.swarm_id == sid]
        total           = len(swarm_drones)
        intercepted     = sum(1 for d in swarm_drones if not d.alive
                              and d.distance_to_target() >= 1.5)
        hit_target      = sum(1 for d in swarm_drones if not d.alive
                              and d.distance_to_target() < 1.5)
        swarm_stats[label] = {
            "total"      : total,
            "intercepted": intercepted,
            "hit_target" : hit_target,
            "int_rate_%" : round(intercepted / total * 100, 1) if total else 0,
        }

    return {**acc, "per_swarm": swarm_stats}


# ──────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(model_path="ppo_drone_swarm", episodes=30):
    print("\n" + "=" * 65)
    print("  RL MODEL ACCURACY EVALUATION")
    print("=" * 65)

    # Load model
    try:
        model = PPO.load(model_path)
        print(f"  Loaded model: {model_path}.zip")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        print("  Run train_rl.py first to generate ppo_drone_swarm.zip")
        return

    strategies = {
        "RL (PPO)"         : None,
        "Heuristic-Closest": STRATEGY_CLOSEST,
        "Heuristic-Fastest": STRATEGY_FASTEST,
        "Heuristic-Threat" : STRATEGY_THREAT,
        "Heuristic-Predict": STRATEGY_PREDICTED,
    }

    all_results = {}

    for name, strategy in strategies.items():
        shot_accs, int_rates, damages = [], [], []
        per_swarm_agg = {}

        print(f"\n  Running {episodes} episodes: {name} ...")
        for ep in range(episodes):
            if strategy is None:
                env = run_rl_episode(model)
            else:
                env = run_heuristic_episode(strategy)

            m = compute_metrics(env)
            shot_accs.append(m["accuracy_pct"])
            int_rates.append(m["interception_rate"])
            damages.append(m["damage_taken"])

            for swarm_label, stats in m["per_swarm"].items():
                if swarm_label not in per_swarm_agg:
                    per_swarm_agg[swarm_label] = []
                per_swarm_agg[swarm_label].append(stats["int_rate_%"])

        all_results[name] = {
            "shot_accuracy"     : np.mean(shot_accs),
            "interception_rate" : np.mean(int_rates),
            "avg_damage"        : np.mean(damages),
            "per_swarm"         : {k: np.mean(v) for k, v in per_swarm_agg.items()},
        }

    # ── Print results table ────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'Method':<22} {'Shot Acc%':>10} {'Int Rate%':>10} {'Avg Damage':>11}")
    print("  " + "-" * 55)
    for name, r in all_results.items():
        marker = "  ◄ RL" if "RL" in name else ""
        print(f"  {name:<22} {r['shot_accuracy']:>9.1f}% "
              f"{r['interception_rate']:>9.1f}% "
              f"{r['avg_damage']:>10.2f}{marker}")

    # ── Per-swarm breakdown ────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PER-SWARM INTERCEPTION RATE  (avg over episodes)")
    print("=" * 65)

    # Header
    swarm_names = list(next(iter(all_results.values()))["per_swarm"].keys())
    header = f"  {'Method':<22}" + "".join(f"{s[:8]:>10}" for s in swarm_names)
    print(header)
    print("  " + "-" * (22 + 10 * len(swarm_names)))

    for name, r in all_results.items():
        row = f"  {name:<22}"
        for s in swarm_names:
            row += f"{r['per_swarm'].get(s, 0):>9.1f}%"
        print(row)

    # ── RL vs best heuristic ───────────────────────────────────
    rl_int  = all_results["RL (PPO)"]["interception_rate"]
    best_h  = max(
        (v["interception_rate"] for k, v in all_results.items() if "RL" not in k)
    )
    diff    = rl_int - best_h
    verdict = "BETTER" if diff >= 0 else "WORSE"
    print(f"\n  RL interception rate : {rl_int:.1f}%")
    print(f"  Best heuristic rate  : {best_h:.1f}%")
    print(f"  RL vs Best heuristic : {diff:+.1f}%  →  RL is {verdict}")
    print("=" * 65 + "\n")

    return all_results


if __name__ == "__main__":
    evaluate(episodes=30)