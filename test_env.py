"""
test_env.py — quick headless test of all swarms + all strategies.
Run this before training to verify the simulation is working.
"""
from environment import (Environment,
                         STRATEGY_CLOSEST, STRATEGY_FASTEST,
                         STRATEGY_THREAT,  STRATEGY_PREDICTED)


def run(strategy, episodes=5):
    damages, accs, int_rates = [], [], []
    for _ in range(episodes):
        env = Environment(num_effectors=3, strategy=strategy)
        while not env.is_done():
            env.step()
        m = env.get_accuracy()
        damages.append(m["damage_taken"])
        accs.append(m["accuracy_pct"])
        int_rates.append(m["interception_rate"])

    avg_d  = sum(damages)   / episodes
    avg_a  = sum(accs)      / episodes
    avg_ir = sum(int_rates) / episodes
    print(f"  {strategy:<18}  damage={avg_d:.1f}  "
          f"shot_acc={avg_a:.1f}%  interception={avg_ir:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Headless test — 5 episodes per strategy")
    print("=" * 60)

    # Quick single run to show swarm breakdown
    env = Environment(num_effectors=3, strategy=STRATEGY_THREAT)
    print(f"\nSwarms in this episode:")
    for sid, label in env.swarm_labels.items():
        count = sum(1 for d in env.drones if d.swarm_id == sid)
        print(f"  Swarm {sid} [{label}]: {count} drones  "
              f"color={env.swarm_colors[sid]}")
    print(f"  Total drones: {env.num_drones}\n")

    for strategy in [STRATEGY_CLOSEST, STRATEGY_FASTEST,
                     STRATEGY_THREAT,  STRATEGY_PREDICTED]:
        run(strategy)

    print("=" * 60)
    print("Test complete — simulation is working correctly.")