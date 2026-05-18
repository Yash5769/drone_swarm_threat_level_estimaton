import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from environment import Environment, STRATEGY_THREAT


def run_visual(num_effectors=3, strategy=STRATEGY_THREAT):
    env = Environment(num_effectors=num_effectors, strategy=strategy)

    # -- Legend patches for each swarm (built once) --
    legend_patches = [
        mpatches.Patch(color=env.swarm_colors[sid], label=f"Swarm {sid}: {lbl}")
        for sid, lbl in env.swarm_labels.items()
    ]
    legend_patches += [
        mpatches.Patch(color="gray",  label="Destroyed"),
        mpatches.Patch(color="red",   label="Target"),
        mpatches.Patch(color="green", label="Effector (ready)"),
        mpatches.Patch(color="orange",label="Effector (cooldown)"),
    ]

    plt.ion()
    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Multi-Swarm Drone Defense Simulation", fontsize=13, fontweight="bold")

    while not env.is_done():
        ax_top.cla()
        ax_side.cla()

        alive_drones = [d for d in env.drones if d.alive]
        dead_drones  = [d for d in env.drones if not d.alive]

        # ── Top view (X-Y) ───────────────────────────────────
        for d in alive_drones:
            color = env.swarm_colors[d.swarm_id]
            ax_top.scatter(d.position[0], d.position[1],
                           color=color, s=45, zorder=3)
            # Trajectory tail
            if len(d.history) > 1:
                hist = np.array(d.history)
                ax_top.plot(hist[:, 0], hist[:, 1],
                            color=color, alpha=0.2, linewidth=1)

        for d in dead_drones:
            ax_top.scatter(d.position[0], d.position[1],
                           color="gray", s=18, alpha=0.3, zorder=2)

        # Target
        ax_top.scatter(*env.target[:2], color="red",
                       s=220, marker="*", zorder=6)

        # Effectors + range circles
        for eff in env.effectors:
            color  = "green" if eff.can_fire() else "orange"
            circle = mpatches.Circle(eff.position[:2], eff.range,
                                     color=color, fill=False,
                                     linestyle="--", alpha=0.4)
            ax_top.add_patch(circle)
            ax_top.scatter(*eff.position[:2], color=color,
                           s=110, marker="^", zorder=5)

        info = env.get_info()
        ax_top.set_xlim(-115, 120)
        ax_top.set_ylim(-90, 90)
        ax_top.set_xlabel("X")
        ax_top.set_ylabel("Y")
        ax_top.set_title(
            f"Top View  |  Step {info['time_step']}  |  "
            f"Alive: {info['alive_drones']}  |  "
            f"Damage: {info['damage']}  |  "
            f"Accuracy: {info['accuracy_pct']}%"
        )
        ax_top.legend(handles=legend_patches, loc="upper right",
                      fontsize=6.5, framealpha=0.8)
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.grid(True, alpha=0.15)

        # ── Side view (X-Z) ──────────────────────────────────
        for d in alive_drones:
            color = env.swarm_colors[d.swarm_id]
            ax_side.scatter(d.position[0], d.position[2],
                            color=color, s=45, zorder=3)
            if len(d.history) > 1:
                hist = np.array(d.history)
                ax_side.plot(hist[:, 0], hist[:, 2],
                             color=color, alpha=0.2, linewidth=1)

        ax_side.scatter(env.target[0], env.target[2],
                        color="red", s=220, marker="*", zorder=6)

        for eff in env.effectors:
            color = "green" if eff.can_fire() else "orange"
            ax_side.scatter(eff.position[0], eff.position[2],
                            color=color, s=110, marker="^", zorder=5)

        ax_side.set_xlim(-10, 120)
        ax_side.set_ylim(-5, 65)
        ax_side.set_xlabel("X")
        ax_side.set_ylabel("Z  (Altitude)")
        ax_side.set_title("Side View  (altitude)")
        ax_side.grid(True, alpha=0.15)

        plt.tight_layout()
        plt.pause(0.08)
        env.step()

    # ── Final accuracy report ─────────────────────────────────
    plt.ioff()
    acc = env.get_accuracy()
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)
    print(f"  Total steps       : {env.time_step}")
    print(f"  Total drones      : {env.num_drones}")
    print(f"  Shots fired       : {acc['total_shots']}")
    print(f"  Drones destroyed  : {acc['drones_destroyed']}")
    print(f"  Damage taken      : {acc['damage_taken']}")
    print(f"  Shot accuracy     : {acc['accuracy_pct']}%")
    print(f"  Interception rate : {acc['interception_rate']}%")
    print("=" * 50)
    plt.show()


if __name__ == "__main__":
    run_visual()