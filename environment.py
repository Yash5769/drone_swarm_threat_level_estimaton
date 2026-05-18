import numpy as np
from drone import (Drone, SWARM_DIRECT, SWARM_FLANKING,
                   SWARM_DIVING, SWARM_FAST, SWARM_ZIGZAG)
from effector import Effector
from utils import predict_trajectory, compute_threat_score


# Targeting strategies
STRATEGY_CLOSEST   = "closest"
STRATEGY_FASTEST   = "fastest"
STRATEGY_THREAT    = "threat"
STRATEGY_PREDICTED = "predicted"

# Swarm configuration: (type, speed_range, spawn_region, size)
SWARM_CONFIGS = [
    {
        "type"        : SWARM_DIRECT,
        "speed_range" : (1.5, 2.5),
        "spawn_x"     : (70, 100),
        "spawn_y"     : (-20, 20),
        "spawn_z"     : (2, 10),
        "size"        : 5,
        "label"       : "Direct",
        "color"       : "royalblue",
    },
    {
        "type"        : SWARM_FLANKING,
        "speed_range" : (1.5, 2.5),
        "spawn_x"     : (50, 80),
        "spawn_y"     : (40, 70),
        "spawn_z"     : (5, 15),
        "size"        : 4,
        "label"       : "Flanking",
        "color"       : "darkorange",
    },
    {
        "type"        : SWARM_DIVING,
        "speed_range" : (2.0, 3.0),
        "spawn_x"     : (60, 90),
        "spawn_y"     : (-40, 40),
        "spawn_z"     : (30, 55),
        "size"        : 4,
        "label"       : "Diving",
        "color"       : "purple",
    },
    {
        "type"        : SWARM_FAST,
        "speed_range" : (3.5, 5.0),
        "spawn_x"     : (80, 110),
        "spawn_y"     : (-30, 30),
        "spawn_z"     : (2, 8),
        "size"        : 4,
        "label"       : "Fast",
        "color"       : "crimson",
    },
    {
        "type"        : SWARM_ZIGZAG,
        "speed_range" : (2.0, 3.0),
        "spawn_x"     : (65, 95),
        "spawn_y"     : (-50, 50),
        "spawn_z"     : (5, 20),
        "size"        : 4,
        "label"       : "Zigzag",
        "color"       : "teal",
    },
]


class Environment:
    def __init__(self,
                 num_effectors=3,
                 strategy=STRATEGY_THREAT,
                 swarm_configs=None):
        """
        Multi-swarm 3D defense simulation.

        Args:
            num_effectors : number of defense effectors
            strategy      : rule-based targeting strategy
            swarm_configs : list of swarm config dicts (uses SWARM_CONFIGS default)
        """
        self.num_effectors  = num_effectors
        self.strategy       = strategy
        self.swarm_configs  = swarm_configs or SWARM_CONFIGS

        self.target         = np.array([0.0, 0.0, 0.0])
        self.damage         = 0
        self.time_step      = 0

        # Accuracy tracking
        self.total_shots    = 0
        self.hits           = 0

        self.drones         = []
        self.effectors      = []
        self.swarm_labels   = {}   # swarm_id -> label string
        self.swarm_colors   = {}   # swarm_id -> color string

        self._init_swarms()
        self._init_effectors()

        self.num_drones = len(self.drones)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_swarms(self):
        drone_id = 0
        for swarm_id, cfg in enumerate(self.swarm_configs):
            self.swarm_labels[swarm_id] = cfg["label"]
            self.swarm_colors[swarm_id] = cfg["color"]

            for _ in range(cfg["size"]):
                pos = np.array([
                    np.random.uniform(*cfg["spawn_x"]),
                    np.random.uniform(*cfg["spawn_y"]),
                    np.random.uniform(*cfg["spawn_z"]),
                ])
                speed = np.random.uniform(*cfg["speed_range"])
                self.drones.append(Drone(
                    position=pos,
                    target=self.target,
                    speed=speed,
                    drone_id=drone_id,
                    swarm_id=swarm_id,
                    swarm_type=cfg["type"],
                ))
                drone_id += 1

    def _init_effectors(self):
        angles = np.linspace(0, 2 * np.pi, self.num_effectors, endpoint=False)
        for i, angle in enumerate(angles):
            pos = np.array([18 * np.cos(angle), 18 * np.sin(angle), 0.0])
            self.effectors.append(Effector(
                position=pos,
                range_radius=35.0,
                cooldown=3,
                effector_id=i,
            ))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, rl_action=None):
        self.time_step += 1

        for drone in self.drones:
            if drone.alive:
                drone.move()
                if drone.has_reached_target():
                    self.damage += 1
                    drone.destroy()

        alive = [d for d in self.drones if d.alive]
        if not alive:
            return

        for eff in self.effectors:
            eff.update()
            if not eff.can_fire():
                continue

            if rl_action is not None:
                target_drone = self._resolve_rl_action(rl_action, eff, alive)
            else:
                target_drone = self._select_target(alive, eff)

            if target_drone is None:
                continue

            self.total_shots += 1
            hit = eff.try_intercept(target_drone)
            if hit:
                self.hits += 1
                alive.remove(target_drone)

    # ------------------------------------------------------------------
    # Targeting
    # ------------------------------------------------------------------
    def _select_target(self, alive_drones, effector):
        candidates = [d for d in alive_drones if effector.in_range(d)]
        if not candidates:
            return None

        if self.strategy == STRATEGY_CLOSEST:
            return min(candidates, key=lambda d: d.distance_to_target())

        elif self.strategy == STRATEGY_FASTEST:
            return max(candidates, key=lambda d: d.speed)

        elif self.strategy == STRATEGY_PREDICTED:
            def predicted_dist(d):
                future = predict_trajectory(d.history, steps=5)
                return np.linalg.norm(future - self.target)
            return min(candidates, key=predicted_dist)

        else:  # STRATEGY_THREAT
            return min(candidates, key=lambda d: compute_threat_score(d, self.target))

    def _resolve_rl_action(self, action_idx, effector, alive_drones):
        if action_idx < len(self.drones):
            chosen = self.drones[action_idx]
            if chosen.alive and effector.in_range(chosen):
                return chosen
        return self._select_target(alive_drones, effector)

    # ------------------------------------------------------------------
    # State — 9 floats per drone + 5 per effector
    # ------------------------------------------------------------------
    def get_state(self):
        drone_states    = np.concatenate([d.get_state() for d in self.drones])
        effector_states = np.concatenate([e.get_state() for e in self.effectors])
        return np.concatenate([drone_states, effector_states]).astype(np.float32)

    def state_size(self):
        return self.num_drones * 9 + self.num_effectors * 5

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------
    def get_accuracy(self):
        """
        Interception accuracy = hits / total_shots fired.
        Also returns per-swarm damage breakdown.
        """
        acc = (self.hits / self.total_shots * 100) if self.total_shots > 0 else 0.0

        # Per-swarm damage
        swarm_damage = {}
        for swarm_id, label in self.swarm_labels.items():
            swarm_drones = [d for d in self.drones if d.swarm_id == swarm_id]
            hit_target   = sum(1 for d in swarm_drones if not d.alive
                               and d.distance_to_target() < 1.5)
            swarm_damage[label] = hit_target

        return {
            "total_shots"       : self.total_shots,
            "hits"              : self.hits,
            "accuracy_pct"      : round(acc, 2),
            "drones_destroyed"  : self.hits,
            "damage_taken"      : self.damage,
            "interception_rate" : round(
                (1 - self.damage / max(self.num_drones, 1)) * 100, 2
            ),
        }

    # ------------------------------------------------------------------
    # Episode control
    # ------------------------------------------------------------------
    def is_done(self):
        return all(not d.alive for d in self.drones)

    def reset(self):
        self.drones.clear()
        self.effectors.clear()
        self.damage      = 0
        self.time_step   = 0
        self.total_shots = 0
        self.hits        = 0
        self._init_swarms()
        self._init_effectors()
        self.num_drones = len(self.drones)
        return self.get_state()

    def get_info(self):
        return {
            "time_step"   : self.time_step,
            "damage"      : self.damage,
            "alive_drones": sum(d.alive for d in self.drones),
            "strategy"    : self.strategy,
            **self.get_accuracy(),
        }