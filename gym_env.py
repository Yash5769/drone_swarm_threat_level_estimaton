import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment import Environment, STRATEGY_THREAT


class DroneSwarmEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_effectors=3, max_steps=300):
        super().__init__()
        self.num_effectors = num_effectors
        self.max_steps     = max_steps

        # Build temp env to get sizes
        _tmp = Environment(num_effectors=num_effectors)
        self._num_drones = _tmp.num_drones
        obs_size         = _tmp.state_size()

        self.observation_space = spaces.Box(
            low=-200.0, high=200.0,
            shape=(obs_size,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self._num_drones)

        self.env    = None
        self._steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env    = Environment(num_effectors=self.num_effectors)
        self._steps = 0
        return self.env.reset(), {}

    def step(self, action):
        self._steps     += 1
        prev_damage      = self.env.damage
        prev_hits        = self.env.hits
        prev_shots       = self.env.total_shots

        self.env.step(rl_action=int(action))

        damage_delta = self.env.damage - prev_damage
        new_hits     = self.env.hits - prev_hits
        new_shots    = self.env.total_shots - prev_shots

        # Reward:
        #   -10 each drone that reaches target
        #   +5  each successful interception
        #   -1  each wasted shot (fired but missed)
        reward  = -10.0 * damage_delta
        reward +=  5.0  * new_hits
        reward -=  1.0  * max(0, new_shots - new_hits)

        terminated = self.env.is_done()
        truncated  = self._steps >= self.max_steps
        obs        = self.env.get_state()
        info       = self.env.get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        info = self.env.get_info()
        print(f"Step {info['time_step']:4d} | Alive: {info['alive_drones']:3d} | "
              f"Damage: {info['damage']} | Accuracy: {info['accuracy_pct']}%")