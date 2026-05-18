import numpy as np


class Effector:
    def __init__(self, position, range_radius=15.0, cooldown=3, effector_id=0):
        """
        Defense effector (interceptor) in 3D space.

        Args:
            position     : (x, y, z) fixed placement
            range_radius : max intercept distance
            cooldown     : steps to wait between shots
            effector_id  : unique identifier
        """
        self.id               = effector_id
        self.position         = np.array(position, dtype=float)
        self.range            = range_radius
        self.cooldown         = cooldown
        self.current_cooldown = 0

        # Tracking
        self.shots_fired      = 0
        self.kills            = 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def can_fire(self):
        return self.current_cooldown == 0

    def in_range(self, drone):
        """Check if a drone is within intercept range."""
        return np.linalg.norm(self.position - drone.position) <= self.range

    # ------------------------------------------------------------------
    # FIX: old code called drone.alive = False directly in gym_env,
    # bypassing range and cooldown entirely.
    # Now ALL interception goes through try_intercept().
    # ------------------------------------------------------------------
    def try_intercept(self, drone):
        """
        Attempt to intercept a drone.
        Returns True if successful, False otherwise.
        """
        if not self.can_fire():
            return False
        if not drone.alive:
            return False
        if not self.in_range(drone):
            return False

        drone.destroy()
        self.current_cooldown = self.cooldown
        self.shots_fired     += 1
        self.kills           += 1
        return True

    def update(self):
        """Decrement cooldown each step."""
        if self.current_cooldown > 0:
            self.current_cooldown -= 1

    # ------------------------------------------------------------------
    # State (Phase 2 — fed into RL state vector)
    # Vector: [x, y, z, cooldown_remaining, range]  → 5 floats
    # ------------------------------------------------------------------
    def get_state(self):
        return np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            float(self.current_cooldown),
            float(self.range),
        ], dtype=float)

    def __repr__(self):
        return (f"Effector(id={self.id}, pos={self.position}, "
                f"cooldown={self.current_cooldown}/{self.cooldown}, "
                f"kills={self.kills})")