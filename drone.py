import numpy as np


# Swarm types — each has distinct movement behavior
SWARM_DIRECT    = "direct"      # straight line to target
SWARM_FLANKING  = "flanking"    # approaches from the side
SWARM_DIVING    = "diving"      # high altitude, dives steeply
SWARM_FAST      = "fast"        # low altitude, very fast
SWARM_ZIGZAG    = "zigzag"      # evasive lateral movement


class Drone:
    def __init__(self, position, target, speed=1.0, drone_id=0,
                 swarm_id=0, swarm_type=SWARM_DIRECT):
        """
        3D drone with swarm identity and behavior type.

        Args:
            position   : (x, y, z) start
            target     : (x, y, z) target
            speed      : base movement speed
            drone_id   : unique drone ID
            swarm_id   : which swarm this drone belongs to
            swarm_type : movement pattern (SWARM_* constant)
        """
        self.id         = drone_id
        self.swarm_id   = swarm_id
        self.swarm_type = swarm_type

        self.position   = np.array(position, dtype=float)
        self.target     = np.array(target,   dtype=float)
        self.speed      = speed
        self.alive      = True

        self.velocity   = np.zeros(3, dtype=float)

        # Zigzag phase offset — unique per drone so they don't sync
        self._zigzag_phase = np.random.uniform(0, 2 * np.pi)
        self._step_count   = 0

        self.history     = []
        self.max_history = 20

    # ------------------------------------------------------------------
    # Movement — each swarm type moves differently
    # ------------------------------------------------------------------
    def move(self):
        if not self.alive:
            return

        self._update_history()
        self._step_count += 1

        if self.swarm_type == SWARM_DIRECT:
            self._move_direct()

        elif self.swarm_type == SWARM_FLANKING:
            self._move_flanking()

        elif self.swarm_type == SWARM_DIVING:
            self._move_diving()

        elif self.swarm_type == SWARM_FAST:
            self._move_direct()          # same direction, just faster (set at spawn)

        elif self.swarm_type == SWARM_ZIGZAG:
            self._move_zigzag()

        else:
            self._move_direct()

        self.position += self.velocity

    def _move_direct(self):
        """Straight line toward target."""
        direction = self.target - self.position
        dist      = np.linalg.norm(direction)
        if dist > 1e-6:
            self.velocity = (direction / dist) * self.speed

    def _move_flanking(self):
        """
        First moves laterally to flank, then closes in.
        Switches to direct approach once within 40 units.
        """
        dist = np.linalg.norm(self.position - self.target)
        if dist > 40:
            # Move toward a flanking waypoint offset on Y axis
            waypoint  = self.target + np.array([0, 35 * np.sign(self.position[1] + 1e-6), 5])
            direction = waypoint - self.position
            d         = np.linalg.norm(direction)
            if d > 1e-6:
                self.velocity = (direction / d) * self.speed
        else:
            self._move_direct()

    def _move_diving(self):
        """
        Maintains altitude until close, then dives toward target.
        """
        dist = np.linalg.norm(self.position[:2] - self.target[:2])  # horizontal dist
        if dist > 30:
            # Fly level — only close horizontal distance
            direction        = self.target - self.position
            direction[2]     = 0                                      # zero out vertical
            d                = np.linalg.norm(direction)
            if d > 1e-6:
                self.velocity = (direction / d) * self.speed
        else:
            # Dive — full 3D straight line
            self._move_direct()

    def _move_zigzag(self):
        """
        Evasive lateral oscillation while advancing toward target.
        """
        direction = self.target - self.position
        dist      = np.linalg.norm(direction)
        if dist < 1e-6:
            return

        forward = (direction / dist) * self.speed * 0.8   # 80% speed forward

        # Perpendicular axis (rotate forward in XY plane by 90°)
        perp     = np.array([-forward[1], forward[0], 0])
        perp_len = np.linalg.norm(perp)
        if perp_len > 1e-6:
            perp = perp / perp_len

        lateral       = np.sin(self._zigzag_phase + self._step_count * 0.4) * self.speed * 0.5
        self.velocity = forward + perp * lateral

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def _update_history(self):
        self.history.append(self.position.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def has_reached_target(self, threshold=1.5):
        return np.linalg.norm(self.position - self.target) < threshold

    def distance_to_target(self):
        return np.linalg.norm(self.position - self.target)

    def time_to_impact(self):
        if self.speed < 1e-6:
            return float("inf")
        return self.distance_to_target() / self.speed

    def is_active(self):
        return self.alive

    # ------------------------------------------------------------------
    # Trajectory Prediction
    # ------------------------------------------------------------------
    def predict_future_position(self, steps=5):
        if len(self.history) < 2:
            return self.position.copy()
        velocity   = self.history[-1] - self.history[-2]
        future_pos = self.position + velocity * steps
        return future_pos

    # ------------------------------------------------------------------
    # State vector — 9 floats (added swarm_id for RL context)
    # [x, y, z, vx, vy, vz, dist_to_target, alive, swarm_id]
    # ------------------------------------------------------------------
    def get_state(self):
        return np.array([
            self.position[0],
            self.position[1],
            self.position[2],
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.distance_to_target(),
            float(self.alive),
            float(self.swarm_id),
        ], dtype=float)

    # ------------------------------------------------------------------
    # Destroy
    # ------------------------------------------------------------------
    def destroy(self):
        self.alive    = False
        self.velocity = np.zeros(3)

    def __repr__(self):
        pos = np.round(self.position, 2)
        return (f"Drone(id={self.id}, swarm={self.swarm_id}/"
                f"{self.swarm_type}, pos={pos}, alive={self.alive})")