import numpy as np


# ------------------------------------------------------------------
# Trajectory Prediction  (Phase 5 hook — replace body with LSTM later)
# ------------------------------------------------------------------
def predict_trajectory(history, steps=5):
    """
    Linear extrapolation of future position.

    Args:
        history : list of np.array positions (3D)
        steps   : how many steps ahead to predict

    Returns:
        np.array future position (3D)
    """
    if len(history) < 2:
        return history[-1].copy() if history else np.zeros(3)

    velocity   = history[-1] - history[-2]
    future_pos = history[-1] + velocity * steps
    return future_pos


# ------------------------------------------------------------------
# Threat Scoring  (Phase 6)
# ------------------------------------------------------------------
def compute_threat_score(drone, target, w1=1.0, w2=0.5, w3=0.3):
    """
    Heuristic threat score for a single drone.
    Lower score = higher threat (consistent with minimization).

    Args:
        drone  : Drone instance
        target : np.array target position (3D)
        w1     : weight for current distance to target
        w2     : weight for time-to-impact
        w3     : weight for speed (inverse — faster = more dangerous)

    Returns:
        float threat score
    """
    dist_now    = drone.distance_to_target()
    tti         = drone.time_to_impact()
    speed_bonus = 1.0 / (drone.speed + 1e-6)   # faster → lower score

    score = w1 * dist_now + w2 * tti + w3 * speed_bonus
    return score


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------
def distance_3d(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))