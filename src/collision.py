import numpy as np
from numpy.typing import ArrayLike
from kinematics import forward_kinematics
from workspace import Workspace


def segment_intersects_circle(p1: ArrayLike, p2: ArrayLike, center: ArrayLike, radius: float) -> bool:
    """
    True if the line segment p1→p2 intersects a circle with given radius.

    Finds the closest point on the line to the circle center,
    then checks if that distance is within the radius.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    center = np.asarray(center, dtype=float)

    line_vec = p2 - p1
    p1_to_center_vec = center - p1

    line_length_sq = np.dot(line_vec, line_vec)
    if line_length_sq == 0.0:
        return np.linalg.norm(p1_to_center_vec) <= radius

    # t is how far along the line the closest point lies (0 = p1, 1 = p2)
    t = np.clip(np.dot(p1_to_center_vec, line_vec) / line_length_sq, 0.0, 1.0)
    closest_point = p1 + t * line_vec

    return np.linalg.norm(center - closest_point) <= radius


def arm_in_collision(workspace: Workspace, theta1: float, theta2: float) -> bool:
    """True if the arm at (theta1, theta2) intersects any obstacle."""
    shoulder = np.asarray(workspace.robot.base, dtype=float)
    elbow, end_eff = forward_kinematics(workspace.robot, theta1, theta2)

    for obs in workspace.obstacles:
        if (segment_intersects_circle(shoulder, elbow, obs.center_point, obs.radius) or
            segment_intersects_circle(elbow, end_eff, obs.center_point, obs.radius)):
            return True

    return False
