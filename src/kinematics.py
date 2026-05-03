import numpy as np
from workspace import Robot


def forward_kinematics(robot: Robot, theta1: float, theta2: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Returns (x, y) coordinates of elbow and end effector based on joint angles.
    shoulder is fixed at robot.base and is not returned.
    """
    base_x, base_y = robot.base
    L1, L2 = robot.link_lengths

    elbow_x = base_x + L1 * np.cos(theta1)
    elbow_y = base_y + L1 * np.sin(theta1)

    end_eff_x = elbow_x + L2 * np.cos(theta1 + theta2)
    end_eff_y = elbow_y + L2 * np.sin(theta1 + theta2)

    return (elbow_x, elbow_y), (end_eff_x, end_eff_y)


def is_point_reachable(robot: Robot, x: float, y: float) -> bool:
    """True if (x, y) lies within the arm's reachable annulus."""
    base_x, base_y = robot.base
    L1, L2 = robot.link_lengths
    dist = np.hypot(x - base_x, y - base_y)
    return abs(L1 - L2) <= dist <= L1 + L2


def inverse_kinematics(robot: Robot, x: float, y: float) -> list[tuple[float, float]]:
    """
    Returns list of (theta1, theta2) solutions placing the end-effector at (x, y).
    [] if geometrically unreachable.
    1 solution on the boundary of the reachable annulus, 2 otherwise (elbow-right, elbow-left).
    """
    base_x, base_y = robot.base
    L1, L2 = robot.link_lengths

    if not is_point_reachable(robot, x, y):
        return []

    diff_x, diff_y = x - base_x, y - base_y
    dist = np.hypot(diff_x, diff_y)

    cos_theta2 = (dist**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # guard against float drift

    angle_to_target = np.arctan2(diff_y, diff_x)

    solutions = []
    for sign in [1, -1]:  # elbow-right, elbow-left
        theta2 = sign * np.arccos(cos_theta2)
        angle_correction = np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
        theta1 = angle_to_target - angle_correction
        solutions.append((theta1, theta2))

    # degenerate: arm fully extended or fully folded
    if np.isclose(abs(cos_theta2), 1.0):
        solutions = [solutions[0]]

    return solutions