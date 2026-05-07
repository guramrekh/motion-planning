from kinematics import forward_kinematics
from workspace import Workspace


def arm_in_collision(workspace: Workspace, theta1: float, theta2: float) -> bool:
    """True if the arm at (theta1, theta2) intersects any obstacle."""
    shoulder = workspace.robot.base
    elbow, end_eff = forward_kinematics(workspace.robot, theta1, theta2)
    for obs in workspace.obstacles:
        if obs.intersects_segment(shoulder, elbow) or obs.intersects_segment(elbow, end_eff):
            return True
    return False
