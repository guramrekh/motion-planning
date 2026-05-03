import numpy as np
from numpy.typing import NDArray
from workspace import Workspace


def build_cspace_bitmap(workspace: Workspace, resolution: int = 200) -> NDArray[np.bool_]:
    """
    Returns a (resolution x resolution) boolean array representing C-space.
    bitmap[i, j] = True  →  arm is in collision at configuration (theta1_i, theta2_j).
    Both axes span [-π, π].

    Uses vectorized numpy operations: computes FK and collision tests over the
    entire grid at once rather than looping over each cell individually.
    """
    base_x, base_y = workspace.robot.base
    L1, L2 = workspace.robot.link_lengths

    theta_vals = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    theta1_grid, theta2_grid = np.meshgrid(theta_vals, theta_vals, indexing='ij')

    # FK over entire grid
    elbow_x = base_x + L1 * np.cos(theta1_grid)
    elbow_y = base_y + L1 * np.sin(theta1_grid)
    end_eff_x = elbow_x + L2 * np.cos(theta1_grid + theta2_grid)
    end_eff_y = elbow_y + L2 * np.sin(theta1_grid + theta2_grid)

    bitmap = np.zeros((resolution, resolution), dtype=bool)

    for obs in workspace.obstacles:
        center_x, center_y = obs.center_point
        radius = obs.radius
        # link 1: shoulder → elbow
        bitmap |= _segment_intersects_circle_grid(base_x, base_y, elbow_x, elbow_y, center_x, center_y, radius)
        # link 2: elbow → end effector
        bitmap |= _segment_intersects_circle_grid(elbow_x, elbow_y, end_eff_x, end_eff_y, center_x, center_y, radius)

    return bitmap


def _segment_intersects_circle_grid(p1_x, p1_y, p2_x, p2_y, center_x, center_y, radius):
    """
    Vectorized segment-circle collision test for a grid of segments.
    p1 can be scalar (fixed shoulder) or a 2D array; p2 is always a 2D array.
    Returns a boolean array of the same shape as p2_x.
    """
    dx = p2_x - p1_x
    dy = p2_y - p1_y
    vx = center_x - p1_x
    vy = center_y - p1_y

    t = np.clip((vx * dx + vy * dy) / (dx * dx + dy * dy), 0.0, 1.0)

    closest_x = p1_x + t * dx
    closest_y = p1_y + t * dy

    return (center_x - closest_x) ** 2 + (center_y - closest_y) ** 2 <= radius ** 2


def config_to_index(theta1: float, theta2: float, resolution: int) -> tuple[int, int]:
    """Maps (theta1, theta2) angles to (i, j) indices into the bitmap."""
    step = 2 * np.pi / resolution
    i = int((theta1 + np.pi) / step) % resolution
    j = int((theta2 + np.pi) / step) % resolution
    return i, j


def index_to_config(i: int, j: int, resolution: int) -> tuple[float, float]:
    """Maps (i, j) bitmap indices to the center angles (theta1, theta2) of that cell."""
    step = 2 * np.pi / resolution
    theta1 = -np.pi + (i + 0.5) * step
    theta2 = -np.pi + (j + 0.5) * step
    return theta1, theta2
