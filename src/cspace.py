import numpy as np
from numpy.typing import NDArray
from workspace import Workspace


def build_cspace_bitmap(workspace: Workspace, resolution: int = 200) -> NDArray[np.bool_]:
    """
    Returns a (resolution x resolution) boolean array representing C-space.
    bitmap[i, j] = True  →  arm is in collision at configuration (theta1_i, theta2_j).

    Axis ranges:
      theta1 ∈ [0, π]  — the workspace floor (y=0) makes negative theta1 always
                         in collision with the boundary, so it is excluded.
      theta2 ∈ [-π, π] — link 2 can fold either way around the elbow.

    Uses vectorized numpy operations: computes FK and collision tests over the
    entire grid at once rather than looping over each cell individually.
    """
    base_x, base_y = workspace.robot.base
    L1, L2 = workspace.robot.link_lengths

    theta1_vals = np.linspace(0, np.pi, resolution, endpoint=False)
    theta2_vals = np.linspace(-np.pi, np.pi, resolution, endpoint=False)
    theta1_grid, theta2_grid = np.meshgrid(theta1_vals, theta2_vals, indexing='ij')

    # FK over entire grid
    elbow_x = base_x + L1 * np.cos(theta1_grid)
    elbow_y = base_y + L1 * np.sin(theta1_grid)
    end_eff_x = elbow_x + L2 * np.cos(theta1_grid + theta2_grid)
    end_eff_y = elbow_y + L2 * np.sin(theta1_grid + theta2_grid)

    bitmap = np.zeros((resolution, resolution), dtype=bool)

    for obs in workspace.obstacles:
        bitmap |= obs.intersects_segment_grid(base_x, base_y, elbow_x, elbow_y)
        bitmap |= obs.intersects_segment_grid(elbow_x, elbow_y, end_eff_x, end_eff_y)

    return bitmap


def config_to_index(theta1: float, theta2: float, resolution: int) -> tuple[int, int]:
    """Maps (theta1, theta2) angles to (i, j) indices into the bitmap.

    Out-of-range angles are clamped to the nearest valid index. Callers that
    need to reject out-of-range configs should validate before calling.
    """
    step1 = np.pi / resolution             # theta1 ∈ [0, π]
    step2 = 2 * np.pi / resolution         # theta2 ∈ [-π, π]
    i = int(theta1 / step1)
    j = int((theta2 + np.pi) / step2)
    i = max(0, min(resolution - 1, i))
    j = max(0, min(resolution - 1, j))
    return i, j


def index_to_config(i: int, j: int, resolution: int) -> tuple[float, float]:
    """Maps (i, j) bitmap indices to the center angles (theta1, theta2) of that cell."""
    step1 = np.pi / resolution
    step2 = 2 * np.pi / resolution
    theta1 = (i + 0.5) * step1
    theta2 = -np.pi + (j + 0.5) * step2
    return theta1, theta2
