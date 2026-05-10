import numpy as np
from numpy.typing import NDArray
from astar import a_star
from cspace import config_to_index, index_to_config


_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)

# theta1 axis spans [0, π], theta2 axis spans [-π, π], so the two axes have
# different step sizes — distances must account for that anisotropy.
_STEP1_FACTOR = np.pi          # multiplied by 1/resolution to get Δθ1 per cell
_STEP2_FACTOR = 2 * np.pi      # multiplied by 1/resolution to get Δθ2 per cell


def grid_neighbors(bitmap: NDArray[np.bool_], cell: tuple[int, int]) -> list[tuple[int, int]]:
    """8-connected neighbor cells that are in-bounds and free (not in collision)."""
    i, j = cell
    rows, cols = bitmap.shape
    result = []
    for di, dj in _NEIGHBOR_OFFSETS:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols and not bitmap[ni, nj]:
            result.append((ni, nj))
    return result


def grid_cost(a: tuple[int, int], b: tuple[int, int], resolution: int) -> float:
    """Euclidean distance in θ-space between cell-center configurations."""
    dtheta1 = (b[0] - a[0]) * _STEP1_FACTOR / resolution
    dtheta2 = (b[1] - a[1]) * _STEP2_FACTOR / resolution
    return float(np.hypot(dtheta1, dtheta2))


def grid_heuristic(a: tuple[int, int], goal: tuple[int, int], resolution: int) -> float:
    """
    Euclidean distance in θ-space — admissible (matches edge metric) and
    consistent, so A* is optimal on the grid.
    """
    return grid_cost(a, goal, resolution)


def plan_path(
    bitmap: NDArray[np.bool_],
    start_angles: tuple[float, float],
    goal_angles: tuple[float, float],
    resolution=200,
) -> list[tuple[float, float]] | None:
    """
    Plans a path from start_angles to goal_angles through free C-space using A*
    over the bitmap.

    Validates start/goal: raises ValueError if either configuration is out of
    range (θ1 ∉ [0, π] or θ2 ∉ [-π, π]) or lands in a collision cell.
    Returns None when both endpoints are valid but lie in different connected
    components of the free C-space.
    """
    _validate_config(bitmap, start_angles, resolution, "start")
    _validate_config(bitmap, goal_angles,  resolution, "goal")

    start_idx = config_to_index(*start_angles, resolution)
    goal_idx  = config_to_index(*goal_angles,  resolution)

    path_indices = a_star(
        start=start_idx,
        goal=goal_idx,
        neighbors_fn=lambda c: grid_neighbors(bitmap, c),
        heuristic_fn=lambda a, b: grid_heuristic(a, b, resolution),
        cost_fn=lambda a, b: grid_cost(a, b, resolution),
    )

    if path_indices is None:
        return None

    return [index_to_config(i, j, resolution) for (i, j) in path_indices]


def _validate_config(bitmap, angles, resolution, label):
    theta1, theta2 = angles
    if not (0 <= theta1 <= np.pi):
        raise ValueError(
            f"{label} theta1={theta1:.3f} is outside [0, π] — "
            f"the C-space excludes negative theta1 (elbow below workspace floor)."
        )
    if not (-np.pi <= theta2 <= np.pi):
        raise ValueError(
            f"{label} theta2={theta2:.3f} is outside [-π, π]."
        )
    if bitmap[config_to_index(theta1, theta2, resolution)]:
        raise ValueError(f"{label} configuration is in collision.")
