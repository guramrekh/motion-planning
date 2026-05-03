import numpy as np
from numpy.typing import NDArray
from astar import a_star
from cspace import config_to_index, index_to_config


_NEIGHBOR_OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
)

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
    step = 2 * np.pi / resolution
    di = b[0] - a[0]
    dj = b[1] - a[1]
    return step * np.hypot(di, dj)


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

    Snaps endpoints to their containing cells. The returned path is a list of
    (theta1, theta2) tuples at cell centers; returns None if start and goal lie
    in different connected components of the free C-space.

    The caller is responsible for separately verifying that start/goal cells are
    free (failure mode 2). If either cell is in collision, this function will
    still return None, but the caller cannot distinguish that from a true
    no-path case (failure mode 3).
    """
    start_idx = config_to_index(*start_angles, resolution)
    goal_idx = config_to_index(*goal_angles, resolution)

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
