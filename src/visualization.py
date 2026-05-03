import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy.typing import NDArray
from kinematics import forward_kinematics
from workspace import Workspace


def draw_workspace(ws: Workspace, theta1: float=0.0, theta2: float=0.0, ax=None, title="Workspace"):
    """
    Renders the workspace: body cavity bounds, obstacles, and the arm at (theta1, theta2).
    Pass an existing ax to embed in a larger figure (e.g. dual-panel), or leave None
    to create a standalone figure.
    Returns the axes object.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 7))

    xmin, xmax, ymin, ymax = ws.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_facecolor('#dff0d8')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # cavity border
    border = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor='#555', facecolor='none', zorder=1
    )
    ax.add_patch(border)

    # obstacles (organs)
    for obs in ws.obstacles:
        ax.add_patch(patches.Circle(
            obs.center_point, obs.radius,
            color='salmon', alpha=0.75, zorder=2
        ))
        ax.add_patch(patches.Circle(
            obs.center_point, obs.radius,
            fill=False, edgecolor='firebrick', linewidth=1.5, zorder=2
        ))

    # arm
    shoulder = ws.robot.base
    elbow, end_effector = forward_kinematics(ws.robot, theta1, theta2)

    ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]],
            color='steelblue', lw=6, solid_capstyle='round', zorder=3)
    ax.plot([elbow[0], end_effector[0]], [elbow[1], end_effector[1]],
            color='cornflowerblue', lw=4, solid_capstyle='round', zorder=3)

    ax.plot(*shoulder,     'ko', ms=10, zorder=4)
    ax.plot(*elbow,        'ko', ms=7,  zorder=4)
    ax.plot(*end_effector, 'r*', ms=12, zorder=4)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def draw_cspace(bitmap: NDArray[np.bool_], ax=None, title="C-space"):
    """
    Renders the C-space bitmap. Black = in-collision, white = free.
    theta1 on x-axis, theta2 on y-axis, both spanning [-π, π].
    Pass an existing ax to embed in a dual-panel figure.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        bitmap.T,           # transpose: rows=theta2, cols=theta1 → x=theta1, y=theta2
        origin='lower',
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        cmap='gray_r',      # black = collision (True), white = free (False)
        aspect='auto',
        interpolation='nearest'
    )

    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    labels = ['-π', '-π/2', '0', 'π/2', 'π']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlabel('θ1')
    ax.set_ylabel('θ2')
    ax.set_title(title)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def draw_plan(
    ws: Workspace,
    bitmap: NDArray[np.bool_],
    path: list[tuple[float, float]],
    start_config: tuple[float, float],
    goal_config: tuple[float, float],
    ax_ws=None,
    ax_cs=None,
):
    """
    Side-by-side path planning visualization.
    Left: workspace with start arm (faded) and goal arm (full color), end-effectors marked.
    Right: C-space bitmap with the A* path overlaid.
    Pass (ax_ws, ax_cs) to embed in an existing figure, or leave both None
    to create a standalone 1x2 figure. Returns (ax_ws, ax_cs).
    """
    standalone = ax_ws is None and ax_cs is None
    if standalone:
        _, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(14, 6))

    thetas = np.array(path)
    _, start_end_eff = forward_kinematics(ws.robot, *start_config)
    _, goal_end_eff  = forward_kinematics(ws.robot, *goal_config)

    # goal arm (full color) drawn by draw_workspace
    draw_workspace(ws, theta1=goal_config[0], theta2=goal_config[1],
                   ax=ax_ws, title="Workspace — start & goal")

    # start arm overlaid faded
    shoulder = ws.robot.base
    start_elbow, _ = forward_kinematics(ws.robot, *start_config)
    ax_ws.plot([shoulder[0], start_elbow[0]], [shoulder[1], start_elbow[1]],
               color='steelblue', lw=6, alpha=0.3, solid_capstyle='round', zorder=3)
    ax_ws.plot([start_elbow[0], start_end_eff[0]], [start_elbow[1], start_end_eff[1]],
               color='cornflowerblue', lw=4, alpha=0.3, solid_capstyle='round', zorder=3)

    ax_ws.plot(*start_end_eff, 'go', ms=10, zorder=5, label='start end-effector')
    ax_ws.plot(*goal_end_eff,  'g*', ms=14, zorder=5, label='goal end-effector')
    ax_ws.legend()

    draw_cspace(bitmap, ax=ax_cs, title="C-space — A* path")
    ax_cs.plot(thetas[:, 0], thetas[:, 1], color='cyan', linewidth=1.5, label='path')
    ax_cs.plot(thetas[0, 0],  thetas[0, 1],  'go', ms=8,  label='start')
    ax_cs.plot(thetas[-1, 0], thetas[-1, 1], 'r*', ms=10, label='goal')
    ax_cs.legend(loc='upper right')

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax_ws, ax_cs
