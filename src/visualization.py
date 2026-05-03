import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.axes import Axes
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
