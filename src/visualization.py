import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from kinematics import forward_kinematics, inverse_kinematics, is_point_reachable
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


def _draw_workspace_bg(ws: Workspace, ax, title: str = "Workspace") -> np.ndarray:
    """Draw the static workspace background (bounds, obstacles, shoulder). Returns shoulder xy."""
    xmin, xmax, ymin, ymax = ws.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_facecolor('#dff0d8')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.add_patch(patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor='#555', facecolor='none', zorder=1,
    ))
    for obs in ws.obstacles:
        ax.add_patch(patches.Circle(obs.center_point, obs.radius,
                                    color='salmon', alpha=0.75, zorder=2))
        ax.add_patch(patches.Circle(obs.center_point, obs.radius,
                                    fill=False, edgecolor='firebrick', linewidth=1.5, zorder=2))
    shoulder = np.array(ws.robot.base, dtype=float)
    ax.plot(*shoulder, 'ko', ms=10, zorder=4)
    return shoulder


def animate_path(
    ws: Workspace,
    bitmap: NDArray[np.bool_],
    path: list[tuple[float, float]],
    interval=50
) -> FuncAnimation:
    """
    Animates the arm stepping through path waypoints.
    Left panel: arm moving in workspace. Right panel: C-space with path and moving dot.
    Returns a FuncAnimation. In a notebook, display with HTML(anim.to_jshtml()).
    """
    fig, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(14, 6))

    # static C-space with full path
    thetas = np.array(path)
    draw_cspace(bitmap, ax=ax_cs, title="C-space — A* path")
    ax_cs.plot(thetas[:, 0], thetas[:, 1], color='cyan', lw=1.5, label='path')
    ax_cs.plot(thetas[0, 0],  thetas[0, 1],  'go', ms=8,  label='start')
    ax_cs.plot(thetas[-1, 0], thetas[-1, 1], 'r*', ms=10, label='goal')
    ax_cs.legend(loc='upper right')
    cs_dot, = ax_cs.plot([], [], 'yo', ms=8, zorder=6)

    # static workspace background
    shoulder = _draw_workspace_bg(ws, ax_ws)

    # dynamic arm artists
    link1,     = ax_ws.plot([], [], color='steelblue',      lw=6, solid_capstyle='round', zorder=3)
    link2,     = ax_ws.plot([], [], color='cornflowerblue', lw=4, solid_capstyle='round', zorder=3)
    elbow_dot, = ax_ws.plot([], [], 'ko', ms=7,  zorder=4)
    end_dot,   = ax_ws.plot([], [], 'r*', ms=12, zorder=4)

    def update(frame):
        theta1, theta2 = path[frame]
        elbow, end_eff = forward_kinematics(ws.robot, theta1, theta2)
        link1.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        link2.set_data([elbow[0], end_eff[0]], [elbow[1], end_eff[1]])
        elbow_dot.set_data([elbow[0]], [elbow[1]])
        end_dot.set_data([end_eff[0]], [end_eff[1]])
        cs_dot.set_data([theta1], [theta2])
        return link1, link2, elbow_dot, end_dot, cs_dot

    anim = FuncAnimation(fig, update, frames=len(path), interval=interval, blit=True)
    plt.tight_layout()
    return anim


def interactive_planner(ws: Workspace, bitmap: NDArray[np.bool_], resolution=200, interval=50):
    """
    Click-to-plan interactive demo. Requires %matplotlib widget.
    First click in the workspace sets the start; second sets the goal.
    IK auto-picks the free elbow solution. All three failure modes are reported.
    Single-cycle: does not reset after the animation completes.
    """
    from path_planning import plan_path
    from cspace import config_to_index

    fig, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(14, 6))

    # static C-space
    draw_cspace(bitmap, ax=ax_cs, title="C-space")
    cs_path_line, = ax_cs.plot([], [], color='cyan', lw=1.5, label='path')
    cs_start_dot, = ax_cs.plot([], [], 'go', ms=8,  zorder=6, label='start')
    cs_goal_dot,  = ax_cs.plot([], [], 'r*', ms=10, zorder=6, label='goal')
    cs_dot,       = ax_cs.plot([], [], 'yo', ms=8,  zorder=7)
    ax_cs.legend(loc='upper right')

    # static workspace background
    shoulder = _draw_workspace_bg(ws, ax_ws, title="Click to set start")

    # start arm artists (faded, shown after first click)
    s_link1,   = ax_ws.plot([], [], color='steelblue',      lw=6, alpha=0.3, solid_capstyle='round', zorder=3)
    s_link2,   = ax_ws.plot([], [], color='cornflowerblue', lw=4, alpha=0.3, solid_capstyle='round', zorder=3)
    s_end_dot, = ax_ws.plot([], [], 'go', ms=10, zorder=5)

    # animated arm artists
    link1,     = ax_ws.plot([], [], color='steelblue',      lw=6, solid_capstyle='round', zorder=3)
    link2,     = ax_ws.plot([], [], color='cornflowerblue', lw=4, solid_capstyle='round', zorder=3)
    elbow_dot, = ax_ws.plot([], [], 'ko', ms=7,  zorder=4)
    end_dot,   = ax_ws.plot([], [], 'r*', ms=12, zorder=4)

    # invalid-click markers in workspace: red X for start, orange X for goal
    invalid_start_marker,    = ax_ws.plot([], [], 'rx', ms=12, mew=2.5, zorder=6)
    invalid_goal_marker,     = ax_ws.plot([], [], 'x',  ms=12, mew=2.5, zorder=6, color='orange')
    # matching invalid-click markers in C-space (only shown when IK has solutions)
    cs_invalid_start_marker, = ax_cs.plot([], [], 'rx', ms=10, mew=2.0, zorder=8)
    cs_invalid_goal_marker,  = ax_cs.plot([], [], 'x',  ms=10, mew=2.0, zorder=8, color='orange')

    state = {'clicks': 0, 'start_config': None, 'anim': None}

    def _resolve_config(x, y):
        """Return (config, None, solutions) on success, or (None, error_str, solutions) on failure.
        solutions is the raw IK list (may be empty if point is unreachable)."""
        if not is_point_reachable(ws.robot, x, y):
            return None, "outside arm's reach", []
        solutions = inverse_kinematics(ws.robot, x, y)
        free = [(t1, t2) for t1, t2 in solutions
                if not bitmap[config_to_index(t1, t2, resolution)]]
        if not free:
            return None, "in collision — try a nearby point", solutions
        return free[0], None, solutions

    def on_click(event):
        if event.inaxes is not ax_ws or state['clicks'] >= 2:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        config, err, solutions = _resolve_config(x, y)

        if config is None:
            # show C-space marker at first IK solution if one exists (unreachable has none)
            cs_theta = solutions[0] if solutions else None
            cs_x = [cs_theta[0]] if cs_theta else []
            cs_y = [cs_theta[1]] if cs_theta else []
            if state['clicks'] == 0:
                invalid_start_marker.set_data([x], [y])
                cs_invalid_start_marker.set_data(cs_x, cs_y)
                ax_ws.set_title(f"Invalid start — {err}, choose another point")
                print(f"  ✗ Start ({x:.1f}, {y:.1f}): {err}")
            else:
                invalid_goal_marker.set_data([x], [y])
                cs_invalid_goal_marker.set_data(cs_x, cs_y)
                ax_ws.set_title(f"Invalid goal — {err}, choose another point")
                print(f"  ✗ Goal ({x:.1f}, {y:.1f}): {err}")
            fig.canvas.draw_idle()
            return

        # valid click — clear all error markers
        invalid_start_marker.set_data([], [])
        invalid_goal_marker.set_data([], [])
        cs_invalid_start_marker.set_data([], [])
        cs_invalid_goal_marker.set_data([], [])

        if state['clicks'] == 0:
            state['start_config'] = config
            state['clicks'] = 1
            start_elbow, start_end = forward_kinematics(ws.robot, *config)
            s_link1.set_data([shoulder[0], start_elbow[0]], [shoulder[1], start_elbow[1]])
            s_link2.set_data([start_elbow[0], start_end[0]], [start_elbow[1], start_end[1]])
            s_end_dot.set_data([start_end[0]], [start_end[1]])
            cs_start_dot.set_data([config[0]], [config[1]])
            ax_ws.set_title("Start set — click to set goal")
            fig.canvas.draw_idle()
            print(f"  ✓ Start  θ1={np.degrees(config[0]):.1f}°  θ2={np.degrees(config[1]):.1f}°")

        else:
            goal_config  = config
            start_config = state['start_config']

            cs_goal_dot.set_data([goal_config[0]], [goal_config[1]])
            ax_ws.set_title("Planning...")
            fig.canvas.draw_idle()
            print(f"  ✓ Goal   θ1={np.degrees(goal_config[0]):.1f}°  θ2={np.degrees(goal_config[1]):.1f}°")

            path = plan_path(bitmap, start_config, goal_config, resolution)
            if path is None:
                print("  ✗ No collision-free path — start and goal are in disconnected free regions.")
                ax_ws.set_title("No path found — click a new goal")
                # reset to allow a new goal selection
                cs_goal_dot.set_data([], [])
                fig.canvas.draw_idle()
                return

            state['clicks'] = 2
            thetas = np.array(path)
            cs_path_line.set_data(thetas[:, 0], thetas[:, 1])
            fig.canvas.draw_idle()
            print(f"     {len(path)} waypoints — animating")

            def update(frame):
                theta1, theta2 = path[frame]
                elbow, end_eff = forward_kinematics(ws.robot, theta1, theta2)
                link1.set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
                link2.set_data([elbow[0], end_eff[0]], [elbow[1], end_eff[1]])
                elbow_dot.set_data([elbow[0]], [elbow[1]])
                end_dot.set_data([end_eff[0]], [end_eff[1]])
                cs_dot.set_data([theta1], [theta2])
                if frame == len(path) - 1:
                    ax_ws.set_title("Goal reached!")
                return link1, link2, elbow_dot, end_dot, cs_dot

            anim = FuncAnimation(fig, update, frames=len(path),
                                 interval=interval, blit=False, repeat=False)
            state['anim'] = anim  # prevent garbage collection
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()
