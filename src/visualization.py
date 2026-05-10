import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray
from kinematics import forward_kinematics, inverse_kinematics, is_point_reachable
from workspace import Workspace
import theme


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------

def _setup_ws_ax(ax, ws: Workspace, title: str) -> None:
    """Apply the metallic-workshop style and draw the static background."""
    xmin, xmax, ymin, ymax = ws.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_facecolor(theme.BG)
    ax.set_title(title, color=theme.FRAME_EDGE)
    ax.set_xlabel('x', color=theme.FRAME_EDGE)
    ax.set_ylabel('y', color=theme.FRAME_EDGE)
    ax.tick_params(colors=theme.FRAME_EDGE)
    for spine in ax.spines.values():
        spine.set_edgecolor(theme.FRAME_EDGE)
    ax.grid(color=theme.GRID, alpha=0.45, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.add_patch(patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        linewidth=2, edgecolor=theme.FRAME_EDGE, facecolor='none', zorder=1,
    ))
    for obs in ws.obstacles:
        obs.draw(ax)


def _draw_link(ax, p1, p2, *, lw_base, lw_hi, color_base, color_hi, alpha=1.0, zorder=3):
    """Two stacked lines for a brushed-metal look. Returns (base, hi) Line2D pair."""
    xs, ys = [p1[0], p2[0]], [p1[1], p2[1]]
    base = ax.plot(xs, ys, color=color_base, lw=lw_base, solid_capstyle='round', alpha=alpha, zorder=zorder)[0]
    hi   = ax.plot(xs, ys, color=color_hi,   lw=lw_hi,   solid_capstyle='round', alpha=alpha, zorder=zorder + 0.1)[0]
    return base, hi


def _draw_joint(ax, xy, ms: int) -> None:
    """Bolt-head style joint marker (dark ring, metallic face)."""
    ax.plot(*xy, 'o', mfc=theme.JOINT_INNER, mec=theme.JOINT_OUTER, mew=2.5, ms=ms, zorder=4, linestyle='none')


def _draw_end_effector_at(ax, xy) -> None:
    """Glowing red LED at xy (static)."""
    ax.plot(*xy, 'o', color=theme.END_EFFECTOR_GLOW, ms=18, alpha=0.4, zorder=5, linestyle='none')
    ax.plot(*xy, 'o', color=theme.END_EFFECTOR, ms=9, zorder=6, linestyle='none')


def _draw_workspace_bg(ws: Workspace, ax, title: str = "Workspace") -> np.ndarray:
    """Draw the static workspace background and shoulder joint. Returns shoulder xy."""
    _setup_ws_ax(ax, ws, title)
    shoulder = np.array(ws.robot.base, dtype=float)
    _draw_joint(ax, shoulder, theme.JOINT_MS_SHOULDER)
    return shoulder


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def draw_workspace(ws: Workspace, theta1: float = 0.0, theta2: float = 0.0, ax=None, title="Workspace"):
    """
    Renders the workspace: dark workshop background, metallic obstacles, and
    the arm at (theta1, theta2). Pass an existing ax to embed in a larger figure.
    Returns the axes object.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(7, 7))

    _setup_ws_ax(ax, ws, title)

    shoulder = ws.robot.base
    elbow, end_effector = forward_kinematics(ws.robot, theta1, theta2)

    _draw_link(ax, shoulder, elbow,
               lw_base=theme.LINK1_LW, lw_hi=theme.LINK1_LW_HI,
               color_base=theme.LINK1_COLOR, color_hi=theme.LINK1_HI)
    _draw_link(ax, elbow, end_effector,
               lw_base=theme.LINK2_LW, lw_hi=theme.LINK2_LW_HI,
               color_base=theme.LINK2_COLOR, color_hi=theme.LINK2_HI)

    _draw_joint(ax, shoulder, theme.JOINT_MS_SHOULDER)
    _draw_joint(ax, elbow,    theme.JOINT_MS_ELBOW)
    _draw_end_effector_at(ax, end_effector)

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax


def draw_cspace(bitmap: NDArray[np.bool_], ax=None, title="C-space"):
    """
    Renders the C-space bitmap. Black = in-collision, white = free.
    theta1 on x-axis (∈ [0, π]), theta2 on y-axis (∈ [-π, π]).
    Pass an existing ax to embed in a dual-panel figure.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        bitmap.T,
        origin='lower',
        extent=[0, np.pi, -np.pi, np.pi],
        cmap='gray_r',      # black = collision, white = free
        aspect='auto',
        interpolation='nearest'
    )

    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
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
    Left: workspace with start arm (faded) and goal arm (full color).
    Right: C-space bitmap with the A* path overlaid.
    Returns (ax_ws, ax_cs).
    """
    standalone = ax_ws is None and ax_cs is None
    if standalone:
        _, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(14, 6))

    thetas = np.array(path)
    _, start_end_eff = forward_kinematics(ws.robot, *start_config)
    _, goal_end_eff  = forward_kinematics(ws.robot, *goal_config)

    # goal arm at full brightness
    draw_workspace(ws, theta1=goal_config[0], theta2=goal_config[1],
                   ax=ax_ws, title="Workspace — start & goal")

    # faded start arm overlay
    shoulder = ws.robot.base
    start_elbow, _ = forward_kinematics(ws.robot, *start_config)
    _draw_link(ax_ws, shoulder, start_elbow,
               lw_base=theme.LINK1_LW, lw_hi=theme.LINK1_LW_HI,
               color_base=theme.LINK1_COLOR, color_hi=theme.LINK1_HI, alpha=0.35)
    _draw_link(ax_ws, start_elbow, start_end_eff,
               lw_base=theme.LINK2_LW, lw_hi=theme.LINK2_LW_HI,
               color_base=theme.LINK2_COLOR, color_hi=theme.LINK2_HI, alpha=0.35)

    ax_ws.plot(*start_end_eff, 'o', color=theme.START_MARKER, ms=10,
               zorder=5, label='start', linestyle='none')
    ax_ws.plot(*goal_end_eff,  '*', color=theme.GOAL_MARKER,  ms=14,
               zorder=5, label='goal', linestyle='none')
    ax_ws.legend(facecolor=theme.BG, edgecolor=theme.FRAME_EDGE,
                 labelcolor=theme.FRAME_EDGE)

    draw_cspace(bitmap, ax=ax_cs, title="C-space — A* path")
    ax_cs.plot(thetas[:, 0], thetas[:, 1], color=theme.PATH_TRACE, linewidth=1.5, label='path')
    ax_cs.plot(thetas[0, 0],  thetas[0, 1],  'o', color=theme.START_MARKER, ms=8,  label='start')
    ax_cs.plot(thetas[-1, 0], thetas[-1, 1], '*', color=theme.GOAL_MARKER,  ms=10, label='goal')
    ax_cs.legend(loc='upper right')

    if standalone:
        plt.tight_layout()
        plt.show()

    return ax_ws, ax_cs


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
    ax_cs.plot(thetas[:, 0], thetas[:, 1], color=theme.PATH_TRACE, lw=1.5, label='path')
    ax_cs.plot(thetas[0, 0],  thetas[0, 1],  'o', color=theme.START_MARKER, ms=8,  label='start')
    ax_cs.plot(thetas[-1, 0], thetas[-1, 1], '*', color=theme.GOAL_MARKER,  ms=10, label='goal')
    ax_cs.legend(loc='upper right')
    cs_dot = ax_cs.plot([], [], 'o', color=theme.TRACKER_DOT, ms=8, zorder=6)[0]

    # static workspace background (shoulder drawn here)
    shoulder = _draw_workspace_bg(ws, ax_ws)

    # animated arm artists (start with empty data)
    link1_base = ax_ws.plot([], [], color=theme.LINK1_COLOR, lw=theme.LINK1_LW,    solid_capstyle='round', zorder=3)[0]
    link1_hi   = ax_ws.plot([], [], color=theme.LINK1_HI,    lw=theme.LINK1_LW_HI, solid_capstyle='round', zorder=3.1)[0]
    link2_base = ax_ws.plot([], [], color=theme.LINK2_COLOR, lw=theme.LINK2_LW,    solid_capstyle='round', zorder=3)[0]
    link2_hi   = ax_ws.plot([], [], color=theme.LINK2_HI,    lw=theme.LINK2_LW_HI, solid_capstyle='round', zorder=3.1)[0]
    elbow_dot  = ax_ws.plot([], [], 'o', mfc=theme.JOINT_INNER, mec=theme.JOINT_OUTER, mew=2.5, ms=theme.JOINT_MS_ELBOW, zorder=4, linestyle='none')[0]
    end_halo   = ax_ws.plot([], [], 'o', color=theme.END_EFFECTOR_GLOW, ms=18, alpha=0.4, zorder=5, linestyle='none')[0]
    end_dot    = ax_ws.plot([], [], 'o', color=theme.END_EFFECTOR, ms=9, zorder=6, linestyle='none')[0]

    def update(frame):
        theta1, theta2 = path[frame]
        elbow, end_eff = forward_kinematics(ws.robot, theta1, theta2)
        d1 = ([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
        d2 = ([elbow[0], end_eff[0]], [elbow[1], end_eff[1]])
        link1_base.set_data(*d1);  link1_hi.set_data(*d1)
        link2_base.set_data(*d2);  link2_hi.set_data(*d2)
        elbow_dot.set_data([elbow[0]], [elbow[1]])
        end_halo.set_data([end_eff[0]], [end_eff[1]])
        end_dot.set_data([end_eff[0]], [end_eff[1]])
        cs_dot.set_data([theta1], [theta2])
        return link1_base, link1_hi, link2_base, link2_hi, elbow_dot, end_halo, end_dot, cs_dot

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
    cs_path_line = ax_cs.plot([], [], color=theme.PATH_TRACE, lw=1.5, label='path')[0]
    cs_start_dot = ax_cs.plot([], [], 'o', color=theme.START_MARKER, ms=8,  zorder=6, label='start')[0]
    cs_goal_dot  = ax_cs.plot([], [], '*', color=theme.GOAL_MARKER,  ms=10, zorder=6, label='goal')[0]
    cs_dot       = ax_cs.plot([], [], 'o', color=theme.TRACKER_DOT,  ms=8,  zorder=7)[0]
    ax_cs.legend(loc='upper right')

    # static workspace background
    shoulder = _draw_workspace_bg(ws, ax_ws, title="Click to set start")

    # faded start arm artists
    s_link1_base = ax_ws.plot([], [], color=theme.LINK1_COLOR, lw=theme.LINK1_LW,    solid_capstyle='round', alpha=0.35, zorder=3)[0]
    s_link1_hi   = ax_ws.plot([], [], color=theme.LINK1_HI,    lw=theme.LINK1_LW_HI, solid_capstyle='round', alpha=0.35, zorder=3.1)[0]
    s_link2_base = ax_ws.plot([], [], color=theme.LINK2_COLOR, lw=theme.LINK2_LW,    solid_capstyle='round', alpha=0.35, zorder=3)[0]
    s_link2_hi   = ax_ws.plot([], [], color=theme.LINK2_HI,    lw=theme.LINK2_LW_HI, solid_capstyle='round', alpha=0.35, zorder=3.1)[0]
    s_end_dot    = ax_ws.plot([], [], 'o', color=theme.START_MARKER, ms=10, zorder=5, linestyle='none')[0]

    # animated arm artists
    link1_base = ax_ws.plot([], [], color=theme.LINK1_COLOR, lw=theme.LINK1_LW,    solid_capstyle='round', zorder=3)[0]
    link1_hi   = ax_ws.plot([], [], color=theme.LINK1_HI,    lw=theme.LINK1_LW_HI, solid_capstyle='round', zorder=3.1)[0]
    link2_base = ax_ws.plot([], [], color=theme.LINK2_COLOR, lw=theme.LINK2_LW,    solid_capstyle='round', zorder=3)[0]
    link2_hi   = ax_ws.plot([], [], color=theme.LINK2_HI,    lw=theme.LINK2_LW_HI, solid_capstyle='round', zorder=3.1)[0]
    elbow_dot  = ax_ws.plot([], [], 'o', mfc=theme.JOINT_INNER, mec=theme.JOINT_OUTER, mew=2.5, ms=theme.JOINT_MS_ELBOW, zorder=4, linestyle='none')[0]
    end_halo   = ax_ws.plot([], [], 'o', color=theme.END_EFFECTOR_GLOW, ms=18, alpha=0.4, zorder=5, linestyle='none')[0]
    end_dot    = ax_ws.plot([], [], 'o', color=theme.END_EFFECTOR, ms=9, zorder=6, linestyle='none')[0]

    # invalid-click markers
    invalid_start_marker    = ax_ws.plot([], [], 'x', color='#ff453a', ms=12, mew=2.5, zorder=6)[0]
    invalid_goal_marker     = ax_ws.plot([], [], 'x', color='#ff9f0a', ms=12, mew=2.5, zorder=6)[0]
    cs_invalid_start_marker = ax_cs.plot([], [], 'x', color='#ff453a', ms=10, mew=2.0, zorder=8)[0]
    cs_invalid_goal_marker  = ax_cs.plot([], [], 'x', color='#ff9f0a', ms=10, mew=2.0, zorder=8)[0]

    state = {'clicks': 0, 'start_config': None, 'anim': None}

    def _resolve_config(x, y):
        if not is_point_reachable(ws.robot, x, y):
            return None, "outside arm's reach", []
        solutions = inverse_kinematics(ws.robot, x, y)
        # Filter to solutions whose theta1 is in the planner's range [0, π]
        # (negative theta1 puts the elbow below the workspace floor)
        in_range = [(t1, t2) for t1, t2 in solutions if 0 <= t1 <= np.pi]
        free = [(t1, t2) for t1, t2 in in_range
                if not bitmap[config_to_index(t1, t2, resolution)]]
        if not free:
            return None, "in collision", in_range or solutions
        return free[0], None, solutions

    def on_click(event):
        if event.inaxes is not ax_ws or state['clicks'] >= 2:
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        config, err, solutions = _resolve_config(x, y)

        if config is None:
            cs_theta = solutions[0] if solutions else None
            cs_x = [cs_theta[0]] if cs_theta else []
            cs_y = [cs_theta[1]] if cs_theta else []
            if state['clicks'] == 0:
                invalid_start_marker.set_data([x], [y])
                cs_invalid_start_marker.set_data(cs_x, cs_y)
                ax_ws.set_title(f"Invalid start — {err} — choose another point", color=theme.FRAME_EDGE)
                print(f"  ✗ Start ({x:.1f}, {y:.1f}): {err}")
            else:
                invalid_goal_marker.set_data([x], [y])
                cs_invalid_goal_marker.set_data(cs_x, cs_y)
                ax_ws.set_title(f"Invalid goal — {err} — choose another point", color=theme.FRAME_EDGE)
                print(f"  ✗ Goal ({x:.1f}, {y:.1f}): {err}")
            fig.canvas.draw_idle()
            return

        invalid_start_marker.set_data([], [])
        invalid_goal_marker.set_data([], [])
        cs_invalid_start_marker.set_data([], [])
        cs_invalid_goal_marker.set_data([], [])

        if state['clicks'] == 0:
            state['start_config'] = config
            state['clicks'] = 1
            start_elbow, start_end = forward_kinematics(ws.robot, *config)
            d1 = ([shoulder[0], start_elbow[0]], [shoulder[1], start_elbow[1]])
            d2 = ([start_elbow[0], start_end[0]], [start_elbow[1], start_end[1]])
            s_link1_base.set_data(*d1);  s_link1_hi.set_data(*d1)
            s_link2_base.set_data(*d2);  s_link2_hi.set_data(*d2)
            s_end_dot.set_data([start_end[0]], [start_end[1]])
            cs_start_dot.set_data([config[0]], [config[1]])
            ax_ws.set_title("Start set — click to set goal", color=theme.FRAME_EDGE)
            fig.canvas.draw_idle()
            print(f"  ✓ Start  θ1={np.degrees(config[0]):.1f}°  θ2={np.degrees(config[1]):.1f}°")

        else:
            goal_config  = config
            start_config = state['start_config']

            cs_goal_dot.set_data([goal_config[0]], [goal_config[1]])
            ax_ws.set_title("Planning...", color=theme.FRAME_EDGE)
            fig.canvas.draw_idle()
            print(f"  ✓ Goal   θ1={np.degrees(goal_config[0]):.1f}°  θ2={np.degrees(goal_config[1]):.1f}°")

            try:
                path = plan_path(bitmap, start_config, goal_config, resolution)
            except ValueError as e:
                print(f"  ✗ {e}")
                ax_ws.set_title(f"Planning failed — {e}", color=theme.FRAME_EDGE)
                cs_goal_dot.set_data([], [])
                fig.canvas.draw_idle()
                return
            if path is None:
                print("  ✗ No collision-free path — start and goal are in disconnected free regions.")
                ax_ws.set_title("No path found — click a new goal", color=theme.FRAME_EDGE)
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
                d1 = ([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
                d2 = ([elbow[0], end_eff[0]], [elbow[1], end_eff[1]])
                link1_base.set_data(*d1);  link1_hi.set_data(*d1)
                link2_base.set_data(*d2);  link2_hi.set_data(*d2)
                elbow_dot.set_data([elbow[0]], [elbow[1]])
                end_halo.set_data([end_eff[0]], [end_eff[1]])
                end_dot.set_data([end_eff[0]], [end_eff[1]])
                cs_dot.set_data([theta1], [theta2])
                if frame == len(path) - 1:
                    ax_ws.set_title("Goal reached!", color=theme.FRAME_EDGE)
                return (link1_base, link1_hi, link2_base, link2_hi,
                        elbow_dot, end_halo, end_dot, cs_dot)

            anim = FuncAnimation(fig, update, frames=len(path),
                                 interval=interval, blit=False, repeat=False)
            state['anim'] = anim
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()
