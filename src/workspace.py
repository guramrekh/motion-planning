from abc import ABC, abstractmethod
import numpy as np
import matplotlib.patches as patches
import theme


class Robot:
    def __init__(self, base: tuple[float, float], link_lengths: tuple[float, float]):
        self.base = base
        self.link_lengths = link_lengths


class Obstacle(ABC):
    @abstractmethod
    def intersects_segment(self, p1, p2): ...

    @abstractmethod
    def intersects_segment_grid(self, p1_x, p1_y, p2_x, p2_y): ...

    @abstractmethod
    def draw(self, ax): ...


class CircleObstacle(Obstacle):
    def __init__(self, center_point: tuple[float, float], radius: float):
        self.center_point = center_point
        self.radius = radius

    def intersects_segment(self, p1, p2) -> bool:
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        center = np.asarray(self.center_point, dtype=float)
        line_vec = p2 - p1
        p1_to_center = center - p1
        len_sq = np.dot(line_vec, line_vec)
        if len_sq == 0.0:
            return bool(np.linalg.norm(p1_to_center) <= self.radius)
        t = np.clip(np.dot(p1_to_center, line_vec) / len_sq, 0.0, 1.0)
        closest = p1 + t * line_vec
        return bool(np.linalg.norm(center - closest) <= self.radius)

    def intersects_segment_grid(self, p1_x, p1_y, p2_x, p2_y):
        cx, cy = self.center_point
        dx = p2_x - p1_x
        dy = p2_y - p1_y
        vx = cx - p1_x
        vy = cy - p1_y
        t = np.clip((vx * dx + vy * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        closest_x = p1_x + t * dx
        closest_y = p1_y + t * dy
        return (cx - closest_x) ** 2 + (cy - closest_y) ** 2 <= self.radius ** 2

    def draw(self, ax) -> None:
        ax.add_patch(patches.Circle(self.center_point, self.radius,
                                    color=theme.OBSTACLE_FILL, alpha=0.92, zorder=2))
        ax.add_patch(patches.Circle(self.center_point, self.radius,
                                    fill=False, edgecolor=theme.OBSTACLE_EDGE,
                                    linewidth=1.6, zorder=2))


class PolygonObstacle(Obstacle):
    def __init__(self, vertices):
        verts = np.asarray(vertices, dtype=float)  # (N, 2), CCW order
        self.vertices = verts
        self._e1 = verts
        self._e2 = np.roll(verts, -1, axis=0)

    @classmethod
    def regular(cls, center: tuple[float, float], radius: float, num_vertices: int, rotation=0.0):
        cx, cy = center
        angles = rotation + np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
        vertices = list(zip(cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
        return cls(vertices)

    @classmethod
    def rectangle(cls, center: tuple[float, float], width: float, height: float, rotation=0.0):
        cx, cy = center
        hw, hh = width / 2, height / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]  # CCW from bottom-left
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        vertices = [
            (cx + cos_r * x - sin_r * y, cy + sin_r * x + cos_r * y)
            for x, y in corners
        ]
        return cls(vertices)

    def intersects_segment(self, p1, p2):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        for e1, e2 in zip(self._e1, self._e2):
            if _segs_cross(p1, p2, e1, e2):
                return True
        if _point_in_convex_poly(p1, self._e1, self._e2):
            return True
        if _point_in_convex_poly(p2, self._e1, self._e2):
            return True
        return False

    def intersects_segment_grid(self, p1_x, p1_y, p2_x, p2_y):
        result = np.zeros_like(p2_x, dtype=bool)
        for e1, e2 in zip(self._e1, self._e2):
            result |= _segs_cross_grid(p1_x, p1_y, p2_x, p2_y,
                                       e1[0], e1[1], e2[0], e2[1])
        result |= _point_in_convex_poly_grid(p1_x, p1_y, self._e1, self._e2)
        result |= _point_in_convex_poly_grid(p2_x, p2_y, self._e1, self._e2)
        return result

    def draw(self, ax) -> None:
        xy = self.vertices
        ax.add_patch(patches.Polygon(xy, closed=True,
                                     color=theme.OBSTACLE_FILL, alpha=0.92, zorder=2))
        ax.add_patch(patches.Polygon(xy, closed=True,
                                     fill=False, edgecolor=theme.OBSTACLE_EDGE,
                                     linewidth=1.6, zorder=2))


def _segs_cross(p1, p2, e1, e2) -> bool:
    def _cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    c1 = _cross(p1, p2, e1)
    c2 = _cross(p1, p2, e2)
    c3 = _cross(e1, e2, p1)
    c4 = _cross(e1, e2, p2)
    return c1 * c2 <= 0 and c3 * c4 <= 0


def _point_in_convex_poly(pt, e1s, e2s) -> bool:
    for e1, e2 in zip(e1s, e2s):
        cross = (e2[0] - e1[0]) * (pt[1] - e1[1]) - (e2[1] - e1[1]) * (pt[0] - e1[0])
        if cross < 0:
            return False
    return True


def _segs_cross_grid(p1x, p1y, p2x, p2y, e1x, e1y, e2x, e2y):
    """Vectorized: True where segment p1→p2 crosses segment e1→e2.
    p1 may be scalar; p2 must be an ndarray. Returns same shape as p2x."""
    c1 = (p2x - p1x) * (e1y - p1y) - (p2y - p1y) * (e1x - p1x)
    c2 = (p2x - p1x) * (e2y - p1y) - (p2y - p1y) * (e2x - p1x)
    c3 = (e2x - e1x) * (p1y - e1y) - (e2y - e1y) * (p1x - e1x)
    c4 = (e2x - e1x) * (p2y - e1y) - (e2y - e1y) * (p2x - e1x)
    return (c1 * c2 <= 0) & (c3 * c4 <= 0)


def _point_in_convex_poly_grid(px, py, e1s, e2s):
    """Vectorized point-in-convex-polygon (CCW). px/py may be scalar or ndarray."""
    inside = True
    for e1, e2 in zip(e1s, e2s):
        cross = (e2[0] - e1[0]) * (py - e1[1]) - (e2[1] - e1[1]) * (px - e1[0])
        inside = inside & (cross >= 0)
    return inside


class BoundaryObstacle(Obstacle):
    """Workspace boundary walls — flags any arm segment whose endpoint leaves the rectangle."""
    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float):
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax

    def _outside(self, x, y):
        return (x < self.xmin) | (x > self.xmax) | (y < self.ymin) | (y > self.ymax)

    def intersects_segment(self, p1, p2) -> bool:
        return bool(self._outside(p1[0], p1[1]) or self._outside(p2[0], p2[1]))

    def intersects_segment_grid(self, p1_x, p1_y, p2_x, p2_y):
        return self._outside(p1_x, p1_y) | self._outside(p2_x, p2_y)

    def draw(self, ax) -> None:
        pass  # frame already rendered by _setup_ws_ax


class Workspace:
    def __init__(self, robot: Robot, obstacles: list[Obstacle], bounds: tuple[float, float, float, float], closed: bool = False):
        self.robot = robot
        self.obstacles = list(obstacles)
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)
        self.closed = closed
        if closed:
            self.obstacles.append(BoundaryObstacle(*bounds))