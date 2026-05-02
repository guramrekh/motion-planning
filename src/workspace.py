class Robot:
    def __init__(self, base: tuple, link_lengths: tuple):
        self.base = base
        self.link_lengths = link_lengths

class Obstacle:
    def __init__(self, center_point: tuple, radius: float):
        self.center_point = center_point
        self.radius = radius


class Workspace:
    def __init__(self, robot: Robot, obstacles: list, bounds: tuple):
        self.robot = robot
        self.obstacles = obstacles
        self.bounds = bounds  # (xmin, xmax, ymin, ymax)