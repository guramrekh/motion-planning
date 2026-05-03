import heapq


def a_star(start, goal, neighbors_fn, heuristic_fn, cost_fn):
    """
    Generic A* search over an implicit graph.

    Returns the shortest path from start to goal as a list [start, ..., goal],
    or None if goal is unreachable.

    Parameters:
        neighbors_fn(node)        -> iterable of neighbor nodes
        heuristic_fn(node, goal)  -> admissible estimate from node to goal
        cost_fn(a, b)             -> edge cost between adjacent nodes

    Assumes the heuristic is consistent (monotone).
    """
    if start == goal:
        return [start]

    g_score = {start: 0.0}
    came_from = {}
    closed = set()

    counter = 0  # tiebreaker so heap never compares raw nodes
    open_heap = [(heuristic_fn(start, goal), counter, start)]

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score[current]
        for neighbor in neighbors_fn(current):
            if neighbor in closed:
                continue
            tentative_g = current_g + cost_fn(current, neighbor)
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic_fn(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return None
