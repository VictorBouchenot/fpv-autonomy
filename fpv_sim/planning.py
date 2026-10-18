"""Deterministic 26-neighbour A* with continuous segment/box collision checks.

Obstacles are expanded by the drone clearance radius (a conservative box
approximation). A* is classical graph search, not a learned model.
"""
from dataclasses import dataclass
import heapq
from itertools import product
import numpy as np


@dataclass(frozen=True)
class Box:
    lower: tuple
    upper: tuple

    def __post_init__(self):
        lo, hi = np.asarray(self.lower), np.asarray(self.upper)
        if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi])) or np.any(lo >= hi):
            raise ValueError('box must have finite increasing 3D bounds')

    def intersects(self, start, end, margin=0.):
        """Closed segment vs expanded closed box, including boundary contact."""
        lo = np.asarray(self.lower) - margin
        hi = np.asarray(self.upper) + margin
        a, b = np.asarray(start), np.asarray(end)
        direction = b - a
        near, far = 0., 1.
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if a[axis] < lo[axis] or a[axis] > hi[axis]:
                    return False
            else:
                t0, t1 = sorted(((lo[axis] - a[axis]) / direction[axis], (hi[axis] - a[axis]) / direction[axis]))
                near, far = max(near, t0), min(far, t1)
                if near > far:
                    return False
        return True


class World:
    def __init__(self, lower, upper, obstacles=(), clearance=0.18):
        self.bounds = Box(tuple(lower), tuple(upper))
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError('clearance must be nonnegative and finite')
        self.obstacles = tuple(obstacles)
        self.clearance = clearance
        if np.any(np.asarray(lower) + clearance >= np.asarray(upper) - clearance):
            raise ValueError('clearance leaves no usable world')

    def free_segment(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite([a, b])):
            return False
        lo, hi = np.asarray(self.bounds.lower) + self.clearance, np.asarray(self.bounds.upper) - self.clearance
        if np.any(np.minimum(a, b) < lo) or np.any(np.maximum(a, b) > hi):
            return False
        return not any(box.intersects(a, b, self.clearance) for box in self.obstacles)


def astar(world, start, goal, resolution=0.25, max_expansions=100000):
    """Shortest path on the connected lattice, with checked endpoint connectors.

    This is not a claim of a globally shortest continuous-space trajectory.
    Raises ValueError for invalid/blocked endpoints or unreachable destinations.
    """
    if not np.isfinite(resolution) or resolution <= 0 or max_expansions <= 0:
        raise ValueError('resolution and expansion budget must be positive')
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if not world.free_segment(start, start) or not world.free_segment(goal, goal):
        raise ValueError('start or goal is outside free space')
    if np.array_equal(start, goal):
        return np.array([start])
    origin = np.asarray(world.bounds.lower) + world.clearance
    shape = np.floor((np.asarray(world.bounds.upper) - world.clearance - origin) / resolution).astype(int) + 1

    def point(key):
        return origin + np.asarray(key) * resolution

    def connector(position):
        base = np.floor((position - origin) / resolution).astype(int)
        candidates = []
        for offset in product([0, 1], repeat=3):
            key = tuple(base + offset)
            if np.all(np.asarray(key) >= 0) and np.all(np.asarray(key) < shape) and world.free_segment(position, point(key)):
                candidates.append(key)
        if not candidates:
            raise ValueError('endpoint has no safe grid connector; use a finer resolution')
        return candidates

    first, last = connector(start), set(connector(goal))
    frontier, distance, parent = [], {}, {}
    for key in first:
        cost = float(np.linalg.norm(point(key) - start))
        distance[key], parent[key] = cost, None
        heapq.heappush(frontier, (cost + np.linalg.norm(point(key) - goal), cost, key))
    offsets = [x for x in product([-1, 0, 1], repeat=3) if x != (0, 0, 0)]
    best, best_cost, expanded = None, float('inf'), 0
    while frontier:
        estimate, cost, key = heapq.heappop(frontier)
        if estimate >= best_cost:
            break
        if cost > distance[key] + 1e-12:
            continue
        expanded += 1
        if expanded > max_expansions:
            raise ValueError('A* expansion budget exceeded')
        if key in last:
            total = cost + np.linalg.norm(point(key) - goal)
            if total < best_cost:
                best, best_cost = key, total
        for offset in offsets:
            neighbour = tuple(np.asarray(key) + offset)
            if np.any(np.asarray(neighbour) < 0) or np.any(np.asarray(neighbour) >= shape):
                continue
            candidate = cost + resolution * np.linalg.norm(offset)
            if candidate >= distance.get(neighbour, float('inf')) - 1e-12:
                continue
            if not world.free_segment(point(key), point(neighbour)):
                continue
            distance[neighbour], parent[neighbour] = candidate, key
            heapq.heappush(frontier, (candidate + np.linalg.norm(point(neighbour) - goal), candidate, neighbour))
    if best is None:
        raise ValueError('no collision-free path found on this grid')
    keys = []
    while best is not None:
        keys.append(point(best))
        best = parent[best]
    path = [start, *reversed(keys), goal]
    return np.array([p for i, p in enumerate(path) if i == 0 or np.linalg.norm(p - path[i-1]) > 1e-10])


def simplify(world, path):
    """Greedy line-of-sight shortcuts, each checked continuously."""
    result, i = [path[0]], 0
    while i < len(path) - 1:
        for j in range(len(path) - 1, i, -1):
            if world.free_segment(path[i], path[j]):
                result.append(path[j])
                i = j
                break
        else:
            raise ValueError('input path contains a blocked segment')
    return np.array(result)
