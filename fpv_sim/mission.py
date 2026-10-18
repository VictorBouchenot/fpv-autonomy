"""Plan and fly checked waypoint missions in conservative or dynamic profiles."""
from dataclasses import dataclass, replace
import numpy as np
from .physics import Quadrotor, State
from .control import FlightController
from .planning import Box, World, astar, simplify


@dataclass(frozen=True)
class FlightProfile:
    """Reference-motion and acceptance settings for one validated flight style.

    The profile does not fake attitude: the same rigid-body model and controller
    produce the different speed and banking response.
    """

    name: str
    reference_speed: float
    planning_buffer: float
    stop_at_corners: bool
    arrival_tolerance: float
    arrival_speed: float

    def __post_init__(self):
        values = [
            self.reference_speed,
            self.planning_buffer,
            self.arrival_tolerance,
            self.arrival_speed,
        ]
        if not self.name or not all(np.isfinite(values)) or any(value <= 0 for value in values):
            raise ValueError('profile values must be named, positive and finite')


CONSERVATIVE = FlightProfile(
    name='conservative',
    reference_speed=.25,
    planning_buffer=.35,
    stop_at_corners=True,
    arrival_tolerance=.06,
    arrival_speed=.08,
)

# On the 3D gate course, 2.15 m/s collides. Keep 1.85 m/s: faster and more
# banked than the first dynamic preview, with checked clearance remaining.
DYNAMIC = FlightProfile(
    name='dynamic',
    reference_speed=1.85,
    planning_buffer=.35,
    stop_at_corners=False,
    arrival_tolerance=.08,
    arrival_speed=.25,
)

PROFILES = {profile.name: profile for profile in (CONSERVATIVE, DYNAMIC)}
SCENARIOS = ('demo', 'reverse', 'walls')


@dataclass
class MissionResult:
    times: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    motors: np.ndarray
    targets: np.ndarray
    path: np.ndarray
    arrived: bool
    collision: bool
    profile: str
    final_error: float
    max_tracking_error: float
    max_speed: float
    max_tilt_degrees: float
    min_point_clearance: float


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ', '.join(sorted(PROFILES))
        raise ValueError(f'unknown flight mode {name!r}; choose {choices}') from exc


def demo_world(scenario='demo'):
    """Return a named synthetic course, its start and its goal.

    `demo` keeps the same two wall stations as the first preview, but replaces
    each solid slab with four panels around a rectangular opening. The route
    must climb through the high gate, descend through the low gate, then climb
    to the goal. `walls` preserves the original solid-wall course for comparison.
    """
    if scenario == 'walls':
        world = World([0, 0, 0], [6, 5, 3.5], [
            Box((2, .25, 0), (2.6, 3.1, 3.5)),
            Box((4.0, 4.0, 0), (4.4, 4.75, 3.5)),
        ], clearance=.18)
        return world, np.array([.65, 1., 1.]), np.array([5.3, 3.2, 1.3])

    world = World([0, 0, 0], [6, 5, 3.5], [
        # First station: a 1.8 m x 1.5 m high window in the original wall.
        Box((2.0, .25, 0), (2.6, 3.1, 1.5)),
        Box((2.0, .25, 3.0), (2.6, 3.1, 3.5)),
        Box((2.0, .25, 1.5), (2.6, .8, 3.0)),
        Box((2.0, 2.6, 1.5), (2.6, 3.1, 3.0)),
        # Second station: a 1.8 m x 1.4 m low window across the course.
        Box((4.0, .8, 0), (4.4, 4.8, .5)),
        Box((4.0, .8, 1.9), (4.4, 4.8, 3.5)),
        Box((4.0, .8, .5), (4.4, 2.0, 1.9)),
        Box((4.0, 3.8, .5), (4.4, 4.8, 1.9)),
    ], clearance=.18)
    if scenario == 'demo':
        return world, np.array([.65, 1., .8]), np.array([5.3, 3.2, 2.2])
    if scenario == 'reverse':
        return world, np.array([5.3, 3.2, 2.2]), np.array([.65, 1., .8])
    choices = ', '.join(repr(name) for name in SCENARIOS)
    raise ValueError(f'unknown scenario; choose {choices}')


def point_clearance(world, point):
    """Signed distance from a point to the nearest world/obstacle boundary."""
    point = np.asarray(point, dtype=float)
    lower, upper = np.asarray(world.bounds.lower), np.asarray(world.bounds.upper)
    margins = [float(np.min(point - lower)), float(np.min(upper - point))]
    for box in world.obstacles:
        lo, hi = np.asarray(box.lower), np.asarray(box.upper)
        outside = np.maximum.reduce([lo - point, np.zeros(3), point - hi])
        if np.any(outside > 0):
            margins.append(float(np.linalg.norm(outside)))
        else:
            margins.append(float(-np.min(np.minimum(point - lo, hi - point))))
    return min(margins)


def _metrics(result_values):
    times, positions, rotations, motors, targets, path, arrived, collision, profile = result_values
    velocity_steps = np.diff(positions, axis=0) / np.diff(times)[:, None]
    max_speed = float(np.max(np.linalg.norm(velocity_steps, axis=1))) if len(velocity_steps) else 0.
    tilt_cosine = np.clip(rotations[:, 2, 2], -1, 1)
    return {
        'final_error': float(np.linalg.norm(positions[-1] - path[-1])),
        'max_tracking_error': float(np.max(np.linalg.norm(positions - targets, axis=1))),
        'max_speed': max_speed,
        'max_tilt_degrees': float(np.max(np.degrees(np.arccos(tilt_cosine)))),
        'min_point_clearance': float('inf'),  # replaced by fly(), which knows the world
    }


def fly(world, start, goal, dt=.005, duration=90., resolution=.25, profile=CONSERVATIVE):
    """Simulate one mission and check every actual position segment.

    A collision or timeout is returned as data, never converted into a success.
    """
    if isinstance(profile, str):
        profile = get_profile(profile)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    if not np.isfinite(dt) or not 0 < dt <= .02:
        raise ValueError('dt must be in (0, 0.02]')
    planning_world = World(
        world.bounds.lower,
        world.bounds.upper,
        world.obstacles,
        clearance=world.clearance + profile.planning_buffer,
    )
    path = simplify(planning_world, astar(planning_world, start, goal, resolution))
    q = Quadrotor(state=State(position=np.asarray(start, dtype=float).copy()))
    controller = FlightController(q)
    index, arrived, collision = min(1, len(path)-1), False, False
    reference = path[0].copy()
    times = [0.]
    positions = [q.state.position.copy()]
    rotations = [q.state.rotation.copy()]
    motors_log = [np.zeros(4)]
    targets = [reference.copy()]

    for step in range(round(duration / dt)):
        waypoint = path[index]
        delta = waypoint - reference
        distance = np.linalg.norm(delta)
        direction = delta / distance if distance > 1e-12 else np.zeros(3)
        advance = min(distance, profile.reference_speed * dt)
        reference = reference + direction * advance
        target_velocity = np.zeros(3) if profile.stop_at_corners else direction * (advance / dt)
        reached_reference_waypoint = distance <= advance + 1e-12

        old = q.state.position.copy()
        motors = controller.command(reference, dt, target_velocity=target_velocity)
        q.step(motors, dt)
        times.append((step+1)*dt)
        positions.append(q.state.position.copy())
        rotations.append(q.state.rotation.copy())
        motors_log.append(motors)
        targets.append(reference.copy())

        if not world.free_segment(old, q.state.position):
            collision = True
            break

        actual_error = np.linalg.norm(q.state.position - waypoint)
        actual_speed = np.linalg.norm(q.state.velocity)
        if profile.stop_at_corners:
            ready = (
                actual_error < profile.arrival_tolerance
                and actual_speed < profile.arrival_speed
                and distance < .001
            )
            if ready:
                if index == len(path)-1:
                    arrived = True
                    break
                index += 1
        else:
            if reached_reference_waypoint and index < len(path)-1:
                index += 1
            elif (
                reached_reference_waypoint
                and index == len(path)-1
                and actual_error < profile.arrival_tolerance
                and actual_speed < profile.arrival_speed
            ):
                arrived = True
                break

    arrays = [np.array(x) for x in [times, positions, rotations, motors_log, targets]]
    metrics = _metrics([*arrays, path, arrived, collision, profile.name])
    metrics['min_point_clearance'] = min(point_clearance(world, point) for point in arrays[1])
    return MissionResult(*arrays, path, arrived, collision, profile.name, **metrics)
