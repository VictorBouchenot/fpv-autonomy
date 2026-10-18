"""Z-up rigid-body dynamics with body-frame angular velocity.

Generic toy parameters, not measured Aquila 20 specifications. Motors apply
positive body-z thrust. World acceleration includes gravity and linear drag.
"""
from dataclasses import dataclass, field
import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_vector(v):
    """SO(3) exponential map; continuous near zero."""
    angle = float(np.linalg.norm(v))
    k = skew(v)
    if angle < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + np.sin(angle) / angle * k + (1 - np.cos(angle)) / angle**2 * k @ k


@dataclass(frozen=True)
class Parameters:
    mass: float = 0.1
    gravity: float = 9.81
    arm: float = 0.06
    inertia: tuple = (0.00025, 0.00025, 0.00045)
    max_thrust: float = 0.6
    yaw_coefficient: float = 0.01
    linear_drag: float = 0.05
    angular_drag: float = 0.0001

    def __post_init__(self):
        positive = [self.mass, self.gravity, self.arm, self.max_thrust, self.yaw_coefficient, *self.inertia]
        if any(not np.isfinite(x) or x <= 0 for x in positive):
            raise ValueError('physical parameters must be positive and finite')
        if any(not np.isfinite(x) or x < 0 for x in [self.linear_drag, self.angular_drag]):
            raise ValueError('drag must be nonnegative and finite')


@dataclass
class State:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self):
        return State(*(a.copy() for a in [self.position, self.velocity, self.rotation, self.omega]))


class Quadrotor:
    def __init__(self, params=None, state=None):
        self.params = params or Parameters()
        self.state = state.copy() if state else State()
        p = self.params
        a = p.arm / np.sqrt(2)
        # Four X-frame arms; torque = arm cross thrust. Alternate spin directions.
        self.motor_positions = np.array([[a, a, 0], [-a, a, 0], [-a, -a, 0], [a, -a, 0]])
        self.allocation = np.array([
            [1, 1, 1, 1],
            self.motor_positions[:, 1],
            -self.motor_positions[:, 0],
            p.yaw_coefficient * np.array([1, -1, 1, -1]),
        ])
        self.inverse_allocation = np.linalg.inv(self.allocation)

    def mix(self, thrust, torque):
        """Return saturated motor thrusts in N; saturation changes the wrench."""
        command = np.r_[thrust, torque]
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError('wrench must contain four finite values')
        return np.clip(self.inverse_allocation @ command, 0, self.params.max_thrust)

    def step(self, motors, dt, disturbance=None):
        """Semi-implicit Euler at small dt; attitude advanced on SO(3).

        No ground/contact model: callers must detect collisions. Disturbance is
        a world-frame external force in N, not a velocity shortcut.
        """
        if not np.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError('dt must be in (0, 0.02] seconds')
        motors = np.asarray(motors, dtype=float)
        force = np.zeros(3) if disturbance is None else np.asarray(disturbance, dtype=float)
        if motors.shape != (4,) or force.shape != (3,) or not np.all(np.isfinite(np.r_[motors, force])):
            raise ValueError('motors and disturbance have invalid shape or values')
        p, s = self.params, self.state
        thrust, *torque = self.allocation @ np.clip(motors, 0, p.max_thrust)
        inertia = np.asarray(p.inertia)
        acceleration = (s.rotation[:, 2] * thrust + force - p.linear_drag * s.velocity) / p.mass
        acceleration[2] -= p.gravity
        angular_acceleration = (torque - np.cross(s.omega, inertia * s.omega) - p.angular_drag * s.omega) / inertia
        s.velocity += acceleration * dt
        s.position += s.velocity * dt
        s.omega += angular_acceleration * dt
        s.rotation = s.rotation @ rotation_vector(s.omega * dt)
        # Remove accumulated roundoff while preserving a proper rotation.
        u, _, vt = np.linalg.svd(s.rotation)
        s.rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        if not all(np.all(np.isfinite(a)) for a in [s.position, s.velocity, s.omega, s.rotation]):
            raise FloatingPointError('simulation became nonfinite')
        return s.copy()
