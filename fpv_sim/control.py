"""Position PID outside a geometric attitude PD loop.

Integral accumulation freezes when acceleration/motor commands saturate.
All settings are educational, not safe real-flight firmware.
"""
import numpy as np


def vee(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])


class FlightController:
    def __init__(self, quad):
        self.quad = quad
        self.integral = np.zeros(3)
        self.kp = np.array([2.5, 2.5, 4.])
        self.kd = np.array([2.4, 2.4, 3.])
        self.ki = np.array([0.35, 0.35, 0.6])
        self.saturated = False

    def reset(self):
        self.integral[:] = 0
        self.saturated = False

    def command(self, target, dt, target_velocity=None, yaw=0.):
        target = np.asarray(target, dtype=float)
        velocity = np.zeros(3) if target_velocity is None else np.asarray(target_velocity, dtype=float)
        if target.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite(np.r_[target, velocity, yaw])):
            raise ValueError('target, velocity and yaw must be finite with valid shape')
        if not np.isfinite(dt) or not 0 < dt <= 0.02:
            raise ValueError('dt must be in (0, 0.02]')
        q, s = self.quad, self.quad.state
        p = q.params
        error = target - s.position
        candidate = np.clip(self.integral + error * dt, -2, 2)
        acceleration = self.kp * error + self.kd * (velocity - s.velocity) + self.ki * candidate
        bounded = np.clip(acceleration, [-4, -4, -4], [4, 4, 5])
        desired_force = p.mass * (bounded + np.array([0, 0, p.gravity]))
        body_z = desired_force / np.linalg.norm(desired_force)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])
        body_y = np.cross(body_z, heading)
        body_y /= np.linalg.norm(body_y)
        desired_rotation = np.column_stack([np.cross(body_y, body_z), body_y, body_z])
        attitude_error = 0.5 * vee(desired_rotation.T @ s.rotation - s.rotation.T @ desired_rotation)
        inertia = np.asarray(p.inertia)
        torque = -inertia * (90 * attitude_error + 18 * s.omega)
        torque += np.cross(s.omega, inertia * s.omega)
        thrust = float(desired_force @ s.rotation[:, 2])
        raw = q.inverse_allocation @ np.r_[thrust, torque]
        motors = np.clip(raw, 0, p.max_thrust)
        self.saturated = not (np.allclose(raw, motors) and np.allclose(acceleration, bounded))
        if not self.saturated:
            self.integral = candidate
        return motors
