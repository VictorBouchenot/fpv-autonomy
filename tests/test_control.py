import unittest
import numpy as np
from fpv_sim.physics import Quadrotor, State, rotation_vector
from fpv_sim.control import FlightController


class ControlTests(unittest.TestCase):
    def simulate(self, position=None, tilt=None, force=False, dt=0.005):
        q = Quadrotor(state=State(position=np.array(position if position is not None else [0., 0., 1.]), rotation=rotation_vector(np.array(tilt if tilt is not None else [0., 0., 0.]))))
        controller = FlightController(q)
        target = np.array([0., 0., 1.])
        states = []
        for i in range(round(12 / dt)):
            t = i * dt
            disturbance = np.array([0.08, -0.04, 0.]) if force and 3 <= t < 4 else np.zeros(3)
            motors = controller.command(target, dt)
            states.append(q.step(motors, dt, disturbance))
        return q, controller, states

    def test_hover(self):
        q, _, _ = self.simulate()
        np.testing.assert_allclose(q.state.position, [0, 0, 1], atol=1e-8)

    def test_offset_recovery(self):
        q, _, _ = self.simulate([0.5, -0.3, 0.6])
        self.assertLess(np.linalg.norm(q.state.position - [0, 0, 1]), 0.04)

    def test_tilt_recovery(self):
        q, _, _ = self.simulate(tilt=[0.25, -0.2, 0.1])
        self.assertLess(np.linalg.norm(q.state.position - [0, 0, 1]), 0.04)
        self.assertLess(np.linalg.norm(q.state.omega), 0.02)

    def test_disturbance_recovery(self):
        q, _, _ = self.simulate(force=True)
        self.assertLess(np.linalg.norm(q.state.position - [0, 0, 1]), 0.04)

    def test_combined_offset_tilt_and_disturbance(self):
        q, _, _ = self.simulate([0.5, -0.3, 0.6], [0.2, -0.15, 0], force=True)
        self.assertLess(np.linalg.norm(q.state.position - [0, 0, 1]), 0.05)
        self.assertLess(np.linalg.norm(q.state.omega), 0.02)

    def test_timestep_convergence(self):
        a, _, _ = self.simulate([0.5, -0.3, 0.6], dt=0.005)
        b, _, _ = self.simulate([0.5, -0.3, 0.6], dt=0.0025)
        self.assertLess(np.linalg.norm(a.state.position - b.state.position), 0.002)

    def test_anti_windup(self):
        q = Quadrotor()
        c = FlightController(q)
        for _ in range(100):
            motors = c.command([100, 100, 100], 0.005)
        self.assertTrue(c.saturated)
        np.testing.assert_array_equal(c.integral, 0)
        self.assertTrue(np.all((motors >= 0) & (motors <= q.params.max_thrust)))

    def test_reset(self):
        c = FlightController(Quadrotor())
        c.command([0, 0, 0.1], 0.005)
        c.reset()
        np.testing.assert_array_equal(c.integral, 0)

    def test_invalid_target(self):
        c = FlightController(Quadrotor())
        for target in [[1, 2], [0, 0, float('nan')]]:
            with self.assertRaises(ValueError): c.command(target, 0.005)


if __name__ == '__main__':
    unittest.main()
