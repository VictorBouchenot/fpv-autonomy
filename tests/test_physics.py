import unittest
import numpy as np
from fpv_sim.physics import Parameters, Quadrotor, State, rotation_vector


class PhysicsTests(unittest.TestCase):
    def test_hover(self):
        q = Quadrotor()
        for _ in range(1000):
            q.step(np.full(4, q.params.mass * q.params.gravity / 4), 0.005)
        np.testing.assert_allclose(q.state.position, 0, atol=1e-12)
        np.testing.assert_allclose(q.state.rotation, np.eye(3), atol=1e-12)

    def test_free_fall(self):
        q = Quadrotor(Parameters(linear_drag=0))
        for _ in range(100):
            q.step(np.zeros(4), 0.01)
        self.assertAlmostEqual(q.state.velocity[2], -9.81)
        self.assertAlmostEqual(q.state.position[2], -9.81 * 0.01**2 * 100 * 101 / 2)

    def test_allocation_reconstruction(self):
        q = Quadrotor()
        wrench = np.array([0.981, 0.002, -0.001, 0.0001])
        np.testing.assert_allclose(q.allocation @ q.mix(wrench[0], wrench[1:]), wrench)

    def test_saturation(self):
        q = Quadrotor()
        motors = q.mix(100, [1, -1, 1])
        self.assertTrue(np.all(motors >= 0))
        self.assertTrue(np.all(motors <= q.params.max_thrust))

    def test_tilt_changes_thrust_direction(self):
        q = Quadrotor(state=State(rotation=rotation_vector(np.array([0., 0.2, 0.]))))
        q.step(np.full(4, 0.25), 0.005)
        self.assertGreater(q.state.velocity[0], 0)

    def test_differential_motors_rotate(self):
        q = Quadrotor()
        q.step(q.mix(0.981, [0.005, 0, 0]), 0.005)
        self.assertGreater(q.state.omega[0], 0)

    def test_rotation_remains_proper(self):
        q = Quadrotor(state=State(omega=np.array([1., 2., 3.])))
        for _ in range(500):
            q.step(np.zeros(4), 0.005)
        np.testing.assert_allclose(q.state.rotation.T @ q.state.rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(q.state.rotation), 1)

    def test_invalid_inputs(self):
        q = Quadrotor()
        for dt in [0, -1, 0.1, float('nan')]:
            with self.assertRaises(ValueError): q.step(np.zeros(4), dt)
        with self.assertRaises(ValueError): q.step([1, 2], 0.005)
        with self.assertRaises(ValueError): Parameters(mass=0)

    def test_copies_are_independent(self):
        state = State()
        q = Quadrotor(state=state)
        q.step(np.zeros(4), 0.005)
        self.assertEqual(state.position[2], 0)
        copy = q.state.copy()
        copy.position[0] = 100
        self.assertEqual(q.state.position[0], 0)

    def test_deterministic(self):
        a, b = Quadrotor(), Quadrotor()
        for _ in range(100):
            a.step([0.2, 0.3, 0.25, 0.24], 0.005)
            b.step([0.2, 0.3, 0.25, 0.24], 0.005)
        np.testing.assert_array_equal(a.state.position, b.state.position)


if __name__ == '__main__':
    unittest.main()
