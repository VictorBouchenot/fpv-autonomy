import unittest
from dataclasses import replace
import numpy as np
from fpv_sim.mission import CONSERVATIVE, DYNAMIC, FlightProfile, demo_world, fly


class ConservativeMissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.start, cls.goal = demo_world()
        cls.result = fly(cls.world, cls.start, cls.goal, profile=CONSERVATIVE)

    def test_arrives(self):
        self.assertTrue(self.result.arrived)
        self.assertFalse(self.result.collision)
        self.assertLess(self.result.final_error, .061)

    def test_all_actual_segments_clear(self):
        r = self.result
        self.assertTrue(all(self.world.free_segment(a, b) for a, b in zip(r.positions, r.positions[1:])))

    def test_motors_bounded(self):
        self.assertTrue(np.all((self.result.motors >= 0) & (self.result.motors <= .6)))

    def test_tracking_error(self):
        self.assertLess(self.result.max_tracking_error, .3)

    def test_timeout_is_not_success(self):
        r = fly(self.world, self.start, self.goal, duration=.1, profile=CONSERVATIVE)
        self.assertFalse(r.arrived)
        self.assertFalse(r.collision)


class DynamicMissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.start, cls.goal = demo_world()
        cls.result = fly(cls.world, cls.start, cls.goal, profile=DYNAMIC)
        cls.conservative = fly(cls.world, cls.start, cls.goal, profile=CONSERVATIVE)

    def test_arrives_without_collision(self):
        self.assertTrue(self.result.arrived)
        self.assertFalse(self.result.collision)
        self.assertLess(self.result.final_error, .081)

    def test_all_actual_segments_clear(self):
        r = self.result
        self.assertTrue(all(self.world.free_segment(a, b) for a, b in zip(r.positions, r.positions[1:])))
        self.assertGreater(r.min_point_clearance, self.world.clearance + .04)

    def test_real_faster_banked_response(self):
        self.assertGreater(self.result.max_speed, 6 * self.conservative.max_speed)
        self.assertGreater(self.result.max_tilt_degrees, 12)
        self.assertLess(self.result.times[-1], self.conservative.times[-1] / 6)

    def test_motors_bounded(self):
        self.assertTrue(np.all((self.result.motors >= 0) & (self.result.motors <= .6)))

    def test_route_climbs_descends_then_climbs(self):
        z = self.result.path[:, 2]
        self.assertGreater(np.max(z[:3]), 2.2)
        self.assertLess(np.min(z[2:-1]), 1.4)
        self.assertGreater(z[-1], 2.1)

    def test_reverse_scenario_variant(self):
        world, start, goal = demo_world('reverse')
        r = fly(world, start, goal, profile=DYNAMIC)
        self.assertTrue(r.arrived)
        self.assertFalse(r.collision)
        self.assertTrue(all(world.free_segment(a, b) for a, b in zip(r.positions, r.positions[1:])))

    def test_timestep_robustness(self):
        coarse = fly(self.world, self.start, self.goal, dt=.01, profile=DYNAMIC)
        fine = fly(self.world, self.start, self.goal, dt=.0025, profile=DYNAMIC)
        self.assertTrue(coarse.arrived and fine.arrived)
        self.assertFalse(coarse.collision or fine.collision)
        self.assertLess(abs(coarse.times[-1] - fine.times[-1]), .5)
        self.assertLess(max(coarse.final_error, fine.final_error), .1)

    def test_too_fast_setting_fails_honestly(self):
        reckless = replace(DYNAMIC, name='unchecked-fast', reference_speed=2.15)
        r = fly(self.world, self.start, self.goal, profile=reckless)
        self.assertFalse(r.arrived)
        self.assertTrue(r.collision)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fly(self.world, self.start, self.goal, duration=0)
        with self.assertRaises(ValueError):
            fly(self.world, self.start, self.goal, dt=0)
        with self.assertRaises(ValueError):
            fly(self.world, self.start, self.goal, profile='sport')
        with self.assertRaises(ValueError):
            demo_world('unknown')
        with self.assertRaises(ValueError):
            FlightProfile('bad', 0, .35, False, .08, .2)


if __name__ == '__main__':
    unittest.main()
