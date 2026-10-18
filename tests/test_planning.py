import unittest
import numpy as np
from fpv_sim.planning import Box, World, astar, simplify


class PlanningTests(unittest.TestCase):
    def test_box_segment(self):
        box = Box((1, 1, 1), (2, 2, 2))
        self.assertTrue(box.intersects([0, 1.5, 1.5], [3, 1.5, 1.5]))
        self.assertFalse(box.intersects([0, 0, 0], [3, 0, 0]))
        self.assertTrue(box.intersects([1.5]*3, [1.5]*3))
        self.assertTrue(box.intersects([0, 1, 1], [3, 1, 1]))

    def test_clearance(self):
        world = World([0]*3, [4]*3, [Box((1,1,1), (2,2,2))], clearance=.2)
        self.assertFalse(world.free_segment([.9,.5,1.5], [.9,3,1.5]))
        self.assertFalse(world.free_segment([0,0,0], [1,0,0]))

    def test_open_path(self):
        world = World([0]*3,[3]*3,clearance=.1)
        path = astar(world,[.2]*3,[2.7]*3,resolution=.5)
        np.testing.assert_array_equal(path[0],[.2]*3)
        np.testing.assert_array_equal(path[-1],[2.7]*3)
        self.assertTrue(all(world.free_segment(a,b) for a,b in zip(path,path[1:])))

    def test_around_obstacle(self):
        world = World([0]*3,[4]*3,[Box((1.5,.5,.5),(2.5,3.5,2.5))],clearance=.15)
        path = astar(world,[.5,2,1],[3.5,2,1],resolution=.4)
        self.assertTrue(all(world.free_segment(a,b) for a,b in zip(path,path[1:])))
        short = simplify(world,path)
        self.assertTrue(all(world.free_segment(a,b) for a,b in zip(short,short[1:])))
        self.assertLessEqual(len(short),len(path))

    def test_unreachable(self):
        world = World([0]*3,[2]*3,[Box((.9,0,0),(1.1,2,2))],clearance=.1)
        with self.assertRaises(ValueError): astar(world,[.2]*3,[1.8]*3,resolution=.3)

    def test_diagonal_corner_not_cut(self):
        world = World([0]*3,[3]*3,[Box((1,0,0),(2,1,3))],clearance=0)
        self.assertFalse(world.free_segment([.5,.5,1],[1.5,1.5,1]))

    def test_blocked_endpoint(self):
        world = World([0]*3,[3]*3,clearance=.2)
        with self.assertRaises(ValueError): astar(world,[0]*3,[2]*3)

    def test_same_endpoint(self):
        world = World([0]*3,[3]*3)
        self.assertEqual(len(astar(world,[1]*3,[1]*3)),1)

    def test_deterministic(self):
        world = World([0]*3,[2]*3)
        np.testing.assert_array_equal(astar(world,[.3]*3,[1.5]*3),astar(world,[.3]*3,[1.5]*3))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError): Box((0,0,0),(0,1,1))
        with self.assertRaises(ValueError): World([0]*3,[1]*3,clearance=1)
        with self.assertRaises(ValueError): astar(World([0]*3,[2]*3),[.5]*3,[1]*3,resolution=0)


if __name__ == '__main__': unittest.main()
