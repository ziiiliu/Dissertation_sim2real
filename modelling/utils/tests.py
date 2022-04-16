from metrics import avg_distance_from_start, parallel_degree_avg, parallel_degree_step, propagated_error
import unittest
import numpy as np

class TestMetrics(unittest.TestCase):
    def test_avg_distance_from_start(self):
        # test case
        traj1 = [[1,2], [1,3], [2,2]]
        traj2 = [[4,5], [5,6], [2,2]]
        actual_mean, _ = avg_distance_from_start(traj1, traj2)
        expected_mean = (3 * np.sqrt(2) + 5)/3
        self.assertEqual(actual_mean, expected_mean)

if __name__ == '__main__':
    unittest.main()

    vel1 = [(1,1), (1,2), (2,3)]
    vel2 = [(1,0), (0,0), (1,4)]

    cos_sim = parallel_degree_step(vel1, vel2)

    coords1 = [(1,1), (1,2), (2,3)]
    coords2 = [(1,0), (0,0), (1,4)]

    sim_avg, sim_std = parallel_degree_avg(coords1, coords2)
    print(sim_avg, sim_std)

    avg_change = propagated_error(coords1, coords2, n=1)
    print(avg_change)