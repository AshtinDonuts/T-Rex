import unittest

import numpy as np

from d405_hand_keypoints import lift_landmarks, robust_depth_m


class D405KeypointTest(unittest.TestCase):
    def test_robust_depth_rejects_holes_and_outlier(self):
        depth = np.zeros((9, 9), dtype=np.float32)
        depth[2:7, 2:7] = 0.25
        depth[4, 4] = 0.49
        self.assertAlmostEqual(robust_depth_m(depth, 4, 4, 2, 0.07, 0.5, 5), 0.25)

    def test_lift_landmarks_metric_and_missing_confidence(self):
        depth = np.full((100, 200), 0.3, dtype=np.float32)
        depth[45:56, 95:106] = 0.0
        landmarks = np.tile([[0.25, 0.25]], (21, 1))
        landmarks[0] = [0.5, 0.5]

        def deproject(pixel, value):
            return [(pixel[0] - 100.0) * value / 100.0, (pixel[1] - 50.0) * value / 100.0, value]

        result = lift_landmarks(landmarks, depth, deproject, 0.8, 1, 0.07, 0.5, 3)
        self.assertEqual(result.shape, (21, 4))
        self.assertEqual(result[0, 3], 0.0)
        self.assertAlmostEqual(result[1, 2], 0.3)
        self.assertAlmostEqual(result[1, 3], 0.8)


if __name__ == "__main__":
    unittest.main()
