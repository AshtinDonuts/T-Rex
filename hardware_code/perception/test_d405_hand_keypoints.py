import unittest

import numpy as np

from d405_hand_keypoints import hand_geometry_diagnostics, lift_landmarks, robust_depth_m


class D405KeypointTest(unittest.TestCase):
    def test_geometry_diagnostics_flags_depth_outlier(self):
        points = np.zeros((21, 4), dtype=float)
        points[:, 3] = 1.0
        for finger, start in enumerate((1, 5, 9, 13, 17)):
            x = (finger - 2) * 0.015
            for offset in range(4):
                points[start + offset, :3] = [x, 0.03 + offset * 0.02, 0.30]
        points[8, 2] = 0.42
        result = hand_geometry_diagnostics(points)
        self.assertEqual(result["valid_count"], 21)
        self.assertIn(8, result["suspicious_indices"])
        self.assertGreater(result["depth_span_m"], 0.1)

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
