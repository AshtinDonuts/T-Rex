import json
import unittest

import numpy as np

from hand_keypoint_retarget import HandSample, fit_similarity, palm_normalize, parse_packet


class HandKeypointRetargetTest(unittest.TestCase):
    def test_packet_with_confidence(self):
        points = np.arange(84, dtype=float).reshape(21, 4)
        packet = parse_packet(json.dumps({"timestamp": 12.5, "right": points.tolist()}))
        self.assertEqual(packet.timestamp, 12.5)
        np.testing.assert_allclose(packet.hands["right"].xyz, points[:, :3])
        np.testing.assert_allclose(packet.hands["right"].confidence, points[:, 3])

    def test_palm_normalization_is_similarity_invariant(self):
        points = np.zeros((21, 3), dtype=float)
        points[5] = [0.04, 0.08, 0.0]
        points[9] = [0.0, 0.10, 0.0]
        points[13] = [-0.02, 0.08, 0.0]
        points[17] = [-0.04, 0.06, 0.0]
        for index in set(range(21)) - {0, 5, 9, 13, 17}:
            points[index] = [index * 0.001, 0.02 + index * 0.003, index * 0.0005]
        theta = 0.7
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]
        )
        transformed = 2.3 * points @ rotation.T + np.array([1.0, -2.0, 0.4])
        local_a, *_ = palm_normalize(HandSample(points, np.ones(21)))
        local_b, *_ = palm_normalize(HandSample(transformed, np.ones(21)))
        np.testing.assert_allclose(local_a, local_b, atol=1e-10)

    def test_similarity_recovery(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(20, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        target = 0.083 * source @ rotation.T + np.array([0.01, -0.03, 0.02])
        transform = fit_similarity(source, target)
        np.testing.assert_allclose(transform.apply(source), target, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
