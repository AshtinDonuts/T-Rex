import json
import unittest

import numpy as np

from hand_keypoint_retarget import (
    HandSample,
    PalmPose,
    assemble_trex_vector,
    fit_similarity,
    palm_normalize,
    palm_pose,
    parse_packet,
    relative_pose_target,
)


class HandKeypointRetargetTest(unittest.TestCase):
    def test_packet_with_confidence(self):
        points = np.arange(84, dtype=float).reshape(21, 4)
        packet = parse_packet(
            json.dumps({"timestamp": 12.5, "frame_id": "head_camera", "right": points.tolist()})
        )
        self.assertEqual(packet.timestamp, 12.5)
        self.assertEqual(packet.frame_id, "head_camera")
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

    def test_palm_pose_follows_rigid_transform(self):
        points = np.zeros((21, 3), dtype=float)
        points[5] = [0.04, 0.08, 0.0]
        points[9] = [0.0, 0.10, 0.0]
        points[13] = [-0.02, 0.08, 0.0]
        points[17] = [-0.04, 0.06, 0.0]
        theta = 0.4
        rotation = np.array(
            [[np.cos(theta), 0.0, np.sin(theta)], [0.0, 1.0, 0.0], [-np.sin(theta), 0.0, np.cos(theta)]]
        )
        translation = np.array([0.3, -0.2, 0.8])
        sample = HandSample(points, np.ones(21))
        transformed = HandSample(points @ rotation.T + translation, np.ones(21))
        first = palm_pose(sample)
        second = palm_pose(transformed)
        np.testing.assert_allclose(second.position, rotation @ first.position + translation)
        np.testing.assert_allclose(second.rotation, rotation @ first.rotation)

    def test_relative_pose_target(self):
        initial = PalmPose(np.array([0.1, 0.2, 0.3]), np.eye(3))
        current = PalmPose(np.array([0.2, 0.2, 0.3]), np.eye(3))
        robot_initial = PalmPose(np.array([0.6, -0.2, 0.9]), np.eye(3))
        camera_to_robot = np.diag([-1.0, -1.0, 1.0])
        target = relative_pose_target(current, initial, robot_initial, camera_to_robot, 0.5)
        np.testing.assert_allclose(target.position, [0.55, -0.2, 0.9])
        np.testing.assert_allclose(target.rotation, np.eye(3))

    def test_trex_vector_order(self):
        result = assemble_trex_vector(
            np.arange(7), 100 + np.arange(22), 200 + np.arange(7), 300 + np.arange(22)
        )
        self.assertEqual(result.shape, (58,))
        np.testing.assert_array_equal(result[:7], np.arange(7))
        np.testing.assert_array_equal(result[7:29], 100 + np.arange(22))
        np.testing.assert_array_equal(result[29:36], 200 + np.arange(7))
        np.testing.assert_array_equal(result[36:], 300 + np.arange(22))


if __name__ == "__main__":
    unittest.main()
