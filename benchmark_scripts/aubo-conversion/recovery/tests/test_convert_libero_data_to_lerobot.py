"""Small simulator-free checks for the recovered-v2 LeRobot exporter."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "convert_libero_data_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("aubo_lerobot_v21_exporter", SCRIPT)
EXPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPORTER
SPEC.loader.exec_module(EXPORTER)


class AuboLeRobotV21ExporterTest(unittest.TestCase):
    def test_contract_has_no_machine_specific_path_defaults(self):
        self.assertEqual(EXPORTER.DEFAULT_FPS, 10)
        self.assertFalse(hasattr(EXPORTER, "DEFAULT_DATA_DIR"))
        self.assertFalse(hasattr(EXPORTER, "DEFAULT_OUTPUT_ROOT"))
        self.assertFalse(hasattr(EXPORTER, "DEFAULT_REPO_NAME"))
        self.assertFalse(hasattr(EXPORTER, "V3_VERSION"))

    def test_noops_are_detected_without_removing_rows(self):
        actions = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1],
                [0.1, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.assertEqual(EXPORTER._noop_steps(actions), (2, 4))

    def test_seven_dimensional_state_and_action_features(self):
        positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        quaternions = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
            ]
        )
        gripper = np.array([[0.25, 9.0], [-0.5, 8.0]])
        state = EXPORTER._policy_state(positions, quaternions, gripper)
        features = EXPORTER._features((256, 256, 3))

        self.assertEqual(state.shape, (2, 7))
        self.assertEqual(state.dtype, np.float32)
        np.testing.assert_allclose(state[1, 3:6], [0.0, 0.0, np.pi / 2])
        self.assertEqual(set(features), {"image", "wrist_image", "state", "actions"})
        self.assertEqual(features["state"]["shape"], (7,))
        self.assertEqual(features["actions"]["shape"], (7,))
        self.assertEqual(features["image"]["dtype"], "image")

    def test_hdf5_horizontal_flip_matches_pi_orientation(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        oriented = EXPORTER._policy_image(image)
        np.testing.assert_array_equal(oriented, image[:, ::-1])
        np.testing.assert_array_equal(oriented, image[::-1][::-1, ::-1])
        self.assertTrue(oriented.flags.c_contiguous)

    def test_canary_selects_one_episode_per_suite(self):
        episodes = [
            EXPORTER.Episode(suite, Path("task.hdf5"), f"demo_{index}", 5, "task", "wrist", (2, 3, 3))
            for suite in ("libero_spatial", "libero_object")
            for index in range(2)
        ]
        selected = EXPORTER._limit_episodes_per_suite(episodes, 1)
        self.assertEqual([episode.demo for episode in selected], ["demo_0", "demo_0"])


if __name__ == "__main__":
    unittest.main()
