import unittest

import numpy as np
import robosuite

from libero.libero.utils.init_state_utils import (
    EVALUATION_EMBODIMENTS,
    adapt_evaluation_actions,
    available_evaluation_embodiments,
    evaluation_requires_state_transcode,
    get_evaluation_source_profile,
)


class SingleArmEvaluationEmbodimentsTest(unittest.TestCase):
    def test_installed_robosuite_exposes_supported_single_arm_profiles(self):
        expected = {
            "panda",
            "aubo_i5_ctag2f90d",
            "iiwa",
            "jaco",
            "kinova3",
            "sawyer",
            "ur5e",
        }
        version = tuple(int(part) for part in robosuite.__version__.split(".")[:2])
        if version >= (1, 5):
            expected.update({"panda_dex_lh", "panda_dex_rh", "xarm7"})
        self.assertEqual(expected, set(available_evaluation_embodiments()))

    def test_new_fixed_base_profiles_reuse_panda_task_states(self):
        available = set(available_evaluation_embodiments())
        for name, profile in EVALUATION_EMBODIMENTS.items():
            if name not in available:
                continue
            with self.subTest(name=name):
                source = get_evaluation_source_profile(name)
                if name in {"panda", "aubo_i5_ctag2f90d"}:
                    self.assertFalse(evaluation_requires_state_transcode(name))
                else:
                    self.assertEqual(source.name, "panda")
                    self.assertTrue(evaluation_requires_state_transcode(name))

    def test_dexterous_hands_broadcast_the_panda_gripper_command(self):
        version = tuple(int(part) for part in robosuite.__version__.split(".")[:2])
        if version < (1, 5):
            self.skipTest("PandaDex is available only under RoboSuite 1.5+")
        actions = np.array([[1, 2, 3, 4, 5, 6, -0.25]], dtype=float)
        adapted = adapt_evaluation_actions(actions, "panda_dex_lh", 12)
        np.testing.assert_array_equal(adapted[:, :6], actions[:, :6])
        np.testing.assert_array_equal(adapted[:, 6:], np.full((1, 6), -0.25))

    def test_incompatible_action_dimension_fails_with_context(self):
        with self.assertRaisesRegex(ValueError, "ur5e.*8"):
            adapt_evaluation_actions(np.zeros((2, 7)), "ur5e", 8)


if __name__ == "__main__":
    unittest.main()
