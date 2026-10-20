import unittest

import numpy as np
import robosuite as suite
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS, create_robot
from robosuite.robots import ROBOT_CLASS_MAPPING

from libero.libero.envs import robots as libero_robots


class MountedRobosuiteFixedBaseTest(unittest.TestCase):
    def test_models_are_registered_for_installed_robosuite(self):
        expected = {
            "MountedIIWA",
            "MountedJaco",
            "MountedKinova3",
            "MountedSawyer",
            "MountedUR5e",
        }
        is_v15 = tuple(int(part) for part in suite.__version__.split(".")[:2]) >= (1, 5)
        if is_v15:
            expected.update({"MountedPandaDexLH", "MountedPandaDexRH", "MountedXArm7"})

        self.assertEqual(
            expected,
            {model.__name__ for model in libero_robots.MOUNTED_ROBOSUITE_FIXED_BASE_MODELS},
        )
        for name in expected:
            self.assertIn(name, REGISTERED_ROBOTS)
            self.assertTrue(hasattr(libero_robots, name))
            self.assertIs(ROBOT_CLASS_MAPPING[name], libero_robots.BaseRobotClass)

            model = create_robot(name, idn=0)
            if is_v15:
                self.assertEqual(model.arms, ["right"])
            self.assertTrue(
                {"study_table", "kitchen_table", "industry_workbench"}
                <= model.base_xpos_offset.keys()
            )
            self.assertTrue(np.isfinite(model.init_qpos).all())
            self.assertTrue(model.default_gripper)
            self.assertTrue(model.default_controller_config)

        if is_v15:
            self.assertEqual(
                GRIPPER_MAPPING["XArm7Gripper"].__module__,
                "robosuite.models.grippers.xarm7_gripper",
            )

    def test_existing_panda_and_aubo_routes_remain_registered(self):
        for name in (
            "MountedPanda",
            "OnTheGroundPanda",
            "MountedAuboI5",
            "OnTheGroundAuboI5",
        ):
            self.assertIn(name, REGISTERED_ROBOTS)
            self.assertIs(ROBOT_CLASS_MAPPING[name], libero_robots.BaseRobotClass)


if __name__ == "__main__":
    unittest.main()
