from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda
from .mounted_aubo_i5 import MountedAuboI5
from .on_the_ground_aubo_i5 import OnTheGroundAuboI5
from . import grippers as _grippers  # Register LIBERO grippers with robosuite.

# from robosuite.robots.single_arm import SingleArm
import robosuite as suite
from robosuite.robots import ROBOT_CLASS_MAPPING

_version_parts = suite.__version__.split(".")
IS_V1_5_OR_GREATER = int(_version_parts[0]) >= 1 and int(_version_parts[1]) >= 5

if IS_V1_5_OR_GREATER:
    # 1.5.2 Architecture
    from robosuite.robots.fixed_base_robot import FixedBaseRobot as BaseRobotClass
    
    ROBOT_CLASS_MAPPING.update(
        {
            "MountedPanda": BaseRobotClass,
            "OnTheGroundPanda": BaseRobotClass,
            "MountedAuboI5": BaseRobotClass,
            "OnTheGroundAuboI5": BaseRobotClass,
        }
    )

else:
    # 1.4.0 Legacy Architecture
    from robosuite.robots.single_arm import SingleArm as BaseRobotClass
    
    ROBOT_CLASS_MAPPING.update(
        {
            "MountedPanda": BaseRobotClass,
            "OnTheGroundPanda": BaseRobotClass,
            "MountedAuboI5": BaseRobotClass,
            "OnTheGroundAuboI5": BaseRobotClass,
        }
    )
