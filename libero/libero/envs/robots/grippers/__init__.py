from .suction_gripper import SuctionGripper
from .ctag2f90d_gripper import CTAG2F90DGripper
from robosuite.models.grippers import GRIPPER_MAPPING

GRIPPER_MAPPING.update(
    {
    "SuctionGripper": SuctionGripper,
    "CTAG2F90DGripper": CTAG2F90DGripper,
    }
)
