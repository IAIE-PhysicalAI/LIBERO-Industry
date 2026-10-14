import os
import numpy as np
from robosuite.models.grippers import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion

class CTAG2F90DGripperBase(GripperModel):
    """
    Base class containing physical definitions for the CTAG2F90-D Gripper.
    """
    def __init__(self, idn=0):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(
            current_dir, 
            "..", "..", "..", 
            "assets", "grippers", "ctag2f90d_gripper", "ctag2f90d_gripper.xml"
        )
        super().__init__(os.path.normpath(xml_path), idn=idn)

    # @property
    # def joints(self):
    #     """
    #     CRITICAL OVERRIDE: Mask the 5 passive joints from the controller.
    #     This forces Robosuite's JOINT_POSITION controller to initialize with jnt_dim = 1.
    #     """
    #     return ["gripper0_right_Left_1_Joint"]

    @property
    def init_qpos(self):
        """
        The XML exposes one actuator for every joint so that Robosuite 1.5 can
        build its gripper controller.  Consequently this vector must match all
        six joints, even though the equality constraints keep five of them in
        sync with the driver joint.  Zero is the fully-open factory pose.
        """
        return np.zeros(6)

    def format_action(self, action):
        return action

    @property
    def _important_sites(self):
        """Sites defined by the CTAG gripper XML and used by Robosuite."""
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_site_cylinder",
        }

    @property
    def _important_geoms(self):
        """Collision-only rubber contact surfaces used for grasp detection."""
        return {
            "left_finger": ["left_finger_pad_collision"],
            "right_finger": ["right_finger_pad_collision"],
            "left_fingerpad": ["left_finger_pad_collision"],
            "right_fingerpad": ["right_finger_pad_collision"],
        }


class CTAG2F90DGripper(CTAG2F90DGripperBase):
    @property
    def dof(self):
        return 1

    @property
    def speed(self):
        # Robosuite supplies a discrete open / close signal. Reaching the
        # endpoint in one policy step leaves the entire grasp phase available
        # for the position actuator to establish clamping force.
        return 2.0

    def format_action(self, action):
        """
        Maps the Robosuite scalar action to normalized position commands for
        the six XML actuators.  Robosuite scales these values into each
        actuator's ctrlrange, so this method must operate in [-1, 1] rather
        than returning joint angles in radians.

        -1 opens the gripper and +1 closes it; smaller magnitudes move it
        proportionally so contact-heavy releases can unload gradually.
        """
        assert len(action) == self.dof

        # The model resets gripper qpos to its open pose. Start the command at
        # the matching normalized endpoint as well. Keeping this state here
        # makes repeated close commands maintain clamping force on an object.
        if getattr(self, "_master_action", None) is None:
            self._master_action = 1.0

        command = float(np.clip(action[0], -1.0, 1.0))
        self._master_action = np.clip(
            self._master_action - self.speed * command, -1.0, 1.0
        )

        # Factory URDF mimic multipliers, ordered as the six XML actuators:
        # [Left_1, Left_Support, Left_2, Right_2, Right_1, Right_Support].
        # The right driver and support use -1, matching their factory mimic
        # joints and the corrected MuJoCo equality constraints.
        multipliers = np.array([1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
        self.current_action = self._master_action * multipliers
        return self.current_action
