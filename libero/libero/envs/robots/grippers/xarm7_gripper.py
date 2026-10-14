"""
Gripper for UFactory's XArm7 (has two fingers).
"""
import os
import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class XArm7GripperBase(GripperModel):
    """
    Gripper for UFactory's XArm7.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        # Resolve local absolute path to assets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(
            current_dir, 
            "..", "..", "..", 
            "assets", "grippers", "xarm7_gripper", "xarm7_gripper.xml"
        )
        xml_path = os.path.normpath(xml_path)
        
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.02, 0.0, 0.0, -0.02, 0.0, 0.0])

    @property
    def _important_geoms(self):
        # Mapped to the exact <geom> names defined in xarm7_gripper.xml
        return {
            "left_fingerpad": ["left_finger_pad_1", "left_finger_pad_2"],
            "right_fingerpad": ["right_finger_pad_1", "right_finger_pad_2"],
        }


class XArm7Gripper(XArm7GripperBase):
    """
    Modifies XArm7 Gripper to only take one action.
    """

    def format_action(self, action):
        """
        -1 => fully open, +1 => fully closed
        """
        assert len(action) == self.dof  # i.e., 1
        # Suppose self.current_action is also shape (1,). Then:
        action_multiplier = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        delta = self.speed * np.sign(action)
        # self.current_action = np.clip(self.current_action + delta, -1.0, 1.0)
        if getattr(self, "current_action", None) is None or np.isscalar(self.current_action) or len(self.current_action) != 6:
            self.current_action = np.zeros(6)
            
        self.current_action = np.clip(
            self.current_action + action_multiplier * delta, 
            -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1
