"""
Suction Gripper implementation.
OriginalImplementation:
Author: Rakshitha Ireddi
Source: https://github.com/ARISE-Initiative/robosuite/tree/feature/suction-gripper
Commit: 584d4ebe2a0cdc3d6366edc737de8f4bef45dfea

Adapted for LIBERO-Industry Benchmark.
"""
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class SuctionGripper(GripperModel):
    """
    Suction Gripper with 1 actuator (adhesion).
    
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        # Resolve local absolute path to assets
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(
            current_dir, 
            "..", "..", "..", 
            "assets", "grippers", "suction_gripper", "suction_gripper.xml"
        )
        xml_path = os.path.normpath(xml_path)
        
        super().__init__(xml_path, idn=idn)

    def format_action(self, action):
        """
        Args:
            action (np.array): boolean or continuous action for suction.
        """
        assert len(action) == 1
        self.current_action = np.clip(action, -1.0, 1.0)
        # Map [-1, 1] to [0, 1] for adhesion actuator
        return (self.current_action + 1.0) / 2.0

    @property
    def init_qpos(self):
        return np.array([])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
            "cup_geom": ["cup_visual", "cup_collision"],
        }