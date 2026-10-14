import os
import numpy as np
import robosuite as suite
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

_ROBOSUITE_V15 = tuple(int(part) for part in suite.__version__.split(".")[:2]) >= (1, 5)

class MountedAuboI5(ManipulatorModel):
    """
    Mounted Aubo i5 (6-DoF) robot wrapper for LIBERO and Robosuite 1.5.2.
    """

    # 1.5.2 Composite Controller architecture requirement
    arms = ["right"]

    def __init__(self, idn=0):
        # Dynamically resolve absolute path to the local assets folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Traverses up to 'libero/' then down into 'assets/robots/aubo_i5/'
        xml_path = os.path.join(
            current_dir, 
            "..", 
            "..", 
            "assets", 
            "robots", 
            "aubo_i5", 
            "aubo_i5.xml"
        )
        xml_path = os.path.normpath(xml_path)

        super().__init__(xml_path, idn=idn)

        # Set joint damping specifically for 6 joints, not 7
        self.set_joint_attribute(
            attrib="damping", 
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01))
        )

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        if _ROBOSUITE_V15:
            return {"right": "PandaGripper"}
        return "PandaGripper"

    @property
    def default_controller_config(self):
        if _ROBOSUITE_V15:
            return {"right": "default_panda"}
        return "OSC_POSE"

    @property
    def init_qpos(self):
        """Table-domain configuration solved against the ``MountedPanda`` TCP."""
        return np.array(
            [0.26348551, 0.06863280, 1.63365572, -0.06057558, 1.58557463, 0.26258903]
        )

    @property
    def base_xpos_offset(self):
        # Carried over strictly from the LIBERO MountedPanda standard
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "industry_workbench": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        # Target site parsed from your provided snippet
        if _ROBOSUITE_V15:
            return {"right": "tool0_link"}
        return "tool0_link"
