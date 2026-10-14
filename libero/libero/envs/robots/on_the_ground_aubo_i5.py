"""Floor-mounted Aubo i5 model for LIBERO floor-domain tasks."""

import numpy as np

from .mounted_aubo_i5 import MountedAuboI5


class OnTheGroundAuboI5(MountedAuboI5):
    """Aubo i5 without the Rethink mount used by table-domain tasks.

    The floor domain places the robot through the inherited ``empty`` base
    offset. The arm geometry, end-effector frame, and gripper defaults are
    shared with ``MountedAuboI5``; its initial joint configuration is not.
    """

    @property
    def default_base(self):
        """Robosuite 1.5.x fixed-base equivalent of a null mount."""
        return "NullMount"

    @property
    def default_mount(self):
        """Robosuite 1.4.x mount selection for floor-domain placement."""
        return None

    @property
    def base_xpos_offset(self):
        """Match on-the-ground Panda placement in low-table LIBERO domains."""
        offsets = super().base_xpos_offset.copy()
        offsets.update(
            {
                "coffee_table": lambda table_length: (
                    -0.16 - table_length / 2,
                    0,
                    0.41,
                ),
                "living_room_table": lambda table_length: (
                    -0.16 - table_length / 2,
                    0,
                    0.42,
                ),
            }
        )
        return offsets

    @property
    def init_qpos(self):
        """Floor-domain configuration solved against the ``OnTheGroundPanda`` gripper TCP."""
        return np.array(
            [0.26363208, 0.06862388, 1.63352407, -0.06072957, 1.58557244, 0.26298595]
        )
