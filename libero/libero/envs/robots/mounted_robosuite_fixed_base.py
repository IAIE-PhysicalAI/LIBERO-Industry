"""Mounted wrappers for RoboSuite's fixed-base single-arm robots.

The wrappers reuse RoboSuite's robot XML, gripper, controller, and mount. They
only add LIBERO table-domain offsets and Panda-aligned initial joint poses.
"""

import numpy as np
import robosuite as suite
from robosuite.models.robots import IIWA, Jaco, Kinova3, Sawyer, UR5e


_ROBOSUITE_V15 = tuple(
    int(part) for part in suite.__version__.split(".")[:2]
) >= (1, 5)


class _IndustryWorkbenchMixin:
    """Add LIBERO table domains while retaining the upstream robot model."""

    @property
    def base_xpos_offset(self):
        offsets = super().base_xpos_offset.copy()
        offsets.update(
            {
                "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
                "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                "industry_workbench": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            }
        )
        return offsets

    @property
    def init_qpos(self):
        return np.array(self._panda_aligned_init_qpos, dtype=float)


class MountedIIWA(_IndustryWorkbenchMixin, IIWA):
    _panda_aligned_init_qpos = (
        -0.00009104,
        0.23739111,
        0.00010775,
        -1.67933576,
        -0.00002559,
        1.28164117,
        -0.00046590,
    )


class MountedJaco(_IndustryWorkbenchMixin, Jaco):
    _panda_aligned_init_qpos = (
        3.18348255,
        3.58143174,
        -0.03159039,
        1.66737946,
        0.01478607,
        4.42613006,
        3.15042516,
    )


class MountedKinova3(_IndustryWorkbenchMixin, Kinova3):
    # The upstream Kinova3 kinematic model differs slightly between 1.4 and 1.5.
    _panda_aligned_init_qpos = (
        (
            -0.06024857,
            0.41445850,
            0.01087922,
            1.46959965,
            -0.00745548,
            1.31426781,
            -1.61836938,
        )
        if _ROBOSUITE_V15
        else (
            -0.06040041,
            0.41245492,
            0.01114963,
            1.46112890,
            -0.00752042,
            1.32475759,
            -1.61832006,
        )
    )


class MountedSawyer(_IndustryWorkbenchMixin, Sawyer):
    _panda_aligned_init_qpos = (
        -0.23781719,
        -1.47019042,
        -0.14250679,
        2.02613464,
        -0.00774400,
        1.06860681,
        -1.77287264,
    )


class MountedUR5e(_IndustryWorkbenchMixin, UR5e):
    _panda_aligned_init_qpos = (
        -0.29217420,
        -1.66189117,
        1.85957843,
        -1.71410324,
        -1.58714485,
        -1.86301527,
    )


MOUNTED_ROBOSUITE_FIXED_BASE_MODELS = (
    MountedIIWA,
    MountedJaco,
    MountedKinova3,
    MountedSawyer,
    MountedUR5e,
)


if _ROBOSUITE_V15:
    from robosuite.models.robots import PandaDexLH, PandaDexRH, XArm7


    class MountedPandaDexLH(_IndustryWorkbenchMixin, PandaDexLH):
        _panda_aligned_init_qpos = (
            -0.37137582,
            0.59012762,
            0.40583290,
            -1.81752675,
            -0.02332810,
            0.73678457,
            -0.57936464,
        )


    class MountedPandaDexRH(_IndustryWorkbenchMixin, PandaDexRH):
        _panda_aligned_init_qpos = (
            0.37191004,
            0.59016925,
            -0.40608108,
            -1.81758530,
            0.02235313,
            0.73696816,
            2.15026106,
        )


    class MountedXArm7(_IndustryWorkbenchMixin, XArm7):
        _panda_aligned_init_qpos = (
            -0.00578358,
            -0.02276087,
            0.00554994,
            1.09903415,
            0.00014266,
            1.17860020,
            1.56962757,
        )

    MOUNTED_ROBOSUITE_FIXED_BASE_MODELS += (
        MountedPandaDexLH,
        MountedPandaDexRH,
        MountedXArm7,
    )
