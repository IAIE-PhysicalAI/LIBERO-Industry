"""
Noise profiles for industrial-grade multi-embodiment simulation.
Named robot values are millimetres and are converted to metres when resolved.
Benchmark camera extrinsics are generated in ``set_scene`` rather than here.
"""

from typing import Union

ROBOT_NOISE_MAP = {
    "baseline": 0.0,      # mm standard deviation
    "general": 0.1,       # mm standard deviation
    "flexible": 2.0,      # mm standard deviation
}

def resolve_robot_noise(noise_arg: Union[str, float]) -> float:
    """Resolve a named mm profile to metres; numeric values are already metres."""
    if isinstance(noise_arg, str):
        return ROBOT_NOISE_MAP.get(noise_arg, 0.0) * 1e-3
    return float(noise_arg)
