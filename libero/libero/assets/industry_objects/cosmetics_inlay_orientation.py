import numpy as np


_COSMETICS_FAMILIES = (
    "pump_bottle",
    "serum_bottle",
    "cream_bottle",
    "cream_jar",
)


def cosmetics_inlay_alignment(obj_name, target_name, relative_rotation):
    """Return symmetry-aware alignment, or ``None`` outside cosmetics inlays."""
    obj_lower = obj_name.lower()
    target_lower = target_name.lower()
    family = next(
        (name for name in _COSMETICS_FAMILIES if name in obj_lower),
        None,
    )
    if family is None or family not in target_lower or "inlay" not in target_lower:
        return None

    rotation = np.asarray(relative_rotation, dtype=float).reshape(3, 3)
    if "flat" in obj_lower:
        object_top = -rotation[:, 1]
    elif "upright" in obj_lower:
        object_top = rotation[:, 2]
    else:
        return None

    if "flat_inlay" in target_lower:
        target_top = np.array([0.0, -1.0, 0.0])
    elif "upright_inlay" in target_lower:
        target_top = np.array([0.0, 0.0, 1.0])
    else:
        return None

    top_alignment = float(np.dot(object_top, target_top))
    if family == "serum_bottle":
        top_alignment = abs(top_alignment)
    if top_alignment <= 0.9:
        return False

    # The cream bottle's oval section is equivalent after a 180-degree roll,
    # not after an arbitrary roll around its signed bottom-to-top axis.
    if family == "cream_bottle":
        return abs(float(np.dot(rotation[:, 0], [1.0, 0.0, 0.0]))) > 0.9
    return True
