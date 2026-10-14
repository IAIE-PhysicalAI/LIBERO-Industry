"""Measure procedural-layout space from the compiled MuJoCo scene.

This script deliberately does not read fixture XML positions.  Fixture poses are
overwritten while the environment is reset, so all measurements below are taken
from ``sim.data`` after the scene has stepped.
"""

from __future__ import annotations

import glob
import os
import pprint
import re
import sys
from collections import defaultdict
from typing import Any

import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# This lets the analyzer read the current training configuration without relying
# on a caller's working directory.
_AUTOGEN_DIR = os.path.dirname(__file__)
if _AUTOGEN_DIR not in sys.path:
    sys.path.insert(0, _AUTOGEN_DIR)

import configs


TARGET_DIR = os.path.join(
    _REPO_ROOT,
    "libero",
    "libero",
    "bddl_files",
    "libero_industry_draft",
    "modular_test",
    "bddl_files",
)

# These are the training-distribution limits, not a fixture-local region.
DIST_BOUNDS = {
    "x_min": -0.41,
    "x_max": 0.20,
    "y_min": -0.27,
    "y_max": 0.32,
}
SETTLE_STEPS = 25

# MuJoCo's mjtGeom values. Keeping these local avoids a version-specific import
# while still allowing us to calculate an exact world-space AABB per geom.
GEOM_PLANE = 0
GEOM_HFIELD = 1
GEOM_SPHERE = 2
GEOM_CAPSULE = 3
GEOM_ELLIPSOID = 4
GEOM_CYLINDER = 5
GEOM_BOX = 6
GEOM_MESH = 7


def _as_floats(values: np.ndarray | list[float]) -> list[float]:
    """Return ordinary Python floats so the final result is config-pasteable."""
    return [float(value) for value in values]


def _body_id(env: Any, name: str) -> int:
    """Resolve an instance to its physical ``*_main`` body, never a region/site."""
    candidates = (f"{name}_main", name)
    for candidate in candidates:
        try:
            return int(env.sim.model.body_name2id(candidate))
        except (KeyError, ValueError):
            pass

    # ``obj_body_id`` is the environment's compiled-scene mapping. It is only a
    # fallback because the explicit *_main lookup above documents the invariant.
    body_id = getattr(env, "obj_body_id", {}).get(name)
    if body_id is not None:
        return int(body_id)
    raise KeyError(f"Could not find a physical main body for '{name}'.")


def _main_body_com(sim: Any, body_id: int) -> np.ndarray:
    """Return the exact world CoM of the physical ``*_main`` body subtree.

    ``subtree_com`` includes child bodies/geometries, which is the correct CoM
    for LIBERO objects whose visual and collision pieces are nested below the
    ``*_main`` body.  It is calculated by MuJoCo after each forward/step.
    """
    if hasattr(sim.data, "subtree_com"):
        return sim.data.subtree_com[body_id].copy()
    # Older bindings do not expose subtree_com. xipos is still MuJoCo's global
    # inertial-frame CoM for the requested main body, never a fixture region.
    return sim.data.xipos[body_id].copy()


def _is_descendant(sim: Any, body_id: int, root_body_id: int) -> bool:
    """Whether body_id belongs to root_body_id's compiled body subtree."""
    current = int(body_id)
    while current:
        if current == root_body_id:
            return True
        current = int(sim.model.body_parentid[current])
    return root_body_id == 0


def _geom_world_aabb(sim: Any, geom_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Calculate a primitive or mesh's world-space AABB from MuJoCo runtime data."""
    geom_type = int(sim.model.geom_type[geom_id])
    if geom_type in (GEOM_PLANE, GEOM_HFIELD):
        return None

    center = sim.data.geom_xpos[geom_id].copy()
    rotation = sim.data.geom_xmat[geom_id].reshape(3, 3)
    size = sim.model.geom_size[geom_id]

    if geom_type == GEOM_SPHERE:
        extent = np.full(3, size[0])
    elif geom_type in (GEOM_BOX, GEOM_ELLIPSOID):
        extent = np.abs(rotation) @ size[:3]
    elif geom_type in (GEOM_CYLINDER, GEOM_CAPSULE):
        # The local z-axis is the cylinder/capsule axis. This is the exact
        # projected extent (unlike rotating a box around the primitive).
        axis = rotation[:, 2]
        radial_extent = size[0] * np.sqrt(np.maximum(0.0, 1.0 - axis * axis))
        axial_half_length = size[1]
        extent = radial_extent + axial_half_length * np.abs(axis)
        if geom_type == GEOM_CAPSULE:
            # A capsule is a cylinder plus spherical caps: the cap contributes
            # its radius in every world direction, not just along its axis.
            extent = size[0] + axial_half_length * np.abs(axis)
    elif geom_type == GEOM_MESH:
        mesh_id = int(sim.model.geom_dataid[geom_id])
        try:
            start = int(sim.model.mesh_vertadr[mesh_id])
            count = int(sim.model.mesh_vertnum[mesh_id])
            local_vertices = sim.model.mesh_vert[start : start + count]
            world_vertices = center + local_vertices @ rotation.T
            return world_vertices.min(axis=0), world_vertices.max(axis=0)
        except (AttributeError, IndexError, TypeError):
            # Every conveyor mesh in the current assets exposes vertices. The
            # fallback is retained for custom assets compiled without them.
            extent = np.full(3, sim.model.geom_rbound[geom_id])
    else:
        # A conservative runtime fallback for any future MuJoCo geom type.
        extent = np.full(3, sim.model.geom_rbound[geom_id])

    return center - extent, center + extent


def _fixture_footprint(sim: Any, root_body_id: int) -> dict[str, Any]:
    """Return the actual global footprint of a fixture's compiled geoms.

    In particular, this reads the runtime transforms of all conveyor_belt_1
    geoms after reset/stepping. It therefore cannot accidentally use the static
    ``conveyor_belt.xml`` position that reset later overwrites.
    """
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    geom_ids: list[int] = []

    for geom_id, geom_body_id in enumerate(sim.model.geom_bodyid):
        if not _is_descendant(sim, int(geom_body_id), root_body_id):
            continue
        aabb = _geom_world_aabb(sim, geom_id)
        if aabb is None:
            continue
        geom_ids.append(geom_id)
        mins.append(aabb[0])
        maxs.append(aabb[1])

    if not mins:
        raise ValueError(f"Body {root_body_id} has no finite geometry footprint.")

    world_min = np.min(np.stack(mins), axis=0)
    world_max = np.max(np.stack(maxs), axis=0)
    return {
        "body_position": _as_floats(sim.data.body_xpos[root_body_id]),
        "x": [float(world_min[0]), float(world_max[0])],
        "y": [float(world_min[1]), float(world_max[1])],
        "z": [float(world_min[2]), float(world_max[2])],
        "geom_count": len(geom_ids),
    }


def _point_margins(position: np.ndarray) -> dict[str, float]:
    """Margins for a translated CoM to remain inside the strict distribution."""
    x_negative = float(position[0] - DIST_BOUNDS["x_min"])
    x_positive = float(DIST_BOUNDS["x_max"] - position[0])
    y_negative = float(position[1] - DIST_BOUNDS["y_min"])
    y_positive = float(DIST_BOUNDS["y_max"] - position[1])
    return {
        "x_negative": x_negative,
        "x_positive": x_positive,
        "y_negative": y_negative,
        "y_positive": y_positive,
        "symmetric_shift_range": max(0.0, min(x_negative, x_positive, y_negative, y_positive)),
    }


def _footprint_margins(footprint: dict[str, Any]) -> dict[str, float]:
    """Physical footprint margins, used for the conveyor-belt X-axis check."""
    x_negative = float(footprint["x"][0] - DIST_BOUNDS["x_min"])
    x_positive = float(DIST_BOUNDS["x_max"] - footprint["x"][1])
    y_negative = float(footprint["y"][0] - DIST_BOUNDS["y_min"])
    y_positive = float(DIST_BOUNDS["y_max"] - footprint["y"][1])
    return {
        "x_negative": x_negative,
        "x_positive": x_positive,
        "y_negative": y_negative,
        "y_positive": y_positive,
        "fits_training_x_bounds": x_negative >= 0.0 and x_positive >= 0.0,
        "x_symmetric_shift_range": max(0.0, min(x_negative, x_positive)),
        "symmetric_shift_range": max(0.0, min(x_negative, x_positive, y_negative, y_positive)),
    }


def _is_in_strict_bounds(position: np.ndarray) -> bool:
    return bool(
        DIST_BOUNDS["x_min"] <= position[0] <= DIST_BOUNDS["x_max"]
        and DIST_BOUNDS["y_min"] <= position[1] <= DIST_BOUNDS["y_max"]
    )


def _suite_and_level(filename: str) -> tuple[str | None, int | None]:
    match = re.search(r"\b(D2D|D2S|S2D)(?:_TEST)?_LV(\d+)", filename.upper())
    if match is None:
        return None, None
    return match.group(1), int(match.group(2))


def _configured_shift_range(suite: str | None, level: int | None) -> float | None:
    if suite is None or level is None:
        return None
    suite_config = getattr(configs, suite, {})
    if level not in suite_config:
        return None
    return float(suite_config[level].get("shift_range", 0.0))


def _make_env(bddl_path: str) -> Any:
    """Build the regular LIBERO environment without the collection-only helpers."""
    # Keep the heavy simulator import here. Besides speeding up static checks,
    # this guarantees __main__ still prints its final config dictionary with an
    # actionable error if a host is missing a MuJoCo / robosuite dependency.
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        robots=["Panda"],
        gripper_types="default",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        ignore_done=True,
    )


def _settle(env: Any) -> None:
    env.reset()
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(env.action_dim))
    env.sim.forward()


def _measure_dynamic_grooves(env: Any) -> dict[str, Any] | None:
    groove_names = list(getattr(env, "dynamic_groove_names", ()))
    if not groove_names:
        return None

    grooves: dict[str, Any] = {}
    safe_ranges: list[float] = []
    in_bounds_count = 0
    for groove_name in groove_names:
        body_id = _body_id(env, groove_name)
        com = _main_body_com(env.sim, body_id)
        in_bounds = _is_in_strict_bounds(com)
        margins = _point_margins(com)
        in_bounds_count += int(in_bounds)
        safe_ranges.append(margins["symmetric_shift_range"])
        grooves[groove_name] = {
            "com": _as_floats(com),
            "in_strict_bounds": in_bounds,
            "margins": margins,
        }

    # A ghost layout is valid only when two complete dynamic grooves remain in
    # bounds. For symmetric ±shift perturbations, the second-largest individual
    # margin is the largest range that guarantees two such grooves.
    ordered_ranges = sorted(safe_ranges, reverse=True)
    required_two_range = ordered_ranges[1] if len(ordered_ranges) >= 2 else 0.0
    return {
        "required_in_bounds_grooves": 2,
        "in_bounds_grooves": in_bounds_count,
        "is_in_bounds": in_bounds_count >= 2,
        "max_safe_shift_range": float(required_two_range),
        "grooves": grooves,
    }


def _measure_layout(bddl_path: str) -> dict[str, Any]:
    filename = os.path.basename(bddl_path)
    suite, level = _suite_and_level(filename)
    env = _make_env(bddl_path)
    try:
        _settle(env)

        placements: dict[str, Any] = {}
        placement_ranges: list[float] = []
        for object_name in sorted(env.objects_dict):
            if object_name.startswith("dyn_"):
                continue
            body_id = _body_id(env, object_name)
            com = _main_body_com(env.sim, body_id)
            margins = _point_margins(com)
            placements[object_name] = {
                "com": _as_floats(com),
                "in_strict_bounds": _is_in_strict_bounds(com),
                "margins": margins,
            }
            placement_ranges.append(margins["symmetric_shift_range"])

        fixture_footprints: dict[str, Any] = {}
        for fixture_name in sorted(env.fixtures_dict):
            if fixture_name == "industry_workbench_1":
                continue
            try:
                fixture_footprints[fixture_name] = _fixture_footprint(
                    env.sim, _body_id(env, fixture_name)
                )
            except (KeyError, ValueError):
                # A fixture without finite geoms cannot contribute a spatial
                # footprint; object and groove checks remain valid.
                continue

        ghost = _measure_dynamic_grooves(env)
        belt = fixture_footprints.get("conveyor_belt_1")
        belt_x = _footprint_margins(belt) if belt is not None else None

        constraints = list(placement_ranges)
        if ghost is not None:
            constraints.append(ghost["max_safe_shift_range"])
        # The conveyor is intentionally long along its travel direction. Its
        # full Y footprint is not a training-position constraint, but its actual
        # X footprint must leave room for any X perturbation.
        if belt_x is not None:
            constraints.append(belt_x["x_symmetric_shift_range"])
        maximum_shift = max(0.0, min(constraints)) if constraints else 0.0

        configured_shift = _configured_shift_range(suite, level)
        return {
            "suite": suite,
            "level": level,
            "configured_shift_range": configured_shift,
            "max_safe_shift_range": float(maximum_shift),
        "configured_range_is_safe": (
                None if configured_shift is None else configured_shift <= maximum_shift
            ),
            "configured_belt_x_range_is_safe": (
                None
                if configured_shift is None or belt_x is None
                else configured_shift <= belt_x["x_symmetric_shift_range"]
            ),
            "placements": placements,
            "fixture_footprints": fixture_footprints,
            "conveyor_belt_1_x_margin": belt_x,
            "conveyor_ghost": ghost,
        }
    finally:
        env.close()


def _print_measurement(filename: str, result: dict[str, Any]) -> None:
    """Human-readable diagnostics; the final print remains a raw dict literal."""
    belt_x = result["conveyor_belt_1_x_margin"]
    ghost = result["conveyor_ghost"]
    print(f"\n[->] {filename}")
    print(f"     max safe symmetric shift: ±{result['max_safe_shift_range']:.6f} m")
    if belt_x is not None:
        print(
            "     conveyor_belt_1 runtime X footprint: "
            f"[{belt_x['x_negative'] + DIST_BOUNDS['x_min']:+.6f}, "
            f"{DIST_BOUNDS['x_max'] - belt_x['x_positive']:+.6f}] | "
            f"left margin={belt_x['x_negative']:+.6f}, "
            f"right margin={belt_x['x_positive']:+.6f}, "
            f"safe ±X shift={belt_x['x_symmetric_shift_range']:.6f}"
        )
    if ghost is not None:
        print(
            "     conveyor_ghost dynamic grooves in strict bounds: "
            f"{ghost['in_bounds_grooves']}/{len(ghost['grooves'])} "
            f"(requires {ghost['required_in_bounds_grooves']})"
        )


def analyze_compact_space() -> dict[str, Any]:
    bddl_files = sorted(glob.glob(os.path.join(TARGET_DIR, "**", "*.bddl"), recursive=True))
    if not bddl_files:
        raise FileNotFoundError(f"No BDDL files found in {TARGET_DIR}")

    by_layout: dict[str, Any] = {}
    grouped_limits: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    belt_x_limits: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    errors: dict[str, str] = {}

    print("=" * 72)
    print("RUNTIME MuJoCo SPATIAL ANALYZER")
    print("=" * 72)
    for bddl_path in bddl_files:
        filename = os.path.basename(bddl_path)
        try:
            result = _measure_layout(bddl_path)
            by_layout[filename] = result
            _print_measurement(filename, result)

            suite, level = result["suite"], result["level"]
            if suite is not None and level is not None:
                grouped_limits[suite][level].append(result["max_safe_shift_range"])
                belt_x = result["conveyor_belt_1_x_margin"]
                if belt_x is not None:
                    belt_x_limits[suite][level].append(belt_x["x_symmetric_shift_range"])
        except Exception as error:
            errors[filename] = f"{type(error).__name__}: {error}"
            print(f"\n[!] {filename}: {errors[filename]}")

    # A level can have several generated examples. Taking the minimum makes the
    # output safe for every measured procedural layout, not just the easiest one.
    max_safe_shift_range = {
        suite: {level: float(min(limits)) for level, limits in sorted(levels.items())}
        for suite, levels in sorted(grouped_limits.items())
    }
    conveyor_belt_x_shift_limit = {
        suite: {level: float(min(limits)) for level, limits in sorted(levels.items())}
        for suite, levels in sorted(belt_x_limits.items())
    }

    return {
        "DIST_BOUNDS": {key: float(value) for key, value in DIST_BOUNDS.items()},
        "MAX_SAFE_SHIFT_RANGE": max_safe_shift_range,
        "CONVEYOR_BELT_1_X_SHIFT_LIMIT": conveyor_belt_x_shift_limit,
        "LAYOUT_MEASUREMENTS": by_layout,
        "ERRORS": errors,
    }


if __name__ == "__main__":
    # Keep this as the last output: it is a raw Python dictionary literal that
    # can be pasted directly into configs.py (no NumPy scalars or JSON syntax).
    print(pprint.pformat(analyze_compact_space(), sort_dicts=False, width=120))
