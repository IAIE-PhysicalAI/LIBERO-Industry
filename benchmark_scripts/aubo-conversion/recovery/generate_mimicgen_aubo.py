#!/usr/bin/env python
"""Recover failed LIBERO-original Aubo identities with a MimicGen-style loop.

The authoritative Aubo input is ``LIBERO-datasets-auboi5h-raw``. Original
Panda HDF5 files are opened only through the input paths recorded by that raw
dataset. This runner does not import either Aubo trajectory-repair solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np

RAW_ROOT_BASENAME = "LIBERO-datasets-auboi5h-raw"
EXPECTED_ROBOSUITE_VERSION = "1.4.0"
_ACTION_DIMENSION = 7
INITIAL_STATE_ABSOLUTE_TOLERANCE = 1e-12
RAW_INITIALIZATION_SEED = 0
NOOP_EPSILON = 1e-4
DEFAULT_MINIMUM_JACOBIAN_SIGMA = 0.02
DEFAULT_MAXIMUM_JACOBIAN_CONDITION = 80.0
DEFAULT_JACOBIAN_ROTATION_LENGTH_M = 0.10
_OBJECT_NAMES = (
    "alphabet_soup",
    "bbq_sauce",
    "butter",
    "chocolate_pudding",
    "cream_cheese",
    "ketchup",
    "milk",
    "orange_juice",
    "salad_dressing",
    "tomato_sauce",
)
_SPATIAL_TASKS = (
    (
        "spatial_between_plate_ramekin",
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_table_center",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_top_drawer",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_cookie_box",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_next_to_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_next_to_ramekin",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_on_cookie_box",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_on_ramekin",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_on_stove",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo",
    ),
    (
        "spatial_on_wooden_cabinet",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo",
    ),
)
SIMPLE_TASKS = tuple(
    [
        {
            "case_id": f"object_{name}",
            "suite": "libero_object",
            "task_stem": f"pick_up_the_{name}_and_place_it_in_the_basket_demo",
            "object_name": f"{name}_1",
        }
        for name in _OBJECT_NAMES
    ]
    + [
        {
            "case_id": case_id,
            "suite": "libero_spatial",
            "task_stem": task_stem,
            "object_name": "akita_black_bowl_1",
            "goal_name": "plate_1",
        }
        for case_id, task_stem in _SPATIAL_TASKS
    ]
)
for _task_index, _task in enumerate(SIMPLE_TASKS):
    _task["task_index"] = _task_index


def _load_converter() -> Any:
    path = Path(__file__).resolve().parents[1] / "convert_to_aubo.py"
    spec = importlib.util.spec_from_file_location("aubo_recovery_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load converter from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def detect_gripper_events(actions: np.ndarray) -> Dict[int, str]:
    """Return every source close/open transition; multiple cycles are supported."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != _ACTION_DIMENSION:
        raise ValueError("Source actions must have shape (T, 7)")
    gripper = values[:, -1]
    if not np.all(np.isfinite(gripper)) or np.max(np.abs(gripper)) > 1.0 + 1e-12:
        raise ValueError("Source gripper actions must be finite and within [-1, 1]")
    events: Dict[int, str] = {}
    for index in range(1, len(gripper)):
        previous, current = float(gripper[index - 1]), float(gripper[index])
        if previous <= 0.0 < current:
            events[index] = "close"
        elif previous > 0.0 >= current:
            events[index] = "open"
    return events


def _bddl_path(raw: Any, converter: Any) -> Optional[str]:
    if raw is None:
        return None
    value = _decode(raw).replace("\\", "/")
    marker = "bddl_files/"
    if marker not in value:
        raise ValueError(f"Recorded BDDL path has no {marker!r} component: {value}")
    relative = value.split(marker, 1)[1]
    repository_root = Path(converter.__file__).resolve().parents[2]
    bddl_root = repository_root / "libero" / "libero" / "bddl_files"
    if not bddl_root.is_dir():
        bddl_root = Path(converter.get_libero_path("bddl_files"))
    resolved = bddl_root / relative
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Recorded BDDL {value!r} resolves to missing active-repository file "
            f"{resolved}"
        )
    return str(resolved)


def _make_environments(
    source_path: Path, raw_path: Path, converter: Any
) -> Tuple[Any, Any, str]:
    with h5py.File(source_path, "r") as source, h5py.File(raw_path, "r") as raw:
        source_meta = json.loads(_decode(source["data"].attrs["env_args"]))
        target_meta = json.loads(_decode(raw["data"].attrs["env_args"]))
        bddl_file = source["data"].attrs.get("bddl_file_name")

    for metadata in (source_meta, target_meta):
        kwargs = metadata.setdefault("env_kwargs", {})
        kwargs.update(
            {
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "use_camera_obs": False,
                "reward_shaping": False,
            }
        )
    env_name = source_meta.get("env_name") or target_meta.get("env_name")
    if not isinstance(env_name, str) or not env_name:
        raise ValueError("Source/raw HDF5 metadata has no environment name")
    expected_panda, expected_aubo = converter._domain_robot_models(env_name)
    resolved_bddl = _bddl_path(bddl_file, converter)
    if resolved_bddl is not None:
        source_meta["env_kwargs"]["bddl_file_name"] = resolved_bddl
        target_meta["env_kwargs"]["bddl_file_name"] = resolved_bddl
    source_meta["env_kwargs"]["robots"] = ["Panda"]
    source_meta["env_kwargs"]["gripper_types"] = ["PandaGripper"]
    target_meta["env_kwargs"]["robots"] = ["AuboI5"]
    target_meta["env_kwargs"]["gripper_types"] = ["CTAG2F90DGripper"]
    converter._set_controller_metadata(source_meta, preserve_legacy_config=True)
    converter._set_controller_metadata(target_meta)

    reference_env = converter.EnvUtils.create_env_for_data_processing(
        env_meta=source_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    target_env = converter.EnvUtils.create_env_for_data_processing(
        env_meta=target_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )
    converter._assert_runtime_robot(
        reference_env, expected_panda, "Panda recovery reference"
    )
    converter._assert_runtime_robot(target_env, expected_aubo, "Aubo recovery target")
    if converter._action_dimension(reference_env) != _ACTION_DIMENSION:
        raise RuntimeError("Panda recovery action dimension is not seven")
    if converter._action_dimension(target_env) != _ACTION_DIMENSION:
        raise RuntimeError("Aubo recovery action dimension is not seven")
    return reference_env, target_env, env_name


def _scene_values(environment: Any, converter: Any) -> Tuple[np.ndarray, int]:
    base = converter._get_base_env(environment)
    body_ids = getattr(base, "obj_body_id", {})
    values = []
    for name in sorted(body_ids):
        body_id = body_ids[name]
        values.extend(np.asarray(base.sim.data.body_xpos[body_id], dtype=np.float64))
        values.extend(np.asarray(base.sim.data.body_xquat[body_id], dtype=np.float64))
    return np.asarray(values, dtype=np.float64), len(body_ids)


def _initialize(
    reference_env: Any,
    target_env: Any,
    source_initial_state: np.ndarray,
    initial_gripper_action: float,
    seed: int,
    converter: Any,
    *,
    expected_target_state: Any = None,
) -> Dict[str, Any]:
    converter.initialize_target_replay(
        reference_env,
        target_env,
        source_initial_state,
        initial_gripper_action,
        _ACTION_DIMENSION,
        seed=seed,
    )
    state = converter.get_target_sim_state(target_env)
    state_difference = None
    if expected_target_state is not None:
        expected = np.asarray(expected_target_state)
        if expected.ndim != 1 or not np.all(np.isfinite(expected)):
            raise ValueError("Expected target state must be a finite vector")
        if state.shape != expected.shape:
            raise RuntimeError(
                f"Reconstructed Aubo initial-state shape {state.shape} differs "
                f"from stored raw state {expected.shape}"
            )
        state_difference = float(np.max(np.abs(state - expected)))
        if not np.array_equal(state, expected):
            base = converter._get_base_env(target_env)
            set_flattened = getattr(base.sim, "set_state_from_flattened", None)
            if not callable(set_flattened):
                raise AttributeError(
                    "Target simulator cannot restore the exact stored Aubo state"
                )
            set_flattened(expected)
            base.sim.forward()
            state = converter.get_target_sim_state(target_env)
            if not np.array_equal(state, expected):
                raise RuntimeError(
                    "Target simulator did not retain the exact stored Aubo state"
                )
    scene, body_count = _scene_values(target_env, converter)
    result = {
        "state": state,
        "scene": scene,
        "state_dimension": int(state.size),
        "named_task_body_count": body_count,
    }
    if expected_target_state is not None:
        result.update(
            {
                "target_state_max_absolute_difference": state_difference,
                "target_state_restored_exactly": bool(
                    np.array_equal(state, expected_target_state)
                ),
            }
        )
    return result


def _kinematic_metrics(
    target_env: Any, converter: Any, rotation_length_m: float
) -> Dict[str, float]:
    base = converter._get_base_env(target_env)
    robot = base.robots[0]
    arm_qpos, _ = converter.get_target_robot_qpos(target_env)
    raw_indexes = getattr(robot, "_ref_joint_indexes", None)
    if raw_indexes is None:
        raise RuntimeError("Aubo runtime exposes no arm joint indexes")
    indexes = np.asarray(
        converter._flatten_index_container(raw_indexes), dtype=np.int64
    )
    if indexes.shape != (len(arm_qpos),):
        raise RuntimeError(
            f"Aubo joint-index shape {indexes.shape} differs from arm qpos {arm_qpos.shape}"
        )
    sim = base.sim
    limits = np.asarray(sim.model.jnt_range[indexes], dtype=np.float64)
    margins = np.minimum(arm_qpos - limits[:, 0], limits[:, 1] - arm_qpos)
    if margins.shape != arm_qpos.shape or not np.all(np.isfinite(margins)):
        raise RuntimeError("Aubo joint margin is invalid")

    site_id = converter._right_eef_site_id(robot)
    site_name = sim.model.site_id2name(site_id)
    dof_indexes = np.asarray(sim.model.jnt_dofadr[indexes], dtype=np.int64)
    jacobian = np.vstack(
        (
            np.asarray(sim.data.get_site_jacp(site_name), dtype=np.float64).reshape(
                3, sim.model.nv
            ),
            np.asarray(sim.data.get_site_jacr(site_name), dtype=np.float64).reshape(
                3, sim.model.nv
            ),
        )
    )[:, dof_indexes]
    weighted = np.vstack(
        (jacobian[:3], float(rotation_length_m) * jacobian[3:])
    )
    singular_values = np.linalg.svd(weighted, compute_uv=False)
    if singular_values.shape != (len(arm_qpos),) or not np.all(
        np.isfinite(singular_values)
    ):
        raise RuntimeError("Aubo weighted Jacobian singular values are invalid")
    sigma_min = float(singular_values[-1])
    sigma_max = float(singular_values[0])
    condition = (
        float(np.finfo(np.float64).max)
        if sigma_min <= 1e-12
        else float(sigma_max / sigma_min)
    )

    base_position = None
    for body_name in ("robot0_base", "robot0_base_link"):
        try:
            body_id = sim.model.body_name2id(body_name)
        except ValueError:
            continue
        base_position = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
        break
    if base_position is None:
        raise RuntimeError("Aubo base body is unavailable for TCP reach measurement")
    tcp_position = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64)
    return {
        "minimum_arm_joint_margin_rad": float(np.min(margins)),
        "minimum_jacobian_sigma": sigma_min,
        "maximum_jacobian_condition_number": condition,
        "j3_rad": float(arm_qpos[2]),
        "tcp_reach_m": float(np.linalg.norm(tcp_position - base_position)),
    }


def _summarize_kinematics(rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Kinematic summary requires at least one state")
    j3 = np.asarray([row["j3_rad"] for row in rows], dtype=np.float64)
    return {
        "minimum_arm_joint_margin_rad": min(
            row["minimum_arm_joint_margin_rad"] for row in rows
        ),
        "minimum_jacobian_sigma": min(row["minimum_jacobian_sigma"] for row in rows),
        "maximum_jacobian_condition_number": max(
            row["maximum_jacobian_condition_number"] for row in rows
        ),
        "minimum_absolute_j3_rad": float(np.min(np.abs(j3))),
        "j3_zero_crossing_count": int(
            np.count_nonzero(np.signbit(j3[1:]) != np.signbit(j3[:-1]))
        ),
        "maximum_tcp_reach_m": max(row["tcp_reach_m"] for row in rows),
        "kinematic_state_count": len(rows),
    }


def _kinematic_rejection_reasons(
    metrics: Mapping[str, Any],
    minimum_joint_margin_rad: float,
    minimum_jacobian_sigma: float,
    maximum_jacobian_condition: float,
    prefix: str,
) -> list[str]:
    reasons = []
    if metrics["minimum_arm_joint_margin_rad"] < minimum_joint_margin_rad:
        reasons.append(f"{prefix}_joint_margin")
    if metrics["minimum_jacobian_sigma"] < minimum_jacobian_sigma:
        reasons.append(f"{prefix}_jacobian_sigma")
    if metrics["maximum_jacobian_condition_number"] > maximum_jacobian_condition:
        reasons.append(f"{prefix}_jacobian_condition")
    return reasons


def _trajectory_kinematics(
    states: np.ndarray,
    target_env: Any,
    converter: Any,
    rotation_length_m: float,
) -> Dict[str, Any]:
    base = converter._get_base_env(target_env)
    restore = getattr(base.sim, "set_state_from_flattened", None)
    if not callable(restore):
        raise AttributeError("Target simulator cannot restore stored Aubo states")
    rows = []
    for state in states:
        restore(state)
        base.sim.forward()
        rows.append(_kinematic_metrics(target_env, converter, rotation_length_m))
    return _summarize_kinematics(rows)


def path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True

LIBERO10_SUITE = "libero_10"

def _phase(
    kind: str,
    actor: str,
    predicate: Sequence[str],
    *,
    release: bool = True,
    goal_required: bool = True,
):
    if kind not in {"object", "fixture", "push"}:
        raise ValueError(f"Unsupported recovery phase kind: {kind}")
    return {
        "kind": kind,
        "actor": actor,
        "predicate": tuple(predicate),
        "release_required": bool(release),
        "goal_required": bool(goal_required),
    }


LIBERO10_TASKS = (
    {
        "case_id": "libero10_stove_moka",
        "task_stem": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
        "phases": (
            _phase("fixture", "flat_stove_1", ("turnon", "flat_stove_1")),
            _phase(
                "object",
                "moka_pot_1",
                ("on", "moka_pot_1", "flat_stove_1_cook_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_bowl_drawer",
        "task_stem": "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
        "phases": (
            _phase(
                "object",
                "akita_black_bowl_1",
                ("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
            ),
            _phase(
                "fixture",
                "white_cabinet_1",
                ("close", "white_cabinet_1_bottom_region"),
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_mug_microwave",
        "task_stem": "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo",
        "phases": (
            _phase(
                "object",
                "white_yellow_mug_1",
                ("in", "white_yellow_mug_1", "microwave_1_heating_region"),
            ),
            _phase("fixture", "microwave_1", ("close", "microwave_1")),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_two_moka",
        "task_stem": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo",
        "phases": (
            _phase(
                "object",
                "moka_pot_2",
                ("on", "moka_pot_2", "flat_stove_1_cook_region"),
            ),
            _phase(
                "object",
                "moka_pot_1",
                ("on", "moka_pot_1", "flat_stove_1_cook_region"),
                release=False,
            ),
        ),
        "invariants": (("turnon", "flat_stove_1"),),
    },
    {
        "case_id": "libero10_soup_cream",
        "task_stem": "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo",
        "phases": (
            _phase(
                "object",
                "alphabet_soup_1",
                ("in", "alphabet_soup_1", "basket_1_contain_region"),
            ),
            _phase(
                "object",
                "cream_cheese_1",
                ("in", "cream_cheese_1", "basket_1_contain_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_soup_tomato",
        "task_stem": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo",
        "phases": (
            _phase(
                "object",
                "alphabet_soup_1",
                ("in", "alphabet_soup_1", "basket_1_contain_region"),
            ),
            _phase(
                "object",
                "tomato_sauce_1",
                ("in", "tomato_sauce_1", "basket_1_contain_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_cream_butter",
        "task_stem": "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo",
        "phases": (
            _phase(
                "object",
                "cream_cheese_1",
                ("in", "cream_cheese_1", "basket_1_contain_region"),
            ),
            _phase(
                "object",
                "butter_1",
                ("in", "butter_1", "basket_1_contain_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_two_mugs",
        "task_stem": "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo",
        "independent_phase_sources": True,
        "phases": (
            _phase(
                "object",
                "porcelain_mug_1",
                ("on", "porcelain_mug_1", "plate_1"),
            ),
            _phase(
                "object",
                "white_yellow_mug_1",
                ("on", "white_yellow_mug_1", "plate_2"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_mug_pudding",
        "task_stem": "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo",
        "inter_phase_clearance": True,
        "phases": (
            _phase(
                "object",
                "porcelain_mug_1",
                ("on", "porcelain_mug_1", "plate_1"),
            ),
            _phase(
                "object",
                "chocolate_pudding_1",
                (
                    "on",
                    "chocolate_pudding_1",
                    "living_room_table_plate_right_region",
                ),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "libero10_book_caddy",
        "task_stem": "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo",
        "phases": (
            _phase(
                "object",
                "black_book_1",
                ("in", "black_book_1", "desk_caddy_1_back_contain_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
)

for _index, _task in enumerate(LIBERO10_TASKS):
    _task["suite"] = LIBERO10_SUITE
    _task["task_index"] = 100 + _index

GOAL_SUITE = "libero_goal"
GOAL_TASKS = (
    {
        "case_id": "goal_open_middle_drawer",
        "task_stem": "open_the_middle_drawer_of_the_cabinet_demo",
        "phases": (
            _phase(
                "fixture",
                "wooden_cabinet_1",
                ("open", "wooden_cabinet_1_middle_region"),
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_top_drawer_bowl",
        "task_stem": "open_the_top_drawer_and_put_the_bowl_inside_demo",
        "phases": (
            _phase(
                "fixture",
                "wooden_cabinet_1",
                ("open", "wooden_cabinet_1_top_region"),
                goal_required=False,
            ),
            _phase(
                "object",
                "akita_black_bowl_1",
                ("in", "akita_black_bowl_1", "wooden_cabinet_1_top_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_push_plate",
        "task_stem": "push_the_plate_to_the_front_of_the_stove_demo",
        "phases": (
            _phase(
                "push",
                "plate_1",
                ("on", "plate_1", "main_table_stove_front_region"),
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_bowl_plate",
        "task_stem": "put_the_bowl_on_the_plate_demo",
        "phases": (
            _phase(
                "object",
                "akita_black_bowl_1",
                ("on", "akita_black_bowl_1", "plate_1"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_bowl_stove",
        "task_stem": "put_the_bowl_on_the_stove_demo",
        "phases": (
            _phase(
                "object",
                "akita_black_bowl_1",
                ("on", "akita_black_bowl_1", "flat_stove_1_cook_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_bowl_cabinet",
        "task_stem": "put_the_bowl_on_top_of_the_cabinet_demo",
        "phases": (
            _phase(
                "object",
                "akita_black_bowl_1",
                ("on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_cream_cheese_bowl",
        "task_stem": "put_the_cream_cheese_in_the_bowl_demo",
        "phases": (
            _phase(
                "object",
                "cream_cheese_1",
                ("on", "cream_cheese_1", "akita_black_bowl_1"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_wine_rack",
        "task_stem": "put_the_wine_bottle_on_the_rack_demo",
        "phases": (
            _phase(
                "object",
                "wine_bottle_1",
                ("on", "wine_bottle_1", "wine_rack_1_top_region"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_wine_cabinet",
        "task_stem": "put_the_wine_bottle_on_top_of_the_cabinet_demo",
        "phases": (
            _phase(
                "object",
                "wine_bottle_1",
                ("on", "wine_bottle_1", "wooden_cabinet_1_top_side"),
                release=False,
            ),
        ),
        "invariants": (),
    },
    {
        "case_id": "goal_turn_on_stove",
        "task_stem": "turn_on_the_stove_demo",
        "phases": (
            _phase(
                "fixture", "flat_stove_1", ("turnon", "flat_stove_1")
            ),
        ),
        "invariants": (),
    },
)

for _index, _task in enumerate(GOAL_TASKS):
    _task["suite"] = GOAL_SUITE
    _task["task_index"] = 200 + _index

TASKS = SIMPLE_TASKS + LIBERO10_TASKS + GOAL_TASKS


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _demo_number(name: str) -> int:
    if not name.startswith("demo_"):
        raise ValueError(f"Invalid demonstration name: {name}")
    return int(name.split("_")[-1])


def _failed_demo_names(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(
        [name for name, row in rows.items() if row["terminal_task_success"] is False],
        key=_demo_number,
    )


def _raw_admission_reasons(
    row: Mapping[str, Any], actions: np.ndarray, rewards: np.ndarray
) -> list[str]:
    reasons = []
    if row["terminal_task_success"] is not True:
        reasons.append("online_task_failure")
    if rewards.shape != (len(actions),) or len(rewards) < 5 or not np.all(rewards[-5:] > 0):
        reasons.append("final_five_not_successful")
    if (
        actions.ndim != 2
        or actions.shape[1] != 7
        or len(actions) == 0
        or not np.all(np.isfinite(actions))
    ):
        reasons.append("invalid_actions")
    else:
        if np.max(np.abs(actions)) > 1.0 + 1e-6:
            reasons.append("action_out_of_range")
        noops = (np.linalg.norm(actions[1:, :6], axis=1) < NOOP_EPSILON) & (
            np.abs(np.diff(actions[:, 6])) <= NOOP_EPSILON
        )
        if np.any(noops):
            reasons.append("contains_noop")
    return reasons


def _attempt_rng(
    seed: int, task_index: int, target_demo: str, identity_attempt: int
) -> np.random.RandomState:
    value = seed + task_index * 100_000 + _demo_number(target_demo) * 1_000
    return np.random.RandomState(value + identity_attempt)


def _max_true_run(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _raw_success_clipping_limits(
    rows: Mapping[str, Mapping[str, Any]],
) -> Tuple[float, int]:
    """Use the selected raw terminal successes as the task's clipping envelope."""
    successes = [row for row in rows.values() if row["terminal_task_success"] is True]
    if not successes:
        raise ValueError("Task has no terminal-success raw trajectory for calibration")
    percentages = [float(row["clipped_component_pct"]) for row in successes]
    runs = [int(row["max_clipped_run_steps"]) for row in successes]
    if not all(math.isfinite(value) and 0.0 <= value <= 100.0 for value in percentages):
        raise ValueError("Raw successful clipping percentages are invalid")
    if any(value < 0 for value in runs):
        raise ValueError("Raw successful clipped-run lengths are invalid")
    return max(percentages), max(runs)


def _stagewise_source_rows(
    raw_rows: Mapping[str, Mapping[str, Any]],
    admitted_rows: Mapping[str, Mapping[str, Any]],
) -> Tuple[Mapping[str, Mapping[str, Any]], bool]:
    """Fall back to successful raw seeds without admitting them for training."""
    if admitted_rows:
        return admitted_rows, False
    return {
        name: row
        for name, row in raw_rows.items()
        if row["terminal_task_success"] is True
    }, True


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def make_pose(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def transform_segment(
    current_object_pose: np.ndarray,
    source_eef_poses: np.ndarray,
    source_object_pose: np.ndarray,
) -> np.ndarray:
    """Preserve every EEF pose relative to the segment's source object pose."""
    return np.asarray(
        [
            current_object_pose @ np.linalg.inv(source_object_pose) @ pose
            for pose in source_eef_poses
        ],
        dtype=np.float64,
    )


def _body_names(environment: Any, converter: Any) -> Sequence[str]:
    return sorted(converter._get_base_env(environment).obj_body_id)


def _resolve_body(environment: Any, object_name: str, converter: Any) -> str:
    if object_name not in _body_names(environment, converter):
        raise RuntimeError(
            f"Task body {object_name!r} is absent; "
            f"available={_body_names(environment, converter)}"
        )
    return object_name


def _body_pose(environment: Any, body_name: str, converter: Any) -> np.ndarray:
    task = converter._get_base_env(environment)
    body_id = task.obj_body_id[body_name]
    return make_pose(
        np.asarray(task.sim.data.body_xpos[body_id], dtype=np.float64),
        np.asarray(task.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3),
    )


def _pose_distance(left: np.ndarray, right: np.ndarray, converter: Any) -> float:
    position = float(np.linalg.norm(left[:3, 3] - right[:3, 3]))
    rotation = float(
        np.linalg.norm(converter.get_orientation_error(left[:3, :3], right[:3, :3]))
    )
    return position + 0.05 * rotation


def _diverse_indices(poses: Sequence[np.ndarray], count: int, converter: Any) -> list[int]:
    if count < 1 or len(poses) < count:
        raise ValueError("Source-library size exceeds eligible Panda demonstrations")
    selected = [0]
    while len(selected) < count:
        remaining = [index for index in range(len(poses)) if index not in selected]
        selected.append(
            max(
                remaining,
                key=lambda index: min(
                    _pose_distance(poses[index], poses[chosen], converter)
                    for chosen in selected
                ),
            )
        )
    return selected


def _segment_events(actions: np.ndarray) -> Tuple[int, int]:
    events = detect_gripper_events(actions)
    closes = [index for index, event in events.items() if event == "close"]
    if not closes:
        raise ValueError("Panda demonstration has no close event")
    close = closes[0]
    opens = [
        index for index, event in events.items() if event == "open" and index > close
    ]
    if not opens:
        raise ValueError("Panda demonstration has no release after its close event")
    return close, opens[0]


def _task_artifacts(raw_root: Path, task: Mapping[str, str]) -> Dict[str, Any]:
    suite_root = raw_root / task["suite"]
    raw_hdf5 = suite_root / f"{task['task_stem']}.hdf5"
    report_path = suite_root / f"{task['task_stem']}.conversion_report.json"
    for path in (raw_hdf5, report_path):
        if not path.is_file() or not path_is_within(path, raw_root):
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_hdf5 = Path(report.get("input_hdf5", "")).resolve()
    if not source_hdf5.is_file():
        raise FileNotFoundError(source_hdf5)
    if Path(report.get("output_hdf5", "")).resolve() != raw_hdf5.resolve():
        raise ValueError(f"{report_path} points to another Aubo HDF5")
    rows = report.get("trajectories")
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError(f"{report_path} must contain the 50 raw conversion outcomes")
    by_demo = {row.get("trajectory"): row for row in rows}
    if len(by_demo) != 50 or any(not str(name).startswith("demo_") for name in by_demo):
        raise ValueError(f"{report_path} has invalid or duplicate demo names")
    with h5py.File(raw_hdf5, "r") as raw:
        target_env_args = _decode(raw["data"].attrs["env_args"])
        raw_actions = {
            name: np.asarray(raw[f"data/{name}/actions"][()], dtype=np.float64)
            for name in by_demo
        }
        raw_rewards = {
            name: np.asarray(raw[f"data/{name}/rewards"][()], dtype=np.float64)
            for name in by_demo
        }
        raw_states = {
            name: np.asarray(raw[f"data/{name}/states"][()], dtype=np.float64)
            for name in by_demo
        }
        raw_initial_states = {}
        for name in by_demo:
            group = raw[f"data/{name}"]
            initial_state = np.asarray(group.attrs["init_state"])
            states = raw_states[name]
            states_zero = states[0]
            if (
                initial_state.ndim != 1
                or states.ndim != 2
                or len(states) != len(raw_actions[name])
                or not np.all(np.isfinite(states))
                or not np.all(np.isfinite(initial_state))
                or not np.array_equal(initial_state, states_zero)
            ):
                raise ValueError(
                    f"{raw_hdf5}:{name} has an invalid stored initial state"
                )
            raw_initial_states[name] = initial_state
    return {
        "source_hdf5": source_hdf5,
        "raw_hdf5": raw_hdf5,
        "report_path": report_path,
        "raw_rows": by_demo,
        "raw_actions": raw_actions,
        "raw_rewards": raw_rewards,
        "raw_states": raw_states,
        "raw_initial_states": raw_initial_states,
        "target_env_args": target_env_args,
    }


def _raw_kinematic_summaries(
    artifacts: Mapping[str, Any],
    target_env: Any,
    converter: Any,
    rotation_length_m: float,
) -> Dict[str, Dict[str, Any]]:
    return {
        name: _trajectory_kinematics(
            states, target_env, converter, rotation_length_m
        )
        for name, states in artifacts["raw_states"].items()
    }


def _screen_raw_kinematics(
    artifacts: Mapping[str, Any], converter: Any, rotation_length_m: float
) -> Dict[str, Dict[str, Any]]:
    reference_env = target_env = None
    try:
        reference_env, target_env, _ = _make_environments(
            artifacts["source_hdf5"], artifacts["raw_hdf5"], converter
        )
        return _raw_kinematic_summaries(
            artifacts, target_env, converter, rotation_length_m
        )
    finally:
        converter._close_env(reference_env)
        converter._close_env(target_env)


def _load_source(source_hdf5: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Any]]:
    demos: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(source_hdf5, "r") as source:
        data = source["data"]
        metadata = {
            "env_args": _decode(data.attrs["env_args"]),
            "bddl_file_name": _decode(data.attrs.get("bddl_file_name", "")),
        }
        for name in sorted(data, key=lambda value: int(value.split("_")[-1])):
            group = data[name]
            states = np.asarray(group["states"][()], dtype=np.float64)
            actions = np.asarray(group["actions"][()], dtype=np.float64)
            if states.ndim != 2 or actions.shape != (len(states), 7) or len(states) < 2:
                raise ValueError(f"{source_hdf5}:{name} has invalid states/actions")
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError(f"{source_hdf5}:{name} contains NaN or inf")
            demos[name] = {"states": states, "actions": actions}
    if len(demos) != 50:
        raise ValueError(f"{source_hdf5} must contain exactly 50 Panda demonstrations")
    return demos, metadata


def _state_action_demos(
    hdf5_path: Path,
    names: Sequence[str],
    *,
    require_exact_names: bool = False,
    recovered_case_id: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load finite target-native state/action trajectories without images."""
    selected = list(names)
    if len(selected) != len(set(selected)):
        raise ValueError(f"{hdf5_path} source demo names are duplicated")
    demos: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(hdf5_path, "r") as source:
        data = source["data"]
        if require_exact_names and set(data) != set(selected):
            raise ValueError(f"{hdf5_path} demos differ from its recovery summary")
        if recovered_case_id is not None and data.attrs.get(
            "state_action_observation_qualified"
        ) not in (True, np.bool_(True)):
            raise ValueError(f"{hdf5_path} is not a qualified recovery artifact")
        if recovered_case_id is not None and data.attrs.get(
            "kinematically_qualified"
        ) not in (True, np.bool_(True)):
            raise ValueError(f"{hdf5_path} is not kinematically qualified")
        for name in selected:
            if name not in data:
                raise ValueError(f"{hdf5_path} is missing {name}")
            group = data[name]
            states = np.asarray(group["states"][()], dtype=np.float64)
            actions = np.asarray(group["actions"][()], dtype=np.float64)
            if states.ndim != 2 or actions.shape != (len(states), 7):
                raise ValueError(f"{hdf5_path}:{name} has invalid states/actions")
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError(f"{hdf5_path}:{name} contains NaN or inf")
            if recovered_case_id is not None:
                if group.attrs.get("recovered_from_failed_identity") not in (
                    True,
                    np.bool_(True),
                ) or _decode(group.attrs.get("task_case_id", "")) != recovered_case_id:
                    raise ValueError(f"{hdf5_path}:{name} has invalid recovery metadata")
                rewards = np.asarray(group["rewards"][()])
                if rewards.shape != (len(states),) or not bool(rewards[-1] > 0.0):
                    raise ValueError(f"{hdf5_path}:{name} is not terminally successful")
            demos[name] = {"states": states, "actions": actions}
    return demos


def _recovered_source_demos(
    recovery_root: Optional[Path], task: Mapping[str, Any]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load accepted identities from one completed authoritative suite overlay."""
    if recovery_root is None:
        return {}
    root = Path(recovery_root).resolve()
    summary_path = root / task["case_id"] / "summary.json"
    if not path_is_within(summary_path, root):
        raise ValueError(f"Recovery summary escapes its source root: {summary_path}")
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("suite") != task["suite"]
        or summary.get("task_stem") != task["task_stem"]
    ):
        raise ValueError(f"{summary_path} does not describe the requested task")
    names = summary.get("recovered_demo_ids")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{summary_path} has invalid recovered demo names")
    if summary.get("recovered_identity_count") != len(names):
        raise ValueError(f"{summary_path} has inconsistent recovered counts")
    merged_value = summary.get("merged_hdf5")
    if not names:
        if merged_value is not None:
            raise ValueError(f"{summary_path} has an unexpected recovered HDF5")
        return {}
    merged = Path(merged_value).resolve() if isinstance(merged_value, str) else None
    if merged is None or not merged.is_file() or not path_is_within(merged, root):
        raise FileNotFoundError(merged)
    return _state_action_demos(
        merged,
        names,
        require_exact_names=True,
        recovered_case_id=task["case_id"],
    )


def _source_library(
    demos: Mapping[str, Mapping[str, np.ndarray]],
    reference_env: Any,
    object_name: str,
    count: int,
    converter: Any,
    goal_name: Optional[str] = None,
) -> list[Dict[str, Any]]:
    eligible = []
    for name, arrays in demos.items():
        try:
            close, release = _segment_events(arrays["actions"])
        except ValueError:
            continue
        reference_env.reset_to({"states": arrays["states"][0]})
        eligible.append(
            {
                "demo": name,
                "close": close,
                "release": release,
                "initial_object_pose": _body_pose(reference_env, object_name, converter),
            }
        )
    if not eligible:
        raise ValueError(f"No eligible source demonstration for {object_name}")
    chosen = _diverse_indices(
        [row["initial_object_pose"] for row in eligible],
        min(count, len(eligible)),
        converter,
    )
    library = []
    for index in chosen:
        row = eligible[index]
        arrays = demos[row["demo"]]
        eef_poses = []
        object_poses = []
        place_reference_poses = []
        for state in arrays["states"]:
            reference_env.reset_to({"states": state})
            position, rotation = converter.get_eef_pose(reference_env)
            eef_poses.append(make_pose(position, rotation))
            object_poses.append(_body_pose(reference_env, object_name, converter))
            place_reference_poses.append(
                _body_pose(reference_env, goal_name or object_name, converter)
            )
        eef_poses = np.asarray(eef_poses)
        # Actions at t target the next pose; the final target repeats the last pose.
        target_poses = np.concatenate((eef_poses[1:], eef_poses[-1:]), axis=0)
        initial_z = float(object_poses[0][2, 3])
        lifted = [
            step
            for step in range(row["close"], row["release"])
            if float(object_poses[step][2, 3]) >= initial_z + 0.005
        ]
        grasp = lifted[0] if lifted else min(row["close"] + 5, row["release"] - 1)
        library.append(
            {
                **row,
                "grasp": int(grasp),
                "target_poses": target_poses,
                "object_poses": np.asarray(object_poses),
                "place_reference_poses": np.asarray(place_reference_poses),
                "gripper": arrays["actions"][:, -1].copy(),
            }
        )
    return library


def _nearest_source(
    library: Sequence[Mapping[str, Any]],
    current_object_pose: np.ndarray,
    pose_key: str,
    nn_k: int,
    rng: np.random.RandomState,
    converter: Any,
) -> Mapping[str, Any]:
    ranked = sorted(
        library,
        key=lambda row: _pose_distance(
            np.asarray(row[pose_key]), current_object_pose, converter
        ),
    )
    return ranked[int(rng.randint(0, min(nn_k, len(ranked))))]


def _interpolate_pose(
    start: np.ndarray, end: np.ndarray, steps: int, converter: Any
) -> np.ndarray:
    if steps < 1:
        return np.empty((0, 4, 4), dtype=np.float64)
    delta_rotation = converter.get_orientation_error(end[:3, :3], start[:3, :3])
    rows = []
    for alpha in np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)[1:]:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * end[:3, 3]
        pose[:3, :3] = (
            converter.R.from_rotvec(alpha * delta_rotation).as_matrix()
            @ start[:3, :3]
        )
        rows.append(pose)
    return np.asarray(rows)


def _densify_reference_trajectory(
    poses: np.ndarray,
    gripper: np.ndarray,
    feedforward: Any,
    substeps: int,
    converter: Any,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Time-dilate a reference while preserving its endpoints and commands."""
    pose_rows = np.asarray(poses, dtype=np.float64)
    gripper_rows = np.asarray(gripper, dtype=np.float64)
    if pose_rows.ndim != 3 or pose_rows.shape[1:] != (4, 4) or not len(pose_rows):
        raise ValueError("Reference poses must be a non-empty sequence of 4x4 poses")
    if gripper_rows.shape != (len(pose_rows),) or not np.all(
        np.isfinite(gripper_rows)
    ):
        raise ValueError("Reference gripper commands must be finite shape (T,)")
    if substeps < 1:
        raise ValueError("Reference substeps must be positive")
    feedforward_rows = None
    if feedforward is not None:
        feedforward_rows = np.asarray(feedforward, dtype=np.float64)
        if feedforward_rows.shape != (len(pose_rows), 6) or not np.all(
            np.isfinite(feedforward_rows)
        ):
            raise ValueError("Feedforward Cartesian actions must be finite shape (T, 6)")
    if substeps == 1:
        return pose_rows, gripper_rows, feedforward_rows

    dense_poses = []
    dense_gripper = []
    dense_feedforward = []
    for index, target in enumerate(pose_rows):
        start = pose_rows[index - 1] if index else target
        dense_poses.append(_interpolate_pose(start, target, substeps, converter))
        dense_gripper.append(np.full(substeps, gripper_rows[index]))
        if feedforward_rows is not None:
            dense_feedforward.append(
                np.repeat(
                    (feedforward_rows[index] / substeps)[None], substeps, axis=0
                )
            )
    return (
        np.concatenate(dense_poses),
        np.concatenate(dense_gripper),
        np.concatenate(dense_feedforward) if dense_feedforward else None,
    )


def _new_rollout(
    initialization: Mapping[str, Any],
    rotation_length_m: float = DEFAULT_JACOBIAN_ROTATION_LENGTH_M,
) -> Dict[str, Any]:
    return {
        "actions": [],
        "states": [],
        "rewards": [],
        "dones": [],
        "subtask": [],
        "raw_clipped_components": [],
        "kinematics": [],
        "jacobian_rotation_length_m": float(rotation_length_m),
        "initial_state": np.asarray(initialization["state"]).copy(),
        "initial_scene": np.asarray(initialization["scene"]).copy(),
        "first_success_step": None,
        "terminal_success_streak": 0,
        "skipped_noop_count": 0,
        "success_validation_steps": 0,
    }


def _execute(
    environment: Any,
    poses: Iterable[np.ndarray],
    gripper: Iterable[float],
    subtask: int,
    rollout: Dict[str, Any],
    converter: Any,
    feedforward: Any = None,
) -> None:
    pose_rows = list(poses)
    gripper_rows = list(gripper)
    if len(pose_rows) != len(gripper_rows):
        raise ValueError("Pose and gripper trajectories must have equal lengths")
    if feedforward is None:
        feedforward_rows = [None] * len(pose_rows)
    else:
        feedforward_rows = list(feedforward)
        if len(feedforward_rows) != len(pose_rows):
            raise ValueError("Pose and feedforward trajectories must have equal lengths")
    for target, gripper_action, source_action in zip(
        pose_rows, gripper_rows, feedforward_rows
    ):
        current_position, current_rotation = converter.get_eef_pose(environment)
        _, _, raw_position, raw_rotation = (
            converter.calculate_tracking_action(
                current_position,
                current_rotation,
                target[:3, 3],
                target[:3, :3],
                return_raw=True,
            )
        )
        if source_action is not None:
            source_action = np.asarray(source_action, dtype=np.float64)
            if source_action.shape != (6,) or not np.all(np.isfinite(source_action)):
                raise ValueError("Feedforward Cartesian action must be finite shape (6,)")
            raw_position = raw_position + source_action[:3]
            raw_rotation = raw_rotation + source_action[3:]
        action_position = np.clip(raw_position, -1.0, 1.0)
        action_rotation = np.clip(raw_rotation, -1.0, 1.0)
        action = np.concatenate(
            (action_position, action_rotation, [float(gripper_action)])
        )
        if (
            rollout["actions"]
            and np.linalg.norm(action[:6]) < NOOP_EPSILON
            and abs(float(action[-1]) - float(rollout["actions"][-1][-1]))
            <= NOOP_EPSILON
        ):
            rollout["skipped_noop_count"] += 1
            continue
        rollout["states"].append(converter.get_target_sim_state(environment))
        rollout["kinematics"].append(
            _kinematic_metrics(
                environment, converter, rollout["jacobian_rotation_length_m"]
            )
        )
        rollout["actions"].append(action)
        rollout["subtask"].append(subtask)
        rollout["raw_clipped_components"].append(
            np.abs(np.concatenate((raw_position, raw_rotation))) > 1.0 + 1e-12
        )
        _, reward, done, _ = environment.step(action)
        rollout["rewards"].append(float(reward))
        rollout["dones"].append(bool(done))
        success = converter._task_success(environment)
        if success:
            rollout["terminal_success_streak"] += 1
            if rollout["first_success_step"] is None:
                rollout["first_success_step"] = len(rollout["actions"]) - 1
        else:
            rollout["terminal_success_streak"] = 0


def _validate_success_dwell(
    environment: Any,
    gripper_action: float,
    maximum_steps: int,
    required_streak: int,
    rollout: Dict[str, Any],
    converter: Any,
) -> None:
    """Settle the simulator without adding no-op transitions to the dataset."""
    if not converter._task_success(environment):
        return
    target_position, target_rotation = converter.get_eef_pose(environment)
    steps = 0
    while steps < maximum_steps and rollout["terminal_success_streak"] < required_streak:
        current_position, current_rotation = converter.get_eef_pose(environment)
        position, rotation = converter.calculate_tracking_action(
            current_position,
            current_rotation,
            target_position,
            target_rotation,
        )
        action = np.concatenate(
            (np.clip(position, -1.0, 1.0), np.clip(rotation, -1.0, 1.0), [gripper_action])
        )
        rollout["kinematics"].append(
            _kinematic_metrics(
                environment, converter, rollout["jacobian_rotation_length_m"]
            )
        )
        environment.step(action)
        steps += 1
        if converter._task_success(environment):
            rollout["terminal_success_streak"] += 1
        else:
            rollout["terminal_success_streak"] = 0
    rollout["success_validation_steps"] = steps


def _execute_subtask(
    environment: Any,
    transformed: np.ndarray,
    gripper: np.ndarray,
    subtask: int,
    rollout: Dict[str, Any],
    interpolation_steps: int,
    fixed_steps: int,
    converter: Any,
    feedforward: Any = None,
    reference_substeps: int = 1,
) -> None:
    current_position, current_rotation = converter.get_eef_pose(environment)
    current = make_pose(current_position, current_rotation)
    interpolation = _interpolate_pose(
        current, transformed[0], interpolation_steps, converter
    )
    bridge = np.concatenate(
        (interpolation, np.repeat(transformed[:1], fixed_steps, axis=0)), axis=0
    )
    _execute(
        environment,
        bridge,
        np.full(len(bridge), float(gripper[0])),
        subtask,
        rollout,
        converter,
    )
    dense_poses, dense_gripper, dense_feedforward = _densify_reference_trajectory(
        transformed,
        gripper,
        feedforward,
        reference_substeps,
        converter,
    )
    _execute(
        environment,
        dense_poses,
        dense_gripper,
        subtask,
        rollout,
        converter,
        feedforward=dense_feedforward,
    )


def _finalize_rollout(rollout: Dict[str, Any], environment: Any, converter: Any) -> Dict[str, Any]:
    rollout["kinematics"].append(
        _kinematic_metrics(
            environment, converter, rollout["jacobian_rotation_length_m"]
        )
    )
    arrays = {
        key: np.asarray(rollout[key])
        for key in ("actions", "states", "rewards", "dones", "subtask")
    }
    clipped = np.asarray(rollout["raw_clipped_components"], dtype=bool)
    clipped_steps = np.any(clipped, axis=1)
    finite = all(np.all(np.isfinite(arrays[key])) for key in ("actions", "states", "rewards"))
    bounded = bool(len(arrays["actions"]) and np.max(np.abs(arrays["actions"])) <= 1.0 + 1e-12)
    return {
        **arrays,
        "initial_state": rollout["initial_state"],
        "initial_scene": rollout["initial_scene"],
        "first_success_step": rollout["first_success_step"],
        "terminal_task_success": bool(converter._task_success(environment)),
        "terminal_success_streak": int(rollout["terminal_success_streak"]),
        "finite": bool(finite),
        "bounded_actions": bounded,
        **_summarize_kinematics(rollout["kinematics"]),
        "clipped_cartesian_component_pct": float(100.0 * np.mean(clipped)),
        "max_consecutive_clipped_steps": _max_true_run(clipped_steps),
        "skipped_noop_count": int(rollout["skipped_noop_count"]),
        "success_validation_steps": int(rollout["success_validation_steps"]),
    }


def _generate_attempt(
    *,
    target_demo: str,
    demos: Mapping[str, Mapping[str, np.ndarray]],
    library: Sequence[Mapping[str, Any]],
    object_name: str,
    goal_name: Optional[str],
    reference_env: Any,
    target_env: Any,
    expected_raw_initial_state: np.ndarray,
    rng: np.random.RandomState,
    arguments: argparse.Namespace,
    converter: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    initial = demos[target_demo]
    initialization = _initialize(
        reference_env,
        target_env,
        initial["states"][0],
        float(initial["actions"][0, -1]),
        arguments.seed,
        converter,
        expected_target_state=expected_raw_initial_state,
    )
    rollout = _new_rollout(initialization, arguments.jacobian_rotation_length_m)
    initial_object_pose = _body_pose(target_env, object_name, converter)
    pick = _nearest_source(
        library,
        initial_object_pose,
        "initial_object_pose",
        arguments.nearest_neighbors,
        rng,
        converter,
    )
    pick_offset = int(rng.randint(-arguments.boundary_offset, arguments.boundary_offset + 1))
    pick_end = int(np.clip(pick["grasp"] + pick_offset, pick["close"] + 1, pick["release"] - 1))
    pick_poses = transform_segment(
        initial_object_pose,
        pick["target_poses"][:pick_end],
        pick["object_poses"][0],
    )
    _execute_subtask(
        target_env,
        pick_poses,
        pick["gripper"][:pick_end],
        0,
        rollout,
        arguments.interpolation_steps,
        arguments.fixed_steps,
        converter,
        reference_substeps=arguments.reference_substeps,
    )

    current_object_pose = _body_pose(target_env, object_name, converter)
    current_place_reference_pose = (
        _body_pose(target_env, goal_name, converter)
        if goal_name is not None
        else current_object_pose
    )
    place_candidates = []
    for row in library:
        start = int(row["grasp"])
        place_candidates.append(
            {**row, "place_reference_pose": row["place_reference_poses"][start]}
        )
    place = _nearest_source(
        place_candidates,
        current_place_reference_pose,
        "place_reference_pose",
        arguments.nearest_neighbors,
        rng,
        converter,
    )
    place_offset = int(rng.randint(-arguments.boundary_offset, arguments.boundary_offset + 1))
    place_start = int(np.clip(place["grasp"] + place_offset, place["close"] + 1, place["release"] - 1))
    place_poses = transform_segment(
        current_place_reference_pose,
        place["target_poses"][place_start:],
        place["place_reference_poses"][place_start],
    )
    _execute_subtask(
        target_env,
        place_poses,
        place["gripper"][place_start:],
        1,
        rollout,
        arguments.interpolation_steps,
        arguments.fixed_steps,
        converter,
        reference_substeps=arguments.reference_substeps,
    )
    _validate_success_dwell(
        target_env,
        float(place["gripper"][-1]),
        arguments.final_hold_steps,
        arguments.success_dwell_steps,
        rollout,
        converter,
    )
    result = _finalize_rollout(rollout, target_env, converter)
    selection = {
        "target_initial_demo": target_demo,
        "pick_source_demo": pick["demo"],
        "place_source_demo": place["demo"],
        "pick_boundary_offset": pick_offset,
        "place_boundary_offset": place_offset,
        "place_reference_body": goal_name or object_name,
        "source_place_reference_position": place["place_reference_poses"][
            place_start
        ][:3, 3].tolist(),
        "target_place_reference_position": current_place_reference_pose[:3, 3].tolist(),
        "planned_place_endpoint_position": place_poses[-1, :3, 3].tolist(),
        "initial_object_position": initial_object_pose[:3, 3].tolist(),
        "post_pick_object_position": current_object_pose[:3, 3].tolist(),
        "raw_initial_state_max_absolute_difference": initialization[
            "target_state_max_absolute_difference"
        ],
        "raw_initial_state_restored_exactly": initialization[
            "target_state_restored_exactly"
        ],
    }
    return result, selection


def _replay(
    actions: np.ndarray,
    initial: Mapping[str, np.ndarray],
    reference_env: Any,
    target_env: Any,
    expected_raw_initial_state: np.ndarray,
    seed: int,
    converter: Any,
    rotation_length_m: float,
) -> Dict[str, Any]:
    initialization = _initialize(
        reference_env,
        target_env,
        initial["states"][0],
        float(actions[0, -1]),
        seed,
        converter,
        expected_target_state=expected_raw_initial_state,
    )
    states = []
    rewards = []
    dones = []
    kinematics = []
    success_streak = 0
    for action in actions:
        states.append(converter.get_target_sim_state(target_env))
        kinematics.append(
            _kinematic_metrics(target_env, converter, rotation_length_m)
        )
        _, reward, done, _ = target_env.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        success_streak = success_streak + 1 if converter._task_success(target_env) else 0
    kinematics.append(_kinematic_metrics(target_env, converter, rotation_length_m))
    return {
        "states": np.asarray(states),
        "rewards": np.asarray(rewards),
        "dones": np.asarray(dones),
        "initial_state": np.asarray(initialization["state"]),
        "initial_scene": np.asarray(initialization["scene"]),
        "terminal_task_success": bool(converter._task_success(target_env)),
        "terminal_success_streak": int(success_streak),
        **_summarize_kinematics(kinematics),
    }


def _qualified(
    generation: Mapping[str, Any],
    minimum_margin: float,
    success_dwell: int,
    maximum_clipped_component_pct: float,
    maximum_clipped_run_steps: int,
    maximum_horizon: int,
    minimum_jacobian_sigma: float = DEFAULT_MINIMUM_JACOBIAN_SIGMA,
    maximum_jacobian_condition: float = DEFAULT_MAXIMUM_JACOBIAN_CONDITION,
) -> Tuple[bool, list[str]]:
    reasons = []
    if generation["terminal_task_success"] is not True:
        reasons.append("generation_task_failure")
    if generation["terminal_success_streak"] < success_dwell:
        reasons.append("generation_success_dwell")
    if not generation["finite"] or not generation["bounded_actions"]:
        reasons.append("invalid_generation_arrays")
    reasons.extend(
        _kinematic_rejection_reasons(
            generation,
            minimum_margin,
            minimum_jacobian_sigma,
            maximum_jacobian_condition,
            "generation",
        )
    )
    if (
        generation["clipped_cartesian_component_pct"]
        > maximum_clipped_component_pct
    ):
        reasons.append("generation_clipped_component_pct")
    if generation["max_consecutive_clipped_steps"] > maximum_clipped_run_steps:
        reasons.append("generation_clipped_run")
    if len(generation["actions"]) > maximum_horizon:
        reasons.append("generation_horizon")
    return not reasons, reasons


def _camera_environment(
    source_hdf5: Path, raw_hdf5: Path, converter: Any
) -> Any:
    with h5py.File(source_hdf5, "r") as source, h5py.File(raw_hdf5, "r") as raw:
        target_meta = json.loads(_decode(raw["data"].attrs["env_args"]))
        bddl_file = source["data"].attrs.get("bddl_file_name")
    target_meta["env_kwargs"]["robots"] = ["AuboI5"]
    target_meta["env_kwargs"]["gripper_types"] = ["CTAG2F90DGripper"]
    if bddl_file is not None:
        target_meta["env_kwargs"]["bddl_file_name"] = _bddl_path(bddl_file, converter)
    converter._set_controller_metadata(target_meta)
    converter._set_target_observation_metadata(target_meta)
    return converter.EnvUtils.create_env_for_data_processing(
        env_meta=target_meta,
        camera_names=list(converter._TARGET_CAMERA_NAMES),
        camera_height=converter._TARGET_CAMERA_HEIGHT,
        camera_width=converter._TARGET_CAMERA_WIDTH,
        reward_shaping=False,
    )


def _observations(states: np.ndarray, environment: Any, converter: Any) -> Dict[str, np.ndarray]:
    values: Dict[str, list[np.ndarray]] = {
        "agentview_image": [],
        converter._WRIST_IMAGE_KEY: [],
        converter._ARM_JOINT_OBS_KEY: [],
        converter._GRIPPER_QPOS_OBS_KEY: [],
        converter._EEF_POSITION_OBS_KEY: [],
        converter._EEF_QUATERNION_OBS_KEY: [],
    }
    for state in states:
        observation = environment.reset_to({"states": state})
        if not isinstance(observation, dict):
            observation = converter._refresh_target_observation(environment)
        converter._assert_target_observation_contract(observation)
        if not np.array_equal(converter.get_target_sim_state(environment), state):
            raise RuntimeError("Observation materialization changed the stored Aubo state")
        arm, gripper = converter.get_target_robot_qpos(environment)
        position, rotation = converter.get_eef_pose(environment)
        values["agentview_image"].append(observation["agentview_image"])
        values[converter._WRIST_IMAGE_KEY].append(observation[converter._WRIST_IMAGE_KEY])
        values[converter._ARM_JOINT_OBS_KEY].append(arm)
        values[converter._GRIPPER_QPOS_OBS_KEY].append(gripper)
        values[converter._EEF_POSITION_OBS_KEY].append(position)
        values[converter._EEF_QUATERNION_OBS_KEY].append(
            converter.R.from_matrix(rotation).as_quat()
        )
    return {key: np.asarray(rows) for key, rows in values.items()}


def _write_episode(
    path: Path,
    task: Mapping[str, str],
    metadata: Mapping[str, Any],
    generation: Mapping[str, Any],
    observations: Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    with h5py.File(temporary, "w") as output:
        data = output.create_group("data")
        data.attrs["env_args"] = metadata["env_args"]
        data.attrs["bddl_file_name"] = metadata["bddl_file_name"]
        data.attrs["generation_method"] = "object_centric_subtask_recombination"
        data.attrs["state_action_observation_qualified"] = True
        data.attrs["kinematically_qualified"] = True
        data.attrs["comparable_full_dataset_ready"] = False
        demo_name = str(selection["target_initial_demo"])
        demo = data.create_group(demo_name)
        for key in ("actions", "states", "rewards", "dones", "subtask"):
            demo.create_dataset(key, data=generation[key])
        obs = demo.create_group("obs")
        for key, values in observations.items():
            obs.create_dataset(key, data=values, compression="gzip")
        demo.attrs["num_samples"] = len(generation["actions"])
        demo.attrs["init_state"] = generation["states"][0]
        demo.attrs["task_case_id"] = task["case_id"]
        demo.attrs["recovered_from_failed_identity"] = True
        demo.attrs["terminal_success_streak"] = generation["terminal_success_streak"]
        demo.attrs["skipped_noop_count"] = generation["skipped_noop_count"]
        demo.attrs["success_validation_steps"] = generation[
            "success_validation_steps"
        ]
        for key in (
            "minimum_arm_joint_margin_rad",
            "minimum_jacobian_sigma",
            "maximum_jacobian_condition_number",
            "minimum_absolute_j3_rad",
            "j3_zero_crossing_count",
            "maximum_tcp_reach_m",
            "kinematic_state_count",
        ):
            demo.attrs[key] = generation[key]
        if "maximum_horizon" in generation:
            demo.attrs["maximum_horizon"] = generation["maximum_horizon"]
        for key, value in selection.items():
            demo.attrs[key] = json.dumps(value) if isinstance(value, list) else value
    with h5py.File(temporary, "r") as output:
        demo = output[f"data/{selection['target_initial_demo']}"]
        if not np.array_equal(demo["actions"][()], generation["actions"]):
            raise RuntimeError("Written generated actions differ on readback")
        if not np.array_equal(demo["states"][()], generation["states"]):
            raise RuntimeError("Written generated states differ on readback")
        if len(demo["obs/agentview_image"]) != len(generation["actions"]):
            raise RuntimeError("Written observation horizon differs on readback")
    os.replace(temporary, path)


def _merge_episodes(paths: Sequence[Path], output_path: Path) -> None:
    if not paths:
        return
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    copied = 0
    with h5py.File(temporary, "w") as output:
        data = output.create_group("data")
        for index, path in enumerate(paths):
            with h5py.File(path, "r") as source:
                if index == 0:
                    for key, value in source["data"].attrs.items():
                        data.attrs[key] = value
                demo_names = list(source["data"])
                if not demo_names or any(name in data for name in demo_names):
                    raise RuntimeError(f"Invalid or duplicate recovered identity in {path}")
                for name in demo_names:
                    source.copy(f"data/{name}", data, name=name)
                    copied += 1
        data.attrs["total"] = copied
    os.replace(temporary, output_path)
    with h5py.File(output_path, "r") as output:
        if len(output["data"]) != copied:
            raise RuntimeError("Merged recovery count differs on readback")


def _resume_tasks(
    resume_root: Path,
) -> Dict[str, Optional[Mapping[str, Any]]]:
    """Load completed and explicitly errored tasks from a prior round."""
    summary_path = Path(resume_root).resolve() / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tasks = summary.get("tasks")
    errors = summary.get("errors")
    if (
        not isinstance(tasks, list)
        or not isinstance(errors, list)
        or summary.get("errored_task_count") != len(errors)
    ):
        raise ValueError(f"Resume summary is incomplete or errored: {summary_path}")
    by_case = {}
    for task in tasks:
        case_id = task.get("case_id") if isinstance(task, dict) else None
        if not isinstance(case_id, str) or case_id in by_case:
            raise ValueError(f"Resume summary has an invalid task: {summary_path}")
        for key in ("recovered_demo_ids", "unrecovered_demo_ids", "not_attempted_demo_ids"):
            if not isinstance(task.get(key), list):
                raise ValueError(f"Resume task has no valid {key}: {case_id}")
        by_case[case_id] = task
    for error in errors:
        case_id = error.get("case_id") if isinstance(error, dict) else None
        if not isinstance(case_id, str) or case_id in by_case:
            raise ValueError(f"Resume summary has an invalid task error: {summary_path}")
        by_case[case_id] = None
    return by_case


def _resume_partition(
    task: Mapping[str, Any], all_targets: Sequence[str]
) -> Tuple[list[str], list[str], Optional[Path]]:
    carried = list(task["recovered_demo_ids"])
    remaining = list(task["unrecovered_demo_ids"]) + list(
        task["not_attempted_demo_ids"]
    )
    if (
        len(set(carried + remaining)) != len(carried) + len(remaining)
        or set(carried) | set(remaining) != set(all_targets)
    ):
        raise ValueError(
            f"Resume summary does not partition targets for {task['case_id']}"
        )
    merged_value = task.get("merged_hdf5")
    merged = Path(merged_value) if merged_value else None
    if carried:
        if merged is None or not merged.is_file():
            raise FileNotFoundError(merged)
        with h5py.File(merged, "r") as previous:
            if set(previous["data"]) != set(carried):
                raise ValueError(
                    f"Resume HDF5 differs from {task['case_id']} summary"
                )
    elif merged is not None:
        raise ValueError(f"Resume task has HDF5 but no recoveries: {task['case_id']}")
    return carried, remaining, merged


def _write_video(path: Path, frames: np.ndarray) -> None:
    import imageio.v2 as imageio

    path = path.resolve()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    imageio.mimsave(temporary, list(frames), fps=10)
    os.replace(temporary, path)


def _run_task(
    task: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    arguments: argparse.Namespace,
    converter: Any,
) -> Dict[str, Any]:
    artifacts = _task_artifacts(raw_root, task)
    raw_failed_demos = _failed_demo_names(artifacts["raw_rows"])
    raw_kinematics = _screen_raw_kinematics(
        artifacts, converter, arguments.jacobian_rotation_length_m
    )
    raw_rejection_reasons = {
        name: _raw_admission_reasons(
            row,
            artifacts["raw_actions"][name],
            artifacts["raw_rewards"][name],
        )
        + _kinematic_rejection_reasons(
            raw_kinematics[name],
            arguments.min_arm_joint_margin_rad,
            arguments.minimum_jacobian_sigma,
            arguments.maximum_jacobian_condition,
            "raw",
        )
        for name, row in artifacts["raw_rows"].items()
    }
    all_target_demos = sorted(
        (name for name, reasons in raw_rejection_reasons.items() if reasons),
        key=_demo_number,
    )
    verified_source_rows = {
        name: row
        for name, row in artifacts["raw_rows"].items()
        if not raw_rejection_reasons[name]
    }
    raw_clipped_pct, raw_clipped_run = _raw_success_clipping_limits(
        verified_source_rows
    )
    maximum_clipped_component_pct = (
        raw_clipped_pct
        if arguments.max_clipped_component_pct is None
        else arguments.max_clipped_component_pct
    )
    maximum_clipped_run_steps = (
        raw_clipped_run
        if arguments.max_clipped_run_steps is None
        else arguments.max_clipped_run_steps
    )
    demos, source_metadata = _load_source(artifacts["source_hdf5"])
    maximum_horizon = max(len(row["actions"]) for row in demos.values()) + (
        2 * (arguments.interpolation_steps + arguments.fixed_steps)
    )
    reference_env = target_env = camera_env = None
    attempt_rows = []
    identity_rows = []
    accepted_paths: list[Path] = []
    accepted_demos: list[str] = []
    task_dir = output_root / task["case_id"]
    task_dir.mkdir()
    attempts_path = task_dir / "attempts.jsonl"
    ledger_path = task_dir / "recovery_ledger.jsonl"
    try:
        reference_env, target_env, environment_name = _make_environments(
            artifacts["source_hdf5"], artifacts["raw_hdf5"], converter
        )
        object_name = _resolve_body(reference_env, task["object_name"], converter)
        if _resolve_body(target_env, task["object_name"], converter) != object_name:
            raise RuntimeError("Panda and Aubo object body names differ")
        goal_name = task.get("goal_name")
        if goal_name is not None:
            goal_name = _resolve_body(reference_env, goal_name, converter)
            if _resolve_body(target_env, goal_name, converter) != goal_name:
                raise RuntimeError("Panda and Aubo goal body names differ")
        if arguments.recovered_source_root is None:
            library_demos = demos
            library_environment = reference_env
            source_policy = "original_panda"
        else:
            raw_source_names = sorted(
                (
                    name
                    for name, row in verified_source_rows.items()
                    if row["terminal_task_success"] is True
                ),
                key=_demo_number,
            )
            library_demos = _state_action_demos(
                artifacts["raw_hdf5"], raw_source_names
            )
            recovered_sources = _recovered_source_demos(
                arguments.recovered_source_root, task
            )
            if set(library_demos) & set(recovered_sources):
                raise ValueError("Raw and recovered source identities overlap")
            library_demos.update(recovered_sources)
            library_environment = target_env
            source_policy = "verified_raw_plus_recovered_aubo"
        library = _source_library(
            library_demos,
            library_environment,
            object_name,
            arguments.source_demos,
            converter,
            goal_name,
        )
        print(
            f"[TASK READY] {task['case_id']} object={object_name} "
            f"source_policy={source_policy} "
            f"source_pool={len(library_demos)} sources={len(library)} "
            f"clipping_limits={maximum_clipped_component_pct:.6g}%/"
            f"{maximum_clipped_run_steps}_steps",
            flush=True,
        )
        resume_task = arguments.resume_tasks.get(task["case_id"])
        carried_forward_demos: list[str] = []
        if resume_task is not None:
            carried_forward_demos, previous_remaining, previous_merged = (
                _resume_partition(resume_task, all_target_demos)
            )
            if previous_merged is not None:
                accepted_paths.append(previous_merged)
                accepted_demos.extend(carried_forward_demos)
        if arguments.target_demo:
            if arguments.target_demo not in all_target_demos:
                raise ValueError(
                    f"{arguments.target_demo} already passes raw admission "
                    f"for {task['case_id']}"
                )
            target_demos = [arguments.target_demo]
        elif resume_task is not None:
            target_demos = previous_remaining
        else:
            target_demos = all_target_demos[: arguments.limit_identities]
        if not target_demos and resume_task is None:
            raise RuntimeError(f"{task['case_id']} has an empty target reset stratum")
        for target_demo in target_demos:
            raw_row = artifacts["raw_rows"][target_demo]
            accepted_path = None
            attempts_for_identity = 0
            last_reasons: list[str] = []
            for identity_attempt in range(1, arguments.attempts_per_identity + 1):
                attempts_for_identity = identity_attempt
                rng = _attempt_rng(
                    arguments.seed,
                    int(task["task_index"]),
                    target_demo,
                    identity_attempt,
                )
                generation, selection = _generate_attempt(
                    target_demo=target_demo,
                    demos=demos,
                    library=library,
                    object_name=object_name,
                    goal_name=goal_name,
                    reference_env=reference_env,
                    target_env=target_env,
                    expected_raw_initial_state=artifacts["raw_initial_states"][
                        target_demo
                    ],
                    rng=rng,
                    arguments=arguments,
                    converter=converter,
                )
                generation["maximum_horizon"] = maximum_horizon
                accepted, reasons = _qualified(
                    generation,
                    arguments.min_arm_joint_margin_rad,
                    arguments.success_dwell_steps,
                    maximum_clipped_component_pct,
                    maximum_clipped_run_steps,
                    maximum_horizon,
                    arguments.minimum_jacobian_sigma,
                    arguments.maximum_jacobian_condition,
                )
                diagnostic_replay = None
                if accepted:
                    diagnostic_replay = _replay(
                        generation["actions"],
                        demos[target_demo],
                        reference_env,
                        target_env,
                        artifacts["raw_initial_states"][target_demo],
                        arguments.seed,
                        converter,
                        arguments.jacobian_rotation_length_m,
                    )
                last_reasons = reasons
                row = {
                    "attempt": len(attempt_rows) + 1,
                    "identity_attempt": identity_attempt,
                    **selection,
                    "raw_before_terminal_success": raw_row["terminal_task_success"],
                    "raw_before_ever_success": raw_row["ever_task_success"],
                    "raw_rejection_reasons": raw_rejection_reasons[target_demo],
                    "generated_terminal_success": generation[
                        "terminal_task_success"
                    ],
                    "generation_success_streak": generation[
                        "terminal_success_streak"
                    ],
                    "success_validation_steps": generation[
                        "success_validation_steps"
                    ],
                    "diagnostic_replay_terminal_success": (
                        diagnostic_replay["terminal_task_success"]
                        if diagnostic_replay is not None
                        else None
                    ),
                    "diagnostic_replay_success_streak": (
                        diagnostic_replay["terminal_success_streak"]
                        if diagnostic_replay is not None
                        else None
                    ),
                    "accepted": accepted,
                    "rejection_reasons": reasons,
                    "control_steps": len(generation["actions"]),
                    "minimum_arm_joint_margin_rad": generation[
                        "minimum_arm_joint_margin_rad"
                    ],
                    "minimum_jacobian_sigma": generation["minimum_jacobian_sigma"],
                    "maximum_jacobian_condition_number": generation[
                        "maximum_jacobian_condition_number"
                    ],
                    "minimum_absolute_j3_rad": generation["minimum_absolute_j3_rad"],
                    "j3_zero_crossing_count": generation["j3_zero_crossing_count"],
                    "maximum_tcp_reach_m": generation["maximum_tcp_reach_m"],
                    "clipped_cartesian_component_pct": generation[
                        "clipped_cartesian_component_pct"
                    ],
                    "max_consecutive_clipped_steps": generation[
                        "max_consecutive_clipped_steps"
                    ],
                    "skipped_noop_count": generation["skipped_noop_count"],
                    "maximum_clipped_component_pct": (
                        maximum_clipped_component_pct
                    ),
                    "maximum_clipped_run_steps": maximum_clipped_run_steps,
                    "maximum_horizon": maximum_horizon,
                }
                if accepted:
                    if camera_env is None:
                        camera_env = _camera_environment(
                            artifacts["source_hdf5"],
                            artifacts["raw_hdf5"],
                            converter,
                        )
                    observations = _observations(
                        generation["states"], camera_env, converter
                    )
                    if arguments.diagnostic_video:
                        _write_video(
                            Path(arguments.diagnostic_video),
                            observations["agentview_image"],
                        )
                    else:
                        accepted_path = task_dir / f"recovered_{target_demo}.hdf5"
                        episode_metadata = dict(source_metadata)
                        episode_metadata["env_args"] = artifacts["target_env_args"]
                        _write_episode(
                            accepted_path,
                            task,
                            episode_metadata,
                            generation,
                            observations,
                            selection,
                        )
                        accepted_paths.append(accepted_path)
                        row["accepted_hdf5"] = str(accepted_path)
                    accepted_demos.append(target_demo)
                    row["diagnostic_video"] = arguments.diagnostic_video
                attempt_rows.append(row)
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                print(
                    f"[ATTEMPT] task={task['case_id']} identity={target_demo} "
                    f"n={identity_attempt}/{arguments.attempts_per_identity} "
                    f"generated={generation['terminal_task_success']} "
                    f"diagnostic_replay={row['diagnostic_replay_terminal_success']} "
                    f"accepted={accepted}",
                    flush=True,
                )
                if accepted:
                    break
            identity_row = {
                "task": task["case_id"],
                "target_initial_demo": target_demo,
                "status": "recovered" if target_demo in accepted_demos else "unrecovered",
                "attempts": attempts_for_identity,
                "raw_before_terminal_success": raw_row["terminal_task_success"],
                "raw_before_ever_success": raw_row["ever_task_success"],
                "diagnostic_replay_terminal_success": next(
                    (
                        row["diagnostic_replay_terminal_success"]
                        for row in reversed(attempt_rows)
                        if row["target_initial_demo"] == target_demo
                        and row["accepted"]
                    ),
                    None,
                ),
                "accepted_hdf5": str(accepted_path) if accepted_path else None,
                "diagnostic_video": arguments.diagnostic_video
                if target_demo in accepted_demos
                else None,
                "final_rejection_reasons": last_reasons
                if target_demo not in accepted_demos
                else [],
            }
            identity_rows.append(identity_row)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(identity_row, sort_keys=True, allow_nan=False) + "\n"
                )
            print(
                f"[IDENTITY END] task={task['case_id']} identity={target_demo} "
                f"status={identity_row['status']} attempts={attempts_for_identity}",
                flush=True,
            )
            if arguments.diagnostic_video and target_demo in accepted_demos:
                break
        merged = task_dir / f"{task['task_stem']}.generated.hdf5"
        if not arguments.diagnostic_video:
            _merge_episodes(accepted_paths, merged)
        processed_demos = carried_forward_demos + [
            row["target_initial_demo"] for row in identity_rows
        ]
        unrecovered_demos = [
            row["target_initial_demo"]
            for row in identity_rows
            if row["status"] == "unrecovered"
        ]
        summary = {
            "case_id": task["case_id"],
            "suite": task["suite"],
            "task_stem": task["task_stem"],
            "environment": environment_name,
            "authoritative_raw_root": str(raw_root),
            "raw_aubo_hdf5": str(artifacts["raw_hdf5"]),
            "raw_conversion_report": str(artifacts["report_path"]),
            "linked_original_panda_hdf5": str(artifacts["source_hdf5"]),
            "source_library": [row["demo"] for row in library],
            "target_initial_stratum": "raw_without_stable_final_five_success",
            "raw_failed_identity_count": len(raw_failed_demos),
            "raw_unstable_success_identity_count": sum(
                artifacts["raw_rows"][name]["terminal_task_success"] is True
                for name in all_target_demos
            ),
            "raw_retained_identity_count": len(verified_source_rows),
            "raw_kinematic_rejection_identity_count": sum(
                any(
                    reason.startswith("raw_jacobian_")
                    or reason == "raw_joint_margin"
                    for reason in reasons
                )
                for reasons in raw_rejection_reasons.values()
            ),
            "target_identity_count": len(all_target_demos),
            "selected_failed_identity_count": len(carried_forward_demos)
            + len(target_demos),
            "processed_identity_count": len(processed_demos),
            "attempt_count": len(attempt_rows),
            "diagnostic_replay_count": sum(
                row["diagnostic_replay_terminal_success"] is not None
                for row in attempt_rows
            ),
            "diagnostic_replay_success_count": sum(
                row["diagnostic_replay_terminal_success"] is True
                for row in attempt_rows
            ),
            "generated_terminal_success_count": sum(
                row["generated_terminal_success"] for row in attempt_rows
            ),
            "recovered_identity_count": len(accepted_demos),
            "unrecovered_identity_count": len(unrecovered_demos),
            "recovered_demo_ids": accepted_demos,
            "unrecovered_demo_ids": unrecovered_demos,
            "not_attempted_demo_ids": [
                name for name in all_target_demos if name not in processed_demos
            ],
            "all_selected_identities_processed": len(identity_rows)
            == len(target_demos)
            and len(processed_demos)
            == len(carried_forward_demos) + len(target_demos),
            "merged_hdf5": str(merged) if accepted_paths else None,
            "continuation_from": str(arguments.resume_from)
            if arguments.resume_from
            else None,
            "carried_forward_recovered_identity_count": len(
                carried_forward_demos
            ),
            "diagnostic_only": bool(arguments.diagnostic_video),
            "diagnostic_video": arguments.diagnostic_video,
            "hdf5_output_enabled": not bool(arguments.diagnostic_video),
            "camera_observations_materialized_for_accepted_only": True,
            "source_library_policy": source_policy,
            "source_pool_identity_count": len(library_demos),
            "recovered_source_root": str(arguments.recovered_source_root)
            if arguments.recovered_source_root is not None
            else None,
            "clipping_acceptance_policy": (
                "command_line_override"
                if arguments.max_clipped_component_pct is not None
                or arguments.max_clipped_run_steps is not None
                else "task_raw_terminal_success_maximum"
            ),
            "maximum_clipped_component_pct": maximum_clipped_component_pct,
            "maximum_clipped_run_steps": maximum_clipped_run_steps,
            "maximum_horizon": maximum_horizon,
        }
        _atomic_json(task_dir / "summary.json", summary)
        return summary
    finally:
        converter._close_env(camera_env)
        converter._close_env(reference_env)
        converter._close_env(target_env)


def _preflight_task(
    task: Mapping[str, Any],
    raw_root: Path,
    converter: Any,
    arguments: argparse.Namespace,
) -> Dict[str, Any]:
    artifacts = _task_artifacts(raw_root, task)
    reference_env = target_env = None
    try:
        reference_env, target_env, environment_name = _make_environments(
            artifacts["source_hdf5"], artifacts["raw_hdf5"], converter
        )
        reference_body = _resolve_body(reference_env, task["object_name"], converter)
        target_body = _resolve_body(target_env, task["object_name"], converter)
        if reference_body != target_body:
            raise RuntimeError("Panda and Aubo object body names differ")
        raw_kinematics = _raw_kinematic_summaries(
            artifacts,
            target_env,
            converter,
            arguments.jacobian_rotation_length_m,
        )
        raw_kinematic_reasons = {
            name: _kinematic_rejection_reasons(
                metrics,
                arguments.min_arm_joint_margin_rad,
                arguments.minimum_jacobian_sigma,
                arguments.maximum_jacobian_condition,
                "raw",
            )
            for name, metrics in raw_kinematics.items()
        }
        return {
            "case_id": task["case_id"],
            "suite": task["suite"],
            "environment": environment_name,
            "object_name": reference_body,
            "failed_identity_count": len(
                _failed_demo_names(artifacts["raw_rows"])
            ),
            "target_identity_count": sum(
                bool(
                    _raw_admission_reasons(
                        row,
                        artifacts["raw_actions"][name],
                        artifacts["raw_rewards"][name],
                    )
                    + raw_kinematic_reasons[name]
                )
                for name, row in artifacts["raw_rows"].items()
            ),
            "raw_kinematic_rejection_identity_count": sum(
                bool(reasons) for reasons in raw_kinematic_reasons.values()
            ),
        }
    finally:
        converter._close_env(reference_env)
        converter._close_env(target_env)


def _goal(environment: Any, predicate: Sequence[str], converter: Any) -> bool:
    base_environment = converter._get_base_env(environment)
    evaluator = getattr(base_environment, "_eval_predicate", None)
    if not callable(evaluator):
        raise RuntimeError("LIBERO environment exposes no live predicate evaluator")
    return bool(evaluator(list(predicate)))


def _goal_snapshot(
    environment: Any, predicates: Iterable[Sequence[str]], converter: Any
) -> Dict[str, bool]:
    return {
        " ".join(predicate): _goal(environment, predicate, converter)
        for predicate in predicates
    }


def _phase_snapshot_predicates(
    task: Mapping[str, Any],
    persistent_completed: Sequence[Sequence[str]],
    current_phase: Mapping[str, Any],
) -> list[Tuple[str, ...]]:
    """Check temporary prerequisites now, but persist only declared goal phases."""
    return [
        *(tuple(predicate) for predicate in task["invariants"]),
        *(tuple(predicate) for predicate in persistent_completed),
        tuple(current_phase["predicate"]),
    ]


def _replay_phase_snapshot_predicates(
    task: Mapping[str, Any], current_subtask: int
) -> list[Tuple[str, ...]]:
    persistent = [
        tuple(phase["predicate"])
        for phase in task["phases"][:current_subtask]
        if phase.get("goal_required", True)
    ]
    return _phase_snapshot_predicates(
        task, persistent, task["phases"][current_subtask]
    )


def _validate_goal_partition(task: Mapping[str, Any], environment: Any, converter: Any):
    base_environment = converter._get_base_env(environment)
    parsed = [tuple(str(value).lower() for value in state) for state in base_environment.parsed_problem["goal_state"]]
    declared = [
        tuple(value.lower() for value in phase["predicate"])
        for phase in task["phases"]
        if phase.get("goal_required", True)
    ]
    declared.extend(tuple(value.lower() for value in state) for state in task["invariants"])
    if Counter(parsed) != Counter(declared):
        raise ValueError(
            f"{task['case_id']} phase predicates do not exactly partition BDDL goal: "
            f"parsed={parsed}, declared={declared}"
        )


def _pose_from_body_id(base_environment: Any, body_id: int) -> np.ndarray:
    return make_pose(
        np.asarray(base_environment.sim.data.body_xpos[body_id], dtype=np.float64),
        np.asarray(base_environment.sim.data.body_xmat[body_id], dtype=np.float64).reshape(
            3, 3
        ),
    )


def _live_reference_pose(
    environment: Any, reference_name: str, converter: Any
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve a predicate target from the live MuJoCo model, never a pose table."""
    base_environment = converter._get_base_env(environment)
    model = base_environment.sim.model
    try:
        site_id = int(model.site_name2id(reference_name))
    except ValueError:
        site_id = -1
    if site_id >= 0:
        return (
            make_pose(
                np.asarray(base_environment.sim.data.site_xpos[site_id], dtype=np.float64),
                np.asarray(base_environment.sim.data.site_xmat[site_id], dtype=np.float64).reshape(
                    3, 3
                ),
            ),
            {"kind": "site", "name": reference_name, "model_id": site_id},
        )

    body_ids = getattr(base_environment, "obj_body_id", {})
    if reference_name in body_ids:
        body_id = int(body_ids[reference_name])
    else:
        try:
            body_id = int(model.body_name2id(reference_name))
        except ValueError as error:
            raise RuntimeError(
                f"Predicate reference {reference_name!r} is neither a live site nor body"
            ) from error
    if body_id < 0:
        raise RuntimeError(f"Predicate reference {reference_name!r} has no live body")
    body_name = model.body_id2name(body_id) or reference_name
    return (
        _pose_from_body_id(base_environment, body_id),
        {"kind": "body", "name": str(body_name), "model_id": body_id},
    )


def _fixture_reference_pose(
    environment: Any, phase: Mapping[str, Any], converter: Any
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve the live moving site/body controlled by a fixture predicate."""
    reference_name = str(phase["predicate"][-1])
    base_environment = converter._get_base_env(environment)
    state = getattr(base_environment, "object_states_dict", {}).get(reference_name)
    if state is None:
        raise RuntimeError(f"Fixture predicate has no live state: {reference_name}")

    if getattr(state, "object_state_type", None) == "site":
        return _live_reference_pose(environment, reference_name, converter)

    object_name = getattr(state, "object_name", reference_name)
    model_object = base_environment.get_object(object_name)
    joints = tuple(getattr(model_object, "joints", ()))
    if len(joints) != 1:
        raise RuntimeError(
            f"Fixture {reference_name!r} must expose exactly one articulated joint; "
            f"found {list(joints)}"
        )
    model = base_environment.sim.model
    joint_id = int(model.joint_name2id(joints[0]))
    body_id = int(model.jnt_bodyid[joint_id])
    body_name = model.body_id2name(body_id)
    if not body_name:
        raise RuntimeError(f"Fixture joint {joints[0]!r} has no named moving body")
    return (
        _pose_from_body_id(base_environment, body_id),
        {
            "kind": "articulation_body",
            "name": str(body_name),
            "joint": str(joints[0]),
            "model_id": body_id,
        },
    )


def _object_phase_split(
    phase: Mapping[str, Any],
    events: Mapping[int, str],
    actor_poses: Sequence[np.ndarray],
    start: int,
    end: int,
) -> int:
    """Split one object phase after grasp/lift and before destination placement."""
    if end - start < 2:
        raise ValueError(f"Object phase for {phase['actor']} is too short to split")
    closes = [
        index
        for index, event in events.items()
        if event == "close" and start <= index < end
    ]
    if not closes:
        raise ValueError(f"Object phase for {phase['actor']} has no close event")
    close = closes[-1]
    initial_z = float(np.asarray(actor_poses[start])[2, 3])
    lifted = next(
        (
            index
            for index in range(close, end)
            if float(np.asarray(actor_poses[index])[2, 3]) >= initial_z + 0.005
        ),
        None,
    )
    split = lifted if lifted is not None else close + 5
    return int(np.clip(split, start + 1, end - 1))


def _phase_window(
    phase: Mapping[str, Any],
    events: Mapping[int, str],
    predicate_end: int,
    previous_end: int,
    horizon: int,
    *,
    final_phase: bool,
) -> Tuple[int, int]:
    if not previous_end < predicate_end <= horizon:
        raise ValueError("Phase predicate did not transition after the previous phase")
    if phase["kind"] in {"fixture", "push"}:
        return previous_end, predicate_end
    closes = [
        index
        for index, event in events.items()
        if event == "close" and previous_end <= index < predicate_end
    ]
    if not closes:
        raise ValueError("Object phase has no close event before its predicate")
    close = closes[-1]
    earlier_opens = [
        index for index, event in events.items() if event == "open" and index < close
    ]
    start = max(previous_end, earlier_opens[-1] + 1 if earlier_opens else previous_end)
    later_opens = [
        index for index, event in events.items() if event == "open" and index > close
    ]
    if later_opens:
        end = max(predicate_end, later_opens[0] + 1)
    elif phase["release_required"]:
        raise ValueError("Intermediate object phase has no release event")
    elif final_phase:
        end = horizon
    else:
        end = predicate_end
    if not start < end <= horizon:
        raise ValueError("Derived phase action window is invalid")
    return start, end


def _trace_source_demo(
    arrays: Mapping[str, np.ndarray],
    task: Mapping[str, Any],
    reference_env: Any,
    converter: Any,
) -> list[Dict[str, Any]]:
    states = arrays["states"]
    actions = arrays["actions"]
    eef_poses = []
    phase_references = [
        {
            "actor": [],
            "destination": [],
            "fixture": [],
            "metadata": None,
            "destination_metadata": None,
        }
        for _ in task["phases"]
    ]
    predicate_values = [[] for _ in task["phases"]]

    def observe() -> None:
        position, rotation = converter.get_eef_pose(reference_env)
        eef_poses.append(make_pose(position, rotation))
        for index, phase in enumerate(task["phases"]):
            references = phase_references[index]
            if phase["kind"] == "object":
                references["actor"].append(
                    _body_pose(reference_env, phase["actor"], converter)
                )
                destination_pose, metadata = _live_reference_pose(
                    reference_env, str(phase["predicate"][-1]), converter
                )
                references["destination"].append(destination_pose)
            elif phase["kind"] == "push":
                fixture_pose, metadata = _live_reference_pose(
                    reference_env, phase["actor"], converter
                )
                references["fixture"].append(fixture_pose)
                destination_pose, destination_metadata = _live_reference_pose(
                    reference_env, str(phase["predicate"][-1]), converter
                )
                references["destination"].append(destination_pose)
                if references["destination_metadata"] is None:
                    references["destination_metadata"] = destination_metadata
                elif references["destination_metadata"] != destination_metadata:
                    raise RuntimeError(
                        f"Live destination identity changed during phase {index}"
                    )
            else:
                fixture_pose, metadata = _fixture_reference_pose(
                    reference_env, phase, converter
                )
                references["fixture"].append(fixture_pose)
            if references["metadata"] is None:
                references["metadata"] = metadata
            elif references["metadata"] != metadata:
                raise RuntimeError(
                    f"Live reference identity changed during phase {index}: "
                    f"{references['metadata']} != {metadata}"
                )
            predicate_values[index].append(
                _goal(reference_env, phase["predicate"], converter)
            )

    for state in states:
        reference_env.reset_to({"states": state})
        observe()
    reference_env.reset_to({"states": states[-1]})
    reference_env.step(actions[-1])
    observe()

    reference_poses = np.asarray(eef_poses[:-1])
    events = detect_gripper_events(actions)
    segments = []
    previous_end = 0
    for index, phase in enumerate(task["phases"]):
        transition = next(
            (
                state_index
                for state_index in range(previous_end + 1, len(predicate_values[index]))
                if predicate_values[index][state_index]
            ),
            None,
        )
        if transition is None:
            raise ValueError(f"Source never satisfies {' '.join(phase['predicate'])}")
        start, end = _phase_window(
            phase,
            events,
            transition,
            previous_end,
            len(actions),
            final_phase=index == len(task["phases"]) - 1,
        )
        references = phase_references[index]
        common = {
            "phase_index": index,
            "kind": phase["kind"],
            "actor": phase["actor"],
            "predicate": phase["predicate"],
            "reference": references["metadata"],
            "start": start,
            "end": end,
        }
        if phase["kind"] == "object":
            split = _object_phase_split(
                phase, events, references["actor"], start, end
            )
            segments.append(
                {
                    **common,
                    "split": split,
                    "selection_reference_pose": np.asarray(
                        references["actor"][start]
                    ),
                    "pickup_reference_pose": np.asarray(
                        references["actor"][start]
                    ),
                    "pickup_target_poses": reference_poses[start:split].copy(),
                    "pickup_gripper": actions[start:split, -1].copy(),
                    "pickup_feedforward": actions[start:split, :6].copy(),
                    "destination_reference_pose": np.asarray(
                        references["destination"][split]
                    ),
                    "place_target_poses": reference_poses[split:end].copy(),
                    "place_gripper": actions[split:end, -1].copy(),
                    "place_feedforward": actions[split:end, :6].copy(),
                }
            )
        else:
            segment = {
                **common,
                "selection_reference_pose": np.asarray(
                    references["fixture"][start]
                ),
                "fixture_reference_pose": np.asarray(
                    references["fixture"][start]
                ),
                "target_poses": reference_poses[start:end].copy(),
                "gripper": actions[start:end, -1].copy(),
                "feedforward": actions[start:end, :6].copy(),
            }
            if phase["kind"] == "push":
                segment.update(
                    {
                        "destination_reference_pose": np.asarray(
                            references["destination"][start]
                        ),
                        "destination_reference": references[
                            "destination_metadata"
                        ],
                    }
                )
            segments.append(segment)
        previous_end = end
    return segments


def _phase_libraries(
    demos: Mapping[str, Mapping[str, np.ndarray]],
    task: Mapping[str, Any],
    reference_env: Any,
    source_count: int,
    converter: Any,
    *,
    source_embodiment: str,
) -> list[list[Dict[str, Any]]]:
    candidates: list[list[Dict[str, Any]]] = [[] for _ in task["phases"]]
    skipped = 0
    for demo_index, (name, arrays) in enumerate(demos.items(), start=1):
        try:
            segments = _trace_source_demo(arrays, task, reference_env, converter)
        except ValueError:
            skipped += 1
        else:
            for index, segment in enumerate(segments):
                candidates[index].append(
                    {
                        **segment,
                        "demo": name,
                        "source_embodiment": source_embodiment,
                    }
                )
        if demo_index % 10 == 0 or demo_index == len(demos):
            print(
                f"[SOURCE] task={task['case_id']} scanned={demo_index}/{len(demos)} "
                f"eligible={[len(rows) for rows in candidates]} skipped={skipped}",
                flush=True,
            )
    libraries = []
    for index, rows in enumerate(candidates):
        if not rows:
            raise RuntimeError(
                f"{task['case_id']} has no source segment for phase {index}"
            )
        count = min(source_count, len(rows))
        chosen = _diverse_indices(
            [row["selection_reference_pose"] for row in rows], count, converter
        )
        libraries.append([rows[row_index] for row_index in chosen])
    return libraries


def _raw_success_demos(
    raw_hdf5: Path, raw_rows: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load target-native phase seeds from successful raw Aubo rollouts."""
    names = sorted(
        (
            name
            for name, row in raw_rows.items()
            if row["terminal_task_success"] is True
        ),
        key=_demo_number,
    )
    if not names:
        raise ValueError("Task has no successful raw Aubo phase seed")
    demos: Dict[str, Dict[str, np.ndarray]] = {}
    with h5py.File(raw_hdf5, "r") as raw:
        for name in names:
            group = raw[f"data/{name}"]
            states = np.asarray(group["states"][()], dtype=np.float64)
            actions = np.asarray(group["actions"][()], dtype=np.float64)
            if states.ndim != 2 or actions.shape != (len(states), 7):
                raise ValueError(f"{raw_hdf5}:{name} has invalid states/actions")
            if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
                raise ValueError(f"{raw_hdf5}:{name} contains NaN or inf")
            demos[name] = {"states": states, "actions": actions}
    return demos


def _compatible_phase_library(
    library: Sequence[Mapping[str, Any]],
    previous_source_demo: Any,
    independent_phases: bool = False,
) -> Sequence[Mapping[str, Any]]:
    if independent_phases or previous_source_demo is None:
        return library
    same_demo = [row for row in library if row["demo"] == previous_source_demo]
    return same_demo or library


def _initialize_failed_state(
    reference_env: Any,
    target_env: Any,
    source_initial_state: np.ndarray,
    initial_gripper_action: float,
    expected_raw_initial_state: np.ndarray,
    converter: Any,
) -> Dict[str, Any]:
    initialization = _initialize(
        reference_env,
        target_env,
        source_initial_state,
        initial_gripper_action,
        RAW_INITIALIZATION_SEED,
        converter,
    )
    actual = np.asarray(initialization["state"])
    expected = np.asarray(expected_raw_initial_state)
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"Reconstructed Aubo initial-state shape {actual.shape} differs from "
            f"raw failed state {expected.shape}"
        )
    maximum_difference = float(np.max(np.abs(actual - expected)))
    if not np.allclose(
        actual,
        expected,
        rtol=0.0,
        atol=INITIAL_STATE_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(
            "Reconstructed Aubo initial state differs from the raw failed state; "
            f"maximum difference={maximum_difference:.3e}"
        )
    if not np.array_equal(actual, expected):
        base_environment = converter._get_base_env(target_env)
        set_flattened = getattr(
            base_environment.sim, "set_state_from_flattened", None
        )
        if not callable(set_flattened):
            raise AttributeError(
                "Target simulator cannot restore the exact raw Aubo state"
            )
        set_flattened(expected)
        base_environment.sim.forward()
        restored = converter.get_target_sim_state(target_env)
        if not np.array_equal(restored, expected):
            raise RuntimeError("Target simulator did not retain the exact raw Aubo state")
        initialization["state"] = restored
        scene, body_count = _scene_values(target_env, converter)
        initialization["scene"] = scene
        initialization["named_task_body_count"] = body_count
    initialization["raw_initial_state_within_absolute_tolerance"] = True
    initialization["raw_initial_state_max_absolute_difference"] = maximum_difference
    initialization["raw_initial_state_restored_exactly"] = True
    return initialization


def _anchor_push_trajectory(
    source_poses: np.ndarray,
    current_actor_pose: np.ndarray,
    source_actor_pose: np.ndarray,
    source_destination_pose: np.ndarray,
    target_destination_pose: np.ndarray,
    destination_inset_m: float = 0.0,
) -> np.ndarray:
    """Warp a push from the current object pose to the fixed goal region."""
    poses = np.asarray(source_poses, dtype=np.float64).copy()
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError("Push trajectory must contain at least two 4x4 poses")
    translation = current_actor_pose[:3, 3] - source_actor_pose[:3, 3]
    poses[:, :3, 3] += translation
    predicted_destination = source_destination_pose[:3, 3] + translation
    goal = target_destination_pose[:3, 3].copy()
    direction = goal - current_actor_pose[:3, 3]
    direction[2] = 0.0
    direction_norm = float(np.linalg.norm(direction))
    if destination_inset_m < 0.0 or not np.isfinite(destination_inset_m):
        raise ValueError("Push destination inset must be a finite non-negative value")
    if direction_norm > 0.0:
        goal += direction / direction_norm * destination_inset_m
    correction = goal - predicted_destination
    weights = np.linspace(0.0, 1.0, len(poses), dtype=np.float64)[:, None]
    poses[:, :3, 3] += weights * correction
    return poses


def _clearance_transition_poses(
    current_pose: np.ndarray,
    next_pose: np.ndarray,
    clearance_z: float,
    steps: int,
    converter: Any,
) -> np.ndarray:
    """Lift and traverse at a demonstrated, already-reachable phase height."""
    current = np.asarray(current_pose, dtype=np.float64)
    target = np.asarray(next_pose, dtype=np.float64)
    if current.shape != (4, 4) or target.shape != (4, 4):
        raise ValueError("Clearance transition poses must be 4x4")
    if not np.isfinite(clearance_z) or clearance_z < max(
        float(current[2, 3]), float(target[2, 3])
    ):
        raise ValueError("Clearance height must be finite and above both endpoints")
    lift = current.copy()
    lift[2, 3] = clearance_z
    traverse = target.copy()
    traverse[2, 3] = clearance_z
    return np.concatenate(
        (
            _interpolate_pose(current, lift, steps, converter),
            _interpolate_pose(lift, traverse, steps, converter),
        ),
        axis=0,
    )


def recover_libero10_trajectory(
    target_demo: str,
    demos: Mapping[str, Mapping[str, np.ndarray]],
    task: Mapping[str, Any],
    libraries: Sequence[Sequence[Mapping[str, Any]]],
    reference_env: Any,
    target_env: Any,
    expected_raw_initial_state: np.ndarray,
    rng: np.random.RandomState,
    arguments: argparse.Namespace,
    converter: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Recover one LIBERO-10 identity with live, phase-specific references."""
    initial = demos[target_demo]
    initialization = _initialize_failed_state(
        reference_env,
        target_env,
        initial["states"][0],
        float(initial["actions"][0, -1]),
        expected_raw_initial_state,
        converter,
    )
    rollout = _new_rollout(initialization, arguments.jacobian_rotation_length_m)
    persistent_completed: list[Tuple[str, ...]] = []
    snapshots = []
    selections = []
    last_poses = None
    last_gripper = None
    previous_source_demo = None
    for index, (phase, library) in enumerate(zip(task["phases"], libraries)):
        push_destination_inset_m = None
        if phase["kind"] == "object":
            current_reference = _body_pose(
                target_env, phase["actor"], converter
            )
        elif phase["kind"] == "push":
            current_reference, target_reference = _live_reference_pose(
                target_env, phase["actor"], converter
            )
            destination_pose, target_destination_reference = _live_reference_pose(
                target_env, str(phase["predicate"][-1]), converter
            )
        else:
            current_reference, target_reference = _fixture_reference_pose(
                target_env, phase, converter
            )
        source = _nearest_source(
            _compatible_phase_library(
                library,
                previous_source_demo,
                task.get("independent_phase_sources", False),
            ),
            current_reference,
            "selection_reference_pose",
            arguments.nearest_neighbors,
            rng,
            converter,
        )
        previous_source_demo = source["demo"]
        if phase["kind"] == "object":
            pickup_poses = transform_segment(
                current_reference,
                source["pickup_target_poses"],
                source["pickup_reference_pose"],
            )
            if index and task.get("inter_phase_clearance"):
                current_position, current_rotation = converter.get_eef_pose(target_env)
                current_pose = make_pose(current_position, current_rotation)
                clearance_z = max(
                    float(current_pose[2, 3]),
                    float(np.max(pickup_poses[:, 2, 3])),
                )
                transition = _clearance_transition_poses(
                    current_pose,
                    pickup_poses[0],
                    clearance_z,
                    arguments.interpolation_steps,
                    converter,
                )
                _execute(
                    target_env,
                    transition,
                    np.full(len(transition), float(source["pickup_gripper"][0])),
                    index,
                    rollout,
                    converter,
                )
            _execute_subtask(
                target_env,
                pickup_poses,
                source["pickup_gripper"],
                index,
                rollout,
                arguments.interpolation_steps,
                arguments.fixed_steps,
                converter,
                feedforward=source["pickup_feedforward"],
                reference_substeps=arguments.reference_substeps,
            )
            destination_pose, target_reference = _live_reference_pose(
                target_env, str(phase["predicate"][-1]), converter
            )
            if source["reference"]["kind"] != target_reference["kind"]:
                raise RuntimeError(
                    f"Source/target reference type changed for phase {index}: "
                    f"{source['reference']} != {target_reference}"
                )
            transformed = transform_segment(
                destination_pose,
                source["place_target_poses"],
                source["destination_reference_pose"],
            )
            phase_gripper = source["place_gripper"]
            _execute_subtask(
                target_env,
                transformed,
                phase_gripper,
                index,
                rollout,
                arguments.interpolation_steps,
                arguments.fixed_steps,
                converter,
                feedforward=source["place_feedforward"],
                reference_substeps=arguments.reference_substeps,
            )
        else:
            if source["reference"]["kind"] != target_reference["kind"]:
                raise RuntimeError(
                    f"Source/target fixture reference changed for phase {index}: "
                    f"{source['reference']} != {target_reference}"
                )
            if phase["kind"] == "push":
                if (
                    source["destination_reference"]["kind"]
                    != target_destination_reference["kind"]
                ):
                    raise RuntimeError(
                        "Source/target push destinations have different types"
                    )
                base_environment = converter._get_base_env(target_env)
                site_size = np.asarray(
                    base_environment.sim.model.site_size[
                        target_destination_reference["model_id"]
                    ],
                    dtype=np.float64,
                )
                if site_size.shape != (3,) or not np.all(site_size > 0.0):
                    raise RuntimeError("Push destination site has an invalid size")
                push_destination_inset_m = 0.5 * float(np.min(site_size[:2]))
                transformed = _anchor_push_trajectory(
                    source["target_poses"],
                    current_reference,
                    source["fixture_reference_pose"],
                    source["destination_reference_pose"],
                    destination_pose,
                    push_destination_inset_m,
                )
            else:
                transformed = transform_segment(
                    current_reference,
                    source["target_poses"],
                    source["fixture_reference_pose"],
                )
            phase_gripper = source["gripper"]
            _execute_subtask(
                target_env,
                transformed,
                phase_gripper,
                index,
                rollout,
                arguments.interpolation_steps,
                arguments.fixed_steps,
                converter,
                feedforward=source["feedforward"],
                reference_substeps=arguments.reference_substeps,
            )
        checked = _phase_snapshot_predicates(task, persistent_completed, phase)
        snapshot = _goal_snapshot(target_env, checked, converter)
        snapshots.append(snapshot)
        if phase.get("goal_required", True):
            persistent_completed.append(tuple(phase["predicate"]))
        selections.append(
            {
                "phase_index": index,
                "actor": phase["actor"],
                "predicate": list(phase["predicate"]),
                "source_demo": source["demo"],
                "source_embodiment": source["source_embodiment"],
                "source_window": [int(source["start"]), int(source["end"])],
                "source_split": int(source["split"])
                if phase["kind"] == "object"
                else None,
                "live_reference": target_reference,
                "live_destination": target_destination_reference
                if phase["kind"] == "push"
                else None,
                "push_destination_inset_m": push_destination_inset_m,
            }
        )
        last_poses = transformed
        last_gripper = phase_gripper
        if not all(snapshot.values()):
            break
    phase_checks_passed = len(snapshots) == len(task["phases"]) and all(
        all(snapshot.values()) for snapshot in snapshots
    )
    if phase_checks_passed and last_gripper is not None:
        _validate_success_dwell(
            target_env,
            float(last_gripper[-1]),
            arguments.final_hold_steps,
            arguments.success_dwell_steps,
            rollout,
            converter,
        )
    generation = _finalize_rollout(rollout, target_env, converter)
    generation["phase_snapshots"] = snapshots
    generation["phase_checks_passed"] = phase_checks_passed
    return generation, {
        "target_initial_demo": target_demo,
        "raw_initial_state_within_absolute_tolerance": initialization[
            "raw_initial_state_within_absolute_tolerance"
        ],
        "raw_initial_state_max_absolute_difference": initialization[
            "raw_initial_state_max_absolute_difference"
        ],
        "raw_initial_state_restored_exactly": initialization[
            "raw_initial_state_restored_exactly"
        ],
        "phase_sources": selections,
    }


def _raw_initial_state(raw_hdf5: Path, demo: str) -> np.ndarray:
    with h5py.File(raw_hdf5, "r") as raw:
        group = raw[f"data/{demo}"]
        stored = np.asarray(group.attrs["init_state"])
        states_zero = np.asarray(group["states"][0])
    if stored.ndim != 1 or not np.all(np.isfinite(stored)):
        raise ValueError(f"{raw_hdf5}:{demo} has an invalid init_state")
    if not np.array_equal(stored, states_zero):
        raise ValueError(f"{raw_hdf5}:{demo} init_state differs from states[0]")
    return stored


def _stagewise_replay(
    actions: np.ndarray,
    subtasks: np.ndarray,
    initial: Mapping[str, np.ndarray],
    expected_raw_initial_state: np.ndarray,
    task: Mapping[str, Any],
    reference_env: Any,
    target_env: Any,
    converter: Any,
    rotation_length_m: float,
) -> Dict[str, Any]:
    initialization = _initialize_failed_state(
        reference_env,
        target_env,
        initial["states"][0],
        float(actions[0, -1]),
        expected_raw_initial_state,
        converter,
    )
    states = []
    rewards = []
    dones = []
    kinematics = []
    phase_snapshots = []
    success_streak = 0
    for index, action in enumerate(actions):
        states.append(converter.get_target_sim_state(target_env))
        kinematics.append(
            _kinematic_metrics(target_env, converter, rotation_length_m)
        )
        _, reward, done, _ = target_env.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        success = converter._task_success(target_env)
        success_streak = success_streak + 1 if success else 0
        current_subtask = int(subtasks[index])
        next_subtask = (
            int(subtasks[index + 1]) if index + 1 < len(subtasks) else current_subtask
        )
        if current_subtask < len(task["phases"]) and (
            index + 1 == len(subtasks) or next_subtask != current_subtask
        ):
            phase_snapshots.append(
                _goal_snapshot(
                    target_env,
                    _replay_phase_snapshot_predicates(task, current_subtask),
                    converter,
                )
            )
    kinematics.append(_kinematic_metrics(target_env, converter, rotation_length_m))
    return {
        "states": np.asarray(states),
        "rewards": np.asarray(rewards),
        "dones": np.asarray(dones),
        "initial_state": np.asarray(initialization["state"]),
        "initial_scene": np.asarray(initialization["scene"]),
        "terminal_task_success": bool(converter._task_success(target_env)),
        "terminal_success_streak": int(success_streak),
        **_summarize_kinematics(kinematics),
        "phase_snapshots": phase_snapshots,
        "phase_checks_passed": len(phase_snapshots) == len(task["phases"])
        and all(all(snapshot.values()) for snapshot in phase_snapshots),
    }


def _stagewise_qualified(
    generation: Mapping[str, Any],
    arguments: argparse.Namespace,
    maximum_clipped_component_pct: float,
    maximum_clipped_run_steps: int,
    maximum_horizon: int,
) -> Tuple[bool, list[str]]:
    _, reasons = _qualified(
        generation,
        arguments.min_arm_joint_margin_rad,
        arguments.success_dwell_steps,
        maximum_clipped_component_pct,
        maximum_clipped_run_steps,
        maximum_horizon,
        arguments.minimum_jacobian_sigma,
        arguments.maximum_jacobian_condition,
    )
    if not generation["phase_checks_passed"]:
        reasons.append("generation_phase_predicate")
    return not reasons, reasons


def _preflight(
    task: Mapping[str, Any],
    raw_root: Path,
    converter: Any,
    arguments: argparse.Namespace,
) -> Dict[str, Any]:
    artifacts = _task_artifacts(raw_root, task)
    reference_env = target_env = None
    try:
        reference_env, target_env, environment_name = _make_environments(
            artifacts["source_hdf5"], artifacts["raw_hdf5"], converter
        )
        _validate_goal_partition(task, reference_env, converter)
        _validate_goal_partition(task, target_env, converter)
        references = []
        for phase in task["phases"]:
            _resolve_body(reference_env, phase["actor"], converter)
            _resolve_body(target_env, phase["actor"], converter)
            if phase["kind"] == "object":
                _, source_reference = _live_reference_pose(
                    reference_env, str(phase["predicate"][-1]), converter
                )
                _, target_reference = _live_reference_pose(
                    target_env, str(phase["predicate"][-1]), converter
                )
            elif phase["kind"] == "push":
                _, source_reference = _live_reference_pose(
                    reference_env, phase["actor"], converter
                )
                _, target_reference = _live_reference_pose(
                    target_env, phase["actor"], converter
                )
            else:
                _, source_reference = _fixture_reference_pose(
                    reference_env, phase, converter
                )
                _, target_reference = _fixture_reference_pose(
                    target_env, phase, converter
                )
            if source_reference["kind"] != target_reference["kind"]:
                raise RuntimeError(
                    f"Panda/Aubo live-reference types differ: "
                    f"{source_reference} != {target_reference}"
                )
            references.append(
                {
                    "phase": list(phase["predicate"]),
                    "source": source_reference,
                    "target": target_reference,
                }
            )
        raw_kinematics = _raw_kinematic_summaries(
            artifacts,
            target_env,
            converter,
            arguments.jacobian_rotation_length_m,
        )
        raw_kinematic_reasons = {
            name: _kinematic_rejection_reasons(
                metrics,
                arguments.min_arm_joint_margin_rad,
                arguments.minimum_jacobian_sigma,
                arguments.maximum_jacobian_condition,
                "raw",
            )
            for name, metrics in raw_kinematics.items()
        }
        return {
            "case_id": task["case_id"],
            "suite": task["suite"],
            "environment": environment_name,
            "failed_identity_count": len(
                _failed_demo_names(artifacts["raw_rows"])
            ),
            "target_identity_count": sum(
                bool(
                    _raw_admission_reasons(
                        row,
                        artifacts["raw_actions"][name],
                        artifacts["raw_rewards"][name],
                    )
                    + raw_kinematic_reasons[name]
                )
                for name, row in artifacts["raw_rows"].items()
            ),
            "raw_kinematic_rejection_identity_count": sum(
                bool(reasons) for reasons in raw_kinematic_reasons.values()
            ),
            "phases": [list(phase["predicate"]) for phase in task["phases"]],
            "invariants": [list(state) for state in task["invariants"]],
            "live_references": references,
        }
    finally:
        converter._close_env(reference_env)
        converter._close_env(target_env)

def _run_stagewise_task(
    task: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    arguments: argparse.Namespace,
    converter: Any,
) -> Dict[str, Any]:
    """Recover every selected failed identity for one stage-wise task."""
    artifacts = _task_artifacts(raw_root, task)
    raw_failed_demos = _failed_demo_names(artifacts["raw_rows"])
    raw_kinematics = _screen_raw_kinematics(
        artifacts, converter, arguments.jacobian_rotation_length_m
    )
    raw_rejection_reasons = {
        name: _raw_admission_reasons(
            row, artifacts["raw_actions"][name], artifacts["raw_rewards"][name]
        )
        + _kinematic_rejection_reasons(
            raw_kinematics[name],
            arguments.min_arm_joint_margin_rad,
            arguments.minimum_jacobian_sigma,
            arguments.maximum_jacobian_condition,
            "raw",
        )
        for name, row in artifacts["raw_rows"].items()
    }
    all_target_demos = sorted(
        (name for name, reasons in raw_rejection_reasons.items() if reasons),
        key=_demo_number,
    )
    verified_source_rows = {
        name: row
        for name, row in artifacts["raw_rows"].items()
        if not raw_rejection_reasons[name]
    }
    stagewise_source_rows, raw_bootstrap_used = _stagewise_source_rows(
        artifacts["raw_rows"], verified_source_rows
    )
    raw_clipped_pct, raw_clipped_run = _raw_success_clipping_limits(
        stagewise_source_rows
    )
    maximum_clipped_component_pct = (
        raw_clipped_pct
        if arguments.max_clipped_component_pct is None
        else arguments.max_clipped_component_pct
    )
    maximum_clipped_run_steps = (
        raw_clipped_run
        if arguments.max_clipped_run_steps is None
        else arguments.max_clipped_run_steps
    )
    demos, source_metadata = _load_source(artifacts["source_hdf5"])
    maximum_horizon = max(len(row["actions"]) for row in demos.values()) + (
        len(task["phases"])
        * (arguments.interpolation_steps + arguments.fixed_steps)
    )
    task_dir = output_root / task["case_id"]
    task_dir.mkdir()
    attempts_path = task_dir / "attempts.jsonl"
    ledger_path = task_dir / "recovery_ledger.jsonl"
    reference_env = target_env = camera_env = None
    attempt_rows = []
    identity_rows = []
    accepted_paths: list[Path] = []
    accepted_demos: list[str] = []
    try:
        reference_env, target_env, environment_name = _make_environments(
            artifacts["source_hdf5"], artifacts["raw_hdf5"], converter
        )
        _validate_goal_partition(task, reference_env, converter)
        _validate_goal_partition(task, target_env, converter)
        source_demos = _raw_success_demos(
            artifacts["raw_hdf5"], stagewise_source_rows
        )
        recovered_sources = _recovered_source_demos(
            arguments.recovered_source_root, task
        )
        if set(source_demos) & set(recovered_sources):
            raise ValueError("Raw and recovered source identities overlap")
        source_demos.update(recovered_sources)
        if raw_bootstrap_used:
            source_policy = (
                "terminal_success_raw_bootstrap_plus_recovered_aubo"
                if arguments.recovered_source_root
                else "terminal_success_raw_bootstrap_only"
            )
        else:
            source_policy = (
                "online_success_raw_plus_recovered_aubo"
                if arguments.recovered_source_root
                else "online_success_raw_aubo_only"
            )
        libraries = _phase_libraries(
            source_demos,
            task,
            target_env,
            arguments.source_demos,
            converter,
            source_embodiment="online_success_raw_or_recovered_aubo",
        )
        resume_task = arguments.resume_tasks.get(task["case_id"])
        carried_forward_demos: list[str] = []
        if resume_task is not None:
            carried_forward_demos, previous_remaining, previous_merged = (
                _resume_partition(resume_task, all_target_demos)
            )
            if previous_merged is not None:
                accepted_paths.append(previous_merged)
                accepted_demos.extend(carried_forward_demos)
        if arguments.target_demo:
            if arguments.target_demo not in all_target_demos:
                raise ValueError(
                    f"{arguments.target_demo} already passes raw admission "
                    f"for {task['case_id']}"
                )
            target_demos = [arguments.target_demo]
        elif resume_task is not None:
            target_demos = previous_remaining
        else:
            target_demos = all_target_demos[: arguments.limit_identities]
        if not target_demos and resume_task is None:
            raise RuntimeError(f"{task['case_id']} has no selected failed identities")
        print(
            f"[TASK READY] {task['case_id']} failed={len(all_target_demos)} "
            f"selected={len(target_demos)} "
            f"source_policy={source_policy} source_pool={len(source_demos)} "
            f"phase_sources={[len(library) for library in libraries]} "
            f"clipping_limits={maximum_clipped_component_pct:.6g}%/"
            f"{maximum_clipped_run_steps}_steps",
            flush=True,
        )

        for target_demo in target_demos:
            raw_initial_state = _raw_initial_state(
                artifacts["raw_hdf5"], target_demo
            )
            raw_row = artifacts["raw_rows"][target_demo]
            accepted_path = None
            last_reasons: list[str] = []
            attempts_for_identity = 0
            for identity_attempt in range(1, arguments.attempts_per_identity + 1):
                attempts_for_identity = identity_attempt
                rng = _attempt_rng(
                    arguments.seed,
                    int(task["task_index"]),
                    target_demo,
                    identity_attempt,
                )
                generation, selection = recover_libero10_trajectory(
                    target_demo,
                    demos,
                    task,
                    libraries,
                    reference_env,
                    target_env,
                    raw_initial_state,
                    rng,
                    arguments,
                    converter,
                )
                generation["maximum_horizon"] = maximum_horizon
                accepted, reasons = _stagewise_qualified(
                    generation,
                    arguments,
                    maximum_clipped_component_pct,
                    maximum_clipped_run_steps,
                    maximum_horizon,
                )
                diagnostic_replay = (
                    _stagewise_replay(
                        generation["actions"],
                        generation["subtask"],
                        demos[target_demo],
                        raw_initial_state,
                        task,
                        reference_env,
                        target_env,
                        converter,
                        arguments.jacobian_rotation_length_m,
                    )
                    if accepted
                    else None
                )
                last_reasons = reasons
                row = {
                    "identity_attempt": identity_attempt,
                    **selection,
                    "raw_before_terminal_success": raw_row[
                        "terminal_task_success"
                    ],
                    "raw_before_ever_success": raw_row["ever_task_success"],
                    "raw_rejection_reasons": raw_rejection_reasons[target_demo],
                    "generated_terminal_success": generation[
                        "terminal_task_success"
                    ],
                    "generation_success_streak": generation[
                        "terminal_success_streak"
                    ],
                    "success_validation_steps": generation[
                        "success_validation_steps"
                    ],
                    "generation_phase_snapshots": generation["phase_snapshots"],
                    "diagnostic_replay_terminal_success": (
                        diagnostic_replay["terminal_task_success"]
                        if diagnostic_replay is not None
                        else None
                    ),
                    "accepted": accepted,
                    "rejection_reasons": reasons,
                    "control_steps": len(generation["actions"]),
                    "minimum_arm_joint_margin_rad": generation[
                        "minimum_arm_joint_margin_rad"
                    ],
                    "minimum_jacobian_sigma": generation["minimum_jacobian_sigma"],
                    "maximum_jacobian_condition_number": generation[
                        "maximum_jacobian_condition_number"
                    ],
                    "minimum_absolute_j3_rad": generation["minimum_absolute_j3_rad"],
                    "j3_zero_crossing_count": generation["j3_zero_crossing_count"],
                    "maximum_tcp_reach_m": generation["maximum_tcp_reach_m"],
                    "clipped_cartesian_component_pct": generation[
                        "clipped_cartesian_component_pct"
                    ],
                    "max_consecutive_clipped_steps": generation[
                        "max_consecutive_clipped_steps"
                    ],
                    "maximum_clipped_component_pct": (
                        maximum_clipped_component_pct
                    ),
                    "maximum_clipped_run_steps": maximum_clipped_run_steps,
                    "maximum_horizon": maximum_horizon,
                }
                if accepted:
                    if camera_env is None:
                        camera_env = _camera_environment(
                            artifacts["source_hdf5"],
                            artifacts["raw_hdf5"],
                            converter,
                        )
                    observations = _observations(
                        generation["states"], camera_env, converter
                    )
                    accepted_path = task_dir / f"recovered_{target_demo}.hdf5"
                    episode_metadata = dict(source_metadata)
                    episode_metadata["env_args"] = artifacts["target_env_args"]
                    _write_episode(
                        accepted_path,
                        task,
                        episode_metadata,
                        generation,
                        observations,
                        selection,
                    )
                    accepted_paths.append(accepted_path)
                    accepted_demos.append(target_demo)
                    row["accepted_hdf5"] = str(accepted_path)
                attempt_rows.append(row)
                with attempts_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                    )
                print(
                    f"[ATTEMPT] task={task['case_id']} identity={target_demo} "
                    f"n={identity_attempt}/{arguments.attempts_per_identity} "
                    f"phases={generation['phase_checks_passed']} "
                    f"terminal={generation['terminal_task_success']} "
                    f"diagnostic_replay={row['diagnostic_replay_terminal_success']} "
                    f"accepted={accepted}",
                    flush=True,
                )
                if accepted:
                    break

            identity_row = {
                "task": task["case_id"],
                "target_initial_demo": target_demo,
                "status": "recovered"
                if target_demo in accepted_demos
                else "unrecovered",
                "attempts": attempts_for_identity,
                "raw_before_terminal_success": raw_row["terminal_task_success"],
                "raw_before_ever_success": raw_row["ever_task_success"],
                "diagnostic_replay_terminal_success": next(
                    (
                        row["diagnostic_replay_terminal_success"]
                        for row in reversed(attempt_rows)
                        if row["target_initial_demo"] == target_demo
                        and row["accepted"]
                    ),
                    None,
                ),
                "accepted_hdf5": str(accepted_path) if accepted_path else None,
                "final_rejection_reasons": []
                if target_demo in accepted_demos
                else last_reasons,
            }
            identity_rows.append(identity_row)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(identity_row, sort_keys=True, allow_nan=False) + "\n"
                )
            print(
                f"[IDENTITY END] task={task['case_id']} identity={target_demo} "
                f"status={identity_row['status']} attempts={attempts_for_identity}",
                flush=True,
            )
        merged = task_dir / f"{task['task_stem']}.generated.hdf5"
        _merge_episodes(accepted_paths, merged)
        processed_demos = carried_forward_demos + [
            row["target_initial_demo"] for row in identity_rows
        ]
        unrecovered_demos = [
            row["target_initial_demo"]
            for row in identity_rows
            if row["status"] == "unrecovered"
        ]
        summary = {
            "case_id": task["case_id"],
            "suite": task["suite"],
            "task_stem": task["task_stem"],
            "environment": environment_name,
            "authoritative_raw_root": str(raw_root),
            "raw_aubo_hdf5": str(artifacts["raw_hdf5"]),
            "linked_original_panda_hdf5": str(artifacts["source_hdf5"]),
            "target_initial_stratum": "raw_without_stable_terminal_success",
            "raw_failed_identity_count": len(raw_failed_demos),
            "raw_unstable_success_identity_count": sum(
                artifacts["raw_rows"][name]["terminal_task_success"] is True
                for name in all_target_demos
            ),
            "raw_retained_identity_count": len(verified_source_rows),
            "raw_kinematic_rejection_identity_count": sum(
                any(
                    reason.startswith("raw_jacobian_")
                    or reason == "raw_joint_margin"
                    for reason in reasons
                )
                for reasons in raw_rejection_reasons.values()
            ),
            "target_identity_count": len(all_target_demos),
            "source_library_policy": source_policy,
            "source_pool_identity_count": len(source_demos),
            "raw_bootstrap_source_and_clipping_demo_ids": (
                sorted(stagewise_source_rows, key=_demo_number)
                if raw_bootstrap_used
                else []
            ),
            "recovered_source_root": str(
                arguments.recovered_source_root.resolve()
            )
            if arguments.recovered_source_root
            else None,
            "selected_failed_identity_count": len(carried_forward_demos)
            + len(target_demos),
            "processed_identity_count": len(processed_demos),
            "attempt_count": len(attempt_rows),
            "diagnostic_replay_count": sum(
                row["diagnostic_replay_terminal_success"] is not None
                for row in attempt_rows
            ),
            "diagnostic_replay_success_count": sum(
                row["diagnostic_replay_terminal_success"] is True
                for row in attempt_rows
            ),
            "recovered_identity_count": len(accepted_demos),
            "unrecovered_identity_count": len(unrecovered_demos),
            "recovered_demo_ids": accepted_demos,
            "unrecovered_demo_ids": unrecovered_demos,
            "not_attempted_demo_ids": [
                name for name in all_target_demos if name not in processed_demos
            ],
            "all_selected_identities_processed": len(identity_rows)
            == len(target_demos)
            and len(processed_demos)
            == len(carried_forward_demos) + len(target_demos),
            "merged_hdf5": str(merged) if accepted_paths else None,
            "continuation_from": str(arguments.resume_from)
            if arguments.resume_from
            else None,
            "carried_forward_recovered_identity_count": len(
                carried_forward_demos
            ),
            "hdf5_output_enabled": True,
            "maximum_clipped_component_pct": maximum_clipped_component_pct,
            "maximum_clipped_run_steps": maximum_clipped_run_steps,
            "maximum_horizon": maximum_horizon,
        }
        _atomic_json(task_dir / "summary.json", summary)
        return summary
    finally:
        converter._close_env(camera_env)
        converter._close_env(reference_env)
        converter._close_env(target_env)

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover failed Aubo identities across all original LIBERO suites."
    )
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted({task["suite"] for task in TASKS}),
        help="Run every registered task in this suite; repeat to select suites.",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=[task["case_id"] for task in TASKS],
        help="Run only this task; repeat to select more than one.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--diagnostic-video",
        help="Write the first accepted replay as MP4 and disable HDF5 output.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-demos", type=int, default=10)
    parser.add_argument("--attempts-per-identity", type=int, default=10)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Completed suite output whose unrecovered identities should be retried.",
    )
    parser.add_argument("--limit-identities", type=int)
    parser.add_argument(
        "--target-demo", help="Run one exact failed demo."
    )
    parser.add_argument(
        "--recovered-source-root",
        type=Path,
        help="Completed same-suite overlay whose accepted Aubo demos seed recovery.",
    )
    parser.add_argument("--nearest-neighbors", type=int, default=3)
    parser.add_argument("--boundary-offset", type=int, default=3)
    parser.add_argument("--interpolation-steps", type=int, default=10)
    parser.add_argument("--reference-substeps", type=int, default=1)
    parser.add_argument("--fixed-steps", type=int, default=3)
    parser.add_argument("--final-hold-steps", type=int, default=20)
    parser.add_argument("--success-dwell-steps", type=int, default=5)
    parser.add_argument("--min-arm-joint-margin-rad", type=float, default=0.03)
    parser.add_argument(
        "--minimum-jacobian-sigma",
        type=float,
        default=DEFAULT_MINIMUM_JACOBIAN_SIGMA,
    )
    parser.add_argument(
        "--maximum-jacobian-condition",
        type=float,
        default=DEFAULT_MAXIMUM_JACOBIAN_CONDITION,
    )
    parser.add_argument(
        "--jacobian-rotation-length-m",
        type=float,
        default=DEFAULT_JACOBIAN_ROTATION_LENGTH_M,
    )
    parser.add_argument("--max-clipped-component-pct", type=float)
    parser.add_argument("--max-clipped-run-steps", type=int)
    arguments = parser.parse_args()
    positive = (
        arguments.source_demos,
        arguments.attempts_per_identity,
        arguments.nearest_neighbors,
        arguments.interpolation_steps,
        arguments.reference_substeps,
        arguments.fixed_steps,
        arguments.final_hold_steps,
        arguments.success_dwell_steps,
    )
    if any(value < 1 for value in positive):
        parser.error("Counts and step settings must be positive")
    if arguments.source_demos > 50:
        parser.error("Source-library size cannot exceed 50")
    if arguments.limit_identities is not None and arguments.limit_identities < 1:
        parser.error("--limit-identities must be positive")
    if arguments.target_demo and arguments.limit_identities is not None:
        parser.error("--target-demo and --limit-identities are mutually exclusive")
    if arguments.resume_from and (
        arguments.target_demo or arguments.limit_identities is not None
    ):
        parser.error(
            "--resume-from cannot be combined with target, limit, or replay filters"
        )
    if arguments.preflight_only and arguments.recovered_source_root:
        parser.error(
            "--preflight-only does not accept recovered-source inputs"
        )
    non_negative = (
        arguments.boundary_offset,
        arguments.min_arm_joint_margin_rad,
        arguments.minimum_jacobian_sigma,
    )
    if any(value < 0 for value in non_negative):
        parser.error("Boundary offset, joint margin, and sigma must be non-negative")
    if (
        arguments.maximum_jacobian_condition <= 0
        or arguments.jacobian_rotation_length_m <= 0
    ):
        parser.error("Jacobian condition and rotation length must be positive")
    if arguments.max_clipped_component_pct is not None and not (
        0 <= arguments.max_clipped_component_pct <= 100
    ):
        parser.error("Clipped-component percentage must be within [0, 100]")
    if (
        arguments.max_clipped_run_steps is not None
        and arguments.max_clipped_run_steps < 0
    ):
        parser.error("Clipped-run limit must be non-negative")
    if not arguments.preflight_only and not arguments.out_dir:
        parser.error("--out-dir is required unless --preflight-only is used")
    return arguments


def main() -> int:
    arguments = _arguments()
    raw_root = Path(arguments.raw_root).resolve()
    if raw_root.name != RAW_ROOT_BASENAME or not raw_root.is_dir():
        raise ValueError(f"--raw-root must be the authoritative {RAW_ROOT_BASENAME}")
    converter = _load_converter()
    if converter.suite.__version__ != EXPECTED_ROBOSUITE_VERSION:
        raise RuntimeError(
            f"Expected libero-industry Robosuite {EXPECTED_ROBOSUITE_VERSION}; "
            f"received {converter.suite.__version__}"
        )
    selected_tasks = [
        task
        for task in TASKS
        if (not arguments.suite or task["suite"] in arguments.suite)
        and (not arguments.task or task["case_id"] in arguments.task)
    ]
    if not selected_tasks:
        raise ValueError("Suite/task filters selected no recovery tasks")
    arguments.resume_tasks = (
        _resume_tasks(arguments.resume_from) if arguments.resume_from else {}
    )
    if arguments.resume_from:
        missing_resume_tasks = sorted(
            task["case_id"]
            for task in selected_tasks
            if task["case_id"] not in arguments.resume_tasks
        )
        if missing_resume_tasks:
            raise ValueError(
                f"Resume summary is missing selected tasks: {missing_resume_tasks}"
            )
    if arguments.diagnostic_video and len(selected_tasks) != 1:
        raise ValueError("--diagnostic-video requires exactly one selected task")
    if arguments.diagnostic_video and "phases" in selected_tasks[0]:
        raise ValueError(
            "Stage-wise tasks must be collected, then viewed with the trajectory renderer"
        )
    if arguments.target_demo and len(selected_tasks) != 1:
        raise ValueError("--target-demo requires exactly one selected task")
    if arguments.preflight_only:
        rows = []
        for task in selected_tasks:
            row = (
                _preflight(task, raw_root, converter, arguments)
                if "phases" in task
                else _preflight_task(task, raw_root, converter, arguments)
            )
            rows.append(row)
            print(
                f"[PREFLIGHT] {row['case_id']} "
                f"failed={row['failed_identity_count']} "
                f"kinematic={row['raw_kinematic_rejection_identity_count']} "
                f"regenerate={row['target_identity_count']}",
                flush=True,
            )
        print(
            f"[PREFLIGHT END] tasks={len(rows)} "
            f"failed_identities={sum(row['failed_identity_count'] for row in rows)} "
            f"regenerate={sum(row['target_identity_count'] for row in rows)}",
            flush=True,
        )
        return 0

    output_root = Path(arguments.out_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing existing output directory: {output_root}")
    output_root.mkdir(parents=True)
    config = {
        "method": "MimicGen-style object-centric subtask recombination",
        "tasks": [task["case_id"] for task in selected_tasks],
        "source_demos_per_task": arguments.source_demos,
        "attempts_per_failed_identity": arguments.attempts_per_identity,
        "seed": arguments.seed,
        "raw_initialization_seed": RAW_INITIALIZATION_SEED,
        "resume_from": str(arguments.resume_from) if arguments.resume_from else None,
        "limit_identities_per_task": arguments.limit_identities,
        "target_failed_demo": arguments.target_demo,
        "replay_policy": "one saved-action replay per accepted episode; diagnostic only",
        "success_validation_policy": (
            "unrecorded final settling; no no-op dataset transitions"
        ),
        "success_validation_max_steps": arguments.final_hold_steps,
        "target_stratum": "raw task failures or unstable final-five success",
        "clipping_acceptance_policy": "task_raw_terminal_success_maximum",
        "varied": [
            "target_initial_demo",
            "pick_source_segment",
            "place_source_segment",
            "subtask_boundary_offset",
        ],
        "fixed": {
            "nearest_neighbor_pool": arguments.nearest_neighbors,
            "boundary_offset_range": [-arguments.boundary_offset, arguments.boundary_offset],
            "action_noise": 0.0,
            "interpolation_steps": arguments.interpolation_steps,
            "reference_substeps": arguments.reference_substeps,
            "fixed_steps": arguments.fixed_steps,
            "final_hold_steps": arguments.final_hold_steps,
            "success_dwell_steps": arguments.success_dwell_steps,
            "minimum_arm_joint_margin_rad": arguments.min_arm_joint_margin_rad,
            "minimum_jacobian_sigma": arguments.minimum_jacobian_sigma,
            "maximum_jacobian_condition": arguments.maximum_jacobian_condition,
            "jacobian_rotation_length_m": arguments.jacobian_rotation_length_m,
            "maximum_clipped_cartesian_component_pct_override": (
                arguments.max_clipped_component_pct
            ),
            "maximum_consecutive_clipped_steps_override": (
                arguments.max_clipped_run_steps
            ),
        },
        "raw_root_read_only": str(raw_root),
        "recovered_source_root": str(arguments.recovered_source_root.resolve())
        if arguments.recovered_source_root
        else None,
        "other_aubo_roots_used": bool(arguments.recovered_source_root),
        "broken_ik_reference_solver_used": False,
        "old_event_pinned_solver_used": False,
        "video_enabled": bool(arguments.diagnostic_video),
        "diagnostic_video": arguments.diagnostic_video,
        "hdf5_output_enabled": not bool(arguments.diagnostic_video),
        "accepted_outputs_include_target_camera_observations": not bool(
            arguments.diagnostic_video
        ),
        "robosuite_version": converter.suite.__version__,
        "mujoco_version": converter.mujoco.__version__,
    }
    _atomic_json(output_root / "run_config.json", config)
    summaries = []
    errors = []
    for task in selected_tasks:
        try:
            summaries.append(
                (
                    _run_stagewise_task
                    if "phases" in task
                    else _run_task
                )(task, raw_root, output_root, arguments, converter)
            )
        except Exception as error:
            payload = {
                "case_id": task["case_id"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(output_root / f"{task['case_id']}.error.json", payload)
            errors.append(payload)
            print(f"[TASK ERROR] {task['case_id']}: {error}", flush=True)
    final = {
        "completed_task_count": len(summaries),
        "errored_task_count": len(errors),
        "raw_failed_identity_count": sum(
            row["raw_failed_identity_count"] for row in summaries
        ),
        "raw_unstable_success_identity_count": sum(
            row["raw_unstable_success_identity_count"] for row in summaries
        ),
        "raw_retained_identity_count": sum(
            row["raw_retained_identity_count"] for row in summaries
        ),
        "target_identity_count": sum(
            row["target_identity_count"] for row in summaries
        ),
        "selected_failed_identity_count": sum(
            row["selected_failed_identity_count"] for row in summaries
        ),
        "processed_identity_count": sum(
            row["processed_identity_count"] for row in summaries
        ),
        "recovered_identity_count": sum(
            row["recovered_identity_count"] for row in summaries
        ),
        "unrecovered_identity_count": sum(
            row["unrecovered_identity_count"] for row in summaries
        ),
        "all_selected_identities_processed": len(summaries) == len(selected_tasks)
        and all(row["all_selected_identities_processed"] for row in summaries),
        "diagnostic_only": bool(arguments.diagnostic_video),
        "tasks": summaries,
        "errors": errors,
    }
    _atomic_json(output_root / "summary.json", final)
    print(
        f"[RUN END] tasks={len(summaries)}/{len(selected_tasks)} "
        f"processed={final['processed_identity_count']} "
        f"recovered={final['recovered_identity_count']} "
        f"unrecovered={final['unrecovered_identity_count']} errors={len(errors)}",
        flush=True,
    )
    if errors:
        return 1
    if arguments.diagnostic_video and final["recovered_identity_count"] != 1:
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
