"""Audit stored Aubo joint configurations for Jacobian singularities.

This is a state-only audit: it restores recorded arm qpos, calls sim.forward(),
and evaluates the repository's existing Jacobian safety metrics. It never steps
the controller, renders images, or modifies the input dataset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np


os.environ.setdefault("MUJOCO_GL", "osmesa")

SCRIPT_DIR = Path(__file__).resolve().parent
CONVERSION_DIR = SCRIPT_DIR.parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[2]
AUTO_DEMO_DIR = REPOSITORY_ROOT / "notebooks" / "auto_demo"
for path in (CONVERSION_DIR, AUTO_DEMO_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import convert_to_aubo as converter  # noqa: E402
from kinematic_safety import KinematicAnalyzer  # noqa: E402


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sorted_demo_names(group) -> list[str]:
    return sorted(group, key=lambda name: int(name[5:]))


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return []
    breaks = np.flatnonzero(np.diff(indexes) > 1)
    starts = np.r_[indexes[0], indexes[breaks + 1]]
    ends = np.r_[indexes[breaks], indexes[-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _resolve_bddl(hdf5_path: Path, env_meta: dict, data_bddl_hint=None) -> Path:
    bddl_root = REPOSITORY_ROOT / "libero" / "libero" / "bddl_files"
    task_name = hdf5_path.stem[:-5] if hdf5_path.stem.endswith("_demo") else hdf5_path.stem
    hints = (
        data_bddl_hint,
        env_meta.get("env_kwargs", {}).get("bddl_file_name"),
        f"{task_name}.bddl",
        env_meta.get("bddl_file"),
    )
    names = list(dict.fromkeys(Path(_text(hint)).name for hint in hints if hint))
    for name in names:
        path = bddl_root / hdf5_path.parent.name / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No BDDL for {hdf5_path}; tried "
        + ", ".join(str(bddl_root / hdf5_path.parent.name / name) for name in names)
    )


def _create_environment(hdf5_path: Path, env_meta: dict, data_bddl_hint=None):
    env_meta = json.loads(json.dumps(env_meta))
    kwargs = env_meta.setdefault("env_kwargs", {})
    kwargs.update(
        {
            "bddl_file_name": str(_resolve_bddl(hdf5_path, env_meta, data_bddl_hint)),
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "camera_names": [],
        }
    )
    return converter.EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[],
        camera_height=84,
        camera_width=84,
        reward_shaping=False,
    )


def _window_summary(
    start: int,
    end: int,
    *,
    sigma: np.ndarray,
    condition: np.ndarray,
    margin: np.ndarray,
    elbow: np.ndarray,
    reach: np.ndarray,
    actions: np.ndarray,
    fps: float,
) -> dict:
    selection = slice(start, end + 1)
    closed = actions[selection, 6] > 0.0
    reach_delta = np.diff(reach, prepend=reach[0])[selection]
    return {
        "start_frame": start,
        "end_frame": end,
        "frame_count": end - start + 1,
        "start_seconds": start / fps,
        "end_seconds": end / fps,
        "minimum_sigma": float(np.min(sigma[selection])),
        "maximum_condition_number": float(np.max(condition[selection])),
        "minimum_joint_margin_rad": float(np.min(margin[selection])),
        "j3_range_rad": [
            float(np.min(elbow[selection])),
            float(np.max(elbow[selection])),
        ],
        "maximum_tcp_reach_m": float(np.max(reach[selection])),
        "closed_gripper_fraction": float(np.mean(closed)),
        "closed_and_extending_frame_count": int(np.sum(closed & (reach_delta > 0.0))),
        "maximum_translation_action_norm": float(
            np.max(np.linalg.norm(actions[selection, :3], axis=1))
        ),
        "maximum_rotation_action_norm": float(
            np.max(np.linalg.norm(actions[selection, 3:6], axis=1))
        ),
    }


def _counter() -> dict:
    return {
        "demo_count": 0,
        "frame_count": 0,
        "singular_demo_count": 0,
        "singular_frame_count": 0,
        "joint_margin_demo_count": 0,
        "joint_margin_frame_count": 0,
    }


def _add_counts(target: dict, demo: dict) -> None:
    target["demo_count"] += 1
    target["frame_count"] += demo["frame_count"]
    target["singular_demo_count"] += int(demo["singular_frame_count"] > 0)
    target["singular_frame_count"] += demo["singular_frame_count"]
    target["joint_margin_demo_count"] += int(demo["joint_margin_frame_count"] > 0)
    target["joint_margin_frame_count"] += demo["joint_margin_frame_count"]


def _scan_file(path: Path, arguments) -> dict:
    with h5py.File(path, "r") as source:
        data = source["data"]
        env_meta = json.loads(_text(data.attrs["env_args"]))
        data_bddl_hint = data.attrs.get("bddl_file_name")

    environment = _create_environment(path, env_meta, data_bddl_hint)
    raw = converter._get_base_env(environment)
    analyzer = KinematicAnalyzer(raw)
    sim, eef_name, eef_id, qpos_addrs, dof_addrs, limits = analyzer._joint_data()

    demos = []
    maximum_tcp_error = 0.0
    try:
        with h5py.File(path, "r") as source:
            for demo_name in _sorted_demo_names(source["data"]):
                group = source["data"][demo_name]
                qpos = np.asarray(group["obs/robot0_joint_pos"][:], dtype=float)
                actions = np.asarray(group["actions"][:], dtype=float)
                expected_tcp = np.asarray(group["obs/robot0_eef_pos"][:], dtype=float)
                if qpos.shape[0] != actions.shape[0] or qpos.shape[0] != expected_tcp.shape[0]:
                    raise ValueError(f"{path}:{demo_name}: action/observation lengths differ")
                if qpos.shape[1] != len(qpos_addrs) or actions.shape[1] != 7:
                    raise ValueError(
                        f"{path}:{demo_name}: expected 6D qpos and 7D action, got "
                        f"{qpos.shape} and {actions.shape}"
                    )

                frame_count = len(qpos)
                sigma = np.empty(frame_count)
                condition = np.empty(frame_count)
                margin = np.empty(frame_count)
                elbow = qpos[:, 2].copy()
                reach = np.empty(frame_count)
                for frame, (joints, tcp) in enumerate(zip(qpos, expected_tcp)):
                    sim.data.qpos[qpos_addrs] = joints
                    sim.forward()
                    jacobian = analyzer._arm_jacobian(sim, eef_name, eef_id, dof_addrs)
                    sigma[frame], condition[frame] = analyzer._jacobian_metrics(
                        jacobian, arguments.rotation_length_m
                    )
                    margin[frame] = np.min(
                        np.minimum(joints - limits[:, 0], limits[:, 1] - joints)
                    )
                    live_tcp = np.asarray(sim.data.site_xpos[eef_id], dtype=float)
                    reach[frame] = np.linalg.norm(live_tcp - analyzer.base_pos)
                    maximum_tcp_error = max(
                        maximum_tcp_error, float(np.linalg.norm(live_tcp - tcp))
                    )

                singular = (sigma < arguments.minimum_sigma) | (
                    condition > arguments.maximum_condition_number
                )
                joint_margin = margin < arguments.minimum_joint_margin_rad
                windows = [
                    _window_summary(
                        start,
                        end,
                        sigma=sigma,
                        condition=condition,
                        margin=margin,
                        elbow=elbow,
                        reach=reach,
                        actions=actions,
                        fps=arguments.fps,
                    )
                    for start, end in _contiguous_runs(singular | joint_margin)
                ]
                origin = _text(group.attrs.get("materialization_origin", "unknown"))
                demos.append(
                    {
                        "demo": demo_name,
                        "materialization_origin": origin,
                        "frame_count": frame_count,
                        "minimum_sigma": float(np.min(sigma)),
                        "maximum_condition_number": float(np.max(condition)),
                        "minimum_joint_margin_rad": float(np.min(margin)),
                        "j3_range_rad": [float(np.min(elbow)), float(np.max(elbow))],
                        "maximum_tcp_reach_m": float(np.max(reach)),
                        "singular_frame_count": int(np.sum(singular)),
                        "joint_margin_frame_count": int(np.sum(joint_margin)),
                        "flagged_windows": windows,
                    }
                )
    finally:
        converter._close_env(environment)

    counts = _counter()
    origins = defaultdict(_counter)
    for demo in demos:
        _add_counts(counts, demo)
        _add_counts(origins[demo["materialization_origin"]], demo)
    return {
        "suite": path.parent.name,
        "task": path.stem,
        "hdf5": str(path),
        **counts,
        "maximum_reconstructed_tcp_error_m": maximum_tcp_error,
        "by_materialization_origin": dict(origins),
        "demos": demos,
    }


def _self_test() -> None:
    mask = np.asarray([False, True, True, False, True, False])
    assert _contiguous_runs(mask) == [(1, 2), (4, 4)]
    sigma, condition = KinematicAnalyzer._jacobian_metrics(np.eye(6), 1.0)
    assert sigma == 1.0 and condition == 1.0
    counter = _counter()
    _add_counts(
        counter,
        {
            "frame_count": 3,
            "singular_frame_count": 2,
            "joint_margin_frame_count": 0,
        },
    )
    assert counter["demo_count"] == 1 and counter["singular_demo_count"] == 1
    print("self-test passed", flush=True)


def _parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-sigma", type=float, default=0.02)
    parser.add_argument("--maximum-condition-number", type=float, default=80.0)
    parser.add_argument("--minimum-joint-margin-rad", type=float, default=0.03)
    parser.add_argument("--rotation-length-m", type=float, default=0.10)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if arguments.self_test:
        _self_test()
        return 0
    if arguments.input_root is None or arguments.output is None:
        raise ValueError("--input-root and --output are required unless --self-test is used")
    if tuple(int(part) for part in converter.suite.__version__.split(".")[:3]) != (1, 4, 0):
        raise RuntimeError(
            f"This recovered-dataset audit requires Robosuite 1.4.0, got "
            f"{converter.suite.__version__}"
        )

    files = sorted(arguments.input_root.glob("*/*.hdf5"))
    if arguments.limit_files is not None:
        files = files[: arguments.limit_files]
    if not files:
        raise FileNotFoundError(f"No suite HDF5 files found under {arguments.input_root}")

    print(
        f"[START] files={len(files)} input={arguments.input_root} "
        f"sigma<{arguments.minimum_sigma} condition>{arguments.maximum_condition_number} "
        f"joint_margin<{arguments.minimum_joint_margin_rad} "
        f"rotation_length={arguments.rotation_length_m}",
        flush=True,
    )
    started = time.time()
    tasks = []
    totals = _counter()
    origins = defaultdict(_counter)
    for index, path in enumerate(files, start=1):
        task_started = time.time()
        task = _scan_file(path, arguments)
        tasks.append(task)
        task_summary = {
            key: task[key]
            for key in (
                "frame_count",
                "singular_frame_count",
                "joint_margin_frame_count",
            )
        }
        for demo in task["demos"]:
            _add_counts(totals, demo)
            _add_counts(origins[demo["materialization_origin"]], demo)
        print(
            f"[{index:02d}/{len(files):02d}] {task['suite']}/{task['task']} "
            f"frames={task_summary['frame_count']} "
            f"singular={task_summary['singular_frame_count']} "
            f"joint_margin={task_summary['joint_margin_frame_count']} "
            f"seconds={time.time() - task_started:.2f}",
            flush=True,
        )

    report = {
        "status": "complete",
        "method": "stored_arm_qpos_sim_forward_no_rollout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.time() - started,
        "input_root": str(arguments.input_root),
        "robosuite_version": converter.suite.__version__,
        "thresholds": {
            "minimum_sigma": arguments.minimum_sigma,
            "maximum_condition_number": arguments.maximum_condition_number,
            "minimum_joint_margin_rad": arguments.minimum_joint_margin_rad,
            "rotation_length_m": arguments.rotation_length_m,
        },
        "task_count": len(tasks),
        **totals,
        "by_materialization_origin": dict(origins),
        "tasks": tasks,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, arguments.output)
    print(
        f"[COMPLETE] demos={totals['demo_count']} frames={totals['frame_count']} "
        f"singular_demos={totals['singular_demo_count']} "
        f"singular_frames={totals['singular_frame_count']} "
        f"joint_margin_demos={totals['joint_margin_demo_count']} "
        f"joint_margin_frames={totals['joint_margin_frame_count']} "
        f"seconds={report['duration_seconds']:.2f} report={arguments.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
