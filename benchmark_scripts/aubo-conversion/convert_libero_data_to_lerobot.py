"""Convert recovered-v2 Aubo demonstrations to the OpenPI LeRobot v2.1 contract."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_FPS = 10
V21_VERSION = "v2.1"
EXPECTED_SUITE_COUNT = 4
EXPECTED_TASK_COUNT = 40
EXPECTED_EPISODE_COUNT = 2000
EXPECTED_SUITE_TASK_COUNT = 10
EXPECTED_SUITE_EPISODE_COUNT = 500
EXPECTED_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
NOOP_EPSILON = 1e-4

EEF_POSITION_KEY = "obs/robot0_eef_pos"
EEF_QUATERNION_KEY = "obs/robot0_eef_quat"
GRIPPER_STATE_KEY = "obs/robot0_gripper_qpos"
BASE_IMAGE_KEY = "obs/agentview_image"
WRIST_IMAGE_KEYS = (
    "obs/gripper0_eye_in_hand_image",
    "obs/gripper0_right_eye_in_hand_image",
    "obs/robot0_eye_in_hand_image",
)

BASE_FEATURE_KEY = "image"
WRIST_FEATURE_KEY = "wrist_image"
STATE_FEATURE_KEY = "state"
ACTION_FEATURE_KEY = "actions"
STATE_NAMES = [
    "eef_x_m",
    "eef_y_m",
    "eef_z_m",
    "eef_axis_angle_x_rad",
    "eef_axis_angle_y_rad",
    "eef_axis_angle_z_rad",
    "ctag_driver_qpos",
]
ACTION_NAMES = [
    "eef_delta_x",
    "eef_delta_y",
    "eef_delta_z",
    "eef_delta_axis_angle_x",
    "eef_delta_axis_angle_y",
    "eef_delta_axis_angle_z",
    "ctag_command",
]


@dataclasses.dataclass(frozen=True)
class Episode:
    suite: str
    hdf5_path: Path
    demo: str
    horizon: int
    task: str
    wrist_image_key: str
    image_shape: tuple[int, int, int]


def _load_lerobot_dataset() -> Any:
    try:
        from lerobot.common.datasets.lerobot_dataset import (
            CODEBASE_VERSION,
            LeRobotDataset,
        )
    except ImportError as error:
        raise RuntimeError(
            "This exporter requires a LeRobot v2.1 installation."
        ) from error
    if CODEBASE_VERSION != V21_VERSION:
        raise RuntimeError(f"Expected LeRobot {V21_VERSION}, found {CODEBASE_VERSION}.")
    return LeRobotDataset


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    return str(value)


def _task_instruction(data_group: h5py.Group, path: Path) -> str:
    if "problem_info" not in data_group.attrs:
        raise KeyError(f"Missing problem_info in {path}")
    problem_info = json.loads(_decode(data_group.attrs["problem_info"]))
    instruction = problem_info.get("language_instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Missing language instruction in {path}")
    return instruction.strip()


def _wrist_key(group: h5py.Group, path: Path, demo: str) -> str:
    for key in WRIST_IMAGE_KEYS:
        if key in group:
            return key
    raise KeyError(f"Missing wrist image in {path}:{demo}")


def _quat_xyzw_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    """Match OpenPI LIBERO's xyzw-quaternion to axis-angle projection."""
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (T, 4), found {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Quaternion contains NaN or inf")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("Quaternion is not normalized")

    scalar = np.clip(values[:, 3], -1.0, 1.0)
    denominator = np.sqrt(np.maximum(0.0, 1.0 - np.square(scalar)))
    scale = np.zeros_like(scalar)
    nonzero = denominator > 1e-12
    scale[nonzero] = 2.0 * np.arccos(scalar[nonzero]) / denominator[nonzero]
    return values[:, :3] * scale[:, None]


def _policy_state(
    eef_position: np.ndarray,
    eef_quaternion: np.ndarray,
    gripper_qpos: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(eef_position, dtype=np.float64)
    gripper = np.asarray(gripper_qpos, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected EEF-position shape (T, 3), found {positions.shape}")
    if gripper.ndim != 2 or gripper.shape[1] < 1:
        raise ValueError(f"Expected gripper-qpos shape (T, >=1), found {gripper.shape}")
    if len(positions) != len(eef_quaternion) or len(positions) != len(gripper):
        raise ValueError("EEF and gripper state lengths differ")
    state = np.concatenate(
        [positions, _quat_xyzw_to_axis_angle(eef_quaternion), gripper[:, :1]],
        axis=1,
    )
    if not np.all(np.isfinite(state)):
        raise ValueError("Policy state contains NaN or inf")
    return state.astype(np.float32)


def _policy_image(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("Policy image must be HWC uint8 RGB")
    # The recovered HDF5 is vertically inverted from the simulator render, so
    # a horizontal flip matches OpenPI's published 180-degree LIBERO view.
    return np.ascontiguousarray(values[:, ::-1])


def _noop_steps(actions: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"Actions must have shape (T, 7), found {values.shape}")
    mask = np.linalg.norm(values[1:, :6], axis=1) < NOOP_EPSILON
    mask &= (
        np.abs(np.diff(values[:, 6])) <= NOOP_EPSILON
    )
    return tuple(int(index) for index in np.flatnonzero(mask) + 1)


def _features(image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    return {
        BASE_FEATURE_KEY: {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        WRIST_FEATURE_KEY: {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": STATE_NAMES,
        },
        ACTION_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (7,),
            "names": ACTION_NAMES,
        },
    }


def _validate_materialized_source(data_root: Path) -> dict[str, Any]:
    report_path = data_root / "materialization_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Missing final recovered-dataset report: {report_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = {
        "ready_for_training": True,
        "suite_count": EXPECTED_SUITE_COUNT,
        "task_count": EXPECTED_TASK_COUNT,
    }
    mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in required.items()
        if report.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Recovered dataset is not ready for export: {mismatches}")
    final_count = report.get("final_demo_count")
    missing_count = report.get("unrecovered_identity_count")
    complete = final_count == EXPECTED_EPISODE_COUNT and missing_count == 0
    approved_incomplete = (
        report.get("incomplete_dataset_approved") is True
        and type(final_count) is int
        and type(missing_count) is int
        and final_count > 0
        and final_count + missing_count == EXPECTED_EPISODE_COUNT
    )
    if not complete and not approved_incomplete:
        raise ValueError("Recovered dataset totals are neither complete nor approved")
    if Path(report.get("output_root", "")).resolve() != data_root:
        raise ValueError("Materialization report points to a different output root")
    return report


def _validate_episode(suite: str, path: Path, demo: str) -> Episode:
    with h5py.File(path, "r") as hdf5_file:
        data = hdf5_file["data"]
        group = data[demo]
        for key in (
            "actions",
            "rewards",
            EEF_POSITION_KEY,
            EEF_QUATERNION_KEY,
            GRIPPER_STATE_KEY,
            BASE_IMAGE_KEY,
        ):
            if key not in group:
                raise KeyError(f"Missing {key} in {path}:{demo}")
        wrist_key = _wrist_key(group, path, demo)

        actions = group["actions"]
        eef_positions = group[EEF_POSITION_KEY]
        eef_quaternions = group[EEF_QUATERNION_KEY]
        gripper_state = group[GRIPPER_STATE_KEY]
        base_images = group[BASE_IMAGE_KEY]
        wrist_images = group[wrist_key]
        horizon = len(actions)
        expected_shapes = {
            "actions": (horizon, 7),
            EEF_POSITION_KEY: (horizon, 3),
            EEF_QUATERNION_KEY: (horizon, 4),
        }
        for key, expected_shape in expected_shapes.items():
            if group[key].shape != expected_shape:
                raise ValueError(
                    f"Expected {expected_shape} for {key} in {path}:{demo}"
                )
        if gripper_state.ndim != 2 or gripper_state.shape[0] != horizon:
            raise ValueError(f"Invalid gripper state in {path}:{demo}")
        if base_images.shape[0] != horizon or wrist_images.shape[0] != horizon:
            raise ValueError(f"Image/action length mismatch in {path}:{demo}")
        if base_images.shape[1:] != wrist_images.shape[1:]:
            raise ValueError(f"Camera shapes differ in {path}:{demo}")
        if base_images.ndim != 4 or base_images.shape[-1] != 3:
            raise ValueError(f"Images are not HWC RGB in {path}:{demo}")
        if base_images.dtype != np.uint8 or wrist_images.dtype != np.uint8:
            raise TypeError(f"Images are not uint8 in {path}:{demo}")
        rewards = np.asarray(group["rewards"][()])
        stable_success = int(group.attrs.get("terminal_success_streak", 0)) >= 5
        if rewards.shape != (horizon,) or horizon < 5 or (
            not stable_success and not np.all(rewards[-5:] > 0.0)
        ):
            raise ValueError(f"Demo lacks stable-success validation: {path}:{demo}")

        action_values = actions[()]
        _policy_state(eef_positions[()], eef_quaternions[()], gripper_state[()])
        if not np.all(np.isfinite(action_values)):
            raise ValueError(f"Actions contain NaN or inf in {path}:{demo}")
        if np.max(np.abs(action_values)) > 1.0 + 1e-6:
            raise ValueError(f"Action outside [-1, 1] in {path}:{demo}")
        if _noop_steps(action_values):
            raise ValueError(f"Demo contains a no-op action: {path}:{demo}")

        return Episode(
            suite=suite,
            hdf5_path=path,
            demo=demo,
            horizon=horizon,
            task=_task_instruction(data, path),
            wrist_image_key=wrist_key,
            image_shape=tuple(int(value) for value in base_images.shape[1:]),
        )


def _select_hdf5_paths(paths: list[Path], suite: str | None) -> list[Path]:
    if suite is None:
        return paths
    if suite not in EXPECTED_SUITES:
        raise ValueError(f"Unsupported suite: {suite}")
    selected = [path for path in paths if path.parent.name == suite]
    if len(selected) != EXPECTED_SUITE_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SUITE_TASK_COUNT} task HDF5s in {suite}, "
            f"found {len(selected)}"
        )
    return selected


def discover_episodes(data_dir: str | Path, suite: str | None = None) -> list[Episode]:
    data_root = Path(data_dir).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    report = _validate_materialized_source(data_root)
    hdf5_paths = sorted(data_root.glob("*/*.hdf5"))
    if len(hdf5_paths) != EXPECTED_TASK_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TASK_COUNT} task HDF5s, found {len(hdf5_paths)}"
        )
    hdf5_paths = _select_hdf5_paths(hdf5_paths, suite)

    episodes = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as hdf5_file:
            demos = sorted(hdf5_file["data"].keys())
        episodes.extend(
            _validate_episode(path.parent.name, path, demo) for demo in demos
        )
    if suite is None:
        expected_episode_count = report["final_demo_count"]
    else:
        expected_episode_count = next(
            row["final_demo_count"] for row in report["suites"] if row["suite"] == suite
        )
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"Expected {expected_episode_count} demos, found {len(episodes)}"
        )
    if len({episode.image_shape for episode in episodes}) != 1:
        raise ValueError("Image shapes are inconsistent")
    return episodes


def _limit_episodes_per_suite(
    episodes: list[Episode], limit: int | None
) -> list[Episode]:
    if limit is None:
        return episodes
    if limit < 1:
        raise ValueError("episodes_per_suite must be positive")
    selected = []
    counts = {suite: 0 for suite in {episode.suite for episode in episodes}}
    for episode in episodes:
        if counts[episode.suite] < limit:
            selected.append(episode)
            counts[episode.suite] += 1
    if any(count != limit for count in counts.values()):
        raise ValueError(f"Cannot select {limit} episodes from every suite: {counts}")
    return selected


def main(
    data_dir: str,
    *,
    output_root: str,
    repo_name: str,
    suite: str | None = None,
    episodes_per_suite: int | None = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> None:
    LeRobotDataset = _load_lerobot_dataset()
    episodes = discover_episodes(data_dir, suite=suite)
    episodes = _limit_episodes_per_suite(episodes, episodes_per_suite)
    output_path = Path(output_root).resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"LeRobot output exists: {output_path}; use --overwrite to replace it"
            )
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="aubo_i5_ctag_2f90d",
        fps=DEFAULT_FPS,
        features=_features(episodes[0].image_shape),
        use_videos=False,
        image_writer_threads=8,
    )

    manifest = []
    for index, episode in enumerate(episodes, start=1):
        with h5py.File(episode.hdf5_path, "r") as hdf5_file:
            group = hdf5_file[f"data/{episode.demo}"]
            actions = np.asarray(group["actions"][()], dtype=np.float32)
            states = _policy_state(
                group[EEF_POSITION_KEY][()],
                group[EEF_QUATERNION_KEY][()],
                group[GRIPPER_STATE_KEY][()],
            )
            base_images = group[BASE_IMAGE_KEY][()]
            wrist_images = group[episode.wrist_image_key][()]
            for step in range(episode.horizon):
                dataset.add_frame(
                    {
                        BASE_FEATURE_KEY: _policy_image(base_images[step]),
                        WRIST_FEATURE_KEY: _policy_image(wrist_images[step]),
                        STATE_FEATURE_KEY: states[step],
                        ACTION_FEATURE_KEY: actions[step],
                        "task": episode.task,
                    }
                )
        dataset.save_episode()
        manifest.append(
            {
                "episode_index": index - 1,
                "suite": episode.suite,
                "hdf5": episode.hdf5_path.name,
                "demo": episode.demo,
                "horizon": episode.horizon,
                "task": episode.task,
            }
        )
        print(
            f"[{index}/{len(episodes)}] "
            f"{episode.suite}/{episode.hdf5_path.name}:{episode.demo}",
            flush=True,
        )

    summary = {
        "source_dataset": "Lifelong-Robot-Learning/LIBERO",
        "source_release": Path(data_dir).resolve().name,
        "source_license": "CC-BY-4.0",
        "dataset_id": repo_name,
        "lerobot_dataset_version": V21_VERSION,
        "suite_filter": suite,
        "episodes_per_suite": episodes_per_suite,
        "episode_count": len(episodes),
        "frame_count": sum(episode.horizon for episode in episodes),
        "fps": DEFAULT_FPS,
        "image_transform": "horizontal_flip_to_openpi_180",
        "state": {
            "dimension": 7,
            "components": STATE_NAMES,
            "gripper_source": f"{GRIPPER_STATE_KEY}[0]",
        },
        "action": {"dimension": 7, "components": ACTION_NAMES},
        "features": _features(episodes[0].image_shape),
        "episodes": manifest,
    }
    (output_path / "aubo_conversion_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if dataset.num_episodes != len(episodes):
        raise RuntimeError(
            f"Finalized LeRobot dataset has {dataset.num_episodes} episodes"
        )
    if dataset.num_frames != summary["frame_count"]:
        raise RuntimeError(
            f"Finalized LeRobot dataset has {dataset.num_frames} frames"
        )
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "aubo-i5", "ctag-2f90d", f"lerobot-{V21_VERSION}"],
            private=True,
            push_videos=True,
            license="cc-by-4.0",
        )
    print(
        f"Converted {summary['episode_count']} demos / "
        f"{summary['frame_count']} frames to {output_path}.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--suite", choices=EXPECTED_SUITES)
    parser.add_argument("--episodes-per-suite", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.data_dir,
        output_root=arguments.output_root,
        repo_name=arguments.repo_name,
        suite=arguments.suite,
        episodes_per_suite=arguments.episodes_per_suite,
        overwrite=arguments.overwrite,
        push_to_hub=arguments.push_to_hub,
    )
