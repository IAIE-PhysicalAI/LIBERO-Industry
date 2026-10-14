"""Render one stored Aubo trajectory without modifying either HDF5 file.

The source Panda HDF5 supplies the recorded initial task scene.  The matching
Aubo HDF5 supplies the action sequence to replay.  This intentionally mirrors
the initialization used by ``convert_to_aubo.py`` so a recorded failure can be
viewed independently of the conversion batch.
"""

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


_SCRIPT_DIR = Path(__file__).resolve().parent
_CONVERSION_DIR = _SCRIPT_DIR.parent


def _import_helpers():
    sys.path[:0] = [str(_SCRIPT_DIR), str(_CONVERSION_DIR)]
    import convert_to_aubo as converter
    import diagnose_aubo_grasp_stability as diagnostic

    return converter, diagnostic


def _read_demo(source_path, converted_path, demo, source_demo=None):
    source_demo = source_demo or demo
    with h5py.File(source_path, "r") as source_file, h5py.File(
        converted_path, "r"
    ) as converted_file:
        if f"data/{source_demo}" not in source_file:
            raise KeyError(f"{source_path.name} does not contain data/{source_demo}.")
        if f"data/{demo}" not in converted_file:
            raise KeyError(f"{converted_path.name} does not contain data/{demo}.")
        source_demo_group = source_file[f"data/{source_demo}"]
        converted_demo = converted_file[f"data/{demo}"]
        states = np.asarray(source_demo_group["states"])
        actions = np.asarray(converted_demo["actions"])

    if len(states) == 0:
        raise ValueError(f"{source_path.name} data/{demo} has no states.")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"{converted_path.name} data/{demo} actions must have shape (T, 7), "
            f"got {actions.shape}."
        )
    return states, actions


def render_trajectory(
    source_path, converted_path, demo, video_path, fps, seed, source_demo=None
):
    converter, diagnostic = _import_helpers()
    source_path = Path(source_path).resolve()
    converted_path = Path(converted_path).resolve()
    video_path = Path(video_path).resolve()
    states, actions = _read_demo(source_path, converted_path, demo, source_demo)

    reference_env = None
    target_env = None
    try:
        reference_env, target_env, env_name = diagnostic._make_environments(
            source_path, converted_path, True, converter
        )
        observation = converter.initialize_target_replay(
            reference_env,
            target_env,
            states[0],
            actions[0, -1],
            7,
            seed=seed,
        )

        frames = []
        first_success_step = None
        for step, action in enumerate(actions):
            frame = observation.get("agentview_image")
            if frame is None:
                raise KeyError("agentview_image is unavailable; cannot render video.")
            frames.append(np.asarray(frame))
            observation, _, _, _ = target_env.step(action)
            if converter._task_success(target_env) and first_success_step is None:
                first_success_step = step

        final_frame = observation.get("agentview_image")
        if final_frame is not None:
            frames.append(np.asarray(final_frame))
        if not frames:
            raise RuntimeError("Replay did not produce any video frames.")

        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(video_path, frames, fps=fps)
        terminal_success = converter._task_success(target_env)
        print(f"[WROTE] {video_path}")
        print(
            f"[RESULT] {demo}: {'SUCCESS' if terminal_success else 'FAIL'} "
            f"| first_success_step={first_success_step if first_success_step is not None else 'none'} "
            f"| frames={len(frames)} | environment={env_name}"
        )
    finally:
        converter._close_env(reference_env)
        converter._close_env(target_env)


def main():
    parser = argparse.ArgumentParser(
        description="Render a stored converted Aubo trajectory as an MP4 without HDF5 writes."
    )
    parser.add_argument("--source-hdf5", required=True, help="Original Panda HDF5.")
    parser.add_argument("--converted-hdf5", required=True, help="Matching converted Aubo HDF5.")
    parser.add_argument("--demo", required=True, help="Demo name, e.g. demo_7.")
    parser.add_argument(
        "--source-demo",
        help="Source Panda demo supplying the initial state (defaults to --demo).",
    )
    parser.add_argument("--video-out", required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=20, help="Output frames per second (default: 20).")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic target-reset seed shared with the converter (default: 0).",
    )
    arguments = parser.parse_args()
    if arguments.fps <= 0:
        parser.error("--fps must be positive")
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    render_trajectory(
        arguments.source_hdf5,
        arguments.converted_hdf5,
        arguments.demo,
        arguments.video_out,
        arguments.fps,
        arguments.seed,
        arguments.source_demo,
    )


if __name__ == "__main__":
    main()
