"""Save a Panda or Aubo wrist-camera image from an Industry or Lift scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
import numpy as np
import robosuite

from libero.libero.envs import OffScreenRenderEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BDDL = REPO_ROOT / (
    "libero/libero/bddl_files/libero_industry_d2d/"
    "D2D_LV1_002_V5_put_all_objects_from_the_conveyor_into_the_moving_tray.bddl"
)
ROBOSUITE_V15 = tuple(
    int(part) for part in robosuite.__version__.split(".")[:2]
) >= (1, 5)
CAMERA_REQUEST_NAME = (
    "gripper0_right_eye_in_hand" if ROBOSUITE_V15 else "gripper0_eye_in_hand"
)
RUNTIME_CAMERA_NAMES = (
    CAMERA_REQUEST_NAME,
    "gripper0_eye_in_hand",
    "gripper0_right_eye_in_hand",
    "robot0_eye_in_hand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a live window for the simulated Aubo wrist camera."
    )
    parser.add_argument(
        "--bddl",
        type=Path,
        default=DEFAULT_BDDL,
        help="LIBERO Industry BDDL scene to load.",
    )
    parser.add_argument(
        "--robot",
        choices=("panda", "aubo"),
        default="aubo",
        help="Robot embodiment to render (default: aubo).",
    )
    parser.add_argument(
        "--scene",
        choices=("industry", "lift"),
        default="industry",
        help="Scene to render (default: industry).",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=REPO_ROOT / "wrist_camera.png",
        help="PNG written by the S key or in --no-window mode.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Render one frame to --snapshot without opening a window.",
    )
    parser.add_argument(
        "--no-crosshair",
        action="store_true",
        help="Do not draw the green image-center marker.",
    )
    return parser.parse_args()


def find_wrist_image(observation: dict) -> tuple[str, np.ndarray]:
    preferred_keys = [f"{name}_image" for name in RUNTIME_CAMERA_NAMES]
    preferred_keys.append(f"{CAMERA_REQUEST_NAME}_image")
    for key in preferred_keys:
        image = observation.get(key)
        if image is not None:
            return key, image

    available = sorted(key for key in observation if key.endswith("_image"))
    raise KeyError(
        "No wrist-camera image was returned. Available image keys: "
        + ", ".join(available)
    )


def find_runtime_camera(env: OffScreenRenderEnv) -> tuple[str, int]:
    for name in RUNTIME_CAMERA_NAMES:
        try:
            return name, int(env.sim.model.camera_name2id(name))
        except (KeyError, ValueError):
            continue
    raise ValueError(
        "Could not find a compiled wrist camera named "
        + ", ".join(RUNTIME_CAMERA_NAMES)
    )


def prepare_frame(rgb_image: np.ndarray, crosshair: bool) -> np.ndarray:
    # Robosuite's off-screen observation has an OpenGL-style bottom-left origin.
    frame = cv2.cvtColor(np.flipud(rgb_image), cv2.COLOR_RGB2BGR)
    if crosshair:
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        cv2.drawMarker(
            frame,
            center,
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=1,
        )
    return frame


def save_snapshot(path: Path, frame: np.ndarray) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise OSError(f"OpenCV could not write {path}")
    print(f"Saved {path}")


def main() -> None:
    args = parse_args()
    bddl_path = args.bddl.expanduser().resolve()
    if args.scene == "industry" and not bddl_path.is_file():
        raise FileNotFoundError(f"BDDL file does not exist: {bddl_path}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    camera_request_name = (
        "robot0_eye_in_hand" if args.robot == "panda" else CAMERA_REQUEST_NAME
    )
    env_kwargs = dict(
        robots=[
            ("Panda" if args.robot == "panda" else "AuboI5")
            if args.scene == "industry"
            else ("MountedPanda" if args.robot == "panda" else "MountedAuboI5")
        ],
        camera_names=[camera_request_name],
        camera_widths=args.width,
        camera_heights=args.height,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        initialization_noise=None,
    )
    if args.scene == "industry":
        env_kwargs["bddl_file_name"] = str(bddl_path)
    if args.robot == "aubo":
        env_kwargs["gripper_types"] = "CTAG2F90DGripper"
    env = (
        OffScreenRenderEnv(**env_kwargs)
        if args.scene == "industry"
        else robosuite.make(env_name="Lift", **env_kwargs)
    )

    try:
        observation = env.reset()
        image_key, rgb_image = find_wrist_image(observation)
        camera_name, camera_id = find_runtime_camera(env)
        model = env.sim.model
        print(
            f"Robot: {args.robot}\n"
            f"Camera: {camera_name} (observation: {image_key})\n"
            f"Resolution: {rgb_image.shape[1]}x{rgb_image.shape[0]}\n"
            f"pos={model.cam_pos[camera_id]} quat={model.cam_quat[camera_id]} "
            f"fovy={model.cam_fovy[camera_id]:g}"
        )

        frame = prepare_frame(rgb_image, crosshair=not args.no_crosshair)
        if args.no_window:
            save_snapshot(args.snapshot, frame)
            return

        window_name = f"{args.robot.title()} wrist camera - S: save, Q/Esc: close"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)
        while True:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_snapshot(args.snapshot, frame)
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
