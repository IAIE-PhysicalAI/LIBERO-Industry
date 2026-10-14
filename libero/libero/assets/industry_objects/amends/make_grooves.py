"""Generate groove fixtures from an object's compiled collision geometry.

The default mode is read-only. Pass ``--write`` to replace the selected XML.
This script intentionally does not edit the Python object registry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


OBJECTS_DIR = Path(__file__).resolve().parent.parent

# Total clearance across each cavity dimension (5 mm on each side).
TOLERANCE = 0.01
WALL_THICKNESS = 0.002
FLOOR_THICKNESS = 0.0025
WALL_HEIGHT = 0.01

DEFAULT_TARGETS = (
    "thomas",
    "porter",
    "mario",
    "mini_fire_engine",
    "mini_roller",
    "school_bus",
)

CUSTOM_EXTENTS = {
    "headset": (0.11971 / 2.0, 0.13238 / 2.0, 0.06407 / 2.0),
}


def _source_xml(obj_name: str, orientation: str, base_dir: Path) -> Path:
    pose = "flat" if "flat" in orientation else "upright"
    candidates = (
        base_dir / obj_name / f"{obj_name}_{pose}.xml",
        base_dir / obj_name / f"{obj_name}.xml",
        base_dir / obj_name / f"{obj_name}_upright.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No source XML for {obj_name}; tried: {tried}")


def _is_descendant(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    current = body_id
    while current > 0:
        if current == ancestor_id:
            return True
        current = int(model.body_parentid[current])
    return current == ancestor_id


def collision_points_from_xml(xml_path: Path) -> np.ndarray:
    """Return compiled contact vertices in the named object's body frame."""

    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    if object_id < 0:
        raise ValueError(f"{xml_path} has no body named 'object'")

    object_pos = np.asarray(data.xpos[object_id], dtype=float)
    object_mat = np.asarray(data.xmat[object_id], dtype=float).reshape(3, 3)
    points: list[np.ndarray] = []

    for geom_id in range(model.ngeom):
        if not _is_descendant(model, int(model.geom_bodyid[geom_id]), object_id):
            continue
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError(
                f"Unsupported non-mesh collision geom {geom_id} in {xml_path}"
            )

        mesh_id = int(model.geom_dataid[geom_id])
        vertex_address = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vertex_address : vertex_address + vertex_count],
            dtype=float,
        )
        geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=float)
        geom_mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world_vertices = vertices @ geom_mat.T + geom_pos
        points.append((world_vertices - object_pos) @ object_mat)

    if not points:
        raise ValueError(f"{xml_path} has no contact-enabled mesh geometry")

    return np.vstack(points)


def collision_bounds_from_xml(xml_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return contact-geometry bounds in the named object's local body frame."""

    vertices = collision_points_from_xml(xml_path)
    return vertices.min(axis=0), vertices.max(axis=0)


def canonical_placement_quat(
    upright_xml: Path, target_xml: Path
) -> np.ndarray:
    """Fit the compiled upright-to-target collision-frame rotation."""

    upright_points = collision_points_from_xml(upright_xml)
    target_points = collision_points_from_xml(target_xml)
    if upright_points.shape != target_points.shape:
        raise ValueError(
            "Canonical assets have different compiled collision topology: "
            f"{upright_xml} {upright_points.shape} vs "
            f"{target_xml} {target_points.shape}"
        )

    covariance = upright_points.T @ target_points
    left, _, right_t = np.linalg.svd(covariance)
    row_rotation = left @ right_t
    if np.linalg.det(row_rotation) < 0.0:
        left[:, -1] *= -1.0
        row_rotation = left @ right_t
    rotation = row_rotation.T
    residual = np.sqrt(
        np.mean(
            np.sum(
                (upright_points @ rotation.T - target_points) ** 2,
                axis=1,
            )
        )
    )
    if residual > 1e-6:
        raise ValueError(
            "Canonical collision assets are not related by one rotation: "
            f"{upright_xml} -> {target_xml}, residual={residual:.6g} m"
        )

    quaternion = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def object_bounds(
    obj_name: str, orientation: str, base_dir: Path
) -> tuple[Path | None, np.ndarray, np.ndarray]:
    source_xml = _source_xml(obj_name, orientation, base_dir)
    if obj_name in CUSTOM_EXTENTS:
        half_extents = np.asarray(CUSTOM_EXTENTS[obj_name], dtype=float)
        return source_xml, -half_extents, half_extents

    minimum, maximum = collision_bounds_from_xml(source_xml)
    return source_xml, minimum, maximum


def _normal_groove(
    name: str,
    orientation: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
    goal_orientation: np.ndarray,
) -> tuple[str, str]:
    clearance = TOLERANCE / 2.0
    tw = WALL_THICKNESS
    ft = FLOOR_THICKNESS

    cavity_min = minimum[:2] - clearance
    cavity_max = maximum[:2] + clearance
    cavity_center = (cavity_min + cavity_max) / 2.0
    cavity_half = (cavity_max - cavity_min) / 2.0

    # The base reaches the outer faces of all four walls.
    outer_min = cavity_min - 2.0 * tw
    outer_max = cavity_max + 2.0 * tw
    floor_center = (outer_min + outer_max) / 2.0
    floor_half = (outer_max - outer_min) / 2.0
    wall_center_z = 2.0 * ft + WALL_HEIGHT
    top_z = 2.0 * ft + 2.0 * WALL_HEIGHT

    # A body origin within this site keeps its complete collision envelope
    # inside the cavity. The target Z is where the collision bottom meets the
    # top of the groove floor.
    goal_z = 2.0 * ft - float(minimum[2])
    goal_half = np.array([clearance, clearance])

    orient_str = "flat" if "flat" in orientation else "upright"
    groove_name = f"{name}_{orient_str}_groove"

    far_x = outer_min[0] if abs(outer_min[0]) > abs(outer_max[0]) else outer_max[0]
    far_y = outer_min[1] if abs(outer_min[1]) > abs(outer_max[1]) else outer_max[1]

    geoms = f"""              <geom name="groove_bottom" type="box" size="{floor_half[0]:.6f} {floor_half[1]:.6f} {ft:.6f}" pos="{floor_center[0]:.6f} {floor_center[1]:.6f} {ft:.6f}" rgba="0.6 0.6 0.6 1" friction="0.6 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="groove_wall_left" type="box" size="{tw:.6f} {cavity_half[1]:.6f} {WALL_HEIGHT:.6f}" pos="{cavity_min[0] - tw:.6f} {cavity_center[1]:.6f} {wall_center_z:.6f}" rgba="0.6 0.6 0.6 1" friction="0.6 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="groove_wall_right" type="box" size="{tw:.6f} {cavity_half[1]:.6f} {WALL_HEIGHT:.6f}" pos="{cavity_max[0] + tw:.6f} {cavity_center[1]:.6f} {wall_center_z:.6f}" rgba="0.6 0.6 0.6 1" friction="0.6 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="groove_wall_front" type="box" size="{cavity_half[0]:.6f} {tw:.6f} {WALL_HEIGHT:.6f}" pos="{cavity_center[0]:.6f} {cavity_min[1] - tw:.6f} {wall_center_z:.6f}" rgba="0.6 0.6 0.6 1" friction="0.6 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="groove_wall_back" type="box" size="{cavity_half[0]:.6f} {tw:.6f} {WALL_HEIGHT:.6f}" pos="{cavity_center[0]:.6f} {cavity_max[1] + tw:.6f} {wall_center_z:.6f}" rgba="0.6 0.6 0.6 1" friction="0.6 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <site name="goal_region" type="box" size="{goal_half[0]:.6f} {goal_half[1]:.6f} 0.001000" pos="0 0 {goal_z:.6f}" quat="1 0 0 0" rgba="0 1 0 0.4" group="0" />
              <site name="goal_orientation" type="sphere" size="0.001" pos="0 0 {goal_z:.6f}" quat="{' '.join(f'{value:.7f}' for value in goal_orientation)}" rgba="0 0 0 0" group="0" />"""

    return groove_name, _xml_document(
        groove_name,
        geoms,
        bottom_z=0.0,
        top_z=top_z,
        horizontal_site=(float(far_x), float(far_y)),
    )


def _v_groove(
    name: str,
    orientation: str,
    minimum: np.ndarray,
    maximum: np.ndarray,
    goal_orientation: np.ndarray,
) -> tuple[str, str]:
    """Retain the legacy 45-degree generator for non-cream-bottle fixtures."""

    h_x, h_y, h_z = (maximum - minimum) / 2.0
    hx = float(h_x + TOLERANCE / 2.0)
    hy = float(h_y + TOLERANCE / 2.0)
    hz = float(h_z + TOLERANCE / 2.0)
    tw = WALL_THICKNESS
    ft = FLOOR_THICKNESS
    max_wall = 0.045
    # A V-groove must support the same 45-degree object pose on either face.
    # Size both slants from the larger relevant object half-extent so the
    # generated fixture remains centred and symmetric even when hx != hz.
    wall_len = min(max(hx, hz), max_wall)
    left_len = wall_len
    right_len = wall_len
    left_reach = left_len * 2.0 * 0.7071
    right_reach = right_len * 2.0 * 0.7071
    max_height = max(left_len, right_len) * 2.0 * 0.7071
    center_x = (right_reach - left_reach) / 2.0
    floor_x = (left_reach + right_reach) / 2.0 + 2.0 * tw
    floor_y = hy + 2.0 * tw
    top_z = 2.0 * ft + max_height + tw
    cap_size_x = (left_reach + right_reach) / 2.0
    shift = tw * 0.7071
    cx_left = left_len * 0.7071 + shift
    cz_left = 2.0 * ft + left_len * 0.7071 - shift
    cx_right = right_len * 0.7071 + shift
    cz_right = 2.0 * ft + right_len * 0.7071 - shift
    goal_z = 2.0 * ft + 0.005

    suffix = "_flat" if "flat" in orientation else ""
    groove_name = f"{name}{suffix}_v_groove"
    geoms = f"""              <geom name="v_bottom" type="box" size="{floor_x:.6f} {floor_y:.6f} {ft:.6f}" pos="{center_x:.6f} 0 {ft:.6f}" rgba="0.4 0.4 0.4 1" friction="0.8 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="v_outer_left" type="box" size="{tw:.6f} {floor_y:.6f} {max_height / 2.0:.6f}" pos="{-left_reach - tw:.6f} 0 {2.0 * ft + max_height / 2.0:.6f}" rgba="0.4 0.4 0.4 1" friction="0.8 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="v_outer_right" type="box" size="{tw:.6f} {floor_y:.6f} {max_height / 2.0:.6f}" pos="{right_reach + tw:.6f} 0 {2.0 * ft + max_height / 2.0:.6f}" rgba="0.4 0.4 0.4 1" friction="0.8 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="v_cap_front" type="box" size="{cap_size_x:.6f} {tw:.6f} {max_height / 2.0:.6f}" pos="{center_x:.6f} {-hy - tw:.6f} {2.0 * ft + max_height / 2.0:.6f}" rgba="0.4 0.4 0.4 1" friction="0.8 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="v_cap_back" type="box" size="{cap_size_x:.6f} {tw:.6f} {max_height / 2.0:.6f}" pos="{center_x:.6f} {hy + tw:.6f} {2.0 * ft + max_height / 2.0:.6f}" rgba="0.4 0.4 0.4 1" friction="0.8 0.005 0.001" group="1" contype="1" conaffinity="1" />
              <geom name="v_slant_left" type="box" size="{tw:.6f} {hy:.6f} {left_len:.6f}" pos="{-cx_left:.6f} 0 {cz_left:.6f}" euler="0 -45 0" rgba="0.45 0.45 0.45 1" friction="2.0 0.1 0.001" solref="0.01 1" group="1" contype="1" conaffinity="1" />
              <geom name="v_slant_right" type="box" size="{tw:.6f} {hy:.6f} {right_len:.6f}" pos="{cx_right:.6f} 0 {cz_right:.6f}" euler="0 45 0" rgba="0.45 0.45 0.45 1" friction="2.0 0.1 0.001" solref="0.01 1" group="1" contype="1" conaffinity="1" />
              <site name="goal_region" type="box" size="0.005000 {hy:.6f} 0.001000" pos="0 0 {goal_z:.6f}" quat="1 0 0 0" rgba="0 1 0 0.4" group="0" />
              <site name="goal_orientation" type="sphere" size="0.001" pos="0 0 {goal_z:.6f}" quat="{' '.join(f'{value:.7f}' for value in goal_orientation)}" rgba="0 0 0 0" group="0" />"""
    return groove_name, _xml_document(
        groove_name,
        geoms,
        bottom_z=0.0,
        top_z=top_z,
        horizontal_site=(floor_x, floor_y),
    )


def _xml_document(
    groove_name: str,
    geoms: str,
    *,
    bottom_z: float,
    top_z: float,
    horizontal_site: tuple[float, float],
) -> str:
    return f"""<mujoco model="{groove_name}">
  <worldbody>
    <body>
      <body name="object" pos="0 0 0" quat="1 0 0 0">
        <body name="{groove_name}" pos="0 0 0" quat="1 0 0 0">
          <inertial pos="0 0 0" mass="0.5" diaginertia="1e-3 1e-3 1e-3" />
{geoms}
        </body>
      </body>
      <site rgba="0 0 0 0" size="0.01" pos="0 0 {bottom_z:.6f}" quat="1 0 0 0" name="bottom_site" />
      <site rgba="0 0 0 0" size="0.01" pos="0 0 {top_z:.6f}" quat="1 0 0 0" name="top_site" />
      <site rgba="0 0 0 0" size="0.01" pos="{horizontal_site[0]:.6f} {horizontal_site[1]:.6f} 0" quat="1 0 0 0" name="horizontal_radius_site" />
    </body>
  </worldbody>
</mujoco>
"""


def generate(
    obj_name: str,
    orientation: str,
    groove_type: str,
    base_dir: Path,
    *,
    write: bool,
) -> Path:
    source_xml, minimum, maximum = object_bounds(obj_name, orientation, base_dir)
    upright_xml = _source_xml(obj_name, "upright", base_dir)
    goal_orientation = canonical_placement_quat(upright_xml, source_xml)
    if groove_type == "normal":
        groove_name, content = _normal_groove(
            obj_name,
            orientation,
            minimum,
            maximum,
            goal_orientation,
        )
    else:
        groove_name, content = _v_groove(
            obj_name,
            orientation,
            minimum,
            maximum,
            goal_orientation,
        )

    output_path = base_dir / groove_name / f"{groove_name}.xml"
    source_label = str(source_xml) if source_xml is not None else "custom extents"
    print(f"{obj_name}: {source_label}")
    print(f"  collision min: {' '.join(f'{value:.6f}' for value in minimum)}")
    print(f"  collision max: {' '.join(f'{value:.6f}' for value in maximum)}")

    existing = output_path.read_text(encoding="utf-8") if output_path.is_file() else None
    if existing == content:
        print(f"  unchanged: {output_path}")
    elif write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        print(f"  generated: {output_path}")
    else:
        print(f"  [check] would generate: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Object base name; repeat for multiple targets (default: legacy toy set).",
    )
    parser.add_argument(
        "--orientation",
        choices=("upright", "flat", "flat_x", "flat_y"),
        default="upright",
    )
    parser.add_argument(
        "--groove-type", choices=("normal", "v_groove"), default="v_groove"
    )
    parser.add_argument("--base-dir", type=Path, default=OBJECTS_DIR)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write generated XML. Without this flag the command is read-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = tuple(args.targets) if args.targets else DEFAULT_TARGETS
    base_dir = args.base_dir.resolve()
    for target in targets:
        generate(
            target,
            args.orientation,
            args.groove_type,
            base_dir,
            write=args.write,
        )


if __name__ == "__main__":
    main()
