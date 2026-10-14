import random, math
import os
from dataclasses import dataclass
import numpy as np
import xml.etree.ElementTree as ET

from libero.libero.benchmark.industry_benchmark_config import sample_scene


# S2D uses the six placement sites on grey_plate rather than the plate mesh as
# the task footprint.  The farthest placement-site centre is at (+/- .095,
# +/- .095), so this is the yaw-invariant circumradius of every possible
# spawned object centre.  A circle, instead of a yaw-specific AABB, is what
# makes the bounds proof valid for every continuous yaw.
S2D_BOUNDS = {
    "x_min": -0.41,
    "x_max": 0.20,
    "y_min": -0.27,
    "y_max": 0.32,
}
S2D_GEOMETRIC_CENTER = (-0.105, 0.0)
S2D_TRAY_SPAWN_OFFSETS = (
    (-0.095, -0.095), (0.0, -0.095), (0.095, -0.095),
    (-0.095, 0.095), (0.0, 0.095), (0.095, 0.095),
)
S2D_TRAY_SPAWN_RADIUS = max(math.hypot(x, y) for x, y in S2D_TRAY_SPAWN_OFFSETS)
# conveyor_ghost is an anchor/site-only fixture.  Its moving inlays are
# validated separately: requiring the whole 1.2 m virtual track to fit in the
# training rectangle is neither necessary nor physically possible.
S2D_GHOST_FIXTURE_RADIUS = 0.0
S2D_MAX_PERTURBATION = 0.05
# ``conveyor_belt.xml`` exposes this small transport-centre site. It is a
# reachability reference, not a placement or spatial-boundary envelope: source
# objects may initialize anywhere along the long conveyor surface.
CONVEYOR_CENTRE_REGION_HALF_EXTENTS = (0.04, 0.06)
CONVEYOR_CENTRE_REGION_RADIUS = math.hypot(*CONVEYOR_CENTRE_REGION_HALF_EXTENTS)
LAYOUT_CLEARANCE = 0.01
# S2D applies the sampled XY perturbation to the plate and ghost together, so
# it cannot reduce their relative collision clearance.  Independent yaw
# perturbations are handled exactly by the projected OBB support below.
S2D_TRACK_RELATIVE_SHIFT_RESERVE = 0.0
S2D_TRACK_REQUIRED_CLEARANCE = (
    S2D_TRACK_RELATIVE_SHIFT_RESERVE + LAYOUT_CLEARANCE
)
# Four-decimal speed strata spanning 0.03--0.08 m/s exactly once. The
# midpoints between the V3/V5/V8 anchors (0.04 and 0.065) are the watersheds.
D2D_SPEED_BAND_TICKS = {
    3: (300, 399),
    5: (400, 649),
    8: (650, 800),
}


@dataclass(frozen=True)
class LayoutValidationResult:
    """A rejection-sampler result with an actionable failure reason."""

    valid: bool
    reason: str = ""

    def __bool__(self):
        return self.valid


def _rotate_xy(point, yaw_deg):
    """Rotate a local XY vector using the fixture's Z yaw."""
    yaw = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x, y = point
    return (x * cos_yaw - y * sin_yaw, x * sin_yaw + y * cos_yaw)


def _obb_support(half_extents, yaw_deg, axis):
    """Project a yawed XY box's half-extents onto one world-space axis."""
    local_x = _rotate_xy((1.0, 0.0), yaw_deg)
    local_y = _rotate_xy((0.0, 1.0), yaw_deg)
    return (
        half_extents[0] * abs(local_x[0] * axis[0] + local_x[1] * axis[1])
        + half_extents[1]
        * abs(local_y[0] * axis[0] + local_y[1] * axis[1])
    )


def _obb_separating_gap(
    first_pos,
    first_half_extents,
    first_yaw_deg,
    second_pos,
    second_half_extents,
    second_yaw_deg,
):
    """Return the largest SAT separating gap between two planar OBBs.

    A negative result is penetration.  A positive result is a conservative
    lower bound on planar clearance, measured on the best separating axis.
    """
    delta = (second_pos[0] - first_pos[0], second_pos[1] - first_pos[1])
    axes = (
        _rotate_xy((1.0, 0.0), first_yaw_deg),
        _rotate_xy((0.0, 1.0), first_yaw_deg),
        _rotate_xy((1.0, 0.0), second_yaw_deg),
        _rotate_xy((0.0, 1.0), second_yaw_deg),
    )
    return max(
        abs(delta[0] * axis[0] + delta[1] * axis[1])
        - _obb_support(first_half_extents, first_yaw_deg, axis)
        - _obb_support(second_half_extents, second_yaw_deg, axis)
        for axis in axes
    )


def _circle_inside_bounds(point, radius, bounds=S2D_BOUNDS):
    x, y = point
    return (
        bounds["x_min"] <= x - radius
        and x + radius <= bounds["x_max"]
        and bounds["y_min"] <= y - radius
        and y + radius <= bounds["y_max"]
    )


def get_safe_symmetric_base_anchors(
    source_radius,
    destination_radius,
    axis_deg=0.0,
    max_perturbation=S2D_MAX_PERTURBATION,
):
    """Return source/destination anchors with a ±5 cm clearance proof.

    ``axis_deg`` rotates the side-by-side relation for Level-3
    transpositions.  The anchors themselves are unperturbed base points; the
    caller samples the existing configured XY shift afterwards.
    """
    source_anchor, destination_anchor, separation = get_symmetric_base_anchors(
        source_radius,
        destination_radius,
        axis_deg=axis_deg,
        max_perturbation=max_perturbation,
    )

    if not _circle_inside_bounds(
        source_anchor, source_radius + max_perturbation
    ):
        raise ValueError("source fixture exceeds spatial workspace bounds")
    if not _circle_inside_bounds(
        destination_anchor, destination_radius + max_perturbation
    ):
        raise ValueError("destination fixture exceeds spatial workspace bounds")
    return source_anchor, destination_anchor, separation


def get_symmetric_base_anchors(
    source_radius,
    destination_radius,
    axis_deg=0.0,
    max_perturbation=S2D_MAX_PERTURBATION,
):
    """Return S2D-style symmetric anchors without a workspace-envelope proof."""
    separation = source_radius + destination_radius + 2.0 * max_perturbation
    axis_x, axis_y = _rotate_xy((1.0, 0.0), axis_deg)
    half_separation = separation / 2.0
    centre_x, centre_y = S2D_GEOMETRIC_CENTER
    source_anchor = (
        centre_x - half_separation * axis_x,
        centre_y - half_separation * axis_y,
    )
    destination_anchor = (
        centre_x + half_separation * axis_x,
        centre_y + half_separation * axis_y,
    )

    return source_anchor, destination_anchor, separation


def get_s2d_safe_base_anchors(
    max_perturbation=S2D_MAX_PERTURBATION,
    axis_deg=0.0,
):
    """Return symmetric S2D anchors with a yaw-independent clearance proof.

    The fixtures are separated by their circumradii plus two 5 cm
    perturbations.  The tray anchor is then checked against its six spawn
    centres, including the maximum translation perturbation.
    """
    return get_safe_symmetric_base_anchors(
        S2D_TRAY_SPAWN_RADIUS,
        S2D_GHOST_FIXTURE_RADIUS,
        axis_deg=axis_deg,
        max_perturbation=max_perturbation,
    )


def get_s2d_collision_clearance_anchors(
    plate_half_extents,
    slot_half_extents,
    plate_yaw_deg,
    ghost_yaw_deg,
    load_axis="y",
    required_clearance=S2D_TRACK_REQUIRED_CLEARANCE,
):
    """Place the plate and ghost apart on the dynamic track's normal.

    Runtime always moves slot centres on ghost-local Y. ``LoadSlot ... x``
    rotates the slot mesh by 90 degrees, while ``... y`` leaves it aligned to
    the ghost.  The cross-track collision half-width is therefore local Y for
    X loading and local X for Y loading.
    """
    if load_axis not in {"x", "y"}:
        raise ValueError("LoadSlot axis must be 'x' or 'y'")
    track_normal = _rotate_xy((1.0, 0.0), ghost_yaw_deg)
    plate_support = _obb_support(
        plate_half_extents, plate_yaw_deg, track_normal
    )
    slot_cross_track_half_extent = (
        slot_half_extents[1] if load_axis == "x" else slot_half_extents[0]
    )
    separation = (
        plate_support + slot_cross_track_half_extent + required_clearance
    )
    half_separation = separation / 2.0
    centre_x, centre_y = S2D_GEOMETRIC_CENTER
    plate_anchor = (
        centre_x - half_separation * track_normal[0],
        centre_y - half_separation * track_normal[1],
    )
    ghost_anchor = (
        centre_x + half_separation * track_normal[0],
        centre_y + half_separation * track_normal[1],
    )
    return plate_anchor, ghost_anchor, separation

def perturb_speed(base_speed, speed_noise=0.0):
    """Sample one signed speed without changing its nominal direction."""
    noise = random.uniform(-speed_noise, speed_noise) if speed_noise > 0 else 0
    return round(
        base_speed * (1 + noise) if speed_noise > 0.01 else base_speed + noise,
        4,
    )


def d2d_speed_band(base_speed):
    """Return the non-overlapping magnitude band for a D2D filename speed."""
    speed_code = int(round(abs(base_speed) * 100))
    if speed_code not in D2D_SPEED_BAND_TICKS:
        raise ValueError("D2D speed must use a 0.03, 0.05, or 0.08 anchor")
    low, high = D2D_SPEED_BAND_TICKS[speed_code]
    return low / 10000.0, high / 10000.0


def sample_d2d_speed(base_speed, rng=None):
    """Uniformly sample one signed four-decimal speed inside its watershed."""
    speed_code = int(round(abs(base_speed) * 100))
    if speed_code not in D2D_SPEED_BAND_TICKS:
        raise ValueError("D2D speed must use a 0.03, 0.05, or 0.08 anchor")
    low, high = D2D_SPEED_BAND_TICKS[speed_code]
    rng = rng or random
    tick = low + int(rng.random() * (high - low + 1))
    return math.copysign(tick / 10000.0, base_speed)


def calc_belt_timing(base_speed, num_objects, speed_noise=0.0):
    """Calculate actual speed, takt (0.75 m max span), and the 1.5 s offset."""
    actual_speed = perturb_speed(base_speed, speed_noise)

    offset = 1.5
    distance_between = 0.75 / max(1, num_objects)
    takt = round(distance_between / abs(actual_speed), 1)
    return actual_speed, takt, offset

def sample_vis(level, rng=None):
    """Sample the shared canonical visual policy for an Industry level."""
    return sample_scene(level, rng=rng)

def get_lv3_transposition(angle_deg, shift_x, shift_y, suite_type):
    """Returns suite-specific fixture anchors after their sampled translation."""
    if suite_type == "D2D":
        o_b, o_g = 0.18 + random.uniform(-0.02, 0.02), 0.18 + random.uniform(-0.02, 0.02)
        if angle_deg == 90: return (0.12, o_b + shift_y), (shift_x, -o_g + shift_y)
        if angle_deg == 180: return (0.05, shift_y), (-0.25, shift_y)
        if angle_deg == 270: return (0.12, -o_b + shift_y), (shift_x, o_g + shift_y)
        return (-0.25, shift_y), (0.05, shift_y)

    if suite_type == "D2S":
        if angle_deg == 90: return (0.1 + shift_x, 0.08 + shift_y)
        if angle_deg == 180: return (-0.35 + shift_x, shift_y)
        if angle_deg == 270: return (0.1 + shift_x, -0.08 + shift_y)
        return (0.08 + shift_x, shift_y)
        
    if suite_type == "S2D":
        # Preserve the historical yaw-independent relation for downstream
        # callers.  The canonical generator uses the geometry-aware helper.
        tray_anchor, ghost_anchor, _ = get_s2d_safe_base_anchors()
        return (
            (tray_anchor[0] + shift_x, tray_anchor[1] + shift_y),
            (ghost_anchor[0] + shift_x, ghost_anchor[1] + shift_y),
        )

def get_s2d_lv2_tray_x(X_g, D_safe, groove_yaw):
    # [USER MISTAKE MARKER]: Prevent divide-by-zero if yaw is exactly 90 deg (pi/2)
    safe_cos = max(abs(math.cos(groove_yaw)), 0.05) * (1 if math.cos(groove_yaw) >= 0 else -1)
    return X_g - (D_safe / safe_cos)

class DynamicSpatialValidator:
    def __init__(self, asset_base_dir, base_pos=(-0.56, 0.0, 0.912)):
        self.asset_base_dir = asset_base_dir
        
        # Original Libero Spatial Distribution
        self.bounds = S2D_BOUNDS.copy()
        
        # Robot Kinematics & Physical Base Pillar
        self.base_pos = np.asarray(base_pos, dtype=float)
        self.robot_base_radius = 0.12  # 12cm radius around the robot pillar to prevent clipping
        self.max_reach = 0.855
        self.table_z = 0.80 
        
        self._bbox_cache = {}
        self._collision_bbox_cache = {}
        self._slot_spacing_half_extents_cache = {}
        self._xml_path_cache = {}
        self._contact_region_yaw_offset_cache = {}
        self._ghost_track_half_length_cache = None

        # Master list of all possible inlays/grooves to measure for the Ghost
        self.ALL_INLAYS = [
            "pump_bottle_flat_groove", "pump_bottle_upright_inlay",
            "serum_bottle_flat_groove", "serum_bottle_flat_inlay",
            "cream_bottle_flat_groove", "cream_bottle_flat_inlay",
            "cream_jar_v_groove", "cream_jar_upright_inlay",
            "electronics_tray", "cosmetics_tray"
        ]

    def _get_base_name(self, xml_name):
        """Strips suffixes to locate the correct directory."""
        for suffix in ["_upright", "_flat", "_inverted", "_inlay", "_groove", "_v_groove"]:
            if suffix in xml_name:
                # Some directories just use the base name (e.g., pump_bottle)
                return xml_name.split(suffix)[0]
        return xml_name

    def _find_xml_path(self, xml_name):
        """Locate an asset by its real directory before using legacy aliases."""
        if xml_name in self._xml_path_cache:
            return self._xml_path_cache[xml_name]

        candidates = [
            os.path.join(self.asset_base_dir, xml_name, f"{xml_name}.xml"),
            os.path.join(self.asset_base_dir, "plates", f"{xml_name}.xml"),
            os.path.join(self.asset_base_dir, self._get_base_name(xml_name), f"{xml_name}.xml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                self._xml_path_cache[xml_name] = path
                return path

        # Industry assets do not all share a directory naming convention.
        for root, _, filenames in os.walk(self.asset_base_dir):
            if f"{xml_name}.xml" in filenames:
                path = os.path.join(root, f"{xml_name}.xml")
                self._xml_path_cache[xml_name] = path
                return path
        return None

    def _parse_xml_extents(self, xml_name):
        """Reads the MuJoCo <geom> tags to find the exact half-extents (hx, hy)."""
        if xml_name in self._bbox_cache:
            return self._bbox_cache[xml_name]

        # 1. The ghost fixture itself contains only marker sites and disabled
        # end-block geoms.  Its dynamic inlays are validated as track slots in
        # validate_s2d_layout(), rather than inflating this anchor footprint.
        if xml_name == "conveyor_ghost":
            self._bbox_cache[xml_name] = (0.0, 0.0)
            return self._bbox_cache[xml_name]

        # 2. STANDARD CASE: Parse the actual XML file
        xml_path = self._find_xml_path(xml_name)
        
        if xml_path is None:
            # Fallbacks for standard plates if XML paths differ slightly
            if "plate" in xml_name or "tray" in xml_name:
                return (0.125, 0.16) 
            if "belt" in xml_name:
                return (0.15, 0.065)
            raise FileNotFoundError(f"Missing asset XML: {xml_path}")

        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        max_hx, max_hy = 0.0, 0.0
        
        # Scan all collision geoms and the horizontal radius site.  Mesh
        # inlays often have no ``size`` attribute; their radius site is the
        # authoritative XY envelope used by the physical track.
        for geom in root.iter('geom'):
            size_str = geom.attrib.get('size')
            geom_type = geom.attrib.get('type', 'sphere')
            
            if size_str:
                sizes = [float(s) for s in size_str.split()]
                if geom_type == 'box' and len(sizes) >= 2:
                    hx, hy = sizes[0], sizes[1]
                elif geom_type in ['cylinder', 'sphere'] and len(sizes) >= 1:
                    hx, hy = sizes[0], sizes[0] 
                else:
                    hx, hy = 0.0, 0.0
                    
                max_hx = max(max_hx, hx)
                max_hy = max(max_hy, hy)

        for site in root.iter('site'):
            if "horizontal_radius_site" not in site.attrib.get("name", ""):
                continue
            pos = [abs(float(value)) for value in site.attrib.get("pos", "0 0 0").split()]
            if len(pos) >= 2:
                max_hx = max(max_hx, pos[0])
                max_hy = max(max_hy, pos[1])
                
        final_extents = (max_hx + 0.015, max_hy + 0.015) # 1.5cm physical safety buffer
        self._bbox_cache[xml_name] = final_extents
        return final_extents

    @staticmethod
    def _planar_quat_yaw_deg(attributes):
        """Return one XML element's planar quaternion yaw in degrees."""
        quat = attributes.get("quat")
        if quat is None:
            return 0.0
        w, x, y, z = (float(value) for value in quat.split())
        if not math.isclose(x, 0.0, abs_tol=1e-9) or not math.isclose(
            y, 0.0, abs_tol=1e-9
        ):
            raise ValueError("S2D collision geometry has a non-planar rotation")
        return math.degrees(2.0 * math.atan2(z, w))

    def collision_half_extents(self, xml_name):
        """Return the actual asset-local XY collision envelope.

        Unlike ``_parse_xml_extents()``, this reads referenced OBJ vertices
        and inline mesh vertices and does not add the legacy 15 mm buffer.
        This matters for generated ``dyn_*`` slots: their collision hull can
        extend slightly beyond ``horizontal_radius_site`` metadata.
        """
        if xml_name in self._collision_bbox_cache:
            return self._collision_bbox_cache[xml_name]

        xml_path = self._find_xml_path(xml_name)
        if xml_path is None:
            raise FileNotFoundError("Missing collision asset XML: {}".format(xml_name))
        root = ET.parse(xml_path).getroot()
        parents = {child: parent for parent in root.iter() for child in parent}

        default_geoms = {}
        for default in root.iter("default"):
            class_name = default.attrib.get("class")
            geom = default.find("geom")
            if class_name and geom is not None:
                default_geoms[class_name] = dict(geom.attrib)

        mesh_elements = {
            mesh.attrib["name"]: mesh
            for mesh in root.findall("./asset/mesh")
            if mesh.attrib.get("name")
        }
        mesh_vertices = {}

        def load_mesh_vertices(mesh_name):
            if mesh_name in mesh_vertices:
                return mesh_vertices[mesh_name]
            mesh = mesh_elements.get(mesh_name)
            if mesh is None:
                raise ValueError(
                    "Collision geom references missing mesh '{}' in {}".format(
                        mesh_name, xml_path
                    )
                )
            values = []
            if mesh.attrib.get("vertex"):
                raw = [float(value) for value in mesh.attrib["vertex"].split()]
                values = [tuple(raw[index:index + 3]) for index in range(0, len(raw), 3)]
            elif mesh.attrib.get("file"):
                mesh_path = os.path.join(
                    os.path.dirname(xml_path), mesh.attrib["file"]
                )
                with open(mesh_path, "r", encoding="utf-8", errors="ignore") as stream:
                    for line in stream:
                        fields = line.strip().split()
                        if fields and fields[0] == "v" and len(fields) >= 4:
                            values.append(tuple(float(value) for value in fields[1:4]))
            scale = [float(value) for value in mesh.attrib.get("scale", "1 1 1").split()]
            if len(scale) == 1:
                scale *= 3
            if len(scale) != 3:
                raise ValueError("Invalid mesh scale in {}".format(xml_path))
            scaled = [
                (vertex[0] * scale[0], vertex[1] * scale[1])
                for vertex in values
            ]
            mesh_vertices[mesh_name] = scaled
            return scaled

        def effective_geom_attributes(geom):
            attributes = dict(default_geoms.get(geom.attrib.get("class"), {}))
            attributes.update(geom.attrib)
            return attributes

        def element_pose(geom, attributes):
            chain = []
            node = geom
            while node is not None:
                if node.tag == "body":
                    chain.append((node.attrib, node.tag))
                node = parents.get(node)
            chain.reverse()
            chain.append((attributes, "geom"))
            position = (0.0, 0.0)
            yaw_deg = 0.0
            for element_attributes, _ in chain:
                raw_pos = [
                    float(value)
                    for value in element_attributes.get("pos", "0 0 0").split()
                ]
                local_pos = (raw_pos[0], raw_pos[1])
                rotated_pos = _rotate_xy(local_pos, yaw_deg)
                position = (
                    position[0] + rotated_pos[0],
                    position[1] + rotated_pos[1],
                )
                yaw_deg += self._planar_quat_yaw_deg(element_attributes)
            return position, yaw_deg

        max_abs_x, max_abs_y = 0.0, 0.0
        for geom in root.iter("geom"):
            node = parents.get(geom)
            has_body_ancestor = False
            while node is not None:
                if node.tag == "body":
                    has_body_ancestor = True
                    break
                node = parents.get(node)
            if not has_body_ancestor:
                continue
            attributes = effective_geom_attributes(geom)
            if (
                int(attributes.get("contype", "1")) == 0
                and int(attributes.get("conaffinity", "1")) == 0
            ):
                continue
            position, yaw_deg = element_pose(geom, attributes)
            geom_type = attributes.get(
                "type", "mesh" if attributes.get("mesh") else "sphere"
            )
            local_points = []
            if geom_type == "mesh" and attributes.get("mesh"):
                local_points = load_mesh_vertices(attributes["mesh"])
            elif attributes.get("size"):
                sizes = [float(value) for value in attributes["size"].split()]
                if geom_type == "box" and len(sizes) >= 2:
                    local_points = [
                        (x, y)
                        for x in (-sizes[0], sizes[0])
                        for y in (-sizes[1], sizes[1])
                    ]
                elif geom_type in {"sphere", "cylinder", "capsule"} and sizes:
                    radius = sizes[0]
                    local_points = [
                        (-radius, -radius),
                        (-radius, radius),
                        (radius, -radius),
                        (radius, radius),
                    ]
            for local_point in local_points:
                rotated_point = _rotate_xy(local_point, yaw_deg)
                world_point = (
                    position[0] + rotated_point[0],
                    position[1] + rotated_point[1],
                )
                max_abs_x = max(max_abs_x, abs(world_point[0]))
                max_abs_y = max(max_abs_y, abs(world_point[1]))

        if max_abs_x <= 0.0 or max_abs_y <= 0.0:
            raise ValueError("No planar collision geometry found in {}".format(xml_path))
        self._collision_bbox_cache[xml_name] = (max_abs_x, max_abs_y)
        return self._collision_bbox_cache[xml_name]

    def _ghost_track_half_length(self):
        """Read the finite ghost-local Y half-length used by LoadSlot."""
        if self._ghost_track_half_length_cache is not None:
            return self._ghost_track_half_length_cache
        ghost_path = self._find_xml_path("conveyor_ghost")
        if ghost_path is None:
            raise FileNotFoundError("Missing conveyor_ghost asset XML")
        root = ET.parse(ghost_path).getroot()
        active_region = next(
            (
                site
                for site in root.iter("site")
                if site.attrib.get("name") == "ghost_active_region"
            ),
            None,
        )
        if active_region is None or not active_region.attrib.get("size"):
            raise ValueError("conveyor_ghost has no sized ghost_active_region")
        sizes = [float(value) for value in active_region.attrib["size"].split()]
        self._ghost_track_half_length_cache = sizes[1]
        return self._ghost_track_half_length_cache

    def _load_slot_spacing_half_extents(self, inlay_name):
        """Mirror LoadSlot's horizontal-radius spacing dimensions."""
        if inlay_name in self._slot_spacing_half_extents_cache:
            return self._slot_spacing_half_extents_cache[inlay_name]
        inlay_path = self._find_xml_path(inlay_name)
        if inlay_path is None:
            raise FileNotFoundError(
                "Missing dynamic slot asset XML: {}".format(inlay_name)
            )
        root = ET.parse(inlay_path).getroot()
        radius_site = next(
            (
                site
                for site in root.iter("site")
                if "horizontal_radius_site" in site.attrib.get("name", "")
            ),
            None,
        )
        if radius_site is None or not radius_site.attrib.get("pos"):
            raise ValueError(
                "Dynamic slot has no horizontal_radius_site: {}".format(
                    inlay_name
                )
            )
        values = [
            abs(float(value)) for value in radius_site.attrib["pos"].split()
        ]
        if len(values) < 2:
            raise ValueError(
                "Dynamic slot has invalid horizontal_radius_site: {}".format(
                    inlay_name
                )
            )
        half_x, half_y = values[:2]
        if half_x <= 0.0 or half_y <= 0.0:
            radius = max(values)
            half_x = half_y = radius
        self._slot_spacing_half_extents_cache[inlay_name] = (half_x, half_y)
        return self._slot_spacing_half_extents_cache[inlay_name]

    def dynamic_slot_loop_half_length(self, inlay_name, load_axis="y"):
        """Return the generated dyn-slot centre loop half-length.

        Runtime packs an integer number of metadata-sized slots into the ghost
        active region, then replaces the nominal endpoints with the resulting
        perfect loop length.  X loading rotates the mesh by 90 degrees, so its
        local X dimension becomes the along-track spacing dimension.
        """
        if load_axis not in {"x", "y"}:
            raise ValueError("LoadSlot axis must be 'x' or 'y'")
        half_x, half_y = self._load_slot_spacing_half_extents(inlay_name)
        along_half_extent = half_x if load_axis == "x" else half_y
        groove_step = 2.0 * along_half_extent + 0.01
        nominal_length = 2.0 * self._ghost_track_half_length()
        num_slots = int(nominal_length / groove_step)
        if num_slots < 2:
            raise ValueError(
                "Dynamic slot track generates fewer than two slots for {}".format(
                    inlay_name
                )
            )
        return num_slots * groove_step / 2.0

    def _contact_region_yaw_offset(self, xml_name):
        """Return the XML-local Z yaw of a conveyor contact region.

        Fixture placement yaw is applied to the root body, while the physical
        conveyor contact geom can live beneath one or more yawed child bodies.
        The base-clearance OBB must therefore use their composed orientation,
        not the BDDL placement yaw alone. Normalized conveyor assets have no
        child-body yaw, while this composition remains compatible with older
        fixture XML variants.
        """
        if xml_name in self._contact_region_yaw_offset_cache:
            return self._contact_region_yaw_offset_cache[xml_name]

        xml_path = self._find_xml_path(xml_name)
        if xml_path is None:
            raise FileNotFoundError("Missing fixture XML for {}".format(xml_name))
        root = ET.parse(xml_path).getroot()
        contact_region = next(
            (
                geom
                for geom in root.iter("geom")
                if geom.attrib.get("name") == "contact_region"
            ),
            None,
        )
        if contact_region is None:
            # Only conveyor belts use the physical-rail base check today.
            # Retain a neutral offset for assets without this named geom.
            self._contact_region_yaw_offset_cache[xml_name] = 0.0
            return 0.0

        parents = {child: parent for parent in root.iter() for child in parent}
        chain = []
        node = contact_region
        while node is not None:
            chain.append(node)
            node = parents.get(node)

        yaw_offset = 0.0
        for node in reversed(chain):
            quat = node.attrib.get("quat")
            if quat is None:
                continue
            w, x, y, z = (float(value) for value in quat.split())
            if not math.isclose(x, 0.0, abs_tol=1e-9) or not math.isclose(
                y, 0.0, abs_tol=1e-9
            ):
                raise ValueError(
                    "contact_region in {} has a non-planar XML rotation".format(
                        xml_name
                    )
                )
            yaw_offset += math.degrees(2.0 * math.atan2(z, w))

        self._contact_region_yaw_offset_cache[xml_name] = yaw_offset
        return yaw_offset

    def circumradius(self, xml_name):
        """Return a yaw-invariant XY envelope radius for an asset."""
        if xml_name == "conveyor_belt":
            return CONVEYOR_CENTRE_REGION_RADIUS
        if xml_name == "grey_plate":
            # The physical plate exceeds the narrow training rectangle when
            # represented as a full mesh circle.  Its task-relevant footprint
            # is the six object-CoM spawn sites, as in S2D.
            return S2D_TRAY_SPAWN_RADIUS
        half_x, half_y = self._parse_xml_extents(xml_name)
        return math.hypot(half_x, half_y)

    def _fixture_radius(self, fixture):
        if "safety_radius" in fixture:
            return fixture["safety_radius"]
        return self.circumradius(fixture["xml_name"])

    def d2d_lv3_cross_track_half_widths(self):
        """Measure the belt and ghost-region widths used by D2S layout checks."""
        belt_path = self._find_xml_path("conveyor_belt")
        ghost_path = self._find_xml_path("conveyor_ghost")
        if belt_path is None or ghost_path is None:
            raise FileNotFoundError("conveyor asset XML is missing")

        belt_root = ET.parse(belt_path).getroot()
        ghost_root = ET.parse(ghost_path).getroot()
        belt_width = next(
            abs(float(site.attrib["pos"].split()[0]))
            for site in belt_root.iter("site")
            if site.attrib.get("name") == "horizontal_radius_site"
        )
        slot_width = next(
            float(site.attrib["size"].split()[0])
            for site in ghost_root.iter("site")
            if site.attrib.get("name") == "ghost_active_region"
        )
        return belt_width, slot_width

    def is_in_distribution(self, x, y):
        in_x = self.bounds["x_min"] <= x <= self.bounds["x_max"]
        in_y = self.bounds["y_min"] <= y <= self.bounds["y_max"]
        return in_x and in_y

    def is_kinematically_safe(self, x, y, buffer_m=0.03):
        target_pos = np.array([x, y, self.table_z])
        dist = np.linalg.norm(target_pos - self.base_pos)
        return dist <= (self.max_reach - buffer_m)

    def _circle_is_in_distribution(self, x, y, radius):
        """Check an XY circumcircle against the rectangular training bounds."""
        return (
            self.bounds["x_min"] <= x - radius
            and x + radius <= self.bounds["x_max"]
            and self.bounds["y_min"] <= y - radius
            and y + radius <= self.bounds["y_max"]
        )

    def _circle_is_kinematically_safe(self, x, y, radius=0.0, buffer_m=0.03):
        target_pos = np.array([x, y, self.table_z])
        dist = np.linalg.norm(target_pos - self.base_pos)
        return dist + radius <= (self.max_reach - buffer_m)

    def validate_conveyor_centre_region(self, belt):
        """Check reachability at the conveyor centre without bounding its rail.

        The BDDL anchor is deliberately used as the centre-region proxy.  The
        XML site's local 2 cm X offset is retained by the asset/runtime and is
        not folded back into generated coordinates.
        """
        if not self._circle_is_kinematically_safe(*belt["pos"]):
            return LayoutValidationResult(
                False, "conveyor centre_region exceeds kinematic workspace bounds"
            )
        return LayoutValidationResult(True)

    def validate_conveyor_base_clearance(self, belt):
        """Reject a yawed physical conveyor envelope that overlaps the robot base.

        The conveyor rail is intentionally exempt from the bounded task
        rectangle, but its physical MuJoCo footprint must still clear the
        robot-base pillar.  Use the asset's measured XY half-extents and an
        exact point-to-oriented-rectangle distance to the circular base
        keep-out; checking only the centre region would miss a transverse
        rail that crosses the base.
        """
        half_x, half_y = self._parse_xml_extents(belt["xml_name"])
        physical_yaw_deg = belt["yaw_deg"] + self._contact_region_yaw_offset(
            belt["xml_name"]
        )
        yaw = math.radians(physical_yaw_deg)
        dx = self.base_pos[0] - belt["pos"][0]
        dy = self.base_pos[1] - belt["pos"][1]
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        edge_x = max(abs(local_x) - half_x, 0.0)
        edge_y = max(abs(local_y) - half_y, 0.0)
        base_distance = math.hypot(edge_x, edge_y)
        required_clearance = self.robot_base_radius + LAYOUT_CLEARANCE
        if base_distance < required_clearance:
            return LayoutValidationResult(
                False,
                "conveyor physical envelope overlaps robot-base keep-out "
                "({:.3f} m < {:.3f} m)".format(
                    base_distance, required_clearance
                ),
            )
        return LayoutValidationResult(True)

    def negative_x_translation_to_safe_region(
        self,
        kinematic_circles,
        spatial_circles=(),
    ):
        """Return the smallest common non-positive X shift that is safe.

        Kinematic items are ``(name, (x, y), radius)``. Spatial items are
        ``(name, (x, y), half_x, half_y)``. Kinematic circles must stay within
        the robot-reach semicircle; spatial envelopes must stay in the
        rectangular task bounds. ``None`` means that no whole-layout shift
        along negative X can repair the candidate.
        """
        lower, upper = -math.inf, 0.0
        safe_reach = self.max_reach - 0.03
        base_x, base_y = self.base_pos[:2]

        for name, (x_pos, y_pos), radius in kinematic_circles:
            available = safe_reach - radius
            dy = y_pos - base_y
            remaining_sq = available * available - dy * dy
            if remaining_sq < 0.0:
                return None
            horizontal = math.sqrt(remaining_sq)
            lower = max(lower, base_x - horizontal - x_pos)
            upper = min(upper, base_x + horizontal - x_pos)

        for name, (x_pos, y_pos), half_x, half_y in spatial_circles:
            if (
                y_pos - half_y < self.bounds["y_min"]
                or y_pos + half_y > self.bounds["y_max"]
            ):
                return None
            lower = max(lower, self.bounds["x_min"] + half_x - x_pos)
            upper = min(upper, self.bounds["x_max"] - half_x - x_pos)

        if lower > upper + 1e-9:
            return None
        return min(0.0, upper)

    def _validate_fixture_circles(self, fixtures):
        """Validate physical fixture envelopes and return a reasoned result."""
        for name, fixture in fixtures.items():
            radius = self._fixture_radius(fixture)
            position = fixture["pos"]
            if not fixture.get("skip_spatial_bounds", False) and not self._circle_is_in_distribution(*position, radius):
                return LayoutValidationResult(
                    False,
                    "{} exceeds spatial workspace bounds".format(name),
                )
            if not fixture.get("skip_kinematic_bounds", False) and not self._circle_is_kinematically_safe(*position, radius):
                return LayoutValidationResult(
                    False,
                    "{} exceeds kinematic workspace bounds".format(name),
                )

        items = list(fixtures.items())
        for index, (first_name, first) in enumerate(items):
            first_radius = self._fixture_radius(first)
            for second_name, second in items[index + 1 :]:
                second_radius = self._fixture_radius(second)
                if math.dist(first["pos"], second["pos"]) < first_radius + second_radius + LAYOUT_CLEARANCE:
                    return LayoutValidationResult(
                        False,
                        "{} and {} collide".format(first_name, second_name),
                    )
        return LayoutValidationResult(True)

    def dynamic_slot_track_half_extents(self, inlay_name, load_axis="y"):
        """Return one dyn-slot's (cross-track, along-track) half-extents."""
        if load_axis not in {"x", "y"}:
            raise ValueError("LoadSlot axis must be 'x' or 'y'")
        half_x, half_y = self.collision_half_extents(inlay_name)
        # Runtime rotates an X-loaded mesh 90 degrees relative to the ghost.
        return (half_y, half_x) if load_axis == "x" else (half_x, half_y)

    def d2d_required_track_separation(
        self, inlay_name, load_axis="y", required_clearance=LAYOUT_CLEARANCE
    ):
        """Measure the belt and selected dynamic groove's cross-track width."""
        belt_cross_half = _obb_support(
            self.collision_half_extents("conveyor_belt"),
            self._contact_region_yaw_offset("conveyor_belt"),
            (1.0, 0.0),
        )
        slot_cross_half, _ = self.dynamic_slot_track_half_extents(
            inlay_name, load_axis=load_axis
        )
        return belt_cross_half + slot_cross_half + required_clearance

    def d2d_track_clearance(self, layout_dict, inlay_name, load_axis="y"):
        """Return SAT clearance from the belt to the selected moving groove track."""
        belt = layout_dict["belt"]
        ghost = layout_dict["ghost"]
        slot_cross_half, slot_along_half = self.dynamic_slot_track_half_extents(
            inlay_name, load_axis=load_axis
        )
        swept_track_half_extents = (
            slot_cross_half,
            self.dynamic_slot_loop_half_length(inlay_name, load_axis=load_axis)
            + slot_along_half,
        )
        return _obb_separating_gap(
            belt["pos"],
            self.collision_half_extents(belt["xml_name"]),
            belt["yaw_deg"] + self._contact_region_yaw_offset(belt["xml_name"]),
            ghost["pos"],
            swept_track_half_extents,
            ghost["yaw_deg"],
        )

    def s2d_plate_track_clearance(
        self, layout_dict, inlay_name, load_axis="y"
    ):
        """Return SAT clearance from grey_plate to the finite swept slot OBB."""
        plate = layout_dict["plate"]
        ghost = layout_dict["ghost"]
        slot_cross_half, slot_along_half = self.dynamic_slot_track_half_extents(
            inlay_name, load_axis=load_axis
        )
        swept_track_half_extents = (
            slot_cross_half,
            self.dynamic_slot_loop_half_length(
                inlay_name, load_axis=load_axis
            )
            + slot_along_half,
        )
        return _obb_separating_gap(
            plate["pos"],
            self.collision_half_extents(plate["xml_name"]),
            plate["yaw_deg"],
            ghost["pos"],
            swept_track_half_extents,
            ghost["yaw_deg"],
        )

    def _reachable_track_slots(self, ghost_pos, yaw_deg, inlay_name, load_axis="y"):
        """Return the two central slot target positions on the virtual ghost track.

        The complete ghost track is intentionally longer than the training
        rectangle.  S2D only needs the two slots nearest the transfer point to
        be reachable, so those are the relevant dynamic envelopes to validate.
        """
        if load_axis not in {"x", "y"}:
            raise ValueError("LoadSlot axis must be 'x' or 'y'")
        half_x, half_y = self._parse_xml_extents(inlay_name)
        # Keep the legacy reachable-slot envelope used by the seeded layout
        # gate. The swept collision OBB above uses the exact runtime loop.
        physical_half_extent = max(
            0.001, (half_x if load_axis == "x" else half_y) - 0.015
        )
        groove_step = 2.0 * physical_half_extent + 0.01
        num_slots = max(2, int(1.2 / groove_step))
        local_ys = [(-num_slots * groove_step / 2.0) + i * groove_step for i in range(num_slots)]
        nearest_ys = sorted(local_ys, key=abs)[:2]

        yaw_rad = math.radians(yaw_deg)
        sin_yaw, cos_yaw = math.sin(yaw_rad), math.cos(yaw_rad)
        return [
            (
                ghost_pos[0] - local_y * sin_yaw,
                ghost_pos[1] + local_y * cos_yaw,
            )
            for local_y in nearest_ys
        ]

    def s2d_safety_circles(self, layout_dict, inlay_name, load_axis="y"):
        """Return the exact S2D CoM/slot checks for a common X translation."""
        tray = layout_dict["plate"]
        ghost = layout_dict["ghost"]
        yaw_rad = math.radians(tray["yaw_deg"])
        sine, cosine = math.sin(yaw_rad), math.cos(yaw_rad)
        spawn_coms = []
        for index, (local_x, local_y) in enumerate(S2D_TRAY_SPAWN_OFFSETS):
            spawn_coms.append(
                (
                    "tray object CoM {}".format(index + 1),
                    (
                        tray["pos"][0] + local_x * cosine - local_y * sine,
                        tray["pos"][1] + local_x * sine + local_y * cosine,
                    ),
                    0.0,
                )
            )
        slot_radius = self.circumradius(inlay_name)
        slots = [
            ("dynamic groove {}".format(index + 1), position, slot_radius)
            for index, position in enumerate(
                self._reachable_track_slots(
                    ghost["pos"],
                    ghost["yaw_deg"],
                    inlay_name,
                    load_axis=load_axis,
                )
            )
        ]
        tray_radius = tray.get("safety_radius", S2D_TRAY_SPAWN_RADIUS)
        spatial = [("tray spawn envelope", tray["pos"], tray_radius, tray_radius)] + [
            (name, position, radius, radius) for name, position, radius in slots
        ]
        return spawn_coms + slots, spatial

    # Kept for existing S2D callers and downstream notebooks.
    _s2d_reachable_track_slots = _reachable_track_slots

    def _validate_dynamic_slots(
        self,
        ghost_pos,
        ghost_yaw_deg,
        inlay_name,
        minimum_slots=2,
        slot_radius=None,
        load_axis="y",
    ):
        try:
            if slot_radius is None:
                slot_radius = self.circumradius(inlay_name)
        except FileNotFoundError:
            return LayoutValidationResult(False, "missing dynamic slot geometry for {}".format(inlay_name))
        reachable_slots = self._reachable_track_slots(
            ghost_pos, ghost_yaw_deg, inlay_name, load_axis=load_axis
        )
        usable_slots = [
            slot for slot in reachable_slots
            if self._circle_is_in_distribution(*slot, slot_radius)
            and self._circle_is_kinematically_safe(*slot, slot_radius)
        ]
        if len(usable_slots) < minimum_slots:
            return LayoutValidationResult(
                False,
                "only {} dynamic slots are reachable; require {}".format(
                    len(usable_slots), minimum_slots
                ),
            )
        return LayoutValidationResult(True)

    def validate_d2d_layout(
        self,
        layout_dict,
        dynamic_inlay_name,
        slot_radius=None,
        load_axis="y",
    ):
        """Validate D2D's reachable centre and two dynamic ghost slots.

        Belt-source objects intentionally initialize on the long transport
        surface and may start outside the training rectangle. They are not
        spatial layout targets and are therefore excluded from this validator.
        """
        belt = layout_dict["belt"]
        ghost = layout_dict["ghost"]
        fixture_result = self._validate_fixture_circles({"belt": belt, "ghost": ghost})
        if not fixture_result:
            return fixture_result
        base_clearance_result = self.validate_conveyor_base_clearance(belt)
        if not base_clearance_result:
            return base_clearance_result
        centre_result = self.validate_conveyor_centre_region(belt)
        if not centre_result:
            return centre_result
        try:
            track_clearance = self.d2d_track_clearance(
                layout_dict, dynamic_inlay_name, load_axis=load_axis
            )
        except (FileNotFoundError, ValueError) as error:
            return LayoutValidationResult(False, str(error))
        if track_clearance + 1e-9 < LAYOUT_CLEARANCE:
            return LayoutValidationResult(
                False,
                "conveyor belt and dynamic groove track violate collision "
                "clearance ({:.3f} m < {:.3f} m)".format(
                    track_clearance, LAYOUT_CLEARANCE
                ),
            )
        return self._validate_dynamic_slots(
            ghost["pos"],
            ghost["yaw_deg"],
            dynamic_inlay_name,
            minimum_slots=2,
            slot_radius=slot_radius,
            load_axis=load_axis,
        )

    def validate_d2s_layout(self, layout_dict):
        """Validate D2S's reachable belt centre and static destination fixtures."""
        belt = layout_dict["belt"]
        centre_result = self.validate_conveyor_centre_region(belt)
        if not centre_result:
            return centre_result
        fixtures = {"belt": belt}
        fixtures.update(layout_dict["static_fixtures"])
        return self._validate_fixture_circles(fixtures)

    def validate_s2d_layout(
        self,
        layout_dict,
        inlay_name,
        load_axis="y",
        required_track_clearance=None,
    ):
        """Validate S2D's physical track clearance and usable dynamic slots.

        When ``required_track_clearance`` is supplied, the full grey-plate OBB
        must clear the finite swept dynamic-slot OBB by that distance.  The
        default retains the legacy spawn-envelope check for existing callers.
        Inactive individual slots may still extend beyond the training area.
        """
        tray = layout_dict["plate"]
        ghost = layout_dict["ghost"]
        tray_radius = tray.get("safety_radius", S2D_TRAY_SPAWN_RADIUS)
        ghost_radius = ghost.get("safety_radius", S2D_GHOST_FIXTURE_RADIUS)

        if required_track_clearance is None:
            # Level 1's moving tray is vertically clear of the plate. Preserve
            # its existing spawn-envelope separation without imposing an XY
            # swept-track exclusion that its kinematic bounds cannot satisfy.
            minimum_separation = (
                tray_radius
                + ghost_radius
                + 2.0 * S2D_MAX_PERTURBATION
            )
            if math.dist(tray["pos"], ghost["pos"]) + 1e-9 < minimum_separation:
                return LayoutValidationResult(
                    False, "tray and ghost violate fixture separation"
                )
        else:
            try:
                track_clearance = self.s2d_plate_track_clearance(
                    layout_dict, inlay_name, load_axis=load_axis
                )
            except (FileNotFoundError, ValueError) as error:
                return LayoutValidationResult(False, str(error))
            if track_clearance + 1e-9 < required_track_clearance:
                return LayoutValidationResult(
                    False,
                    "grey plate and dynamic slot track violate collision clearance "
                    "({:.3f} m < {:.3f} m)".format(
                        track_clearance, required_track_clearance
                    ),
                )
        if not self._circle_is_in_distribution(*tray["pos"], tray_radius):
            return LayoutValidationResult(False, "tray spawn envelope exceeds spatial workspace bounds")
        if not self._circle_is_in_distribution(*ghost["pos"], ghost_radius):
            return LayoutValidationResult(False, "ghost anchor exceeds spatial workspace bounds")

        kinematic_circles, _ = self.s2d_safety_circles(
            layout_dict, inlay_name, load_axis=load_axis
        )
        for name, position, radius in kinematic_circles:
            if not self._circle_is_kinematically_safe(*position, radius):
                return LayoutValidationResult(
                    False, "{} exceeds kinematic workspace bounds".format(name)
                )

        # Reachability is a target-site property. A ghost can legitimately
        # carry oversized inactive mesh slots outside the training rectangle;
        # only two dynamic slot *envelopes* need to be accessible for transfer.
        return self._validate_dynamic_slots(
            ghost["pos"],
            ghost["yaw_deg"],
            inlay_name,
            minimum_slots=2,
            load_axis=load_axis,
        )

    def is_collision_free(self, layout_dict):
        items = list(layout_dict.items())
        
        for i in range(len(items)):
            id1, data1 = items[i]
            hx1, hy1 = self._parse_xml_extents(data1["xml_name"])
            pos1 = data1["pos"]
            
            # Swap Dimensions for 90/270 degree rotations (Y-Axis Parallel Conveyors)
            yaw1_mod = data1["yaw_deg"] % 180
            if 85 < yaw1_mod < 95:
                hx1, hy1 = hy1, hx1
                
            # 1. ROBOT BASE COLLISION CHECK (AABB against robot0_base)
            # Compares the fixture's half-extents against the robot base's half-extents
            overlap_base_x = abs(pos1[0] - self.base_pos[0]) < (hx1 + self.robot_base_radius)
            overlap_base_y = abs(pos1[1] - self.base_pos[1]) < (hy1 + self.robot_base_radius)
            if overlap_base_x and overlap_base_y:
                print(f"      [!] REJECTED: '{data1['xml_name']}' at (X={pos1[0]:.3f}, Y={pos1[1]:.3f}) clipped the robot base! (Requires > {self.robot_base_radius}m clearance)")
                return False # Fixture hits the robot base!

            # 2. FIXTURE-TO-FIXTURE COLLISION CHECK
            for j in range(i + 1, len(items)):
                id2, data2 = items[j]
                hx2, hy2 = self._parse_xml_extents(data2["xml_name"])
                pos2 = data2["pos"]
                
                yaw2_mod = data2["yaw_deg"] % 180
                if 85 < yaw2_mod < 95:
                    hx2, hy2 = hy2, hx2
                    
                overlap_x = abs(pos1[0] - pos2[0]) < (hx1 + hx2)
                overlap_y = abs(pos1[1] - pos2[1]) < (hy1 + hy2)
                
                if overlap_x and overlap_y:
                    return False # Fixtures overlap each other!
                    
        return True

    def validate_layout(self, layout_dict):
        # 1. Prevent physical overlap
        if not self.is_collision_free(layout_dict):
            return False
            
        # 2. Check Reach & Bounding Boxes
        for obj_id, data in layout_dict.items():
            x, y = data["pos"]
            if not self.is_kinematically_safe(x, y):
                return False
            if not self.is_in_distribution(x, y):
                return False
                
        return True
