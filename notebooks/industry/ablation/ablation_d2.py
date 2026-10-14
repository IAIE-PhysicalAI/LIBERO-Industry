import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
SOURCE_DIR = REPO_ROOT / "libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d3"
TARGET_DIR = REPO_ROOT / "libero/libero/bddl_files/ablation_industry_flow/libero_ablation_d2"

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ASSET_XML = {
    "conveyor_belt": REPO_ROOT
    / "libero/libero/assets/industry_objects/conveyor_belt/conveyor_belt.xml",
    "grey_plate": REPO_ROOT
    / "libero/libero/assets/industry_objects/plates/grey_plate.xml",
}


def _block(lines, marker):
    for index, line in enumerate(lines):
        if line.strip() != marker:
            continue
        result = []
        depth = 0
        for candidate in lines[index:]:
            result.append(candidate)
            depth += candidate.count("(") - candidate.count(")")
            if depth == 0:
                return result
    raise ValueError("Missing BDDL block: {}".format(marker))


def _regions(lines):
    region_lines = _block(lines, "(:regions")
    result = {}
    index = 1
    while index < len(region_lines) - 1:
        stripped = region_lines[index].strip()
        if not stripped.startswith("(") or stripped.startswith("(:"):
            index += 1
            continue
        name = stripped[1:].split()[0].rstrip(")")
        entry = _block(region_lines[index:], "({}".format(name))
        result[name] = entry
        index += len(entry)
    return result


def _region_values(region_lines):
    target = None
    bounds = None
    yaw = 0.0
    mode = None
    for line in region_lines:
        stripped = line.strip()
        target_match = re.search(r"\(:target\s+(\S+)\)", stripped)
        if target_match:
            target = target_match.group(1)
        if stripped.startswith("(:ranges"):
            mode = "ranges"
            continue
        if stripped.startswith("(:yaw_rotation"):
            mode = "yaw"
            continue
        values = [float(value) for value in re.findall(FLOAT_RE, stripped)]
        if mode == "ranges" and len(values) == 4 and bounds is None:
            bounds = tuple(values)
            mode = None
        elif mode == "yaw" and len(values) == 2:
            yaw = 0.5 * (values[0] + values[1])
            mode = None
    return target, bounds, yaw


def _declarations(lines, marker):
    result = {}
    for line in _block(lines, marker):
        match = re.match(r"\s*(\S+(?:\s+\S+)*)\s+-\s+(\S+)\s*$", line)
        if not match:
            continue
        for instance in match.group(1).split():
            result[instance] = match.group(2)
    return result


def _on_target(lines, instance):
    pattern = re.compile(r"^\s*\(On\s+{}\s+(\S+)\)".format(re.escape(instance)))
    for line in _block(lines, "(:init"):
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise ValueError("Missing On predicate for {}".format(instance))


def _split_site_reference(reference, fixtures):
    for fixture in sorted(fixtures, key=len, reverse=True):
        prefix = fixture + "_"
        if reference.startswith(prefix):
            return fixture, reference[len(prefix) :]
    raise ValueError("Cannot resolve fixture/site reference: {}".format(reference))


def _site_geometry(category, site_name):
    xml_path = ASSET_XML.get(category)
    if xml_path is None:
        raise ValueError("No static-site source registered for fixture type {}".format(category))
    site = ET.parse(str(xml_path)).find(".//site[@name='{}']".format(site_name))
    if site is None:
        raise ValueError("Missing site {} in {}".format(site_name, xml_path))
    position = [float(value) for value in site.get("pos", "0 0 0").split()]
    size = [float(value) for value in site.get("size", "0 0 0").split()]
    return position[:2], size[:2]


def _world_region(lines, fixture, site_name, fixtures, regions):
    placement = _on_target(lines, fixture)
    workbench_prefix = "industry_workbench_"
    if not placement.startswith(workbench_prefix):
        raise ValueError("{} is not placed on the Industry workbench".format(fixture))
    base_name = placement[len(workbench_prefix) :]
    _, bounds, yaw = _region_values(regions[base_name])
    if bounds is None:
        raise ValueError("Region {} has no Cartesian bounds".format(base_name))

    (site_x, site_y), (site_half_x, site_half_y) = _site_geometry(
        fixtures[fixture], site_name
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    base_x = 0.5 * (bounds[0] + bounds[2])
    base_y = 0.5 * (bounds[1] + bounds[3])
    center_x = base_x + cos_yaw * site_x - sin_yaw * site_y
    center_y = base_y + sin_yaw * site_x + cos_yaw * site_y

    # The ablation environment uses uniform sampling and halves object ranges.
    # Encode the fixture jitter twice so D2 preserves the D3 world distribution.
    base_half_x = 0.5 * (bounds[2] - bounds[0])
    base_half_y = 0.5 * (bounds[3] - bounds[1])
    rotated_half_x = abs(cos_yaw) * site_half_x + abs(sin_yaw) * site_half_y
    rotated_half_y = abs(sin_yaw) * site_half_x + abs(cos_yaw) * site_half_y
    half_x = rotated_half_x + 2.0 * base_half_x
    half_y = rotated_half_y + 2.0 * base_half_y
    return (
        center_x - half_x,
        center_y - half_y,
        center_x + half_x,
        center_y + half_y,
    )


def _format_region(name, target, bounds=None, yaw=0.0):
    lines = ["      ({}\n".format(name), "          (:target {})\n".format(target)]
    if bounds is not None:
        values = " ".join("{:.12g}".format(value) for value in bounds)
        lines.extend(
            [
                "          (:ranges (\n",
                "              ({})\n".format(values),
                "            )\n",
                "          )\n",
                "          (:yaw_rotation (\n",
                "              ({0:.12g} {0:.12g})\n".format(yaw),
                "            )\n",
                "          )\n",
            ]
        )
    lines.append("      )\n")
    return lines


def convert_stage(source_path, target_path):
    source_path = Path(source_path)
    target_path = Path(target_path)
    lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    text = "".join(lines)
    regions = _regions(lines)
    fixtures = _declarations(lines, "(:fixtures")
    objects = _declarations(lines, "(:objects")

    object_lines = _block(lines, "(:obj_of_interest")
    object_names = [line.strip() for line in object_lines[1:-1] if line.strip()]
    if len(object_names) != 1:
        raise ValueError("D2 expects one object of interest, got {}".format(object_names))
    object_name = object_names[0]
    object_reference = _on_target(lines, object_name)
    source_fixture, source_site = _split_site_reference(object_reference, fixtures)
    object_bounds = _world_region(lines, source_fixture, source_site, fixtures, regions)

    language_match = re.search(r"\(:language\s+(.*?)\)", text)
    if language_match is None:
        raise ValueError("Missing language declaration")
    language = language_match.group(1).replace("moving tray", "tray").replace(" from the conveyor", "")

    goal_match = re.search(
        r"\((?:DynamicOn|ReleasedOn|Routed|On)\s+{}\s+([^\s\)]+)\)".format(
            re.escape(object_name)
        ),
        "".join(_block(lines, "(:goal")),
    )
    if goal_match is None:
        raise ValueError("Missing goal for {}".format(object_name))
    goal_target = goal_match.group(1)

    dynamic_target = goal_target == "conveyor_ghost_1"
    tray_name = None
    tray_type = None
    tray_bounds = None
    tray_yaw = 0.0
    if dynamic_target:
        load_match = re.search(r"\(LoadSlot\s+\S+\s+(\S+)", text)
        if load_match is None:
            raise ValueError("Dynamic D2 target has no LoadSlot declaration")
        tray_type = load_match.group(1)
        tray_name = tray_type + "_1"
        ghost_reference = _on_target(lines, "conveyor_ghost_1")
        ghost_region = ghost_reference[len("industry_workbench_") :]
        _, tray_bounds, tray_yaw = _region_values(regions[ghost_region])
        if tray_bounds is None:
            raise ValueError("Dynamic D2 target region has no Cartesian bounds")
        tray_center_x = 0.5 * (tray_bounds[0] + tray_bounds[2])
        tray_center_y = 0.5 * (tray_bounds[1] + tray_bounds[3])
        # D2 turns the tray fixture into an object; the ablation sampler halves
        # object ranges, so expand this range once to preserve the same jitter.
        tray_bounds = (
            tray_center_x + 2.0 * (tray_bounds[0] - tray_center_x),
            tray_center_y + 2.0 * (tray_bounds[1] - tray_center_y),
            tray_center_x + 2.0 * (tray_bounds[2] - tray_center_x),
            tray_center_y + 2.0 * (tray_bounds[3] - tray_center_y),
        )
        goal_name = tray_name
        goal_type = tray_type
    else:
        goal_name, _ = _split_site_reference(goal_target, fixtures)
        goal_type = fixtures[goal_name]

    output = [
        "(define (problem LIBERO_Industry_Workbench_Ablation)\n",
        "  (:domain robosuite)\n",
        "  (:language {})\n".format(language),
        "    (:regions\n",
    ]
    output.extend(_format_region("object_init", "industry_workbench", object_bounds, 0.0))

    target_init = None
    if dynamic_target:
        output.extend(_format_region("tray_init", "industry_workbench", tray_bounds, tray_yaw))
        output.extend(_format_region("goal_region", tray_name))
    else:
        target_reference = _on_target(lines, goal_name)
        target_init = target_reference[len("industry_workbench_") :]
        target_target, target_bounds, target_yaw = _region_values(regions[target_init])
        output.extend(_format_region(target_init, target_target, target_bounds, target_yaw))
        for region_name, region_lines in regions.items():
            region_target, _, _ = _region_values(region_lines)
            if region_target == goal_name:
                output.extend(_format_region(region_name, goal_name))
    output.extend(["    )\n\n", "  (:fixtures\n", "    industry_workbench - industry_workbench\n"])
    if not dynamic_target:
        output.append("    {} - {}\n".format(goal_name, goal_type))
    output.extend(["  )\n\n", "  (:objects\n"])
    if dynamic_target:
        output.append("    {} - {}\n".format(tray_name, tray_type))
    output.extend(
        [
            "    {} - {}\n".format(object_name, objects[object_name]),
            "  )\n\n",
            "  (:obj_of_interest\n",
            "    {}\n".format(object_name),
            "  )\n\n",
            "  (:init\n",
        ]
    )
    if dynamic_target:
        output.append("    (On {} industry_workbench_tray_init)\n".format(tray_name))
    else:
        output.append("    (On {} industry_workbench_{})\n".format(goal_name, target_init))
    output.extend(
        [
            "    (On {} industry_workbench_object_init)\n".format(object_name),
            "  )\n\n",
            "  (:goal\n",
            "    (And (ReleasedOn {} {}_goal_region))\n".format(object_name, goal_name),
            "  )\n",
            ")\n",
        ]
    )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(output))
    print("[SUCCESS] Converted to D2: {}".format(target_path.name))


if __name__ == "__main__":
    raise SystemExit("Use regenerate_ablation_flow.py with an empty --output-root.")
