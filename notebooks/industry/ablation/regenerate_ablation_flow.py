"""Regenerate D2-D6 ablation BDDLs from the current LV1 held-out split."""

import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BDDL_ROOT = REPO_ROOT / "libero/libero/bddl_files"
FLOW_DIR = "ablation_industry_flow"
SOURCE_SUITES = {
    "D2D": "libero_industry_d2d",
    "D2S": "libero_industry_d2s",
    "S2D": "libero_industry_s2d",
}
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
HISTORICAL_S2D_LV1_REGIONS = {
    "002": {
        "ghost_init": (0.0950, -0.0165, 0.1150, 0.0035),
        "tray_init": (-0.2050, -0.0165, -0.1850, 0.0035),
    },
    "006": {
        "ghost_init": (0.1221, -0.0558, 0.1421, -0.0358),
        "tray_init": (-0.1779, -0.0558, -0.1579, -0.0358),
    },
    "010": {
        "ghost_init": (0.1034, 0.0149, 0.1234, 0.0349),
        "tray_init": (-0.1966, 0.0149, -0.1766, 0.0349),
    },
    "015": {
        "ghost_init": (0.0454, -0.0238, 0.0654, -0.0038),
        "tray_init": (-0.2546, -0.0238, -0.2346, -0.0038),
    },
}


def _load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _held_out_lv1_tasks():
    from libero.libero.benchmark import libero_task_map

    held_out = set(libero_task_map["libero_industry_test"])
    selected = {}
    for family, suite in SOURCE_SUITES.items():
        tasks = sorted(
            task
            for task in libero_task_map[suite]
            if task in held_out and "_LV1_" in task
        )
        if len(tasks) != 4:
            raise RuntimeError("Expected four held-out {} LV1 tasks, got {}".format(family, tasks))
        selected[family] = tasks
    return selected


def _run_object_tier(module, source_dir, target_dir, target_attribute):
    setattr(module, "SOURCE_DIR", str(source_dir))
    setattr(module, target_attribute, str(target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)
    source_by_family = {
        family: sorted(path for path in source_dir.glob("*.bddl") if path.name.startswith(family + "_"))
        for family in SOURCE_SUITES
    }
    for task in module.BALANCED_TASKS:
        source = source_by_family[task["type"]].pop(0)
        module.process_file(str(source), task)


def _init_text(text):
    match = re.search(r"\(:init(.*?)\n\s*\)\s*\n\s*\(:goal", text, re.DOTALL)
    if match is None:
        raise AssertionError("Cannot isolate :init block")
    return match.group(1)


def _region_bounds(text, name):
    match = re.search(
        r"\(\s*{}\s+.*?\(:ranges\s+\(\s*\(([^()]*)\)".format(re.escape(name)),
        text,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("Cannot read {} bounds".format(name))
    values = tuple(float(value) for value in re.findall(FLOAT_RE, match.group(1)))
    if len(values) != 4:
        raise AssertionError("{} has invalid bounds {}".format(name, values))
    return values


def _restore_historical_s2d_regions(path):
    """Keep D5/D6 bowls clear of the moving tray at reset."""
    source_id = re.search(r"_S2D_LV1_(\d{3})_", path.name).group(1)
    expected = HISTORICAL_S2D_LV1_REGIONS[source_id]
    text = path.read_text(encoding="utf-8")
    for region_name, bounds in expected.items():
        pattern = re.compile(
            r"(\(\s*{}\s+.*?\(:ranges\s+\(\s*\()([^()]*)(\))".format(
                re.escape(region_name)
            ),
            re.DOTALL,
        )
        text, count = pattern.subn(
            lambda match: match.group(1)
            + " ".join("{:.10g}".format(value) for value in bounds)
            + match.group(3),
            text,
            count=1,
        )
        if count != 1:
            raise AssertionError("Cannot replace {} in {}".format(region_name, path))
    path.write_text(text, encoding="utf-8")


def _validate(flow_root, source_tasks):
    expected_ids = {
        family: {re.search(r"_LV1_(\d{3})_", task).group(1) for task in tasks}
        for family, tasks in source_tasks.items()
    }
    report = {"source_tasks": source_tasks, "tiers": {}}
    for tier in range(2, 7):
        suite = "libero_ablation_d{}".format(tier)
        paths = sorted((flow_root / suite).glob("*.bddl"))
        if len(paths) != 12:
            raise AssertionError("{} has {} BDDLs instead of 12".format(suite, len(paths)))
        actual_ids = {family: set() for family in SOURCE_SUITES}
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if text.count("(") != text.count(")"):
                raise AssertionError("Unbalanced parentheses: {}".format(path))
            if "_d2d_" in text or "_d2d_" in path.name:
                raise AssertionError("Legacy _d2d_ region name remains in {}".format(path))
            if "LIBERO_Industry_Workbench_Ablation" not in text:
                raise AssertionError("Wrong problem class in {}".format(path))

            identity = re.search(r"_(D2D|D2S|S2D)_LV1_(\d{3})_", path.name)
            if identity is None:
                raise AssertionError("Cannot parse task identity: {}".format(path.name))
            family, source_id = identity.groups()
            actual_ids[family].add(source_id)
            init_text = _init_text(text)

            if tier in {5, 6} and family == "S2D":
                expected = HISTORICAL_S2D_LV1_REGIONS[source_id]
                for region_name, bounds in expected.items():
                    actual = _region_bounds(text, region_name)
                    if any(abs(left - right) > 1e-9 for left, right in zip(actual, bounds)):
                        raise AssertionError(
                            "{} lost historical S2D {} bounds".format(path, region_name)
                        )

            if tier >= 4 and family in {"D2D", "D2S"}:
                at_speed = re.search(r"\(AtSpeed\s+\S+\s+([-\d.]+).*?\)", init_text)
                if at_speed is None or "takt_4.0" not in at_speed.group(0) or "offset_1.5" not in at_speed.group(0):
                    raise AssertionError("Missing frozen timing metadata in {}".format(path))
                bowl_sites = re.findall(
                    r"\(On\s+ablation_akita_black_bowl_\d+\s+conveyor_belt_1_((?:offset|tail)_bowl_[123])\)",
                    init_text,
                )
                if len(bowl_sites) != tier - 3:
                    raise AssertionError("Wrong bowl-site count in {}: {}".format(path, bowl_sites))
            if tier >= 4 and family in {"D2D", "S2D"}:
                load_slot = re.search(r"\(LoadSlot.*?\)", init_text)
                if load_slot is None or "service_1.5" not in load_slot.group(0):
                    raise AssertionError("Missing service metadata in {}".format(path))
            if tier == 3:
                at_speed = re.search(r"\(AtSpeed\s+\S+\s+([-\d.]+).*?\)", init_text)
                if family in {"D2D", "D2S"}:
                    if at_speed is None or float(at_speed.group(1)) != 0.0:
                        raise AssertionError("D3 conveyor is not stationary in {}".format(path))
                    if "takt_4.0" not in at_speed.group(0) or "offset_1.5" not in at_speed.group(0):
                        raise AssertionError("D3 lost source timing metadata in {}".format(path))
                if family in {"D2D", "S2D"}:
                    load_slot = re.search(r"\(LoadSlot\s+\S+\s+\S+\s+\S+\s+\S+\s+([-\d.]+).*?\)", init_text)
                    if load_slot is None or float(load_slot.group(1)) != 0.0:
                        raise AssertionError("D3 tray is not stationary in {}".format(path))
                    if "service_1.5" not in load_slot.group(0):
                        raise AssertionError("D3 lost service metadata in {}".format(path))
            if tier == 2 and ("(AtSpeed" in init_text or "(LoadSlot" in init_text):
                raise AssertionError("D2 still contains dynamic timing in {}".format(path))
            if tier == 2 and family in {"D2D", "S2D"}:
                d3_path = flow_root / "libero_ablation_d3" / path.name.replace("ABLD2", "ABLD3")
                d3_text = d3_path.read_text(encoding="utf-8")
                d2_bounds = _region_bounds(text, "tray_init")
                d3_bounds = _region_bounds(d3_text, "ghost_init")
                for low, high in ((0, 2), (1, 3)):
                    same_center = abs(
                        (d2_bounds[low] + d2_bounds[high])
                        - (d3_bounds[low] + d3_bounds[high])
                    ) < 1e-9
                    doubled_range = abs(
                        (d2_bounds[high] - d2_bounds[low])
                        - 2.0 * (d3_bounds[high] - d3_bounds[low])
                    ) < 1e-9
                    if not same_center or not doubled_range:
                        raise AssertionError("D2 tray distribution differs from {}".format(d3_path))

        if actual_ids != expected_ids:
            raise AssertionError("{} source IDs differ: {} != {}".format(suite, actual_ids, expected_ids))
        report["tiers"][suite] = [path.name for path in paths]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Empty staging directory that will receive bddl_files/ablation_industry_flow.",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        parser.error("--output-root must be empty: {}".format(output_root))

    flow_root = output_root / "bddl_files" / FLOW_DIR
    source_dir = flow_root / "libero_original_lv1"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_tasks = _held_out_lv1_tasks()
    for family, tasks in source_tasks.items():
        source_suite = BDDL_ROOT / SOURCE_SUITES[family]
        for task in tasks:
            shutil.copy2(str(source_suite / (task + ".bddl")), str(source_dir / (task + ".bddl")))

    d4 = _load("ablation_d4")
    d5 = _load("ablation_d5")
    d6 = _load("ablation_d6")
    d3 = _load("ablation_d3")
    d2 = _load("ablation_d2")
    _run_object_tier(d4, source_dir, flow_root / "libero_ablation_d4", "TARGET_DIR_L4")
    _run_object_tier(d5, source_dir, flow_root / "libero_ablation_d5", "TARGET_DIR")
    _run_object_tier(d6, source_dir, flow_root / "libero_ablation_d6", "TARGET_DIR")
    for tier in (5, 6):
        for path in sorted((flow_root / "libero_ablation_d{}".format(tier)).glob("*_S2D_*.bddl")):
            _restore_historical_s2d_regions(path)

    d3_target = flow_root / "libero_ablation_d3"
    for source in sorted((flow_root / "libero_ablation_d4").glob("*.bddl")):
        d3.process_stage_02_bddl(str(source), str(d3_target))

    d2_target = flow_root / "libero_ablation_d2"
    for source in sorted(d3_target.glob("*.bddl")):
        d2.convert_stage(source, d2_target / source.name.replace("ABLD3", "ABLD2"))

    report = _validate(flow_root, source_tasks)
    report_path = output_root / "regeneration.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Validated 60 staged D2-D6 BDDLs: {}".format(flow_root))
    print("Report: {}".format(report_path))


if __name__ == "__main__":
    main()
