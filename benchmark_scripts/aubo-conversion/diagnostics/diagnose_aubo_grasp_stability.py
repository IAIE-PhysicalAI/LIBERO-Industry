#!/usr/bin/env python
"""Diagnose Panda-to-Aubo gripper differences without modifying any dataset.

The tool has two deliberately separate parts:

* XML inventory: compares Panda's stock gripper definition with the local
  CTAG2F90D definition. This can run on a workstation without any HDF5 data.
* Stability replay: restores each source LIBERO scene, replays Panda's source
  actions and Aubo's converted actions up to the first close command, then
  holds the gripper closed and applies a small vertical lift. It reports
  finger-pad contact, relative object drift, and whether a contacted movable
  object followed the lift.

All HDF5 files are opened read-only. The only files this script writes are a
JSON diagnosis report and optional MP4 videos.

Examples
--------
Compare XML locally::

    python benchmark_scripts/aubo-conversion/diagnostics/diagnose_aubo_grasp_stability.py --xml-only

Sample up to two failed demonstrations per LIBERO Object HDF5::

    python benchmark_scripts/aubo-conversion/diagnostics/diagnose_aubo_grasp_stability.py \
      --source-dir DATASET_ROOT/LIBERO-datasets/libero_object \
      --converted-dir DATASET_ROOT/LIBERO-datasets-auboi5/libero_object \
      --failed-only --max-demos-per-file 2 \
      --report-out .industry-runs/gripper_diagnostics/libero_object.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


_CONVERSION_DIR = Path(__file__).resolve().parent.parent
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CTAG_XML = (
    _REPOSITORY_ROOT
    / "libero"
    / "libero"
    / "assets"
    / "grippers"
    / "ctag2f90d_gripper"
    / "ctag2f90d_gripper.xml"
)
_CLOSE_ACTION = 1.0


def _ascii_table(headers, rows):
    """Return a compact ASCII table without an external formatting dependency."""
    rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row):
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    return "\n".join(
        [separator, format_row(headers), separator]
        + [format_row(row) for row in rows]
        + [separator]
    )


def _xml_path_for_panda():
    """Resolve Panda's installed Robosuite XML without assuming a version path."""
    from robosuite.utils.mjcf_utils import xml_path_completion

    return Path(xml_path_completion("grippers/panda_gripper.xml"))


def _xml_inventory(xml_path):
    """Extract contact-relevant fields while retaining raw attributes in JSON."""
    root = ET.parse(xml_path).getroot()

    # Only worldbody joints are physical DoFs. Equality constraints also use
    # ``<joint>`` tags, but they are relationships, not additional joints.
    joints = [dict(element.attrib) for element in root.findall(".//worldbody//joint")]
    actuators = []
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for element in actuator_root:
            entry = {"type": element.tag}
            entry.update(element.attrib)
            actuators.append(entry)

    collision_geoms = []
    for element in root.findall(".//geom"):
        attributes = dict(element.attrib)
        contype = int(attributes.get("contype", "1"))
        conaffinity = int(attributes.get("conaffinity", "1"))
        if contype != 0 and conaffinity != 0:
            collision_geoms.append(attributes)

    grip_sites = [
        dict(element.attrib)
        for element in root.findall(".//site")
        if "grip_site" in element.attrib.get("name", "")
    ]
    equalities = []
    equality_root = root.find("equality")
    if equality_root is not None:
        for element in equality_root:
            entry = {"type": element.tag}
            entry.update(element.attrib)
            equalities.append(entry)

    total_mass = 0.0
    for element in root.findall(".//inertial"):
        try:
            total_mass += float(element.attrib.get("mass", 0.0))
        except ValueError:
            pass

    return {
        "path": str(xml_path),
        "joint_count": len(joints),
        "actuator_count": len(actuators),
        "collision_geom_count": len(collision_geoms),
        "equality_count": len(equalities),
        "declared_mass_kg": total_mass,
        "grip_sites": grip_sites,
        "joints": joints,
        "actuators": actuators,
        "collision_geoms": collision_geoms,
        "equalities": equalities,
    }


def compare_gripper_xml():
    """Return a structural comparison; this function does not instantiate MuJoCo."""
    panda_path = _xml_path_for_panda()
    if not _CTAG_XML.is_file():
        raise FileNotFoundError(f"CTAG XML is missing: {_CTAG_XML}")

    comparison = {
        "panda": _xml_inventory(panda_path),
        "ctag2f90d": _xml_inventory(_CTAG_XML),
        "action_contract": {
            "panda": "Robosuite PandaGripper: -1 opens, +1 closes; two finger joints.",
            "ctag2f90d": (
                "CTAG2F90DGripper: -1 opens, +1 closes; one policy action is "
                "expanded to six XML actuators with five equality-linked followers."
            ),
        },
    }
    return comparison


def _print_xml_comparison(comparison):
    panda = comparison["panda"]
    ctag = comparison["ctag2f90d"]
    rows = []
    for label, record in (("Panda", panda), ("CTAG2F90D", ctag)):
        frictions = ", ".join(
            geom.get("friction", "MuJoCo default") for geom in record["collision_geoms"]
        ) or "none"
        grip_site_positions = ", ".join(
            site.get("pos", "0 0 0") for site in record["grip_sites"]
        ) or "missing"
        rows.append(
            (
                label,
                record["joint_count"],
                record["actuator_count"],
                record["collision_geom_count"],
                record["equality_count"],
                f"{record['declared_mass_kg']:.4f}",
                frictions,
                grip_site_positions,
            )
        )
    print("\n=== GRIPPER XML / CONTACT INVENTORY ===")
    print(
        _ascii_table(
            (
                "gripper",
                "joints",
                "actuators",
                "active collision geoms",
                "equalities",
                "declared mass (kg)",
                "finger friction",
                "grip-site local pos",
            ),
            rows,
        )
    )


def _write_json(path, value):
    """Atomically write a report so an interrupted diagnostic cannot look complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        json.dump(value, temporary_file, indent=2, sort_keys=True)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)


def _load_failed_demos(report_path):
    """Read terminal failures from a completed converter sidecar report."""
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    trajectories = report.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError(f"Report has no 'trajectories' list: {report_path}")
    return [
        {
            "demo": trajectory["trajectory"],
            "ever_task_success": bool(trajectory.get("ever_task_success", False)),
        }
        for trajectory in trajectories
        if not bool(trajectory.get("terminal_task_success", False))
        and isinstance(trajectory.get("trajectory"), str)
    ]


def _representative_failed_demos(failures, maximum):
    """Sample both failure signatures before taking additional failed demos.

    A failure that never reached the task predicate and a failure that reached
    it then lost it point to different causes. With the default two-demo cap,
    each HDF5 therefore contributes one of each whenever both exist.
    """
    ordered = []
    for reached_success in (False, True):
        ordered.extend(
            failure["demo"]
            for failure in failures
            if failure["ever_task_success"] is reached_success
        )
    if maximum is not None:
        ordered = ordered[:maximum]
    return ordered


def _sorted_demos(data_group):
    demos = list(data_group.keys())
    try:
        # Python 3.8 is still used by the Robosuite 1.4 deployment.
        return sorted(demos, key=lambda name: int(name[5:]))
    except ValueError as error:
        raise ValueError(f"Expected demo_<integer> names, received {demos}") from error


def _select_pairs(arguments):
    """Return (source, converted, selected_demos) tuples without changing either HDF5."""
    source_file = Path(arguments.source_hdf5) if arguments.source_hdf5 else None
    converted_file = Path(arguments.converted_hdf5) if arguments.converted_hdf5 else None
    if (source_file is None) != (converted_file is None):
        raise ValueError("Use --source-hdf5 and --converted-hdf5 together.")
    if source_file is not None:
        raw_pairs = [(source_file, converted_file)]
    else:
        if not arguments.source_dir or not arguments.converted_dir:
            raise ValueError(
                "Use either the two --*-hdf5 options or both --source-dir and --converted-dir."
            )
        source_dir = Path(arguments.source_dir)
        converted_dir = Path(arguments.converted_dir)
        raw_pairs = [
            (source_path, converted_dir / source_path.name)
            for source_path in sorted(source_dir.glob("*.hdf5"))
        ]

    explicit_demos = None
    if arguments.demos:
        explicit_demos = [demo.strip() for demo in arguments.demos.split(",") if demo.strip()]
        if not explicit_demos:
            raise ValueError("--demos must contain at least one demo name.")
        if len(raw_pairs) != 1:
            raise ValueError("--demos is only unambiguous with one --source-hdf5 pair.")

    selected_pairs = []
    for source_path, converted_path in raw_pairs:
        if not source_path.is_file():
            print(f"[SKIP] Source HDF5 does not exist: {source_path}")
            continue
        if not converted_path.is_file():
            print(f"[SKIP] Converted HDF5 does not exist: {converted_path}")
            continue

        with h5py.File(source_path, "r") as source_data:
            available_demos = _sorted_demos(source_data["data"])

        if explicit_demos is not None:
            selected = explicit_demos
        elif arguments.failed_only:
            report_path = converted_path.with_suffix(".conversion_report.json")
            if not report_path.is_file():
                print(f"[SKIP] No completed conversion report: {report_path}")
                continue
            selected = _representative_failed_demos(
                _load_failed_demos(report_path), arguments.max_demos_per_file
            )
        else:
            selected = available_demos

        missing = sorted(set(selected) - set(available_demos))
        if missing:
            raise ValueError(
                f"Selected demos are absent from {source_path.name}: {', '.join(missing)}"
            )
        if not arguments.failed_only and arguments.max_demos_per_file is not None:
            selected = selected[: arguments.max_demos_per_file]
        if selected:
            selected_pairs.append((source_path, converted_path, selected))
    return selected_pairs


def _import_converter():
    """Load the conversion compatibility layer only when a simulator test is requested."""
    sys.path.insert(0, str(_CONVERSION_DIR))
    import convert_to_aubo as converter

    return converter


def _decode_hdf5_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _resolved_bddl_path(bddl_file, converter):
    """Map dataset machine-specific BDDL paths into the active checkout."""
    if not bddl_file:
        return None
    bddl_file = _decode_hdf5_text(bddl_file)
    relative = bddl_file.split("bddl_files/", 1)[-1]
    return str(Path(converter.get_libero_path("bddl_files")) / relative)


def _prepare_environment_metadata(source_path, converted_path, converter):
    """Build fresh Panda and Aubo metadata, never trusting an edited output attr."""
    with h5py.File(source_path, "r") as source_data, h5py.File(
        converted_path, "r"
    ) as converted_data:
        source_args = _decode_hdf5_text(source_data["data"].attrs.get("env_args", "{}"))
        target_args = _decode_hdf5_text(converted_data["data"].attrs.get("env_args", "{}"))
        bddl_file = source_data["data"].attrs.get("bddl_file_name")

    reference_meta = json.loads(source_args) if source_args != "{}" else {}
    target_meta = json.loads(target_args) if target_args != "{}" else {}
    for metadata in (reference_meta, target_meta):
        metadata.setdefault("env_kwargs", {})

    env_name = reference_meta.get("env_name") or target_meta.get("env_name")
    if not env_name:
        raise KeyError(f"No env_name in {source_path} or {converted_path}")
    expected_panda, expected_aubo = converter._domain_robot_models(env_name)
    bddl_path = _resolved_bddl_path(bddl_file, converter)
    if bddl_path is not None:
        reference_meta["env_kwargs"]["bddl_file_name"] = bddl_path
        target_meta["env_kwargs"]["bddl_file_name"] = bddl_path

    reference_meta["env_kwargs"]["robots"] = ["Panda"]
    reference_meta["env_kwargs"]["gripper_types"] = ["PandaGripper"]
    target_meta["env_kwargs"]["robots"] = ["AuboI5"]
    target_meta["env_kwargs"]["gripper_types"] = ["CTAG2F90DGripper"]
    converter._set_controller_metadata(reference_meta, preserve_legacy_config=True)
    converter._set_controller_metadata(target_meta)
    return reference_meta, target_meta, env_name, expected_panda, expected_aubo


def _make_environments(source_path, converted_path, want_video, converter):
    """Create the two environments with the converter's 1.4 / 1.5 compatibility path."""
    reference_meta, target_meta, env_name, expected_panda, expected_aubo = (
        _prepare_environment_metadata(source_path, converted_path, converter)
    )
    cameras = ["agentview"] if want_video else []
    reference_env = converter.EnvUtils.create_env_for_data_processing(
        env_meta=reference_meta,
        camera_names=cameras,
        camera_height=256,
        camera_width=256,
        reward_shaping=False,
    )
    target_env = converter.EnvUtils.create_env_for_data_processing(
        env_meta=target_meta,
        camera_names=cameras,
        camera_height=256,
        camera_width=256,
        reward_shaping=False,
    )
    converter._assert_runtime_robot(reference_env, expected_panda, "Panda diagnostic")
    converter._assert_runtime_robot(target_env, expected_aubo, "Aubo diagnostic")
    for label, environment in (("Panda", reference_env), ("Aubo", target_env)):
        action_dimension = converter._action_dimension(environment)
        if action_dimension != 7:
            raise RuntimeError(
                f"{label} diagnostic expects seven actions, got {action_dimension}."
            )
    return reference_env, target_env, env_name


def _pad_geom_ids(environment, converter):
    """Find the currently-prefixed collision geoms for the active gripper pads."""
    base_env = converter._get_base_env(environment)
    requested_names = []
    for key in ("left_fingerpad", "right_fingerpad", "left_finger", "right_finger"):
        requested_names.extend(base_env.robots[0].gripper.important_geoms.get(key, []))

    found = set()
    for geom_id in range(base_env.sim.model.ngeom):
        geom_name = base_env.sim.model.geom_id2name(geom_id)
        if geom_name and any(
            geom_name == requested or geom_name.endswith(requested)
            for requested in requested_names
        ):
            found.add(geom_id)
    if not found:
        raise RuntimeError(
            f"Could not locate finger-pad collision geoms for {type(base_env.robots[0].gripper).__name__}."
        )
    return found


def _task_body_index(base_env):
    """Associate a MuJoCo body with the nearest named LIBERO task root body."""
    root_by_id = {int(body_id): name for name, body_id in base_env.obj_body_id.items()}
    movable = set(getattr(base_env, "objects_dict", {}))
    fixtures = set(getattr(base_env, "fixtures_dict", {}))
    parents = base_env.sim.model.body_parentid

    def task_name(body_id):
        body_id = int(body_id)
        while body_id >= 0:
            name = root_by_id.get(body_id)
            if name is not None:
                return name
            parent_id = int(parents[body_id])
            if parent_id == body_id:
                break
            body_id = parent_id
        return None

    return task_name, movable, fixtures, root_by_id


def _pad_contacts(environment, pad_ids, converter, task_name_for_body):
    """Return contacts between a finger pad and task bodies for the current step."""
    base_env = converter._get_base_env(environment)
    contacts = []
    for contact_id in range(base_env.sim.data.ncon):
        contact = base_env.sim.data.contact[contact_id]
        first_geom, second_geom = int(contact.geom1), int(contact.geom2)
        if first_geom in pad_ids:
            other_geom = second_geom
        elif second_geom in pad_ids:
            other_geom = first_geom
        else:
            continue
        other_body = int(base_env.sim.model.geom_bodyid[other_geom])
        contacts.append(
            {
                "task_body": task_name_for_body(other_body),
                "body": base_env.sim.model.body_id2name(other_body) or str(other_body),
                "geom": base_env.sim.model.geom_id2name(other_geom) or str(other_geom),
            }
        )
    return contacts


def _snapshot(environment, converter, root_by_id):
    base_env = converter._get_base_env(environment)
    tcp_position, _ = converter.get_eef_pose(environment)
    body_positions = {
        task_name: np.array(base_env.sim.data.body_xpos[body_id], dtype=float)
        for body_id, task_name in root_by_id.items()
    }
    return np.array(tcp_position, dtype=float), body_positions


def _append_frame(observation, frames):
    if frames is None:
        return
    image = observation.get("agentview_image")
    if image is None:
        raise KeyError("Video was requested but agentview_image is unavailable.")
    frames.append(np.asarray(image))


def _step(environment, action, frames):
    observation, _, _, _ = environment.step(action)
    _append_frame(observation, frames)


def _first_close_step(actions, close_threshold):
    close_steps = np.flatnonzero(np.asarray(actions)[:, -1] > close_threshold)
    if close_steps.size == 0:
        return None
    return int(close_steps[0])


def _evaluate_stability(
    environment,
    actions,
    close_step,
    hold_steps,
    settle_steps,
    lift_steps,
    lift_action,
    threshold_mm,
    frames,
    converter,
):
    """Replay through grasp close, then isolate the closed-gripper hold / lift test."""
    pad_ids = _pad_geom_ids(environment, converter)
    base_env = converter._get_base_env(environment)
    task_name_for_body, movable, fixtures, root_by_id = _task_body_index(base_env)
    contact_counts = Counter()
    non_task_contacts = Counter()

    def record_contacts():
        for contact in _pad_contacts(environment, pad_ids, converter, task_name_for_body):
            if contact["task_body"] is None:
                non_task_contacts[f"{contact['body']}::{contact['geom']}"] += 1
            else:
                contact_counts[contact["task_body"]] += 1

    for action in actions[: close_step + 1]:
        _step(environment, action, frames)
        record_contacts()

    close_hold = np.zeros(7)
    close_hold[-1] = _CLOSE_ACTION
    for _ in range(settle_steps):
        _step(environment, close_hold, frames)
        record_contacts()

    start_tcp, start_positions = _snapshot(environment, converter, root_by_id)
    static_relative_drift = Counter()
    max_static_drop = Counter()
    for _ in range(hold_steps):
        _step(environment, close_hold, frames)
        record_contacts()
        tcp_position, body_positions = _snapshot(environment, converter, root_by_id)
        for name in movable:
            if name not in start_positions or name not in body_positions:
                continue
            relative_drift = np.linalg.norm(
                (body_positions[name] - tcp_position) - (start_positions[name] - start_tcp)
            )
            static_relative_drift[name] = max(static_relative_drift[name], relative_drift)
            max_static_drop[name] = min(
                max_static_drop[name], body_positions[name][2] - start_positions[name][2]
            )

    lift_command = np.array(close_hold)
    lift_command[2] = lift_action
    for _ in range(lift_steps):
        _step(environment, lift_command, frames)
        record_contacts()
    for _ in range(settle_steps):
        _step(environment, close_hold, frames)
        record_contacts()
    end_tcp, end_positions = _snapshot(environment, converter, root_by_id)

    movable_contacts = {
        name: count for name, count in contact_counts.items() if name in movable
    }
    candidate = max(movable_contacts, key=movable_contacts.get) if movable_contacts else None
    threshold_m = threshold_mm / 1000.0
    # OSC tracks position imperfectly under contact. Require a meaningful lift
    # but do not require the full commanded displacement: a genuine Panda grasp
    # in the reference test can be slightly below that ideal displacement.
    min_lift_m = max(0.008, abs(lift_action) * 0.05 * lift_steps * 0.10)

    if candidate is None or candidate not in start_positions or candidate not in end_positions:
        result = {
            "classification": "no_movable_finger_pad_contact",
            "candidate_object": None,
            "candidate_contact_steps": 0,
            "static_relative_drift_mm": None,
            "static_height_drop_mm": None,
            "lifted_with_tcp_mm": None,
            "final_relative_drift_mm": None,
        }
    else:
        static_drift = float(static_relative_drift[candidate])
        static_drop = float(max_static_drop[candidate])
        lifted_with_tcp = float(end_positions[candidate][2] - start_positions[candidate][2])
        final_relative_drift = float(
            np.linalg.norm(
                (end_positions[candidate] - end_tcp) - (start_positions[candidate] - start_tcp)
            )
        )
        if static_drift > threshold_m or static_drop < -threshold_m:
            classification = "slips_during_stationary_hold"
        elif lifted_with_tcp < min_lift_m:
            classification = "contacted_but_not_lifted"
        elif final_relative_drift > threshold_m:
            classification = "loses_grip_during_lift"
        else:
            classification = "stable_closed_grasp"
        result = {
            "classification": classification,
            "candidate_object": candidate,
            "candidate_contact_steps": int(movable_contacts[candidate]),
            "static_relative_drift_mm": round(static_drift * 1000.0, 3),
            "static_height_drop_mm": round(static_drop * 1000.0, 3),
            "lifted_with_tcp_mm": round(lifted_with_tcp * 1000.0, 3),
            "final_relative_drift_mm": round(final_relative_drift * 1000.0, 3),
        }

    result.update(
        {
            "first_close_step": int(close_step),
            "movable_objects": sorted(movable),
            "fixtures": sorted(fixtures),
            "all_task_pad_contact_steps": dict(sorted(contact_counts.items())),
            "non_task_pad_contacts": dict(non_task_contacts.most_common(8)),
            "hold_steps": int(hold_steps),
            "settle_steps": int(settle_steps),
            "lift_steps": int(lift_steps),
            "lift_action": float(lift_action),
            "minimum_lift_required_mm": round(min_lift_m * 1000.0, 3),
        }
    )
    return result


def _save_video(video_path, frames):
    if not frames:
        return None
    import imageio.v2 as imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=20)
    return str(video_path)


def _run_demo(source_path, converted_path, demo, arguments, converter):
    """Run both embodiments from the same source scene, with no HDF5 mutation."""
    reference_env = None
    target_env = None
    panda_frames = [] if arguments.video_dir else None
    aubo_frames = [] if arguments.video_dir else None
    try:
        reference_env, target_env, env_name = _make_environments(
            source_path, converted_path, bool(arguments.video_dir), converter
        )
        with h5py.File(source_path, "r") as source_data, h5py.File(
            converted_path, "r"
        ) as converted_data:
            source_demo = source_data[f"data/{demo}"]
            target_demo = converted_data[f"data/{demo}"]
            states = source_demo["states"][()]
            panda_actions = source_demo["actions"][()]
            aubo_actions = target_demo["actions"][()]

        if panda_actions.ndim != 2 or panda_actions.shape[1] != 7:
            raise ValueError(f"Panda {demo} actions must have shape (T, 7), got {panda_actions.shape}")
        if aubo_actions.ndim != 2 or aubo_actions.shape[1] != 7:
            raise ValueError(f"Aubo {demo} actions must have shape (T, 7), got {aubo_actions.shape}")
        if len(states) == 0:
            raise ValueError(f"{source_path.name} {demo} has no states")

        panda_close = _first_close_step(panda_actions, arguments.close_threshold)
        aubo_close = _first_close_step(aubo_actions, arguments.close_threshold)
        if panda_close is None or aubo_close is None:
            return {
                "source_hdf5": str(source_path),
                "converted_hdf5": str(converted_path),
                "demo": demo,
                "environment": env_name,
                "classification": "missing_close_command",
                "panda_first_close_step": panda_close,
                "aubo_first_close_step": aubo_close,
            }

        target_observation = converter.initialize_target_replay(
            reference_env,
            target_env,
            states[0],
            panda_actions[0, -1],
            7,
            seed=arguments.seed,
        )
        _append_frame(target_observation, aubo_frames)

        panda_result = _evaluate_stability(
            reference_env,
            panda_actions,
            panda_close,
            arguments.hold_steps,
            arguments.settle_steps,
            arguments.lift_steps,
            arguments.lift_action,
            arguments.relative_drift_threshold_mm,
            panda_frames,
            converter,
        )
        aubo_result = _evaluate_stability(
            target_env,
            aubo_actions,
            aubo_close,
            arguments.hold_steps,
            arguments.settle_steps,
            arguments.lift_steps,
            arguments.lift_action,
            arguments.relative_drift_threshold_mm,
            aubo_frames,
            converter,
        )

        result = {
            "source_hdf5": str(source_path),
            "converted_hdf5": str(converted_path),
            "demo": demo,
            "environment": env_name,
            "reset_seed": int(arguments.seed),
            "panda": panda_result,
            "aubo": aubo_result,
        }
        if arguments.video_dir:
            root = Path(arguments.video_dir)
            stem = f"{source_path.stem}__{demo}__run{arguments._repeat_index}"
            result["panda_video"] = _save_video(root / f"{stem}__panda_hold_lift.mp4", panda_frames)
            result["aubo_video"] = _save_video(root / f"{stem}__aubo_hold_lift.mp4", aubo_frames)
        return result
    finally:
        converter._close_env(reference_env)
        converter._close_env(target_env)


def _print_stability_summary(results):
    rows = []
    for result in results:
        if "panda" not in result:
            rows.append(
                (
                    Path(result["source_hdf5"]).name,
                    result["demo"],
                    result.get("repeat", 1),
                    "n/a",
                    result["classification"],
                    "-",
                    "-",
                    "-",
                )
            )
            continue
        for robot_name in ("panda", "aubo"):
            stability = result[robot_name]
            rows.append(
                (
                    Path(result["source_hdf5"]).name,
                    result["demo"],
                    result.get("repeat", 1),
                    robot_name,
                    stability["classification"],
                    stability["candidate_object"] or "none",
                    stability["static_relative_drift_mm"],
                    stability["lifted_with_tcp_mm"],
                )
            )
    print("\n=== CLOSED-GRIPPER STABILITY SUMMARY ===")
    print(
        _ascii_table(
            (
                "HDF5",
                "demo",
                "repeat",
                "robot",
                "diagnosis",
                "contacted movable object",
                "static relative drift (mm)",
                "object lift (mm)",
            ),
            rows,
        )
    )


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare Panda / CTAG XML and run non-mutating grasp hold-lift diagnostics."
    )
    parser.add_argument("--xml-only", action="store_true", help="Only compare XML; no HDF5 or MuJoCo replay.")
    parser.add_argument("--source-hdf5", help="One original Panda LIBERO demonstration HDF5.")
    parser.add_argument("--converted-hdf5", help="The corresponding converted Aubo HDF5.")
    parser.add_argument("--source-dir", help="Directory of original Panda HDF5 files, e.g. LIBERO-datasets/libero_object.")
    parser.add_argument("--converted-dir", help="Matching directory of converted Aubo HDF5 files.")
    parser.add_argument("--failed-only", action="store_true", help="Read converter reports and replay only terminally failed demos.")
    parser.add_argument("--demos", help="Comma-separated demo names for one HDF5 pair, e.g. demo_12,demo_22.")
    parser.add_argument(
        "--max-demos-per-file",
        type=int,
        default=2,
        help="Cross-object sample cap per HDF5 (default: 2; use 0 for no cap).",
    )
    parser.add_argument("--hold-steps", type=int, default=120, help="Closed stationary hold duration (default: 120).")
    parser.add_argument("--settle-steps", type=int, default=10, help="Closed-gripper settle duration before / after lift.")
    parser.add_argument("--lift-steps", type=int, default=5, help="Small closed-gripper lift duration (default: 5).")
    parser.add_argument("--lift-action", type=float, default=0.25, help="World-Z OSC command during the diagnostic lift.")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent environment replays per selected demo (default: 1; use 3 to quantify contact instability).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic target-reset seed shared with the converter (default: 0).",
    )
    parser.add_argument("--close-threshold", type=float, default=0.0, help="Action above this value is treated as a close command.")
    parser.add_argument(
        "--relative-drift-threshold-mm",
        type=float,
        default=3.0,
        help="Maximum allowed object-to-TCP drift for a stable grasp (default: 3 mm).",
    )
    parser.add_argument("--video-dir", help="Optional directory for Panda and Aubo hold-lift MP4s; disabled by default.")
    parser.add_argument(
        "--report-out",
        default=".industry-runs/gripper_diagnostics/gripper_stability_report.json",
        help="Output JSON report (default: .industry-runs/gripper_diagnostics/gripper_stability_report.json).",
    )
    arguments = parser.parse_args()
    for name in ("max_demos_per_file", "hold_steps", "settle_steps", "lift_steps", "repeats"):
        value = getattr(arguments, name)
        if value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if arguments.max_demos_per_file == 0:
        arguments.max_demos_per_file = None
    if arguments.relative_drift_threshold_mm <= 0:
        parser.error("--relative-drift-threshold-mm must be positive")
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if not arguments.xml_only and not (arguments.source_hdf5 or arguments.source_dir):
        parser.error("Provide HDF5 input options, or use --xml-only.")
    return arguments


def main():
    arguments = _parse_arguments()
    comparison = compare_gripper_xml()
    _print_xml_comparison(comparison)
    report = {
        "gripper_xml_comparison": comparison,
        "diagnostic_config": {
            "hold_steps": arguments.hold_steps,
            "settle_steps": arguments.settle_steps,
            "lift_steps": arguments.lift_steps,
            "lift_action": arguments.lift_action,
            "relative_drift_threshold_mm": arguments.relative_drift_threshold_mm,
            "repeats": arguments.repeats,
            "reset_seed": int(arguments.seed),
        },
        "trials": [],
    }

    if not arguments.xml_only:
        converter = _import_converter()
        pairs = _select_pairs(arguments)
        if not pairs:
            raise RuntimeError("No valid source / converted HDF5 pair with selected demonstrations.")
        demo_count = sum(len(demos) for _, _, demos in pairs)
        print(f"\n[INFO] Running {demo_count} diagnostic demo(s) x {arguments.repeats} repeat(s).")
        for source_path, converted_path, demos in pairs:
            for demo in demos:
                for repeat_index in range(1, arguments.repeats + 1):
                    arguments._repeat_index = repeat_index
                    print(f"[DIAGNOSE] {source_path.name} :: {demo} :: run {repeat_index}")
                    try:
                        result = _run_demo(
                            source_path, converted_path, demo, arguments, converter
                        )
                        result["repeat"] = repeat_index
                        report["trials"].append(result)
                    except Exception as error:
                        report["trials"].append(
                            {
                                "source_hdf5": str(source_path),
                                "converted_hdf5": str(converted_path),
                                "demo": demo,
                                "repeat": repeat_index,
                                "classification": "diagnostic_error",
                                "error": str(error),
                            }
                        )
                        print(f"[ERROR] {source_path.name} :: {demo}: {error}")
        _print_stability_summary(report["trials"])

    _write_json(arguments.report_out, report)
    print(f"\n[WROTE] {Path(arguments.report_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
