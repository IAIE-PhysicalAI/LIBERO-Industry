#!/usr/bin/env python
"""Combine online-successful raw Aubo demos with accepted MimicGen recoveries.

The raw and recovery roots are read-only. Each output task is written through
a hidden temporary HDF5 inside the final recovered root, checked against its
selected source arrays, and then atomically renamed.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


RAW_ROOT_BASENAME = "LIBERO-datasets-auboi5h-raw"
OUTPUT_ROOT_BASENAME = "LIBERO-datasets-auboi5h-recovered-v2"
SUPPORTED_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)
WRIST_KEYS = (
    "gripper0_eye_in_hand_image",
    "gripper0_right_eye_in_hand_image",
    "robot0_eye_in_hand_image",
)
REQUIRED_OBSERVATIONS = {
    "agentview_image",
    "robot0_joint_pos",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
}
NOOP_EPSILON = 1e-4


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _demo_number(name: str) -> int:
    if not name.startswith("demo_"):
        raise ValueError(f"Invalid demonstration name: {name}")
    return int(name.split("_", 1)[1])


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _wrist_key(observations: h5py.Group) -> str:
    matches = [key for key in WRIST_KEYS if key in observations]
    if len(matches) != 1:
        raise ValueError(
            f"{observations.name} must contain exactly one wrist image key; "
            f"found {matches}"
        )
    return matches[0]


def _validate_demo(demo: h5py.Group, canonical_wrist: str | None = None) -> int:
    horizon = demo.attrs.get("num_samples")
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError(f"{demo.name} has an invalid num_samples")
    horizon = int(horizon)
    for key in ("actions", "states", "rewards", "dones", "obs"):
        if key not in demo:
            raise ValueError(f"{demo.name} is missing {key}")
    if demo["actions"].shape != (horizon, 7):
        raise ValueError(f"{demo.name} actions must have shape ({horizon}, 7)")
    if demo["states"].ndim != 2 or demo["states"].shape[0] != horizon:
        raise ValueError(f"{demo.name} states have a wrong horizon")
    if demo["rewards"].shape != (horizon,) or demo["dones"].shape != (horizon,):
        raise ValueError(f"{demo.name} rewards or dones have a wrong horizon")
    for key in ("actions", "states", "rewards"):
        if not np.all(np.isfinite(demo[key][()])):
            raise ValueError(f"{demo.name}/{key} contains NaN or inf")
    actions = np.asarray(demo["actions"][()])
    if np.max(np.abs(actions)) > 1.0 + 1e-6:
        raise ValueError(f"{demo.name} contains an action outside [-1, 1]")
    rewards = np.asarray(demo["rewards"][()])
    stable_success = int(demo.attrs.get("terminal_success_streak", 0)) >= 5
    if horizon < 5 or (not stable_success and not np.all(rewards[-5:] > 0.0)):
        raise ValueError(f"{demo.name} does not pass stable-success validation")
    noops = (np.linalg.norm(actions[1:, :6], axis=1) < NOOP_EPSILON) & (
        np.abs(np.diff(actions[:, 6])) <= NOOP_EPSILON
    )
    if np.any(noops):
        raise ValueError(f"{demo.name} contains a no-op action")
    maximum_horizon = demo.attrs.get("maximum_horizon")
    if maximum_horizon is not None and horizon > int(maximum_horizon):
        raise ValueError(f"{demo.name} exceeds its maximum horizon")
    if not np.array_equal(np.asarray(demo.attrs["init_state"]), demo["states"][0]):
        raise ValueError(f"{demo.name} init_state differs from states[0]")

    observations = demo["obs"]
    wrist = _wrist_key(observations)
    if canonical_wrist is not None and wrist != canonical_wrist:
        raise ValueError(f"{demo.name} uses {wrist}, expected {canonical_wrist}")
    if not REQUIRED_OBSERVATIONS.issubset(observations):
        missing = sorted(REQUIRED_OBSERVATIONS - set(observations))
        raise ValueError(f"{demo.name}/obs is missing {missing}")
    for key in REQUIRED_OBSERVATIONS | {wrist}:
        if observations[key].shape[0] != horizon:
            raise ValueError(f"{demo.name}/obs/{key} has a wrong horizon")
    for key in ("agentview_image", wrist):
        if observations[key].shape[1:] != (256, 256, 3):
            raise ValueError(f"{demo.name}/obs/{key} has a wrong image shape")
        if observations[key].dtype != np.uint8:
            raise ValueError(f"{demo.name}/obs/{key} must be uint8")
    for key in REQUIRED_OBSERVATIONS - {"agentview_image"}:
        if not np.all(np.isfinite(observations[key][()])):
            raise ValueError(f"{demo.name}/obs/{key} contains NaN or inf")
    quaternions = observations["robot0_eef_quat"][()]
    if quaternions.shape != (horizon, 4) or not np.allclose(
        np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError(f"{demo.name} has invalid EEF quaternions")
    return horizon


def _datasets_equal(source: h5py.Dataset, target: h5py.Dataset) -> bool:
    """Compare arrays directly without loading a full image trajectory at once."""
    if source.shape != target.shape or source.dtype != target.dtype:
        return False
    if source.ndim == 0:
        return bool(np.array_equal(source[()], target[()]))
    row_bytes = max(1, int(np.prod(source.shape[1:])) * source.dtype.itemsize)
    rows = max(1, (8 * 1024 * 1024) // row_bytes)
    return all(
        np.array_equal(source[start : start + rows], target[start : start + rows])
        for start in range(0, source.shape[0], rows)
    )


def _copied_arrays_equal(
    source: h5py.Group, target: h5py.Group, canonical_wrist: str
) -> bool:
    for key in ("actions", "states", "rewards", "dones"):
        if not _datasets_equal(source[key], target[key]):
            return False
    source_obs = source["obs"]
    target_obs = target["obs"]
    source_wrist = _wrist_key(source_obs)
    mapped = {
        canonical_wrist if key == source_wrist else key: key for key in source_obs
    }
    if set(mapped) != set(target_obs):
        return False
    return all(
        _datasets_equal(source_obs[source_key], target_obs[target_key])
        for target_key, source_key in mapped.items()
    )


def _copy_demo(
    source: h5py.Group,
    name: str,
    target: h5py.Group,
    canonical_wrist: str,
    origin: str,
) -> int:
    _validate_demo(source[name])
    source.copy(name, target, name=name)
    demo = target[name]
    observations = demo["obs"]
    wrist = _wrist_key(observations)
    if wrist != canonical_wrist:
        observations.move(wrist, canonical_wrist)
        demo.attrs["materialized_wrist_source_key"] = wrist
    demo.attrs["materialization_origin"] = origin
    demo.attrs["materialization_source_hdf5"] = str(source.file.filename)
    horizon = _validate_demo(demo, canonical_wrist)
    if not _copied_arrays_equal(source[name], demo, canonical_wrist):
        raise RuntimeError(f"{demo.name} differs from its selected source arrays")
    return horizon


def _report_rows(report: Mapping[str, Any], report_path: Path) -> dict[str, Mapping[str, Any]]:
    rows = report.get("trajectories")
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError(f"{report_path} must describe exactly 50 trajectories")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("trajectory"), str):
            raise ValueError(f"{report_path} has an invalid trajectory record")
        name = row["trajectory"]
        if name in result or type(row.get("terminal_task_success")) is not bool:
            raise ValueError(f"{report_path} has a duplicate or invalid {name}")
        result[name] = row
    return result


def _gate5_partition(
    task: Mapping[str, Any], identities: set[str], suite: str, stem: str
) -> tuple[set[str], set[str]]:
    partitions = {}
    for key in (
        "recovered_demo_ids",
        "unrecovered_demo_ids",
        "not_attempted_demo_ids",
    ):
        names = task.get(key)
        if (
            not isinstance(names, list)
            or any(not isinstance(name, str) for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(f"Gate 5 summary has invalid {key} for {suite}/{stem}")
        partitions[key] = set(names)
    target_groups = tuple(partitions.values())
    targets = set().union(*target_groups)
    if sum(map(len, target_groups)) != len(targets):
        raise ValueError(f"Gate 5 summary has duplicate targets for {suite}/{stem}")
    unexpected = targets - identities
    if unexpected:
        raise ValueError(
            f"Gate 5 summary has unexpected targets for {suite}/{stem}: "
            f"{sorted(unexpected, key=_demo_number)}"
        )
    raw_retained = identities - targets
    if task.get("target_identity_count") != len(targets) or task.get(
        "raw_retained_identity_count"
    ) != len(raw_retained):
        raise ValueError(f"Gate 5 identity counts differ for {suite}/{stem}")
    return raw_retained, targets


def _same_task_metadata(raw: h5py.Group, recovered: h5py.Group) -> None:
    for key in ("bddl_file_name", "env_args"):
        if key not in raw.attrs or key not in recovered.attrs:
            raise ValueError(f"Missing task metadata attribute: {key}")
    if _decode(raw.attrs["bddl_file_name"]) != _decode(
        recovered.attrs["bddl_file_name"]
    ):
        raise ValueError("Raw and recovered BDDL metadata differ")
    if json.loads(_decode(raw.attrs["env_args"])) != json.loads(
        _decode(recovered.attrs["env_args"])
    ):
        raise ValueError("Raw and recovered environment metadata differ")


def _recovery_tasks(
    recovery_roots: Sequence[Path],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Collect completed task records from suite, wrapper, or task summaries."""
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    sources: dict[tuple[str, str], Path] = {}
    for recovery_root in recovery_roots:
        recovery_root = Path(recovery_root).resolve()
        if not recovery_root.is_dir():
            raise FileNotFoundError(recovery_root)
        for summary_path in sorted(recovery_root.rglob("summary.json")):
            if {"work", "archive"} & set(
                summary_path.relative_to(recovery_root).parts[:-1]
            ):
                continue
            summary = _read_json(summary_path)
            if isinstance(summary.get("case_id"), str):
                tasks = [summary]
            else:
                tasks = summary.get("tasks", [])
            if not isinstance(tasks, list):
                raise ValueError(f"{summary_path} has an invalid tasks field")
            for task in tasks:
                if not isinstance(task, dict):
                    raise ValueError(f"{summary_path} has an invalid task record")
                suite = task.get("suite")
                stem = task.get("task_stem")
                if suite not in SUPPORTED_SUITES or not isinstance(stem, str):
                    continue
                key = (suite, stem)
                if key in result and result[key] != task:
                    raise ValueError(
                        f"Conflicting recovery summaries for {suite}/{stem}: "
                        f"{sources[key]} and {summary_path}"
                    )
                result[key] = task
                sources[key] = summary_path
    return result


def _recovery_sources(
    recovery_roots: Sequence[Path],
) -> dict[tuple[str, str, str], Path]:
    """Resolve every accepted recovery from suite summaries and final rounds."""
    result: dict[tuple[str, str, str], Path] = {}

    def add(key: tuple[str, str, str], path: Path) -> None:
        path = path.resolve()
        if key in result and result[key] != path:
            raise ValueError(
                f"Conflicting recovery sources for {key}: {result[key]} and {path}"
            )
        result[key] = path

    for (suite, stem), task in _recovery_tasks(recovery_roots).items():
        recovered = task.get("recovered_demo_ids")
        merged = task.get("merged_hdf5")
        if not isinstance(recovered, list) or bool(recovered) != bool(merged):
            raise ValueError(f"Invalid accepted recovery summary for {suite}/{stem}")
        for demo in recovered:
            if not isinstance(demo, str):
                raise ValueError(f"Invalid recovered identity for {suite}/{stem}")
            add((suite, stem, demo), Path(merged))

    for root in recovery_roots:
        summary_path = Path(root).resolve() / "summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        results = summary.get("results")
        if results is None:
            continue
        if summary.get("status") != "COMPLETE" or not isinstance(results, list):
            raise ValueError(f"{summary_path} is not a completed recovery round")
        for row in results:
            if not isinstance(row, dict) or row.get("status") != "recovered":
                continue
            suite, stem, demo = (
                row.get("suite"),
                row.get("task_stem"),
                row.get("demo"),
            )
            if (
                suite not in SUPPORTED_SUITES
                or not isinstance(stem, str)
                or not isinstance(demo, str)
                or not isinstance(row.get("merged_hdf5"), str)
            ):
                raise ValueError(f"{summary_path} has an invalid recovered result")
            add((suite, stem, demo), Path(row["merged_hdf5"]))
    return result


def _materialize_task(
    suite: str,
    raw_hdf5: Path,
    report_path: Path,
    gate5_task: Mapping[str, Any],
    recovery_sources: Mapping[tuple[str, str, str], Path],
    output_hdf5: Path,
    allow_incomplete: bool = False,
) -> Mapping[str, Any]:
    report = _read_json(report_path)
    rows = _report_rows(report, report_path)
    stem = raw_hdf5.stem
    raw_terminal_successes = {
        name for name, row in rows.items() if row["terminal_task_success"]
    }
    if output_hdf5.exists():
        raise FileExistsError(output_hdf5)
    temporary = output_hdf5.with_name(f".{output_hdf5.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)

    with h5py.File(raw_hdf5, "r") as raw_file:
        raw_data = raw_file["data"]
        if set(raw_data) != set(rows):
            raise ValueError(f"{raw_hdf5} demos differ from its conversion report")
        raw_retained, recovery_targets = _gate5_partition(
            gate5_task, set(rows), suite, stem
        )
        raw_names = sorted(raw_retained, key=_demo_number)
        accepted_recoveries = set(gate5_task["recovered_demo_ids"])
        unresolved_names = sorted(
            recovery_targets - accepted_recoveries, key=_demo_number
        )
        recovered_names = sorted(
            accepted_recoveries if allow_incomplete else recovery_targets,
            key=_demo_number,
        )
        task_sources = {
            demo: recovery_sources[(suite, stem, demo)]
            for demo in recovered_names
            if (suite, stem, demo) in recovery_sources
        }
        if set(task_sources) != set(recovered_names):
            missing = sorted(set(recovered_names) - set(task_sources), key=_demo_number)
            raise ValueError(f"{raw_hdf5} is missing accepted recoveries: {missing}")
        canonical_wrist = _wrist_key(raw_data[min(rows, key=_demo_number)]["obs"])
        with h5py.File(temporary, "w") as output_file:
            output = output_file.create_group("data")
            for key, value in raw_data.attrs.items():
                if not key.startswith("aubo_"):
                    output.attrs[key] = value
            total_samples = 0
            for name in raw_names:
                total_samples += _copy_demo(
                    raw_data, name, output, canonical_wrist, "raw_gate5_retained"
                )
            grouped: dict[Path, list[str]] = defaultdict(list)
            for name, path in task_sources.items():
                grouped[path].append(name)
            for recovered_hdf5, names in sorted(
                grouped.items(), key=lambda item: str(item[0])
            ):
                if not recovered_hdf5.is_file():
                    raise FileNotFoundError(recovered_hdf5)
                with h5py.File(recovered_hdf5, "r") as recovered_file:
                    recovered_data = recovered_file["data"]
                    _same_task_metadata(raw_data, recovered_data)
                    if not bool(
                        recovered_data.attrs.get(
                            "state_action_observation_qualified", False
                        )
                    ):
                        raise ValueError(f"{recovered_hdf5} is not quality-qualified")
                    if set(recovered_data) != set(names):
                        raise ValueError(
                            f"{recovered_hdf5} demos differ from its recovery summary"
                        )
                    for name in sorted(names, key=_demo_number):
                        total_samples += _copy_demo(
                            recovered_data,
                            name,
                            output,
                            canonical_wrist,
                            "recovered_gate5_target",
                        )
            output.attrs["total"] = total_samples
            output.attrs["num_demos"] = len(output)
            output.attrs["demo_count"] = len(output)
            output.attrs["materialization_method"] = (
                "gate5_raw_retained_plus_mimicgen_recovery"
            )
            output.attrs["contains_only_terminal_success"] = True
            output.attrs["raw_terminal_success_count"] = len(raw_terminal_successes)
            output.attrs["raw_retained_count"] = len(raw_names)
            output.attrs["recovered_identity_count"] = len(recovered_names)
            output.attrs["unrecovered_identity_count"] = len(unresolved_names)
            output.attrs["comparable_full_dataset_ready"] = len(output) == 50
            output.attrs["mixed_generation_provenance"] = True
    os.replace(temporary, output_hdf5)

    expected = set(raw_names) | set(recovered_names)
    with h5py.File(output_hdf5, "r") as output_file:
        output = output_file["data"]
        if set(output) != expected:
            raise RuntimeError(f"{output_hdf5} has wrong materialized identities")
        for name in output:
            _validate_demo(output[name], canonical_wrist)

    return {
        "task_stem": raw_hdf5.stem,
        "output_hdf5": str(output_hdf5),
        "raw_terminal_success_count": len(raw_terminal_successes),
        "raw_retained_count": len(raw_names),
        "recovered_identity_count": len(recovered_names),
        "final_demo_count": len(expected),
        "recovered_demo_ids": recovered_names,
        "recovery_source_hdf5s": sorted({str(path) for path in task_sources.values()}),
        "unrecovered_demo_ids": unresolved_names,
        "canonical_wrist_image_key": canonical_wrist,
    }


def _prepare_inputs(
    raw_root: Path,
    recovery_roots: Sequence[Path],
    suites: Sequence[str],
    allow_incomplete: bool = False,
) -> tuple[list[tuple[str, Path, list[Path]]], dict, dict, Mapping[str, Any]]:
    raw_root = raw_root.resolve()
    recovery_roots = [Path(path).resolve() for path in recovery_roots]
    if raw_root.name != RAW_ROOT_BASENAME:
        raise ValueError(f"Authoritative raw root must be named {RAW_ROOT_BASENAME}")
    if not recovery_roots:
        raise ValueError("At least one recovery root is required")
    if len(set(suites)) != len(suites):
        raise ValueError("Suites must not be repeated")

    recovery_tasks = _recovery_tasks(recovery_roots)
    recovery_sources = _recovery_sources(recovery_roots)
    suite_inputs = []
    all_identities = set()
    raw_terminal_successes = set()
    raw_retained = set()
    required_recoveries = set()
    unresolved_identities = set()
    suffix = ".conversion_report.json"
    for suite in suites:
        raw_suite = raw_root / suite
        reports = sorted(raw_suite.glob("*.conversion_report.json"))
        if len(reports) != 10:
            raise ValueError(f"{suite} must contain exactly ten conversion reports")
        for report_path in reports:
            stem = report_path.name[: -len(suffix)]
            rows = _report_rows(_read_json(report_path), report_path)
            task_ids = {(suite, stem, demo) for demo in rows}
            task_terminal = {
                (suite, stem, demo)
                for demo, row in rows.items()
                if row["terminal_task_success"]
            }
            raw_hdf5 = raw_suite / f"{stem}.hdf5"
            with h5py.File(raw_hdf5, "r") as raw_file:
                raw_data = raw_file["data"]
                if set(raw_data) != set(rows):
                    raise ValueError(
                        f"{raw_hdf5} demos differ from its conversion report"
                    )
                task = recovery_tasks.get((suite, stem))
                if task is None:
                    raise ValueError(f"Missing Gate 5 summary for {suite}/{stem}")
                retained_names, target_names = _gate5_partition(
                    task, set(rows), suite, stem
                )
                for name in retained_names:
                    _validate_demo(raw_data[name])
            task_retained = {(suite, stem, demo) for demo in retained_names}
            accepted_names = set(task["recovered_demo_ids"])
            unresolved_names = target_names - accepted_names
            selected_names = accepted_names if allow_incomplete else target_names
            task_targets = {(suite, stem, demo) for demo in selected_names}
            task_unresolved = {(suite, stem, demo) for demo in unresolved_names}
            all_identities.update(task_ids)
            raw_terminal_successes.update(task_terminal)
            raw_retained.update(task_retained)
            required_recoveries.update(task_targets)
            unresolved_identities.update(task_unresolved)
        suite_inputs.append((suite, raw_suite, reports))

    selected_suites = set(suites)
    selected_recoveries = {
        key: path
        for key, path in recovery_sources.items()
        if key[0] in selected_suites
    }
    if set(selected_recoveries) != required_recoveries:
        missing = sorted(required_recoveries - set(selected_recoveries))
        unexpected = sorted(set(selected_recoveries) - required_recoveries)
        raise ValueError(
            f"Recovery resolution is incomplete; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )
    grouped: dict[Path, set[str]] = defaultdict(set)
    for (_, _, demo), path in selected_recoveries.items():
        grouped[path].add(demo)
    for path, demos in grouped.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as recovered_file:
            data = recovered_file["data"]
            if not bool(data.attrs.get("state_action_observation_qualified", False)):
                raise ValueError(f"{path} is not quality-qualified")
            if set(data) != demos:
                raise ValueError(f"{path} does not contain its resolved identities")
            for name in data:
                _validate_demo(data[name])

    if set(suites) == set(SUPPORTED_SUITES) and len(all_identities) != 2000:
        raise ValueError(
            f"Full-dataset preflight found {len(all_identities)} identities, expected 2000"
        )

    summary = {
        "status": "PREFLIGHT_PASSED",
        "suite_count": len(suites),
        "task_count": sum(len(reports) for _, _, reports in suite_inputs),
        "identity_count": len(all_identities),
        "raw_terminal_success_count": len(raw_terminal_successes),
        "raw_retained_count": len(raw_retained),
        "recovered_identity_count": len(required_recoveries),
        "recovery_hdf5_count": len(grouped),
        "missing_identity_count": len(unresolved_identities),
        "duplicate_identity_count": 0,
    }
    return suite_inputs, recovery_tasks, selected_recoveries, summary


def materialize(
    raw_root: Path,
    recovery_roots: Sequence[Path],
    output_root: Path,
    suites: Sequence[str],
    preflight_only: bool = False,
    allow_incomplete: bool = False,
) -> Mapping[str, Any]:
    raw_root = raw_root.resolve()
    recovery_roots = [Path(path).resolve() for path in recovery_roots]
    output_root = output_root.resolve()
    if output_root.name != OUTPUT_ROOT_BASENAME:
        raise ValueError(f"Output root must be named {OUTPUT_ROOT_BASENAME}")
    if output_root.exists():
        raise FileExistsError(output_root)
    suite_inputs, recovery_tasks, recovery_sources, preflight = _prepare_inputs(
        raw_root, recovery_roots, suites, allow_incomplete=allow_incomplete
    )
    print(
        f"[PREFLIGHT] tasks={preflight['task_count']} "
        f"identities={preflight['identity_count']} "
        f"raw_retained={preflight['raw_retained_count']} "
        f"recovered={preflight['recovered_identity_count']}",
        flush=True,
    )
    if preflight_only:
        return preflight

    output_root.mkdir()
    suite_results = []
    suffix = ".conversion_report.json"
    for suite, raw_suite, reports in suite_inputs:
        output_suite = output_root / suite
        output_suite.mkdir()
        task_results = []
        for report_path in reports:
            stem = report_path.name[: -len(suffix)]
            raw_hdf5 = raw_suite / f"{stem}.hdf5"
            result = _materialize_task(
                suite,
                raw_hdf5,
                report_path,
                recovery_tasks[(suite, stem)],
                recovery_sources,
                output_suite / f"{stem}.hdf5",
                allow_incomplete=allow_incomplete,
            )
            task_results.append(result)
            print(
                f"[TASK] suite={suite} task={stem} "
                f"raw_retained={result['raw_retained_count']} "
                f"recovered={result['recovered_identity_count']} "
                f"final={result['final_demo_count']}",
                flush=True,
            )
        suite_result = {
            "suite": suite,
            "task_count": len(task_results),
            "raw_terminal_success_count": sum(
                row["raw_terminal_success_count"] for row in task_results
            ),
            "raw_retained_count": sum(
                row["raw_retained_count"] for row in task_results
            ),
            "recovered_identity_count": sum(
                row["recovered_identity_count"] for row in task_results
            ),
            "final_demo_count": sum(row["final_demo_count"] for row in task_results),
            "unrecovered_identity_count": sum(
                len(row["unrecovered_demo_ids"]) for row in task_results
            ),
            "tasks": task_results,
        }
        suite_results.append(suite_result)
        print(
            f"[SUITE] suite={suite} "
            f"raw_retained={suite_result['raw_retained_count']} "
            f"recovered={suite_result['recovered_identity_count']} "
            f"final={suite_result['final_demo_count']} "
            f"unrecovered={suite_result['unrecovered_identity_count']}",
            flush=True,
        )

    report = {
        "status": "MATERIALIZED_RECOVERED_DATASET",
        "authoritative_raw_root": str(raw_root),
        "recovery_roots": [str(path) for path in recovery_roots],
        "output_root": str(output_root),
        "suites": suite_results,
        "suite_count": len(suite_results),
        "task_count": sum(row["task_count"] for row in suite_results),
        "final_demo_count": sum(row["final_demo_count"] for row in suite_results),
        "raw_terminal_success_count": sum(
            row["raw_terminal_success_count"] for row in suite_results
        ),
        "raw_retained_count": sum(
            row["raw_retained_count"] for row in suite_results
        ),
        "recovered_identity_count": sum(
            row["recovered_identity_count"] for row in suite_results
        ),
        "unrecovered_identity_count": sum(
            row["unrecovered_identity_count"] for row in suite_results
        ),
        "raw_root_modified": False,
        "recovery_roots_modified": False,
        "selected_source_arrays_compared_directly": True,
        "incomplete_dataset_approved": allow_incomplete,
        "ready_for_training": False,
    }
    complete = (
        report["suite_count"] == 4
        and report["task_count"] == 40
        and report["final_demo_count"] == 2000
        and report["unrecovered_identity_count"] == 0
    )
    accounted_for = (
        report["suite_count"] == 4
        and report["task_count"] == 40
        and report["final_demo_count"] + report["unrecovered_identity_count"] == 2000
    )
    if not complete and not (allow_incomplete and accounted_for):
        raise RuntimeError(f"Final recovered dataset totals are invalid: {report}")
    report["ready_for_training"] = True
    report["comparable_full_dataset_ready"] = complete
    _atomic_json(output_root / "materialization_report.json", report)
    print(
        f"[END] suites={len(suite_results)} final={report['final_demo_count']} "
        f"recovered={report['recovered_identity_count']} "
        f"unrecovered={report['unrecovered_identity_count']}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--recovery-root", type=Path, action="append", required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUPPORTED_SUITES,
        required=True,
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    arguments = parser.parse_args()
    materialize(
        arguments.raw_root,
        arguments.recovery_root,
        arguments.output_root,
        arguments.suite,
        preflight_only=arguments.preflight_only,
        allow_incomplete=arguments.allow_incomplete,
    )


if __name__ == "__main__":
    main()
