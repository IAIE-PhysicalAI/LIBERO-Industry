"""Audit embodiment-specific init files and optionally reset-smoke them."""

import argparse
import hashlib
import importlib
import json
import os
import re
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from libero.libero import benchmark
from libero.libero.benchmark.industry_benchmark_config import scene_for_task
from libero.libero.utils.init_state_utils import (
    apply_init_state_profile,
    available_init_state_profiles,
    get_init_state_profile,
    load_existing_state_pair,
    resolve_init_state_path,
    resolve_init_states_root,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = REPO_ROOT / "libero" / "libero"
DEFAULT_BDDL_ROOT = LIBERO_ROOT / "bddl_files"
DEFAULT_INIT_ROOT = LIBERO_ROOT / "init_files"


def report_path(path):
    """Keep paths inside the repository portable in generated reports."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def task_family(task_name):
    match = re.match(r"^(D2D|D2S|S2D)_", task_name.upper())
    return match.group(1) if match is not None else None


def task_level(task_name):
    match = re.match(r"^(?:D2D|D2S|S2D)_LV([123])_", task_name.upper())
    if match is None:
        return None
    return int(match.group(1))


def select_tasks(task_suite, levels=None, families=None, task_names=None):
    tasks = [task_suite.get_task(task_id) for task_id in range(task_suite.n_tasks)]
    requested = set(task_names or ())
    missing = requested - {task.name for task in tasks}
    if missing:
        raise ValueError("Tasks are not in the selected suite: {}".format(", ".join(sorted(missing))))

    selected_levels = set(levels or ())
    selected_families = set(families or ())
    return [
        task
        for task in tasks
        if (not requested or task.name in requested)
        and (not selected_levels or task_level(task.name) in selected_levels)
        and (not selected_families or task_family(task.name) in selected_families)
    ]


def stable_seed(base_seed, task_name):
    offset = int(hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + offset) % (2**32)


def write_json_atomic(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Validate init-state pairs for one embodiment without changing them."
    )
    parser.add_argument("--suite", required=True)
    parser.add_argument(
        "--embodiment",
        choices=available_init_state_profiles(),
        default="panda",
    )
    parser.add_argument("--level", type=int, nargs="+", choices=(1, 2, 3))
    parser.add_argument(
        "--family", nargs="+", choices=("D2D", "D2S", "S2D")
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Validate one exact task name. Repeat to select multiple tasks.",
    )
    parser.add_argument("--bddl-root", type=Path, default=DEFAULT_BDDL_ROOT)
    parser.add_argument(
        "--init-root",
        type=Path,
        help="Final profile root containing physical suite directories.",
    )
    parser.add_argument(
        "--target-states",
        type=int,
        help="Require exactly this many states in every selected file.",
    )
    parser.add_argument(
        "--reset-smoke",
        action="store_true",
        help="Instantiate each task and restore states without rendering.",
    )
    parser.add_argument(
        "--states-per-task",
        type=int,
        default=1,
        help="Number of saved states to restore per task during reset smoke (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()

    if args.target_states is not None and args.target_states <= 0:
        parser.error("--target-states must be positive")
    if args.states_per_task <= 0:
        parser.error("--states-per-task must be positive")

    bddl_root = args.bddl_root.expanduser().resolve()
    init_root = resolve_init_states_root(
        DEFAULT_INIT_ROOT,
        embodiment=args.embodiment,
        explicit_init_root=args.init_root,
    )
    profile = get_init_state_profile(args.embodiment)

    benchmark_module = benchmark
    if args.suite.startswith("libero_ablation_") and bddl_root != DEFAULT_BDDL_ROOT.resolve():
        matches = [path for path in bddl_root.rglob(args.suite) if path.is_dir()]
        if len(matches) != 1:
            parser.error(
                "Expected one '{}' directory under {}, found {}".format(
                    args.suite, bddl_root, len(matches)
                )
            )
        staged_tasks = sorted(path.stem for path in matches[0].glob("*.bddl"))
        if not staged_tasks:
            parser.error("No staged BDDLs found in {}".format(matches[0]))
        benchmark_module.libero_task_map[args.suite] = staged_tasks
        benchmark_module = importlib.reload(benchmark_module)

    benchmark_dict = benchmark_module.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        parser.error("Suite '{}' is not registered".format(args.suite))
    task_suite = benchmark_dict[args.suite]()
    try:
        selected_tasks = select_tasks(
            task_suite,
            levels=args.level,
            families=args.family,
            task_names=args.task,
        )
    except ValueError as error:
        parser.error(str(error))
    if not selected_tasks:
        parser.error("No tasks matched the requested selectors")

    report = {
        "schema_version": 1,
        "suite": args.suite,
        "embodiment": profile.name,
        "profile": profile.manifest_dict(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "bddl_root": report_path(bddl_root),
        "init_root": report_path(init_root),
        "reset_smoke": bool(args.reset_smoke),
        "tasks": [],
    }
    failures = 0

    manifest_path = init_root / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as manifest_stream:
            manifest = json.load(manifest_stream)
        if manifest.get("embodiment") != profile.name:
            parser.error(
                "Manifest embodiment '{}' does not match requested '{}'".format(
                    manifest.get("embodiment"), profile.name
                )
            )
    elif profile.name != "panda":
        parser.error("Embodiment manifest does not exist: {}".format(manifest_path))

    for task_index, task in enumerate(selected_tasks, start=1):
        print(
            "[VALIDATE {}/{}] {}".format(
                task_index, len(selected_tasks), task.name
            )
        )
        task_report = {
            "task": task.name,
            "problem_folder": task.problem_folder,
        }
        env = None
        try:
            bddl_path = bddl_root / task.problem_folder / task.bddl_file
            init_path = resolve_init_state_path(
                init_root,
                task.problem_folder,
                task.name,
                pruned=False,
                embodiment=profile.name,
            )
            pruned_path = resolve_init_state_path(
                init_root,
                task.problem_folder,
                task.name,
                pruned=True,
                embodiment=profile.name,
            )
            for required_path in (bddl_path, init_path, pruned_path):
                if not required_path.is_file():
                    raise FileNotFoundError("Required file does not exist: {}".format(required_path))

            pruned_states = load_existing_state_pair(init_path, pruned_path)
            if (
                args.target_states is not None
                and pruned_states.shape[0] != args.target_states
            ):
                raise ValueError(
                    "Expected {} states, found {}".format(
                        args.target_states, pruned_states.shape[0]
                    )
                )

            task_report.update(
                {
                    "state_count": int(pruned_states.shape[0]),
                    "state_width": int(pruned_states.shape[1]),
                }
            )

            if args.reset_smoke:
                from libero.libero.envs import OffScreenRenderEnv

                env_args = {
                    "bddl_file_name": str(bddl_path),
                    "has_renderer": False,
                    "has_offscreen_renderer": False,
                    "use_camera_obs": False,
                    "control_freq": 20,
                    "renderer": "mujoco",
                }
                env_args = apply_init_state_profile(env_args, profile.name)
                family = task_family(task.name)
                if family is not None:
                    level = task_level(task.name)
                    env_args["object_sampling"] = "uniform" if level == 1 else "random"
                    try:
                        env_args["set_scene"] = scene_for_task(task.name, seed=args.seed)
                    except ValueError:
                        pass

                env = OffScreenRenderEnv(**env_args)
                env.seed(stable_seed(args.seed, task.name))
                env.reset()
                live_width = np.asarray(env.get_init_state()).reshape(-1).shape[0]
                if live_width != pruned_states.shape[1]:
                    raise ValueError(
                        "Saved state width {} does not match live simulator width {}".format(
                            pruned_states.shape[1], live_width
                        )
                    )
                restored_count = min(args.states_per_task, pruned_states.shape[0])
                for state_index in range(restored_count):
                    env.set_init_state(pruned_states[state_index])
                    restored = np.asarray(env.get_init_state()).reshape(-1)
                    if restored.shape[0] != live_width or not np.isfinite(restored).all():
                        raise ValueError(
                            "Restored state {} is invalid".format(state_index)
                        )
                task_report["reset_states_checked"] = restored_count

            task_report["status"] = "passed"
        except Exception as error:
            failures += 1
            task_report.update({"status": "failed", "error": str(error)})
            print("[ERROR] {}".format(error))
            traceback.print_exc()
        finally:
            report["tasks"].append(task_report)
            if env is not None:
                try:
                    env.close()
                except Exception as close_error:
                    print("[WARNING] Failed to close environment: {}".format(close_error))

    report["summary"] = {
        "selected_tasks": len(selected_tasks),
        "passed_tasks": len(selected_tasks) - failures,
        "failed_tasks": failures,
    }
    output_report_path = args.report_file or (
        init_root / "validation_report_{}.json".format(args.suite)
    )
    write_json_atomic(report, output_report_path)
    print(
        "[SUMMARY] {}/{} tasks passed; report: {}".format(
            len(selected_tasks) - failures, len(selected_tasks), output_report_path
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
