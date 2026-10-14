import os
import re
import json
import argparse
import hashlib
import platform
import subprocess
import sys
import tempfile
import traceback
import numpy as np
import importlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.init_state_utils import (
    apply_init_state_profile,
    as_init_state_array,
    available_init_state_profiles,
    filesystem_path,
    get_init_state_profile,
    init_state_filename,
    load_existing_state_pair,
    resolve_init_states_root,
)

# ==============================================================================
# CONFIGURATION VARIABLES
# Edit these directly for dry runs, bypassing physics, or auto-fixing files.
# ==============================================================================
SEED = 42
TARGET_STATES = 20       # Set to an int to override target states (default 50 for industry, 20 otherwise)

# ==============================================================================
# GLOBALS & PATH SETUPS
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
LIBERO_ROOT = REPO_ROOT / "libero" / "libero"
SEARCH_DIR = LIBERO_ROOT / "bddl_files"


def is_ablation_suite(suite_name):
    """Return whether a suite uses the legacy ablation init-state policy."""
    return str(suite_name).strip().lower().startswith("libero_ablation_")


def is_ablation_task(task_name):
    """Return whether a task uses the canonical ABL ablation naming scheme."""
    return Path(task_name).stem.upper().startswith("ABL")


def suite_tasks_from_root(task_suite_name, search_dir):
    """Return one staged suite's physical BDDL stems without editing live maps."""
    matches = [path for path in Path(search_dir).rglob(task_suite_name) if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            "Expected one '{}' directory under {}, found {}".format(
                task_suite_name, search_dir, len(matches)
            )
        )
    tasks = sorted(path.stem for path in matches[0].glob("*.bddl"))
    if not tasks:
        raise ValueError("No BDDLs found in {}".format(matches[0]))
    return tasks

def sync_task_map(task_suite_name, search_dir, dry_run=False):
    """
    Validates libero_suite_task_map.py against physical .bddl files.
    Identifies missing/ghost tasks and surgically prunes them from the dictionary.

    Returns ``(success, changed)``. In dry-run mode, ``changed`` reports that
    the map would be updated even though the source file is left untouched.
    """
    # 1. Locate physical .bddl files (The Ground Truth)
    suite_dir = None
    for root, dirs, files in os.walk(search_dir):
        if task_suite_name in dirs:
            suite_dir = os.path.join(root, task_suite_name)
            break
            
    if not suite_dir:
        print(f"[WARNING] Task suite directory '{task_suite_name}' not found under {search_dir}.")
        return False, False
        
    actual_tasks = [f.replace(".bddl", "") for f in os.listdir(suite_dir) if f.endswith(".bddl")]
    actual_tasks.sort()
    if not actual_tasks:
        print(f"[WARNING] No .bddl files found in {suite_dir}; refusing to empty the task map.")
        return False, False
    
    # 2. Extract current memory state from libero_suite_task_map.py
    task_map_path = os.path.join(LIBERO_ROOT, "benchmark", "libero_suite_task_map.py")
    current_tasks = []
    content = ""
    
    if os.path.exists(task_map_path):
        temp_namespace = {}
        with open(task_map_path, "r") as f:
            content = f.read()
            try:
                exec(content, temp_namespace)
                current_map = temp_namespace.get("libero_task_map", {})
                current_tasks = current_map.get(task_suite_name, [])
            except Exception as e:
                print(f"[ERROR] Failed to parse existing task map: {e}")
                return False, False
    else:
        content = "libero_task_map = {}\n"

    # 3. Analytical Set Comparison
    set_actual = set(actual_tasks)
    set_current = set(current_tasks)
    
    to_remove = set_current - set_actual
    to_add = set_actual - set_current
    
    if not to_remove and not to_add:
        print(f"[INFO] Task map is perfectly synced for '{task_suite_name}'. No pruning needed.")
        return True, False

    # 4. Actionable Readout
    print(f"\n[SYNC REQUIRED] Desynchronization detected in '{task_suite_name}':")
    if to_remove:
        print(f"  [-] Ghost tasks detected (No .bddl found). Pruning {len(to_remove)} entries:")
        for t in sorted(list(to_remove)): print(f"      - {t}")
    if to_add:
        print(f"  [+] New tasks detected. Injecting {len(to_add)} entries:")
        for t in sorted(list(to_add)): print(f"      + {t}")

    # 5. Dry Run Gatekeeper
    if dry_run:
        print("\n[DRY RUN] Bypassing matrix write. No changes were made to libero_suite_task_map.py.")
        return True, True

    # 6. Format the new strictly-aligned list
    new_list_str = f'"{task_suite_name}": [\n'
    for t in actual_tasks:
        new_list_str += f'        "{t}",\n'
    new_list_str += '    ]'
    
    # Regex search to replace the entire array block for this suite
    pattern = rf'"{task_suite_name}"\s*:\s*\[.*?\]'
    
    if re.search(pattern, content, flags=re.DOTALL):
        updated_content = re.sub(pattern, new_list_str, content, count=1, flags=re.DOTALL)
    else:
        # If the suite didn't exist at all, append it safely
        last_brace_idx = content.rfind('}')
        if last_brace_idx != -1:
            updated_content = content[:last_brace_idx] + f'    {new_list_str},\n' + content[last_brace_idx:]
        else:
            updated_content = content + f"\nlibero_task_map = {{\n    {new_list_str}\n}}\n"

    # 7. Execute Write
    try:
        atomic_text_write(updated_content, task_map_path)
        print(f"\n[SUCCESS] Matrix updated. '{task_suite_name}' now contains exactly {len(actual_tasks)} verified tasks.")
        return True, True
    except Exception as e:
        print(f"\n[ERROR] Matrix write failure: {e}")
        return False, False

def count_task_objects(bddl_path):
    """Return target and source-conveyor-confounder counts from a BDDL file."""
    path = Path(bddl_path)
    text = path.read_text(encoding="utf-8")

    init_match = re.search(
        r"\(\s*:init\b(.*?)(?=\(\s*:goal\b)", text, flags=re.DOTALL | re.IGNORECASE
    )
    goal_match = re.search(r"\(\s*:goal\b(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if init_match is None or goal_match is None:
        raise ValueError("{} is missing an :init or :goal section".format(path))

    predicate_pattern = r"On|ReleasedOn|AlignedOn|Routed|DynamicOn"
    target_instances = set(
        re.findall(
            r"\(\s*(?:{})\s+([^\s\)]+)\s+[^\s\)]+\s*\)".format(predicate_pattern),
            goal_match.group(1),
            flags=re.IGNORECASE,
        )
    )
    source_instances = set()
    for instance, region in re.findall(
        r"\(\s*On\s+([^\s\)]+)\s+([^\s\)]+)\s*\)",
        init_match.group(1),
        flags=re.IGNORECASE,
    ):
        if region.startswith("conveyor_belt_1_") and instance not in target_instances:
            source_instances.add(instance)

    if not target_instances:
        raise ValueError("{} has no target objects in :goal".format(path))
    return len(target_instances), len(source_instances)


def get_industry_timing(bddl_path):
    """Read the conveyor timing parameters actually used by an Industry BDDL."""
    path = Path(bddl_path)
    text = path.read_text(encoding="utf-8")
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"

    at_speed = re.search(
        r"\(\s*AtSpeed\s+conveyor_belt_1\s+({0})\s+steady\s+"
        r"takt_({0})\s+offset_({0})\s*\)".format(number),
        text,
        flags=re.IGNORECASE,
    )
    load_slot = re.search(
        r"\(\s*LoadSlot\s+\S+\s+([^\s\)]+)\s+([xy])\s+"
        r"pulse\s+({0})\s+service_({0})\s*\)".format(number),
        text,
        flags=re.IGNORECASE,
    )

    return {
        "at_speed": (
            {
                "speed": float(at_speed.group(1)),
                "takt": float(at_speed.group(2)),
                "offset": float(at_speed.group(3)),
            }
            if at_speed is not None
            else None
        ),
        "load_slot": (
            {
                "groove_type": load_slot.group(1).lower(),
                "axis": load_slot.group(2).lower(),
                "speed": float(load_slot.group(3)),
                "service": float(load_slot.group(4)),
            }
            if load_slot is not None
            else None
        ),
    }


def get_takt_steps(task_name, bddl_path, suite_name=None):
    """Return an Industry episode horizon, or ``None`` for ablation tasks.

    The BDDL is the source of truth for conveyor speed, takt, offset, and
    service. Task names are deliberately not parsed for speed tokens because
    fractional tuning suffixes such as ``_v3_t4`` are not part of the base
    task identifier.
    """
    if is_ablation_suite(suite_name) or is_ablation_task(task_name):
        return None

    fallback_steps = 400
    try:
        from libero.libero.benchmark.industry_benchmark_config import (
            takt_steps_for_bddl,
        )

        return takt_steps_for_bddl(Path(task_name).stem, bddl_path)
    except Exception as error:
        print(
            "[WARNING] Could not calculate Industry takt horizon for '{}': {}. "
            "Using {}-step compatibility fallback.".format(task_name, error, fallback_steps)
        )
        return fallback_steps

def get_scene_from_tracker(task_name, seed=SEED):
    """Return the shared deterministic visual policy for an Industry task."""
    from libero.libero.benchmark.industry_benchmark_config import scene_for_task

    try:
        return scene_for_task(task_name, seed=seed)
    except ValueError:
        return None


def get_task_level(task_name):
    """Return the numeric Industry level encoded in a canonical task name."""
    match = re.match(r"^(?:D2D|D2S|S2D)_LV([123])_", task_name.upper())
    if match is None:
        raise ValueError(
            "Expected an Industry task name such as D2S_LV1_001_..., got '{}'".format(
                task_name
            )
        )
    return int(match.group(1))


def get_task_family(task_name):
    match = re.match(r"^(D2D|D2S|S2D)_", task_name.upper())
    return match.group(1) if match is not None else None


def is_industry_task(task_name):
    return get_task_family(task_name) is not None


def build_init_env_args(bddl_path, profile_name, suite_name, task_name, seed=SEED):
    """Build reset-only environment arguments for one init-state task."""
    env_args = {
        "bddl_file_name": str(bddl_path),
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "control_freq": 20,
        "renderer": "mujoco",
    }
    env_args = apply_init_state_profile(env_args, profile_name)

    # Ablation init states intentionally use the BDDL/default reset policy.
    # In particular, there is no takt horizon, scene override, or sampling
    # override, even if a future ablation task uses an Industry-like name.
    if is_ablation_suite(suite_name):
        return env_args

    if is_industry_task(task_name):
        parts = task_name.split("_")
        level = parts[1] if len(parts) > 1 else "LV1"
        env_args["object_sampling"] = "uniform" if level == "LV1" else "random"
        takt_steps = get_takt_steps(task_name, bddl_path, suite_name=suite_name)
        if takt_steps is not None:
            env_args["takt_steps"] = takt_steps
        scene_cfg = get_scene_from_tracker(task_name, seed=seed)
        if scene_cfg:
            env_args["set_scene"] = scene_cfg

    return env_args


def select_tasks(task_suite, levels=None, families=None, task_names=None):
    """Return deterministic benchmark tasks matching the requested selectors."""

    tasks = [task_suite.get_task(task_id) for task_id in range(task_suite.n_tasks)]
    available_names = {task.name for task in tasks}
    requested_names = set(task_names or ())
    missing = sorted(requested_names - available_names)
    if missing:
        raise ValueError("Requested task(s) are not in the suite: {}".format(", ".join(missing)))

    selected_levels = set(levels or ())
    selected_families = {family.upper() for family in (families or ())}
    selected = []
    for task in tasks:
        family = get_task_family(task.name)
        if requested_names and task.name not in requested_names:
            continue
        if selected_families and family not in selected_families:
            continue
        if selected_levels:
            if family is None or get_task_level(task.name) not in selected_levels:
                continue
        selected.append(task)
    return selected


def task_state_seed(base_seed, task_name, state_index):
    """Return a selection-order-independent seed for deterministic resume."""

    task_offset = int(hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + task_offset + int(state_index)) % (2**32)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def atomic_text_write(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(value)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _run_torch_bridge(script, arguments, operation):
    """Run Torch-only serialization work outside the MuJoCo process."""

    try:
        return subprocess.run(
            [sys.executable, "-c", script, *map(str, arguments)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError("{} failed: {}".format(operation, detail)) from error


def atomic_torch_save_pair(value, init_path, pruned_path):
    """Atomically save one NumPy state array to both Torch-compatible paths.

    PyTorch is deliberately imported only in the child process. Loading it in
    the MuJoCo worker can initialize a second libiomp runtime and abort Windows
    init-state generation with OpenMP error 15.
    """

    destinations = [Path(init_path), Path(pruned_path)]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    array_path = None
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(destinations[0].parent),
            prefix=".states-",
            suffix=".npy",
            delete=False,
        ) as temporary:
            array_path = Path(temporary.name)
        with array_path.open("wb") as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)

        for destination in destinations:
            with tempfile.NamedTemporaryFile(
                dir=str(destination.parent),
                prefix=".init-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_paths.append(Path(temporary.name))

        script = (
            "import numpy as np, sys, torch; "
            "value=np.load(sys.argv[1], allow_pickle=False); "
            "[torch.save(value, path) for path in sys.argv[2:]]"
        )
        _run_torch_bridge(
            script,
            [array_path, *temporary_paths],
            "Torch init-state serialization",
        )
        for temporary_path, destination in zip(temporary_paths, destinations):
            os.replace(filesystem_path(temporary_path), filesystem_path(destination))
    finally:
        if array_path is not None and array_path.exists():
            array_path.unlink()
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()


def atomic_json_dump(value, path):
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


def package_version(module_name):
    distribution_name = module_name.replace("_", "-")
    try:
        return distribution_version(distribution_name)
    except PackageNotFoundError:
        module = sys.modules.get(module_name)
        return str(getattr(module, "__version__", "unknown")) if module else None


def source_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def report_path(path):
    """Keep paths inside the repository portable in generated reports."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate embodiment-specific initial states. Ablation suites synchronize their "
            "task map before generation."
        )
    )
    parser.add_argument("--suite", type=str, required=True, help="Target benchmark suite name.")
    parser.add_argument(
        "--level",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        help="Only generate these Industry levels, for example: --level 1 2.",
    )
    parser.add_argument(
        "--family",
        type=str,
        nargs="+",
        choices=("D2D", "D2S", "S2D"),
        help="Only generate these Industry task families.",
    )
    parser.add_argument(
        "--task",
        action="append",
        help="Generate one exact task name. Repeat this option to select multiple tasks.",
    )
    parser.add_argument(
        "--embodiment",
        choices=available_init_state_profiles(),
        default="panda",
        help="Robot/gripper profile that owns the generated states (default: %(default)s).",
    )
    parser.add_argument(
        "--bddl-root",
        type=Path,
        default=Path(SEARCH_DIR),
        help=(
            "Root containing the suite BDDL directory. Use the staging run directory "
            "to avoid reading live BDDLs."
        ),
    )
    parser.add_argument(
        "--init-root",
        type=Path,
        default=None,
        help=(
            "Final root where physical suite directories will be written. When omitted, "
            "Panda uses the legacy init_files root and other embodiments use their "
            "embodiments/<profile> root."
        ),
    )
    parser.add_argument(
        "--target-states",
        type=int,
        default=TARGET_STATES,
        help="Total states to retain per task (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate selected state pairs instead of resuming them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the selected files and output paths without writing states.",
    )
    args = parser.parse_args()

    if args.target_states <= 0:
        parser.error("--target-states must be positive")

    suite_name = args.suite

    bddl_root = args.bddl_root.resolve()
    if not bddl_root.is_dir():
        parser.error("BDDL root directory does not exist: {}".format(bddl_root))

    ablation_suite = is_ablation_suite(suite_name)
    task_map_changed = False
    staged_ablation_tasks = None
    live_bddl_root = Path(SEARCH_DIR).resolve()
    if ablation_suite and bddl_root != live_bddl_root:
        try:
            staged_ablation_tasks = suite_tasks_from_root(suite_name, bddl_root)
        except ValueError as error:
            parser.error(str(error))
        print(
            "[ABLATION POLICY] Using {} staged BDDLs in memory; the live task map "
            "will not be changed.".format(len(staged_ablation_tasks))
        )
    elif ablation_suite:
        sync_ok, task_map_changed = sync_task_map(
            suite_name, bddl_root, dry_run=args.dry_run
        )
        if not sync_ok:
            parser.error("Could not synchronize task map for '{}'".format(suite_name))
        if args.dry_run and task_map_changed:
            parser.error(
                "Ablation task map is out of date. Rerun without --dry-run to synchronize "
                "it before generating or previewing states."
            )
        print(
            "[ABLATION POLICY] Task-map sync enabled; takt horizon, Industry scene "
            "overrides, and Industry object-sampling overrides are disabled."
        )

    # Ablation map synchronization must happen before this import because the
    # benchmark package snapshots libero_suite_task_map while it initializes.
    benchmark = importlib.import_module("libero.libero.benchmark")
    if staged_ablation_tasks is not None:
        benchmark.libero_task_map[suite_name] = staged_ablation_tasks
        benchmark = importlib.reload(benchmark)
    elif ablation_suite and task_map_changed:
        task_map_module = importlib.import_module(
            "libero.libero.benchmark.libero_suite_task_map"
        )
        importlib.reload(task_map_module)
        benchmark = importlib.reload(benchmark)

    legacy_init_root = Path(LIBERO_ROOT) / "init_files"
    init_root = resolve_init_states_root(
        legacy_init_root,
        embodiment=args.embodiment,
        explicit_init_root=args.init_root,
    )
    profile = get_init_state_profile(args.embodiment)

    print(
        "[RUN CONFIGURATION] Suite: {} | Embodiment: {} | Levels: {} | Families: {} | "
        "Target States: {} | BDDL Root: {} | Init Root: {}".format(
            suite_name,
            profile.name,
            sorted(args.level) if args.level else "all",
            sorted(args.family) if args.family else "all",
            args.target_states,
            bddl_root,
            init_root,
        )
    )
    if args.dry_run:
        print("[DRY RUN] Filesystem writes and state generation will be bypassed.")

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        if suite_name not in benchmark_dict:
            parser.error("Suite '{}' is not registered".format(suite_name))
        task_suite = benchmark_dict[suite_name]()
    except Exception as e:
        parser.error("Benchmark initialization failure: {}".format(e))

    all_tasks_are_industry = all(
        is_industry_task(task_suite.get_task(task_id).name)
        for task_id in range(task_suite.n_tasks)
    )
    if (args.level or args.family) and not all_tasks_are_industry:
        parser.error("--level and --family are only supported for Industry task suites")
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
        parser.error("No tasks in '{}' matched the requested selectors".format(suite_name))

    success_count = 0
    failure_count = 0
    run_started = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": 1,
        "suite": suite_name,
        "embodiment": profile.name,
        "profile": profile.manifest_dict(),
        "source_commit": source_commit(),
        "started_at": run_started,
        "seed": args.seed,
        "target_states": args.target_states,
        "bddl_root": report_path(bddl_root),
        "init_root": report_path(init_root),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "init_policy": "ablation" if ablation_suite else "standard",
        "task_map_synchronized": bool(ablation_suite and staged_ablation_tasks is None),
        "task_map_source": "staged_memory" if staged_ablation_tasks is not None else "live",
        "takt_horizon_applied": False,
        "selectors": {
            "levels": sorted(args.level) if args.level else None,
            "families": sorted(args.family) if args.family else None,
            "tasks": sorted(args.task) if args.task else None,
        },
        "tasks": [],
    }

    for task_index, task in enumerate(selected_tasks, start=1):
        task_name = task.name
        print(
            "\n--- Processing Task {}/{}: {} ---".format(
                task_index, len(selected_tasks), task_name
            )
        )

        env = None
        task_report = {
            "task": task_name,
            "problem_folder": task.problem_folder,
        }
        try:
            bddl_filename = Path(task.bddl_file).name
            bddl_path = bddl_root / task.problem_folder / bddl_filename
            if not bddl_path.is_file():
                raise FileNotFoundError("BDDL file does not exist: {}".format(bddl_path))

            init_save_dir = init_root / task.problem_folder

            pruned_path = init_save_dir / init_state_filename(
                task_name, pruned=True, embodiment=profile.name
            )
            init_path = init_save_dir / init_state_filename(
                task_name, embodiment=profile.name
            )
            existing_array = load_existing_state_pair(
                init_path, pruned_path, overwrite=args.overwrite
            )
            existing_states = (
                [] if existing_array is None else [state.copy() for state in existing_array]
            )

            states_needed = args.target_states - len(existing_states)
            if states_needed <= 0:
                print(f"[SKIP] Already has {len(existing_states)} states.")
                task_report.update(
                    {
                        "status": "complete_existing",
                        "init_file": init_path.name,
                        "pruned_init_file": pruned_path.name,
                        "state_count": len(existing_states),
                        "state_width": int(existing_array.shape[1]),
                        "state_sha256": sha256_array(existing_array),
                        "bddl_sha256": sha256_file(bddl_path),
                    }
                )
                success_count += 1
                continue

            if args.dry_run:
                print(f"[DRY RUN] Would generate {states_needed} states to {pruned_path}")
                task_report.update(
                    {
                        "status": "dry_run",
                        "init_file": init_path.name,
                        "pruned_init_file": pruned_path.name,
                        "existing_state_count": len(existing_states),
                        "states_to_generate": states_needed,
                        "bddl_sha256": sha256_file(bddl_path),
                    }
                )
                success_count += 1
                continue

            init_save_dir.mkdir(parents=True, exist_ok=True)

            env_args = build_init_env_args(
                bddl_path,
                profile.name,
                suite_name,
                task_name,
                seed=args.seed,
            )

            env = OffScreenRenderEnv(**env_args)
            init_states = existing_states.copy()
            expected_width = existing_array.shape[1] if existing_array is not None else None

            for state_index in range(len(existing_states), args.target_states):
                env.seed(task_state_seed(args.seed, task_name, state_index))
                env.reset()
                state = np.asarray(env.get_init_state()).reshape(-1)
                if not np.isfinite(state).all():
                    raise ValueError("Generated state {} contains NaN or infinity".format(state_index))
                if expected_width is None:
                    expected_width = state.shape[0]
                elif state.shape[0] != expected_width:
                    raise ValueError(
                        "Generated state width changed from {} to {} at index {}".format(
                            expected_width, state.shape[0], state_index
                        )
                    )
                init_states.append(state)

            init_states_array = as_init_state_array(
                np.stack(init_states), source="generated states for {}".format(task_name)
            )
            atomic_torch_save_pair(init_states_array, init_path, pruned_path)
            print(f"[SUCCESS] Saved initial physics states for {task_name}.")
            task_report.update(
                {
                    "status": "generated",
                    "init_file": init_path.name,
                    "pruned_init_file": pruned_path.name,
                    "state_count": int(init_states_array.shape[0]),
                    "state_width": int(init_states_array.shape[1]),
                    "state_sha256": sha256_array(init_states_array),
                    "bddl_sha256": sha256_file(bddl_path),
                }
            )
            success_count += 1

        except Exception as e:
            print(f"[ERROR] Logic fault processing task {task_name}: {e}")
            traceback.print_exc()
            task_report.update({"status": "failed", "error": str(e)})
            failure_count += 1
        finally:
            report["tasks"].append(task_report)
            if env is not None:
                try:
                    env.close()
                except Exception as close_error:
                    print("[WARNING] Failed to close environment: {}".format(close_error))

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = {
        "selected_tasks": len(selected_tasks),
        "successful_tasks": success_count,
        "failed_tasks": failure_count,
    }
    if not args.dry_run:
        manifest = {
            "schema_version": 1,
            "embodiment": profile.name,
            "profile": profile.manifest_dict(),
            "source_commit": report["source_commit"],
            "generated_at": report["finished_at"],
            "state_format": "torch-saved numpy.ndarray[num_states, flattened_mujoco_state]",
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": package_version("torch"),
                "robosuite": package_version("robosuite"),
                "mujoco": package_version("mujoco"),
                "mujoco_py": package_version("mujoco_py"),
            },
        }
        atomic_json_dump(manifest, init_root / "manifest.json")
        atomic_json_dump(
            report, init_root / "generation_report_{}.json".format(suite_name)
        )

    print(
        "\n[SUMMARY] Successfully processed {}/{} selected tasks for suite: {}; "
        "{} failed.".format(
            success_count, len(selected_tasks), suite_name, failure_count
        )
    )
    if failure_count:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
