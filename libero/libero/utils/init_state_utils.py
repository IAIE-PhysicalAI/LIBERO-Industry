"""Embodiment-aware initial-state profiles and safe state loading helpers."""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


DEFAULT_INIT_STATE_PROFILE = "panda"
COMPACT_INIT_STATE_STEM_LENGTH = 88

_FRACTIONAL_TIMING_RE = re.compile(
    r"(?P<timing>_(?:v\d+(?:\.\d+)?_t\d+(?:\.\d+)?|svc\d+(?:\.\d+)?))$",
    re.IGNORECASE,
)
_FRACTIONAL_ACTION_MARKERS = ("_pick_up_", "_route_", "_sort_")


def filesystem_path(path: Path) -> str:
    """Return a path that supports long filenames on Windows."""

    resolved = os.path.abspath(os.fspath(Path(path).expanduser()))
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    if resolved.startswith("\\\\"):
        return "\\\\?\\UNC\\" + resolved[2:]
    return "\\\\?\\" + resolved


@dataclass(frozen=True)
class InitStateProfile:
    """Simulator configuration that owns a set of flattened MuJoCo states."""

    name: str
    robots: Tuple[str, ...]
    gripper_types: str
    relative_init_root: Optional[Path]
    compact_filenames: bool = False

    def env_kwargs(self) -> Dict[str, Any]:
        return {
            "robots": list(self.robots),
            "gripper_types": self.gripper_types,
        }

    def resolve_root(self, legacy_init_root: Path) -> Path:
        legacy_init_root = Path(legacy_init_root).expanduser().resolve()
        if self.relative_init_root is None:
            return legacy_init_root
        return legacy_init_root / self.relative_init_root

    def manifest_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "robots": list(self.robots),
            "gripper_types": self.gripper_types,
            "relative_init_root": (
                str(self.relative_init_root).replace("\\", "/")
                if self.relative_init_root is not None
                else None
            ),
            "compact_filenames": self.compact_filenames,
        }


INIT_STATE_PROFILES: Mapping[str, InitStateProfile] = {
    "panda": InitStateProfile(
        name="panda",
        robots=("Panda",),
        gripper_types="default",
        # Preserve the original LIBERO layout for existing users.
        relative_init_root=None,
    ),
    "aubo_i5_ctag2f90d": InitStateProfile(
        name="aubo_i5_ctag2f90d",
        robots=("AuboI5",),
        gripper_types="CTAG2F90DGripper",
        relative_init_root=Path("embodiments") / "aubo_i5_ctag2f90d",
        compact_filenames=True,
    ),
}


@dataclass(frozen=True)
class EvaluationEmbodiment:
    """Robot configuration and source state used for policy evaluation."""

    name: str
    robots: Tuple[str, ...]
    gripper_types: str
    source_init_profile: str
    upstream_controller: bool = False
    minimum_robosuite: Tuple[int, int] = (1, 4)
    gripper_action: str = "identity"

    def env_kwargs(self) -> Dict[str, Any]:
        configured = {
            "robots": list(self.robots),
            "gripper_types": self.gripper_types,
        }
        if self.upstream_controller and installed_robosuite_version() >= (1, 5):
            from robosuite.controllers import load_composite_controller_config

            configured["controller"] = load_composite_controller_config(
                robot=self.robots[0]
            )
        return configured


EVALUATION_EMBODIMENTS: Mapping[str, EvaluationEmbodiment] = {
    "panda": EvaluationEmbodiment(
        name="panda",
        robots=("Panda",),
        gripper_types="default",
        source_init_profile="panda",
    ),
    "aubo_i5_ctag2f90d": EvaluationEmbodiment(
        name="aubo_i5_ctag2f90d",
        robots=("AuboI5",),
        gripper_types="CTAG2F90DGripper",
        source_init_profile="aubo_i5_ctag2f90d",
    ),
    "iiwa": EvaluationEmbodiment(
        name="iiwa",
        robots=("IIWA",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
    ),
    "jaco": EvaluationEmbodiment(
        name="jaco",
        robots=("Jaco",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
    ),
    "kinova3": EvaluationEmbodiment(
        name="kinova3",
        robots=("Kinova3",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
    ),
    "sawyer": EvaluationEmbodiment(
        name="sawyer",
        robots=("Sawyer",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
    ),
    "ur5e": EvaluationEmbodiment(
        name="ur5e",
        robots=("UR5e",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
    ),
    "panda_dex_lh": EvaluationEmbodiment(
        name="panda_dex_lh",
        robots=("PandaDexLH",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
        minimum_robosuite=(1, 5),
        gripper_action="broadcast",
    ),
    "panda_dex_rh": EvaluationEmbodiment(
        name="panda_dex_rh",
        robots=("PandaDexRH",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
        minimum_robosuite=(1, 5),
        gripper_action="broadcast",
    ),
    "xarm7": EvaluationEmbodiment(
        name="xarm7",
        robots=("XArm7",),
        gripper_types="default",
        source_init_profile="panda",
        upstream_controller=True,
        minimum_robosuite=(1, 5),
    ),
}


def installed_robosuite_version() -> Tuple[int, int]:
    import robosuite

    parts = robosuite.__version__.split(".")
    return int(parts[0]), int(parts[1])


def available_evaluation_embodiments() -> Tuple[str, ...]:
    version = installed_robosuite_version()
    return tuple(
        sorted(
            name
            for name, embodiment in EVALUATION_EMBODIMENTS.items()
            if version >= embodiment.minimum_robosuite
        )
    )


def get_evaluation_embodiment(name: Optional[str] = None) -> EvaluationEmbodiment:
    normalized = str(name or DEFAULT_INIT_STATE_PROFILE).strip().lower()
    try:
        embodiment = EVALUATION_EMBODIMENTS[normalized]
    except KeyError as error:
        raise ValueError(
            "Unknown evaluation embodiment '{}'. Available embodiments: {}".format(
                normalized, ", ".join(available_evaluation_embodiments())
            )
        ) from error
    version = installed_robosuite_version()
    if version < embodiment.minimum_robosuite:
        raise ValueError(
            "Evaluation embodiment '{}' requires RoboSuite {}.{}, installed {}.{}".format(
                normalized,
                *embodiment.minimum_robosuite,
                *version,
            )
        )
    return embodiment


def apply_evaluation_embodiment(
    env_args: Mapping[str, Any], embodiment: Optional[str] = None
) -> Dict[str, Any]:
    configured = dict(env_args)
    configured.update(get_evaluation_embodiment(embodiment).env_kwargs())
    return configured


def get_evaluation_source_profile(
    embodiment: Optional[str] = None,
) -> InitStateProfile:
    target = get_evaluation_embodiment(embodiment)
    return get_init_state_profile(target.source_init_profile)


def evaluation_requires_state_transcode(embodiment: Optional[str] = None) -> bool:
    target = get_evaluation_embodiment(embodiment)
    return target.name != target.source_init_profile


def available_init_state_profiles() -> Tuple[str, ...]:
    return tuple(sorted(INIT_STATE_PROFILES))


def get_init_state_profile(name: Optional[str] = None) -> InitStateProfile:
    normalized = str(name or DEFAULT_INIT_STATE_PROFILE).strip().lower()
    try:
        return INIT_STATE_PROFILES[normalized]
    except KeyError as error:
        raise ValueError(
            "Unknown init-state embodiment '{}'. Available profiles: {}".format(
                normalized, ", ".join(available_init_state_profiles())
            )
        ) from error


def get_configured_embodiment(config: Any) -> str:
    """Read an embodiment from dict-like or OmegaConf-like configuration."""

    if config is None:
        return DEFAULT_INIT_STATE_PROFILE
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            value = getter("embodiment", DEFAULT_INIT_STATE_PROFILE)
            return str(value or DEFAULT_INIT_STATE_PROFILE)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    try:
        return str(getattr(config, "embodiment"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return DEFAULT_INIT_STATE_PROFILE


def resolve_init_states_root(
    legacy_init_root: Path,
    embodiment: Optional[str] = None,
    explicit_init_root: Optional[Path] = None,
) -> Path:
    """Resolve the final root that directly contains suite directories.

    ``explicit_init_root`` is already a final profile root. This makes staging
    runs unambiguous and avoids adding an embodiment subdirectory twice.
    """

    if explicit_init_root:
        return Path(explicit_init_root).expanduser().resolve()
    return get_init_state_profile(embodiment).resolve_root(Path(legacy_init_root))


def apply_init_state_profile(
    env_args: Mapping[str, Any], embodiment: Optional[str] = None
) -> Dict[str, Any]:
    """Return environment arguments with the selected robot and gripper."""

    configured = dict(env_args)
    configured.update(get_init_state_profile(embodiment).env_kwargs())
    return configured


def compact_init_state_stem(task_name: str) -> str:
    """Return a deterministic, Git-friendly stem for a long task name.

    Fractional-tuning task names repeat a long natural-language action between
    their identifying prefix and timing suffix. Preserve those useful pieces
    and add a short digest so different full task names cannot collide.
    """

    task_name = str(task_name)
    if len(task_name) <= COMPACT_INIT_STATE_STEM_LENGTH:
        return task_name

    timing_match = _FRACTIONAL_TIMING_RE.search(task_name)
    timing = timing_match.group("timing") if timing_match else ""
    core = task_name[: timing_match.start()] if timing_match else task_name
    identifying_prefix = core
    for marker in _FRACTIONAL_ACTION_MARKERS:
        marker_index = core.lower().find(marker)
        if marker_index >= 0:
            identifying_prefix = core[:marker_index]
            break

    digest = hashlib.sha256(task_name.encode("utf-8")).hexdigest()[:10]
    reserved = len(timing) + len(digest) + 2
    prefix_limit = COMPACT_INIT_STATE_STEM_LENGTH - reserved
    prefix = identifying_prefix[:prefix_limit].rstrip("_-. ")
    if not prefix:
        prefix = core[:prefix_limit].rstrip("_-. ")
    return "{}__{}{}".format(prefix, digest, timing)


def init_state_filename(
    task_name: str,
    *,
    pruned: bool = False,
    embodiment: Optional[str] = None,
) -> str:
    """Return the profile-specific filename for one task's saved states."""

    profile = get_init_state_profile(embodiment)
    stem = compact_init_state_stem(task_name) if profile.compact_filenames else task_name
    return stem + (".pruned_init" if pruned else ".init")


def resolve_init_state_path(
    init_root: Path,
    problem_folder: str,
    task_name: str,
    *,
    pruned: bool = True,
    embodiment: Optional[str] = None,
) -> Path:
    """Resolve a saved state path, accepting legacy long Aubo filenames.

    The compact name is authoritative for profiles that enable it. The legacy
    BDDL-derived name remains a read fallback so existing datasets do not need
    to be regenerated in order to load.
    """

    init_dir = Path(init_root) / problem_folder
    preferred = init_dir / init_state_filename(
        task_name, pruned=pruned, embodiment=embodiment
    )
    legacy_suffix = ".pruned_init" if pruned else ".init"
    legacy = init_dir / (task_name + legacy_suffix)
    if os.path.isfile(filesystem_path(preferred)) or preferred == legacy:
        return preferred
    if os.path.isfile(filesystem_path(legacy)):
        return legacy
    return preferred


def torch_load_compat(path: Path) -> Any:
    """Load trusted repository state files across supported PyTorch versions."""

    import torch

    path = Path(path)
    load_path = filesystem_path(path)
    try:
        return torch.load(load_path, weights_only=False)
    except TypeError:
        return torch.load(load_path)


def as_init_state_array(value: Any, source: str = "init states") -> np.ndarray:
    """Normalize saved init states to a finite two-dimensional numeric array."""

    # Importing PyTorch eagerly can load a second Intel OpenMP runtime before
    # MuJoCo initializes. A tensor can only reach this function after PyTorch
    # has already been imported by the caller, so inspect that existing module
    # without loading it for ordinary NumPy state generation.
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            "{} must have shape [num_states, state_width], got {}".format(
                source, array.shape
            )
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(
            "{} must contain numeric values, got {}".format(source, array.dtype)
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("{} cannot be empty, got {}".format(source, array.shape))
    if not np.isfinite(array).all():
        raise ValueError("{} contains NaN or infinite values".format(source))
    return array


def load_init_state_array(path: Path) -> np.ndarray:
    path = Path(path)
    return as_init_state_array(torch_load_compat(path), source=str(path))


def adapt_evaluation_actions(
    actions: np.ndarray,
    embodiment: Optional[str],
    target_dimension: int,
) -> np.ndarray:
    """Map a Panda-style OSC action to the selected single-arm embodiment."""

    profile = get_evaluation_embodiment(embodiment)
    array = np.asarray(actions)
    if array.ndim == 0:
        raise ValueError("Evaluation actions must have at least one dimension")
    if array.shape[-1] == target_dimension:
        return array
    if (
        profile.gripper_action == "broadcast"
        and array.shape[-1] == 7
        and target_dimension > 6
    ):
        gripper = np.repeat(array[..., 6:7], target_dimension - 6, axis=-1)
        return np.concatenate((array[..., :6], gripper), axis=-1)
    raise ValueError(
        "Policy action dimension {} is incompatible with '{}' action dimension {}".format(
            array.shape[-1], profile.name, target_dimension
        )
    )


def environment_action_dimension(env: Any) -> int:
    """Return the flattened action dimension exposed by a LIBERO environment."""

    action_spec = getattr(env.env, "action_spec", None)
    if action_spec is None:
        raise RuntimeError("Environment does not expose action_spec")
    low, _ = action_spec
    return int(np.asarray(low).size)


def _robot_naming_prefixes(env: Any) -> Tuple[str, ...]:
    prefixes = []
    for robot in env.robots:
        models = [
            robot.robot_model,
            getattr(robot.robot_model, "base", None),
            getattr(robot.robot_model, "mount", None),
        ]
        grippers = getattr(robot, "gripper", None)
        models.extend(grippers.values() if isinstance(grippers, dict) else [grippers])
        for model in models:
            prefix = getattr(model, "naming_prefix", None) if model is not None else None
            if prefix:
                prefixes.append(str(prefix))
    if not prefixes:
        raise RuntimeError("Environment does not expose robot naming prefixes")
    return tuple(sorted(set(prefixes)))


def _named_non_robot_joints(env: Any) -> Dict[str, int]:
    prefixes = _robot_naming_prefixes(env)
    joints = {}
    for joint_id in range(int(env.sim.model.njnt)):
        name = env.sim.model.joint_id2name(joint_id)
        if name is not None and not str(name).startswith(prefixes):
            joints[str(name)] = joint_id
    return joints


def capture_portable_task_state(env: Any) -> Dict[str, Any]:
    """Capture named non-robot joints from a source embodiment state."""

    joints = {}
    for name in _named_non_robot_joints(env):
        joints[name] = {
            "qpos": np.asarray(env.sim.data.get_joint_qpos(name)).copy(),
            "qvel": np.asarray(env.sim.data.get_joint_qvel(name)).copy(),
        }
    return {
        "time": float(env.sim.data.time),
        "joints": joints,
    }


def apply_portable_task_state(
    env: Any,
    portable_state: Mapping[str, Any],
) -> None:
    """Apply named task joints while retaining the target robot's reset pose."""

    source_joints = portable_state.get("joints", {})
    target_joints = _named_non_robot_joints(env)
    source_names = set(source_joints)
    target_names = set(target_joints)
    if source_names != target_names:
        missing = sorted(source_names - target_names)
        extra = sorted(target_names - source_names)
        raise ValueError(
            "Portable task joint mismatch: missing={}, extra={}".format(missing, extra)
        )

    for name in sorted(source_names):
        state = source_joints[name]
        source_qpos = np.asarray(state["qpos"])
        source_qvel = np.asarray(state["qvel"])
        target_qpos = np.asarray(env.sim.data.get_joint_qpos(name))
        target_qvel = np.asarray(env.sim.data.get_joint_qvel(name))
        if source_qpos.shape != target_qpos.shape:
            raise ValueError(
                "qpos shape mismatch for '{}': source={}, target={}".format(
                    name, source_qpos.shape, target_qpos.shape
                )
            )
        if source_qvel.shape != target_qvel.shape:
            raise ValueError(
                "qvel shape mismatch for '{}': source={}, target={}".format(
                    name, source_qvel.shape, target_qvel.shape
                )
            )
        env.sim.data.set_joint_qpos(name, source_qpos)
        env.sim.data.set_joint_qvel(name, source_qvel)

    env.sim.data.time = float(portable_state.get("time", 0.0))
    env.sim.forward()


def transcode_init_state_array(
    source_states: Any,
    source_env: Any,
    target_env: Any,
) -> np.ndarray:
    """Convert source states to target-native states without solving IK online.

    ``target_env.reset()`` installs the mounted robot's precomputed ``init_qpos``.
    Only named non-robot joints are then copied from the source state.
    """

    source_array = as_init_state_array(source_states)
    converted = []
    expected_width = None
    for state_index, source_state in enumerate(source_array):
        source_env.set_state(source_state)
        source_env.sim.forward()
        portable_state = capture_portable_task_state(source_env)

        target_env.reset()
        apply_portable_task_state(target_env, portable_state)
        target_state = np.asarray(target_env.get_init_state()).reshape(-1)
        if not np.isfinite(target_state).all():
            raise ValueError(
                "Converted target state {} contains NaN or infinity".format(state_index)
            )
        if expected_width is None:
            expected_width = target_state.shape[0]
        elif target_state.shape[0] != expected_width:
            raise ValueError(
                "Converted target state width changed from {} to {} at index {}".format(
                    expected_width, target_state.shape[0], state_index
                )
            )
        converted.append(target_state)
    return as_init_state_array(np.stack(converted), source="transcoded init states")


def load_existing_state_pair(
    init_path: Path, pruned_path: Path, overwrite: bool = False
) -> Optional[np.ndarray]:
    """Load an intact state pair without importing Torch beside MuJoCo."""

    if overwrite:
        return None

    sources = [Path(init_path), Path(pruned_path)]
    exists = [os.path.isfile(filesystem_path(path)) for path in sources]
    if exists[0] != exists[1]:
        raise FileNotFoundError(
            "Refusing to resume a partial state pair: {} exists={}, {} exists={}".format(
                sources[0], exists[0], sources[1], exists[1]
            )
        )
    if not exists[0]:
        return None

    temporary_paths = []
    try:
        for source in sources:
            with tempfile.NamedTemporaryFile(
                dir=str(source.parent),
                prefix=".loaded-",
                suffix=".npy",
                delete=False,
            ) as temporary:
                temporary_paths.append(Path(temporary.name))

        bridge = (
            "import numpy as np, sys, torch\n"
            "def load(path):\n"
            "    try:\n"
            "        return torch.load(path, weights_only=False)\n"
            "    except TypeError:\n"
            "        return torch.load(path)\n"
            "for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):\n"
            "    value = load(src)\n"
            "    if torch.is_tensor(value):\n"
            "        value = value.detach().cpu().numpy()\n"
            "    np.save(dst, np.asarray(value), allow_pickle=False)\n"
        )
        arguments = []
        for source, temporary_path in zip(sources, temporary_paths):
            arguments.extend([filesystem_path(source), filesystem_path(temporary_path)])
        try:
            subprocess.run(
                [sys.executable, "-c", bridge, *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError("Torch init-state loading failed: {}".format(detail)) from error

        init_states, pruned_states = [
            as_init_state_array(
                np.load(path, allow_pickle=False), source=str(source)
            )
            for path, source in zip(temporary_paths, sources)
        ]
        if init_states.shape != pruned_states.shape or not np.array_equal(
            init_states, pruned_states
        ):
            raise ValueError(
                "Existing .init and .pruned_init files do not contain the same states"
            )
        return pruned_states
    finally:
        for temporary_path in temporary_paths:
            if temporary_path.exists():
                temporary_path.unlink()
