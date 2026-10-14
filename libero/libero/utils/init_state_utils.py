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
