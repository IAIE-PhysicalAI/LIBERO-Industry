#!/usr/bin/env python3
"""Generate a small fractional-parameter BDDL tuning set.

For each industry suite and difficulty level, the script selects the canonical
``001`` source BDDL and writes one parameterized copy for every requested tuning
value.
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import re


# Speeds are filename labels in centimetres per second.  BDDL and the runtime
# use metres per second, so ``v3`` must be emitted as ``0.03``.
SPEED_TAKT_PAIRS = ((3, 4), (3, 6), (5, 4), (8, 2), (8, 4), (8, 6))
D2D_LV2_TAKTS = (5, 6)
SERVICE_TIMES = ("1.5", "2.0", "2.5", "3.0", "3.5", "4.0")
LEVELS = ("LV1", "LV2", "LV3")
SUITES = ("libero_industry_d2d", "libero_industry_d2s", "libero_industry_s2d")

# Four-object comparison matrix used for fractional timing studies.  D2D keeps
# the source and destination speeds encoded by each canonical filename and
# varies only takt.  D2S retains its speed/takt sweep.  D2S level 1 has four toy
# types; its higher levels have six types, all of which appear across source
# tasks 001--004.
FOUR_OBJECT_SOURCE_IDS = {
    ("D2D", "LV1"): ("001", "013"),
    ("S2D", "LV1"): ("001", "002", "003", "007", "008", "009", "013", "014", "015", "019", "020", "021"),
    ("S2D", "LV2"): ("001", "002", "003", "007", "008", "009", "013", "014", "015", "019", "020", "021"),
    ("S2D", "LV3"): ("001", "002", "003", "007", "008", "009", "013", "014", "015", "019", "020", "021"),
    # Four objects at each authoritative positive source-speed class.
    ("D2D", "LV2"): (
        "001", "002", "003",
        "007", "008", "009",
        "013", "014", "015",
        "019", "020", "021",
    ),
    ("D2D", "LV3"): ("001", "007", "013", "019"),
    ("D2S", "LV1"): ("001", "002", "003", "004"),
    ("D2S", "LV2"): ("001", "002", "003", "004"),
    ("D2S", "LV3"): ("001", "002", "003", "004"),
}


# The optional takt/offset and service portions keep the script compatible with
# pre-existing BDDL files.  Generated files always contain the strict, expanded
# predicates required by the industry task mappers.
AT_SPEED_RE = re.compile(
    r"(?P<prefix>\(\s*AtSpeed\s+)"
    r"(?P<fixture>[^\s()]+)"
    r"(?P<fixture_speed_ws>\s+)"
    r"(?P<speed>[^\s()]+)"
    r"(?P<speed_steady_ws>\s+)steady"
    r"(?:"
    r"(?P<takt_ws>\s+)takt_(?P<takt>[^\s()]+)"
    r"(?P<offset_ws>\s+)offset_(?P<offset>[^\s()]+)"
    r")?"
    r"(?P<closing_ws>\s*\))"
)
LOAD_SLOT_RE = re.compile(
    r"(?P<prefix>\(\s*LoadSlot\s+)"
    r"(?P<fixture>[^\s()]+)"
    r"(?P<fixture_inlay_ws>\s+)"
    r"(?P<inlay>[^\s()]+)"
    r"(?P<inlay_axis_ws>\s+)"
    r"(?P<axis>[^\s()]+)"
    r"(?P<axis_pulse_ws>\s+)pulse"
    r"(?P<pulse_speed_ws>\s+)"
    r"(?P<speed>[^\s()]+)"
    r"(?:"
    r"(?P<service_ws>\s+)service_(?P<service>[^\s()]+)"
    r")?"
    r"(?P<closing_ws>\s*\))"
)
FRACTIONAL_AT_SPEED_SUFFIX_RE = re.compile(
    r"_v(?P<speed>\d+)_t(?P<takt>\d+(?:\.\d+)?)\.bddl$", re.IGNORECASE
)
# D2D keeps this legacy suffix for stable task/result references.  Its earlier
# V/M token pair is authoritative; only the suffix's takt value is applied.
FRACTIONAL_TASK_ID_RE = re.compile(
    r"^(?P<suite>D2D|D2S|S2D)_(?P<level>LV[123])_(?P<task>\d{3})(?:_|$)",
    re.IGNORECASE,
)
SUITE_DIR_BY_CODE = {
    "D2D": "libero_industry_d2d",
    "D2S": "libero_industry_d2s",
    "S2D": "libero_industry_s2d",
}


def repo_root() -> Path:
    """Return the repository root when invoked from any working directory."""

    return Path(__file__).resolve().parents[3]


def choose_sources(bddl_root: Path) -> list[tuple[str, str, Path]]:
    """Select the single canonical ``001`` BDDL for each suite and level."""

    selections: list[tuple[str, str, Path]] = []
    for suite in SUITES:
        suite_dir = bddl_root / suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(f"Suite directory does not exist: {suite_dir}")

        for level in LEVELS:
            task_prefix = re.compile(rf"(?:^|[_-]){level}[_-]001(?:[_-]|$)")
            candidates = sorted(
                path for path in suite_dir.glob("*.bddl") if task_prefix.search(path.stem)
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No canonical {level} task 001 .bddl file in {suite_dir}"
                )
            if len(candidates) != 1:
                matches = ", ".join(path.name for path in candidates)
                raise ValueError(
                    f"Expected exactly one canonical {level} task 001 file in "
                    f"{suite_dir}, found {len(candidates)}: {matches}"
                )
            selections.append((suite, level, candidates[0]))
    return selections


def choose_four_object_sources(bddl_root: Path) -> list[tuple[str, str, Path]]:
    """Select the balanced source matrix for the four-object timing study."""

    selections: list[tuple[str, str, Path]] = []
    for (suite, level), task_ids in FOUR_OBJECT_SOURCE_IDS.items():
        for task_id in task_ids:
            identity = "{}_{}_{}".format(suite, level, task_id)
            selections.append(
                (SUITE_DIR_BY_CODE[suite], level, _canonical_source(bddl_root, suite, identity))
            )
    return selections


def _scaled_speed(speed_code: int, existing_speed: str) -> str:
    """Convert a ``v3``-style code to m/s while preserving BDDL direction."""

    magnitude = speed_code / 100.0
    sign = -1.0 if float(existing_speed) < 0.0 else 1.0
    return format(math.copysign(magnitude, sign), ".6g")


def _s2d_speed_code(source: Path) -> int:
    """Read the V3/V5/V8 speed code from one canonical S2D source name."""

    match = re.search(r"_V(?P<speed>3|5|8)_", source.stem, re.IGNORECASE)
    if match is None:
        raise ValueError(f"S2D source has no V3/V5/V8 speed code: {source.name}")
    return int(match.group("speed"))


def _d2d_source_speed_code(source: Path) -> int:
    """Read the authoritative source speed from a D2D filename pair."""

    match = re.search(
        r"_[VM](?P<speed>3|5|8)_[VM](?:3|5|8)(?:_|$)",
        source.stem,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"D2D source has no V/M speed pair: {source.name}")
    return int(match.group("speed"))


def _speed_takt_pairs(suite: str, level: str, source: Path):
    if suite == "libero_industry_d2d" and level == "LV2":
        speed = _d2d_source_speed_code(source)
        return tuple((speed, takt) for takt in D2D_LV2_TAKTS)
    return SPEED_TAKT_PAIRS


def parameterize_at_speed(
    text: str, speed_code: int | None, takt: int
) -> tuple[str, int]:
    """Set takt and offset, changing belt speed only when requested."""

    def replace(match: re.Match[str]) -> str:
        groups = match.groupdict()
        takt_ws = groups["takt_ws"] or " "
        offset_ws = groups["offset_ws"] or " "
        return (
            f"{groups['prefix']}{groups['fixture']}{groups['fixture_speed_ws']}"
            f"{groups['speed'] if speed_code is None else _scaled_speed(speed_code, groups['speed'])}"
            f"{groups['speed_steady_ws']}steady"
            f"{takt_ws}takt_{takt}{offset_ws}offset_1.5{groups['closing_ws']}"
        )

    return AT_SPEED_RE.subn(replace, text)


def parameterize_load_slot_speed(text: str, speed_code: int) -> tuple[str, int]:
    """Keep D2D's slot conveyor synchronized with its parameterized belt."""

    def replace(match: re.Match[str]) -> str:
        groups = match.groupdict()
        service = "{}service_{}".format(
            groups["service_ws"] or " ", groups["service"] or "1.0"
        )
        return (
            f"{groups['prefix']}{groups['fixture']}{groups['fixture_inlay_ws']}"
            f"{groups['inlay']}{groups['inlay_axis_ws']}{groups['axis']}"
            f"{groups['axis_pulse_ws']}pulse{groups['pulse_speed_ws']}"
            f"{_scaled_speed(speed_code, groups['speed'])}{service}"
            f"{groups['closing_ws']}"
        )

    return LOAD_SLOT_RE.subn(replace, text)


def parameterize_load_slot(text: str, service_time: str) -> tuple[str, int]:
    """Set the service time while preserving each LoadSlot speed value."""

    def replace(match: re.Match[str]) -> str:
        groups = match.groupdict()
        service_ws = groups["service_ws"] or " "
        return (
            f"{groups['prefix']}{groups['fixture']}{groups['fixture_inlay_ws']}"
            f"{groups['inlay']}{groups['inlay_axis_ws']}{groups['axis']}"
            f"{groups['axis_pulse_ws']}pulse{groups['pulse_speed_ws']}"
            f"{groups['speed']}{service_ws}service_{service_time}"
            f"{groups['closing_ws']}"
        )

    return LOAD_SLOT_RE.subn(replace, text)


def _copy_predicate_speed(text: str, source_text: str, pattern: re.Pattern) -> tuple[str, int]:
    """Copy one canonical predicate speed without replacing other BDDL content."""

    source_match = pattern.search(source_text)
    if source_match is None:
        return text, 0
    source_speed = source_match.group("speed")

    def replace(match: re.Match[str]) -> str:
        start, end = match.span("speed")
        offset = match.start()
        value = match.group(0)
        return value[: start - offset] + source_speed + value[end - offset :]

    return pattern.subn(replace, text)


def output_path(source: Path, output_dir: Path, suffix: str) -> Path:
    """Append a tuning suffix before the source file extension."""

    return output_dir / f"{source.stem}{suffix}{source.suffix}"


def _write_bddl(path: Path, text: str) -> None:
    """Write generated BDDL with repository-stable LF line endings."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _task_group(path: Path) -> tuple[str, str] | None:
    """Return the (family, level) encoded by a fractional BDDL filename."""

    match = FRACTIONAL_TASK_ID_RE.match(path.stem)
    if match is None:
        return None
    return match.group("suite").upper(), match.group("level").upper()


def _generated_paths(
    selections: list[tuple[str, str, Path]], output_dir: Path
) -> set[Path]:
    """Return the BDDL paths the selection matrix will emit."""

    paths: set[Path] = set()
    for suite, level, source in selections:
        if suite in {"libero_industry_d2d", "libero_industry_d2s"}:
            paths.update(
                output_path(source, output_dir, f"_v{speed}_t{takt}")
                for speed, takt in _speed_takt_pairs(suite, level, source)
            )
        else:
            paths.update(
                output_path(source, output_dir, f"_svc{service_time}")
                for service_time in SERVICE_TIMES
            )
    return paths


def _sync_fractional_task_map(
    output_dir: Path, *, dry_run: bool, task_names: list[str] | None = None
) -> None:
    """Keep the registered fractional suite aligned with its live BDDLs."""

    task_map_path = repo_root() / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py"
    task_map_text = task_map_path.read_text(encoding="utf-8")
    if task_names is None:
        task_names = [path.stem for path in sorted(output_dir.glob("*.bddl"))]
    if not task_names:
        raise ValueError(f"Refusing to create an empty fractional task map from {output_dir}")

    replacement = '    "fractional_tuning_set": [\n{}\n    ],'.format(
        "".join('        "{}",\n'.format(name) for name in task_names)
    )
    pattern = re.compile(
        r'^    "fractional_tuning_set": \[\n.*?^    \],', re.MULTILINE | re.DOTALL
    )
    updated_text, replacements = pattern.subn(replacement, task_map_text, count=1)
    if replacements != 1:
        raise ValueError(f"Could not locate the fractional task map in {task_map_path}")
    if updated_text == task_map_text:
        print(f"Fractional task map already contains {len(task_names)} tasks.")
        return
    if dry_run:
        print(f"[DRY RUN] Would synchronize {len(task_names)} fractional tasks in {task_map_path}")
        return
    _write_bddl(task_map_path, updated_text)
    print(f"Synchronized {len(task_names)} fractional tasks in {task_map_path}")


def _parse_remake_selections(values: list[str]) -> set[tuple[str, str]]:
    """Parse ``D2D:LV1``-style family/level selectors."""

    selections: set[tuple[str, str]] = set()
    for value in values:
        match = re.fullmatch(r"(D2D|D2S):(?:LV)?([123])", value, re.IGNORECASE)
        if match is None:
            raise ValueError(
                "Invalid remake selection {!r}; expected D2D:LV1 or D2S:LV3".format(
                    value
                )
            )
        selections.add((match.group(1).upper(), "LV{}".format(match.group(2))))
    return selections


def _fractional_identity(path: Path) -> tuple[str, str, str]:
    """Return suite, level, and stable canonical identity for a fractional file."""

    match = FRACTIONAL_TASK_ID_RE.match(path.stem)
    if match is None:
        raise ValueError(f"Unrecognized fractional BDDL name: {path.name}")
    suite = match.group("suite").upper()
    level = match.group("level").upper()
    identity = "{}_{}_{}".format(suite, level, match.group("task"))
    return suite, level, identity


def _canonical_source(bddl_root: Path, suite: str, identity: str) -> Path:
    """Resolve one canonical task by stable ``SUITE_LV#_###`` identity."""

    suite_dir = bddl_root / SUITE_DIR_BY_CODE[suite]
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"Suite directory does not exist: {suite_dir}")
    candidates = sorted(suite_dir.glob(f"{identity}_*.bddl"))
    if len(candidates) != 1:
        matches = ", ".join(path.name for path in candidates) or "none"
        raise ValueError(
            "Expected exactly one canonical source for {} in {}, found {}: {}".format(
                identity, suite_dir, len(candidates), matches
            )
        )
    return candidates[0]


def remake_existing(
    bddl_root: Path,
    template_dir: Path,
    output_dir: Path,
    selection_values: list[str],
) -> tuple[list[tuple[str, Path]], Counter[str]]:
    """Rebuild selected existing D2D/D2S variants from canonical BDDLs.

    Existing filenames are retained so the fractional task map remains stable.
    The complete canonical BDDL is copied before only the requested timing
    predicate values are restored from each filename suffix.
    """

    if not template_dir.is_dir():
        raise FileNotFoundError(f"Template directory does not exist: {template_dir}")
    selections = _parse_remake_selections(selection_values)
    templates_by_identity: dict[tuple[str, str, str], list[Path]] = {}
    for path in sorted(template_dir.glob("*.bddl")):
        suite, level, identity = _fractional_identity(path)
        if (suite, level) in selections:
            templates_by_identity.setdefault((suite, level, identity), []).append(path)

    missing_selections = selections - {
        (suite, level) for suite, level, _identity in templates_by_identity
    }
    if missing_selections:
        raise ValueError(
            "No existing fractional templates for: {}".format(
                ", ".join("{}:{}".format(*item) for item in sorted(missing_selections))
            )
        )

    selected_sources: list[tuple[str, Path]] = []
    generated_by_suite: Counter[str] = Counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    for (suite, _level, identity), templates in sorted(templates_by_identity.items()):
        parsed_templates: list[tuple[Path, int, int]] = []
        actual_pairs: set[tuple[int, int]] = set()
        for template in templates:
            suffix = FRACTIONAL_AT_SPEED_SUFFIX_RE.search(template.name)
            if suffix is None or not suffix.group("takt").isdigit():
                raise ValueError(f"Unsupported D2D/D2S tuning suffix: {template.name}")
            pair = (int(suffix.group("speed")), int(suffix.group("takt")))
            parsed_templates.append((template, *pair))
            actual_pairs.add(pair)
        if len(actual_pairs) != len(parsed_templates):
            raise ValueError(
                "{} has duplicate speed/takt variants: {}".format(
                    identity, sorted(actual_pairs)
                )
            )

        source = _canonical_source(bddl_root, suite, identity)
        source_text = source.read_text(encoding="utf-8")
        selected_sources.append((identity, source))
        for template, speed, takt in parsed_templates:
            parameterized, replacements = parameterize_at_speed(
                source_text, None if suite == "D2D" else speed, takt
            )
            if not replacements:
                raise ValueError(f"No AtSpeed predicate found in {source}")
            parameterized = parameterized.replace("_Ablation", "_Manipulation")
            _write_bddl(output_dir / template.name, parameterized)
            generated_by_suite[suite] += 1

    return selected_sources, generated_by_suite


def generate_sources(
    selections: list[tuple[str, str, Path]], output_dir: Path
) -> Counter[str]:
    """Write timing variants for an explicit set of canonical BDDL sources."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_by_suite: Counter[str] = Counter()

    for suite, level, source in selections:
        source_text = source.read_text(encoding="utf-8")

        if suite in {"libero_industry_d2d", "libero_industry_d2s"}:
            for speed, takt in _speed_takt_pairs(suite, level, source):
                is_d2d = suite == "libero_industry_d2d"
                parameterized, replacements = parameterize_at_speed(
                    source_text, None if is_d2d else speed, takt
                )
                if not replacements:
                    raise ValueError(f"No AtSpeed predicate found in {source}")
                # Required for Robosuite 1.5.2 task mapper compatibility.
                parameterized = parameterized.replace("_Ablation", "_Manipulation")
                target = output_path(source, output_dir, f"_v{speed}_t{takt}")
                _write_bddl(target, parameterized)
                generated_by_suite[suite] += 1
        else:
            parameterized_speed, speed_replacements = parameterize_load_slot_speed(
                source_text, _s2d_speed_code(source)
            )
            if not speed_replacements:
                raise ValueError(f"No LoadSlot predicate found in {source}")
            for service_time in SERVICE_TIMES:
                parameterized, replacements = parameterize_load_slot(
                    parameterized_speed, service_time
                )
                if not replacements:
                    raise ValueError(f"No LoadSlot predicate found in {source}")
                # Required for Robosuite 1.5.2 task mapper compatibility.
                parameterized = parameterized.replace("_Ablation", "_Manipulation")
                target = output_path(source, output_dir, f"_svc{service_time}")
                _write_bddl(target, parameterized)
                generated_by_suite[suite] += 1

    return generated_by_suite


def generate(bddl_root: Path, output_dir: Path) -> tuple[list[tuple[str, str, Path]], Counter[str]]:
    """Create the complete tuning set and return selections plus output counts."""

    selections = choose_sources(bddl_root)
    return selections, generate_sources(selections, output_dir)


def generate_four_object_set(bddl_root: Path, output_dir: Path, *, dry_run: bool) -> tuple[
    list[tuple[str, str, Path]], Counter[str]
]:
    """Synchronize the requested family-level groups to the balanced matrix."""

    selections = choose_four_object_sources(bddl_root)
    expected_paths = _generated_paths(selections, output_dir)
    selected_groups = set(FOUR_OBJECT_SOURCE_IDS)
    stale_paths = [
        path
        for path in sorted(output_dir.glob("*.bddl"))
        if _task_group(path) in selected_groups and path not in expected_paths
    ]

    print("Balanced source selections:")
    for suite, level, source in selections:
        print(f"  {suite} {level}: {source.name}")
    print(
        "Balanced output: {} BDDLs across {} family-level groups; {} stale BDDLs to remove.".format(
            len(expected_paths), len(selected_groups), len(stale_paths)
        )
    )
    if dry_run:
        for path in stale_paths:
            print(f"[DRY RUN] Would remove {path}")
        final_task_names = sorted(
            {
                *(path.stem for path in output_dir.glob("*.bddl") if _task_group(path) not in selected_groups),
                *(path.stem for path in expected_paths),
            }
        )
        _sync_fractional_task_map(
            output_dir, dry_run=True, task_names=final_task_names
        )
        return selections, Counter()

    for path in stale_paths:
        path.unlink()
    generated_by_suite = generate_sources(selections, output_dir)
    _sync_fractional_task_map(output_dir, dry_run=False)
    return selections, generated_by_suite


def repair_existing(bddl_root: Path, output_dir: Path) -> int:
    """Repair speed units in already-generated D2D/D2S fractional BDDLs.

    Earlier versions wrote the filename code (``3``) directly to ``AtSpeed``.
    D2S restores metres-per-second units; D2D copies only its canonical speed
    predicates so the filename-defined conveyor pair remains authoritative.
    """

    repaired = 0
    for path in sorted(output_dir.glob("*.bddl")):
        suffix = FRACTIONAL_AT_SPEED_SUFFIX_RE.search(path.name)
        if suffix is None:
            continue

        task_suite = path.stem.split("_", 1)[0].upper()
        if task_suite not in {"D2D", "D2S"}:
            continue

        text = path.read_text(encoding="utf-8")
        canonical_text = None
        if task_suite == "D2D":
            _suite, _level, identity = _fractional_identity(path)
            canonical_text = _canonical_source(bddl_root, task_suite, identity).read_text(
                encoding="utf-8"
            )
        updated, replacements = parameterize_at_speed(
            text,
            None if task_suite == "D2D" else int(suffix.group("speed")),
            suffix.group("takt"),
        )
        if not replacements:
            raise ValueError(f"No AtSpeed predicate found in {path}")
        if canonical_text is not None:
            updated, source_replacements = _copy_predicate_speed(
                updated, canonical_text, AT_SPEED_RE
            )
            updated, slot_replacements = _copy_predicate_speed(
                updated, canonical_text, LOAD_SLOT_RE
            )
            if not source_replacements or not slot_replacements:
                raise ValueError(f"Canonical D2D speed predicates are missing for {path}")

        if updated != text:
            _write_bddl(path, updated)
            repaired += 1
    return repaired


def parse_args() -> argparse.Namespace:
    root = repo_root()
    bddl_root = root / "libero" / "libero" / "bddl_files"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bddl-root",
        type=Path,
        default=bddl_root,
        help=f"BDDL root directory (default: {bddl_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=bddl_root / "fractional_tuning_set",
        help="Destination directory for generated BDDL files.",
    )
    parser.add_argument(
        "--repair-existing",
        action="store_true",
        help="Repair speed units in an existing fractional D2D/D2S BDDL directory.",
    )
    parser.add_argument(
        "--remake-existing",
        nargs="+",
        metavar="SUITE:LEVEL",
        help=(
            "Rebuild selected existing D2D/D2S family-level variants from canonical "
            "BDDLs while preserving their filenames, for example D2D:LV1 D2S:LV3."
        ),
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=bddl_root / "fractional_tuning_set",
        help="Existing fractional filenames to retain in --remake-existing mode.",
    )
    parser.add_argument(
        "--four-object-set",
        action="store_true",
        help=(
            "Synchronize S2D, D2D, and D2S levels 1--3 "
            "to their balanced four-source timing matrix, then update the task map."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview --four-object-set changes without writing or removing BDDLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bddl_root = args.bddl_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.four_object_set:
        if args.repair_existing or args.remake_existing:
            raise ValueError(
                "--four-object-set cannot be combined with --repair-existing or --remake-existing"
            )
        selections, generated_by_suite = generate_four_object_set(
            bddl_root, output_dir, dry_run=args.dry_run
        )
        if args.dry_run:
            print("[DRY RUN] No BDDLs or task maps were modified.")
            return
        print(f"Files generated: {sum(generated_by_suite.values())}")
        for suite in sorted(generated_by_suite):
            print(f"  {suite}: {generated_by_suite[suite]}")
        print(f"Output directory: {output_dir}")
        return
    if args.remake_existing:
        if args.repair_existing:
            raise ValueError("--remake-existing and --repair-existing are mutually exclusive")
        selected_sources, generated_by_suite = remake_existing(
            bddl_root,
            args.template_dir.resolve(),
            output_dir,
            args.remake_existing,
        )
        print("Selected canonical sources:")
        for identity, source in selected_sources:
            print(f"  {identity}: {source.name}")
        print(f"Files generated: {sum(generated_by_suite.values())}")
        for suite in sorted(generated_by_suite):
            print(f"  {suite}: {generated_by_suite[suite]}")
        print(f"Output directory: {output_dir}")
        return
    if args.repair_existing:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        repaired = repair_existing(bddl_root, output_dir)
        print(f"Repaired {repaired} fractional D2D/D2S BDDL files in: {output_dir}")
        return

    selections, generated_by_suite = generate(bddl_root, output_dir)

    print("Selected canonical task-001 source files:")
    for suite, level, source in selections:
        print(f"  {suite} {level}: {source.name}")
    print(f"Files processed: {len(selections)}")
    print(f"Files generated: {sum(generated_by_suite.values())}")
    for suite in SUITES:
        print(f"  {suite}: {generated_by_suite[suite]}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
