#!/usr/bin/env python3
"""Generate staged Panda init states for ablation D2-D6 suites."""

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "scripts" / "generate_inits_utils_all.py"
SUITES = tuple("libero_ablation_d{}".format(tier) for tier in range(2, 7))
def _suite_dir(bddl_root, suite):
    matches = [path for path in bddl_root.rglob(suite) if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            "Expected one '{}' directory under {}, found {}".format(
                suite, bddl_root, len(matches)
            )
        )
    count = len(list(matches[0].glob("*.bddl")))
    if count != 12:
        raise ValueError("Expected 12 BDDLs in {}, found {}".format(matches[0], count))
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description="Generate Panda init states for staged ablation D2-D6 BDDLs."
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        required=True,
        help="Staging root containing bddl_files/ablation_industry_flow.",
    )
    parser.add_argument("--suite", action="append", choices=SUITES)
    parser.add_argument("--target-states", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stage_root = args.stage_root.resolve()
    bddl_root = stage_root / "bddl_files"
    init_root = stage_root / "init_files"
    if not bddl_root.is_dir():
        parser.error("BDDL root does not exist: {}".format(bddl_root))
    if stage_root == REPO_ROOT or REPO_ROOT not in stage_root.parents:
        parser.error("--stage-root must be a dedicated directory inside the repository")
    if args.target_states <= 0:
        parser.error("--target-states must be positive")

    suites = tuple(args.suite or SUITES)
    try:
        for suite in suites:
            _suite_dir(bddl_root, suite)
    except ValueError as error:
        parser.error(str(error))

    for index, suite in enumerate(suites, start=1):
        print("\n=== Ablation init suite {}/{}: {} ===".format(index, len(suites), suite))
        command = [
            sys.executable,
            str(WORKER),
            "--suite",
            suite,
            "--embodiment",
            "panda",
            "--bddl-root",
            str(bddl_root),
            "--init-root",
            str(init_root),
            "--target-states",
            str(args.target_states),
            "--seed",
            str(args.seed),
        ]
        if args.overwrite:
            command.append("--overwrite")
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)

    print("\nCompleted {} staged ablation init suites under {}".format(len(suites), init_root))


if __name__ == "__main__":
    main()
