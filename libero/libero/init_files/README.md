# Embodiment-specific init states

The existing suite directories directly under this folder remain the default
Panda init states. This preserves compatibility with original LIBERO users.

Additional robot/gripper combinations live below an explicit embodiment root:

```text
init_files/
├── libero_industry_d2s/                 # Panda (legacy default)
└── embodiments/
    └── aubo_i5_ctag2f90d/
        ├── manifest.json
        ├── libero_industry_d2d/
        ├── libero_industry_d2s/
        ├── libero_industry_s2d/
        └── fractional_tuning_set/
```

Init states are flattened MuJoCo simulator states. They are specific to the
robot, gripper, task XML/BDDL, object assets, Robosuite/MuJoCo versions, and the
code revision recorded in `manifest.json`. Never merge Panda and Aubo arrays.

## Generation

Run from the repository root in the `libero-industry` environment. A dry run
selects tasks and shows output paths without creating states:

```bash
conda run -n libero-industry python scripts/generate_inits_utils_all.py \
  --suite fractional_tuning_set \
  --embodiment aubo_i5_ctag2f90d \
  --level 1 --family D2S S2D --target-states 20 --dry-run
```

Generating the logical training split writes each state to its physical D2D,
D2S, or S2D task directory:

```bash
conda run -n libero-industry python scripts/generate_inits_utils_all.py \
  --suite libero_industry_train \
  --embodiment aubo_i5_ctag2f90d \
  --target-states 20
```

Every selected physical BDDL directory must exist below `--bddl-root`. If a
canonical suite (notably D2D in some checkouts) exists only in a verified
benchmark staging run, pass that run's `bddl_files` directory explicitly. Do
not silently fall back to `libero_industry_draft`.

Use `--task <exact-task-name>` for a one-task smoke run. The generator resumes
complete state pairs by default; `--overwrite` explicitly regenerates them.

## Validation

Audit file pairs, shapes, counts, and restore one state per task without video:

```bash
conda run -n libero-industry python scripts/validate_init_states.py \
  --suite libero_industry_train \
  --embodiment aubo_i5_ctag2f90d \
  --target-states 20 --reset-smoke
```

Training and evaluation select a profile with Hydra, for example
`embodiment=aubo_i5_ctag2f90d`. An explicit `init_states_folder` is treated as
the final profile root and is useful for staged validation.
