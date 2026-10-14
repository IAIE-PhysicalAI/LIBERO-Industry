import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import materialize_recovered_dataset as materializer


def _demo(data, name, wrist, success):
    horizon = 5
    demo = data.create_group(name)
    actions = np.zeros((horizon, 7))
    actions[:, 0] = 0.1
    demo.create_dataset("actions", data=actions)
    states = np.arange(horizon * 3, dtype=np.float64).reshape(horizon, 3)
    demo.create_dataset("states", data=states)
    demo.create_dataset("rewards", data=np.full(horizon, float(success)))
    demo.create_dataset("dones", data=np.full(horizon, int(success)))
    demo.attrs["num_samples"] = horizon
    demo.attrs["init_state"] = states[0]
    obs = demo.create_group("obs")
    obs.create_dataset("agentview_image", shape=(horizon, 256, 256, 3), dtype=np.uint8)
    obs.create_dataset(wrist, shape=(horizon, 256, 256, 3), dtype=np.uint8)
    obs.create_dataset("robot0_joint_pos", data=np.zeros((horizon, 6)))
    obs.create_dataset("robot0_gripper_qpos", data=np.zeros((horizon, 6)))
    obs.create_dataset("robot0_eef_pos", data=np.zeros((horizon, 3)))
    obs.create_dataset(
        "robot0_eef_quat", data=np.tile([0.0, 0.0, 0.0, 1.0], (horizon, 1))
    )


class MaterializeRecoveredDatasetTest(unittest.TestCase):
    def test_generated_demo_can_use_unrecorded_stability_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generated.hdf5"
            with h5py.File(path, "w") as file:
                data = file.create_group("data")
                _demo(data, "demo_0", "gripper0_eye_in_hand_image", False)
                data["demo_0/rewards"][-1] = 1.0
                data["demo_0"].attrs["terminal_success_streak"] = 5
                horizon = materializer._validate_demo(data["demo_0"])
            self.assertEqual(horizon, 5)

    def test_gate5_partition_is_authoritative_for_successful_raw_demo(self):
        raw_retained, targets = materializer._gate5_partition(
            {
                "recovered_demo_ids": ["demo_1"],
                "unrecovered_demo_ids": [],
                "not_attempted_demo_ids": [],
                "target_identity_count": 1,
                "raw_retained_identity_count": 1,
            },
            {"demo_0", "demo_1"},
            "libero_object",
            "task",
        )
        self.assertEqual(raw_retained, {"demo_0"})
        self.assertEqual(targets, {"demo_1"})
        with self.assertRaisesRegex(ValueError, "duplicate targets"):
            materializer._gate5_partition(
                {
                    "recovered_demo_ids": ["demo_1"],
                    "unrecovered_demo_ids": ["demo_1"],
                    "not_attempted_demo_ids": [],
                    "target_identity_count": 1,
                    "raw_retained_identity_count": 1,
                },
                {"demo_0", "demo_1"},
                "libero_object",
                "task",
            )

    def test_preflight_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / materializer.OUTPUT_ROOT_BASENAME
            preflight = {
                "task_count": 10,
                "identity_count": 500,
                "raw_retained_count": 400,
                "recovered_identity_count": 100,
            }
            with mock.patch.object(
                materializer,
                "_prepare_inputs",
                return_value=([], {}, {}, preflight),
            ):
                result = materializer.materialize(
                    root / materializer.RAW_ROOT_BASENAME,
                    [root / "recovery"],
                    output,
                    ["libero_object"],
                    preflight_only=True,
                )
            self.assertEqual(result, preflight)
            self.assertFalse(output.exists())

    def test_collects_split_suite_and_task_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            goal = {
                "case_id": "goal_case",
                "suite": "libero_goal",
                "task_stem": "goal_task",
                "all_selected_identities_processed": True,
            }
            libero10 = {
                "case_id": "libero10_case",
                "suite": "libero_10",
                "task_stem": "libero10_task",
                "all_selected_identities_processed": True,
            }
            (first / "summary.json").write_text(
                json.dumps(goal), encoding="utf-8"
            )
            (second / "summary.json").write_text(
                json.dumps({"tasks": [libero10]}), encoding="utf-8"
            )

            tasks = materializer._recovery_tasks([first, second])

            self.assertEqual(tasks[("libero_goal", "goal_task")], goal)
            self.assertEqual(tasks[("libero_10", "libero10_task")], libero10)

    def test_ignores_nested_work_and_archive_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authoritative = {
                "case_id": "spatial_case",
                "suite": "libero_spatial",
                "task_stem": "spatial_task",
                "recovered_demo_ids": ["demo_1"],
            }
            stale = {**authoritative, "recovered_demo_ids": []}
            (root / "summary.json").write_text(
                json.dumps(authoritative), encoding="utf-8"
            )
            for directory in ("work", "archive"):
                scratch = root / directory / "retry"
                scratch.mkdir(parents=True)
                (scratch / "summary.json").write_text(
                    json.dumps(stale), encoding="utf-8"
                )

            tasks = materializer._recovery_tasks([root])

            self.assertEqual(tasks[("libero_spatial", "spatial_task")], authoritative)

    def test_collects_completed_final_round_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            final = root / "final"
            base.mkdir()
            final.mkdir()
            base_hdf5 = root / "base.hdf5"
            final_hdf5 = root / "final.hdf5"
            (base / "summary.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "suite": "libero_object",
                                "task_stem": "task",
                                "recovered_demo_ids": ["demo_1"],
                                "merged_hdf5": str(base_hdf5),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (final / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "results": [
                            {
                                "status": "recovered",
                                "suite": "libero_object",
                                "task_stem": "task",
                                "demo": "demo_2",
                                "merged_hdf5": str(final_hdf5),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            sources = materializer._recovery_sources([base, final])

            self.assertEqual(
                sources[("libero_object", "task", "demo_1")],
                base_hdf5.resolve(),
            )
            self.assertEqual(
                sources[("libero_object", "task", "demo_2")],
                final_hdf5.resolve(),
            )

    def test_combines_gate5_retained_raw_and_recovered_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / materializer.RAW_ROOT_BASENAME
            recovery = root / "recovery"
            output = root / materializer.OUTPUT_ROOT_BASENAME
            suite = "libero_object"
            stem = "task_demo"
            (raw / suite).mkdir(parents=True)
            (recovery / suite / "case").mkdir(parents=True)
            raw_hdf5 = raw / suite / f"{stem}.hdf5"
            recovered_hdf5 = recovery / suite / "case" / f"{stem}.generated.hdf5"
            env_args = json.dumps({"env_name": "test"})
            for path, wrist, demos in (
                (
                    raw_hdf5,
                    "gripper0_eye_in_hand_image",
                    (("demo_0", True), ("demo_1", False)),
                ),
                (
                    recovered_hdf5,
                    "gripper0_right_eye_in_hand_image",
                    (("demo_1", True),),
                ),
            ):
                with h5py.File(path, "w") as file:
                    data = file.create_group("data")
                    data.attrs["env_args"] = env_args
                    data.attrs["bddl_file_name"] = "task.bddl"
                    if path == recovered_hdf5:
                        data.attrs["state_action_observation_qualified"] = True
                    for name, success in demos:
                        _demo(data, name, wrist, success)

            rows = [
                {
                    "trajectory": f"demo_{index}",
                    "terminal_task_success": True,
                }
                for index in range(50)
            ]
            # The production contract is 50 identities. Add minimal failed raw
            # groups so the synthetic report and HDF5 agree.
            with h5py.File(raw_hdf5, "r+") as file:
                for index in range(2, 50):
                    _demo(file["data"], f"demo_{index}", "gripper0_eye_in_hand_image", True)
            report = {"trajectories": rows}
            (raw / suite / f"{stem}.conversion_report.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            output.parent.mkdir(exist_ok=True)
            output.mkdir()
            (output / suite).mkdir()
            sources = {(suite, stem, "demo_1"): recovered_hdf5}
            gate5_task = {
                "recovered_demo_ids": ["demo_1"],
                "unrecovered_demo_ids": [],
                "not_attempted_demo_ids": [],
                "target_identity_count": 1,
                "raw_retained_identity_count": 49,
            }
            result = materializer._materialize_task(
                suite,
                raw_hdf5,
                raw / suite / f"{stem}.conversion_report.json",
                gate5_task,
                sources,
                output / suite / f"{stem}.hdf5",
            )
            self.assertEqual(result["final_demo_count"], 50)
            self.assertEqual(result["raw_retained_count"], 49)
            self.assertEqual(result["recovered_identity_count"], 1)
            with h5py.File(output / suite / f"{stem}.hdf5", "r") as file:
                self.assertEqual(
                    set(file["data"]), {f"demo_{index}" for index in range(50)}
                )
                self.assertEqual(
                    file["data/demo_1"].attrs["materialization_origin"],
                    "recovered_gate5_target",
                )
                for name in file["data"]:
                    self.assertIn(
                        "gripper0_eye_in_hand_image", file[f"data/{name}/obs"]
                    )
                    self.assertNotIn(
                        "gripper0_right_eye_in_hand_image", file[f"data/{name}/obs"]
                    )

            incomplete_task = {
                **gate5_task,
                "unrecovered_demo_ids": ["demo_2"],
                "target_identity_count": 2,
                "raw_retained_identity_count": 48,
            }
            with self.assertRaisesRegex(ValueError, "missing accepted recoveries"):
                materializer._materialize_task(
                    suite,
                    raw_hdf5,
                    raw / suite / f"{stem}.conversion_report.json",
                    incomplete_task,
                    sources,
                    output / suite / "strict.hdf5",
                )
            partial = materializer._materialize_task(
                suite,
                raw_hdf5,
                raw / suite / f"{stem}.conversion_report.json",
                incomplete_task,
                sources,
                output / suite / "partial.hdf5",
                allow_incomplete=True,
            )
            self.assertEqual(partial["final_demo_count"], 49)
            self.assertEqual(partial["unrecovered_demo_ids"], ["demo_2"])


if __name__ == "__main__":
    unittest.main()
