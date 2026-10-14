"""Small simulator-free checks for the object-centric Aubo generator."""

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_mimicgen_aubo as generator

SAFE_KINEMATICS = {
    "minimum_arm_joint_margin_rad": 0.2,
    "minimum_jacobian_sigma": 0.1,
    "maximum_jacobian_condition_number": 10.0,
    "j3_rad": 0.5,
    "tcp_reach_m": 0.5,
}


class MimicGenAuboTest(unittest.TestCase):
    def test_libero_original_recovery_requires_robosuite_14(self):
        self.assertEqual(generator.EXPECTED_ROBOSUITE_VERSION, "1.4.0")

    def test_registry_covers_all_original_libero_tasks(self):
        self.assertEqual(len(generator.TASKS), 40)
        self.assertEqual(len(generator.SIMPLE_TASKS), 20)
        self.assertEqual(
            sum(task["suite"] == "libero_object" for task in generator.SIMPLE_TASKS), 10
        )
        self.assertEqual(
            sum(task["suite"] == "libero_spatial" for task in generator.SIMPLE_TASKS), 10
        )
        self.assertEqual(len({task["case_id"] for task in generator.TASKS}), 40)

    def test_spatial_task_uses_bddl_object_of_interest(self):
        spatial = next(
            task for task in generator.TASKS if task["case_id"] == "spatial_cookie_box"
        )
        self.assertEqual(spatial["object_name"], "akita_black_bowl_1")
        self.assertEqual(spatial["goal_name"], "plate_1")
        self.assertNotIn(
            "goal_name",
            next(task for task in generator.TASKS if task["case_id"] == "object_milk"),
        )

    def test_object_transform_preserves_relative_eef_pose(self):
        source_object = np.eye(4)
        source_object[:3, 3] = [0.1, 0.2, 0.3]
        target_object = np.eye(4)
        target_object[:3, 3] = [-0.2, 0.4, 0.1]
        source_eef = np.repeat(np.eye(4)[None], 2, axis=0)
        source_eef[0, :3, 3] = [0.15, 0.2, 0.4]
        source_eef[1, :3, 3] = [0.2, 0.25, 0.45]
        transformed = generator.transform_segment(
            target_object, source_eef, source_object
        )
        source_relative = np.linalg.inv(source_object) @ source_eef[1]
        target_relative = np.linalg.inv(target_object) @ transformed[1]
        self.assertTrue(np.allclose(source_relative, target_relative))

    def test_reference_substeps_preserve_endpoints_and_scale_feedforward(self):
        class Rotation:
            @staticmethod
            def from_rotvec(_value):
                return SimpleNamespace(as_matrix=lambda: np.eye(3))

        converter = SimpleNamespace(
            get_orientation_error=lambda target, current: np.zeros(3),
            R=Rotation,
        )
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[1, 0, 3] = 0.2
        gripper = np.asarray([-1.0, 1.0])
        feedforward = np.asarray([[0.2] * 6, [0.4] * 6])
        dense_poses, dense_gripper, dense_feedforward = (
            generator._densify_reference_trajectory(
                poses, gripper, feedforward, 2, converter
            )
        )
        self.assertEqual(dense_poses.shape, (4, 4, 4))
        self.assertTrue(np.allclose(dense_poses[-1], poses[-1]))
        self.assertTrue(np.array_equal(dense_gripper, [-1.0, -1.0, 1.0, 1.0]))
        self.assertTrue(
            np.allclose(dense_feedforward.reshape(2, 2, 6).sum(axis=1), feedforward)
        )

    def test_gripper_events_define_one_pick_place_split(self):
        actions = np.zeros((9, 7), dtype=np.float64)
        actions[:, -1] = [-1, -1, 1, 1, 1, 1, -1, -1, -1]
        self.assertEqual(generator._segment_events(actions), (2, 6))

    def test_failed_identities_are_unique_sorted_and_attempts_are_repeatable(self):
        rows = {
            "demo_10": {"terminal_task_success": False},
            "demo_2": {"terminal_task_success": False},
            "demo_1": {"terminal_task_success": True},
        }
        self.assertEqual(generator._failed_demo_names(rows), ["demo_2", "demo_10"])
        first = generator._attempt_rng(0, 3, "demo_2", 1).rand(4)
        repeated = generator._attempt_rng(0, 3, "demo_2", 1).rand(4)
        next_attempt = generator._attempt_rng(0, 3, "demo_2", 2).rand(4)
        self.assertTrue(np.array_equal(first, repeated))
        self.assertFalse(np.array_equal(first, next_attempt))

    def test_clipping_limits_follow_successes_from_the_same_raw_task(self):
        rows = {
            "demo_0": {
                "terminal_task_success": True,
                "clipped_component_pct": 8.5,
                "max_clipped_run_steps": 21,
            },
            "demo_1": {
                "terminal_task_success": False,
                "clipped_component_pct": 99.0,
                "max_clipped_run_steps": 999,
            },
            "demo_2": {
                "terminal_task_success": True,
                "clipped_component_pct": 6.0,
                "max_clipped_run_steps": 30,
            },
        }
        self.assertEqual(generator._raw_success_clipping_limits(rows), (8.5, 30))

    def test_recovered_source_loader_accepts_only_the_summary_partition(self):
        task = {
            "suite": "libero_spatial",
            "case_id": "spatial_test",
            "task_stem": "task_demo",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / task["case_id"]
            task_root.mkdir()
            merged = task_root / "task_demo.generated.hdf5"
            with generator.h5py.File(merged, "w") as output:
                data = output.create_group("data")
                data.attrs["state_action_observation_qualified"] = True
                data.attrs["kinematically_qualified"] = True
                demo = data.create_group("demo_2")
                demo.create_dataset("states", data=np.zeros((2, 4)))
                demo.create_dataset("actions", data=np.zeros((2, 7)))
                demo.create_dataset("rewards", data=np.ones(2))
                demo.attrs["recovered_from_failed_identity"] = True
                demo.attrs["task_case_id"] = task["case_id"]
            (task_root / "summary.json").write_text(
                json.dumps(
                    {
                        **task,
                        "all_selected_identities_processed": True,
                        "recovered_identity_count": 1,
                        "recovered_demo_ids": ["demo_2"],
                        "merged_hdf5": str(merged),
                    }
                ),
                encoding="utf-8",
            )
            loaded = generator._recovered_source_demos(root, task)
            self.assertEqual(set(loaded), {"demo_2"})
            self.assertEqual(loaded["demo_2"]["actions"].shape, (2, 7))
            self.assertEqual(
                generator._recovered_source_demos(
                    root, {**task, "case_id": "task_without_recovery"}
                ),
                {},
            )

    def test_shared_initializer_restores_exact_stored_aubo_state(self):
        expected = np.arange(4, dtype=np.float64)

        class Sim:
            state = expected + 0.1

            def set_state_from_flattened(self, value):
                self.state = np.asarray(value).copy()

            def forward(self):
                pass

        target = SimpleNamespace(sim=Sim())
        converter = SimpleNamespace(
            initialize_target_replay=lambda *args, **kwargs: None,
            get_target_sim_state=lambda environment: environment.sim.state.copy(),
            _get_base_env=lambda environment: environment,
        )
        with mock.patch.object(
            generator, "_scene_values", return_value=(np.zeros(1), 0)
        ):
            initialization = generator._initialize(
                object(),
                target,
                np.zeros(2),
                -1.0,
                0,
                converter,
                expected_target_state=expected,
            )
        self.assertTrue(np.array_equal(initialization["state"], expected))
        self.assertTrue(initialization["target_state_restored_exactly"])

    def test_admission_uses_online_stable_success_not_replay(self):
        generation = {
            "terminal_task_success": True,
            "terminal_success_streak": 5,
            "finite": True,
            "bounded_actions": True,
            "minimum_arm_joint_margin_rad": 0.2,
            "minimum_jacobian_sigma": 0.1,
            "maximum_jacobian_condition_number": 10.0,
            "clipped_cartesian_component_pct": 1.0,
            "max_consecutive_clipped_steps": 1,
            "initial_state": np.zeros(4),
            "initial_scene": np.ones(7),
            "states": np.zeros((3, 4)),
            "actions": np.zeros((5, 7)),
        }
        accepted, reasons = generator._qualified(
            generation, 0.03, 5, 5.0, 4, 5
        )
        self.assertTrue(accepted, reasons)

        unstable = dict(generation)
        unstable["terminal_success_streak"] = 4
        accepted, reasons = generator._qualified(unstable, 0.03, 5, 5.0, 4, 5)
        self.assertFalse(accepted)
        self.assertIn("generation_success_dwell", reasons)

        clipped = dict(generation)
        clipped["clipped_cartesian_component_pct"] = 5.1
        accepted, reasons = generator._qualified(
            clipped, 0.03, 5, 5.0, 4, 5
        )
        self.assertFalse(accepted)
        self.assertIn("generation_clipped_component_pct", reasons)

        overlong = dict(generation)
        overlong["actions"] = np.zeros((6, 7))
        accepted, reasons = generator._qualified(overlong, 0.03, 5, 5.0, 4, 5)
        self.assertFalse(accepted)
        self.assertIn("generation_horizon", reasons)

        singular = dict(generation)
        singular["minimum_jacobian_sigma"] = 0.019
        accepted, reasons = generator._qualified(singular, 0.03, 5, 5.0, 4, 5)
        self.assertFalse(accepted)
        self.assertIn("generation_jacobian_sigma", reasons)

    def test_raw_admission_requires_final_five_successes(self):
        actions = np.zeros((6, 7))
        actions[:, 0] = 0.1
        stable = generator._raw_admission_reasons(
            {"terminal_task_success": True}, actions, np.ones(6)
        )
        unstable = generator._raw_admission_reasons(
            {"terminal_task_success": True}, actions, np.array([0, 0, 1, 1, 1, 1])
        )
        self.assertEqual(stable, [])
        self.assertEqual(unstable, ["final_five_not_successful"])

    def test_noop_is_skipped_before_environment_step(self):
        class Environment:
            def __init__(self):
                self.step_count = 0

            def step(self, _action):
                self.step_count += 1
                return None, 0.0, False, {}

        converter = SimpleNamespace(
            get_eef_pose=lambda _environment: (np.zeros(3), np.eye(3)),
            calculate_tracking_action=lambda *_args, **_kwargs: (
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
            ),
            get_target_sim_state=lambda _environment: np.zeros(2),
            _task_success=lambda _environment: False,
        )
        environment = Environment()
        rollout = generator._new_rollout(
            {"state": np.zeros(2), "scene": np.zeros(2)}
        )
        pose = np.eye(4)
        with mock.patch.object(
            generator, "_kinematic_metrics", return_value=SAFE_KINEMATICS
        ):
            generator._execute(
                environment,
                [pose, pose],
                [-1.0, -1.0],
                0,
                rollout,
                converter,
            )
        self.assertEqual(environment.step_count, 1)
        self.assertEqual(len(rollout["actions"]), 1)
        self.assertEqual(rollout["skipped_noop_count"], 1)

    def test_success_dwell_is_validated_without_adding_dataset_rows(self):
        environment = SimpleNamespace(step_count=0)

        def step(_action):
            environment.step_count += 1

        environment.step = step
        converter = SimpleNamespace(
            get_eef_pose=lambda _environment: (np.zeros(3), np.eye(3)),
            calculate_tracking_action=lambda *_args: (np.zeros(3), np.zeros(3)),
            _task_success=lambda _environment: True,
        )
        rollout = generator._new_rollout(
            {"state": np.zeros(2), "scene": np.zeros(2)}
        )
        rollout["terminal_success_streak"] = 1
        with mock.patch.object(
            generator, "_kinematic_metrics", return_value=SAFE_KINEMATICS
        ):
            generator._validate_success_dwell(
                environment, -1.0, 20, 5, rollout, converter
            )
        self.assertEqual(environment.step_count, 4)
        self.assertEqual(rollout["success_validation_steps"], 4)
        self.assertEqual(rollout["terminal_success_streak"], 5)
        self.assertEqual(rollout["actions"], [])

    def test_merge_preserves_failed_demo_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "prior.hdf5"
            with generator.h5py.File(prior, "w") as output:
                data = output.create_group("data")
                for demo_name in ("demo_2", "demo_10"):
                    data.attrs["env_args"] = "{}"
                    data.create_group(demo_name).create_dataset(
                        "actions", data=np.zeros((1, 7))
                    )
            merged = root / "merged.hdf5"
            generator._merge_episodes([prior], merged)
            with generator.h5py.File(merged, "r") as output:
                self.assertEqual(set(output["data"]), {"demo_2", "demo_10"})

    def test_resume_summary_selects_completed_task_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged = root / "accepted.hdf5"
            with generator.h5py.File(merged, "w") as output:
                output.create_group("data").create_group("demo_2")
            task = {
                "case_id": "spatial_cookie_box",
                "all_selected_identities_processed": False,
                "recovered_demo_ids": ["demo_2"],
                "unrecovered_demo_ids": ["demo_10"],
                "not_attempted_demo_ids": ["demo_11"],
                "merged_hdf5": str(merged),
            }
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "all_selected_identities_processed": False,
                        "errored_task_count": 0,
                        "errors": [],
                        "tasks": [task],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                generator._resume_tasks(root)["spatial_cookie_box"], task
            )
            carried, remaining, recovered = generator._resume_partition(
                task, ["demo_2", "demo_10", "demo_11"]
            )
            self.assertEqual(carried, ["demo_2"])
            self.assertEqual(remaining, ["demo_10", "demo_11"])
            self.assertEqual(recovered, merged)

            with self.assertRaisesRegex(ValueError, "does not partition"):
                generator._resume_partition(
                    {**task, "unrecovered_demo_ids": ["demo_2"]},
                    ["demo_2", "demo_11"],
                )

    def test_resume_allows_only_explicitly_errored_tasks_to_start_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            completed = {
                "case_id": "goal_turn_on_stove",
                "recovered_demo_ids": [],
                "unrecovered_demo_ids": ["demo_0"],
                "not_attempted_demo_ids": [],
            }
            summary = {
                "errored_task_count": 1,
                "errors": [{"case_id": "goal_wine_cabinet"}],
                "tasks": [completed],
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(
                generator._resume_tasks(root),
                {
                    "goal_turn_on_stove": completed,
                    "goal_wine_cabinet": None,
                },
            )

            summary["errored_task_count"] = 0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete or errored"):
                generator._resume_tasks(root)


if __name__ == "__main__":
    unittest.main()
