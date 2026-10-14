"""Small simulator-free contracts for the LIBERO-10 phase runner."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_mimicgen_aubo as phase

SAFE_KINEMATICS = {
    "minimum_arm_joint_margin_rad": 0.2,
    "minimum_jacobian_sigma": 0.1,
    "maximum_jacobian_condition_number": 10.0,
    "j3_rad": 0.5,
    "tcp_reach_m": 0.5,
}


class Libero10PhaseRunnerTest(unittest.TestCase):
    def test_libero10_has_an_independent_14_recovery_entry_point(self):
        self.assertEqual(phase.EXPECTED_ROBOSUITE_VERSION, "1.4.0")
        self.assertTrue(callable(phase.recover_libero10_trajectory))

    def test_terminal_object_phase_can_finish_without_release(self):
        spec = phase._phase("object", "book", ("in", "book", "caddy"), release=False)
        self.assertEqual(
            phase._phase_window(
                spec,
                {20: "close"},
                predicate_end=80,
                previous_end=0,
                horizon=100,
                final_phase=True,
            ),
            (0, 100),
        )

    def test_intermediate_object_phase_requires_release(self):
        spec = phase._phase("object", "first", ("in", "first", "basket"))
        with self.assertRaisesRegex(ValueError, "no release"):
            phase._phase_window(
                spec,
                {20: "close"},
                predicate_end=70,
                previous_end=0,
                horizon=100,
                final_phase=False,
            )
        self.assertEqual(
            phase._phase_window(
                spec,
                {20: "close", 75: "open"},
                predicate_end=70,
                previous_end=0,
                horizon=100,
                final_phase=False,
            ),
            (0, 76),
        )

    def test_second_object_starts_after_first_release(self):
        spec = phase._phase(
            "object", "second", ("on", "second", "plate"), release=False
        )
        self.assertEqual(
            phase._phase_window(
                spec,
                {20: "close", 60: "open", 90: "close", 130: "open"},
                predicate_end=120,
                previous_end=61,
                horizon=150,
                final_phase=True,
            ),
            (61, 131),
        )

    def test_fixture_phase_uses_previous_phase_boundary(self):
        spec = phase._phase("fixture", "microwave", ("close", "microwave"))
        self.assertEqual(
            phase._phase_window(
                spec,
                {},
                predicate_end=140,
                previous_end=81,
                horizon=160,
                final_phase=True,
            ),
            (81, 140),
        )

    def test_push_phase_needs_no_gripper_event(self):
        spec = phase._phase(
            "push", "plate", ("on", "plate", "front_of_stove")
        )
        self.assertEqual(
            phase._phase_window(
                spec,
                {},
                predicate_end=90,
                previous_end=0,
                horizon=120,
                final_phase=True,
            ),
            (0, 90),
        )

    def test_push_trajectory_starts_at_actor_and_ends_at_destination(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = [0.0, 0.5, 1.0]
        source_actor = np.eye(4)
        source_destination = np.eye(4)
        source_destination[0, 3] = 1.0
        current_actor = np.eye(4)
        current_actor[0, 3] = 0.2
        current_actor[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        target_destination = np.eye(4)
        target_destination[0, 3] = 1.0
        anchored = phase._anchor_push_trajectory(
            poses,
            current_actor,
            source_actor,
            source_destination,
            target_destination,
            0.02,
        )
        self.assertAlmostEqual(anchored[0, 0, 3], 0.2)
        self.assertAlmostEqual(anchored[-1, 0, 3], 1.02)

    def test_clearance_transition_lifts_before_traversing(self):
        class Rotation:
            @staticmethod
            def from_rotvec(_value):
                return SimpleNamespace(as_matrix=lambda: np.eye(3))

        converter = SimpleNamespace(
            get_orientation_error=lambda target, current: np.zeros(3),
            R=Rotation,
        )
        current = np.eye(4)
        current[:3, 3] = [0.0, 0.0, 0.2]
        target = np.eye(4)
        target[:3, 3] = [0.4, 0.3, 0.1]
        poses = phase._clearance_transition_poses(
            current, target, 0.5, 2, converter
        )
        self.assertEqual(len(poses), 4)
        self.assertTrue(np.allclose(poses[1, :3, 3], [0.0, 0.0, 0.5]))
        self.assertTrue(np.allclose(poses[-1, :3, 3], [0.4, 0.3, 0.5]))
        self.assertLessEqual(float(np.max(poses[:, 2, 3])), 0.5)

    def test_target_native_feedforward_is_preserved_at_zero_pose_error(self):
        class Environment:
            def step(self, action):
                self.action = np.asarray(action)
                return {}, 0.0, False, {}

        converter = SimpleNamespace(
            get_eef_pose=lambda environment: (np.zeros(3), np.eye(3)),
            get_target_sim_state=lambda environment: np.zeros(2),
            calculate_tracking_action=lambda *args, **kwargs: (
                np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
            ),
            _task_success=lambda environment: False,
        )
        environment = Environment()
        rollout = phase._new_rollout(
            {"state": np.zeros(2), "scene": np.zeros(1)}
        )
        feedforward = np.asarray([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
        with mock.patch.object(
            phase, "_kinematic_metrics", return_value=SAFE_KINEMATICS
        ):
            phase._execute(
                environment,
                np.eye(4)[None],
                np.asarray([-1.0]),
                0,
                rollout,
                converter,
                feedforward=feedforward,
            )
        self.assertTrue(
            np.array_equal(environment.action[:6], feedforward[0])
        )
        self.assertEqual(environment.action[-1], -1.0)

    def test_dependent_phases_prefer_the_same_successful_source_demo(self):
        library = [
            {"demo": "demo_1", "value": 1},
            {"demo": "demo_2", "value": 2},
        ]
        self.assertEqual(
            phase._compatible_phase_library(library, "demo_2"), [library[1]]
        )
        self.assertIs(
            phase._compatible_phase_library(library, "demo_3"), library
        )
        self.assertIs(
            phase._compatible_phase_library(library, "demo_2", True), library
        )
        tasks = {task["case_id"]: task for task in phase.LIBERO10_TASKS}
        self.assertTrue(tasks["libero10_two_mugs"]["independent_phase_sources"])

    def test_admission_requires_online_phase_predicates_not_replay(self):
        generation = {"phase_checks_passed": True}
        with mock.patch.object(
            phase,
            "_qualified",
            return_value=(True, []),
        ):
            accepted, reasons = phase._stagewise_qualified(
                generation,
                SimpleNamespace(
                    min_arm_joint_margin_rad=0.03,
                    minimum_jacobian_sigma=0.02,
                    maximum_jacobian_condition=80.0,
                    success_dwell_steps=5,
                ),
                1.0,
                1,
                100,
            )
        self.assertTrue(accepted)
        self.assertEqual(reasons, [])

        generation["phase_checks_passed"] = False
        with mock.patch.object(phase, "_qualified", return_value=(True, [])):
            accepted, reasons = phase._stagewise_qualified(
                generation,
                SimpleNamespace(
                    min_arm_joint_margin_rad=0.03,
                    minimum_jacobian_sigma=0.02,
                    maximum_jacobian_condition=80.0,
                    success_dwell_steps=5,
                ),
                1.0,
                1,
                100,
            )
        self.assertFalse(accepted)
        self.assertEqual(reasons, ["generation_phase_predicate"])

    def test_every_task_declares_all_phases(self):
        self.assertEqual(len(phase.LIBERO10_TASKS), 10)
        self.assertTrue(all(task["phases"] for task in phase.LIBERO10_TASKS))

    def test_two_moka_follows_the_recorded_source_phase_order(self):
        task = next(
            task
            for task in phase.LIBERO10_TASKS
            if task["case_id"] == "libero10_two_moka"
        )
        self.assertEqual(
            [row["actor"] for row in task["phases"]],
            ["moka_pot_2", "moka_pot_1"],
        )
        self.assertEqual(
            [row["release_required"] for row in task["phases"]],
            [True, False],
        )

    def test_goal_tasks_cover_all_ten_files_and_keep_drawer_open_as_prerequisite(self):
        self.assertEqual(len(phase.GOAL_TASKS), 10)
        self.assertTrue(
            all(task["suite"] == "libero_goal" for task in phase.GOAL_TASKS)
        )
        drawer = next(
            task
            for task in phase.GOAL_TASKS
            if task["case_id"] == "goal_top_drawer_bowl"
        )
        self.assertFalse(drawer["phases"][0]["goal_required"])
        self.assertTrue(drawer["phases"][1]["goal_required"])
        push = next(
            task for task in phase.GOAL_TASKS if task["case_id"] == "goal_push_plate"
        )
        self.assertEqual(push["phases"][0]["kind"], "push")

    def test_temporary_drawer_prerequisite_is_not_required_after_placement(self):
        drawer = next(
            task
            for task in phase.GOAL_TASKS
            if task["case_id"] == "goal_top_drawer_bowl"
        )
        first, second = drawer["phases"]
        self.assertIn(
            tuple(first["predicate"]),
            phase._phase_snapshot_predicates(drawer, [], first),
        )
        after_first = []
        if first["goal_required"]:
            after_first.append(tuple(first["predicate"]))
        second_checks = phase._phase_snapshot_predicates(drawer, after_first, second)
        self.assertNotIn(tuple(first["predicate"]), second_checks)
        self.assertIn(tuple(second["predicate"]), second_checks)
        self.assertIn(
            tuple(first["predicate"]),
            phase._replay_phase_snapshot_predicates(drawer, 0),
        )
        self.assertNotIn(
            tuple(first["predicate"]),
            phase._replay_phase_snapshot_predicates(drawer, 1),
        )

    def test_object_phase_splits_after_live_lift(self):
        poses = np.repeat(np.eye(4)[None], 8, axis=0)
        poses[4:, 2, 3] = 0.006
        spec = phase._phase("object", "mug", ("on", "mug", "plate"))
        self.assertEqual(
            phase._object_phase_split(spec, {2: "close", 7: "open"}, poses, 0, 8),
            4,
        )

    def test_live_references_use_simulator_site_and_articulation_body(self):
        class Model:
            jnt_bodyid = np.array([2], dtype=np.int64)

            def site_name2id(self, name):
                if name != "basket_region":
                    raise ValueError(name)
                return 0

            def body_name2id(self, name):
                return {"plate_1": 0}[name]

            def body_id2name(self, body_id):
                return ["plate_1", "microwave_1", "microwave_1_microdoorroot"][
                    body_id
                ]

            def joint_name2id(self, name):
                if name != "microwave_1_microjoint":
                    raise ValueError(name)
                return 0

        data = SimpleNamespace(
            body_xpos=np.asarray([[1, 0, 0], [0, 0, 0], [2, 3, 4]], dtype=float),
            body_xmat=np.repeat(np.eye(3).reshape(1, 9), 3, axis=0),
            site_xpos=np.asarray([[5, 6, 7]], dtype=float),
            site_xmat=np.eye(3).reshape(1, 9),
        )
        state = SimpleNamespace(
            object_state_type="object", object_name="microwave_1"
        )
        base = SimpleNamespace(
            sim=SimpleNamespace(model=Model(), data=data),
            obj_body_id={"plate_1": 0},
            object_states_dict={"microwave_1": state},
            get_object=lambda name: SimpleNamespace(
                joints=["microwave_1_microjoint"]
            ),
        )
        converter = SimpleNamespace(_get_base_env=lambda environment: environment)

        site_pose, site_reference = phase._live_reference_pose(
            base, "basket_region", converter
        )
        self.assertEqual(site_reference["kind"], "site")
        self.assertTrue(np.array_equal(site_pose[:3, 3], [5, 6, 7]))

        fixture_pose, fixture_reference = phase._fixture_reference_pose(
            base,
            phase._phase("fixture", "microwave_1", ("close", "microwave_1")),
            converter,
        )
        self.assertEqual(fixture_reference["kind"], "articulation_body")
        self.assertEqual(fixture_reference["name"], "microwave_1_microdoorroot")
        self.assertTrue(np.array_equal(fixture_pose[:3, 3], [2, 3, 4]))

    def test_initialization_must_equal_the_stored_raw_aubo_state(self):
        expected = np.arange(4, dtype=float)
        with mock.patch.object(
            phase,
            "_initialize",
            return_value={"state": expected.copy(), "scene": np.zeros(1)},
        ) as initialize:
            result = phase._initialize_failed_state(
                object(), object(), np.zeros(2), -1.0, expected, object()
            )
        self.assertEqual(
            initialize.call_args.args[4], phase.RAW_INITIALIZATION_SEED
        )
        self.assertTrue(result["raw_initial_state_within_absolute_tolerance"])
        self.assertEqual(result["raw_initial_state_max_absolute_difference"], 0.0)

        class Sim:
            state = expected + np.finfo(np.float64).eps

            def set_state_from_flattened(self, value):
                self.state = np.asarray(value).copy()

            def forward(self):
                pass

        target = SimpleNamespace(sim=Sim())
        converter = SimpleNamespace(
            _get_base_env=lambda environment: environment,
            get_target_sim_state=lambda environment: environment.sim.state.copy(),
        )
        with mock.patch.object(
            phase,
            "_initialize",
            return_value={
                "state": expected + np.finfo(np.float64).eps,
                "scene": np.zeros(1),
            },
        ), mock.patch.object(
            phase,
            "_scene_values",
            return_value=(np.zeros(1), 0),
        ):
            rounded = phase._initialize_failed_state(
                object(), target, np.zeros(2), -1.0, expected, converter
            )
        self.assertLessEqual(
            rounded["raw_initial_state_max_absolute_difference"],
            phase.INITIAL_STATE_ABSOLUTE_TOLERANCE,
        )
        self.assertTrue(rounded["raw_initial_state_restored_exactly"])

        with mock.patch.object(
            phase,
            "_initialize",
            return_value={"state": expected + 1.0, "scene": np.zeros(1)},
        ):
            with self.assertRaisesRegex(RuntimeError, "differs from the raw failed state"):
                phase._initialize_failed_state(
                    object(), object(), np.zeros(2), -1.0, expected, object()
                )

    def test_phase_seeds_load_only_successful_raw_aubo_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.hdf5"
            with phase.h5py.File(path, "w") as raw:
                data = raw.create_group("data")
                for name in ("demo_0", "demo_1"):
                    demo = data.create_group(name)
                    demo.create_dataset("states", data=np.zeros((2, 4)))
                    demo.create_dataset("actions", data=np.zeros((2, 7)))
            demos = phase._raw_success_demos(
                path,
                {
                    "demo_0": {"terminal_task_success": False},
                    "demo_1": {"terminal_task_success": True},
                },
            )
        self.assertEqual(list(demos), ["demo_1"])

    def test_stagewise_bootstrap_does_not_change_raw_admission(self):
        raw_rows = {
            "demo_0": {"terminal_task_success": False},
            "demo_1": {"terminal_task_success": True},
        }
        admitted = {"demo_2": {"terminal_task_success": True}}
        self.assertEqual(
            phase._stagewise_source_rows(raw_rows, admitted), (admitted, False)
        )
        sources, bootstrap = phase._stagewise_source_rows(raw_rows, {})
        self.assertTrue(bootstrap)
        self.assertEqual(list(sources), ["demo_1"])
        self.assertNotIn("demo_1", admitted)

if __name__ == "__main__":
    unittest.main()
