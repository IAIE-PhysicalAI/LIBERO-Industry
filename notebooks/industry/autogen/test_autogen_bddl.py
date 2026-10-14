"""BDDL-level composition checks for modular Level 2 and Level 3 tasks."""

import os
import random
import re
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import configs
import gen_d2d
import gen_d2s
import gen_s2d
import generate_fractional_tuning_set as fractional_tuning
import perturbations
import libero.libero.utils.task_generation_utils as task_utils
from libero.libero.utils.task_generation_utils import generate_bddl_from_task_info


def _section(text, name, next_name):
    match = re.search(r"\(\:{}\s*(.*?)\n\s*\(\:{}".format(name, next_name), text, re.DOTALL)
    if not match:
        raise AssertionError("Missing BDDL section: {}".format(name))
    return match.group(1)


def _declared_names(section):
    names = set()
    for line in section.splitlines():
        if " - " in line:
            names.update(line.strip().split(" - ", 1)[0].split())
    return names


class ModularBddlCompositionTest(unittest.TestCase):
    def test_d2d_speed_sampling_uses_non_overlapping_watersheds(self):
        random.seed(42)
        samples = {}
        for anchor in (0.03, 0.05, 0.08, -0.03, -0.05, -0.08):
            values = [perturbations.sample_d2d_speed(anchor) for _ in range(50)]
            lower, upper = perturbations.d2d_speed_band(anchor)
            self.assertTrue(
                all(
                    value * anchor > 0.0 and lower <= abs(value) <= upper
                    for value in values
                )
            )
            self.assertGreater(len(set(values)), 1)
            samples[anchor] = values

        self.assertLess(max(samples[0.03]), min(samples[0.05]))
        self.assertLess(max(samples[0.05]), min(samples[0.08]))

    def test_fractional_remake_preserves_existing_timing_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_dir = root / "libero_industry_d2s"
            canonical_dir.mkdir()
            canonical = canonical_dir / "D2S_LV3_001_TEST.bddl"
            canonical.write_text(
                "(AtSpeed conveyor_belt_1 0.03 steady takt_4 offset_1.5)\n",
                encoding="utf-8",
            )
            templates = root / "templates"
            templates.mkdir()
            template = templates / "D2S_LV3_001_TEST_v6_t2.bddl"
            template.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")
            output = root / "output"

            _, generated = fractional_tuning.remake_existing(
                root, templates, output, ["D2S:LV3"]
            )

            self.assertEqual(generated["D2S"], 1)
            self.assertIn(
                "AtSpeed conveyor_belt_1 0.06 steady takt_2",
                (output / template.name).read_text(encoding="utf-8"),
            )

    def test_d2d_fractional_variants_preserve_filename_speed_pair(self):
        cases = (
            ("LV2", "V5_V3", "0.0478", "0.0264", "v5_t5"),
            ("LV3", "V3_M8", "0.03", "-0.08", "v5_t4"),
        )
        for level, speed_pair, source_speed, slot_speed, suffix in cases:
            with self.subTest(level=level), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / f"D2D_{level}_001_TEST_{speed_pair}_task.bddl"
                source.write_text(
                    "(AtSpeed conveyor_belt_1 {} steady takt_6 offset_1.5)\n"
                    "(LoadSlot conveyor_ghost_1 test_inlay y pulse {} service_2.0)\n".format(
                        source_speed, slot_speed
                    ),
                    encoding="utf-8",
                )
                output = Path(directory) / "output"

                fractional_tuning.generate_sources(
                    [("libero_industry_d2d", level, source)], output
                )
                generated = (output / f"{source.stem}_{suffix}.bddl").read_text(
                    encoding="utf-8"
                )

                self.assertIn(
                    "AtSpeed conveyor_belt_1 {} steady takt_{}".format(
                        source_speed, suffix.rsplit("t", 1)[1]
                    ),
                    generated,
                )
                self.assertIn(
                    f"pulse {slot_speed} service_2.0", generated
                )

    def test_d2d_repair_changes_only_timing_predicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_dir = root / "libero_industry_d2d"
            canonical_dir.mkdir()
            canonical = canonical_dir / "D2D_LV2_007_TEST_V3_V8_task.bddl"
            canonical.write_text(
                "(AtSpeed conveyor_belt_1 0.0287 steady takt_6 offset_1.5)\n"
                "(LoadSlot conveyor_ghost_1 test_inlay y pulse 0.0768 service_2.0)\n",
                encoding="utf-8",
            )
            output = root / "fractional"
            output.mkdir()
            variant = output / "D2D_LV2_007_TEST_V3_V8_task_v5_t4.bddl"
            variant.write_text(
                "custom geometry\n"
                "(AtSpeed conveyor_belt_1 0.05 steady takt_4 offset_1.5)\n"
                "(LoadSlot conveyor_ghost_1 test_inlay y pulse 0.05 service_2.0)\n",
                encoding="utf-8",
            )

            self.assertEqual(fractional_tuning.repair_existing(root, output), 1)
            repaired = variant.read_text(encoding="utf-8")

            self.assertIn("custom geometry", repaired)
            self.assertIn("AtSpeed conveyor_belt_1 0.0287 steady takt_4", repaired)
            self.assertIn("pulse 0.0768 service_2.0", repaired)

    def test_s2d_fractional_variants_scale_load_slot_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "S2D_LV3_002_PUMP_BOTTLE_V5_task.bddl"
            source.write_text(
                "(LoadSlot conveyor_ghost_1 pump_bottle_upright_inlay y pulse -0.0438 service_2.5)\n",
                encoding="utf-8",
            )
            output = Path(directory) / "output"
            generated = fractional_tuning.generate_sources(
                [("libero_industry_s2d", "LV3", source)], output
            )

            self.assertEqual(generated["libero_industry_s2d"], 6)
            self.assertIn(
                "pulse -0.05 service_1.5",
                (output / f"{source.stem}_svc1.5.bddl").read_text(encoding="utf-8"),
            )

    def _build(self, builder, task_id, *args, **kwargs):
        random.seed(1000 + task_id)
        task_utils.TASK_INFO.clear()
        scene_name, _, error, _ = builder(task_id, *args, **kwargs)
        self.assertIsNone(error)
        with tempfile.TemporaryDirectory() as output_dir:
            files, failures = generate_bddl_from_task_info(folder=output_dir)
            self.assertFalse(failures)
            self.assertEqual(len(files), 1)
            with open(files[0], "r") as handle:
                return handle.read()

    def test_d2s_level2_object_map_rejects_pose_mismatched_grooves(self):
        entry = {
            "source_objects": ["mario_flat", "mini_roller_flat"],
            "target_objects": ["mario_flat", "mini_roller_flat"],
            "target_grooves": ["mario_flat_groove", "mini_roller_v_groove"],
        }

        with self.assertRaisesRegex(ValueError, "must match"):
            gen_d2s._sample_composition(configs.D2S[2], 2, None, entry)

    def test_d2s_level2_forward_takt_follows_speed_band(self):
        for task_id, speed, takt in ((221, 0.03, 4), (222, 0.05, 5), (223, 0.08, 6)):
            with self.subTest(speed=speed):
                text = self._build(
                    gen_d2s.build_and_register_d2s,
                    task_id,
                    2,
                    speed,
                    "school_bus_upright",
                )
                self.assertIn("steady takt_{} offset_1.5".format(takt), text)

    def _assert_predicate_objects_declared(self, text):
        fixtures = _declared_names(_section(text, "fixtures", "objects"))
        objects = _declared_names(_section(text, "objects", "init"))
        declared = fixtures | objects
        init = _section(text, "init", "goal")
        goal = text[text.index("(:goal") :]
        for predicate, first_name in re.findall(r"\((On|AlignedOn|Routed|DynamicOn)\s+([^\s\)]+)", init + goal):
            self.assertIn(first_name, declared, "{} references undeclared {}".format(predicate, first_name))

    def _assert_bounded_rejection(self, builder, task_id, *args):
        random.seed(1000 + task_id)
        task_utils.TASK_INFO.clear()
        scene_name, _, error, _ = builder(task_id, *args)
        self.assertIsNone(scene_name)
        self.assertIsNotNone(error)
        self.assertRegex(
            error,
            r"(spatial workspace bounds|kinematic workspace bounds|dynamic slots)",
        )

    def test_level2_contains_only_source_confounders(self):
        s2d = self._build(gen_s2d.build_and_register_s2d, 102, 2, "pump_bottle_flat", 0.03)
        d2d = self._build(gen_d2d.build_and_register_d2d, 302, 2, "mouse_black_taipan_upright", 0.03)
        d2s = self._build(gen_d2s.build_and_register_d2s, 202, 2, 0.03, "school_bus_upright")
        for text in (s2d, d2d, d2s):
            self._assert_predicate_objects_declared(text)

        s2d_init = _section(s2d, "init", "goal")
        s2d_goal = s2d[s2d.index("(:goal") :]
        self.assertNotRegex(s2d_init, r"\(On (?!conveyor_ghost_1\b)[^\s]+ conveyor_ghost_1\)")
        self.assertNotRegex(_section(d2d, "init", "goal"), r"\(On (?!conveyor_ghost_1\b)[^\s]+ conveyor_ghost_1\)")
        self.assertNotIn("conveyor_ghost_1", _section(d2s, "init", "goal"))
        self.assertIn("Routed", s2d_goal)

    def test_level3_preserves_destination_confounder_goal_semantics(self):
        s2d = self._build(gen_s2d.build_and_register_s2d, 103, 3, "pump_bottle", 0.03)
        d2d = self._build(gen_d2d.build_and_register_d2d, 303, 3, "mouse_black_taipan", 0.03)
        d2s = self._build(gen_d2s.build_and_register_d2s, 203, 3, 0.03, "school_bus")
        for text in (s2d, d2d, d2s):
            self._assert_predicate_objects_declared(text)

        s2d_init = _section(s2d, "init", "goal")
        s2d_goal = s2d[s2d.index("(:goal") :]
        s2d_destination = re.findall(r"\(On ([^\s]+) conveyor_ghost_1\)", s2d_init)
        self.assertTrue(s2d_destination)
        for instance_name in s2d_destination:
            self.assertNotIn("Routed {} conveyor_ghost_1".format(instance_name), s2d_goal)

        d2d_init = _section(d2d, "init", "goal")
        d2d_goal = d2d[d2d.index("(:goal") :]
        d2d_destination = re.findall(r"\(On ([^\s]+) conveyor_ghost_1\)", d2d_init)
        self.assertTrue(d2d_destination)
        for instance_name in d2d_destination:
            self.assertIn("(Routed {} conveyor_ghost_1)".format(instance_name), d2d_goal)

        d2s_goal = d2s[d2s.index("(:goal") :]
        self.assertIn("AlignedOn", d2s_goal)

    def test_d2d_objects_of_interest_match_goal_objects(self):
        for task_id, level, target in (
            (301, 1, "mouse_black_taipan_upright"),
            (302, 2, "mouse_black_taipan_upright"),
            (303, 3, "mouse_black_taipan"),
        ):
            with self.subTest(level=level):
                text = self._build(
                    gen_d2d.build_and_register_d2d, task_id, level, target, 0.03
                )
                objects_of_interest = set(
                    line.strip()
                    for line in _section(text, "obj_of_interest", "init").splitlines()
                    if line.strip() != ")"
                )
                goal_objects = set(
                    re.findall(
                        r"\((?:DynamicOn|Routed)\s+([^\s\)]+)\s+conveyor_ghost_1\)",
                        text[text.index("(:goal") :],
                    )
                )
                self.assertEqual(objects_of_interest, goal_objects)

    def test_d2s_level3_prealigned_duplicate_owns_instance_one(self):
        layout = (
            0.0,
            0.0,
            0.0,
            (-0.20, 0.0),
            (0.0, 0.0),
            [(0.0, 0.10), (0.0, 0.0), (0.0, -0.10)],
            [0.0, 0.0, 0.0],
            90.0,
        )
        object_map_entry = {
            "source_objects": ["mario_flat", "mini_fire_engine_flat", "thomas_flat"],
            "target_objects": ["mario_flat", "thomas_flat", "mini_fire_engine_flat"],
            "prealigned_objects": ["mini_fire_engine_flat"],
        }
        with mock.patch.object(gen_d2s, "generate_safe_d2s_layout", return_value=layout):
            text = self._build(
                gen_d2s.build_and_register_d2s,
                202,
                3,
                0.03,
                object_map_entry=object_map_entry,
            )

        init = _section(text, "init", "goal")
        goal = text[text.index("(:goal") :]
        self.assertIn(
            "(AlignedOn mini_fire_engine_flat_1 mini_fire_engine_flat_inlay_1_goal_region)",
            init,
        )
        self.assertRegex(init, r"\(On mini_fire_engine_flat_2 conveyor_belt_1_[^\s]+\)")
        self.assertEqual(goal.count("AlignedOn mini_fire_engine_flat_1 "), 1)
        self.assertNotIn("AlignedOn mini_fire_engine_flat_2 ", goal)

    def test_d2s_level3_uses_pose_matched_fixtures_and_populates_interest(self):
        layout = (
            0.0,
            0.0,
            0.0,
            (-0.20, 0.0),
            (0.0, 0.0),
            [(0.0, 0.10), (0.0, 0.0), (0.0, -0.10)],
            [0.0, 0.0, 0.0],
            90.0,
        )
        object_map_entry = {
            "source_objects": ["thomas_upright", "mario_flat"],
            "target_objects": ["thomas_upright", "mario_flat", "porter_flat"],
            "prealigned_objects": ["porter_flat"],
        }
        with mock.patch.object(gen_d2s, "generate_safe_d2s_layout", return_value=layout):
            text = self._build(
                gen_d2s.build_and_register_d2s,
                203,
                3,
                0.03,
                object_map_entry=object_map_entry,
            )

        self.assertIn("thomas_upright_groove_1 - thomas_upright_groove", text)
        self.assertIn("mario_flat_inlay_1 - mario_flat_inlay", text)
        self.assertNotIn("thomas_flat_inlay", text)
        self.assertNotIn("mario_upright", text)
        objects_of_interest = set(
            line.strip()
            for line in _section(text, "obj_of_interest", "init").splitlines()
            if line.strip() != ")"
        )
        goal_objects = set(
            re.findall(
                r"\(AlignedOn\s+([^\s\)]+)\s+[^\s\)]+\)",
                text[text.index("(:goal") :],
            )
        )
        self.assertEqual(objects_of_interest, goal_objects)

    def test_d2s_and_d2d_serialize_canonical_conveyor_regions(self):
        # Geometry is tested independently. These stubs isolate BDDL text
        # serialization so a deliberate bounded-layout rejection cannot hide
        # a regression back to the legacy _d2d region names.
        d2d_layout = (0.0, 0.0, 0.0, (-0.20, 0.0), (0.0, 0.0), 180.0, 0.0)
        d2s_layout = (
            0.0,
            0.0,
            0.0,
            (-0.20, 0.0),
            (0.0, 0.0),
            [(0.0, 0.10), (0.0, 0.0), (0.0, -0.10)],
            [0.0, 0.0, 0.0],
            90.0,
        )
        with mock.patch.object(gen_d2d, "generate_safe_d2d_layout", return_value=d2d_layout):
            d2d = self._build(
                gen_d2d.build_and_register_d2d, 304, 2, "mouse_black_taipan_upright", 0.03
            )
        with mock.patch.object(gen_d2s, "generate_safe_d2s_layout", return_value=d2s_layout):
            d2s = self._build(
                gen_d2s.build_and_register_d2s, 204, 2, 0.03, "school_bus_upright"
        )
        for text in (d2d, d2s):
            init = _section(text, "init", "goal")
            self.assertNotRegex(
                init, r"conveyor_belt_1_(?:offset|tail)_region_d2d_\d+"
            )
            self.assertRegex(
                text,
                r"\(AtSpeed conveyor_belt_1 [-0-9.]+ steady takt_[-0-9.]+ offset_[-0-9.]+\)",
            )
        self.assertRegex(
            d2d,
            r"\(LoadSlot conveyor_ghost_1 [^\s]+ [xy] pulse [-0-9.]+ service_[-0-9.]+\)",
        )

    def test_d2d_serializes_the_requested_speed_relationships(self):
        layout = (0.0, 0.0, 0.0, (-0.20, 0.0), (0.0, 0.0), 90.0, 90.0)

        def speeds(text):
            source = float(
                re.search(r"\(AtSpeed conveyor_belt_1 ([-0-9.]+)", text).group(1)
            )
            slot = float(
                re.search(
                    r"\(LoadSlot conveyor_ghost_1 [^\s]+ [xy] pulse ([-0-9.]+)",
                    text,
                ).group(1)
            )
            return source, slot

        def takt(text):
            return float(re.search(r"\btakt_([-0-9.]+)", text).group(1))

        with mock.patch.object(gen_d2d, "generate_safe_d2d_layout", return_value=layout):
            level1_slow = self._build(
                gen_d2d.build_and_register_d2d,
                311,
                1,
                "mouse_black_taipan_upright",
                0.03,
                slot_base_speed=0.03,
            )
            level1_fast = self._build(
                gen_d2d.build_and_register_d2d,
                314,
                1,
                "mouse_black_taipan_upright",
                0.05,
                slot_base_speed=0.05,
            )
            level2 = self._build(
                gen_d2d.build_and_register_d2d,
                312,
                2,
                "mouse_black_taipan_upright",
                0.03,
                slot_base_speed=0.05,
            )
            level3 = self._build(
                gen_d2d.build_and_register_d2d,
                313,
                3,
                "mouse_black_taipan",
                0.03,
                slot_base_speed=-0.08,
            )

        level1_source, level1_slot = speeds(level1_slow)
        self.assertEqual(level1_source, level1_slot)
        level2_source, level2_slot = speeds(level2)
        self.assertGreater(level2_source * level2_slot, 0.0)
        self.assertNotEqual(abs(level2_source), abs(level2_slot))
        level3_source, level3_slot = speeds(level3)
        self.assertLess(level3_source * level3_slot, 0.0)
        self.assertEqual(takt(level1_slow), 4.0)
        self.assertEqual(takt(level1_fast), 5.0)
        self.assertEqual(takt(level2), 6.0)
        self.assertEqual(takt(level3), 5.0)



if __name__ == "__main__":
    unittest.main()
