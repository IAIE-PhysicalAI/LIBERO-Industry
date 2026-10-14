"""Deterministic checks for legacy Level 2/3 modular task composition."""

from collections import Counter
from pathlib import Path
import random
import re
import unittest

import configs
import regenerate_industry_benchmark as regenerate
import task_composition as composition
from industry_object_map import industry_object_map


class TaskCompositionTest(unittest.TestCase):
    def test_single_arm_d2s_level3_filenames_match_goals(self):
        bddl_dir = (
            Path(configs.LIBERO_INDUSTRY_BASE)
            / "libero/libero/bddl_files/libero_industry_d2s"
        )
        paths = sorted(bddl_dir.glob("D2S_LV3_*.bddl"))
        self.assertEqual(len(paths), 24)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            init_text = text[text.index("(:init") : text.index("(:goal")]
            goals = re.findall(
                r"\(AlignedOn\s+(\w+)\s+(\w+)\)", text[text.index("(:goal") :]
            )
            self.assertEqual(
                [f"(AlignedOn {obj} {region})" in init_text for obj, region in goals],
                [False, False, True],
            )
            bases = [
                composition.strip_pose(re.sub(r"_\d+$", "", obj)).upper()
                for obj, _region in goals
            ]
            self.assertIn("_{}_".format("_".join(bases)), path.stem)

    def test_single_arm_d2s_level3_has_no_groove_targets(self):
        bddl_dir = (
            Path(configs.LIBERO_INDUSTRY_BASE)
            / "libero/libero/bddl_files/libero_industry_d2s"
        )
        paths = sorted(bddl_dir.glob("D2S_LV3_*.bddl"))
        self.assertEqual(len(paths), 24)
        for path in paths:
            self.assertNotIn("_groove", path.read_text(encoding="utf-8"), path.name)

    def test_track_specific_d2s_level3_task_names(self):
        current = "D2S_LV3_001_SCHOOL_BUS_MINI_ROLLER_MARIO_V3_sort_toys_to_inlays"
        legacy = "D2S_LV3_001_SCHOOL_BUS_MINI_ROLLER_V3_sort_toys_to_inlays"

        self.assertEqual(regenerate._object_map_key(current), legacy)
        self.assertIn(
            current, regenerate._task_names_for_track("D2S", configs.SINGLE_ARM)
        )
        self.assertIn(
            legacy, regenerate._task_names_for_track("D2S", configs.BIMANUAL)
        )

    def test_instance_names_continue_across_destination(self):
        source, counts = composition.named_instances(["a_flat", "a_flat", "b_upright"])
        destination, _ = composition.named_instances(["a_flat", "b_upright"], counts)
        self.assertEqual(source, ["a_flat_1", "a_flat_2", "b_upright_1"])
        self.assertEqual(destination, ["a_flat_3", "b_upright_2"])

    def test_s2d_levels_keep_source_and_destination_roles(self):
        for seed in range(40):
            level2 = composition.sample_s2d_level2(
                configs.S2D[2], "pump_bottle_flat", random.Random(seed)
            )
            self.assertEqual(len(level2["target_assets"]), 3)
            self.assertIn(len(level2["source_confounder_assets"]), (1, 2))
            self.assertFalse(level2["destination_confounder_assets"])
            self.assertTrue(
                all(asset not in {"pump_bottle_flat", "pump_bottle_upright"}
                    for asset in level2["source_confounder_assets"])
            )

            level3 = composition.sample_s2d_level3(
                configs.S2D[3], "pump_bottle", random.Random(seed)
            )
            self.assertEqual(len(level3["target_assets"]), 3)
            self.assertIn(len(level3["source_confounder_assets"]), (1, 2))
            self.assertIn(len(level3["destination_confounder_assets"]), (1, 2))
            self.assertLessEqual(
                len(level3["source_confounder_assets"])
                + len(level3["destination_confounder_assets"]),
                3,
            )
            self.assertTrue(all(asset.startswith("pump_bottle_") for asset in level3["destination_confounder_assets"]))

    def test_d2s_levels_keep_empty_l2_and_prefilled_l3_targets(self):
        for seed in range(40):
            level2 = composition.sample_d2s_level2(
                configs.D2S[2], "school_bus_upright", random.Random(seed)
            )
            self.assertEqual(len(level2["source_assets"]), 4)
            self.assertEqual(len(level2["goal_assets"]), 3)
            self.assertEqual(len(level2["groove_assets"]), 3)
            self.assertFalse(level2["destination_confounder_assets"])
            self.assertEqual(
                len(set(level2["source_assets"]) - set(level2["goal_assets"])), 1
            )

            level3 = composition.sample_d2s_level3(
                configs.D2S[3], "school_bus", random.Random(seed)
            )
            self.assertEqual(len(level3["target_assets"]), 2)
            self.assertIn(len(level3["source_confounder_assets"]), (1, 2))
            self.assertEqual(level3["destination_confounder_assets"], [level3["destination_base"] + "_flat"])
            all_bases = set(level3["target_bases"]) | {
                composition.strip_pose(asset) for asset in level3["source_confounder_assets"]
            } | {level3["destination_base"]}
            self.assertNotIn("mini_fire_engine", all_bases)
            for asset in level3["target_assets"] + level3["source_confounder_assets"]:
                if asset.startswith("mario_"):
                    self.assertEqual(asset, "mario_flat")

    def test_single_arm_d2s_level3_is_balanced_without_fire_engine(self):
        source_counts = Counter()
        moving_counts = Counter()
        prealigned_counts = Counter()
        target_counts = Counter()

        def bases(assets):
            return [composition.strip_pose(asset) for asset in assets]

        task_names = sorted(
            name for name in industry_object_map if name.startswith("D2S_LV3_")
        )
        self.assertEqual(len(task_names), 24)
        for task_name in task_names:
            raw = industry_object_map[task_name]
            single_arm = configs.object_map_entry_for_track(task_name, raw)
            bimanual = configs.object_map_entry_for_track(
                task_name, raw, configs.BIMANUAL
            )
            moving = list(single_arm["target_objects"])
            for asset in single_arm["prealigned_objects"]:
                moving.remove(asset)

            self.assertEqual(bimanual, raw)
            self.assertEqual(len(single_arm["source_objects"]), len(raw["source_objects"]))
            self.assertEqual(len(moving), 2)
            self.assertEqual(len(single_arm["prealigned_objects"]), 1)
            self.assertFalse(
                any(
                    composition.strip_pose(asset) == "mini_fire_engine"
                    for field in single_arm.values()
                    if isinstance(field, list)
                    for asset in field
                )
            )
            source_counts.update(bases(single_arm["source_objects"]))
            moving_counts.update(bases(moving))
            prealigned_counts.update(bases(single_arm["prealigned_objects"]))
            target_counts.update(bases(single_arm["target_objects"]))

        self.assertEqual(set(source_counts.values()), {17})
        self.assertEqual(sorted(moving_counts.values()), [9, 9, 10, 10, 10])
        self.assertEqual(sorted(prealigned_counts.values()), [4, 5, 5, 5, 5])
        self.assertEqual(sorted(target_counts.values()), [14, 14, 14, 15, 15])

    def test_d2s_level2_grooves_match_object_pose(self):
        groove_cfg = configs.D2S[2]["composition"]["grooves"]
        self.assertEqual(
            composition.groove_for_d2s_level2(
                "thomas_flat", groove_cfg, random.Random(0)
            ),
            "thomas_flat_groove",
        )
        self.assertEqual(
            composition.groove_for_d2s_level2(
                "thomas_upright", groove_cfg, random.Random(1)
            ),
            "thomas_upright_groove",
        )

    def test_d2d_levels_keep_counts_and_track_pose_policy(self):
        bimanual_inverted = False
        for seed in range(80):
            level2 = composition.sample_d2d_level2(
                configs.D2D[2], "mouse_black_taipan_upright", random.Random(seed)
            )
            self.assertEqual(len(level2["source_assets"]), 4)
            self.assertIn(len(level2["target_assets"]), (2, 3))
            self.assertEqual(
                len(level2["source_confounder_assets"]), 4 - len(level2["target_assets"])
            )
            self.assertFalse(level2["destination_confounder_assets"])

            level3 = composition.sample_d2d_level3(
                configs.D2D[3], "mouse_black_taipan", random.Random(seed)
            )
            self.assertEqual(len(level3["source_assets"]), 4)
            self.assertIn(len(level3["target_assets"]), (2, 3))
            self.assertEqual(
                len(level3["source_confounder_assets"])
                + len(level3["destination_confounder_assets"]),
                3,
            )
            self.assertTrue(
                all(asset == "mouse_black_taipan_upright" for asset in level3["destination_confounder_assets"])
            )
            self.assertTrue(
                all(asset.endswith("_upright") for asset in level3["source_assets"])
            )

            bimanual = composition.sample_d2d_level3(
                configs.get_suite_config("D2D", 3, configs.BIMANUAL),
                "mouse_black_taipan",
                random.Random(seed),
            )
            bimanual_inverted |= any(
                asset.endswith("_inverted") for asset in bimanual["source_assets"]
            )
        self.assertTrue(bimanual_inverted)

    def test_generation_tracks_keep_separate_names_and_level2_policies(self):
        self.assertEqual(
            configs.get_suite_dirs(configs.BIMANUAL)["S2D"],
            "libero_industry_s2d_bimanual",
        )
        self.assertEqual(configs.D2D[2]["grooves"], ["upright_groove"])
        self.assertEqual(
            configs.get_suite_config("D2D", 2, configs.BIMANUAL)["grooves"],
            ["upright_groove", "v_groove"],
        )
        self.assertEqual(
            configs.S2D[2]["mismatch_map"]["pump_bottle_flat"]["init"],
            "pump_bottle_flat",
        )
        self.assertEqual(
            configs.get_suite_config("S2D", 2, configs.BIMANUAL)["mismatch_map"][
                "pump_bottle_flat"
            ]["init"],
            "pump_bottle_upright",
        )
        self.assertEqual(
            configs.get_suite_config("S2D", 2, configs.BIMANUAL)["mismatch_map"][
                "cream_bottle_flat"
            ]["inlay"],
            "cream_bottle_flat_groove",
        )

        d2s = configs.object_map_entry_for_track(
            "D2S_LV2_001_TEST_V3_sort_marked_toys_to_grooves",
            {
                "target_objects": [
                    "school_bus_flat",
                    "mario_flat",
                    "porter_flat",
                ]
            },
            configs.BIMANUAL,
        )
        self.assertEqual(d2s["target_grooves"][0], "school_bus_v_groove")

        flat_fire_engine = {
            "source_objects": ["mini_fire_engine_flat"],
            "target_objects": ["mini_fire_engine_flat"],
        }
        single_arm_d2s = configs.object_map_entry_for_track(
            "D2S_LV2_002_TEST_V3_sort_marked_toys_to_grooves",
            flat_fire_engine,
        )
        self.assertEqual(
            single_arm_d2s["source_objects"], ["mini_fire_engine_upright"]
        )
        self.assertEqual(
            single_arm_d2s["target_grooves"],
            ["mini_fire_engine_upright_groove"],
        )
        self.assertEqual(
            configs.object_map_entry_for_track(
                "D2S_LV2_002_TEST_V3_sort_marked_toys_to_grooves",
                flat_fire_engine,
                configs.BIMANUAL,
            )["source_objects"],
            ["mini_fire_engine_flat"],
        )

        d2d = configs.object_map_entry_for_track(
            "D2D_LV3_001_TEST_V3_M8_route_objects",
            {
                "source_objects": ["mouse_naga_inverted"],
                "target_objects": ["mouse_naga_inverted"],
            },
        )
        self.assertEqual(d2d["source_objects"], ["mouse_naga_upright"])

        s2d = configs.object_map_entry_for_track(
            "S2D_LV2_001_TEST_V3_route_cosmetics",
            {
                "static_source_objects": ["pump_bottle_upright"] * 3,
                "target_objects": ["pump_bottle_upright"] * 3,
            },
        )
        self.assertEqual(s2d["target_objects"], ["pump_bottle_flat"] * 3)

        bimanual_s2d = configs.object_map_entry_for_track(
            "S2D_LV2_001_TEST_V3_route_cosmetics",
            {
                "static_source_objects": ["pump_bottle_flat"] * 3,
                "target_objects": ["pump_bottle_flat"] * 3,
            },
            configs.BIMANUAL,
        )
        self.assertEqual(
            bimanual_s2d["target_objects"], ["pump_bottle_upright"] * 3
        )

        level3_entry = {
            "static_source_objects": [
                "serum_bottle_upright",
                "serum_bottle_flat",
                "cream_bottle_upright",
            ],
            "destination_objects": ["serum_bottle_flat"],
            "target_objects": [
                "serum_bottle_upright",
                "serum_bottle_flat",
            ],
        }
        single_arm_level3 = configs.object_map_entry_for_track(
            "S2D_LV3_007_TEST_V3_route_cosmetics", level3_entry
        )
        self.assertEqual(
            single_arm_level3["static_source_objects"],
            ["serum_bottle_flat", "serum_bottle_flat", "cream_bottle_upright"],
        )
        self.assertEqual(
            single_arm_level3["target_objects"], ["serum_bottle_flat"] * 2
        )
        self.assertEqual(
            configs.object_map_entry_for_track(
                "S2D_LV3_007_TEST_V3_route_cosmetics",
                level3_entry,
                configs.BIMANUAL,
            ),
            level3_entry,
        )

        cream_jar_level3 = {
            "static_source_objects": ["cream_jar_flat", "cream_jar_upright"],
            "destination_objects": ["cream_jar_upright"],
            "target_objects": ["cream_jar_flat", "cream_jar_upright"],
        }
        self.assertEqual(
            configs.object_map_entry_for_track(
                "S2D_LV3_019_TEST_V3_route_cosmetics", cream_jar_level3
            ),
            cream_jar_level3,
        )


if __name__ == "__main__":
    unittest.main()
