"""Focused geometry and timing checks for bounded Industry generation."""

import math
import random
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONVEYOR_BELT_XML = (
    REPO_ROOT
    / "libero"
    / "libero"
    / "assets"
    / "industry_objects"
    / "conveyor_belt"
    / "conveyor_belt.xml"
)

import configs
import gen_d2d
import gen_d2s
import gen_s2d
import perturbations
from libero.libero.envs.regions.workspace_region_sampler import _conveyor_timing_target


class _FakeSimData:
    def __init__(self, sites):
        self._sites = sites

    def get_site_xpos(self, name):
        return self._sites[name]


class _FakeSim:
    def __init__(self, sites):
        self.data = _FakeSimData(sites)


class BoundedLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = perturbations.DynamicSpatialValidator(configs.ASSET_DIR)

    def test_symmetric_anchor_proof_reserves_two_fixture_shifts(self):
        for rotation in (0, 90, 180, 270):
            source, destination, separation = perturbations.get_safe_symmetric_base_anchors(
                0.05, 0.08, rotation
            )
            self.assertAlmostEqual(separation, 0.23)
            self.assertAlmostEqual(math.dist(source, destination), separation)
            self.assertTrue(perturbations._circle_inside_bounds(source, 0.10))
            self.assertTrue(perturbations._circle_inside_bounds(destination, 0.13))

    def test_s2d_uses_collision_mesh_extents_and_load_axis(self):
        expected_half_extents = {
            "grey_plate": (0.15, 0.18),
            "pump_bottle_flat_groove": (0.02465, 0.06986),
            "serum_bottle_flat_groove": (0.02197, 0.05101),
            "cream_bottle_flat_groove": (0.03, 0.06118),
            "cream_jar_v_groove": (0.058284, 0.042385),
            "pump_bottle_upright_inlay": (0.0506171, 0.0506171),
            "serum_bottle_flat_inlay": (0.04081341, 0.06575342),
            "cream_bottle_flat_inlay": (0.03499836, 0.05922236),
            "cream_jar_upright_inlay": (0.05053989, 0.05053989),
        }
        for asset_name, expected in expected_half_extents.items():
            actual = self.validator.collision_half_extents(asset_name)
            self.assertAlmostEqual(actual[0], expected[0], places=8)
            self.assertAlmostEqual(actual[1], expected[1], places=8)

        pump_half_extents = self.validator.collision_half_extents(
            "pump_bottle_upright_inlay"
        )
        self.assertAlmostEqual(pump_half_extents[0], 0.0506171)
        self.assertAlmostEqual(pump_half_extents[1], 0.0506171)

        serum_half_extents = self.validator.collision_half_extents(
            "serum_bottle_flat_inlay"
        )
        self.assertEqual(
            self.validator.dynamic_slot_track_half_extents(
                "serum_bottle_flat_inlay", load_axis="y"
            ),
            serum_half_extents,
        )
        self.assertEqual(
            self.validator.dynamic_slot_track_half_extents(
                "serum_bottle_flat_inlay", load_axis="x"
            ),
            (serum_half_extents[1], serum_half_extents[0]),
        )
        self.assertAlmostEqual(
            self.validator.dynamic_slot_loop_half_length(
                "serum_bottle_flat_inlay", load_axis="y"
            ),
            0.56032,
        )
        self.assertAlmostEqual(
            self.validator.dynamic_slot_loop_half_length(
                "serum_bottle_flat_inlay", load_axis="x"
            ),
            0.5863,
        )

    def test_s2d_clearance_anchors_sum_yawed_support_mesh_and_gap(self):
        plate_half_extents = self.validator.collision_half_extents("grey_plate")
        slot_half_extents = self.validator.collision_half_extents(
            "pump_bottle_upright_inlay"
        )
        plate_yaw_deg = 30.0
        ghost_yaw_deg = 10.0
        plate, ghost, separation = (
            perturbations.get_s2d_collision_clearance_anchors(
                plate_half_extents,
                slot_half_extents,
                plate_yaw_deg,
                ghost_yaw_deg,
            )
        )
        track_normal = perturbations._rotate_xy((1.0, 0.0), ghost_yaw_deg)
        expected_separation = (
            perturbations._obb_support(
                plate_half_extents, plate_yaw_deg, track_normal
            )
            + slot_half_extents[0]
            + perturbations.LAYOUT_CLEARANCE
        )
        self.assertAlmostEqual(separation, expected_separation)
        self.assertAlmostEqual(math.dist(plate, ghost), expected_separation)
        centre_delta = (ghost[0] - plate[0], ghost[1] - plate[1])
        self.assertAlmostEqual(
            centre_delta[0] / separation, track_normal[0]
        )
        self.assertAlmostEqual(
            centre_delta[1] / separation, track_normal[1]
        )

    def test_s2d_rejects_reported_lv3_plate_track_overlap(self):
        layout = {
            "plate": {
                "pos": (-0.2584007637399376, -0.03082309049017188),
                "yaw_deg": 287.9827098423408,
                "xml_name": "grey_plate",
                "safety_radius": perturbations.S2D_TRAY_SPAWN_RADIUS,
            },
            "ghost": {
                "pos": (-0.02405047531449355, -0.03082309049017188),
                "yaw_deg": 288.0193760670657,
                "xml_name": "conveyor_ghost",
                "safety_radius": perturbations.S2D_GHOST_FIXTURE_RADIUS,
            },
        }
        result = self.validator.validate_s2d_layout(
            layout,
            "pump_bottle_upright_inlay",
            required_track_clearance=perturbations.S2D_TRACK_REQUIRED_CLEARANCE,
        )
        self.assertFalse(result)
        self.assertIn("dynamic slot track", result.reason)

    def test_s2d_collision_clearance_boundary_is_accepted_then_rejected(self):
        plate_half_extents = self.validator.collision_half_extents("grey_plate")
        slot_half_extents = self.validator.collision_half_extents(
            "pump_bottle_upright_inlay"
        )
        plate, ghost, _ = perturbations.get_s2d_collision_clearance_anchors(
            plate_half_extents,
            slot_half_extents,
            0.0,
            0.0,
        )
        layout = {
            "plate": {
                "pos": plate,
                "yaw_deg": 0.0,
                "xml_name": "grey_plate",
                "safety_radius": perturbations.S2D_TRAY_SPAWN_RADIUS,
            },
            "ghost": {
                "pos": ghost,
                "yaw_deg": 0.0,
                "xml_name": "conveyor_ghost",
                "safety_radius": perturbations.S2D_GHOST_FIXTURE_RADIUS,
            },
        }
        valid = self.validator.validate_s2d_layout(
            layout,
            "pump_bottle_upright_inlay",
            required_track_clearance=perturbations.S2D_TRACK_REQUIRED_CLEARANCE,
        )
        self.assertTrue(valid, valid.reason)
        layout["ghost"]["pos"] = (ghost[0] - 0.001, ghost[1])
        invalid = self.validator.validate_s2d_layout(
            layout,
            "pump_bottle_upright_inlay",
            required_track_clearance=perturbations.S2D_TRACK_REQUIRED_CLEARANCE,
        )
        self.assertFalse(invalid)
        self.assertIn("collision clearance", invalid.reason)

    def test_s2d_common_xy_perturbation_preserves_track_clearance(self):
        plate_half_extents = self.validator.collision_half_extents("grey_plate")
        slot_half_extents = self.validator.collision_half_extents(
            "pump_bottle_upright_inlay"
        )
        plate, ghost, _ = perturbations.get_s2d_collision_clearance_anchors(
            plate_half_extents,
            slot_half_extents,
            30.0,
            45.0,
        )
        baseline_layout = {
            "plate": {
                "pos": plate,
                "yaw_deg": 30.0,
                "xml_name": "grey_plate",
            },
            "ghost": {
                "pos": ghost,
                "yaw_deg": 45.0,
                "xml_name": "conveyor_ghost",
            },
        }
        baseline = self.validator.s2d_plate_track_clearance(
            baseline_layout, "pump_bottle_upright_inlay"
        )
        for shift_x in (-perturbations.S2D_MAX_PERTURBATION, 0.05):
            for shift_y in (-perturbations.S2D_MAX_PERTURBATION, 0.05):
                shifted_layout = {
                    name: dict(
                        fixture,
                        pos=(
                            fixture["pos"][0] + shift_x,
                            fixture["pos"][1] + shift_y,
                        ),
                    )
                    for name, fixture in baseline_layout.items()
                }
                shifted = self.validator.s2d_plate_track_clearance(
                    shifted_layout, "pump_bottle_upright_inlay"
                )
                self.assertAlmostEqual(shifted, baseline)

    def test_s2d_sampler_preserves_each_lv3_cardinal_with_clearance(self):
        random_state = random.getstate()
        try:
            inlays = (
                "pump_bottle_upright_inlay",
                "serum_bottle_flat_inlay",
                "cream_bottle_flat_inlay",
                "cream_jar_upright_inlay",
            )
            for inlay_index, inlay_name in enumerate(inlays):
                for transpose_deg in (0, 90, 180, 270):
                    random.seed(2000 + 10 * inlay_index + transpose_deg)
                    layout = gen_s2d.generate_safe_s2d_layout(
                        configs.S2D[3],
                        3,
                        inlay_name,
                        transpose_degrees=(transpose_deg,),
                    )
                    self.assertEqual(layout[2], transpose_deg)
                    candidate = {
                        "plate": {
                            "pos": layout[3],
                            "yaw_deg": layout[5],
                            "xml_name": "grey_plate",
                            "safety_radius": perturbations.S2D_TRAY_SPAWN_RADIUS,
                        },
                        "ghost": {
                            "pos": layout[4],
                            "yaw_deg": layout[6],
                            "xml_name": "conveyor_ghost",
                            "safety_radius": perturbations.S2D_GHOST_FIXTURE_RADIUS,
                        },
                    }
                    clearance = self.validator.s2d_plate_track_clearance(
                        candidate, inlay_name
                    )
                    self.assertGreaterEqual(
                        clearance + 1e-9,
                        perturbations.S2D_TRACK_REQUIRED_CLEARANCE,
                    )
        finally:
            random.setstate(random_state)

    def test_s2d_rejection_commits_rng_at_first_legacy_valid_candidate(self):
        from unittest import mock

        random_state = random.getstate()
        try:
            random.seed(2601)
            expected_rng = random.Random()
            expected_rng.setstate(random.getstate())
            expected_rng.uniform(-0.05, 0.05)
            expected_rng.uniform(-0.05, 0.05)
            expected_rng.uniform(-20.0, 20.0)
            expected_rng.uniform(-20.0, 20.0)
            expected_rng.uniform(-0.05, 0.05)
            expected_rng.uniform(-0.05, 0.05)
            expected_rng.uniform(-20.0, 20.0)
            expected_rng.uniform(-20.0, 20.0)
            expected_next = expected_rng.random()

            rejected = perturbations.LayoutValidationResult(
                False, "synthetic collision rejection"
            )
            accepted = perturbations.LayoutValidationResult(True)
            with mock.patch.object(
                gen_s2d.spatial_validator,
                "validate_s2d_layout",
                side_effect=(rejected, accepted, accepted),
            ):
                gen_s2d.generate_safe_s2d_layout(
                    configs.S2D[3],
                    3,
                    "pump_bottle_upright_inlay",
                    transpose_degrees=(90,),
                )
            self.assertAlmostEqual(random.random(), expected_next)
        finally:
            random.setstate(random_state)

    def test_s2d_source_plate_yaw_offset_only_rotates_the_plate(self):
        from unittest import mock

        random_state = random.getstate()
        try:
            accepted = perturbations.LayoutValidationResult(True)
            with mock.patch.object(
                gen_s2d.spatial_validator,
                "validate_s2d_layout",
                return_value=accepted,
            ):
                random.seed(42)
                baseline = gen_s2d.generate_safe_s2d_layout(
                    configs.S2D[2], 2, "pump_bottle_flat_groove"
                )
                random.seed(42)
                aligned = gen_s2d.generate_safe_s2d_layout(
                    configs.S2D[2],
                    2,
                    "pump_bottle_flat_groove",
                    source_plate_yaw_offset_deg=90.0,
                )

            self.assertEqual(aligned[:3], baseline[:3])
            self.assertAlmostEqual(aligned[5], aligned[6] + 90.0)
            self.assertEqual(aligned[6], baseline[6])
        finally:
            random.setstate(random_state)

    def test_s2d_seeded_retries_preserve_legacy_yaws_and_source_slots(self):
        random_state = random.getstate()
        try:
            cases = (
                (
                    1041145151,
                    270,
                    "pump_bottle_upright_inlay",
                    (-0.036225619527215584, -0.03082309049017188),
                    (287.98271029313884, 288.01937611711406),
                    [5, 1, 6, 2, 3],
                ),
                (
                    2846276042,
                    90,
                    "serum_bottle_flat_inlay",
                    (-0.043640566155539254, 0.022588429743108512),
                    (73.20268538628646, 106.52607334580117),
                    [6, 4, 1, 3],
                ),
            )
            for (
                seed,
                transpose,
                inlay_name,
                expected_shift,
                expected_yaws,
                expected_slots,
            ) in cases:
                random.seed(seed)
                random.uniform(*configs.S2D[3]["speed_range"])
                layout = gen_s2d.generate_safe_s2d_layout(
                    configs.S2D[3],
                    3,
                    inlay_name,
                    transpose_degrees=(transpose,),
                )
                self.assertAlmostEqual(layout[0], expected_shift[0])
                self.assertAlmostEqual(layout[1], expected_shift[1])
                self.assertAlmostEqual(layout[5], expected_yaws[0])
                self.assertAlmostEqual(layout[6], expected_yaws[1])
                self.assertEqual(
                    random.sample(range(1, 7), len(expected_slots)),
                    expected_slots,
                )
                candidate = {
                    "plate": {
                        "pos": layout[3],
                        "yaw_deg": layout[5],
                        "xml_name": "grey_plate",
                    },
                    "ghost": {
                        "pos": layout[4],
                        "yaw_deg": layout[6],
                        "xml_name": "conveyor_ghost",
                    },
                }
                self.assertAlmostEqual(
                    self.validator.s2d_plate_track_clearance(
                        candidate, inlay_name
                    ),
                    perturbations.S2D_TRACK_REQUIRED_CLEARANCE,
                )
        finally:
            random.setstate(random_state)

    def test_s2d_seeded_sampler_uses_other_track_side_when_required(self):
        random_state = random.getstate()
        try:
            random.seed(1969407691)
            random.uniform(*configs.S2D[3]["speed_range"])
            layout = gen_s2d.generate_safe_s2d_layout(
                configs.S2D[3],
                3,
                "pump_bottle_upright_inlay",
                transpose_degrees=(90,),
            )
            self.assertAlmostEqual(layout[0], -0.014038747396864629)
            self.assertAlmostEqual(layout[1], -0.039113229371512306)
            track_normal = perturbations._rotate_xy((1.0, 0.0), layout[6])
            plate_to_ghost = (
                layout[4][0] - layout[3][0],
                layout[4][1] - layout[3][1],
            )
            self.assertLess(
                plate_to_ghost[0] * track_normal[0]
                + plate_to_ghost[1] * track_normal[1],
                0.0,
            )
        finally:
            random.setstate(random_state)

    def test_oversized_composite_formation_is_rejected_by_workspace_proof(self):
        with self.assertRaisesRegex(ValueError, "spatial workspace bounds"):
            perturbations.get_safe_symmetric_base_anchors(
                perturbations.CONVEYOR_CENTRE_REGION_RADIUS, 0.25, 0
            )

    def test_common_negative_x_correction_preserves_reach_and_bounds(self):
        correction = self.validator.negative_x_translation_to_safe_region(
            [("centre_region", (0.25, 0.0), 0.0)],
            [("bounded static fixture", (0.25, 0.0), 0.0, 0.0)],
        )
        self.assertAlmostEqual(correction, -0.05)
        corrected_x = 0.25 + correction
        self.assertTrue(self.validator._circle_is_kinematically_safe(corrected_x, 0.0))
        self.assertTrue(self.validator._circle_is_in_distribution(corrected_x, 0.0, 0.0))

    def test_dynamic_d2d_requires_two_reachable_slots(self):
        valid = self.validator._validate_dynamic_slots(
            (0.0, 0.0), 0.0, "electronics_tray"
        )
        self.assertTrue(valid, valid.reason)
        invalid = self.validator._validate_dynamic_slots(
            (0.0, 0.0), 90.0, "electronics_tray"
        )
        self.assertFalse(invalid)
        self.assertIn("dynamic slots", invalid.reason)

    def test_conveyor_belt_sites_use_the_normalized_transport_frame(self):
        root = ET.parse(CONVEYOR_BELT_XML).getroot()
        belt_body = next(
            body
            for body in root.iter("body")
            if body.attrib.get("name") == "conveyor_belt"
        )
        self.assertNotIn("quat", belt_body.attrib)

        contact_geom = next(
            geom
            for geom in belt_body.findall("geom")
            if geom.attrib.get("name") == "contact_region"
        )
        sites = {
            site.attrib["name"]: site
            for site in belt_body.findall("site")
        }
        self.assertEqual(
            contact_geom.attrib["pos"], sites["contact_region"].attrib["pos"]
        )
        self.assertEqual(
            contact_geom.attrib["size"], sites["contact_region"].attrib["size"]
        )

        centre_x = float(sites["centre_region"].attrib["pos"].split()[0])
        self.assertAlmostEqual(
            float(sites["centre_region"].attrib["pos"].split()[1]), 0.0
        )
        for prefix in ("offset_region", "tail_region", "offset_bowl", "tail_bowl"):
            for name, site in sites.items():
                if name.startswith(prefix + "_"):
                    self.assertAlmostEqual(
                        float(site.attrib["pos"].split()[0]), centre_x
                    )

        self.assertEqual(
            [
                float(sites["offset_region_{}".format(slot)].attrib["pos"].split()[1])
                for slot in range(1, 5)
            ],
            [-0.06, -0.19, -0.32, -0.45],
        )
        self.assertEqual(
            [
                float(sites["tail_region_{}".format(slot)].attrib["pos"].split()[1])
                for slot in range(1, 5)
            ],
            [0.06, 0.19, 0.32, 0.45],
        )

        # This outer marker is radius metadata, not a transport site. Keep
        # its canonical local components instead of rotating it to preserve
        # an unused world-space point.
        radius_site = next(
            site
            for site in root.iter("site")
            if site.attrib.get("name") == "horizontal_radius_site"
        )
        self.assertEqual(radius_site.attrib["pos"], "0.11405 0.57523 0")

    def test_d2d_level12_uses_opposite_x_sides_and_robot_base_clearance(self):
        for level, target in (
            (1, "electronics_tray"),
            (2, "mouse_black_taipan_upright_groove"),
        ):
            # A deterministic seed makes this an anchor invariant rather
            # than a test of the random yaw / XY noise values.
            import random

            random.seed(100 + level)
            layout = gen_d2d.generate_safe_d2d_layout(configs.D2D[level], level, target)
            belt, ghost = layout[3:5]
            belt_yaw, ghost_yaw = layout[5:7]
            self.assertLess(belt[0], 0.0)
            self.assertGreater(ghost[0], 0.0)
            self.assertAlmostEqual(belt[1] - layout[1], 0.30)
            self.assertAlmostEqual(ghost[1] - layout[1], 0.05)
            layout_dict = {
                "belt": {
                    "pos": belt,
                    "yaw_deg": belt_yaw,
                    "xml_name": "conveyor_belt",
                },
                "ghost": {
                    "pos": ghost,
                    "yaw_deg": ghost_yaw,
                    "xml_name": "conveyor_ghost",
                },
            }
            self.assertGreaterEqual(
                self.validator.d2d_track_clearance(layout_dict, target),
                perturbations.LAYOUT_CLEARANCE - 1e-9,
            )
            base_clearance = self.validator.validate_conveyor_base_clearance(
                layout_dict["belt"]
            )
            self.assertTrue(base_clearance, base_clearance.reason)
            if level == 1:
                self.assertAlmostEqual(belt_yaw, 90.0)
                self.assertAlmostEqual(ghost_yaw, 90.0)
            else:
                self.assertLessEqual(abs(belt_yaw - 90.0), configs.D2D[2]["yaw_range"])
                self.assertLessEqual(abs(ghost_yaw - 90.0), configs.D2D[2]["yaw_range"])
            self.assertAlmostEqual(belt_yaw, ghost_yaw)

    def test_d2d_level3_uses_local_parallel_track_convention(self):
        self.assertEqual(configs.D2D[2]["yaw_range"], 5)
        self.assertEqual(
            gen_d2d.D2D_LV3_BASE_ANCHORS[0],
            ((-0.105, 0.30), (-0.055, 0.05)),
        )
        self.assertEqual(
            gen_d2d.D2D_LV3_BASE_ANCHORS[180],
            ((-0.105, -0.30), (-0.15, -0.05)),
        )
        for transpose, expected in gen_d2d.D2D_LV3_BASE_ANCHORS.items():
            self.assertEqual(gen_d2d._d2d_lv3_base_anchors(transpose), expected)
        with self.assertRaisesRegex(ValueError, "cardinal"):
            gen_d2d._d2d_lv3_base_anchors(45)

        headset_y = self.validator.d2d_required_track_separation(
            "headset_upright_inlay", load_axis="y"
        )
        mouse_y = self.validator.d2d_required_track_separation(
            "mouse_white_taipan_upright_inlay", load_axis="y"
        )
        mouse_x = self.validator.d2d_required_track_separation(
            "mouse_white_taipan_upright_inlay", load_axis="x"
        )
        self.assertGreater(headset_y, mouse_y)
        self.assertGreater(mouse_x, mouse_y)

    def test_d2d_level3_samples_each_cardinal_transpose(self):
        import random

        for transpose in (0, 90, 180, 270):
            random.seed(300 + transpose)
            layout = gen_d2d.generate_safe_d2d_layout(
                configs.D2D[3],
                3,
                "mouse_black_taipan_upright_inlay",
                transpose_degrees=(transpose,),
            )
            self.assertEqual(layout[2], transpose)
            self.assertAlmostEqual(layout[5], layout[6])
            if transpose in (0, 180):
                self.assertLessEqual(
                    abs(layout[5] - (transpose - 90.0)), 5.0
                )
                self.assertLessEqual(
                    abs(layout[1]), configs.D2D[3]["shift_x_range"]
                )
                direction = 1.0 if transpose == 0 else -1.0
                self.assertAlmostEqual(
                    layout[3][1] - layout[1], direction * 0.30
                )
                self.assertAlmostEqual(
                    layout[4][1] - layout[1], direction * 0.05
                )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            gen_d2d.generate_safe_d2d_layout(
                configs.D2D[3],
                3,
                "mouse_black_taipan_upright_inlay",
                transpose_degrees=(0, 90),
            )

    def test_d2d_level3_schedule_balances_angles_speeds_and_targets(self):
        from collections import Counter, defaultdict

        import regenerate_industry_benchmark as regen

        catalog = regen.build_catalog(42)
        d2d_level3 = [
            spec for spec in catalog if spec.suite == "D2D" and spec.level == 3
        ]
        self.assertEqual(
            {0: 6, 90: 6, 180: 6, 270: 6},
            dict(Counter(spec.transpose_deg for spec in d2d_level3)),
        )
        by_speed = defaultdict(Counter)
        by_target = defaultdict(Counter)
        for spec in d2d_level3:
            by_speed[spec.speed_code][spec.transpose_deg] += 1
            by_target[spec.target][spec.transpose_deg] += 1
            self.assertEqual(
                (spec.transpose_deg,), regen._transpose_candidates(spec, Counter())
            )
        expected_angles = Counter({0: 1, 90: 1, 180: 1, 270: 1})
        self.assertTrue(all(counts == expected_angles for counts in by_speed.values()))
        self.assertTrue(
            all(
                set(counts) == set(regen.TRANSPOSE_DEGREES)
                and sorted(counts.values()) == [1, 1, 2, 2]
                for counts in by_target.values()
            )
        )
        expected_matrix = {
            "mouse_black_taipan": (0, 90, 180, 270, 90, 180),
            "mouse_white_taipan": (90, 180, 270, 0, 180, 270),
            "mouse_naga": (180, 270, 0, 90, 270, 0),
            "headset": (270, 0, 90, 180, 0, 90),
        }
        actual_matrix = {
            target: tuple(
                next(
                    spec.transpose_deg
                    for spec in d2d_level3
                    if spec.target == target and spec.speed_code == speed_code
                )
                for speed_code in regen.SPEEDS
            )
            for target in expected_matrix
        }
        self.assertEqual(expected_matrix, actual_matrix)

    def test_d2d_catalog_preserves_both_filename_speed_codes(self):
        import regenerate_industry_benchmark as regen

        d2d = [spec for spec in regen.build_catalog(42) if spec.suite == "D2D"]
        self.assertEqual(len(d2d), 72)
        for spec in d2d:
            if spec.level == 1:
                self.assertEqual(spec.speed_code, spec.slot_speed_code)
                self.assertEqual(spec.base_speed, spec.slot_base_speed)
            elif spec.level == 2:
                self.assertGreater(spec.base_speed * spec.slot_base_speed, 0.0)
                self.assertNotEqual(abs(spec.base_speed), abs(spec.slot_base_speed))
            else:
                self.assertLess(spec.base_speed * spec.slot_base_speed, 0.0)

    def test_runtime_horizon_uses_asset_derived_groove_pitch(self):
        from libero.libero.benchmark.industry_benchmark_config import (
            groove_pitch_for_asset,
        )

        self.assertAlmostEqual(
            groove_pitch_for_asset("electronics_tray", "x"), 0.19
        )
        self.assertAlmostEqual(
            groove_pitch_for_asset("electronics_tray", "y"), 0.205
        )

    def test_dynamic_slot_validation_uses_load_axis_spacing(self):
        inlay = "mouse_white_taipan_upright_inlay"
        x_slots = self.validator._reachable_track_slots(
            (0.0, 0.0), 0.0, inlay, load_axis="x"
        )
        y_slots = self.validator._reachable_track_slots(
            (0.0, 0.0), 0.0, inlay, load_axis="y"
        )
        self.assertNotAlmostEqual(
            math.dist(*x_slots), math.dist(*y_slots), places=6
        )
        with self.assertRaisesRegex(ValueError, "LoadSlot axis"):
            self.validator._reachable_track_slots(
                (0.0, 0.0), 0.0, inlay, load_axis="z"
            )

    def test_d2d_base_clearance_uses_contact_region_xml_yaw(self):
        """D2D rejects an exact base-overlap candidate in the XML frame."""
        self.assertAlmostEqual(
            self.validator._contact_region_yaw_offset("conveyor_belt"), 0.0
        )
        for transpose in gen_d2d.D2D_LV3_TRANSPOSE_DEGREES:
            belt, _ = gen_d2d._d2d_lv3_base_anchors(transpose)
            result = self.validator.validate_conveyor_base_clearance(
                {
                    "pos": belt,
                    "yaw_deg": float(transpose - 90.0),
                    "xml_name": "conveyor_belt",
                }
            )
            self.assertTrue(result, result.reason)

    def test_d2d_level3_collision_validator_rejects_overlapping_tracks(self):
        target = "mouse_black_taipan_upright_inlay"
        separation = self.validator.d2d_required_track_separation(target)
        belt, reference_ghost = gen_d2d._d2d_lv3_base_anchors(90)
        ghost = gen_d2d._contract_ghost_anchor(
            belt, reference_ghost, 0.0, separation, 0
        )
        layout = {
            "belt": {
                "pos": belt,
                "yaw_deg": 0.0,
                "xml_name": "conveyor_belt",
                "safety_radius": 0.0,
                "skip_spatial_bounds": True,
                "skip_kinematic_bounds": True,
            },
            "ghost": {
                "pos": ghost,
                "yaw_deg": 0.0,
                "xml_name": "conveyor_ghost",
                "safety_radius": 0.0,
                "skip_spatial_bounds": True,
                "skip_kinematic_bounds": True,
            },
        }
        valid = self.validator.validate_d2d_layout(
            layout, target
        )
        self.assertTrue(valid, valid.reason)
        layout["ghost"]["pos"] = (belt[0] + 0.05, belt[1])
        collision = self.validator.validate_d2d_layout(
            layout, target
        )
        self.assertFalse(collision)
        self.assertIn("dynamic groove track", collision.reason)

    def test_d2s_level2_places_three_grooves_on_negative_x_y_line(self):
        import random

        random.seed(220)
        assets = [
            "mini_fire_engine_upright_groove",
            "mini_roller_v_groove",
            "school_bus_upright_groove",
        ]
        layout = gen_d2s.generate_safe_d2s_layout(configs.D2S[2], 2, assets)
        shift_x = layout[0]
        belt = layout[3]
        groove_positions = layout[5]
        groove_yaws = layout[6]
        self.assertLessEqual(abs(shift_x), gen_d2s.D2S_SHARED_X_PERTURBATION)
        self.assertAlmostEqual(
            belt[0], gen_d2s.D2S_LV12_BELT_ANCHOR[0] + shift_x
        )
        self.assertTrue(
            all(
                math.isclose(
                    position[0], gen_d2s.D2S_STATIC_Y_LINE_X + shift_x
                )
                for position in groove_positions
            )
        )
        self.assertTrue(all(position[0] < 0.0 for position in groove_positions))
        self.assertEqual(groove_positions, sorted(groove_positions, key=lambda pos: pos[1]))
        result = gen_d2s._validate_lv2_y_line(
            belt, groove_positions, assets, groove_yaws
        )
        self.assertTrue(result, result.reason)

    def test_d2s_validator_uses_runtime_workbench_base(self):
        self.assertAlmostEqual(gen_d2s.spatial_validator.base_pos[0], -0.66)
        self.assertAlmostEqual(self.validator.base_pos[0], -0.56)

    def test_d2s_level1_tray_keeps_full_common_x_perturbation(self):
        for shift_x in (
            -gen_d2s.D2S_SHARED_X_PERTURBATION,
            gen_d2s.D2S_SHARED_X_PERTURBATION,
        ):
            belt = (gen_d2s.D2S_LV12_BELT_ANCHOR[0] + shift_x, 0.0)
            tray = (gen_d2s.D2S_LV1_TRAY_ANCHOR[0] + shift_x, 0.0)
            result = gen_d2s._validate_lv1_tray(belt, tray, "toy_tray", 0.0)
            self.assertTrue(result, result.reason)

    def test_d2s_level2_belt_yaw_uses_configured_noise(self):
        import random

        assets = [
            "mini_fire_engine_upright_groove",
            "mini_roller_v_groove",
            "school_bus_upright_groove",
        ]
        random.seed(221)
        belt_yaws = [
            gen_d2s.generate_safe_d2s_layout(configs.D2S[2], 2, assets)[7]
            for _ in range(12)
        ]
        self.assertTrue(
            all(
                abs(yaw) <= configs.D2S[2]["yaw_range"]
                for yaw in belt_yaws
            )
        )
        self.assertGreater(len({round(yaw, 6) for yaw in belt_yaws}), 1)

    def test_d2s_level3_uses_transpose_not_bddl_yaw_for_formation(self):
        import random
        from unittest import mock

        assets = [
            "mario_flat_inlay",
            "mini_fire_engine_flat_inlay",
            "porter_flat_inlay",
        ]
        for rot_deg in (0, 90, 180, 270):
            random.seed(320 + rot_deg)
            with mock.patch.object(gen_d2s.random, "choice", return_value=rot_deg):
                layout = gen_d2s.generate_safe_d2s_layout(configs.D2S[3], 3, assets)
            shift_x = layout[0]
            shift_y = layout[1]
            belt, positions, yaws, belt_yaw = layout[3], layout[5], layout[6], layout[7]
            self.assertLessEqual(abs(shift_x), gen_d2s.D2S_LV3_X_PERTURBATION)
            self.assertLessEqual(
                abs(belt_yaw - rot_deg),
                configs.D2S[3]["yaw_range"],
            )
            for yaw in yaws:
                self.assertLessEqual(
                    abs(math.degrees(yaw) - (90.0 + rot_deg)),
                    configs.D2S[3]["yaw_range"],
                )
            x_span = max(position[0] for position in positions) - min(
                position[0] for position in positions
            )
            y_span = max(position[1] for position in positions) - min(
                position[1] for position in positions
            )
            if rot_deg in (90, 270):
                self.assertAlmostEqual(
                    belt[0], gen_d2s.D2S_LV3_D2D_BELT_ANCHOR[0] + shift_x
                )
                self.assertAlmostEqual(
                    belt[1], gen_d2s.D2S_LV3_D2D_BELT_ANCHOR[1] + shift_y
                )
                self.assertGreaterEqual(belt[1], 0.36)
                self.assertLessEqual(belt[1], 0.44)
                self.assertTrue(all(position[1] < belt[1] for position in positions))
                self.assertLess(x_span, 0.25)
                self.assertGreater(y_span, 0.15)
                self.assertLess(positions[0][1], positions[1][1])
                self.assertLess(positions[0][1], positions[2][1])
                self.assertLess(
                    min(positions[1][0], positions[2][0]),
                    positions[0][0],
                )
                self.assertLess(
                    positions[0][0],
                    max(positions[1][0], positions[2][0]),
                )
                self.assertLessEqual(
                    abs(positions[1][1] - positions[2][1]),
                    2.0 * gen_d2s.D2S_LV3_LOCAL_JITTER,
                )
                if rot_deg == 270:
                    self.assertLess(positions[1][0], positions[2][0])
                else:
                    self.assertGreater(positions[1][0], positions[2][0])
            else:
                self.assertAlmostEqual(belt[0], shift_x)
                self.assertTrue(all(position[0] < belt[0] for position in positions))
                self.assertTrue(
                    all(
                        math.isclose(
                            position[0], gen_d2s.D2S_STATIC_Y_LINE_X + shift_x
                        )
                        for position in positions
                    )
                )
                self.assertLess(x_span, 0.03)
                self.assertGreater(y_span, 0.10)
                self.assertEqual(positions, sorted(positions, key=lambda pos: pos[1]))
            result = gen_d2s._validate_lv3_triangle(
                belt, rot_deg, positions, assets, yaws, belt_yaw_deg=belt_yaw
            )
            self.assertTrue(result, result.reason)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            gen_d2s.generate_safe_d2s_layout(
                configs.D2S[3],
                3,
                assets,
                transpose_degrees=(90, 270),
            )

    def test_d2s_level3_keeps_its_scheduled_transpose(self):
        from collections import Counter

        import regenerate_industry_benchmark as regen

        d2s_level3 = [
            spec
            for spec in regen.build_catalog(42)
            if spec.suite == "D2S" and spec.level == 3
        ]
        self.assertEqual(24, len(d2s_level3))
        for spec in d2s_level3:
            self.assertEqual(
                (spec.transpose_deg,), regen._transpose_candidates(spec, Counter())
            )

    def test_timed_source_can_start_outside_the_workspace(self):
        sim = _FakeSim(
            {
                "conveyor_belt_1_centre_region": (-0.1, 0.0, 0.8),
                "conveyor_belt_1_tail_region_4": (-0.1, 0.2, 0.8),
                "conveyor_belt_1_offset_region_4": (-0.1, -0.2, 0.8),
            }
        )
        start, _, _ = _conveyor_timing_target(
            sim,
            "conveyor_belt_1_offset_region_4",
            {"speed": 0.03, "takt": 6.2, "offset": 1.5, "slot": 4},
        )
        self.assertLess(start[1], perturbations.S2D_BOUNDS["y_min"])

    def test_runtime_timing_uses_canonical_centre_region(self):
        sim = _FakeSim(
            {
                "conveyor_belt_1_centre_region": (-0.1, 0.0, 0.8),
                "conveyor_belt_1_tail_region_4": (-0.1, 0.2, 0.8),
                "conveyor_belt_1_offset_region_1": (-0.1, -0.2, 0.8),
            }
        )
        target, centre, _ = _conveyor_timing_target(
            sim,
            "conveyor_belt_1_offset_region_1",
            {"speed": 0.05, "takt": 1.0, "offset": 1.0, "slot": 1},
        )
        self.assertTrue((centre == (-0.1, 0.0)).all())
        self.assertTrue((target == (-0.1, -0.05)).all())

    def test_generic_slot_number_matches_static_and_timed_arrival_order(self):
        root = ET.parse(CONVEYOR_BELT_XML).getroot()
        site_y = {
            site.attrib["name"]: float(site.attrib["pos"].split()[1])
            for site in root.iter("site")
        }
        for prefix in ("offset", "tail"):
            static_distances = [
                abs(site_y["{}_region_{}".format(prefix, slot)])
                for slot in range(1, 5)
            ]
            self.assertEqual(static_distances, sorted(static_distances))

            sim = _FakeSim(
                {
                    "conveyor_belt_1_centre_region": (0.0, 0.0, 0.8),
                    "conveyor_belt_1_tail_region_4": (0.0, 0.45, 0.8),
                    **{
                        "conveyor_belt_1_{}_region_{}".format(prefix, slot): (
                            0.0,
                            site_y["{}_region_{}".format(prefix, slot)],
                            0.8,
                        )
                        for slot in range(1, 5)
                    },
                }
            )
            speed = 0.05 if prefix == "offset" else -0.05
            starts = [
                _conveyor_timing_target(
                    sim,
                    "conveyor_belt_1_{}_region_{}".format(prefix, slot),
                    {"speed": speed, "takt": 1.0, "offset": 1.5, "slot": slot},
                )[0]
                for slot in range(1, 5)
            ]
            timed_distances = [float(np.linalg.norm(start)) for start in starts]
            self.assertEqual(timed_distances, sorted(timed_distances))
            self.assertEqual(timed_distances[0], abs(speed) * 1.5)
            self.assertEqual(timed_distances[-1], abs(speed) * 4.5)

    def test_timing_uses_sampled_belt_pose_before_sim_fixture_placement(self):
        # Conditional object placement sees the unplaced XML sites. A yaw-90
        # fixture pose must transform the canonical centre and tail axis into
        # the world frame in which the belt will be placed.
        sim = _FakeSim(
            {
                "conveyor_belt_1_centre_region": (0.02, 0.0, 0.049),
                "conveyor_belt_1_offset_region_1": (0.02, -0.06, 0.049),
                "conveyor_belt_1_tail_region_4": (0.02, 0.45, 0.049),
            }
        )
        target, centre, _ = _conveyor_timing_target(
            sim,
            "conveyor_belt_1_offset_region_1",
            {"speed": 0.05, "takt": 1.0, "offset": 1.5, "slot": 1},
            reference_pos=(-0.23, 0.2014, 0.9196),
            # Fixture placement quaternions use wxyz ordering.
            reference_quat=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
        )
        self.assertAlmostEqual(centre[0], -0.23)
        self.assertAlmostEqual(centre[1], 0.2214)
        self.assertAlmostEqual(target[0], -0.155)
        self.assertAlmostEqual(target[1], 0.2214)


if __name__ == "__main__":
    unittest.main()
