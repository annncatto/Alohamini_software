import math
import unittest
from dataclasses import FrozenInstanceError, replace

from alohamini.calibration import EncoderCalibration, JointPositionDecoder, LiftCalibration
from alohamini.calibration.encoder import HostPositionUnits


class HostPositionUnitTests(unittest.TestCase):
    def test_normalized_endpoints_inversion_and_saturation(self):
        for mode, lower, middle, upper in (
            ("range_m100_100", -100, 0, 100),
            ("range_0_100", 0, 50, 100),
        ):
            for drive in (0, 1):
                units = HostPositionUnits(mode, 1000, 3000, drive)
                ends = (3000, 1000) if drive else (1000, 3000)
                self.assertEqual(units.to_tick(lower), ends[0])
                self.assertEqual(units.to_tick(middle), 2000)
                self.assertEqual(units.to_tick(upper), ends[1])
                self.assertEqual(units.to_tick(-1000), ends[0])
                self.assertEqual(units.to_tick(1000), ends[1])
                self.assertEqual(units.from_tick(ends[0]), lower)
                self.assertEqual(units.from_tick(2000), middle)
                self.assertEqual(units.from_tick(ends[1]), upper)

    def test_degree_units_keep_4095_scale_and_integer_truncation(self):
        units = HostPositionUnits("degrees", 1000, 3000, 1)
        self.assertEqual(units.to_tick(12.5), int(12.5 * 4095 / 360 + 2000))
        self.assertEqual(units.from_tick(2200), 200 * 360 / 4095)
        with self.assertRaises(ValueError):
            units.to_tick(300)

    def test_invalid_configuration_or_target_fails(self):
        for args in (
            ("unknown", 1, 2, 0),
            ("degrees", 100, 100, 0),
            ("degrees", -1, 3000, 0),
            ("degrees", 1000, 3000, True),
        ):
            with self.assertRaises(ValueError):
                HostPositionUnits(*args)
        units = HostPositionUnits("degrees", 1000, 3000, 0)
        for value in (True, float("nan"), float("inf"), 1e308):
            with self.assertRaises(ValueError):
                units.to_tick(value)


class EncoderCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = EncoderCalibration(
            4096, 2048, 0.0, 1, position_min_rad=-2, position_max_rad=2
        )

    def test_reference_direction_and_transmission_ratio(self):
        self.assertEqual(self.calibration.position_from_tick(2048), 0)
        reverse = replace(self.calibration, direction=-1, joint_per_encoder_ratio=0.5)
        self.assertAlmostEqual(reverse.position_from_tick(2560), -math.pi / 8)
        self.assertAlmostEqual(reverse.velocity_from_ticks_per_second(4096), -math.pi)

    def test_target_round_trip_respects_tick_quantization(self):
        for direction in (-1, 1):
            c = replace(self.calibration, direction=direction)
            for position in (-2, -1.2, 0, 0.7, 2):
                tick = c.position_to_tick(position)
                self.assertLessEqual(abs(c.position_from_tick(tick) - position), math.pi / 4096)

    def test_initial_branch_matches_calibrated_interval(self):
        c = EncoderCalibration(
            4096, 969, -1.571, -1, position_min_rad=-5.0532, position_max_rad=-1.5617
        )
        self.assertAlmostEqual(c.position_from_tick(3239), -5.0531363885, places=9)

    def test_continuity_across_encoder_zero(self):
        c = EncoderCalibration(4096, 0, 0, 1)
        before = c.position_from_tick(4090)
        after = c.position_from_tick(5, previous_position_rad=before)
        self.assertAlmostEqual(after - before, 11 * math.tau / 4096)

    def test_continuity_across_principal_half_turn(self):
        c = EncoderCalibration(4096, 0, 0, 1)
        before = c.position_from_tick(2040)
        after = c.position_from_tick(2050, previous_position_rad=before)
        self.assertGreater(after, math.pi)
        self.assertAlmostEqual(after - before, 10 * math.tau / 4096)

    def test_ambiguous_initial_turn_requires_explicit_reference(self):
        c = replace(self.calibration, position_min_rad=-8, position_max_rad=8)
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            c.position_from_tick(2048)
        self.assertAlmostEqual(c.position_from_tick(2048, previous_position_rad=6), math.tau)

    def test_feedback_is_not_clamped_at_target_limits(self):
        c = replace(self.calibration, position_min_rad=-0.1, position_max_rad=0.1)
        self.assertGreater(c.position_from_tick(2200), 0.1)

    def test_target_limits_and_existing_boundary_tolerance(self):
        c = self.calibration
        self.assertEqual(c.position_to_tick(2 + 0.00005), c.position_to_tick(2))
        self.assertEqual(c.position_to_tick(-2 - 0.00005), c.position_to_tick(-2))
        for value in (-2.001, 2.001):
            with self.assertRaises(ValueError):
                c.position_to_tick(value)
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            EncoderCalibration(4096, 0, 0, 1).position_to_tick(0)

    def test_invalid_calibration_is_rejected(self):
        invalid = (
            {"ticks_per_revolution": 0},
            {"ticks_per_revolution": True},
            {"reference_tick": -1},
            {"reference_tick": 4096},
            {"reference_tick": 1.2},
            {"direction": 0},
            {"direction": True},
            {"joint_per_encoder_ratio": 0},
            {"joint_per_encoder_ratio": -1},
            {"reference_position_rad": float("nan")},
            {"position_min_rad": 2},
            {"position_max_rad": None},
            {"position_max_rad": float("inf")},
        )
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(ValueError):
                replace(self.calibration, **update)
        with self.assertRaises(FrozenInstanceError):
            self.calibration.direction = -1

    def test_invalid_feedback_and_targets_are_rejected(self):
        for tick in (-1, 4096, 0.1, True, None, "12"):
            with self.subTest(tick=tick), self.assertRaises(ValueError):
                self.calibration.position_from_tick(tick)
        for value in (float("nan"), float("inf"), True, "1", 10**400):
            for method in (
                self.calibration.position_to_tick,
                self.calibration.velocity_from_ticks_per_second,
            ):
                with self.subTest(value=value, method=method), self.assertRaises(ValueError):
                    method(value)


class JointPositionDecoderTests(unittest.TestCase):
    def setUp(self):
        self.calibrations = {
            "left_joint": EncoderCalibration(4096, 0, 0, 1),
            "right_joint": EncoderCalibration(4096, 0, 0.2, -1),
        }
        self.decoder = JointPositionDecoder(self.calibrations)

    def test_sides_have_independent_references_and_directions(self):
        positions = self.decoder.decode({"left_joint": 0, "right_joint": 0})
        self.assertEqual(positions, {"left_joint": 0, "right_joint": 0.2})
        positions = self.decoder.decode({"left_joint": 10, "right_joint": 10})
        self.assertGreater(positions["left_joint"], 0)
        self.assertLess(positions["right_joint"], 0.2)

    def test_reset_starts_a_new_stream(self):
        self.decoder.decode({"left_joint": 2040})
        self.assertGreater(self.decoder.decode({"left_joint": 2050})["left_joint"], math.pi)
        self.decoder.reset()
        self.assertLess(self.decoder.decode({"left_joint": 2050})["left_joint"], 0)

    def test_missing_joint_is_not_filled_and_loses_continuity(self):
        self.decoder.decode({"left_joint": 2040})
        self.assertEqual(self.decoder.decode({"right_joint": 0}), {"right_joint": 0.2})
        self.assertLess(self.decoder.decode({"left_joint": 2050})["left_joint"], 0)
        self.assertEqual(self.decoder.decode({}), {})

    def test_invalid_batch_does_not_leave_partial_or_old_continuity(self):
        self.decoder.decode({"left_joint": 2040})
        with self.assertRaises(ValueError):
            self.decoder.decode({"left_joint": 2050, "right_joint": -1})
        self.assertLess(self.decoder.decode({"left_joint": 2050})["left_joint"], 0)

    def test_unknown_joint_has_no_calibration_fallback(self):
        with self.assertRaisesRegex(ValueError, "No calibration"):
            self.decoder.decode({"unknown": 100})

    def test_caller_cannot_mutate_calibration_or_continuity(self):
        self.calibrations.clear()
        result = self.decoder.decode({"left_joint": 2040})
        result["left_joint"] = -100
        self.assertGreater(self.decoder.decode({"left_joint": 2050})["left_joint"], 0)


class LiftCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = LiftCalibration(0.0, 0.6, -0.3, 0.3)

    def test_endpoints_and_midpoint_in_meters(self):
        for physical, joint in ((0, -0.3), (0.3, 0), (0.6, 0.3)):
            self.assertAlmostEqual(self.calibration.height_to_position(physical), joint)
            self.assertAlmostEqual(self.calibration.position_to_height(joint), physical)

    def test_feedback_overrun_is_preserved_but_targets_are_rejected(self):
        self.assertAlmostEqual(self.calibration.height_to_position(0.605), 0.305)
        self.assertAlmostEqual(self.calibration.height_to_position(-0.005), -0.305)
        for value in (-0.305, 0.305):
            with self.assertRaises(ValueError):
                self.calibration.position_to_height(value)

    def test_nonunit_scale_and_nonzero_origin(self):
        c = LiftCalibration(0.1, 0.5, -0.3, 0.3)
        self.assertAlmostEqual(c.height_to_position(0.3), 0)
        self.assertAlmostEqual(c.position_to_height(0), 0.3)

    def test_invalid_ranges_and_nonfinite_values(self):
        for args in (
            (0, 0, -0.3, 0.3),
            (0, -1, -0.3, 0.3),
            (0, 1, 1, 0),
            (0, float("inf"), 0, 1),
            (True, 1, 0, 1),
            (-1e308, 1e308, 0, 1),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                LiftCalibration(*args)
        for value in (float("nan"), float("inf"), True, "0.1"):
            for method in (
                self.calibration.height_to_position,
                self.calibration.position_to_height,
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    method(value)
