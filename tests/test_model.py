import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from alohamini.model import get_robot_model, loader, robot_models


class ModelTests(unittest.TestCase):
    def test_repeated_model_lookup_performs_no_asset_io(self):
        loader._installed_robot_model.cache_clear()
        self.addCleanup(loader._installed_robot_model.cache_clear)
        with patch.object(loader, "_read_json", wraps=loader._read_json) as read:
            model = get_robot_model("alohamini2pro")
            self.assertEqual(read.call_count, 2)
        with (
            patch.object(loader, "_asset_root", side_effect=AssertionError("asset lookup")),
            patch.object(loader, "asset_path", side_effect=AssertionError("asset IO")),
        ):
            self.assertIs(get_robot_model("alohamini2pro"), model)

    def test_failed_load_is_not_cached(self):
        loader._installed_robot_model.cache_clear()
        self.addCleanup(loader._installed_robot_model.cache_clear)
        with patch.object(loader, "_read_json", side_effect=ValueError("invalid assets")):
            with self.assertRaisesRegex(ValueError, "invalid assets"):
                get_robot_model("alohamini2pro")
        self.assertEqual(get_robot_model("alohamini2pro").model_id, "alohamini2pro")

    def test_supported_layouts_and_bus_addresses(self):
        self.assertEqual(robot_models(), ("alohamini1", "alohamini2", "alohamini2pro"))
        for name, count in (("alohamini1", 16), ("alohamini2", 18), ("alohamini2pro", 18)):
            with self.subTest(name=name):
                model = get_robot_model(name)
                self.assertEqual(len(model.actuators), count)
                self.assertEqual(len({a.name for a in model.actuators}), count)
                self.assertEqual(len({(a.bus, a.motor_id) for a in model.actuators}), count)
                self.assertEqual(
                    [(a.name, a.motor_id) for a in model.actuators if a.name.startswith("base_")],
                    [("base_left_wheel", 8), ("base_back_wheel", 9), ("base_right_wheel", 10)],
                )
                self.assertEqual(model.actuators[-1].name, "lift_axis")
                self.assertEqual(model.actuators[-1].motor_id, 11)
                self.assertEqual(model.actuators[-1].bus, "left")

    def test_pro_motors_match_corrected_hardware(self):
        model = get_robot_model("alohamini2pro")
        for motor in model.actuators:
            expected = (
                "sts3095"
                if motor.name == "lift_axis" or motor.name.endswith(("shoulder_lift", "elbow_flex"))
                else "sts3250"
            )
            self.assertEqual(motor.motor_model, expected, motor.name)

    def test_standard_models_are_not_silently_changed_to_pro(self):
        first = get_robot_model("alohamini1")
        self.assertTrue(all(a.motor_model == "sts3215" for a in first.actuators))
        self.assertFalse(any("wrist_yaw" in a.name for a in first.actuators))
        second = get_robot_model("alohamini2")
        self.assertEqual(second.actuators[0].motor_model, "sts3095")
        self.assertEqual(second.actuators[3].motor_model, "sts3215")

    def test_geometry_is_explicitly_in_meters(self):
        for name, radius, base, lead in (
            ("alohamini1", 0.05, 0.125, 0.084),
            ("alohamini2", 0.063, 0.195, 0.131),
            ("alohamini2pro", 0.063, 0.195, 0.131),
        ):
            model = get_robot_model(name)
            self.assertEqual(
                (model.wheel_radius_m, model.base_radius_m, model.lift_lead_m_per_rev),
                (radius, base, lead),
            )

    def test_catalog_cannot_be_changed_through_returned_objects(self):
        model = get_robot_model("alohamini2pro")
        with self.assertRaises(FrozenInstanceError):
            model.model_id = "different"
        with self.assertRaises(FrozenInstanceError):
            model.actuators[0].motor_model = "different"

    def test_unknown_model_has_no_fallback(self):
        for name in ("unknown", "", None, [], True):
            with self.subTest(name=name), self.assertRaises(ValueError):
                get_robot_model(name)
