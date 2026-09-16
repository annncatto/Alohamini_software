import math
import unittest

from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import get_robot_model, robot_models
from alohamini.schema import BodyVelocity


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.kinematics = OmniBaseKinematics(0.063, 0.195)

    def test_ros_wheel_order_and_axes(self):
        forward = self.kinematics.body_to_wheels(BodyVelocity(x_m_s=0.1))
        left = self.kinematics.body_to_wheels(BodyVelocity(y_m_s=0.1))
        ccw = self.kinematics.body_to_wheels(BodyVelocity(yaw_rad_s=0.2))
        self.assertGreater(forward[0], 0)
        self.assertAlmostEqual(forward[1], 0)
        self.assertLess(forward[2], 0)
        self.assertLess(left[0], 0)
        self.assertGreater(left[1], 0)
        self.assertLess(left[2], 0)
        for value in ccw:
            self.assertAlmostEqual(value, 0.195 * 0.2 / 0.063)

    def test_round_trip_for_each_model_and_motion_axis(self):
        for name in robot_models():
            model = get_robot_model(name)
            kinematics = OmniBaseKinematics(model.wheel_radius_m, model.base_radius_m)
            for velocity in (
                BodyVelocity(),
                BodyVelocity(0.12, -0.03, 0.4),
                BodyVelocity(-0.2, 0.3, -0.5),
                BodyVelocity(0, 0, math.pi),
            ):
                with self.subTest(model=name, velocity=velocity):
                    actual = kinematics.wheels_to_body(kinematics.body_to_wheels(velocity))
                    for field in ("x_m_s", "y_m_s", "yaw_rad_s"):
                        self.assertAlmostEqual(getattr(actual, field), getattr(velocity, field))

    def test_yaw_is_radians_not_legacy_host_degrees(self):
        wheels = self.kinematics.body_to_wheels(BodyVelocity(yaw_rad_s=math.radians(90)))
        for value in wheels:
            self.assertAlmostEqual(value, math.pi / 2 * 0.195 / 0.063)

    def test_no_silent_speed_limiting(self):
        slow = self.kinematics.body_to_wheels(BodyVelocity(0.1, 0.2, 0.3))
        fast = self.kinematics.body_to_wheels(BodyVelocity(1, 2, 3))
        for a, b in zip(slow, fast, strict=True):
            self.assertAlmostEqual(a * 10, b)

    def test_invalid_geometry_and_nonfinite_velocities(self):
        for value in (0, -1, True, "0.1", float("nan"), float("inf")):
            for args in ((value, 0.195), (0.063, value)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    OmniBaseKinematics(*args)
        for value in (True, "1", float("nan"), float("inf"), 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BodyVelocity(x_m_s=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.kinematics.wheels_to_body((0, value, 0))

    def test_invalid_wheel_count_and_overflow(self):
        with self.assertRaises(ValueError):
            self.kinematics.wheels_to_body((0, 0))
        with self.assertRaises(ValueError):
            OmniBaseKinematics(1e-300, 0.195).body_to_wheels(BodyVelocity(x_m_s=1e300))
