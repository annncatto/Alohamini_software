import os
import subprocess
import sys
import unittest
from pathlib import Path


class DependencyTests(unittest.TestCase):
    def test_imports_work_without_site_packages_or_network_access(self):
        code = """
import importlib.util
import socket
import sys
socket.socket = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('network access'))
import alohamini
from alohamini.client import HostClient
from alohamini.protocol import (
    command_target_keys, decode_command_context, encode_command, encode_request,
)
from alohamini.schema import CommandIdentity
assert encode_request('test') == b'test:state'
assert decode_command_context({}, client_id='test') is None
assert encode_command({'x.vel': 0}, CommandIdentity('test', 1, 'session', 0),
                      allowed_targets=command_target_keys('alohamini2pro'))
import alohamini.cli
import alohamini.model
import alohamini.schema
import alohamini.kinematics
import alohamini.runtime.command_owner
import alohamini.calibration
import alohamini.hardware.feetech
import alohamini.hardware.feetech_device
import alohamini.hardware.find_port
import alohamini.runtime.feedback
import alohamini.runtime.lifecycle
import alohamini.runtime.arm_control
import alohamini.runtime.base_lift_control
import alohamini.runtime.host
import alohamini.runtime.camera_stream
import alohamini.runtime.startup
assert not any(name.startswith(('alohamini.apps', 'alohamini.datasets')) for name in sys.modules)
import alohamini.calibration.servo
import alohamini.calibration.procedure
import alohamini.hardware.leader
import alohamini.apps.teleoperation
import alohamini.apps.teleop_monitor
import alohamini.apps.recording
import alohamini.datasets
for name in ('native', 'images', 'tools', 'lerobot'):
    assert importlib.util.find_spec(f'alohamini.datasets.{name}') is not None
for name in ('dataset', 'dataset_images', 'dataset_tools', 'dataset_lerobot',
             'teleoperation', 'recording', 'teleop_monitor', 'visualization'):
    assert importlib.util.find_spec(f'alohamini.{name}') is None
from alohamini.hardware.camera import CameraConfig, OpenCVCamera
camera = OpenCVCamera(CameraConfig('/dev/test-camera'))
camera.close()
client = HostClient('127.0.0.1')
client.close()
forbidden = {'lerobot', 'rclpy', 'torch', 'cv2', 'numpy', 'zmq', 'serial', 'scservo_sdk', 'pynput'}
assert not forbidden & sys.modules.keys()
"""
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.run(
            [sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=5
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
