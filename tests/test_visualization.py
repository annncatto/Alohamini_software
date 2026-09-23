import importlib.util
import json
import sys
import threading
import unittest
from types import ModuleType
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec("numpy"), "PC image dependencies unavailable")
class VisualizationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        from alohamini.apps import visualization

        self.np, self.viz = np, visualization
        self.rr = ModuleType("rerun")
        for name in (
            "log",
            "Scalars",
            "Image",
            "DepthImage",
            "init",
            "spawn",
            "connect_grpc",
            "send_blueprint",
            "rerun_shutdown",
        ):
            setattr(self.rr, name, Mock())
        blueprint = ModuleType("rerun.blueprint")
        for name in ("Spatial2DView", "TimeSeriesView", "Grid", "Blueprint"):
            setattr(blueprint, name, Mock())
        self.blueprint = blueprint
        # Restore only mocked entries; unloading newly imported native extensions
        # (e.g. cv2) through a whole sys.modules snapshot breaks later imports.
        for name, module in {"rerun": self.rr, "rerun.blueprint": blueprint}.items():
            if name in sys.modules:
                self.addCleanup(sys.modules.__setitem__, name, sys.modules[name])
            else:
                self.addCleanup(sys.modules.pop, name, None)
            sys.modules[name] = module
        self.viz.init_rerun()

    def test_images_are_temporal_not_static_entities(self):
        self.viz.log_rerun_data({"forward": self.np.zeros((8, 16, 3), dtype=self.np.uint8)})
        self.assertFalse(self.rr.log.call_args.kwargs.get("static", False))

    def test_camera_arriving_after_first_state_gets_a_view(self):
        self.viz.log_rerun_data({"arm_left_gripper.pos": 10.0}, {"arm_left_gripper.pos": 20.0})
        self.viz.log_rerun_data({"forward": self.np.zeros((8, 16, 3), dtype=self.np.uint8)})
        self.blueprint.Spatial2DView.assert_called_with(
            origin="observation.forward", name="observation.forward"
        )
        self.assertEqual(self.rr.send_blueprint.call_count, 2)
        self.viz.log_rerun_data({"arm_left_gripper.pos": 11.0})
        self.assertEqual(self.rr.send_blueprint.call_count, 2)

    def test_scalar_and_image_names_keep_original_namespaces(self):
        self.viz.log_rerun_data({"joint.pos": 1.0, "observation.other": 2.0}, {"joint.pos": 3.0})
        self.assertEqual(
            [c.args[0] for c in self.rr.log.call_args_list],
            ["observation.joint.pos", "observation.other", "action.joint.pos"],
        )

    def test_init_and_shutdown_use_existing_viewer_operations(self):
        self.rr.init.assert_called_once_with("alohamini_teleop")
        self.rr.spawn.assert_called_once_with(memory_limit="10%")
        self.viz.shutdown_rerun()
        self.rr.rerun_shutdown.assert_called_once()

    def test_snapshot_preserves_rgb_and_does_not_create_missing_camera_frames(self):
        import cv2

        from alohamini.protocol import HostSnapshot

        rgb = self.np.zeros((8, 16, 3), dtype=self.np.uint8)
        rgb[:, :, 0] = 255
        ok, jpeg = cv2.imencode(".jpg", rgb)
        self.assertTrue(ok)
        snapshot = HostSnapshot(
            {"joint.pos": 1.0, "_safety": {}, "_robot_metadata": {}},
            {"forward": jpeg.tobytes()},
            1.0,
            1.01,
        )
        with patch.object(self.viz, "log_rerun_data") as log:
            self.viz.log_snapshot(snapshot, {"joint.pos": 2.0})
        observation, action = log.call_args.args
        self.assertEqual(set(observation), {"joint.pos", "forward"})
        self.assertGreater(int(observation["forward"][0, 0, 0]), 240)
        self.assertLess(int(observation["forward"][0, 0, 2]), 10)
        self.assertEqual(action, {"joint.pos": 2.0})

    def test_bad_jpeg_does_not_replace_real_state_or_fabricate_black_frame(self):
        from alohamini.protocol import HostSnapshot

        snapshot = HostSnapshot({"joint.pos": 1.0}, {"forward": b"bad-jpeg"}, 1.0, 1.01)
        with patch.object(self.viz, "log_rerun_data") as log:
            self.viz.log_snapshot(snapshot, {})
        self.assertEqual(log.call_args.args, ({"joint.pos": 1.0}, {}))

    def test_slow_preview_is_latest_only_and_does_not_block_control(self):
        from test_teleoperation import snapshot

        from alohamini.apps.teleoperation import run_loop

        entered, release, latest_seen = threading.Event(), threading.Event(), threading.Event()
        rendered = []

        def render(_snapshot, action):
            rendered.append(action["joint.pos"])
            if len(rendered) == 1:
                entered.set()
                release.wait(2)
            else:
                latest_seen.set()

        stop = threading.Event()
        client, leader = Mock(client_id="client"), Mock()
        client.read.side_effect = snapshot
        leader.read.return_value = {"joint.pos": 1}
        count = 0

        def send(*_args, **_kwargs):
            nonlocal count
            count += 1
            if count == 4:
                stop.set()
            return Mock()

        client.send_command.side_effect = send
        with patch.object(self.viz, "log_snapshot", side_effect=render):
            with self.viz.TeleopPreview(None, "alohamini2pro", fps=30) as worker:
                try:
                    worker.submit(snapshot(), {"joint.pos": 0})
                    self.assertTrue(entered.wait(1))
                    with patch("alohamini.apps.teleoperation.stop_owned_robot"):
                        run_loop(
                            client,
                            "alohamini2pro",
                            leader,
                            None,
                            on_frame=worker.submit,
                            stop_event=stop,
                        )
                    self.assertEqual(count, 4)
                    for index in range(2, 100):
                        worker.submit(snapshot(), {"joint.pos": index})
                    self.assertEqual(rendered, [0])
                    release.set()
                    self.assertTrue(latest_seen.wait(1))
                finally:
                    release.set()
            self.assertFalse(worker._thread.is_alive())
        self.assertEqual(rendered, [0, 99])
        self.assertTrue(
            all(not call.kwargs.get("include_images") for call in client.read.call_args_list)
        )

    def test_camera_subscription_is_bounded_session_bound_and_standard_rgb(self):
        import cv2
        import zmq
        from test_teleoperation import snapshot

        sample = snapshot()
        sample.payload["_robot_metadata"]["cameras"] = ["forward"]
        sample.payload["_host_timing"] = {"state_sample_monotonic_s": 2.0}
        worker = self.viz.TeleopPreview("test-only", "alohamini2pro", fps=30)

        def packet(seq, *, session="session", name="forward", stamp=2.0, red=True):
            bgr = self.np.zeros((8, 16, 3), dtype=self.np.uint8)
            bgr[:, :, 2 if red else 0] = 255
            ok, jpeg = cv2.imencode(".jpg", bgr)
            self.assertTrue(ok)
            metadata = dict(
                schema_version=1,
                camera_name=name,
                sequence=seq,
                width=16,
                height=8,
                encoding="jpeg",
                capture_monotonic_s=stamp,
                capture_unix_ns=2_000_000_000,
                host_session_id=session,
            )
            return [f"camera/{name}".encode(), json.dumps(metadata).encode(), jpeg.tobytes()]

        context, socket = Mock(), Mock()
        context.socket.return_value = socket
        socket.recv_multipart.side_effect = [
            packet(1, red=False),
            packet(2),
            packet(1, red=False),
            packet(3, session="old-session", red=False),
            packet(3, stamp=1.0, red=False),
            packet(3, name="disabled"),
            [b"bad"],
            zmq.Again(),
        ]
        worker.submit(sample, {})
        with (
            patch("zmq.Context", return_value=context),
            patch.object(self.viz, "log_snapshot"),
            patch.object(
                self.viz, "log_rerun_data", side_effect=lambda *_: worker._stop.set()
            ) as log,
        ):
            worker._run()
        context.socket.assert_called_once_with(zmq.SUB)
        socket.connect.assert_called_once_with("tcp://test-only:5557")
        socket.send.assert_not_called()
        image = log.call_args.args[0]["forward"]
        self.assertGreater(int(image[0, 0, 0]), 240)
        self.assertLess(int(image[0, 0, 2]), 10)
        self.assertEqual(set(log.call_args.args[0]), {"forward"})
        socket.close.assert_called_once_with(linger=0)
        context.term.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("rerun"), "PC viewer dependency unavailable")
class RerunMemoryTests(unittest.TestCase):
    def test_preview_worker_writes_to_real_sdk_without_viewer(self):
        import rerun as rr
        from test_teleoperation import snapshot

        from alohamini.apps import visualization

        rr.init("alohamini-worker-test", spawn=False, strict=True)
        complete = threading.Event()
        original = visualization.log_snapshot

        def render(*args):
            original(*args)
            complete.set()

        try:
            memory = rr.memory_recording()
            visualization.log_rerun_data.paths = (set(), set(), set())
            with patch.object(visualization, "log_snapshot", side_effect=render):
                with visualization.TeleopPreview(None, "alohamini2pro", fps=30) as worker:
                    worker.submit(snapshot(), {"joint.pos": 2.0})
                    self.assertTrue(complete.wait(1))
            self.assertGreater(memory.num_msgs(), 0)
            self.assertTrue(memory.drain_as_bytes())
        finally:
            rr.rerun_shutdown()

    def test_real_sdk_accepts_native_state_action_and_images_without_viewer(self):
        import numpy as np
        import rerun as rr

        from alohamini.apps import visualization

        rr.init("alohamini-offline-test", spawn=False, strict=True)
        try:
            memory = rr.memory_recording()
            visualization.log_rerun_data.paths = (set(), set(), set())
            visualization.log_rerun_data({"joint.pos": 1.0}, {"joint.pos": 2.0})
            visualization.log_rerun_data({"forward": np.zeros((8, 16, 3), dtype=np.uint8)})
            self.assertGreater(memory.num_msgs(), 0)
            self.assertTrue(memory.drain_as_bytes())
        finally:
            rr.rerun_shutdown()
