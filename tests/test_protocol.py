import json
import unittest

from support import state_frames, state_payload

from alohamini.errors import ModelMismatchError, ProtocolError
from alohamini.protocol import (
    MAX_JSON_BYTES,
    HostSnapshot,
    command_target_keys,
    decode_command,
    decode_command_context,
    decode_reply,
    encode_command,
    encode_reply,
    encode_request,
)
from alohamini.schema import CommandIdentity


class CommandProtocolTests(unittest.TestCase):
    def test_lift_stop_is_stop_only_and_cannot_be_combined_with_height(self):
        identity = CommandIdentity("client", 1, "session", 0)
        keys = command_target_keys("alohamini2pro")
        encoded = encode_command({"lift_axis.stop": 1.0}, identity, allowed_targets=keys)
        self.assertEqual(
            decode_command(encoded, allowed_targets=keys), (identity, {"lift_axis.stop": 1.0})
        )
        for fields in (
            {"lift_axis.stop": 0},
            {"lift_axis.stop": True},
            {"lift_axis.stop": -1},
            {"lift_axis.stop": 1, "lift_axis.height_mm": 100},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                encode_command(fields, identity, allowed_targets=keys)
            payload = json.loads(encoded)
            payload.pop("lift_axis.stop")
            payload.update(fields)
            with self.assertRaises(ProtocolError):
                decode_command(json.dumps(payload).encode(), allowed_targets=keys)

    def test_host_command_decoding_and_reply_encoding_round_trip(self):
        identity = CommandIdentity("client", 1, "session", 0)
        keys = command_target_keys("alohamini2pro")
        targets = {"x.vel": 0.1, "arm_left_shoulder_pan.pos": 20}
        self.assertEqual(
            decode_command(
                encode_command(targets, identity, allowed_targets=keys), allowed_targets=keys
            ),
            (identity, targets),
        )
        parts = [b"test:full", *encode_reply(state_payload(), {"front": b"jpeg"})]
        payload, images = decode_reply(parts, token=b"test:full", include_images=True)
        self.assertEqual(images, {"front": b"jpeg"})
        self.assertEqual(payload["_images"], ["front"])

    def test_host_rejects_missing_bad_duplicate_or_nonfinite_command_fields(self):
        for data in (
            b"{}",
            b"[]",
            b'{"x.vel":0}',
            b'{"x.vel":NaN}',
            b'{"x.vel":0,"x.vel":1}',
            b"\xff",
            b'{"x.vel":0,"_command":{"client_id":"x","sequence":true,"host_session_id":"s","control_epoch":0}}',
        ):
            with self.subTest(data=data), self.assertRaises(ProtocolError):
                decode_command(data, allowed_targets=frozenset({"x.vel"}))

    def test_explicitly_invalid_feedback_cannot_bind_a_command_context(self):
        payload = {
            "_safety": {
                "version": 1,
                "control_owner": None,
                "host_session_id": "s",
                "control_epoch": 0,
                "feedback_valid": False,
            }
        }
        self.assertIsNone(decode_command_context(payload, client_id="pc"))

    def test_request_wire_format(self):
        self.assertEqual(encode_request("sample"), b"sample:state")
        self.assertEqual(encode_request("sample", include_images=True), b"sample:full")

    def test_request_rejects_ambiguous_ids_and_modes(self):
        for request_id in (None, "", "a:state", "a b", "a\n", "请求", "a" * 65):
            with self.subTest(request_id=request_id), self.assertRaises(ValueError):
                encode_request(request_id)
        with self.assertRaises(ValueError):
            encode_request("sample", include_images=1)

    def test_context_preserves_session_and_epoch_without_mutating_payload(self):
        for owner in (None, "client"):
            payload = {
                "_safety": {
                    "version": 1,
                    "control_owner": owner,
                    "host_session_id": "session",
                    "control_epoch": 2,
                }
            }
            before = json.dumps(payload)
            self.assertEqual(
                decode_command_context(payload, client_id="client"),
                CommandIdentity("client", 0, "session", 2),
            )
            self.assertEqual(json.dumps(payload), before)

    def test_invalid_or_unavailable_context_does_not_authorize_commands(self):
        valid = {
            "version": 1,
            "control_owner": None,
            "host_session_id": "session",
            "control_epoch": 0,
        }
        candidates = [
            None,
            [],
            {},
            *({k: v for k, v in valid.items() if k != missing} for missing in valid),
        ]
        candidates.extend(
            {**valid, **patch}
            for patch in (
                {"version": True},
                {"version": 2},
                {"control_epoch": True},
                {"control_epoch": -1},
                {"control_epoch": 1.0},
                {"host_session_id": ""},
                {"host_session_id": None},
                {"control_owner": "other"},
            )
        )
        for status in candidates:
            with self.subTest(status=status):
                self.assertIsNone(decode_command_context({"_safety": status}, client_id="client"))
        self.assertIsNone(decode_command_context({}, client_id="client"))

    def test_contact_hold_is_not_a_protocol_send_prohibition(self):
        payload = {
            "_safety": {
                "version": 1,
                "control_owner": None,
                "host_session_id": "session",
                "control_epoch": 0,
                "joint_holds": {"arm_left_shoulder_pan": 0.2},
            }
        }
        self.assertIsNotNone(decode_command_context(payload, client_id="client"))

    def test_all_models_keep_existing_command_envelopes(self):
        # Reference envelope used by the deployed ROS2 and LeRobot Host clients.
        count = 0
        for model, expected_count in (
            ("alohamini1", 16),
            ("alohamini2", 18),
            ("alohamini2pro", 18),
        ):
            keys = command_target_keys(model)
            legacy_keys = keys - {"lift_axis.stop"}
            self.assertEqual(len(legacy_keys), expected_count)
            for epoch in (0, 1, 123):
                identity = CommandIdentity("client", 7, "session", epoch)
                for key in legacy_keys:
                    targets = {key: -12.5}
                    expected = {
                        **targets,
                        "_command": {
                            "client_id": "client",
                            "sequence": 7,
                            "host_session_id": "session",
                            "control_epoch": epoch,
                        },
                    }
                    with self.subTest(model=model, epoch=epoch, key=key):
                        self.assertEqual(
                            encode_command(targets, identity, allowed_targets=keys),
                            json.dumps(expected, allow_nan=False, separators=(",", ":")).encode(),
                        )
                        self.assertEqual(targets, {key: -12.5})
                    count += 1
        self.assertEqual(count, 156)

    def test_invalid_targets_cannot_be_encoded(self):
        keys = command_target_keys("alohamini2pro")
        identity = CommandIdentity("client", 1, "session", 0)
        for targets in (
            None,
            [],
            {},
            {"x.vel": True},
            {"x.vel": float("nan")},
            {"x.vel": float("inf")},
            {"x.vel": "1"},
            {"unknown": 1},
            {"lift_axis.vel": 100},
            {"_command": 1},
        ):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                encode_command(targets, identity, allowed_targets=keys)
        with self.assertRaises(ValueError):
            encode_command({"x.vel": 0}, None, allowed_targets=keys)

    def test_command_metadata_cannot_be_injected_even_with_bad_allowlist(self):
        with self.assertRaises(ValueError):
            encode_command(
                {"_command": 1},
                CommandIdentity("client", 1, "session", 0),
                allowed_targets=frozenset({"_command"}),
            )


class ProtocolTests(unittest.TestCase):
    token = b"sample:state"

    def test_preserves_normalized_units_and_partial_feedback(self):
        payload, images = decode_reply(state_frames(self.token), token=self.token)
        self.assertEqual(payload["arm_left_shoulder_pan.pos"], -12.5)
        self.assertEqual(payload["lift_axis.height_mm"], 120.0)
        feedback = payload["_motor_feedback"]["motors"]["arm_left_shoulder_pan"]
        self.assertNotIn("velocity_raw", feedback)
        self.assertEqual(feedback["current_ma"], 65)
        self.assertEqual(images, {})

    def test_preserves_unknown_extensions(self):
        expected = state_payload()
        expected["_future_extension"] = {"version": 9, "values": [1, None, "a"]}
        actual, _ = decode_reply(state_frames(self.token, expected), token=self.token)
        self.assertEqual(actual, expected)

    def test_reads_do_not_share_dictionaries(self):
        first, _ = decode_reply(state_frames(self.token), token=self.token)
        first["_safety"]["control_owner"] = "someone"
        second, _ = decode_reply(state_frames(self.token), token=self.token)
        self.assertIsNone(second["_safety"]["control_owner"])

    def test_optional_feedback_is_not_required_or_fabricated(self):
        payload = state_payload()
        del payload["_motor_feedback"]
        actual, _ = decode_reply(state_frames(self.token, payload), token=self.token)
        self.assertNotIn("_motor_feedback", actual)

    def test_robot_model_mismatch_is_explicit(self):
        with self.assertRaises(ModelMismatchError):
            decode_reply(state_frames(self.token), token=self.token, expected_model="alohamini1")

    def test_rejects_malformed_envelopes(self):
        for parts in ([], [self.token], [self.token, b"{}", b"camera"], ["sample", b"{}"]):
            with self.subTest(parts=parts), self.assertRaises(ProtocolError):
                decode_reply(parts, token=self.token)

    def test_rejects_wrong_token(self):
        with self.assertRaises(ProtocolError):
            decode_reply(state_frames(b"old:state"), token=self.token)

    def test_rejects_invalid_json_and_duplicate_fields(self):
        for value in (b"[]", b"null", b"{", b"\xff", b'{"a":1,"a":2}'):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                decode_reply([self.token, value], token=self.token)

    def test_rejects_nonfinite_values_including_exponent_overflow(self):
        for value in ("NaN", "Infinity", "-Infinity", "1e999"):
            raw = json.dumps(state_payload())[:-1] + ',"invalid":' + value + "}"
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                decode_reply([self.token, raw.encode()], token=self.token)

    def test_rejects_unsupported_or_missing_metadata(self):
        for metadata in (
            None,
            [],
            {},
            {"schema_version": True, "robot_model": "a"},
            {"schema_version": 2, "robot_model": "a"},
            {"schema_version": 1, "robot_model": ""},
        ):
            payload = state_payload()
            payload["_robot_metadata"] = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ProtocolError):
                decode_reply(state_frames(self.token, payload), token=self.token)

    def test_rejects_invalid_safety_or_timing_containers(self):
        for key in ("_safety", "_host_timing"):
            payload = state_payload()
            payload[key] = []
            with self.subTest(key=key), self.assertRaises(ProtocolError):
                decode_reply(state_frames(self.token, payload), token=self.token)

    def test_rejects_excessive_json_size_and_nesting(self):
        with self.assertRaises(ProtocolError):
            decode_reply([self.token, b" " * (MAX_JSON_BYTES + 1)], token=self.token)
        payload = state_payload()
        nested = {}
        payload["extra"] = nested
        for _ in range(65):
            nested["extra"] = {}
            nested = nested["extra"]
        with self.assertRaises(ProtocolError):
            decode_reply(state_frames(self.token, payload), token=self.token)

    def test_images_are_returned_as_original_bytes(self):
        token = b"sample:full"
        payload = state_payload()
        payload["_images"] = ["forward", "wrist_right"]
        parts = state_frames(token, payload) + [b"forward", b"jpeg1", b"wrist_right", b"jpeg2"]
        actual, images = decode_reply(parts, token=token, include_images=True)
        self.assertEqual(actual, payload)
        self.assertEqual(images, {"forward": b"jpeg1", "wrist_right": b"jpeg2"})
        with self.assertRaises(ProtocolError):
            decode_reply(parts, token=token)

    def test_rejects_inconsistent_camera_metadata(self):
        for names, tail in (
            (["forward"], []),
            (["forward"], [b"wrong", b"jpeg"]),
            (["forward"], [b"forward", b""]),
            (["forward"], [b"\xff", b"jpeg"]),
            (["forward", "forward"], [b"forward", b"a", b"forward", b"b"]),
            ([None], []),
            (None, []),
        ):
            payload = state_payload()
            payload["_images"] = names
            with self.subTest(names=names, tail=tail), self.assertRaises(ProtocolError):
                decode_reply(
                    state_frames(self.token, payload) + tail, token=self.token, include_images=True
                )

    def test_does_not_confuse_host_and_client_clocks(self):
        payload = state_payload()
        snapshot = HostSnapshot(payload, {}, 50000.0, 50000.025)
        self.assertAlmostEqual(snapshot.round_trip_s, 0.025)
        self.assertEqual(snapshot.robot_model, "alohamini2pro")
        self.assertEqual(snapshot.payload["_host_timing"]["state_sample_monotonic_s"], 100.125)


if __name__ == "__main__":
    unittest.main()
