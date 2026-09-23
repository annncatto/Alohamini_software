import json
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from alohamini.model import get_robot_model, load_robot_model


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        self.model = get_robot_model("alohamini2pro")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.data = json.loads(self.model.asset_path("model.json").read_text())
        self.data["descriptions"] = {}

    def write_model(self):
        (self.directory / "model.json").write_text(json.dumps(self.data))

    def test_all_description_mesh_references_are_model_local_and_present(self):
        for name in self.model.descriptions:
            path = self.model.description_path(name)
            root = ET.parse(path).getroot()
            self.assertEqual(root.tag, "robot")
            for mesh in root.findall(".//mesh"):
                relative = mesh.attrib["filename"]
                self.assertNotIn(":", relative)
                self.assertFalse(Path(relative).is_absolute())
                resolved = (path.parent / relative).resolve()
                self.assertTrue(resolved.is_relative_to(self.model.directory))
                self.assertTrue(resolved.is_file(), resolved)

    def test_no_device_calibration_or_old_wheel_preview_is_packaged(self):
        names = {p.name for p in self.model.directory.rglob("*")}
        for excluded in (
            "AlohaMiniRobot.json",
            "hardware_joint_map_left.yaml",
            "hardware_joint_map_right.yaml",
            "arm_home.yaml",
            "intrinsics",
            "hand_eye",
            "lidar_three_wheel",
            "description_state.yaml",
        ):
            self.assertNotIn(excluded, names)

    def test_description_xml_resolves_meshes_without_changing_assets(self):
        from urllib.parse import unquote, urlparse

        for name in self.model.descriptions:
            path = self.model.description_path(name)
            original = path.read_bytes()
            rendered = ET.fromstring(self.model.description_xml(name))
            for mesh in rendered.findall(".//mesh"):
                uri = urlparse(mesh.attrib["filename"])
                self.assertEqual(uri.scheme, "file")
                resource = Path(unquote(uri.path))
                self.assertTrue(resource.is_relative_to(self.model.directory))
                self.assertTrue(resource.is_file())
            self.assertEqual(path.read_bytes(), original)

    def test_semantic_description_retains_groups_and_geometry_references(self):
        urdf = ET.parse(self.model.description_path("collision")).getroot()
        semantic = ET.parse(self.model.description_path("semantic")).getroot()
        self.assertEqual(urdf.attrib["name"], semantic.attrib["name"])
        links = {link.attrib["name"] for link in urdf.findall("link")}
        self.assertTrue(
            {"left_arm", "right_arm", "dual_arms", "lift"}
            <= {group.attrib["name"] for group in semantic.findall("group")}
        )
        for chain in semantic.findall(".//chain"):
            self.assertIn(chain.attrib["base_link"], links)
            self.assertIn(chain.attrib["tip_link"], links)
        for pair in semantic.findall("disable_collisions"):
            self.assertIn(pair.attrib["link1"], links)
            self.assertIn(pair.attrib["link2"], links)

    def test_custom_model_loads_from_its_own_directory(self):
        self.data["model_id"] = "custom-robot"
        self.data["wheel_radius_m"] = 0.07
        self.write_model()
        loaded = load_robot_model(self.directory)
        self.assertEqual(loaded.model_id, "custom-robot")
        self.assertEqual(loaded.wheel_radius_m, 0.07)
        self.assertEqual(loaded.directory, self.directory.resolve())
        self.assertEqual(self.model.wheel_radius_m, 0.063)

    def test_unsupported_version_or_missing_geometry_is_rejected(self):
        original = dict(self.data)
        for updates in (
            {"schema_version": 2},
            {"schema_version": True},
            {"asset_revision": 0},
            {"wheel_radius_m": 0},
            {"base_radius_m": float("inf")},
            {"lift_lead_m_per_rev": -0.1},
            {"model_id": "../outside"},
        ):
            with self.subTest(updates=updates):
                self.data = {**original, **updates}
                self.write_model()
                with self.assertRaises(ValueError):
                    load_robot_model(self.directory)
        self.data = dict(original)
        del self.data["wheel_radius_m"]
        self.write_model()
        with self.assertRaises(ValueError):
            load_robot_model(self.directory)

    def test_duplicate_addresses_and_names_are_rejected(self):
        for field in ("motor_id", "name"):
            self.data = json.loads(self.model.asset_path("model.json").read_text())
            self.data["descriptions"] = {}
            self.data["actuators"][1][field] = self.data["actuators"][0][field]
            self.write_model()
            with self.assertRaises(ValueError):
                load_robot_model(self.directory)

    def test_duplicate_json_keys_are_rejected(self):
        path = self.directory / "model.json"
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_robot_model(self.directory)

    def test_asset_path_cannot_escape_through_paths_or_symlinks(self):
        for relative in ("../model.json", "/tmp/file", "package://something", "..\\file", ""):
            with self.subTest(path=relative), self.assertRaises(ValueError):
                self.model.asset_path(relative)
        self.data["descriptions"] = {"collision": "outside.urdf"}
        self.write_model()
        (self.directory / "outside.urdf").symlink_to(self.model.description_path("collision"))
        with self.assertRaises(ValueError):
            load_robot_model(self.directory)

    def test_missing_description_fails_without_model_fallback(self):
        with self.assertRaises(ValueError):
            get_robot_model("alohamini1").description_path("collision")
        with self.assertRaises(TypeError):
            self.model.descriptions["collision"] = "other.urdf"
        self.data["descriptions"] = {"collision": "missing.urdf"}
        self.write_model()
        with self.assertRaises(FileNotFoundError):
            load_robot_model(self.directory)

    def test_source_assets_work_without_installation_or_current_working_directory(self):
        source = Path(__file__).resolve().parents[1] / "src"
        code = (
            "from alohamini.model import get_robot_model; "
            "m=get_robot_model('alohamini2pro'); "
            "assert m.description_path('collision').is_file()"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=self.directory,
            env={**os.environ, "PYTHONPATH": str(source)},
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
