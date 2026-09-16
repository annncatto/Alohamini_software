"""Keep deployment roles distinct without importing optional runtime packages."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def requirements(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith("-r "):
            result.update(requirements(path.parent / line[3:].strip()))
        match = re.match(r"([\w-]+)==([^\s\\]+)", line)
        if match:
            result[match[1].lower().replace("_", "-")] = match[2]
    return result


class EnvironmentManifestTests(unittest.TestCase):
    def test_pc_and_host_have_unambiguous_names(self):
        self.assertTrue((ROOT / "environment.yml").read_text().startswith("name: alohamini\n"))
        self.assertTrue((ROOT / "env/host.yml").read_text().startswith("name: alohamini_host\n"))

    def test_host_has_hardware_but_not_learning_or_keyboard_dependencies(self):
        expected = {"pyzmq", "pyserial", "feetech-servo-sdk", "numpy", "opencv-python-headless"}
        for architecture in ("linux-64", "linux-aarch64"):
            self.assertEqual(set(requirements(ROOT / f"env/host-{architecture}.lock")), expected)

    def test_pc_includes_local_leader_and_local_data_dependencies(self):
        packages = requirements(ROOT / "env/pc-linux-64.lock")
        expected = {
            "torch",
            "torchvision",
            "pyarrow",
            "pandas",
            "av",
            "torchcodec",
            "pynput",
            "rerun-sdk",
            "pyserial",
            "feetech-servo-sdk",
            "pyzmq",
        }
        self.assertTrue(expected <= packages.keys())
        self.assertFalse(
            {"lerobot", "huggingface-hub", "datasets", "transformers", "rclpy"} & packages.keys()
        )

    def test_pc_direct_dependencies_match_lock(self):
        direct = requirements(ROOT / "env/pc-linux-64.in")
        locked = requirements(ROOT / "env/pc-linux-64.lock")
        self.assertTrue(direct)
        for name, version in direct.items():
            self.assertEqual(locked[name], version, name)

    def test_shared_hardware_versions_match_between_pc_and_host(self):
        pc = requirements(ROOT / "env/pc-linux-64.lock")
        for architecture in ("linux-64", "linux-aarch64"):
            for name, version in requirements(ROOT / f"env/host-{architecture}.lock").items():
                self.assertEqual(pc[name], version, name)


if __name__ == "__main__":
    unittest.main()
