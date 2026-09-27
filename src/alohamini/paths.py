"""Workspace storage locations shared by AlohaMini applications.

Resolving a path does not create directories, move files or load a framework.
Applications create storage only when writing. These are organizational defaults,
not a filesystem sandbox; explicitly supplied external paths remain possible.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_root() -> Path:
    value = os.environ.get("ALOHAMINI_WORKSPACE")
    if value is None:
        return Path.home() / "Alohamini_workspace"
    if not value.strip():
        raise ValueError("ALOHAMINI_WORKSPACE must not be empty")
    return Path(value)


def _name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or value != value.strip()
        or value in (".", "..")
        or any(char in '/\\<>:"|?*' or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("Storage name must be one nonempty filename component")
    return value


@dataclass(frozen=True)
class WorkspacePaths:
    """Use an explicit root, ALOHAMINI_WORKSPACE, or ~/Alohamini_workspace.

    The root must be absolute so changing the launch directory cannot redirect
    files. A WorkspacePaths instance retains its root for the application's lifetime.
    """

    root: Path = field(default_factory=_default_root)

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser()
        if not root.is_absolute():
            raise ValueError("AlohaMini workspace must be an absolute path")
        object.__setattr__(self, "root", root)

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def pretrained(self) -> Path:
        return self.root / "pretrained"

    @property
    def incoming(self) -> Path:
        return self.root / "incoming"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def calibration_file(self, role: str, device_id: str) -> Path:
        if role not in ("robots", "teleoperators"):
            raise ValueError("Calibration role must be robots or teleoperators")
        return self.calibration / role / f"{_name(device_id)}.json"

    def dataset(self, name: str) -> Path:
        return self.datasets / _name(name)

    def run(self, name: str) -> Path:
        return self.runs / _name(name)

    def incoming_batch(self, name: str) -> Path:
        return self.incoming / _name(name)
