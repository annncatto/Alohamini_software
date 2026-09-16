"""Versioned JSON model loading without ROS, YAML or repository-path dependencies."""

import json
import re
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

from alohamini._validation import finite_number, identifier

from .types import ActuatorSpec, RobotModel, asset_path


def _asset_root() -> Path:
    try:
        resource = files("alohamini._assets")
    except ModuleNotFoundError as exc:
        if exc.name != "alohamini._assets":
            raise
        # Uninstalled source checkouts (including Python -S) have no package mapping.
        # Bind to this module's checkout, never the cwd or an unrelated old repo.
        checkout = Path(__file__).resolve().parents[3]
        if not (checkout / "pyproject.toml").is_file():
            raise RuntimeError("Model resources are missing; reinstall alohamini-platform") from exc
        resource = checkout / "models"
    if not isinstance(resource, Path):
        raise RuntimeError("Model assets require a filesystem installation")
    return resource


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate model field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite model number: {value}")


def _read_json(path: Path) -> dict:
    with path.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("Model metadata exceeds 1 MiB")
    try:
        value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except RecursionError as exc:
        raise ValueError("Model metadata is nested too deeply") from exc
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
        raise ValueError("Model metadata must specify an integer schema_version")
    if value["schema_version"] != 1:
        raise ValueError("Unsupported model schema_version")
    return value


def _model_id(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value):
        raise ValueError("Invalid model_id")


def _catalog() -> tuple[Path, tuple[str, ...]]:
    root = _asset_root()
    names = _read_json(root / "catalog.json").get("models")
    if not isinstance(names, list) or not names:
        raise ValueError("Model catalog must contain a nonempty models list")
    for name in names:
        _model_id(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate model IDs in catalog")
    return root, tuple(names)


def robot_models() -> tuple[str, ...]:
    return _catalog()[1]


def load_robot_model(directory: str | Path) -> RobotModel:
    """Load a self-contained model directory; never synthesize missing geometry."""
    directory = Path(directory).resolve()
    data = _read_json(asset_path(directory, "model.json"))
    required = {
        "schema_version",
        "asset_revision",
        "model_id",
        "actuators",
        "wheel_radius_m",
        "base_radius_m",
        "lift_lead_m_per_rev",
        "descriptions",
    }
    if not required <= data.keys():
        raise ValueError(f"Missing model fields: {sorted(required - data.keys())}")
    _model_id(data["model_id"])
    revision = data["asset_revision"]
    if type(revision) is not int or revision < 1:
        raise ValueError("asset_revision must be a positive integer")
    for name in ("wheel_radius_m", "base_radius_m", "lift_lead_m_per_rev"):
        finite_number(data[name], name)
        if data[name] <= 0:
            raise ValueError(f"{name} must be positive")
    rows = data["actuators"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("actuators must be a nonempty list")
    actuators = []
    names, addresses = set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bus", "motor_id", "motor_model"}:
            raise ValueError("Invalid actuator fields")
        for field in ("name", "bus", "motor_model"):
            identifier(row[field], field)
        if type(row["motor_id"]) is not int or not 1 <= row["motor_id"] <= 253:
            raise ValueError("motor_id must be in [1, 253]")
        address = row["bus"], row["motor_id"]
        if row["name"] in names or address in addresses:
            raise ValueError("Duplicate actuator name or bus address")
        names.add(row["name"])
        addresses.add(address)
        actuators.append(ActuatorSpec(**row))
    descriptions = data["descriptions"]
    if not isinstance(descriptions, dict):
        raise ValueError("descriptions must be an object")
    for name, relative in descriptions.items():
        identifier(name, "description name")
        asset_path(directory, relative)
    return RobotModel(
        model_id=data["model_id"],
        actuators=tuple(actuators),
        wheel_radius_m=data["wheel_radius_m"],
        base_radius_m=data["base_radius_m"],
        lift_lead_m_per_rev=data["lift_lead_m_per_rev"],
        asset_revision=revision,
        descriptions=MappingProxyType(dict(descriptions)),
        directory=directory,
    )


def get_robot_model(model_id: str) -> RobotModel:
    _model_id(model_id)
    root, names = _catalog()
    if model_id not in names:
        raise ValueError(f"Unknown robot model {model_id!r}; expected one of {names}")
    model = load_robot_model(asset_path(root, f"{model_id}/model.json").parent)
    if model.model_id != model_id:
        raise ValueError("Catalog and model manifest IDs disagree")
    return model
