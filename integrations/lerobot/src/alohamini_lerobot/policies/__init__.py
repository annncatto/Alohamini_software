"""LeRobot-backed policies with lazily imported model and processor exports."""

from importlib import import_module


def _load_export(package, exports, name):
    if name not in exports:
        raise AttributeError(f"module {package!r} has no attribute {name!r}")
    module, symbol = exports[name]
    value = getattr(import_module(module, package), symbol)
    setattr(import_module(package), name, value)
    return value
