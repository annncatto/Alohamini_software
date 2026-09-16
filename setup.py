"""Package root-level model assets without a second maintained source copy."""

from pathlib import Path

from setuptools import find_packages, setup

asset_files = [
    path.relative_to("models").as_posix()
    for path in sorted(Path("models").rglob("*"))
    if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".urdf", ".stl"}
]

setup(
    packages=find_packages("src") + ["alohamini._assets"],
    package_dir={"": "src", "alohamini._assets": "models"},
    package_data={"alohamini._assets": asset_files},
)
