from glob import glob

from setuptools import find_packages, setup

package_name = "alohamini_validation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "../../../LICENSE", "../../../NOTICE"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Offline validation tools and golden baselines for AlohaMini.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "validate_assets = alohamini_validation.validate_assets:main",
            "validate_moveit = alohamini_validation.validate_moveit:main",
            "validate_tf = alohamini_validation.validate_tf:main",
        ]
    },
)
