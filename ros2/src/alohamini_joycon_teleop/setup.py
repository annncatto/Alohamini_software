from glob import glob

from setuptools import find_packages, setup

package_name = "alohamini_joycon_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.rviz")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/scripts", glob("scripts/*.py")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Guarded Joy-Con teleoperation for AlohaMini ROS 2.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "teleop_node = alohamini_joycon_teleop.teleop_node:main",
            "joycon_input_log = alohamini_joycon_teleop.input_log:main",
        ]
    },
)
