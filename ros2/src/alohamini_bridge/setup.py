from glob import glob

from setuptools import find_packages, setup

package_name = "alohamini_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "../../../LICENSE", "../../../NOTICE"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="ROS 2 state and command bridge for the native AlohaMini Host.",
    license="Apache-2.0",
    entry_points={"console_scripts": ["bridge_node = alohamini_bridge.bridge_node:main"]},
)
