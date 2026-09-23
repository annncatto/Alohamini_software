from glob import glob

from setuptools import find_packages, setup

package_name = "alohamini_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/models/high_plinth", glob("models/high_plinth/*")),
        (f"share/{package_name}/models/grasp_object", glob("models/grasp_object/*")),
        (f"share/{package_name}/models/aruco_drop_zone", glob("models/aruco_drop_zone/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="anncatto",
    maintainer_email="anncatto@users.noreply.github.com",
    description="Gazebo Fortress integration and deterministic acceptance demos for AlohaMini.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "lift_pick_place_demo = alohamini_gazebo.lift_pick_place_demo:main",
            "omni_base_adapter = alohamini_gazebo.omni_base_adapter:main",
        ],
    },
)
