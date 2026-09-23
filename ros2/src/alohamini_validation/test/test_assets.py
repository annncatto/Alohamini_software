from pathlib import Path

import pytest
from alohamini_validation.validate_assets import UrdfKinematics, main, validate_fk
from ament_index_python.packages import get_package_share_directory

from alohamini.model import get_robot_model


def test_authoritative_offline_assets():
    main()


def test_fk_drift_fails_instead_of_refreshing_the_baseline():
    description = get_robot_model("alohamini2pro")
    model = UrdfKinematics(description.description_path("kinematic"))
    joint = model.joint_by_child["left_tcp"]
    joint.find("origin").set("xyz", "0 0 0.1")
    validation = Path(get_package_share_directory("alohamini_validation"))
    with pytest.raises(AssertionError, match="FK drift"):
        validate_fk(model, description.directory, validation)


def test_cycle_in_model_is_rejected_without_hanging():
    model = UrdfKinematics(get_robot_model("alohamini2pro").description_path("collision"))
    model.joint_by_child["left_tcp"].find("parent").set("link", "left_tcp")
    with pytest.raises(AssertionError, match="cycle"):
        model.chain("root", "left_tcp")
