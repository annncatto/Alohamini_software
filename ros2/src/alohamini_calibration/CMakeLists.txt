cmake_minimum_required(VERSION 3.8)
project(alohamini_calibration)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/calibrate_camera_intrinsics
    scripts/calibrate_hand_eye
    scripts/capture_camera_calibration
    scripts/capture_hand_eye_samples
    scripts/generate_charuco_board
    scripts/sync_arm_mapping
    scripts/calibrate_lift_axis
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES scripts/camera_board.py scripts/calibration_io.py scripts/hardware_capture.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_camera_board test/test_camera_board.py)
  ament_add_pytest_test(test_camera_workflow test/test_camera_workflow.py)
  ament_add_pytest_test(test_hand_eye_stationary_gate test/test_hand_eye_stationary_gate.py)
  ament_add_pytest_test(test_sync_arm_mapping test/test_sync_arm_mapping.py)
  ament_add_pytest_test(test_lift_axis_calibration test/test_lift_axis_calibration.py)
endif()

install(FILES ../../../LICENSE ../../../NOTICE DESTINATION share/${PROJECT_NAME})

ament_package()
