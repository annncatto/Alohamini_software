cmake_minimum_required(VERSION 3.8)
project(alohamini_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

install(FILES ../../../LICENSE ../../../NOTICE DESTINATION share/${PROJECT_NAME})

ament_package()
