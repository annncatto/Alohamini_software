cmake_minimum_required(VERSION 3.16)
project(alohamini_core)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the shared Python package and model assets.
get_filename_component(PLATFORM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
ament_python_install_package(alohamini PACKAGE_DIR "${PLATFORM_ROOT}/src/alohamini")
# Model paths must remain inside one asset root, including in development builds.
# In symlink mode ament aliases CMake's copying install command as _install.
if(AMENT_CMAKE_SYMLINK_INSTALL)
  _install(DIRECTORY "${PLATFORM_ROOT}/models/"
    DESTINATION "${PYTHON_INSTALL_DIR}/alohamini/_assets"
    PATTERN "__pycache__" EXCLUDE)
else()
  install(DIRECTORY "${PLATFORM_ROOT}/models/"
    DESTINATION "${PYTHON_INSTALL_DIR}/alohamini/_assets"
    PATTERN "__pycache__" EXCLUDE)
endif()
install(FILES "${PLATFORM_ROOT}/LICENSE" "${PLATFORM_ROOT}/NOTICE"
  DESTINATION "share/${PROJECT_NAME}")

ament_package()
