cmake_minimum_required(VERSION 3.5)
project(go1_model)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY xacro
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
