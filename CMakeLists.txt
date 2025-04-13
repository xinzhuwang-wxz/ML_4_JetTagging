cmake_minimum_required(VERSION 3.0)
project(BDT4JOI)

set(CMAKE_CXX_STANDARD 20)

find_package(yaml-cpp REQUIRED)

find_package(ROOT REQUIRED COMPONENTS Core)
find_package(Python3 REQUIRED)


include_directories(${YAML_CPP_INCLUDE_DIRS})
include_directories(${ROOT_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(inc)
include_directories(${CMAKE_SOURCE_DIR}/core)


add_executable(bdt_joi main.cpp src/config_reader.cpp src/Bdtrain.cpp)

target_include_directories(bdt_joi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bdt_joi yaml-cpp ${ROOT_LIBRARIES})

add_custom_target(run_python_script
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/chain2csv.py --config ${CMAKE_SOURCE_DIR}/config.yaml
    DEPENDS ${CMAKE_SOURCE_DIR}/core/chain2csv.py
)

add_custom_target(run_python_script_1
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/core/training.py  --config ${CMAKE_SOURCE_DIR}/config.yaml
    DEPENDS ${CMAKE_SOURCE_DIR}/core/training.py
)

install(TARGETS bdt_joi DESTINATION bin)