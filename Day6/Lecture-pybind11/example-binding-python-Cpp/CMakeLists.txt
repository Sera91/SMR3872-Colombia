cmake_minimum_required(VERSION 3.12)

#**********************************************************************
# Compiling Pybind11 + Apache Arrow 
#**********************************************************************
# Requirements:
# - Apache Arrow
# - pybind11
#****************************************************************************


project(arrow_pybind_example VERSION 0.1.0)

find_package(Arrow CONFIG REQUIRED)
find_package(ArrowPython CONFIG REQUIRED HINTS ${Arrow_DIR})
find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

include_directories("src/include")

set(CMAKE_CXX_STANDARD 11)

pybind11_add_module(arrow_pybind_example src/example.cpp)
target_link_libraries(arrow_pybind_example PRIVATE arrow_shared arrow_python_shared)
