cmake_minimum_required(VERSION 3.22)
project(haversine-py-ffi LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Python3 REQUIRED COMPONENTS
	Interpreter
	Development)

include_directories(${Python3_INCLUDE_DIRS})

cuda_add_library(haversine SHARED
	haversine.cu)

target_link_libraries(haversine
	${Python3_LIBRARY_DIRS}
	${CUDA_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set_target_properties(haversine PROPERTIES PREFIX "")

