cmake_minimum_required(VERSION 3.15)
cmake_policy(VERSION 3.15)
project(medficientsam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(BUILD_SHARED_LIBS OFF)

set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libs)

add_compile_definitions(XTENSOR_USE_XSIMD)
add_compile_definitions(XTENSOR_DISABLE_EXCEPTIONS)

set(XTL_INCLUDE_DIR ${LIBS_DIR}/xtl/include)
include_directories(${XTL_INCLUDE_DIR})

set(XSIMD_INCLUDE_DIR ${LIBS_DIR}/xsimd/include)
include_directories(${XSIMD_INCLUDE_DIR})

set(XTENSOR_INCLUDE_DIR ${LIBS_DIR}/xtensor/include)
include_directories(${XTENSOR_INCLUDE_DIR})

set(XTENSOR_IO_INCLUDE_DIR ${LIBS_DIR}/xtensor-io/include)
configure_file(
  "${XTENSOR_IO_INCLUDE_DIR}/xtensor-io/xtensor_io_config.hpp.in"
  "${XTENSOR_IO_INCLUDE_DIR}/xtensor-io/xtensor_io_config.hpp"
)
include_directories(${XTENSOR_IO_INCLUDE_DIR})

set(WITH_GZFILEOP OFF)
set(ZLIB_COMPAT ON)
set(ZLIB_ENABLE_TESTS OFF)
set(ZLIBNG_ENABLE_TESTS OFF)
set(WITH_GTEST OFF)
add_subdirectory(${LIBS_DIR}/zlib-ng)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenVINO REQUIRED)

add_executable(main src/main.cpp)
set_target_properties(main PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
target_link_libraries(main opencv_core opencv_imgproc openvino::runtime zlibstatic)
target_compile_options(main PRIVATE -march=skylake -mtune=skylake)
