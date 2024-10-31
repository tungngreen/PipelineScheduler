cmake_minimum_required(VERSION 3.18)
project(PipePlusPlus)

option(ON_HOST "Enable Test Builds" OFF)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# This include defines standard install locations.
# https://cmake.org/cmake/help/v3.11/module/GNUInstallDirs.html
include(GNUInstallDirs)

# This include provides some helper functions that simplify packaging.
# https://cmake.org/cmake/help/v3.11/module/CMakePackageConfigHelpers.html
include(CMakePackageConfigHelpers)

set(SYSTEM_NAME "PPP" CACHE STRING "System name: ")

message("System name: ${SYSTEM_NAME}")

set(DEVICE_ARCH "amd64" CACHE STRING "Device architecture: ")

# If no build type was specified, set it to Release.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release."
            FORCE)
endif (NOT CMAKE_BUILD_TYPE)

# If no installation prefix is given manually, install locally.
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING
            "The install location"
            FORCE)
endif (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(CMAKE_CXX_STANDARD 17)
    # Set C++ version and optimization level
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fPIC -Wall -Wno-deprecated-declarations -O0") #-Ofast

    # set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fPIC -Wall")
    # set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC -Wall -Wno-deprecated-declarations")

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Testing only available if this is the main app. Note this needs to be done
    # in the main CMakeLists since it calls enable_testing, which must be in the
    # main CMakeLists.
    include(CTest)

    # Docs only available if this is the main app
    find_package(Doxygen)
    # For finding FindTensorRT.cmake
    set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
    if (Doxygen_FOUND)
        add_subdirectory(docs)
    else ()
        message(STATUS "Doxygen not found, not building docs")
    endif ()
endif ()

# Threads
find_package(Threads REQUIRED)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

#OpenCV
find_package(OpenCV REQUIRED)

# Protobuf
option(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)
if (CMAKE_CROSSCOMPILING)
find_program(_PROTOBUF_PROTOC protoc)
else ()
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif ()

# Manually specify the path to the protobufprotocols
set(protobufprotocols_DIR "${CMAKE_CURRENT_BINARY_DIR}/../protobufprotocols")

if (${SYSTEM_NAME} STREQUAL "FCPO")
    set(CUDACXX /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER ${CUDACXX})

    # list(APPEND CMAKE_PREFIX_PATH /libtorch)
    execute_process(
        COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc --list-gpu-arch
        OUTPUT_VARIABLE DETECTED_ARCHITECTURE
        ERROR_QUIET
    )
    # Parse the architecture from the nvcc output
    string(REGEX MATCHALL "compute_([0-9]+)" DETECTED_ARCHITECTURE "${DETECTED_ARCHITECTURE}")
    foreach(ARCH ${DETECTED_ARCHITECTURE})
        string(REGEX REPLACE "compute_" "" ARCH_NUM ${ARCH})
        list(APPEND CMAKE_CUDA_ARCHITECTURES ${ARCH_NUM})
    endforeach()
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        message(WARNING "No CUDA architectures detected, resorting to native")
        set(CMAKE_CUDA_ARCHITECTURES 'native')
    else ()
        message(STATUS "Available CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif ()

    # Automatically detect the CUDA architecture using nvcc
    if (${DEVICE_ARCH} STREQUAL "amd64")
        list(APPEND CMAKE_PREFIX_PATH /libtorch)
    else()
        set (Torch_DIR /home/cdsn/.local/lib/python3.8/site-packages/torch/share/cmake/Torch)
    endif()
    find_package(Torch REQUIRED)
endif()

# Boost
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# gRPC
set(gRPC_DIR /grpc/lib/cmake/grpc)
message(${gRPC_DIR})
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_GRPC_GRPCPP gRPC::grpc++)
if (CMAKE_CROSSCOMPILING)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else ()
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif ()

# SPDLOG logging
find_package(spdlog REQUIRED)

# PostgreSQL database
find_package(PostgreSQL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBPQXX REQUIRED libpqxx)

add_subdirectory(libs/misc)
add_subdirectory(libs/profiler)
add_subdirectory(libs/microservice)
add_subdirectory(libs/utils/protobufprotocols)

if (${SYSTEM_NAME} STREQUAL "FCPO")
    add_subdirectory(libs/container_agent/fcpo_learning)
endif()

add_subdirectory(libs/controller)
add_subdirectory(libs/communicator)


if (ON_HOST STREQUAL "ON")
    message(STATUS "Building for host======================================")

    add_subdirectory(libs/device_agent)
    add_subdirectory(libs/device_agent/sink_agent)

    add_executable(Communicator libs/communicator/main.cpp)
    target_link_libraries(Communicator communicator)

    add_executable(DeviceAgent libs/device_agent/main.cpp)
    target_link_libraries(DeviceAgent device_agent)

    add_executable(SinkAgent libs/device_agent/sink_agent/main.cpp)
    target_link_libraries(SinkAgent sink_agent)
else()
    # Specify the path to TensorRT root dir
    if (NOT TensorRT_DIR)
        set(TensorRT_DIR /TensorRT/)
    endif ()
    find_package(TensorRT REQUIRED)

    add_subdirectory(libs/trtengine)
    add_subdirectory(libs/utils/onnx2trt)
    add_subdirectory(libs/baseprocessors)
    add_subdirectory(libs/container_agent)
    add_subdirectory(libs/workloads/data_source)
    add_subdirectory(libs/workloads/yolov5)
    add_subdirectory(libs/workloads/yolov5-plate)
    add_subdirectory(libs/workloads/retinaface)
    add_subdirectory(libs/workloads/arcface)
    add_subdirectory(libs/workloads/movenet)
    add_subdirectory(libs/workloads/emotionnet)
    add_subdirectory(libs/workloads/gender)
    add_subdirectory(libs/workloads/age)
    add_subdirectory(libs/workloads/carbrand)
    add_subdirectory(libs/device_agent)
    add_subdirectory(libs/device_agent/sink_agent)

    add_library(PipePlusPlusSink
        src/main.cpp)

    target_include_directories(PipePlusPlusSink PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../libs/protobufprotocols>)
    target_link_libraries(PipePlusPlusSink PUBLIC
        data_grpc_proto
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${TensorRT_LIBRARIES}
        ${_GRPC_GRPCPP}
        ${_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
        misc
        microservice
        communicator
        trtengine
        baseprocessors
    )
    add_executable(runSink src/main.cpp)
    target_link_libraries(runSink PipePlusPlusSink)

    add_executable(Communicator libs/communicator/main.cpp)
    target_link_libraries(Communicator communicator)

    add_executable(DeviceAgent libs/device_agent/main.cpp)
    target_link_libraries(DeviceAgent device_agent)

    add_executable(SinkAgent libs/device_agent/sink_agent/main.cpp)
    target_link_libraries(SinkAgent sink_agent)

    add_executable(Controller libs/controller/main.cpp)
    target_link_libraries(Controller controller)

    add_executable(Container_DataSource libs/workloads/data_source/main.cpp)
    target_link_libraries(Container_DataSource data_source)

    add_executable(Container_Yolov5 libs/workloads/yolov5/yolov5.cpp)
    target_link_libraries(Container_Yolov5 yolov5)

    add_executable(Container_PlateDet libs/workloads/yolov5-plate/yolov5-plate.cpp)
    target_link_libraries(Container_PlateDet yolov5-plate)

    add_executable(convert_onnx2trt libs/utils/onnx2trt/onnx2trt.cpp)
    target_link_libraries(convert_onnx2trt onnx2trt)

    add_executable(Container_RetinaFace libs/workloads/retinaface/retinaface.cpp)
    target_link_libraries(Container_RetinaFace retinaface)

    add_executable(Container_ArcFace libs/workloads/arcface/arcface.cpp)
    target_link_libraries(Container_ArcFace arcface)

    add_executable(Container_MoveNet libs/workloads/movenet/movenet.cpp)
    target_link_libraries(Container_MoveNet movenet)

    add_executable(Container_EmotionNet libs/workloads/emotionnet/emotionnet.cpp)
    target_link_libraries(Container_EmotionNet emotionnet)

    add_executable(Container_Gender libs/workloads/gender/gender.cpp)
    target_link_libraries(Container_Gender gender)

    add_executable(Container_Age libs/workloads/age/age.cpp)
    target_link_libraries(Container_Age age)

    add_executable(Container_CarBrand libs/workloads/carbrand/carbrand.cpp)
    target_link_libraries(Container_CarBrand carbrand)
    
endif()