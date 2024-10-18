cmake_minimum_required(VERSION 3.18)

project(container_agent)

# Only public header files go here.
set(
        HEADER_FILES
        container_agent.h
        profiler.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES
        container_agent.cpp
        profiler.cpp
)

# Here we define the library.
add_library(container_agent
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(container_agent::container_agent ALIAS container_agent)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(container_agent
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX container_agent)

find_package(Threads REQUIRED)

set(microservice_DIR "${CMAKE_CURRENT_BINARY_DIR}/../microservice")
find_package(microservice CONFIG REQUIRED)

set(communicator_DIR "${CMAKE_CURRENT_BINARY_DIR}/../communicator")
find_package(communicator CONFIG REQUIRED)

set(controller_DIR "${CMAKE_CURRENT_BINARY_DIR}/../controller")
find_package(controller CONFIG REQUIRED)

set(baseprocessors_DIR "${CMAKE_CURRENT_BINARY_DIR}/../baseprocessors")
find_package(baseprocessors CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(container_agent
        PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${container_agent_SOURCE_DIR}/>
        # Used when installing the library:
        $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
        ${OpenCV_INCLUDE_DIRS}
        ${PostgreSQL_INCLUDE_DIRS}
        ${LIBPQXX_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
        PRIVATE
        # Used only when building the library:
        src)

target_link_libraries(container_agent
        PUBLIC
        device_grpc_proto
        absl::flags
        absl::flags_parse
        ${_GRPC_GRPCPP}
        ${_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
        ${PostgreSQL_LIBRARIES}
        ${LIBPQXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_LIBRARIES}
        CUDA::nvml
        microservice
        controller
        communicator
        baseprocessors
)

# add_executable(ContainerAgent container_agent.cpp)
# target_link_libraries(ContainerAgent container_agent)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/container_agent)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/container_agent-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/container_agent-config.cmake")

configure_package_config_file(cmake/container_agent-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS container_agent
        EXPORT container_agent-targets
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_ARCHIVE_DIR})

# # The INSTALL(FILES) command specifies where to install specific files, such as headers.
#install(FILES ${HEADER_FILES}
#        DESTINATION ${INSTALL_INCLUDE_DIR})

# The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES
        ${PROJECT_CONFIG_VERSION_FILE}
        ${PROJECT_CONFIG_FILE}
        DESTINATION ${INSTALL_CONFIG_DIR})

# The INSTALL(EXPORT) command installs a previously defined export group to a script.
install(EXPORT container_agent-targets
        FILE container_agent-targets.cmake
        NAMESPACE container_agent::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT container_agent-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/container_agent-targets.cmake
        NAMESPACE container_agent::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE container_agent)