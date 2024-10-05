cmake_minimum_required(VERSION 3.18)

project(device_agent)

# This include defines standard install locations.
# https://cmake.org/cmake/help/v3.11/module/GNUInstallDirs.html
include(GNUInstallDirs)

# This include provides some helper functions that simplify packaging.
# https://cmake.org/cmake/help/v3.11/module/CMakePackageConfigHelpers.html
include(CMakePackageConfigHelpers)

# If no build type was specified, set it to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release."
            FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# If no installation prefix is given manually, install locally.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING
            "The install location"
            FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Only public header files go here.
set(
        HEADER_FILES
        device_agent.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES
        device_agent.cpp
)

# Here we define the library.
add_library(device_agent
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(device_agent::device_agent ALIAS device_agent)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(device_agent
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX device_agent)

set(microservice_DIR "${CMAKE_CURRENT_BINARY_DIR}/../microservice")
find_package(microservice CONFIG REQUIRED)

set(container_agent_DIR "${CMAKE_CURRENT_BINARY_DIR}/../container_agent")
find_package(container_agent CONFIG REQUIRED)

set(controller_DIR "${CMAKE_CURRENT_BINARY_DIR}/../controller")
find_package(controller CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(device_agent
        PUBLIC
            # Used when building the library:
            $<BUILD_INTERFACE:${device_agent_SOURCE_DIR}/>
            # Used when installing the library:
            $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
        PRIVATE
            # Used only when building the library:
            src)

target_link_libraries(device_agent
        PUBLIC
        absl::flags
        absl::flags_parse
        ${CMAKE_THREAD_LIBS_INIT}
        microservice
        container_agent
        controller
)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR  ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/device_agent)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/device_agent-config-version.cmake")
set(PROJECT_CONFIG_FILE         "${PROJECT_BINARY_DIR}/device_agent-config.cmake")

configure_package_config_file(cmake/device_agent-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS device_agent
        EXPORT device_agent-targets
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
install(EXPORT device_agent-targets
        FILE device_agent-targets.cmake
        NAMESPACE device_agent::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT device_agent-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/device_agent-targets.cmake
       NAMESPACE device_agent::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE device_agent)