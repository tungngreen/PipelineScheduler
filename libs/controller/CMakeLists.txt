cmake_minimum_required(VERSION 3.18)

project(controller)

# This include defines standard install locations.
# https://cmake.org/cmake/help/v3.11/module/GNUInstallDirs.html
include(GNUInstallDirs)

# This include provides some helper functions that simplify packaging.
# https://cmake.org/cmake/help/v3.11/module/CMakePackageConfigHelpers.html
include(CMakePackageConfigHelpers)

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

message ("Building controller for system: ${SYSTEM_NAME}")
if(SYSTEM_NAME STREQUAL "PPP")
# Only public header files go here.
        set(
                HEADER_FILES
                controller.h
                scheduling-ppp.h
        )

        # All source files and private header files go here.
        set(
                SOURCE_FILES
                controller.cpp
                scheduling-ppp.cpp
        )
elseif (SYSTEM_NAME STREQUAL "JLF")
        set(
                HEADER_FILES
                controller.h
                scheduling-jlf.h
        )

        # All source files and private header files go here.
        set(
                SOURCE_FILES
                controller.cpp
                scheduling-jlf.cpp
        )
elseif (SYSTEM_NAME STREQUAL "DIS")
        set(
                HEADER_FILES
                controller.h
                scheduling-dis.h
        )

        # All source files and private header files go here.
        set(
                SOURCE_FILES
                controller.cpp
                scheduling-dis.cpp
        )
elseif (SYSTEM_NAME STREQUAL "RIM")
        set(
                HEADER_FILES
                controller.h
                scheduling-rim.h
        )

        # All source files and private header files go here.
        set(
                SOURCE_FILES
                controller.cpp
                scheduling-rim.cpp
        )
endif()

# Here we define the library.
add_library(controller
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(controller::controller ALIAS controller)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(controller
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX controller)

set(microservice_DIR "${CMAKE_CURRENT_BINARY_DIR}/../microservice")
find_package(microservice CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(controller
        PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${controller_SOURCE_DIR}/>
        # Used when installing the library:
        $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
        ${OpenCV_INCLUDE_DIRS}
        # ${LIGHTGBM_INCLUDE_DIR}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
        PRIVATE
        # Used only when building the library:
        src)

target_link_libraries(controller
        PUBLIC
        control_comm_grpc_proto
        control_mess_grpc_proto
        absl::flags
        absl::flags_parse
        ${_GRPC_GRPCPP}
        ${_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
        ${CMAKE_THREAD_LIBS_INIT}
        microservice
        # ${LIGHTGBM_LIBRARY}
)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/controller)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/controller-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/controller-config.cmake")

configure_package_config_file(cmake/controller-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS controller
        EXPORT controller-targets
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
install(EXPORT controller-targets
        FILE controller-targets.cmake
        NAMESPACE controller::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT controller-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/controller-targets.cmake
        NAMESPACE controller::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE controller)