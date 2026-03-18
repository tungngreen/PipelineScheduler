cmake_minimum_required(VERSION 3.18)

project(misc)

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
        ./misc.h
        ../utils/json.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES
        ./misc.cpp
)

# Here we define the library.
add_library(misc
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(misc::misc ALIAS misc)

set_target_properties(misc PROPERTIES LINKER_LANGUAGE CXX)

# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(misc
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX misc)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(misc
        PUBLIC
            # Used when building the library:
            $<BUILD_INTERFACE:${misc_SOURCE_DIR}/>
            # Used when installing the library:
            $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
            ${OpenCV_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS} 
            ${PostgreSQL_INCLUDE_DIRS}
            ${LIBPQXX_INCLUDE_DIRS}
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
            
        PRIVATE
            # Used only when building the library:
            src)

target_link_libraries(misc
        PUBLIC
        control_mess_grpc_proto
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${PostgreSQL_LIBRARIES}
        ${LIBPQXX_LIBRARIES}
        ${_GRPC_GRPCPP}
)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR  ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/misc)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/misc-config-version.cmake")
set(PROJECT_CONFIG_FILE         "${PROJECT_BINARY_DIR}/misc-config.cmake")

configure_package_config_file(cmake/misc-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS misc
        EXPORT misc-targets
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_ARCHIVE_DIR})

# # The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES ${HEADER_FILES}
        DESTINATION ${INSTALL_INCLUDE_DIR})

# The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES
            ${PROJECT_CONFIG_VERSION_FILE}
            ${PROJECT_CONFIG_FILE}
        DESTINATION ${INSTALL_CONFIG_DIR})

# The INSTALL(EXPORT) command installs a previously defined export group to a script.
install(EXPORT misc-targets
        FILE misc-targets.cmake
        NAMESPACE misc::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT misc-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/misc-targets.cmake
       NAMESPACE misc::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE misc)