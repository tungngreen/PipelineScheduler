cmake_minimum_required(VERSION 3.18)

project(trtengine)

# This include defines standard install locations.
# https://cmake.org/cmake/help/v3.11/module/GNUInstallDirs.html
include(GNUInstallDirs)

# This include provides some helper functions that simplify packaging.
# https://cmake.org/cmake/help/v3.11/module/CMakePackageConfigHelpers.html
include(CMakePackageConfigHelpers)

# Only public header files go here.
set(
        HEADER_FILES 
        ./trtengine.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES 
        ./trtengine.cpp
)

# Here we define the library.
add_library(trtengine SHARED
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(trtengine::trtengine ALIAS trtengine)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(trtengine
        # Features required to compile the library itself.
        PRIVATE cxx_std_14)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX trtengine)

set(misc_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../misc")
find_package(misc CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(trtengine
        PUBLIC
            # Used when building the library:
            $<BUILD_INTERFACE:${trtengine_SOURCE_DIR}/>
            # Used when installing the library:
            $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
            ${OpenCV_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${TensorRT_INCLUDE_DIRS}
        PRIVATE
            # Used only when building the library:
            src)

message(STATUS "TensorRT_LIBRARIES: ${TensorRT_LIBRARIES}")
target_link_libraries(trtengine
        PUBLIC
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${TensorRT_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        misc
)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR  ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/trtengine)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/trtengine-config-version.cmake")
set(PROJECT_CONFIG_FILE         "${PROJECT_BINARY_DIR}/trtengine-config.cmake")

configure_package_config_file(cmake/trtengine-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS trtengine
        EXPORT trtengine-targets
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
install(EXPORT trtengine-targets
        FILE trtengine-targets.cmake
        NAMESPACE trtengine::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT trtengine-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/trtengine-targets.cmake
       NAMESPACE trtengine::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE trtengine)