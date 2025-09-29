cmake_minimum_required(VERSION 3.18)

project(local_optimization)

# Only public header files go here.
set(
        HEADER_FILES
        local_optimization.h
        fcpo_learning.h
        bcedge.h
        edgevision.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES
        fcpo_learning.cpp
        bcedge.cpp
        edgevision.cpp
)

# Here we define the library.
add_library(local_optimization
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(local_optimization::local_optimization ALIAS local_optimization)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(local_optimization
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX local_optimization)

find_package(Threads REQUIRED)

set(misc_DIR "${CMAKE_CURRENT_BINARY_DIR}/../misc")
find_package(misc CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(local_optimization
        PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${local_optimization_SOURCE_DIR}/>
        # Used when installing the library:
        $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
        ${OpenCV_INCLUDE_DIRS}
        ${PostgreSQL_INCLUDE_DIRS}
        ${LIBPQXX_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
        PRIVATE
        # Used only when building the library:
        src)

target_link_libraries(local_optimization
        PUBLIC
        cppzmq
        control_mess_grpc_proto
        device_mess_grpc_proto
        ${_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
        ${PostgreSQL_LIBRARIES}
        ${LIBPQXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        misc
)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/local_optimization)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/local_optimization-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/local_optimization-config.cmake")

configure_package_config_file(cmake/local_optimization-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS local_optimization
        EXPORT local_optimization-targets
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
install(EXPORT local_optimization-targets
        FILE local_optimization-targets.cmake
        NAMESPACE local_optimization::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT local_optimization-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/local_optimization-targets.cmake
        NAMESPACE local_optimization::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE local_optimization)