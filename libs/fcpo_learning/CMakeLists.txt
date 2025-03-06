cmake_minimum_required(VERSION 3.18)

project(fcpo_learning)

# Only public header files go here.
set(
        HEADER_FILES
        fcpo_learning.h
        bcedge.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES
        fcpo_learning.cpp
        bcedge.cpp
)

# Here we define the library.
add_library(fcpo_learning
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(fcpo_learning::fcpo_learning ALIAS fcpo_learning)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(fcpo_learning
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX fcpo_learning)

find_package(Threads REQUIRED)

set(misc_DIR "${CMAKE_CURRENT_BINARY_DIR}/../misc")
find_package(misc CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(fcpo_learning
        PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${fcpo_learning_SOURCE_DIR}/>
        # Used when installing the library:
        $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
        ${OpenCV_INCLUDE_DIRS}
        ${PostgreSQL_INCLUDE_DIRS}
        ${LIBPQXX_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
        PRIVATE
        # Used only when building the library:
        src)

target_link_libraries(fcpo_learning
        PUBLIC
        control_comm_grpc_proto
        device_comm_grpc_proto
        device_mess_grpc_proto
        ${_GRPC_GRPCPP}
        ${_REFLECTION}
        ${_PROTOBUF_LIBPROTOBUF}
        ${PostgreSQL_LIBRARIES}
        ${LIBPQXX_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES}
        misc
)

# add_executable(ContainerAgent fcpo_learning.cpp)
# target_link_libraries(ContainerAgent fcpo_learning)

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/fcpo_learning)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/fcpo_learning-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/fcpo_learning-config.cmake")

configure_package_config_file(cmake/fcpo_learning-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS fcpo_learning
        EXPORT fcpo_learning-targets
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
install(EXPORT fcpo_learning-targets
        FILE fcpo_learning-targets.cmake
        NAMESPACE fcpo_learning::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT fcpo_learning-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/fcpo_learning-targets.cmake
        NAMESPACE fcpo_learning::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE fcpo_learning)