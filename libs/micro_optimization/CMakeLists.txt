cmake_minimum_required(VERSION 3.18)

project(micro_optimization)

# Only public header files go here.
set(
        HEADER_FILES
        micro_optimization.h
        fcpo_learning.h
        apis_fcrl.h
        bcedge.h
        edgevision.h
)

# All source files and private header files go here.
set(
        SOURCE_FILES
        fcpo_learning.cpp
        apis_fcrl.cpp
        bcedge.cpp
        edgevision.cpp
)

# Here we define the library.
add_library(micro_optimization
        ${HEADER_FILES}
        ${SOURCE_FILES})

# We also add an alias definition so that we shadown
# the export namespace when using add_subdirectory() instead.
add_library(micro_optimization::micro_optimization ALIAS micro_optimization)


# Required compiler features.
# https://cmake.org/cmake/help/v3.1/manual/cmake-compile-features.7.html
target_compile_features(micro_optimization
        # Features required to compile the library itself.
        PRIVATE cxx_std_11)
# By prefixing the installation paths with our name and version
# we can have multiple versions installed at the same time.
set(PROJECT_PREFIX micro_optimization)

find_package(Threads REQUIRED)

set(misc_DIR "${CMAKE_CURRENT_BINARY_DIR}/../misc")
find_package(misc CONFIG REQUIRED)

# Defines the headers for this library.
# https://cmake.org/cmake/help/v3.11/command/target_include_directories.html
target_include_directories(micro_optimization
        PUBLIC
        # Used when building the library:
        $<BUILD_INTERFACE:${micro_optimization_SOURCE_DIR}/>
        # Used when installing the library:
        $<INSTALL_INTERFACE:${PROJECT_PREFIX}>
        ${OpenCV_INCLUDE_DIRS}
        ${PostgreSQL_INCLUDE_DIRS}
        ${LIBPQXX_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../utils/protobufprotocols>
        PRIVATE
        # Used only when building the library:
        src)

target_link_libraries(micro_optimization
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

# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/micro_optimization)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/micro_optimization-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/micro_optimization-config.cmake")

configure_package_config_file(cmake/micro_optimization-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS micro_optimization
        EXPORT micro_optimization-targets
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
install(EXPORT micro_optimization-targets
        FILE micro_optimization-targets.cmake
        NAMESPACE micro_optimization::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT micro_optimization-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/micro_optimization-targets.cmake
        NAMESPACE micro_optimization::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE micro_optimization)

add_executable(RL_OfflineTraining offline_training.cpp)
target_link_libraries(RL_OfflineTraining micro_optimization absl::flags absl::flags_parse)
