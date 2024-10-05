cmake_minimum_required(VERSION 3.18)

project(protobufprotocols)

##################################################################### data_grpc_proto
get_filename_component(data_proto "dataexchange.proto" ABSOLUTE)
get_filename_component(data_proto_path "${data_proto}" PATH)

# Generated sources
set(data_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/dataexchange.pb.cc")
set(data_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/dataexchange.pb.h")
set(data_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/dataexchange.grpc.pb.cc")
set(data_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/dataexchange.grpc.pb.h")
add_custom_command(
        OUTPUT "${data_proto_srcs}" "${data_proto_hdrs}" "${data_grpc_srcs}" "${data_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${data_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${data_proto}"
        DEPENDS "${data_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(data_grpc_proto
        ${data_grpc_srcs}
        ${data_grpc_hdrs}
        ${data_proto_srcs}
        ${data_proto_hdrs})
target_link_libraries(data_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

##################################################################### device_grpc_proto
get_filename_component(device_proto "indevicecommunication.proto" ABSOLUTE)
get_filename_component(device_proto_path "${device_proto}" PATH)

# Generated sources
set(device_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.pb.cc")
set(device_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.pb.h")
set(device_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.grpc.pb.cc")
set(device_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/indevicecommunication.grpc.pb.h")
add_custom_command(
        OUTPUT "${device_proto_srcs}" "${device_proto_hdrs}" "${device_grpc_srcs}" "${device_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${device_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${device_proto}"
        DEPENDS "${device_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(device_grpc_proto
        ${device_grpc_srcs}
        ${device_grpc_hdrs}
        ${device_proto_srcs}
        ${device_proto_hdrs})
target_link_libraries(device_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})


##################################################################### control_comm_grpc_proto
get_filename_component(control_comm_proto "controlcommands.proto" ABSOLUTE)
get_filename_component(control_comm_proto_path "${control_comm_proto}" PATH)

# Generated sources
set(control_comm_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/controlcommands.pb.cc")
set(control_comm_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/controlcommands.pb.h")
set(control_comm_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/controlcommands.grpc.pb.cc")
set(control_comm_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/controlcommands.grpc.pb.h")
add_custom_command(
        OUTPUT "${control_comm_proto_srcs}" "${control_comm_proto_hdrs}" "${control_comm_grpc_srcs}" "${control_comm_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${control_comm_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${control_comm_proto}"
        DEPENDS "${control_comm_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(control_comm_grpc_proto
        ${control_comm_grpc_srcs}
        ${control_comm_grpc_hdrs}
        ${control_comm_proto_srcs}
        ${control_comm_proto_hdrs})
target_link_libraries(control_comm_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})


##################################################################### control_mess_grpc_proto
get_filename_component(control_mess_proto "controlmessages.proto" ABSOLUTE)
get_filename_component(control_mess_proto_path "${control_mess_proto}" PATH)

# Generated sources
set(control_mess_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/controlmessages.pb.cc")
set(control_mess_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/controlmessages.pb.h")
set(control_mess_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/controlmessages.grpc.pb.cc")
set(control_mess_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/controlmessages.grpc.pb.h")
add_custom_command(
        OUTPUT "${control_mess_proto_srcs}" "${control_mess_proto_hdrs}" "${control_mess_grpc_srcs}" "${control_mess_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${control_mess_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${control_mess_proto}"
        DEPENDS "${control_mess_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_library(control_mess_grpc_proto
        ${control_mess_grpc_srcs}
        ${control_mess_grpc_hdrs}
        ${control_mess_proto_srcs}
        ${control_mess_proto_hdrs})
target_link_libraries(control_mess_grpc_proto
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})


set(
        HEADER_FILES
        data_grpc_hdrs
        device_grpc_hdrs
        control_comm_grpc_hdrs
        control_mess_grpc_hdrs
)


# Here we configure variables specifying the target directories.
set(INSTALL_RUNTIME_DIR ${CMAKE_INSTALL_BINDIR})
set(INSTALL_CONFIG_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/cmake)
set(INSTALL_LIBRARY_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE})
set(INSTALL_ARCHIVE_DIR ${CMAKE_INSTALL_LIBDIR}/${PROJECT_PREFIX}/${CMAKE_BUILD_TYPE}/static)
set(INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_PREFIX}/protobufprotocols)

# Define the configuration file locations.
set(PROJECT_CONFIG_VERSION_FILE "${PROJECT_BINARY_DIR}/protobufprotocols-config-version.cmake")
set(PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/protobufprotocols-config.cmake")


configure_package_config_file(cmake/protobufprotocols-config.cmake.in
        ${PROJECT_CONFIG_FILE}
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR})


# The INSTALL(TARGET) command specifies where to install specific targets.
# By specifying EXPORT here, we add these targets to an export group.
install(TARGETS data_grpc_proto device_grpc_proto control_comm_grpc_proto control_mess_grpc_proto
        EXPORT protobufprotocols-targets
        RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR}
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_ARCHIVE_DIR})

# The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES ${HEADER_FILES}
        DESTINATION ${INSTALL_INCLUDE_DIR})

# The INSTALL(FILES) command specifies where to install specific files, such as headers.
install(FILES
        ${PROJECT_CONFIG_VERSION_FILE}
        ${PROJECT_CONFIG_FILE}
        DESTINATION ${INSTALL_CONFIG_DIR})

# The INSTALL(EXPORT) command installs a previously defined export group to a script.
install(EXPORT protobufprotocols-targets
        FILE protobufprotocols-targets.cmake
        NAMESPACE protobufprotocols::
        DESTINATION ${INSTALL_CONFIG_DIR})

export(EXPORT protobufprotocols-targets
        FILE ${CMAKE_CURRENT_BINARY_DIR}/protobufprotocols-targets.cmake
        NAMESPACE protobufprotocols::)

# Register the package in the user's package registry.
# https://cmake.org/cmake/help/v3.11/command/export.html
export(PACKAGE protobufprotocols)