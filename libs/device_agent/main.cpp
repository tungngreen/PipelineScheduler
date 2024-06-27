#include "device_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    std::string name = absl::GetFlag(FLAGS_name);
    std::string type = absl::GetFlag(FLAGS_device_type);
    std::string controller_url = absl::GetFlag(FLAGS_controller_url);
    SystemDeviceType deviceType;
    if (type == "server") {
        deviceType = SystemDeviceType::Server;
    } else if (type == "nxavier") {
        deviceType = SystemDeviceType::NXXavier;
    } else if (type == "agxavier") {
        deviceType = SystemDeviceType::AGXXavier;
    } else if (type == "orinano") {
        deviceType = SystemDeviceType::OrinNano;
    }
    else {
        std::cerr << "Invalid device type, use [server, nxavier, agxavier, orinano]" << std::endl;
        exit(1);
    }

    auto *agent = new DeviceAgent(controller_url, name, deviceType);
    while (agent->isRunning()) {
        agent->collectRuntimeMetrics();
    }
    delete agent;
    return 0;
}