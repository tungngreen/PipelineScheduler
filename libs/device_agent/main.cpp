#include "device_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto *agent = new DeviceAgent(absl::GetFlag(FLAGS_controller_url));

    std::thread scriptThread(&DeviceAgent::limitBandwidth, agent, "../scripts/set_bandwidth.sh",
                             absl::GetFlag(FLAGS_dev_networkInterface));
    scriptThread.detach();

    while (agent->isRunning()) {
        agent->collectRuntimeMetrics();
    }
    delete agent;
    return 0;
}