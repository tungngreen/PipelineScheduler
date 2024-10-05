#include "device_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto *agent = new DeviceAgent(absl::GetFlag(FLAGS_controller_url));

    // Start the runBashScript function in a separate thread
    std::thread scriptThread(&DeviceAgent::limitBandwidth, agent, "../scripts/set_bandwidth.sh", "../jsons/bandwidth.json");
    scriptThread.detach();

    while (agent->isRunning()) {
        agent->collectRuntimeMetrics();
    }
    delete agent;
    return 0;
}