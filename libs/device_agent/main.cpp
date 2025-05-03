#include "device_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto *agent = new DeviceAgent(absl::GetFlag(FLAGS_controller_url));

    // Start the runBashScript function in a separate thread
    std::string bandwidth_limits = "../jsons/bandwidths/";
    if (absl::GetFlag(FLAGS_dev_bandwidthLimitID) == 0) {
        bandwidth_limits += "simple_bandwidth_limit.json";
    } else {
        bandwidth_limits += "bandwidth_limits" + std::to_string(absl::GetFlag(FLAGS_dev_bandwidthLimitID)) + ".json";
    }
    std::thread scriptThread(&DeviceAgent::limitBandwidth, agent, "../scripts/set_bandwidth.sh", bandwidth_limits);
    scriptThread.detach();

    while (agent->isRunning()) {
        agent->collectRuntimeMetrics();
    }
    delete agent;
    return 0;
}