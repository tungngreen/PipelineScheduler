#include "device_agent.h"

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);
    auto *agent = new DeviceAgent();
    agent->SelfReady();
    while (agent->isRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    delete agent;
    return 0;
}