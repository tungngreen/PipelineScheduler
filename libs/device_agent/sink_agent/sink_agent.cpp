#include "sink_agent.h"

SinkAgent::SinkAgent(const std::string &controller_url) : DeviceAgent() {
    dev_logPath += "/sink_agent";
    std::filesystem::create_directories(
            std::filesystem::path(dev_logPath)
    );

    setupLogger(
            dev_logPath,
            "device_agent",
            dev_loggingMode,
            dev_verbose,
            dev_loggerSinks,
            dev_logger
    );

    running = true;
    threads = std::vector<std::thread>();
    threads.emplace_back(&DeviceAgent::HandleControlCommands, this);
    for (auto &thread: threads) {
        thread.detach();
    }
}