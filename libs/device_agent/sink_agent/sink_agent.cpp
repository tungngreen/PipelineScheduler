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
    threads.emplace_back(&SinkAgent::HandleControlRecvRpcs, this);
    for (auto &thread: threads) {
        thread.detach();
    }
}

void SinkAgent::HandleControlRecvRpcs() {
    new StartContainerRequestHandler(&controller_service, controller_cq.get(), this);
    new StopContainerRequestHandler(&controller_service, controller_cq.get(), this);
    void *tag;
    bool ok;
    while (running) {
        if (!controller_cq->Next(&tag, &ok)) {
            break;
        }
        GPR_ASSERT(ok);
        static_cast<RequestHandler *>(tag)->Proceed();
    }
}