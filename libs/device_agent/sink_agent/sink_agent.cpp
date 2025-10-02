#include "sink_agent.h"

SinkAgent::SinkAgent(const std::string &ctrl_url) : DeviceAgent() {
    controller_url = ctrl_url;
    controller_ctx = context_t(1);
    std::string server_address = absl::StrFormat("tcp://%s:%d", controller_url, CONTROLLER_MESSAGE_QUEUE_PORT + dev_system_port_offset  - dev_agent_port_offset);
    controller_message_queue = socket_t(controller_ctx, ZMQ_SUB);
    controller_message_queue.setsockopt(ZMQ_SUBSCRIBE, (dev_name + "|").c_str(), dev_name.size() + 1);
    controller_message_queue.connect(server_address);
    controller_message_queue.set(zmq::sockopt::rcvtimeo, 1000);

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