#ifndef SINK_AGENT_H
#define SINK_AGENT_H

#include "device_agent.h"

class SinkAgent: public DeviceAgent {
public:
    explicit SinkAgent(const std::string &controller_url);

private:
    void HandleControlRecvRpcs() override;
};

#endif //SINK_AGENT_H
