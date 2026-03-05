#ifndef SINK_AGENT_H
#define SINK_AGENT_H

#include "device_agent.h"

class SinkAgent: private DeviceAgent {
public:
    SinkAgent(const std::string &controller_url);
};

#endif //SINK_AGENT_H
