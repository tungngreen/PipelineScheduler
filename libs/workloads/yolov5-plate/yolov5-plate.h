#include "container_agent.h"

class YoloV5Agent : public ContainerAgent {
public:
    YoloV5Agent(const json &configs) : ContainerAgent(configs) {}
};