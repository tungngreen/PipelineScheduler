#include "container_agent.h"

class FireDetAgent : public ContainerAgent {
public:
    FireDetAgent(const json &configs) : ContainerAgent(configs) {}
};