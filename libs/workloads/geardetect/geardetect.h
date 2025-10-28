#include "container_agent.h"

class GearDetAgent : public ContainerAgent {
public:
    GearDetAgent(const json &configs) : ContainerAgent(configs) {}
};