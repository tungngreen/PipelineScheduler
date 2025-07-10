#include "container_agent.h"

class carbrandAgent: public ContainerAgent{
public:
    carbrandAgent(const json &configs) : ContainerAgent(configs) {}
};