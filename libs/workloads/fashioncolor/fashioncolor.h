#include "container_agent.h"

class FashionColorAgent: public ContainerAgent{
public:
    FashionColorAgent(const json &configs) : ContainerAgent(configs) {}
};