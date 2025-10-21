#include "container_agent.h"

class CarColorAgent: public ContainerAgent{
public:
    CarColorAgent(const json &configs) : ContainerAgent(configs) {}
};