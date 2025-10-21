#include "container_agent.h"

class CarDamageAgent : public ContainerAgent {
public:
    CarDamageAgent(const json &configs) : ContainerAgent(configs) {}
};