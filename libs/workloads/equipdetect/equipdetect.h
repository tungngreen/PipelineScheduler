#include "container_agent.h"

class EquipDetAgent : public ContainerAgent {
public:
    EquipDetAgent(const json &configs) : ContainerAgent(configs) {}
};