#include "container_agent.h"
#include <baseprocessor.h>

class GenderAgent: public ContainerAgent{
public:
    GenderAgent(
        const json &configs
    );
};