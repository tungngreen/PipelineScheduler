#include "container_agent.h"
#include <baseprocessor.h>


class AgeAgent: public ContainerAgent{
public:
    AgeAgent(
        const json &configs
    );
};