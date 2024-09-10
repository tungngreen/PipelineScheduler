#include "container_agent.h"
#include <baseprocessor.h>


class MoveNetAgent: public ContainerAgent{
public:
    MoveNetAgent(
        const json& configs
    );
};