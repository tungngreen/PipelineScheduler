#include "container_agent.h"
#include "baseprocessor.h"


class EmotionNetAgent: public ContainerAgent{
public:
    EmotionNetAgent(
        const json& configs
    );
};