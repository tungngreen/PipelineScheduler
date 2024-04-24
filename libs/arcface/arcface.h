#include <baseprocessor.h>
#include <trtengine.h>
#include <misc.h>
#include "container_agent.h"
#include "receiver.h"
#include "sender.h"

class ArcFaceAgent : public ContainerAgent {
public:
    ArcFaceAgent(
        const std::string &name,
        uint16_t own_port,
        int8_t devIndex,
        std::string logPath,
        std::vector<Microservice*> services
    );
};