#include "container_agent.h"

using namespace spdlog;


class ArcFaceAgent : public ContainerAgent {
public:
    ArcFaceAgent(const json &configs) : ContainerAgent(configs) {}
};

class ArcFaceDataSource : public ContainerAgent {
public:
    ArcFaceDataSource(const json &configs) : ContainerAgent(configs) {}
};