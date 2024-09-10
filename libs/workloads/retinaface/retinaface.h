#include "container_agent.h"
#include <baseprocessor.h>
#include "data_source.h"

class RetinaFaceAgent : public ContainerAgent {
public:
    RetinaFaceAgent(const json &configs): ContainerAgent(configs) {}
};

class RetinaFaceDataSource : public DataSourceAgent {
public:
    RetinaFaceDataSource(const json &configs): DataSourceAgent(configs) {}
};