#ifndef PIPEPLUSPLUS_DATA_SOURCE_H
#define PIPEPLUSPLUS_DATA_SOURCE_H

#include <list>
#include "container_agent.h"

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(const json &configs) : ContainerAgent(configs) {}

    void runService(const json &pipeConfigs, const json &configs) override;
};

#endif //PIPEPLUSPLUS_DATA_SOURCE_H
