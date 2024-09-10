#include "container_agent.h"
#include <baseprocessor.h>
#include "data_source.h"

using namespace spdlog;

class YoloV5Agent : public ContainerAgent {
public:
    YoloV5Agent(const json &configs): ContainerAgent(configs) {};
};

class YoloV5DataSource : public DataSourceAgent {
public:
    YoloV5DataSource( const json &configs): DataSourceAgent(configs) {};
};