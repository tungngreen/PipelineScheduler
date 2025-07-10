#include "yolov5.h"

int main(int argc, char **argv) {

    json configs = loadRunArgs(argc, argv);

    ContainerAgent *agent;

    json pipeConfigs = configs["container"]["cont_pipeline"];

    if (pipeConfigs[0].at("msvc_type") == MicroserviceType::DataReader) {
        agent = new YoloV5DataSource(configs);
    } else {
        agent = new YoloV5Agent(configs);
    }

    agent->runService(pipeConfigs, configs);
    delete agent;
    return 0;
}