#include "retinaface.h"

int main(int argc, char **argv) {

    json configs = loadRunArgs(argc, argv);

    ContainerAgent *agent;

    json pipeConfigs = configs["container"]["cont_pipeline"];

    if (pipeConfigs[0].at("msvc_type") == MicroserviceType::DataReader) {
        agent = new RetinaFaceDataSource(configs);
    } else {
        agent = new RetinaFaceAgent(configs);
    }

    agent->runService(pipeConfigs, configs);
    delete agent;
    return 0;
}