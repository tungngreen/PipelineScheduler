#include "data_source.h"

int main(int argc, char **argv) {
    json configs = loadRunArgs(argc, argv);
    json pipeConfigs = configs["container"]["cont_pipeline"];
    ContainerAgent *agent = new DataSourceAgent(configs);

    // std::vector<Microservice *> msvcsList;
    // msvcsList.push_back(new DataReader(pipeConfigs[0]));
    // for (uint64_t i = 1; i < pipeConfigs.size(); i++) {
    //     msvcsList.push_back(new RemoteCPUSender(pipeConfigs[i]));
    //     msvcsList[i]->SetInQueue({msvcsList[0]->GetOutQueue(i-1)});
    // }

    // agent->addMicroservice(msvcsList);

    agent->runService(pipeConfigs, configs);
    delete agent;
    return 0;
}
