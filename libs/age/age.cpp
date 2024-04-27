#include "age.h"

#include <utility>

AgeAgent::AgeAgent(
    const std::string &name,
    uint16_t own_port,
    int8_t devIndex,
    std::string logPath,
    RUNMODE runmode,
    const json &profiling_configs
) : ContainerAgent(name, own_port, devIndex, logPath, runmode, profiling_configs) {
    // for (uint16_t i = 4; i < msvcs.size(); i++) {
    //     msvcs[i]->dispatchThread();
    // }
}

int main(int argc, char **argv) {

    contRunArgs cont_args = loadRunArgs(argc, argv);

    ContainerAgent *agent;
    agent = new AgeAgent(cont_args.cont_name, cont_args.cont_port, cont_args.cont_devIndex, cont_args.cont_logPath, cont_args.cont_runmode, cont_args.cont_profilingConfigs);

    std::vector<Microservice*> msvcsList;

    msvcsList.push_back(new Receiver(cont_args.cont_pipeConfigs[0]));
    msvcsList.push_back(new BaseReqBatcher(cont_args.cont_pipeConfigs[1]));
    msvcsList[1]->SetInQueue(msvcsList[0]->GetOutQueue());
    msvcsList.push_back(new BaseBatchInferencer(cont_args.cont_pipeConfigs[2]));
    msvcsList[2]->SetInQueue(msvcsList[1]->GetOutQueue());
    msvcsList.push_back(new BaseClassifier(cont_args.cont_pipeConfigs[3]));
    msvcsList[3]->SetInQueue(msvcsList[2]->GetOutQueue());
    if (cont_args.cont_runmode == RUNMODE::PROFILING) {
        msvcsList[0]->SetInQueue(msvcsList[3]->GetOutQueue());
    } else {
        for (uint16_t i = 4; i < cont_args.cont_pipeConfigs.size(); i++) {
            if (cont_args.cont_pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::localGPU) {
                msvcsList.push_back(new GPUSender(cont_args.cont_pipeConfigs[i]));
            } else if (cont_args.cont_pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::sharedMemory) {
                msvcsList.push_back(new LocalCPUSender(cont_args.cont_pipeConfigs[i]));
            } else if (cont_args.cont_pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::sharedMemory) {
                msvcsList.push_back(new RemoteCPUSender(cont_args.cont_pipeConfigs[i]));
            }
            msvcsList[i]->SetInQueue(msvcsList[i - 1]->GetOutQueue());
        }
    }

    agent->addMicroservice(msvcsList);
    agent->dispatchMicroservices();

    agent->checkReady();
    
    while (agent->running()) {
        std::this_thread::sleep_for(std::chrono::seconds(4));
        agent->SendState();
    }
    delete agent;
    return 0;
}