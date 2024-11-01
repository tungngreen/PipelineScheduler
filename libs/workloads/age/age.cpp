#include "age.h"

#include <utility>

AgeAgent::AgeAgent(
        const json &configs
) : ContainerAgent(configs) {
    // for (uint16_t i = 4; i < msvcs.size(); i++) {
    //     msvcs[i]->dispatchThread();
    // }
}

int main(int argc, char **argv) {

    json configs = loadRunArgs(argc, argv);

    ContainerAgent *agent;

    json pipeConfigs = configs["container"]["cont_pipeline"];

    agent = new AgeAgent(configs);

    // std::vector<Microservice *> msvcsList;

    // msvcsList.push_back(new Receiver(pipeConfigs[0]));
    // msvcsList.push_back(new BaseReqBatcher(pipeConfigs[1]));
    // msvcsList[1]->SetInQueue(msvcsList[0]->GetOutQueue());
    // msvcsList.push_back(new BaseBatchInferencer(pipeConfigs[2]));
    // msvcsList[2]->SetInQueue(msvcsList[1]->GetOutQueue());
    // msvcsList.push_back(new BaseClassifier(pipeConfigs[3]));
    // msvcsList[3]->SetInQueue(msvcsList[2]->GetOutQueue());
    // // dynamic_cast<BaseBBoxCropper*>(msvcsList[3])->setInferenceShape(dynamic_cast<BaseBatchInferencer*>(msvcsList[2])->getInputShapeVector());
    // if (configs["container"]["cont_RUNMODE"] == RUNMODE::PROFILING) {
    //     msvcsList.push_back(new BaseSink(pipeConfigs[4]));
    //     msvcsList[4]->SetInQueue(msvcsList[3]->GetOutQueue());
    // } else {
    //     for (uint16_t i = 4; i < pipeConfigs.size(); i++) {
    //         if (pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::localGPU) {
    //             spdlog::get("container_agent")->info("Local GPU Sender");
    //             msvcsList.push_back(new GPUSender(pipeConfigs[i]));
    //         } else if (pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::sharedMemory) {
    //             spdlog::get("container_agent")->info("Local CPU Sender");
    //             msvcsList.push_back(new LocalCPUSender(pipeConfigs[i]));
    //         } else if (pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::serialized ||
    //                    pipeConfigs[i].at("msvc_dnstreamMicroservices")[0].at("nb_commMethod") == CommMethod::encodedCPU) {
    //             spdlog::get("container_agent")->info("Remote CPU Sender");
    //             msvcsList.push_back(new RemoteCPUSender(pipeConfigs[i]));
    //         }
    //         std::string name = msvcsList[i]->msvc_name;
    //         msvcsList[i]->SetInQueue({msvcsList[3]->GetOutQueue(i-4)});
    //     }
    // }
    // agent->addMicroservice(msvcsList);

    agent->runService(pipeConfigs, configs);
    delete agent;
    return 0;
}