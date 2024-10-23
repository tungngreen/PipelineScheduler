#ifndef PIPEPLUSPLUS_DATA_SOURCE_H
#define PIPEPLUSPLUS_DATA_SOURCE_H

#include <list>
#include "container_agent.h"

class DataSourceAgent : public ContainerAgent {
public:
    DataSourceAgent(const json &configs) : ContainerAgent(configs) {}

    void runService(const json &pipeConfigs, const json &configs) override;

    class SetStartFrameRequestHandler : public RequestHandler {
    public:
        SetStartFrameRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
                                    std::vector<Microservice*> *msvcs)
                : RequestHandler(service, cq), msvcs(msvcs) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommands::Int32 request;
        std::vector<Microservice*> *msvcs;
    };

    void HandleRecvRpcs() override;
};

#endif //PIPEPLUSPLUS_DATA_SOURCE_H
