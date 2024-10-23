#include "data_source.h"

void DataSourceAgent::runService(const json &pipeConfigs, const json &configs) {
    this->dispatchMicroservices();
    if (configs["container"]["cont_allocationMode"] == 0) {
        while (!cont_msvcsGroups["receiver"].msvcList[0]->checkReady()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / pipeConfigs[0]["msvc_idealBatchSize"].get<int>()));
        }
    } else {
        if (configs["container"]["cont_RUNMODE"] == 0) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        cont_msvcsGroups["receiver"].msvcList[0]->setReady();
    }

    this->waitReady();
    this->START();
    collectRuntimeMetrics();

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    exit(0);
}


void DataSourceAgent::SetStartFrameRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSetStartFrame(&ctx, &request, &responder, cq, cq, this);
    } else if (status == PROCESS) {
        msvcs->front()->SetCurrFrameID(request.value() - 1);
        msvcs->front()->setReady();
        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

void DataSourceAgent::HandleRecvRpcs() {
    auto msvcsList = getAllMicroservices();
    new SetStartFrameRequestHandler(&service, server_cq.get(), &msvcsList);
    ContainerAgent::HandleRecvRpcs();
}
