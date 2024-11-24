#include "receiver.h"

ReceiverConfigs Receiver::loadConfigsFromJson(const json &jsonConfigs) {
    ReceiverConfigs configs;
    return configs;
}

void Receiver::loadConfigs(const json &jsonConfigs, bool isConstructing) {
    spdlog::get("container_agent")->trace("{0:s} is LOANDING configs...", __func__);

    if (!isConstructing) { // If this is not called from the constructor, then we are loading configs from a file for Microservice class
        Microservice::loadConfigs(jsonConfigs);
    }

    ReceiverConfigs configs = loadConfigsFromJson(jsonConfigs);

    if (msvc_RUNMODE == RUNMODE::EMPTY_PROFILING) {
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
    } else if (msvc_RUNMODE == RUNMODE::DEPLOYMENT || msvc_RUNMODE == RUNMODE::PROFILING) {
        grpc::EnableDefaultHealthCheckService(true);
        grpc::reflection::InitProtoReflectionServerBuilderPlugin();
        ServerBuilder builder;
        builder.AddListeningPort(upstreamMicroserviceList.front().link[0], grpc::InsecureServerCredentials());
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        builder.SetMaxMessageSize(1024 * 1024 * 1024);
        builder.SetMaxReceiveMessageSize(1024 * 1024 * 1024);
        builder.RegisterService(&service);
        cq = builder.AddCompletionQueue();
        server = builder.BuildAndStart();
        msvc_OutQueue[0]->setActiveQueueIndex(msvc_activeOutQueueIndex[0]);
    }
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->trace("{0:s} FINISHED loading configs...", __func__);
}

Receiver::Receiver(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name); 
}

template<typename ReqDataType>
void Receiver::processInferTimeReport(Request<ReqDataType> &timeReport) {
    BatchSizeType batchSize = timeReport.req_batchSize;

    BatchSizeType numTimeStamps = (BatchSizeType) (timeReport.req_origGenTime.size() / batchSize);
    for (BatchSizeType i = 0; i < batchSize; i++) {
        msvc_logFile << timeReport.req_travelPath[i] << ",";
        for (BatchSizeType j = 0; j < numTimeStamps - 1; j++) {
            msvc_logFile << timePointToEpochString(timeReport.req_origGenTime[i * numTimeStamps + j]) << ",";
        }
        msvc_logFile << timePointToEpochString(timeReport.req_origGenTime[i * numTimeStamps + numTimeStamps - 1])
                     << "|";

        for (BatchSizeType j = 1; j < numTimeStamps - 1; j++) {
            msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(
                    timeReport.req_origGenTime[i * numTimeStamps + j] -
                    timeReport.req_origGenTime[i * numTimeStamps + j - 1]).count() << ",";
        }
        msvc_logFile << std::chrono::duration_cast<TimePrecisionType>(
                timeReport.req_origGenTime[(i + 1) * numTimeStamps - 1] -
                timeReport.req_origGenTime[(i + 1) * numTimeStamps - 2]).count() << std::endl;
    }
}

Receiver::GpuPointerRequestHandler::GpuPointerRequestHandler(
    DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
    ThreadSafeFixSizedDoubleQueue *out,
    uint64_t &msvc_inReqCount, Receiver *receiver
) : RequestHandler(service, cq, out, msvc_inReqCount, receiver), responder(&ctx) {
    GpuPointerRequestHandler::Proceed();
}

void Receiver::GpuPointerRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestGpuPointerTransfer(&ctx, &request, &responder, cq, cq,
                                           this);
    } else if (status == PROCESS) {
        spdlog::get("container_agent")->trace("GpuPointerRequestHandler::{0:s} is processing request...", __func__);
        if (OutQueue->getActiveQueueIndex() != 2) OutQueue->setActiveQueueIndex(2);
        new GpuPointerRequestHandler(service, cq, OutQueue, msvc_inReqCount, receiverInstance);

        if (request.mutable_elements()->empty()) {
            responder.Finish(reply, Status(grpc::INVALID_ARGUMENT, "No valid data"), this);
            return;
        }

        std::vector<RequestData<LocalGPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto timestamps = std::vector<ClockType>();
            for (auto ts: el.timestamp()) {
                timestamps.push_back(std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(ts)));
            }
            if (validateReq(timestamps[0], el.path())) {
                continue;
            }
            timestamps.push_back(std::chrono::system_clock::now());
            void* data;
            cudaIpcMemHandle_t ipcHandle;
            memcpy(&ipcHandle, el.data().c_str(), sizeof(cudaIpcMemHandle_t));
            cudaError_t cudaStatus = cudaIpcOpenMemHandle(&data, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
            if (cudaStatus != cudaSuccess) {
                std::cerr << "cudaIpcOpenMemHandle failed: " << cudaStatus << std::endl;
                continue;
            }
            auto gpu_image = cv::cuda::GpuMat(el.height(), el.width(), CV_8UC3, data).clone();
            elements = {{{gpu_image.channels(), el.height(), el.width()}, gpu_image}};

            cudaIpcCloseMemHandle(data);

            if (elements.empty()) continue;

            Request<LocalGPUReqDataType> req = {
                    {timestamps},
                    {el.slo()},
                    {el.path()},
                    1,
                    elements
            };
            OutQueue->emplace(req);
            spdlog::get("container_agent")->trace("GpuPointerRequestHandler::{0:s} emplaced request with path: {1:s}", __func__, el.path());
        }

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Receiver::SharedMemoryRequestHandler::SharedMemoryRequestHandler(
    DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
    ThreadSafeFixSizedDoubleQueue *out,
    uint64_t &msvc_inReqCount,
    Receiver *receiver
) : RequestHandler(service, cq, out, msvc_inReqCount, receiver), responder(&ctx) {
    Proceed();
}

void Receiver::SharedMemoryRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSharedMemTransfer(&ctx, &request, &responder, cq, cq,
                                          this);
    } else if (status == PROCESS) {
        spdlog::get("container_agent")->trace("SharedMemoryRequestHandler::{0:s} is processing request...", __func__);
        if (OutQueue->getActiveQueueIndex() != 1) OutQueue->setActiveQueueIndex(1);
        new SharedMemoryRequestHandler(service, cq, OutQueue, msvc_inReqCount, receiverInstance);

        std::vector<RequestData<LocalCPUReqDataType>> elements = {};
        for (const auto &el: *request.mutable_elements()) {
            auto timestamps = std::vector<ClockType>();
            for (auto ts: el.timestamp()) {
                timestamps.emplace_back(std::chrono::time_point<std::chrono::system_clock>(TimePrecisionType(ts)));
            }
            if (validateReq(timestamps[0], el.path())) {
                continue;
            }
            timestamps.push_back(std::chrono::system_clock::now());
            auto name = el.data().c_str();
            boost::interprocess::shared_memory_object shm{open_only, name, read_only};
            boost::interprocess::mapped_region region{shm, read_only};
            auto image = static_cast<cv::Mat *>(region.get_address());
            elements = {{{image->channels(), el.height(), el.width()}, *image}};

            boost::interprocess::shared_memory_object::remove(name);

            Request<LocalCPUReqDataType> req = {
                    {timestamps},
                    {el.slo()},
                    {el.path()},
                    1,
                    elements
            };
            OutQueue->emplace(req);
            spdlog::get("container_agent")->trace("SharedMemoryRequestHandler::{0:s} emplaced request with path: {1:s}", __func__, el.path());
        }

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

Receiver::SerializedDataRequestHandler::SerializedDataRequestHandler(
    DataTransferService::AsyncService *service, ServerCompletionQueue *cq,
    ThreadSafeFixSizedDoubleQueue *out,
    uint64_t &msvc_inReqCount,
    Receiver *receiver
) : RequestHandler(service, cq, out, msvc_inReqCount, receiver), responder(&ctx) {
    Proceed();
}

void Receiver::SerializedDataRequestHandler::Proceed() {
    if (status == CREATE) {
        status = PROCESS;
        service->RequestSerializedDataTransfer(&ctx, &request, &responder, cq, cq,
                                               this);
    } else if (status == PROCESS) {
        spdlog::get("container_agent")->trace("SerializedDataRequestHandler::{0:s} is processing request {1:s}...", __func__, request.mutable_elements()->at(0).path());
        new SerializedDataRequestHandler(service, cq, OutQueue, msvc_inReqCount, receiverInstance);
        std::vector<RequestData<LocalCPUReqDataType>> elements = {};

        for (const auto &el: *request.mutable_elements()) {
            if (!validateReq(ClockType(TimePrecisionType(el.timestamp(0))), el.path())) {
                continue;
            }
            uint length = el.data().length();
            if (length != el.datalen()) {
                spdlog::get("container_agent")->error("SerializedDataRequestHandler::{0:s} data length does not match", __func__);
                continue;
            }
            auto timestamps = std::vector<ClockType>();
            for (auto ts: el.timestamp()) {
                timestamps.emplace_back(TimePrecisionType(ts));
            }
            auto receivedTime = std::chrono::system_clock::now();
            receiverInstance->updateStats(receivedTime);
            timestamps.push_back(receivedTime);
            cv::Mat image;
            if (el.is_encoded()){
                std::vector<uchar> buf(el.data().c_str(), el.data().c_str() + length);
                image = cv::imdecode(buf, cv::IMREAD_COLOR);
            } else {
                image = cv::Mat(el.height(), el.width(), CV_8UC3,const_cast<char *>(el.data().c_str())).clone();
            }
            elements = {{{image.channels(), el.height(), el.width()}, image}};

            Request<LocalCPUReqDataType> req = {
                    {timestamps},
                    {el.slo()},
                    {el.path()},
                    1,
                    elements
            };

            OutQueue->emplace(req);
            spdlog::get("container_agent")->trace("SerializedDataRequestHandler::{0:s} emplaced request with path: {1:s}", __func__, el.path());

            /**
             * @brief Request now should carry 4 timestamps
             * 1. The very moment request is originally generated at the beggining of the pipeline. (FIRST_TIMESTAMP)
             * 2. The moment request is put into outqueue of last immediate upstream processor. (SECOND_TIMESTAMP)
             * 3. The moment request is sent by immediate upstream sender. (THIRD_TIMESTAMP)
             * 4. The moment request is received by the receiver. (FOURTH_TIMESTAMP)
             */
        }

        status = FINISH;
        responder.Finish(reply, Status::OK, this);
    } else {
        GPR_ASSERT(status == FINISH);
        delete this;
    }
}

// This can be run in multiple threads if needed.
void Receiver::HandleRpcs() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    new GpuPointerRequestHandler(&service, cq.get(), msvc_OutQueue[0], msvc_overallTotalReqCount, this);
    new SharedMemoryRequestHandler(&service, cq.get(), msvc_OutQueue[0], msvc_overallTotalReqCount, this);
    new SerializedDataRequestHandler(&service, cq.get(), msvc_OutQueue[0], msvc_overallTotalReqCount, this);
    void *tag;  // uniquely identifies a request.
    bool ok;
    READY = true;
    while (true) {
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                /*void* target;
                auto test = cv::cuda::GpuMat(1, 1, CV_8UC3);
                cudaIpcMemHandle_t ipcHandle;
                cudaIpcGetMemHandle(&ipcHandle, test.data);
                cudaError_t cudaStatus = cudaIpcOpenMemHandle(&target, ipcHandle, cudaIpcMemLazyEnablePeerAccess);
                cudaIpcCloseMemHandle(target);
                test.release();
                if (cudaStatus != cudaSuccess) {
                    std::cout << "cudaIpcOpenMemHandle failed: " << cudaStatus << std::endl;
                    setDevice();
                }*/
                RELOADING = false;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            //spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        GPR_ASSERT(cq->Next(&tag, &ok));
        GPR_ASSERT(ok);
        std::thread thread_([tag]() {static_cast<RequestHandler *>(tag)->Proceed();});
        thread_.detach();
    }
    msvc_logFile.close();
    STOPPED = true;
}