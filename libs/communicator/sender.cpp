#include "sender.h"

SenderConfigs Sender::loadConfigsFromJson(const json &jsonConfigs) {
    SenderConfigs configs;
    configs.msvc_name = jsonConfigs["msvc_name"];
    return configs;
}

void Sender::loadConfigs(const json &jsonConfigs, bool isConstructing) {

    if (!isConstructing) { //If this is not called from the constructor, we need to load the configs for Sender's base, Micrsoservice class
        Microservice::loadConfigs(jsonConfigs, isConstructing);
    }

    SenderConfigs configs = loadConfigsFromJson(jsonConfigs);

    stubs = std::vector<std::unique_ptr<DataTransferService::Stub>>();
    for (auto &link: dnstreamMicroserviceList.front().link) {
        stubs.push_back(
                DataTransferService::NewStub(
                        grpc::CreateChannel(link, grpc::InsecureChannelCredentials())));
    }
    if (stubs.size() > 1) {
        multipleStubs = true;
    } else {
        multipleStubs = false;
    }
    READY = true;
}

Sender::Sender(const json &jsonConfigs) : Microservice(jsonConfigs) {
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name);
}

void Sender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    while (READY) {
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                RELOADING = false;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop1();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(request.req_travelPath[0].c_str(), "empty") == 0) {
            continue;
            /**
             * @brief ONLY IN PROFILING MODE
             * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
             */
        } else if (strcmp(request.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(request);
            continue;
        }
        /**
         * @brief An outgoing request should contain exactly 3 timestamps:
         * 1. The time when the request was generated at the very beginning of the pipeline, this timestamp is always at the front.
         * 2. The time when the request was putin the out queue of the previous microservice, which is either a postprocessor (regular container) or a data reader (data source).
         * 3. The time this request is sent, which is right about now().
         */
        auto t = std::chrono::system_clock::now();
        for (auto &ts: request.req_origGenTime) {
            ts.emplace_back(t);
        }

        SendData(
                request.req_data,
                request.req_origGenTime,
                request.req_travelPath,
                request.req_e2eSLOLatency
        );
    }
    msvc_logFile.close();
    STOPPED = true;
}

GPUSender::GPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "GPU");
    spdlog::get("container_agent")->trace("{0:s} GPUSender is created.", msvc_name);
}

void GPUSender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    while (READY) {
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                setDevice();
                RELOADING = false;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop2();
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        if (strcmp(request.req_travelPath[0].c_str(), "empty") == 0) {
            continue;

        /**
         * @brief ONLY IN PROFILING MODE
         * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
         */
        } else if (strcmp(request.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
            STOP_THREADS = true;
            msvc_OutQueue[0]->emplace(request);
            continue;
        }
        /**
         * @brief An outgoing request should contain exactly 3 timestamps:
         * 1. The time when the request was generated at the very beginning of the pipeline, this timestamp is always at the front.
         * 2. The time when the request was putin the out queue of the previous microservice, which is either a postprocessor (regular container) or a data reader (data source).
         * 3. The time this request is sent, which is right about now().
         */
        auto t = std::chrono::system_clock::now();
        for (auto &ts: request.req_origGenTime) {
            ts.emplace_back(t);
        }

        SendData(
                request.req_data,
                request.req_origGenTime,
                request.req_travelPath,
                request.req_e2eSLOLatency
        );
    }
    msvc_logFile.close();
    STOPPED = true;
}

std::string GPUSender::SendData(std::vector<RequestData<LocalGPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                                std::vector<std::string> &path, RequestSLOType &slo) {
    CompletionQueue cq;

    ImageDataPayload request;
    for (unsigned int i = 0; i < elements.size(); i++) {
        cudaIpcMemHandle_t ipcHandle;
        char *serializedData[sizeof(cudaIpcMemHandle_t)];
        cudaError_t cudaStatus = cudaIpcGetMemHandle(&ipcHandle, elements[i].data.template ptr<uchar>());
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaIpcGetMemHandle failed: " << cudaStatus << std::endl;
            continue;
        }
        memcpy(&serializedData, &ipcHandle, sizeof(cudaIpcMemHandle_t));

        ImageData* ref = request.add_elements();
        ref->set_is_encoded(false);
        ref->set_data(serializedData, sizeof(cudaIpcMemHandle_t));
        ref->set_height(elements[i].shape[1]);
        ref->set_width(elements[i].shape[2]);
        for (auto ts: timestamp[i]) {
            ref->add_timestamp(std::chrono::duration_cast<TimePrecisionType>(ts.time_since_epoch()).count());
        }
        ref->set_path(path[i]);
        ref->set_slo(slo[i]);
    }

    if (request.elements_size() == 0) {
        return "No elements to send";
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;


    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncGpuPointerTransfer(&context, request, &cq));
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(cq.Next(&got_tag, &ok));
        GPR_ASSERT(ok);

        if (status.ok()) {
            return "Complete";
        } else {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return "RPC failed";
        };
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncGpuPointerTransfer(&context, request, &cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(ok);

    if (status.ok()) {
        return "Complete";
    } else {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "RPC failed";
    }
}

LocalCPUSender::LocalCPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "LocalCPU");
    spdlog::get("container_agent")->trace("{0:s} LocalCPUSender is created.", msvc_name);
}

std::string LocalCPUSender::SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                                     std::vector<std::string> &path, RequestSLOType &slo) {
    CompletionQueue cq;
    ImageDataPayload request;
    char *name;
    for (unsigned int i = 0; i < elements.size(); i++) {
        sprintf(name, "shared %d", rand_int(0, 1000));
        boost::interprocess::shared_memory_object shm{create_only, name, read_write};
        shm.truncate(elements[i].data.total() * elements[i].data.elemSize());
        boost::interprocess::mapped_region region{shm, read_write};
        std::memcpy(region.get_address(), elements[i].data.data, elements[i].data.total() * elements[i].data.elemSize());

        auto ref = request.add_elements();
        ref->set_is_encoded(false);
        ref->set_data(name);
        ref->set_height(elements[i].shape[1]);
        ref->set_width(elements[i].shape[2]);
        for (auto ts: timestamp[i]) {
            ref->add_timestamp(std::chrono::duration_cast<TimePrecisionType>(ts.time_since_epoch()).count());
        }
        ref->set_path(path[i]);
        ref->set_slo(slo[i]);
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncSharedMemTransfer(&context, request, &cq));
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(cq.Next(&got_tag, &ok));
        GPR_ASSERT(ok);

        if (status.ok()) {
            return "Complete";
        } else {
            std::cout << status.error_code() << ": " << status.error_message() << std::endl;
            return "RPC failed";
        }
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncSharedMemTransfer(&context, request, &cq));
    rpc->Finish(&reply, &status, (void *) 1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(ok);

    if (status.ok()) {
        return "Complete";
    } else {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "RPC failed";
    }
}

RemoteCPUSender::RemoteCPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "RemoteCPU");
    spdlog::get("container_agent")->trace("{0:s} RemoteCPUSender is created.", msvc_name);
}

std::string RemoteCPUSender::SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                                     std::vector<std::string> &path, RequestSLOType &slo) {
    CompletionQueue cq;
    ImageDataPayload request;
    for (unsigned int i = 0; i < elements.size(); i++) {
        auto ref = request.add_elements();
        ref->set_is_encoded(msvc_InQueue.front()->getEncoded());
        ref->set_data(elements[i].data.data, elements[i].data.total() * elements[i].data.elemSize());

        //Metadata meta;
        ref->set_height(elements[i].shape[1]);
        ref->set_width(elements[i].shape[2]);
        for (auto ts: timestamp[i]) {
            ref->add_timestamp(std::chrono::duration_cast<TimePrecisionType>(ts.time_since_epoch()).count());
        }
        ref->set_path(path[i]);
        ref->set_slo(slo[i]);
        ref->set_datalen(elements[i].data.total() * elements[i].data.elemSize());
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncSerializedDataTransfer(&context, request, &cq));
        rpc->Finish(&reply, &status, (void *) 1);
        void *got_tag;
        bool ok = false;
        GPR_ASSERT(cq.Next(&got_tag, &ok));
        GPR_ASSERT(ok);

        if (status.ok()) {
            return "Complete";
        } else {
            spdlog::get("container_agent")->error("{0:s} error {1:d}: {2:s}", msvc_name, status.error_code(), status.error_message());
            return "RPC failed";
        }
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncSerializedDataTransfer(&context, request, &cq));
    rpc->Finish(&reply, &status, (void *) 1);
//    void *got_tag;
//    bool ok = false;
//    GPR_ASSERT(cq.Next(&got_tag, &ok));
//    GPR_ASSERT(ok);
//
//    if (status.ok()) {
//        return "Complete";
//    } else {
//        spdlog::get("container_agent")->error("{0:s} error {1:d}: {2:s}", msvc_name, status.error_code(), status.error_message());
//        return "RPC failed";
//    }
    return "Complete";
}