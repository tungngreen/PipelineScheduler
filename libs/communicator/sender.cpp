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

    //stubs = std::vector<std::unique_ptr<DataTransferService::Stub>>();
    stubs = StubVector();
    for (auto &link : dnstreamMicroserviceList.front().link) {
        stubs.emplace_back(comm_ctx, link, (int) ((msvc_pipelineSLO - msvc_contSLO) / 1000));
    }

    if (stubs.size() > 1) {
        multipleStubs = true;
        if (dnstreamMicroserviceList.front().portions.size() == 0){
            dnstreamMicroserviceList.front().portions.push_back(1.0f);
        }
    } else {
        multipleStubs = false;
    }

    READY = true;
}

void Sender::reloadDnstreams() {
    READY = false;
    Microservice::reloadDnstreams();
    stubs = StubVector();
    for (auto &link: dnstreamMicroserviceList.front().link) {
        stubs.emplace_back(comm_ctx, link, (int) ((msvc_pipelineSLO - msvc_contSLO) / 1000));
    }
    if (stubs.size() > 1) {
        multipleStubs = true;
        if (dnstreamMicroserviceList.front().portions.size() == 0){
            dnstreamMicroserviceList.front().portions.push_back(1.0f);
        }
    } else {
        multipleStubs = false;
    }
    READY = true;
}

Sender::Sender(const json &jsonConfigs) : Microservice(jsonConfigs) {
    comm_ctx = context_t(1);
    loadConfigs(jsonConfigs, true);
    msvc_toReloadConfigs = false;
    spdlog::get("container_agent")->info("{0:s} is created.", msvc_name);
}

void Sender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    while (true) {
        if (STOP_THREADS) {
            spdlog::get("container_agent")->info("{0:s} STOPS.", msvc_name);
            break;
        } else if (PAUSE_THREADS) {
            if (RELOADING) {
                spdlog::get("container_agent")->trace("{0:s} is BEING (re)loaded...", msvc_name);
                RELOADING = false;
                spdlog::get("container_agent")->info("{0:s} is (RE)LOADED.", msvc_name);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto request = msvc_InQueue[0]->pop1(msvc_name);
        // Meaning the the timeout in pop() has been reached and no request was actually popped
        // before comparing, check if the re_travelPath size = 0, if so, continue
        if (request.req_travelPath.empty()) continue;
        if (strcmp(request.req_travelPath[0].c_str(), "empty") == 0) continue;
            /**
             * @brief ONLY IN PROFILING MODE
             * Check if the profiling is to be stopped, if true, then send a signal to the downstream microservice to stop profiling
             */
        if (strcmp(request.req_travelPath[0].c_str(), "STOP_PROFILING") == 0) {
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

std::string Sender::completeSending(message_t &zmq_msg) {
    socket_t *sock;
    if (!multipleStubs) {
        sock = stubs.first();
    } else if (dnstreamMicroserviceList.front().portions[0] == 1.0f) {
        sock = stubs.random();
    } else {
        sock = stubs.next();
    }

    // Check if peer is available immediately via polling
    zmq::pollitem_t items[] = { { sock->handle(), 0, ZMQ_POLLOUT, 0 } };
    zmq::poll(&items[0], 1, std::chrono::milliseconds(0));
    if (!(items[0].revents & ZMQ_POLLOUT)) {
        spdlog::get("container_agent")->error("{0:s} can not send to {1:s}, because Receiver is unavailable.", msvc_name, sock->get(zmq::sockopt::last_endpoint));
        return "RPC failed";
    }

    if (!sock->send(zmq_msg, send_flags::none)) {
        spdlog::get("container_agent")->error("{0:s} failed to send data to {1:s}, because of Send Timeout ({2:d}).", msvc_name, sock->get(zmq::sockopt::last_endpoint), sock->get(zmq::sockopt::sndtimeo));
        return "RPC failed";
    }

    message_t reply;
    if (!sock->recv(reply, recv_flags::none)) {
        spdlog::get("container_agent")->error("{0:s} failed to send data to {1:s}, because of Recv Timeout ({2:d}).", msvc_name, sock->get(zmq::sockopt::last_endpoint), sock->get(zmq::sockopt::rcvtimeo));
        return "RPC failed";
    }
    return reply.to_string();
}

GPUSender::GPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "GPU");
    spdlog::get("container_agent")->trace("{0:s} GPUSender is created.", msvc_name);
}

void GPUSender::Process() {
    msvc_logFile.open(msvc_microserviceLogPath, std::ios::out);
    while (true) {
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
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto request = msvc_InQueue[0]->pop2(msvc_name);
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

void GPUSender::SendData(std::vector<RequestData<LocalGPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
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
        return;
    }
    message_t zmq_msg(std::to_string(pipelinescheduler::GPU) + " " + request.SerializeAsString());
    completeSending(zmq_msg);
}

LocalCPUSender::LocalCPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "LocalCPU");
    spdlog::get("container_agent")->trace("{0:s} LocalCPUSender is created.", msvc_name);
}

void LocalCPUSender::SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
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
    if (request.elements_size() == 0) {
        return;
    }
    message_t zmq_msg(std::to_string(pipelinescheduler::MEMORY) + " " + request.SerializeAsString());
    completeSending(zmq_msg);
}

RemoteCPUSender::RemoteCPUSender(const json &jsonConfigs) : Sender(jsonConfigs) {
    addToName("sender", "RemoteCPU");
    spdlog::get("container_agent")->trace("{0:s} RemoteCPUSender is created.", msvc_name);
}

void RemoteCPUSender::SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
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

    if (request.elements_size() == 0) {
        return;
    }
    message_t zmq_msg(std::to_string(pipelinescheduler::REMOTE) + " " + request.SerializeAsString());
    completeSending(zmq_msg);
}