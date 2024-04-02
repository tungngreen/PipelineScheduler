#include "sender.h"


Sender::Sender(const BaseMicroserviceConfigs &configs) : Microservice(
        configs) {
    stubs = std::vector<std::unique_ptr<DataTransferService::Stub>>();
    stubs.push_back(
            DataTransferService::NewStub(grpc::CreateChannel(configs.msvc_dnstreamMicroservices.front().link[0], grpc::InsecureChannelCredentials())));
    multipleStubs = false;
    READY = true;
}


std::string
Sender::HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> &rpc, CompletionQueue &cq,
                   EmptyMessage &reply, Status &status) {
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

GPUSender::GPUSender(const BaseMicroserviceConfigs &configs) : Sender(configs) {
    tagToGpuPointer = std::map<void *, std::vector<RequestData<LocalGPUReqDataType>> *>();
}

void GPUSender::Process() {
    setDevice();
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop2();
        SendGpuPointer(request.req_data, request.req_origGenTime, request.req_travelPath, request.req_e2eSLOLatency);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::string GPUSender::SendGpuPointer(
        std::vector<RequestData<LocalGPUReqDataType>> &elements,
        const ClockType &timestamp, const std::string &path, const uint32_t &slo) {
    CompletionQueue cq;

    GpuPointerPayload request;
    request.set_timestamp(std::chrono::system_clock::to_time_t(timestamp));
    request.set_path(path);
    request.set_slo(slo);
    for (RequestData<LocalGPUReqDataType> el: elements) {
        auto ref = request.add_elements();
        ref->set_data(&el.data, sizeof(el.data));
        ref->set_width(el.shape[0]);
        ref->set_height(el.shape[1]);
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    auto tag = (void *) (uintptr_t) (rand_int(0, 1000));
    while (tagToGpuPointer.find(tag) != tagToGpuPointer.end()) {
        tag = (void *) (uintptr_t) (rand_int(0, 1000));
    }
    tagToGpuPointer[tag] = &elements;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncGpuPointerTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status, tag);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncGpuPointerTransfer(&context, request, &cq));
    return HandleRpcs(rpc, cq, reply, status, tag);
}

std::string GPUSender::HandleRpcs(std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> &rpc, CompletionQueue &cq,
                                  EmptyMessage &reply, Status &status, void *tag) {
    rpc->Finish(&reply, &status, tag);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(cq.Next(&got_tag, &ok));
    GPR_ASSERT(ok);
    if (status.ok()) {
        if (got_tag == tag) {
            for (RequestData<LocalGPUReqDataType> el: *tagToGpuPointer[tag]) {
                cudaFree(&el.data);
            }
            delete tagToGpuPointer[tag];
            tagToGpuPointer.erase(tag);
        } else {
            return "Complete but Wrong Tag Received";
        }
        return "Complete";
    } else {
        std::cout << status.error_code() << ": " << status.error_message() << std::endl;
        return "RPC failed";
    }
}

LocalCPUSender::LocalCPUSender(const BaseMicroserviceConfigs &configs) : Sender(
        configs) {}

void LocalCPUSender::Process() {
    setDevice();
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop1();
        SendSharedMemory(request.req_data, request.req_origGenTime, request.req_travelPath, request.req_e2eSLOLatency);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::string LocalCPUSender::SendSharedMemory(const std::vector<RequestData<LocalCPUReqDataType>> &elements, const ClockType &timestamp,
                                             const std::string &path,
                                             const uint32_t &slo) {
    CompletionQueue cq;
    SharedMemPayload request;
    request.set_timestamp(std::chrono::system_clock::to_time_t(timestamp));
    request.set_path(path);
    request.set_slo(slo);
    char* name;
    for (RequestData<LocalCPUReqDataType> el: elements) {
        auto ref = request.add_elements();
        sprintf(name, "shared %d", rand_int(0, 1000));
        boost::interprocess::shared_memory_object shm{create_only, name, read_write};
        shm.truncate(el.data.total() * el.data.elemSize());
        boost::interprocess::mapped_region region{shm, read_write};
        std::memcpy(region.get_address(), el.data.data, el.data.total() * el.data.elemSize());
        ref->set_name(name);
        ref->set_width(el.shape[0]);
        ref->set_height(el.shape[1]);
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncSharedMemTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncSharedMemTransfer(&context, request, &cq));
    return HandleRpcs(rpc, cq, reply, status);
}

RemoteCPUSender::RemoteCPUSender(const BaseMicroserviceConfigs &configs) : Sender(
        configs) {}

void RemoteCPUSender::Process() {
    setDevice();
    while (READY) {
        if (this->STOP_THREADS) {
            spdlog::info("{0:s} STOPS.", msvc_name);
            break;
        }
        else if (this->PAUSE_THREADS) {
            ///spdlog::info("{0:s} is being PAUSED.", msvc_name);
            continue;
        }
        auto request = msvc_InQueue[0]->pop1();
        SendSerializedData(request.req_data, request.req_origGenTime, request.req_travelPath, request.req_e2eSLOLatency);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::string RemoteCPUSender::SendSerializedData(
        const std::vector<RequestData<LocalCPUReqDataType>> &elements, const ClockType &timestamp, const std::string &path,
        const uint32_t &slo) { // We use unix time encoded to int64
    CompletionQueue cq;

    SerializedDataPayload request;
    request.set_timestamp(std::chrono::system_clock::to_time_t(timestamp));
    request.set_path(path);
    request.set_slo(slo);
    for (RequestData<LocalCPUReqDataType> el: elements) {
        auto ref = request.add_elements();
        ref->set_data(el.data.data, el.data.total() * el.data.elemSize());
        ref->set_width(el.shape[0]);
        ref->set_height(el.shape[1]);
        ref->set_datalen(el.data.total() * el.data.elemSize());
    }
    EmptyMessage reply;
    ClientContext context;
    Status status;

    if (!multipleStubs) {
        std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
                stubs[0]->AsyncSerializedDataTransfer(&context, request, &cq));
        return HandleRpcs(rpc, cq, reply, status);
    }

    std::unique_ptr<ClientAsyncResponseReader<EmptyMessage>> rpc(
            stubs[rand_int(0, stubs.size() - 1)]->AsyncSerializedDataTransfer(&context, request, &cq));
    return HandleRpcs(rpc, cq, reply, status);
}