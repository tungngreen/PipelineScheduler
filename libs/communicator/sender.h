#ifndef PIPEPLUSPLUS_SENDER_H
#define PIPEPLUSPLUS_SENDER_H

#include "communicator.h"

using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;
using boost::interprocess::read_write;
using boost::interprocess::create_only;
using json = nlohmann::ordered_json;


struct SenderConfigs : BaseMicroserviceConfigs {
    // Empty for now
    uint8_t dummy;
};

class Sender : public Microservice {
public:
    Sender(const json &jsonConfigs);

    void Process();

    SenderConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

    virtual std::string SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) = 0;

protected:
    static inline std::mt19937 &generator() {
        // the generator will only be seeded once (per thread) since it's static
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }

    static int rand_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator());
    }

    void addToName(const std::string substring, const std::string strToAdd) {
        msvc_name.replace(msvc_name.find(substring), substring.length(), strToAdd + substring);
        msvc_microserviceLogPath.replace(msvc_microserviceLogPath.find(substring), substring.length(), strToAdd + substring);
    }

    std::vector<std::unique_ptr<DataTransferService::Stub>> stubs;
    bool multipleStubs;
    std::atomic<bool> run{};
};

class GPUSender : public Sender {
public:
    explicit GPUSender(const json &jsonConfigs);

    ~GPUSender() {
        waitStop();
        spdlog::get("container_agent")->info("{0:s} has stopped", msvc_name);
    }

    void Process();

    void dispatchThread() {
        std::thread sender(&GPUSender::Process, this);
        sender.detach();
    }

    std::string SendData(std::vector<RequestData<LocalGPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo);

    std::string SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) final {return "";};

private:

    void serializeIpcMemHandle(const cudaIpcMemHandle_t& handle, char* buffer) {
        memcpy(buffer, &handle, sizeof(cudaIpcMemHandle_t));
    }
};

class LocalCPUSender : public Sender {
public:
    LocalCPUSender(const json &jsonConfigs);

    ~LocalCPUSender() {
        waitStop();
        spdlog::get("container_agent")->info("{0:s} has stopped", msvc_name);
    }

    void dispatchThread() final {
        std::thread sender(&LocalCPUSender::Process, this);
        sender.detach();
    }

    std::string SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) final;
};

class RemoteCPUSender : public Sender {
public:
    RemoteCPUSender(const json &jsonConfigs);

    ~RemoteCPUSender() {
        waitStop();
        spdlog::get("container_agent")->info("{0:s} has stopped", msvc_name);
    }

    void dispatchThread() final {
        std::thread sender(&RemoteCPUSender::Process, this);
        sender.detach();
    }

    std::string SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) final;
};

#endif //PIPEPLUSPLUS_SENDER_H
