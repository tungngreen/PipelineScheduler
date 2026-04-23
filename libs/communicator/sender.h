#ifndef PIPEPLUSPLUS_SENDER_H
#define PIPEPLUSPLUS_SENDER_H

#include "communicator.h"

using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::CompletionQueue;
using boost::interprocess::read_write;
using boost::interprocess::create_only;
using json = nlohmann::ordered_json;

class StubVector {
public:
    explicit StubVector()
            : currentIndex_(0) {
        stubs = std::vector<std::unique_ptr<socket_t>>();
    }

    // Overload the ++ operator to cycle through the stubs
    [[nodiscard]] socket_t *next() {
        auto oldIndex = currentIndex_++;
        currentIndex_ = currentIndex_ % stubs.size();
        return stubs[oldIndex].get();
    }

    [[nodiscard]] socket_t *random() const  {
        return stubs[rand_int(0, stubs.size() - 1)].get();
    }

    [[nodiscard]] socket_t *current() const {
        return stubs[currentIndex_].get();
    }

    [[nodiscard]] socket_t *first() const {
        return stubs[0].get();
    }

    void emplace_back(context_t &ctx, std::string &link, int slo) {
        auto sock = std::make_unique<socket_t>(ctx, ZMQ_REQ);
        sock->set(zmq::sockopt::req_relaxed, 1);
        sock->set(zmq::sockopt::req_correlate, 1);
        sock->set(zmq::sockopt::immediate, 1);
        sock->set(zmq::sockopt::linger, slo);
        sock->set(zmq::sockopt::sndtimeo, slo);
        sock->set(zmq::sockopt::rcvtimeo, slo);
        sock->connect(link);
        stubs.push_back(std::move(sock));
    }

    [[nodiscard]] size_t size() const {
        return stubs.size();
    }

private:
    static inline std::mt19937 &generator() {
        // the generator will only be seeded once (per thread) since it's static
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }

    static int rand_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator());
    }

    std::vector<std::unique_ptr<socket_t>> stubs;
    size_t currentIndex_;
};

struct SenderConfigs : BaseMicroserviceConfigs {
    // Empty for now
    uint8_t dummy;
};

class Sender : public Microservice {
public:
    Sender(const json &jsonConfigs);

    ~Sender() {
        waitStop();
        spdlog::get("container_agent")->info("{0:s} has stopped", msvc_name);
    }

    virtual void Process();

    SenderConfigs loadConfigsFromJson(const json &jsonConfigs);

    virtual void loadConfigs(const json &jsonConfigs, bool isConstructing = false) override;

    virtual void SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) = 0;

    void reloadDnstreams() override;

protected:
    std::string completeSending(message_t &zmq_msg);

    void addToName(const std::string substring, const std::string strToAdd) {
        msvc_name.replace(msvc_name.find(substring), substring.length(), strToAdd + substring);
        msvc_microserviceLogPath.replace(msvc_microserviceLogPath.find(substring), substring.length(), strToAdd + substring);
    }

    static inline std::mt19937 &generator() {
        // the generator will only be seeded once (per thread) since it's static
        static thread_local std::mt19937 gen(std::random_device{}());
        return gen;
    }

    static int rand_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(generator());
    }

    StubVector stubs;
    bool multipleStubs;
    context_t comm_ctx;
    std::atomic<bool> run{};
};

class GPUSender : public Sender {
public:
    explicit GPUSender(const json &jsonConfigs);

    void Process() final;

    void dispatchThread() {
        std::thread sender(&GPUSender::Process, this);
        sender.detach();
    }

    void SendData(std::vector<RequestData<LocalGPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo);

    void SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) final {};

private:

    void serializeIpcMemHandle(const cudaIpcMemHandle_t& handle, char* buffer) {
        memcpy(buffer, &handle, sizeof(cudaIpcMemHandle_t));
    }
};

class LocalCPUSender : public Sender {
public:
    LocalCPUSender(const json &jsonConfigs);

    void dispatchThread() final {
        std::thread sender(&LocalCPUSender::Process, this);
        sender.detach();
    }

    void SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) final;
};

class RemoteCPUSender : public Sender {
public:
    explicit RemoteCPUSender(const json &jsonConfigs);

    void dispatchThread() final {
        std::thread sender(&RemoteCPUSender::Process, this);
        sender.detach();
    }

    void SendData(std::vector<RequestData<LocalCPUReqDataType>> &elements, std::vector<RequestTimeType> &timestamp,
                         std::vector<std::string> &path, RequestSLOType &slo) final;
};

#endif //PIPEPLUSPLUS_SENDER_H
