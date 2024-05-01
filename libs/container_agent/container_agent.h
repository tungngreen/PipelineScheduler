#ifndef CONTAINER_AGENT_H
#define CONTAINER_AGENT_H

#include <vector>
#include <thread>
#include <fstream>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/health_check_service_interface.h>
#include <google/protobuf/empty.pb.h>
#include <filesystem>

#include "microservice.h"
#include "sender.h"
#include "indevicecommunication.grpc.pb.h"

ABSL_DECLARE_FLAG(std::string, name);
ABSL_DECLARE_FLAG(std::optional<std::string>, json);
ABSL_DECLARE_FLAG(std::optional<std::string>, json_path);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json_path);
ABSL_DECLARE_FLAG(uint16_t, port);
ABSL_DECLARE_FLAG(int16_t, device);
ABSL_DECLARE_FLAG(uint16_t, verbose);
ABSL_DECLARE_FLAG(std::string, log_dir);
ABSL_DECLARE_FLAG(std::string, profiling_configs);

using json = nlohmann::json;

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommunication::InDeviceCommunication;
using indevicecommunication::State;
using indevicecommunication::Signal;
using indevicecommunication::Connection;
using indevicecommunication::ProcessData;
using EmptyMessage = google::protobuf::Empty;

enum TransferMethod {
    LocalCPU,
    RemoteCPU,
    GPU
};

namespace msvcconfigs {

    std::tuple<json, json> loadJson();
    std::vector<BaseMicroserviceConfigs> LoadFromJson();
}

struct contRunArgs {
    std::string cont_name;
    uint16_t cont_port;
    int8_t cont_devIndex;
    std::string cont_logPath;
    RUNMODE cont_runmode;
    json cont_pipeConfigs;
    json cont_profilingConfigs;
};

contRunArgs loadRunArgs(int argc, char **argv);

class ContainerAgent {
public:
    ContainerAgent(const std::string &name, uint16_t own_port, int8_t devIndex, const std::string &logPath);

    ContainerAgent(const std::string &name, uint16_t own_port, int8_t devIndex, const std::string &logPath, RUNMODE runmode, const json &profiling_configs);

    ~ContainerAgent() {
        for (auto msvc: msvcs) {
            delete msvc;
        }
        server->Shutdown();
        server_cq->Shutdown();
        sender_cq->Shutdown();
    };

    [[nodiscard]] bool running() const {
        return run;
    }

    void SendState();
    void START() {
        for (auto msvc : msvcs) {
            msvc->unpauseThread();
        }

        spdlog::trace("===========================================CONTAINER STARTS===========================================");
    }
    void checkReady();

    void addMicroservice(std::vector<Microservice*> msvcs) {
        this->msvcs = msvcs;
    }

    void dispatchMicroservices() {
        for (auto msvc : msvcs) {
            msvc->dispatchThread();
        }
    }

    void profiling();

    void loadProfilingConfigs();

protected:
    uint8_t deviceIndex = -1;
    void ReportStart();

    class RequestHandler {
    public:
        RequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : service(service), cq(cq), status(CREATE), responder(&ctx) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        InDeviceCommunication::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class StopRequestHandler : public RequestHandler {
    public:
        StopRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                           std::atomic<bool> *run)
                : RequestHandler(service, cq), run(run) {
            Proceed();
        }

        void Proceed() final;

    private:
        Signal request;
        std::atomic<bool> *run;
    };

    class UpdateSenderRequestHandler : public RequestHandler {
    public:
        UpdateSenderRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                           std::atomic<bool> *run)
                : RequestHandler(service, cq)  {
            Proceed();
        }

        void Proceed() final;

    private:
        Connection request;
        std::vector<Microservice*> *msvcs;
    };

    void HandleRecvRpcs();

    std::string name;
    std::vector<Microservice*> msvcs;
    float arrivalRate;
    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run;

    std::string cont_logDir;
    std::string cont_profilingConfigsPath;
    RUNMODE cont_RUNMODE;

    struct ProfilingConfigs {
        BatchSizeType minBatch;
        BatchSizeType maxBatch;
        uint16_t stepMode; // 0: linear, 1: double
        uint16_t step;
        std::string templateModelPath;
    };

    ProfilingConfigs cont_profilingConfigs;
};

#endif //CONTAINER_AGENT_H