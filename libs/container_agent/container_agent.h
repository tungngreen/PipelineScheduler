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
#include <pqxx/pqxx>

#include "profiler.h"
#include "microservice.h"
#include "receiver.h"
#include "sender.h"
#include "indevicecommunication.grpc.pb.h"
#include "controller.h"
#include "batch_learning.h"

ABSL_DECLARE_FLAG(std::optional<std::string>, json);
ABSL_DECLARE_FLAG(std::optional<std::string>, json_path);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json);
ABSL_DECLARE_FLAG(std::optional<std::string>, trt_json_path);
ABSL_DECLARE_FLAG(uint16_t, port);
ABSL_DECLARE_FLAG(uint16_t, port_offset);
ABSL_DECLARE_FLAG(int16_t, device);
ABSL_DECLARE_FLAG(uint16_t, verbose);
ABSL_DECLARE_FLAG(uint16_t, logging_mode);
ABSL_DECLARE_FLAG(std::string, log_dir);
ABSL_DECLARE_FLAG(uint16_t, profiling_mode);

using json = nlohmann::ordered_json;

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommunication::InDeviceCommunication;
using indevicecommunication::ContainerSignal;
using indevicecommunication::Connection;
using indevicecommunication::ProcessData;
using indevicecommunication::TimeKeeping;
using EmptyMessage = google::protobuf::Empty;

enum TransferMethod {
    LocalCPU,
    RemoteCPU,
    GPU
};

std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::milliseconds> timePointCastMillisecond(
    std::chrono::system_clock::time_point tp);

namespace msvcconfigs {

    json loadJson();

    std::vector<BaseMicroserviceConfigs> LoadFromJson();
}

json loadRunArgs(int argc, char **argv);

void addProfileConfigs(json &msvcConfigs, const json &profileConfigs);

std::vector<float> getRatesInPeriods(const std::vector<ClockType> &timestamps, const std::vector<uint32_t> &periodMillisec);


class ContainerAgent {
public:
    ContainerAgent(const json &configs);

    virtual ~ContainerAgent() {
        for (auto msvc: cont_msvcsList) {
            delete msvc;
        }
        server->Shutdown();
        server_cq->Shutdown();
        sender_cq->Shutdown();
    };

    [[nodiscard]] bool running() const {
        return run;
    }

    void START() {
        for (auto msvc: cont_msvcsList) {
            msvc->unpauseThread();
        }
        spdlog::get("container_agent")->info("=========================================== STARTS ===========================================");
    }

    void PROFILING_START(BatchSizeType batch) {
        for (auto msvc: cont_msvcsList) {
            msvc->unpauseThread();
        }

        spdlog::get("container_agent")->info(
                "======================================= PROFILING MODEL BATCH {0:d} =======================================",
                batch);
    }

    void waitReady();

    bool checkReady();

    void waitPause();

    bool checkPause();

    void addMicroservice(std::vector<Microservice *> msvcs) {
        this->cont_msvcsList = msvcs;
    }

    void dispatchMicroservices() {
        for (auto &msvc: cont_msvcsList) {
            msvc->dispatchThread();
        }
    }

    void transferFrameID(std::string url);

    void profiling(const json &pipeConfigs, const json &profileConfigs);

    virtual void runService(const json &pipeConfigs, const json &configs);

protected:
    void ReportStart();

    void collectRuntimeMetrics();

    void updateArrivalRecords(RunningArrivalRecord &perSecondArrivalRecords, unsigned int lateCount, unsigned int queueDrops);

    void updateProcessRecords();

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

    class KeepAliveRequestHandler : public RequestHandler {
    public:
        KeepAliveRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq)
                : RequestHandler(service, cq) {
            Proceed();
        }

        void Proceed() final;

    private:
        EmptyMessage request;
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
        ContainerSignal request;
        std::atomic<bool> *run;
    };

    class UpdateSenderRequestHandler : public RequestHandler {
    public:
        UpdateSenderRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                   std::vector<Microservice *> *msvcs)
                : RequestHandler(service, cq), msvcs(msvcs) {
            Proceed();
        }

        void Proceed() final;

    private:
        Connection request;
        std::vector<Microservice *> *msvcs;
    };

    class UpdateBatchSizeRequestHandler : public RequestHandler {
    public:
        UpdateBatchSizeRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                      std::vector<Microservice *> *msvcs)
                : RequestHandler(service, cq), msvcs(msvcs) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::Int32 request;
        std::vector<Microservice *> *msvcs;
    };

    class UpdateResolutionRequestHandler : public RequestHandler {
    public:
        UpdateResolutionRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                      ContainerAgent *container_agent)
                : RequestHandler(service, cq), container_agent(container_agent) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::Dimensions request;
        ContainerAgent *container_agent;
    };

    class UpdateTimeKeepingRequestHandler : public RequestHandler {
    public:
        UpdateTimeKeepingRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                       ContainerAgent *container_agent)
                : RequestHandler(service, cq), container_agent(container_agent) {
            Proceed();
        }

        void Proceed() final;

    private:
        TimeKeeping request;
        ContainerAgent *container_agent;
    };

    class SyncDatasourcesRequestHandler : public RequestHandler {
    public:
        SyncDatasourcesRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                      ContainerAgent *containerAgent)
                : RequestHandler(service, cq), containerAgent(containerAgent) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommunication::Int32 request;
        ContainerAgent *containerAgent;
    };

    class FederatedLearningReturnRequestHandler : public RequestHandler {
    public:
        FederatedLearningReturnRequestHandler(InDeviceCommunication::AsyncService *service, ServerCompletionQueue *cq,
                                              PPOAgent *ppoAgent)
                : RequestHandler(service, cq), ppoAgent(ppoAgent) {
            Proceed();
        }

        void Proceed() final;

    private:
        FlData request;
        PPOAgent *ppoAgent;
    };

    virtual void HandleRecvRpcs();

    bool readModelProfile(const json &profile);

    std::string cont_experimentName;
    std::string cont_systemName;
    std::string cont_name;
    std::vector<Microservice *> cont_msvcsList;
    std::string cont_pipeName;
    std::string cont_taskName;
    // Name of the host where the container is running
    std::string cont_hostDevice;
    std::string cont_hostDeviceType;
    std::string cont_inferModel;

    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::shared_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run;

    unsigned int pid;
    Profiler *profiler;

    std::string cont_logDir;
    RUNMODE cont_RUNMODE;
    uint8_t cont_deviceIndex;

    bool reportHwMetrics;
    std::string cont_hwMetricsTableName;
    SummarizedHardwareMetrics cont_hwMetrics;
    BatchInferProfileListType cont_batchInferProfileList;

    std::string cont_batchInferTableName;
    std::string cont_arrivalTableName;
    std::string cont_processTableName;
    std::string cont_networkTableName;

    MetricsServerConfigs cont_metricsServerConfigs;
    std::unique_ptr<pqxx::connection> cont_metricsServerConn = nullptr;

    std::vector<spdlog::sink_ptr> cont_loggerSinks = {};
    std::shared_ptr<spdlog::logger> cont_logger;    

    PPOAgent *cont_ppo;
    uint64_t cont_rlIntervalMillisec = 1000;
    ClockType cont_nextRLDecisionTime = std::chrono::system_clock::now();
};

#endif //CONTAINER_AGENT_H