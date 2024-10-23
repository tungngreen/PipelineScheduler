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
#include "data_reader.h"
#include "controller.h"
#include "fcpo_learning.h"
#include "baseprocessor.h"

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

using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommands::InDeviceCommands;
using indevicecommands::ContainerSignal;
using indevicecommands::Connection;
using indevicecommands::TimeKeeping;
using indevicemessages::ProcessData;
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

enum class CONTAINER_STATUS {
    CREATED,
    INITIATED,
    RUNNING,
    PAUSED,
    STOPPED,
    RELOADING
};


class ContainerAgent {
public:
    ContainerAgent(const json &configs);

    virtual ~ContainerAgent() {
        for (auto msvc: cont_receiverList) {
            delete msvc;
        }
        for (auto msvc: cont_preprocessorList) {
            delete msvc;
        }
        for (auto msvc: cont_inferencerList) {
            delete msvc;
        }
        for (auto msvc: cont_postprocessorList) {
            delete msvc;
        }
        for (auto msvc: cont_senderList) {
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
        for (auto msvc: cont_receiverList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_preprocessorList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_inferencerList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_postprocessorList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_senderList) {
            msvc->unpauseThread();
        }
        spdlog::get("container_agent")->info("=========================================== STARTS ===========================================");
    }

    void PROFILING_START(BatchSizeType batch) {
        for (auto msvc: cont_receiverList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_preprocessorList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_inferencerList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_postprocessorList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_senderList) {
            msvc->unpauseThread();
        }

        spdlog::get("container_agent")->info(
                "======================================= PROFILING MODEL BATCH {0:d} =======================================",
                batch);
    }

    bool checkReady(std::vector<Microservice *> msvcs);

    void waitReady();

    bool checkPause(std::vector<Microservice *> msvcs);

    void waitPause();

    bool stopAllMicroservices();

    std::vector<Microservice *> getAllMicroservices();

    void addMicroservice(Microservice *msvc) {
        MicroserviceType type = msvc->msvc_type;
        if (type >= MicroserviceType::Receiver &&
            type < MicroserviceType::PreprocessBatcher) {
            cont_receiverList.push_back(msvc);
        } else if (type >= MicroserviceType::PreprocessBatcher &&
                    type < MicroserviceType::TRTInferencer) {
            cont_preprocessorList.push_back(msvc);
        } else if (type >= MicroserviceType::TRTInferencer &&
                    type < MicroserviceType::Postprocessor) {
            cont_inferencerList.push_back(msvc);
        } else if (type >= MicroserviceType::Postprocessor &&
                    type < MicroserviceType::Sender) {
            cont_postprocessorList.push_back(msvc);
        } else if (type >= MicroserviceType::Sender) {
            cont_senderList.push_back(msvc);
        } else {
            throw std::runtime_error("Unknown microservice type: " + std::to_string((int)type));
        }
    }

    void addMicroservice(std::vector<Microservice *> msvcs) {
        for (auto &msvc: msvcs) {
            addMicroservice(msvc);
        }
    }

    void dispatchMicroservices() {
        for (auto &msvc: cont_receiverList) {
            msvc->dispatchThread();
        }
        for (auto &msvc: cont_preprocessorList) {
            msvc->dispatchThread();
        }
        for (auto &msvc: cont_inferencerList) {
            msvc->dispatchThread();
        }
        for (auto &msvc: cont_postprocessorList) {
            msvc->dispatchThread();
        }
        for (auto &msvc: cont_senderList) {
            msvc->dispatchThread();
        }
    }

    void transferFrameID(std::string url);

    void profiling(const json &pipeConfigs, const json &profileConfigs);

    virtual void runService(const json &pipeConfigs, const json &configs);

protected:

    virtual void initiateMicroservices(const json &pipeConfigs);

    bool addPreprocessor(uint8_t totalNumInstances);

    bool removePreprocessor(uint8_t numLeftInstances);

    bool addPostprocessor(uint8_t totalNumInstances);

    bool removePostprocessor(uint8_t numLeftInstances);

    void ReportStart();

    void collectRuntimeMetrics();

    void updateArrivalRecords(ArrivalRecordType arrivalRecords, RunningArrivalRecord &perSecondArrivalRecords,
                              unsigned int lateCount, unsigned int queueDrops);

    void updateProcessRecords(ProcessRecordType processRecords, BatchInferRecordType batchInferRecords);

    class RequestHandler {
    public:
        RequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq)
                : service(service), cq(cq), status(CREATE), responder(&ctx) {};

        virtual ~RequestHandler() = default;

        virtual void Proceed() = 0;

    protected:
        enum CallStatus {
            CREATE, PROCESS, FINISH
        };

        InDeviceCommands::AsyncService *service;
        ServerCompletionQueue *cq;
        ServerContext ctx;
        CallStatus status;
        EmptyMessage reply;
        grpc::ServerAsyncResponseWriter<EmptyMessage> responder;
    };

    class KeepAliveRequestHandler : public RequestHandler {
    public:
        KeepAliveRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq)
                : RequestHandler(service, cq) {
            Proceed();
        }

        void Proceed() final;

    private:
        EmptyMessage request;
    };

    class StopRequestHandler : public RequestHandler {
    public:
        StopRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
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
        UpdateSenderRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
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
        UpdateBatchSizeRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
                                      std::vector<Microservice *> *msvcs)
                : RequestHandler(service, cq), msvcs(msvcs) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommands::Int32 request;
        std::vector<Microservice *> *msvcs;
    };

    class UpdateResolutionRequestHandler : public RequestHandler {
    public:
        UpdateResolutionRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
                                      ContainerAgent *container_agent)
                : RequestHandler(service, cq), container_agent(container_agent) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommands::Dimensions request;
        ContainerAgent *container_agent;
    };

    class UpdateTimeKeepingRequestHandler : public RequestHandler {
    public:
        UpdateTimeKeepingRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
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
        SyncDatasourcesRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
                                      ContainerAgent *containerAgent)
                : RequestHandler(service, cq), containerAgent(containerAgent) {
            Proceed();
        }

        void Proceed() final;

    private:
        indevicecommands::Int32 request;
        ContainerAgent *containerAgent;
    };

    class FederatedLearningReturnRequestHandler : public RequestHandler {
    public:
        FederatedLearningReturnRequestHandler(InDeviceCommands::AsyncService *service, ServerCompletionQueue *cq,
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

    std::mutex cont_pipeStructureMutex;

    CONTAINER_STATUS cont_status = CONTAINER_STATUS::CREATED;

    std::string cont_experimentName;
    std::string cont_systemName;
    std::string cont_name;
    std::string cont_pipeName;
    std::string cont_taskName;
    // Name of the host where the container is running
    std::string cont_hostDevice;
    std::string cont_hostDeviceType;
    std::string cont_inferModel;

    std::unique_ptr<ServerCompletionQueue> server_cq;
    CompletionQueue *sender_cq;
    InDeviceCommands::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::shared_ptr<InDeviceMessages::Stub> stub;
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

    std::vector<Microservice *> cont_receiverList;
    std::vector<Microservice *> cont_senderList;
    std::vector<Microservice *> cont_preprocessorList;
    std::vector<Microservice *> cont_postprocessorList;
    std::vector<Microservice *> cont_inferencerList;

    PPOAgent *cont_ppo;
    uint64_t cont_rlIntervalMillisec = 1000;
    ClockType cont_nextRLDecisionTime = std::chrono::system_clock::now();
};

#endif //CONTAINER_AGENT_H