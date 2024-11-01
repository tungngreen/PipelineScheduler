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
#include "baseprocessor.h"
#include "data_reader.h"

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

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientContext;
using grpc::ClientAsyncResponseReader;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using indevicecommunication::InDeviceCommunication;
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

struct MicroserviceGroup {
    std::vector<Microservice *> msvcList;
    std::vector<ThreadSafeFixSizedDoubleQueue *> outQueue;
};


class ContainerAgent {
public:
    ContainerAgent(const json &configs);

    virtual ~ContainerAgent() {
        for (auto msvc: cont_msvcsGroups["receiver"].msvcList) {
            delete msvc;
        }
        for (auto msvc: cont_msvcsGroups["preprocessor"].msvcList) {
            delete msvc;
        }
        for (auto msvc: cont_msvcsGroups["inference"].msvcList) {
            delete msvc;
        }
        for (auto msvc: cont_msvcsGroups["postprocessor"].msvcList) {
            delete msvc;
        }
        for (auto msvc: cont_msvcsGroups["sender"].msvcList) {
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
        for (auto msvcGroup: cont_msvcsGroups) {
            for (auto msvc: msvcGroup.second.msvcList) {
                msvc->unpauseThread();
            }
        }
        spdlog::get("container_agent")->info("=========================================== STARTS ===========================================");
    }

    void PROFILING_START(BatchSizeType batch) {
        for (auto msvc: cont_msvcsGroups["receiver"].msvcList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_msvcsGroups["preprocessor"].msvcList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_msvcsGroups["inference"].msvcList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_msvcsGroups["postprocessor"].msvcList) {
            msvc->unpauseThread();
        }
        for (auto msvc: cont_msvcsGroups["sender"].msvcList) {
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
            type < MicroserviceType::Preprocessor) {
            cont_msvcsGroups["receiver"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Preprocessor &&
                    type < MicroserviceType::TRTInferencer) {
            cont_msvcsGroups["preprocessor"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Batcher &&
                    type < MicroserviceType::TRTInferencer) {
            cont_msvcsGroups["batcher"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::TRTInferencer &&
                    type < MicroserviceType::Postprocessor) {
            cont_msvcsGroups["inference"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Postprocessor &&
                    type < MicroserviceType::Sender) {
            cont_msvcsGroups["postprocessor"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Sender) {
            cont_msvcsGroups["sender"].msvcList.push_back(msvc);
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
        for (auto &group: cont_msvcsGroups) {
            for (auto &msvc: group.second.msvcList) {
                msvc->dispatchThread();
            }
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

    void updateProfileTable();

    void ReportStart();

    void collectRuntimeMetrics();

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
        Signal request;
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
        indevicecommunication::TimeKeeping request;
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
    InDeviceCommunication::AsyncService service;
    std::unique_ptr<grpc::Server> server;
    std::unique_ptr<InDeviceCommunication::Stub> stub;
    std::atomic<bool> run;

    unsigned int pid;
    Profiler *profiler;

    std::string cont_logDir;
    RUNMODE cont_RUNMODE;
    uint8_t cont_deviceIndex;

    /**
     * @brief Metrics
     */

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

    std::map<std::string, MicroserviceGroup> cont_msvcsGroups;
};

#endif //CONTAINER_AGENT_H