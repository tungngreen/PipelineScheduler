#ifndef CONTAINER_AGENT_H
#define CONTAINER_AGENT_H

#include <vector>
#include <thread>
#include <fstream>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
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

using indevicemessages::ContainerSignal;
using indevicemessages::Connection;
using indevicemessages::TimeKeeping;
using indevicemessages::Dimensions;
using indevicemessages::ProcessData;
using indevicemessages::ContainerMetrics;
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
    };

    [[nodiscard]] bool running() const { return run; }

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
        if (type >= MicroserviceType::Receiver && type < MicroserviceType::Preprocessor) {
            cont_msvcsGroups["receiver"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Preprocessor && type < MicroserviceType::TRTInferencer) {
            cont_msvcsGroups["preprocessor"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Batcher && type < MicroserviceType::TRTInferencer) {
            cont_msvcsGroups["batcher"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::TRTInferencer && type < MicroserviceType::Postprocessor) {
            cont_msvcsGroups["inference"].msvcList.push_back(msvc);
        } else if (type >= MicroserviceType::Postprocessor && type < MicroserviceType::Sender) {
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

    virtual void runService(const json &pipeConfigs, const json &configs);

protected:

    /////////////////////////////////////////// PROTECTED FUNCTIONS ///////////////////////////////////////////

    // START AND STOP
    virtual void initiateMicroservices(const json &pipeConfigs);
    bool addPreprocessor(uint8_t totalNumInstances);
    bool removePreprocessor(uint8_t numLeftInstances);
    bool addPostprocessor(uint8_t totalNumInstances);
    bool removePostprocessor(uint8_t numLeftInstances);
    void reportStart();
    void stopExecution(const std::string &msg);

    // RUNTIME CONTROL
    void applyFramePacking(int resolutionConfig);
    void applyBatchSize(int batchSize);
    void applyBatchingTimeout(int timeoutChoice);
    void applyMultiThreading(int multiThreadingConfig);
    void updateSender(const std::string &msg);
    void updateBatchSize(const std::string &msg);
    void updateResolution(const std::string &msg);
    void updateTimeKeeping(const std::string &msg);
    void transferFrameID(const std::string &msg);
    void setFrameID(const std::string &msg);

    // MESSAGING
    void HandleControlMessages();
    std::string sendMessageToDevice(const std::string &type, const std::string &content);

    // METRICS & PROFILING
    void collectRuntimeMetrics();
    void updateArrivalRecords(ArrivalRecordType arrivalRecords, RunningArrivalRecord &perSecondArrivalRecords,
                              unsigned int lateCount, unsigned int queueDrops);
    void updateProcessRecords(ProcessRecordType processRecords, BatchInferRecordType batchInferRecords);
    bool readModelProfile(const json &profile);

    /////////////////////////////////////////// PROTECTED VARIABLES ///////////////////////////////////////////

    // BASIC INFORMATION
    std::string cont_experimentName;
    std::string cont_systemName;
    std::string cont_name;
    std::string cont_pipeName;
    std::string cont_taskName;
    std::string cont_hostDevice; // Device the Container is running on
    std::string cont_hostDeviceType;
    std::string cont_inferModel;
    std::atomic<bool> hasDataReader;
    std::atomic<bool> isDataSource;
    RUNMODE cont_RUNMODE;
    uint8_t cont_deviceIndex; // GPU ID
    unsigned int pid;

    // RUNTIME VARIABLES
    std::mutex cont_pipeStructureMutex;
    std::map<std::string, MicroserviceGroup> cont_msvcsGroups;
    std::atomic<bool> run;
    int cont_pipeSLO;
    int cont_modelSLO;
    double cont_request_arrival_rate;
    int cont_queue_size;
    int64_t  cont_ewma_latency;
    int cont_late_drops;
    int cont_throughput;
    threadingAction cont_threadingAction = NoMultiThreads;

    // MESSAGING
    context_t messaging_ctx;
    socket_t sending_socket;
    socket_t device_message_queue;
    std::unordered_map<std::string, std::function<void(const std::string&)>> handlers;

    // PROFILING DATA & METRICS
    Profiler *profiler;
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

    // LOGGING
    std::string cont_logDir;
    std::vector<spdlog::sink_ptr> cont_loggerSinks = {};
    std::shared_ptr<spdlog::logger> cont_logger;

    // LOCAL OPTIMIZATION
    FCPOAgent *cont_fcpo_agent;
    uint64_t cont_localOptimizationIntervalMillisec;
    ClockType cont_nextOptimizationMetricsTime = std::chrono::high_resolution_clock ::now();
};

#endif //CONTAINER_AGENT_H

