#ifndef MISC_H
#define MISC_H

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include "../json/json.h"
#include "spdlog/spdlog.h"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "opencv2/opencv.hpp"
#include <unordered_set>
#include <pqxx/pqxx>
#include <grpcpp/grpcpp.h>
#include "absl/strings/str_format.h"
#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include <fstream>

using grpc::Status;
using grpc::CompletionQueue;
using grpc::ClientAsyncResponseReader;

typedef uint16_t NumQueuesType;
typedef uint16_t QueueLengthType;
typedef uint32_t MsvcSLOType;
typedef std::vector<uint32_t> RequestSLOType;
typedef std::vector<std::string> RequestPathType;
typedef uint16_t NumMscvType;
typedef std::chrono::high_resolution_clock::time_point ClockType;
typedef std::vector<ClockType> RequestTimeType;
typedef std::vector<RequestTimeType> BatchTimeType;
const uint8_t CUDA_IPC_HANDLE_LENGTH = 64; // bytes
typedef const char *InterConGPUReqDataType;
typedef std::vector<int32_t> RequestDataShapeType;
typedef std::vector<std::vector<int32_t>> RequestShapeType;
typedef cv::cuda::GpuMat LocalGPUReqDataType;
typedef cv::Mat LocalCPUReqDataType;
typedef uint16_t BatchSizeType;
typedef uint32_t RequestMemSizeType;

// Hw Metrics
typedef int CpuUtilType;
typedef unsigned int GpuUtilType;
typedef int MemUsageType;
typedef unsigned int GpuMemUsageType;

const uint8_t NUM_LANES_PER_GPU = 3;
const uint8_t NUM_GPUS = 4;

struct BatchInferProfile {
    uint64_t p95inferLat;
    
    CpuUtilType cpuUtil;
    MemUsageType memUsage;
    MemUsageType rssMemUsage;
    GpuUtilType gpuUtil;
    GpuMemUsageType gpuMemUsage;
};

typedef std::map<BatchSizeType, BatchInferProfile> BatchInferProfileListType;

struct Record {
    template<typename T>
    T findPercentile(const std::vector<T> &vector, uint8_t &percentile) {
        std::vector<T> vectorCopy = vector;
        std::sort(vectorCopy.begin(), vectorCopy.end());
        return vectorCopy[vectorCopy.size() * percentile / 100];
    }
};

struct PercentilesArrivalRecord {
    uint64_t outQueueingDuration;
    uint64_t transferDuration;
    uint64_t queueingDuration;
    uint32_t totalPkgSize;
};

/**
 * @brief Arrival record structure
 * The point of this is to quickly summarize the arrival records collected during the last period
 * 
 */
struct ArrivalRecord : public Record {
    std::vector<uint64_t> outQueueingDuration; //prevPostProcTime - postproc's outqueue time
    std::vector<uint64_t> transferDuration; //arrivalTime - prevSenderTime
    std::vector<uint64_t> queueingDuration; //batcher's pop time - arrivalTime
    std::vector<ClockType> arrivalTime;
    std::vector<uint32_t> totalPkgSize;
    std::vector<uint32_t> reqSize;

    std::map<uint8_t, PercentilesArrivalRecord> findPercentileAll(const std::vector<uint8_t>& percentiles) {
        std::map<uint8_t, PercentilesArrivalRecord> results;
        for (uint8_t percent : percentiles) {
            results[percent] = {
                findPercentile<uint64_t>(outQueueingDuration, percent), 
                findPercentile<uint64_t>(transferDuration, percent),
                findPercentile<uint64_t>(queueingDuration, percent),
                findPercentile<uint32_t>(reqSize, percent)
            };
        }
        return results;
    }
};

//<reqOriginStream, SenderHost>, Record>>
typedef std::map<std::pair<std::string, std::string>, ArrivalRecord> ArrivalRecordType;

struct PercentilesProcessRecord {
    uint64_t prepDuration;
    uint64_t batchDuration;
    uint64_t inferDuration;
    uint64_t postDuration;
    uint32_t inputSize;
    uint32_t outputSize;
};

/**
 * @brief Process record structure
 * The point of this is to quickly summarize the records collected during the last period
 * 
 */
struct ProcessRecord : public Record {
    std::vector<uint64_t> prepDuration;
    std::vector<uint64_t> batchDuration;
    std::vector<uint64_t> inferDuration;
    std::vector<uint64_t> postDuration;
    std::vector<uint32_t> inputSize;
    std::vector<uint32_t> outputSize;
    std::vector<ClockType> postEndTime;
    std::vector<BatchSizeType> inferBatchSize;

    std::map<uint8_t, PercentilesProcessRecord> findPercentileAll(const std::vector<uint8_t>& percentiles) {
        std::map<uint8_t, PercentilesProcessRecord> results;
        for (uint8_t percent : percentiles) {
            results[percent] = {
                findPercentile<uint64_t>(prepDuration, percent),
                findPercentile<uint64_t>(batchDuration, percent),
                findPercentile<uint64_t>(inferDuration, percent),
                findPercentile<uint64_t>(postDuration, percent),
                findPercentile<uint32_t>(inputSize, percent),
                findPercentile<uint32_t>(outputSize, percent)
            };
        }
        return results;
    }
};

struct PercentilesBatchInferRecord {
    uint64_t inferDuration;
};

struct BatchInferRecord : public Record {
    std::vector<uint64_t> inferDuration;

    std::map<uint8_t, PercentilesBatchInferRecord> findPercentileAll(const std::vector<uint8_t>& percentiles) {
        std::map<uint8_t, PercentilesBatchInferRecord> results;
        for (uint8_t percent : percentiles) {
            results[percent] = {
                findPercentile<uint64_t>(inferDuration, percent)
            };
        }
        return results;
    }
};

typedef std::map<std::pair<std::string, BatchSizeType>, BatchInferRecord> BatchInferRecordType;

/**
 * @brief 
 * 
 */
struct PercentilesNetworkRecord {
    uint32_t totalPkgSize = -1;
    uint64_t transferDuration = -1;
};

/**
 * @brief <<sender, receiver>, Record>
 */
typedef std::map<std::string, PercentilesNetworkRecord> NetworkRecordType;
typedef std::vector<std::pair<uint32_t, uint64_t>> NetworkEntryType;

uint64_t calculateP95(std::vector<uint64_t> &values);

NetworkEntryType aggregateNetworkEntries(const NetworkEntryType &res);

uint64_t estimateNetworkLatency(const NetworkEntryType& res, const uint32_t &totalPkgSize);

// Arrival rate coming to a certain model in the pipeline

// Network profile between two devices
struct NetworkProfile {
    uint64_t p95OutQueueingDuration; // out queue before sender of the last container
    uint64_t p95TransferDuration;
    uint64_t p95QueueingDuration; // in queue of batcher of this container
    uint32_t p95PackageSize;
};

// Device to device network profile
typedef std::map<std::pair<std::string, std::string>, NetworkProfile> D2DNetworkProfile;

// Arrival profile of a certain model
struct ModelArrivalProfile {
    // Network profile between two devices, one of which is the receiver host that runs the model
    D2DNetworkProfile d2dNetworkProfile;
    float arrivalRates;
};

// <<pipelineName, modelName>, ModelArrivalProfile>
typedef std::map<std::pair<std::string, std::string>, ModelArrivalProfile> ModelArrivalProfileList;

struct ModelProfile {
    // p95 latency of preprocessing per query
    uint64_t p95prepLat;
    // p95 latency of batch inference per query
    BatchInferProfileListType batchInfer;
    // p95 latency of postprocessing per query
    uint64_t p95postLat;
    // Average size of incoming queries
    int p95InputSize = 1; // bytes
    // Average total size of outgoing queries
    int p95OutputSize = 1; // bytes
};


//<reqOriginStream, Record>
// Since each stream's content is unique, which causes unique process behaviors, 
// we can use the stream name as the key to store the process records
typedef std::map<std::string, ProcessRecord> ProcessRecordType;

typedef std::map<std::string, ModelProfile> PerDeviceModelProfileType;

struct HardwareMetrics {
    ClockType timestamp;
    CpuUtilType cpuUsage = 0;
    MemUsageType memUsage = 0;
    MemUsageType rssMemUsage = 0;
    GpuUtilType gpuUsage = 0;
    GpuMemUsageType gpuMemUsage = 0;
};

struct DeviceHardwareMetrics {
    ClockType timestamp;
    CpuUtilType cpuUsage = 0;
    MemUsageType memUsage = 0;
    MemUsageType rssMemUsage = 0;
    std::vector<GpuUtilType> gpuUsage;
    std::vector<GpuMemUsageType> gpuMemUsage;
};

struct SummarizedHardwareMetrics {
    CpuUtilType cpuUsage = 0;
    MemUsageType memUsage = 0;
    MemUsageType rssMemUsage = 0;
    GpuUtilType gpuUsage = 0;
    GpuMemUsageType gpuMemUsage = 0;

    bool metricsAvailable = false;

    SummarizedHardwareMetrics& operator= (const SummarizedHardwareMetrics &metrics) {
        metricsAvailable = true;
        cpuUsage = std::max(metrics.cpuUsage, cpuUsage);
        memUsage = std::max(metrics.memUsage, memUsage);
        rssMemUsage = std::max(metrics.rssMemUsage, rssMemUsage);
        gpuUsage = std::max(metrics.gpuUsage, gpuUsage);
        gpuMemUsage = std::max(metrics.gpuMemUsage, gpuMemUsage);
        return *this;
    }

    void clear() {
        metricsAvailable = false;
        cpuUsage = 0;
        memUsage = 0;
        rssMemUsage = 0;
        gpuUsage = 0;
        gpuMemUsage = 0;
    }
};

typedef std::chrono::microseconds TimePrecisionType;

const std::unordered_set<uint16_t> GRAYSCALE_CONVERSION_CODES = {6, 7, 10, 11};


void saveGPUAsImg(const cv::cuda::GpuMat &img, std::string name = "test.jpg", float scale = 1.f);

void saveCPUAsImg(const cv::Mat &img, std::string name = "test.jpg", float scale = 1.f);

struct MetricsServerConfigs {
    std::string ip = "localhost";
    uint64_t port = 60004;
    std::string DBName = "pipeline";
    std::string schema = "public";
    std::string user = "container_agent";
    std::string password = "pipe";
    uint64_t hwMetricsScrapeIntervalMillisec = 50;
    uint64_t metricsReportIntervalMillisec = 60000;
    std::vector<uint64_t> queryArrivalPeriodMillisec;
    ClockType nextHwMetricsScrapeTime;
    ClockType nextMetricsReportTime;

    MetricsServerConfigs(const std::string &path) {
        std::ifstream file(path);
        nlohmann::json j = nlohmann::json::parse(file);
        from_json(j);

    }

    MetricsServerConfigs() = default;

    void from_json(const nlohmann::json &j) {
        j.at("metricsServer_ip").get_to(ip);
        j.at("metricsServer_port").get_to(port);
        j.at("metricsServer_DBName").get_to(DBName);
        j.at("metricsServer_user").get_to(user);
        j.at("metricsServer_password").get_to(password);
        j.at("metricsServer_queryArrivalPeriodMillisec").get_to(queryArrivalPeriodMillisec);
        j.at("metricsServer_hwMetricsScrapeIntervalMillisec").get_to(hwMetricsScrapeIntervalMillisec);
        j.at("metricsServer_metricsReportIntervalMillisec").get_to(metricsReportIntervalMillisec);

        nextHwMetricsScrapeTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(4 * hwMetricsScrapeIntervalMillisec);
        nextMetricsReportTime = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(metricsReportIntervalMillisec);
    }
};

std::unique_ptr<pqxx::connection> connectToMetricsServer(MetricsServerConfigs &metricsServerConfigs, const std::string &name);

enum SystemDeviceType {
    Server,
    NXXavier,
    AGXXavier,
    OrinNano
};

typedef std::map<SystemDeviceType, std::string> DeviceInfoType;

enum PipelineType {
    Traffic,
    Video_Call,
    Building_Security
};

enum MODEL_DATA_TYPE {
    int8 = sizeof(uint8_t),
    fp16 = int(sizeof(float) / 2),
    fp32 = sizeof(float)
};

enum ModelType {
    DataSource,
    Sink,
    Yolov5n,
    Yolov5s,
    Yolov5m,
    Yolov5nDsrc,
    Arcface,
    Retinaface,
    RetinafaceDsrc,
    PlateDet,
    Movenet,
    Emotionnet,
    Gender,
    Age,
    CarBrand
};

extern std::map<std::string, std::string> keywordAbbrs;
extern std::map<SystemDeviceType, std::string> SystemDeviceTypeList;
extern std::map<std::string, SystemDeviceType> SystemDeviceTypeReverseList;
extern std::map<ModelType, std::string> ModelTypeList;
extern std::map<std::string, ModelType> ModelTypeReverseList;

struct ContainerInfo {
    std::string taskName;
    std::string modelName;
    std::string modelPath;
    nlohmann::json templateConfig;
    std::string runCommand;
};

typedef std::map<std::string, ContainerInfo> ContainerLibType;



inline void checkCudaErrorCode(cudaError_t code, std::string func_name) {
    if (code != 0) {
        std::string errMsg = "At " + func_name + "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

namespace trt {
    // TRTConfigs for the network
    struct TRTConfigs {
        // Path to the engine or onnx file
        std::string path = "";
        // Path to save the engine in the case of conversion from onnx
        std::string storePath = "";
        // Precision to use for GPU inference.
        MODEL_DATA_TYPE precision = MODEL_DATA_TYPE::fp32;
        // If INT8 precision is selected, must provide path to calibration dataset directory.
        std::string calibrationDataDirectoryPath;
        // The batch size to be used when computing calibration data for INT8 inference.
        // Should be set to as large a batch number as your GPU will support.
        int32_t calibrationBatchSize = 128;
        // The batch size which should be optimized for.
        int32_t optBatchSize = 1;
        // Maximum batch size  we want to use for inference
        // this will be compared with the maximum batch size set when the engine model was created min(maxBatchSize, engine_max)
        // This determines the GPU memory buffer sizes allocated upon model loading so CANNOT BE CHANGE DURING RUNTIME.
        int32_t maxBatchSize = 128;
        // GPU device index
        int8_t deviceIndex = 0;

        size_t maxWorkspaceSize = 1 << 30;
        bool normalize = false;
        std::array<float, 3> subVals{0.f, 0.f, 0.f};
        std::array<float, 3> divVals{1.f, 1.f, 1.f};
    };

    void to_json(nlohmann::json &j, const TRTConfigs &val);

    void from_json(const nlohmann::json &j, TRTConfigs &val);
}

class Stopwatch {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
    bool running;

public:
    Stopwatch() : running(false) {}

    void start() {
        if (!running) {
            start_time = std::chrono::high_resolution_clock::now();
            running = true;
        }
    }

    void stop() {
        if (running) {
            stop_time = std::chrono::high_resolution_clock::now();
            running = false;
        }
    }

    void reset() {
        running = false;
    }

    uint64_t elapsed_microseconds() const {
        if (running) {
            return std::chrono::duration_cast<TimePrecisionType>(std::chrono::high_resolution_clock::now() - start_time).count();
        } else {
            return std::chrono::duration_cast<TimePrecisionType>(stop_time - start_time).count();
        }
    }

    ClockType getStartTime() {
        return start_time;
    }
};

void setupLogger(
    const std::string &logPath,
    const std::string &loggerName,
    uint16_t loggingMode,
    uint16_t verboseLevel,
    std::vector<spdlog::sink_ptr> &loggerSinks,
    std::shared_ptr<spdlog::logger> &logger
);

float fractionToFloat(const std::string& fraction);

std::string removeSubstring(const std::string& str, const std::string& substring);

std::string timePointToEpochString(const std::chrono::system_clock::time_point& tp);

std::string replaceSubstring(const std::string& input, const std::string& toReplace, const std::string& replacement);

std::vector<std::string> splitString(const std::string& str, const std::string& delimiter) ;

std::string getTimestampString();

uint64_t getTimestamp();

pqxx::result pushSQL(pqxx::connection &conn, const std::string &sql);

pqxx::result pullSQL(pqxx::connection &conn, const std::string &sql);

bool isHypertable(pqxx::connection &conn, const std::string &tableName);

bool tableExists(pqxx::connection &conn, const std::string &schemaName, const std::string &tableName);

std::string abbreviate(const std::string &keyphrase, const std::string delimiter = "_");

bool confirmIntention(const std::string& message, const std::string& magicPhrase);


// ================================================================== Queries functions ==================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================

float queryArrivalRate(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const std::vector<uint8_t> &periods = {1, 3, 7, 15, 30, 60} //seconds
);

NetworkProfile queryNetworkProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const std::string &senderHost,
    const std::string &receiverHost,
    const NetworkEntryType &networkEntries
);

ModelArrivalProfile queryModelArrivalProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &taskName,
    const std::string &modelName,
    const std::vector<std::pair<std::string, std::string>> &commPair,
    const std::map<std::pair<std::string, std::string>, NetworkEntryType> &networkEntries,
    const std::vector<uint8_t> &periods = {1, 3, 7, 15, 30, 60} //seconds
);

void queryBatchInferLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &deviceTypeName,
    const std::string &modelName,
    ModelProfile &profile
);

BatchInferProfileListType queryBatchInferLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &deviceTypeName,
    const std::string &modelName
);

void queryPrePostLatency(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &deviceTypeName,
    const std::string &modelName,
    ModelProfile &profile
);

void queryResourceRequirements(
    pqxx::connection &metricsConn,
    const std::string &deviceTypeName,
    const std::string &modelName,
    ModelProfile &profile
);

ModelProfile queryModelProfile(
    pqxx::connection &metricsConn,
    const std::string &experimentName,
    const std::string &systemName,
    const std::string &pipelineName,
    const std::string &streamName,
    const std::string &deviceName,
    const std::string &deviceTypeName,
    const std::string &modelName
);

// =======================================================================================================================================================
// =======================================================================================================================================================
// =======================================================================================================================================================

bool isFileEmpty(const std::string& filePath);

std::string getDeviceTypeAbbr(const SystemDeviceType &deviceType);

std::string getContainerName(const std::string& deviceTypeName, const std::string& modelName);
std::string getContainerName(const SystemDeviceType& deviceType, const ModelType& modelType);

/**
 * @brief Get the Container Lib object
 * 
 * @param deviceName "all" for controller, a specifc type for each device
 * @return ContainerLibType 
 */
ContainerLibType getContainerLib(const std::string& deviceType);

template <typename T>
void finishGrpc(std::unique_ptr<ClientAsyncResponseReader<T>> &rpc, T &reply, Status &status, CompletionQueue *cq){
    rpc->Finish(&reply, &status, (void *)1);
    void *got_tag;
    bool ok = false;
    GPR_ASSERT(cq->Next(&got_tag, &ok));
    GPR_ASSERT(ok);
}

std::string getDeviceTypeName(SystemDeviceType deviceType);
#endif